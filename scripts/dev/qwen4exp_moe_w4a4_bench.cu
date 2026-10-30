// SPDX-License-Identifier: AGPL-3.0-only
// The routed-MoE prefill on native FP4 tensor cores (moe_prefill_w4a4.cu:
// quant_input, gate_up_silu, down) at the TP=EP=2 shape -- 512 experts of
// which 0..255 are local (256..511 null tables, as on EP rank 0), hidden 2560
// / intermediate 640, top-10 with a lognormal routing skew:
//
//   check  against a CPU reference of the b12x A4 prefill math
//          (prefill_a4.py + quantize_block_fp4):
//          * the input's NVFP4 codes and block scales, bit for bit, on the
//            chunk and on blocks built to land on every rounding tie;
//          * the activation's NVFP4 groups from the CPU's gate / up (FP64
//            sums of the exact products): equal except where the GPU's FP32
//            accumulation order or its ex2 / rcp approximations move a BF16
//            rounding (counted, bounded);
//          * the down rows recomputed on the CPU from the GPU's activation:
//            within one BF16 ulp (FP32 accumulation order);
//          * row invariance: the first half (the SP split), a third and one
//            token of the chunk alone give their rows' bytes of the whole;
//          * EP: rows of null-table experts are never written.
//   time   one layer's chain at TOKENS (default 16000) against the shipped
//          q38 W2 N-major chain (a->e4m3, moe_q38wn_gate_up_silu,
//          moe_q38wn_down) and the BF16 TC chain (ATLAS_QWEN4EXP_PREFILL_MOE_BF16).
//
// Build/run: scripts/dev/qwen4exp_moe_w4a4_bench.sh [check|time] (repo root, GB10).
#include "qwen4exp_moe_c8_bench.h"
#include <cuda_fp8.h>
#include <thread>

static const unsigned EL = 256;   // local experts (global ids < 256)

// ── host forms of the device arithmetic ──
static float e4m3_val(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, (int)e - 7);
    return (b & 0x80u) ? -v : v;
}
static unsigned e4m3_rn(float v) {
    return (unsigned)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
static const float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
static float bf16_rn(float x) {  // cvt.rn.satfinite.bf16
    if (std::isnan(x)) return x;
    unsigned u;
    memcpy(&u, &x, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float r;
    memcpy(&r, &u, 4);
    if (std::isinf(r)) { u = (u & 0x80000000u) | 0x7F7F0000u; memcpy(&r, &u, 4); }
    return r;
}
static float bf16_val(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }

// quantize_block_fp4: scale byte and the 16 codes of one block.
static unsigned quant_block(const float* v, float g, unsigned char* codes) {
    float amax = 0;
    for (int i = 0; i < 16; i++) amax = std::max(amax, std::fabs(v[i]));
    const float sf = std::min(amax * g / 6.0f, 448.0f);
    const unsigned byte = e4m3_rn(sf);
    const float dec = e4m3_val(byte);
    const bool zero = dec == 0.0f || g == 0.0f;
    const float vs = zero ? 0.0f : dec / g;
    const float t[7] = {vs * 0.25f, vs * 0.75f, vs * 1.25f, vs * 1.75f, vs * 2.5f, vs * 3.5f, vs * 5.0f};
    for (int i = 0; i < 16; i++) {
        const float m = std::fabs(v[i]);
        unsigned c = 0;
        if (zero) c = 0;
        else if (m > t[0] && m < t[1]) c = 1;
        else if (m >= t[1] && m <= t[2]) c = 2;
        else if (m > t[2] && m < t[3]) c = 3;
        else if (m >= t[3] && m <= t[4]) c = 4;
        else if (m > t[4] && m < t[5]) c = 5;
        else if (m >= t[5] && m <= t[6]) c = 6;
        else if (m > t[6]) c = 7;
        codes[i] = (unsigned char)((c && v[i] < 0) ? (c | 8) : c);
    }
    return byte;
}
static void quant_rows(const std::vector<float>& x, unsigned rows, unsigned K, float g,
                       std::vector<unsigned char>& packed, std::vector<unsigned char>& scale) {
    packed.assign((size_t)rows * K / 2, 0);
    scale.assign((size_t)rows * K / 16, 0);
    for (size_t gi = 0; gi < (size_t)rows * K / 16; gi++) {
        unsigned char c[16];
        scale[gi] = (unsigned char)quant_block(&x[gi * 16], g, c);
        for (int i = 0; i < 8; i++) packed[gi * 8 + i] = (unsigned char)(c[2 * i] | (c[2 * i + 1] << 4));
    }
}
// The dequantized value of element k of an NVFP4 row (no global scale).
static float nv_at(const unsigned char* p, const unsigned char* s, unsigned k) {
    const unsigned char b = p[k / 2];
    return E2M1[(k & 1) ? b >> 4 : b & 15] * e4m3_val(s[k / 16]);
}

// ── the experts: realistic block scales (ModelOpt: scale2 = amax / 2688) ──
struct W {
    std::vector<unsigned char> p, s;
    float s2;
    unsigned char *dp, *ds;
    unsigned n, k;
    std::vector<float> f;   // dequantized [n, k] (no scale2), built for `check`
    void dequant() {
        f.resize((size_t)n * k);
        for (unsigned r = 0; r < n; r++)
            for (unsigned j = 0; j < k; j++) f[(size_t)r * k + j] = nv_at(&p[(size_t)r * k / 2], &s[(size_t)r * k / 16], j);
    }
};
static W make_w(unsigned n, unsigned k) {
    W w;
    w.n = n;
    w.k = k;
    w.p.resize((size_t)n * k / 2);
    w.s.resize((size_t)n * k / 16);
    for (auto& x : w.p) x = (unsigned char)g_rng();
    std::lognormal_distribution<float> ln(0.f, 0.6f);
    for (auto& x : w.s) x = (unsigned char)e4m3_rn(std::min(448.f, 120.f * ln(g_rng)));
    std::uniform_real_distribution<float> d(0.5f, 2.0f);
    w.s2 = d(g_rng) / 2688.f;
    w.dp = dput(w.p);
    w.ds = dput(w.s);
    return w;
}
struct APool {
    std::vector<W> gate, up, down;   // distinct experts, cycled over the 256 local ids
    u64 *gp, *gs, *upk, *us, *dp, *ds;
    float *g2v, *u2v, *d2v;          // weight_scale_2 (q38 / tcp)
    float *ga, *ua, *da, *a2g;       // the W4A4 alphas and g2
    float is1;
    std::vector<float> is2;
    explicit APool(unsigned n) {
        for (unsigned e = 0; e < n; e++) {
            gate.push_back(make_w(I, H)); up.push_back(make_w(I, H)); down.push_back(make_w(H, I));
        }
        is1 = 6.0f / 2688.f;                         // calibrated |x| max ~6
        std::lognormal_distribution<float> ln(0.f, 0.7f);
        std::vector<u64> t[6];
        for (auto& v : t) v.assign(NE, 0);
        std::vector<float> s[7];
        for (auto& v : s) v.assign(NE, 0.f);
        is2.assign(NE, 0.f);
        const float g1 = 1.0f / is1;
        for (unsigned e = 0; e < EL; e++) {
            const unsigned c = e % n;
            t[0][e] = (u64)gate[c].dp; t[1][e] = (u64)gate[c].ds; t[2][e] = (u64)up[c].dp;
            t[3][e] = (u64)up[c].ds; t[4][e] = (u64)down[c].dp; t[5][e] = (u64)down[c].ds;
            is2[e] = 40.0f / 2688.f * ln(g_rng);
            const float g2 = 1.0f / is2[e];
            s[0][e] = gate[c].s2; s[1][e] = up[c].s2; s[2][e] = down[c].s2;
            s[3][e] = gate[c].s2 / g1; s[4][e] = up[c].s2 / g1; s[5][e] = down[c].s2 / g2; s[6][e] = g2;
        }
        u64** d[6] = {&gp, &gs, &upk, &us, &dp, &ds};
        for (int i = 0; i < 6; i++) *d[i] = dput(t[i]);
        float** f[7] = {&g2v, &u2v, &d2v, &ga, &ua, &da, &a2g};
        for (int i = 0; i < 7; i++) *f[i] = dput(s[i]);
    }
    float g1() const { return 1.0f / is1; }
};

// ── a chunk: tokens x top-10 of 512, every route kept (remote ones too, as
// the production sort does), rows sorted by expert ──
struct Chunk {
    unsigned tokens;
    std::vector<unsigned> ids;          // [tokens * 10]
    std::vector<int> off, tok, pos;     // off [513], tok [rows], pos [tokens * 10] -> row
    std::vector<unsigned short> a;      // [tokens, H] BF16
};
static Chunk make_chunk(unsigned tokens) {
    Chunk c{tokens, {}, std::vector<int>(NE + 1, 0), {}, std::vector<int>((size_t)tokens * TOPK), {}};
    std::lognormal_distribution<double> ln(0.0, 1.0);
    std::vector<double> pop(NE);
    for (auto& p : pop) p = ln(g_rng);
    std::discrete_distribution<unsigned> pick(pop.begin(), pop.end());
    for (unsigned t = 0; t < tokens; t++) {
        std::vector<unsigned> row;
        while (row.size() < TOPK) {
            const unsigned e = pick(g_rng);
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        c.ids.insert(c.ids.end(), row.begin(), row.end());
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    c.a.resize((size_t)tokens * H);
    for (unsigned t = 0; t < tokens; t++)
        for (unsigned k = 0; k < H; k++) c.a[(size_t)t * H + k] = tobf(nd(g_rng) * (t % 7 == 3 ? 4.f : 1.f));
    return c;
}
// The first `tokens` tokens of `c`, routed as there, re-sorted.
static Chunk sub_chunk(const Chunk& c, unsigned tokens) {
    Chunk s{tokens, std::vector<unsigned>(c.ids.begin(), c.ids.begin() + tokens * TOPK), std::vector<int>(NE + 1, 0),
            {}, std::vector<int>((size_t)tokens * TOPK), std::vector<unsigned short>(c.a.begin(), c.a.begin() + (size_t)tokens * H)};
    return s;
}
static void sort_routes(Chunk& c) {
    std::vector<std::pair<unsigned, unsigned>> r;
    for (unsigned i = 0; i < c.tokens * TOPK; i++) r.push_back({c.ids[i], i});
    std::sort(r.begin(), r.end());
    c.off.assign(NE + 1, 0);
    c.tok.clear();
    for (size_t i = 0; i < r.size(); i++) {
        c.off[r[i].first + 1]++;
        c.tok.push_back((int)(r[i].second / TOPK));
        c.pos[r[i].second] = (int)i;
    }
    for (unsigned e = 0; e < NE; e++) c.off[e + 1] += c.off[e];
}

struct Dev {
    void *A, *off, *tok, *a4, *a4s, *act, *acts, *C, *a8, *act8, *actb;
    unsigned rows, gy, ty;
};
static Dev dev_chunk(const Chunk& c) {
    const size_t R = c.tok.size(), T = c.tokens;
    int mx = 0;
    for (unsigned e = 0; e < EL; e++) mx = std::max(mx, c.off[e + 1] - c.off[e]);
    const unsigned avg = (unsigned)((R + NE - 1) / NE);
    Dev d{dput(c.a), dput(c.off), dput(c.tok), dzero(T * H / 2), dzero(T * H / 16), dzero(R * I / 2),
          dzero(R * I / 16), dzero(R * H * 2), dzero(T * H), dzero(R * I), dzero(R * I * 2), (unsigned)R,
          // the production grids: rows of twice the average expert, strided
          std::max(1u, std::min((unsigned)(mx + 127) / 128, (2 * avg + 127) / 128)),
          std::max(1u, std::min((unsigned)(mx + 63) / 64, (2 * avg + 63) / 64))};
    return d;
}

// W4A4_PTX=<stem>: another build of moe_prefill_w4a4.cu (a tile sweep point).
static const char* w4a4_module() { return getenv("W4A4_PTX") ? getenv("W4A4_PTX") : "moe_prefill_w4a4"; }

static void w4a4_chain(const APool& p, Dev& d, unsigned tokens, int part = 7) {
    const char* M = w4a4_module();
    unsigned groups = tokens * H / 16, ne = NE, ni = I, kh = H, nh = H, ki = I;
    float g1 = p.g1();
    if (part & 1)
        launch(load(M, "moe_w4a4_quant_input"), dim3((groups + 255) / 256), dim3(256), {&d.A, &d.a4, &d.a4s, &g1, &groups});
    if (part & 2)
        launch(load(M, "moe_w4a4_gate_up_silu"), dim3(I / 64, d.gy, NE), dim3(256),
               {&d.a4, &d.a4s, (void*)&p.gp, (void*)&p.gs, (void*)&p.ga, (void*)&p.upk, (void*)&p.us, (void*)&p.ua,
                (void*)&p.a2g, &d.act, &d.acts, &d.off, &d.tok, &ne, &ni, &kh});
    if (part & 4)
        launch(load(M, "moe_w4a4_down"), dim3(H / 128, d.gy, NE), dim3(256),
               {&d.act, &d.acts, (void*)&p.dp, (void*)&p.ds, (void*)&p.da, &d.C, &d.off, &ne, &nh, &ki});
}
static void q38wn_chain(const APool& p, Dev& d, unsigned tokens, int part = 3) {
    const char* M = "moe_prefill_q38";
    unsigned n = tokens * H, ne = NE, ni = I, kh = H, nh = H, ki = I;
    if (part & 1) {
        launch(load(M, "moe_q38_a_to_e4m3"), dim3((n / 4 + 255) / 256), dim3(256), {&d.A, &d.a8, &n});
        launch(load(M, "moe_q38wn_gate_up_silu"), dim3(I / 64, d.gy, NE), dim3(256),
               {&d.a8, (void*)&p.gp, (void*)&p.gs, (void*)&p.g2v, (void*)&p.upk, (void*)&p.us, (void*)&p.u2v,
                &d.act8, &d.off, &d.tok, &ne, &ni, &kh});
    }
    if (part & 2)
        launch(load(M, "moe_q38wn_down"), dim3(H / 128, d.gy, NE), dim3(256),
               {&d.act8, (void*)&p.dp, (void*)&p.ds, (void*)&p.d2v, &d.C, &d.off, &ne, &nh, &ki});
}
static void tcp_chain(const APool& p, Dev& d, int part = 3) {
    const char* M = "qwen4exp_moe_tcp";
    unsigned ne = NE;
    if (part & 1)
        launch(load(M, "qwen4exp_moe_tcp_gate_up_nc"), dim3(I / 32, d.ty, NE), dim3(128),
               {&d.A, (void*)&p.gp, (void*)&p.gs, (void*)&p.g2v, (void*)&p.upk, (void*)&p.us, (void*)&p.u2v, &d.actb,
                &d.off, &d.tok, &ne});
    if (part & 2)
        launch(load(M, "qwen4exp_moe_tcp_down"), dim3(H / 64, d.ty, NE), dim3(128),
               {&d.actb, (void*)&p.dp, (void*)&p.ds, (void*)&p.d2v, &d.C, &d.off, &ne});
}

template <class F> static void par_for(size_t n, F f) {
    const unsigned T = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned t = 0; t < T; t++) th.emplace_back([&, t] { for (size_t i = t; i < n; i += T) f(i); });
    for (auto& x : th) x.join();
}

// Blocks whose values sit on every decision threshold of their own grid,
// both signs, plus all-zero and saturating blocks.
static std::vector<float> tie_blocks(float g) {
    std::vector<float> v;
    const float amaxes[] = {6.f, 3.f, 1.f, 0.37f, 1e-3f, 1e3f, 5e4f, 0.f};
    for (float am : amaxes) {
        const float sf = std::min(am * g / 6.0f, 448.0f);
        const float dec = e4m3_val(e4m3_rn(sf));
        const float vs = dec / g;
        const float mult[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        float b[16];
        b[0] = am;
        for (int i = 0; i < 7; i++) {
            b[1 + 2 * i] = bf16_rn(vs * mult[i]);
            b[2 + 2 * i] = -bf16_rn(vs * mult[i]);
        }
        b[15] = 0.f;
        // keep every value inside amax so the block's scale is am's
        for (auto& x : b) if (std::fabs(x) > am) x = std::copysign(am, x);
        for (auto x : b) v.push_back(bf16_rn(x));
    }
    while (v.size() % H) v.push_back(0.f);
    return v;
}

static int check_quant(float g) {
    const auto v = tie_blocks(g);
    const unsigned rows = (unsigned)(v.size() / H);
    std::vector<unsigned short> b(v.size());
    for (size_t i = 0; i < v.size(); i++) b[i] = tobf(v[i]);
    void *x = dput(b), *pk = dzero(v.size() / 2), *sc = dzero(v.size() / 16);
    unsigned groups = rows * H / 16;
    launch(load(w4a4_module(), "moe_w4a4_quant_input"), dim3((groups + 255) / 256), dim3(256), {&x, &pk, &sc, &g, &groups});
    std::vector<unsigned char> rp, rs;
    std::vector<float> vf(v.size());
    for (size_t i = 0; i < v.size(); i++) vf[i] = bf16_val(b[i]);
    quant_rows(vf, rows, H, g, rp, rs);
    const auto gp = dget(pk, rp.size()), gs = dget(sc, rs.size());
    const bool ok = gp == rp && gs == rs;
    printf("  %s  tie blocks, g = %g: codes and scales %s\n", ok ? "ok " : "BAD", g, ok ? "bit-exact" : "DIFFER");
    for (void* q : {x, pk, sc}) cudaFree(q);
    return !ok;
}

static int check(APool& p, unsigned n_distinct) {
    for (auto* v : {&p.gate, &p.up, &p.down})
        for (auto& w : *v) w.dequant();
    int bad = 0;
    bad += check_quant(1.0f);
    bad += check_quant(p.g1());
    bad += check_quant(37.25f);
    const unsigned tokens = getenv("TOKENS") ? atoi(getenv("TOKENS")) : 1024;
    Chunk c = make_chunk(tokens);
    sort_routes(c);
    Dev d = dev_chunk(c);
    CK(cudaMemset(d.C, 0x55, (size_t)d.rows * H * 2));
    CK(cudaMemset(d.act, 0x55, (size_t)d.rows * I / 2));
    w4a4_chain(p, d, tokens);
    const auto a4 = dget(d.a4, (size_t)tokens * H / 2), a4s = dget(d.a4s, (size_t)tokens * H / 16);
    const auto act = dget(d.act, (size_t)d.rows * I / 2), acts = dget(d.acts, (size_t)d.rows * I / 16);
    const auto C = dget(d.C, (size_t)d.rows * H * 2);

    // (1) input quantization, bit for bit
    std::vector<float> x(c.a.size());
    for (size_t i = 0; i < x.size(); i++) x[i] = bf16_val(c.a[i]);
    std::vector<unsigned char> rp, rs;
    quant_rows(x, tokens, H, p.g1(), rp, rs);
    const bool qok = rp == a4 && rs == a4s;
    printf("  %s  input NVFP4 (%u tokens, g1 = %g): codes and scales %s\n", qok ? "ok " : "BAD", tokens, p.g1(),
           qok ? "bit-exact" : "DIFFER");
    bad += !qok;

    // (2) the activation from CPU gate / up; (3) down from the GPU's activation
    std::vector<int> expert_of(d.rows);
    for (unsigned e = 0; e < NE; e++) for (int r = c.off[e]; r < c.off[e + 1]; r++) expert_of[r] = (int)e;
    std::vector<size_t> code_diff(d.rows, 0), scale_diff(d.rows, 0), dn_exact(d.rows, 0), dn_far(d.rows, 0), remote_bad(d.rows, 0);
    std::vector<double> err2(d.rows, 0), ref2(d.rows, 0);
    par_for(d.rows, [&](size_t r) {
        const unsigned e = (unsigned)expert_of[r];
        const unsigned short* Cr = reinterpret_cast<const unsigned short*>(C.data()) + r * H;
        if (e >= EL) {   // remote: untouched
            for (unsigned k = 0; k < I / 2; k++) remote_bad[r] += act[r * (I / 2) + k] != 0x55;
            for (unsigned k = 0; k < H; k++) remote_bad[r] += Cr[k] != 0x5555;
            return;
        }
        const W &wg = p.gate[e % n_distinct], &wu = p.up[e % n_distinct], &wd = p.down[e % n_distinct];
        const unsigned char *xa = &rp[(size_t)c.tok[r] * H / 2], *xs = &rs[(size_t)c.tok[r] * H / 16];
        std::vector<float> xv(H), av(I);
        for (unsigned k = 0; k < H; k++) xv[k] = nv_at(xa, xs, k);
        const float ag = wg.s2 / p.g1(), au = wu.s2 / p.g1(), g2 = 1.0f / p.is2[e], ad = wd.s2 / g2;
        for (unsigned n = 0; n < I; n++) {
            double sg = 0, su = 0;
            for (unsigned k = 0; k < H; k++) {
                sg += (double)xv[k] * wg.f[(size_t)n * H + k];
                su += (double)xv[k] * wu.f[(size_t)n * H + k];
            }
            const float gv = bf16_rn((float)sg * ag), uv = bf16_rn((float)su * au);
            const float sig = 1.0f / (1.0f + exp2f(-gv * 1.4426950408889634f));
            av[n] = bf16_rn(bf16_rn(gv * sig) * uv);
        }
        std::vector<unsigned char> cp, cs;
        quant_rows(av, 1, I, g2, cp, cs);
        for (unsigned k = 0; k < I / 2; k++) {
            const unsigned char a = act[r * (I / 2) + k], b = cp[k];
            code_diff[r] += ((a ^ b) & 0x0F) != 0;
            code_diff[r] += ((a ^ b) & 0xF0) != 0;
        }
        for (unsigned k = 0; k < I / 16; k++) scale_diff[r] += acts[r * (I / 16) + k] != cs[k];
        // down from the GPU's own activation
        const unsigned char *ga = &act[r * (I / 2)], *gs = &acts[r * (I / 16)];
        std::vector<float> gv(I);
        for (unsigned k = 0; k < I; k++) gv[k] = nv_at(ga, gs, k);
        for (unsigned n = 0; n < H; n++) {
            double s = 0;
            for (unsigned k = 0; k < I; k++) s += (double)gv[k] * wd.f[(size_t)n * I + k];
            const float ref = bf16_rn((float)s * ad), got = bf16_val(Cr[n]);
            dn_exact[r] += ref == got;
            dn_far[r] += std::fabs(got - ref) > std::ldexp(std::fabs(ref), -7) + 1e-30f;
            err2[r] += ((double)got - s * ad) * ((double)got - s * ad);
            ref2[r] += (s * ad) * (s * ad);
        }
    });
    size_t local = 0, cd = 0, sd = 0, ex = 0, far = 0, rb = 0;
    double e2 = 0, r2 = 0;
    for (size_t r = 0; r < d.rows; r++) {
        local += expert_of[r] < (int)EL;
        cd += code_diff[r]; sd += scale_diff[r]; ex += dn_exact[r]; far += dn_far[r]; rb += remote_bad[r];
        e2 += err2[r]; r2 += ref2[r];
    }
    const double codes = (double)local * I, outs = (double)local * H;
    const bool aok = cd <= codes * 2e-3 && sd <= local * (I / 16) * 2e-3;
    printf("  %s  activation NVFP4 vs CPU gate/up: %zu of %.0f codes, %zu of %zu scales differ (%.4f%%; BF16 "
           "rounding moved by FP32 order / ex2 / rcp)\n", aok ? "ok " : "BAD", cd, codes, sd, local * (I / 16),
           100.0 * cd / codes);
    const bool dok = far == 0;
    printf("  %s  down vs CPU from the GPU activation: %.4f%% exact, %zu beyond one BF16 ulp; rel L2 vs FP64 %.2e\n",
           dok ? "ok " : "BAD", 100.0 * ex / outs, far, std::sqrt(e2 / r2));
    printf("  %s  EP: %zu bytes of null-table expert rows written (%zu remote rows)\n", rb ? "BAD" : "ok ", rb,
           d.rows - local);
    bad += !aok + !dok + (rb != 0);

    // (4) row invariance
    for (unsigned part : {tokens / 2, tokens / 3 + 5, 1u}) {
        Chunk s = sub_chunk(c, part);
        sort_routes(s);
        Dev e = dev_chunk(s);
        w4a4_chain(p, e, part);
        const auto Cs = dget(e.C, (size_t)e.rows * H * 2);
        size_t diff = 0, n = 0;
        for (unsigned i = 0; i < part * TOPK; i++) {
            if (c.ids[i] >= EL) continue;
            n++;
            diff += memcmp(&Cs[(size_t)s.pos[i] * H * 2], &C[(size_t)c.pos[i] * H * 2], H * 2) != 0;
        }
        printf("  %s  rowinv: first %u of %u tokens alone: %zu of %zu local rows differ\n", diff ? "BAD" : "ok ", part,
               tokens, diff, n);
        bad += diff != 0;
        for (void* q : {e.A, e.off, e.tok, e.a4, e.a4s, e.act, e.acts, e.C, e.a8, e.act8, e.actb}) cudaFree(q);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}

static void time_chunk(const APool& p) {
    Timer tm;
    const unsigned tokens = getenv("TOKENS") ? atoi(getenv("TOKENS")) : 16000;
    Chunk c = make_chunk(tokens);
    sort_routes(c);
    Dev d = dev_chunk(c);
    size_t local = (size_t)c.off[EL];
    const int it = 10;
    const double w = tm.run(it, [&](int) { w4a4_chain(p, d, tokens); });
    const double wq = tm.run(it, [&](int) { w4a4_chain(p, d, tokens, 1); });
    const double wg = tm.run(it, [&](int) { w4a4_chain(p, d, tokens, 2); });
    const double q = tm.run(it, [&](int) { q38wn_chain(p, d, tokens); });
    const double qg = tm.run(it, [&](int) { q38wn_chain(p, d, tokens, 1); });
    const double b = tm.run(it, [&](int) { tcp_chain(p, d); });
    const double bg = tm.run(it, [&](int) { tcp_chain(p, d, 1); });
    const double fl = 2.0 * local * 3.0 * H * I;
    printf("  %u tokens, %zu local rows (grid.y %u):\n", tokens, local, d.gy);
    printf("    W4A4        %7.2f ms  (quant %.2f, gate_up+silu %.2f, down %.2f)  %.0f TFLOP/s\n", w / 1e3, wq / 1e3,
           wg / 1e3, (w - wq - wg) / 1e3, fl / w / 1e6);
    printf("    q38 W2 NM   %7.2f ms  (a->e4m3 + gate_up+silu %.2f, down %.2f)\n", q / 1e3, qg / 1e3, (q - qg) / 1e3);
    printf("    BF16 tcp    %7.2f ms  (gate_up %.2f, down %.2f)\n", b / 1e3, bg / 1e3, (b - bg) / 1e3);
    printf("    x48 layers: W4A4 %.0f, q38 %.0f, tcp %.0f ms\n", 48 * w / 1e3, 48 * q / 1e3, 48 * b / 1e3);
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "time";
    CU(cuInit(0));
    CK(cudaFree(0));
    const unsigned distinct = mode == "check" ? 8 : EL;
    APool p(distinct);
    if (mode == "check") return check(p, distinct);
    time_chunk(p);
    return 0;
}
