// SPDX-License-Identifier: AGPL-3.0-only
// ATLAS_QWEN4EXP_PREFILL_MOE_BF16: the routed-MoE prefill on tensor cores
// (qwen4exp_moe_tcp.cu) against
//   check  the TC decode kernels (qwen4exp_moe_c8_tc.cu) on the same rows,
//          byte for byte: sampled tokens of a prefill chunk alone (1 row) and
//          as 16-row waves, every local expert's down row; with the routed
//          clamp and without (ATLAS_QWEN4EXP_MOE_NO_CLAMP)
//   time   today's prefill chain, moe_prefill_q38's W2 kernels
//          (ATLAS_QWEN4EXP_PREFILL_MOE[_W2]: a->e4m3, gate_up_silu, down)
// at the TP=EP=2 shape: 256 local experts of 512, top-10, lognormal routing
// skew (as qwen4exp_moe_prefill_bench.cu), TOKENS per chunk (default 16000).
//
// Build/run: scripts/dev/qwen4exp_moe_tcp_bench.sh [check|time] (repo root, GB10).
#include "qwen4exp_moe_c8_bench.h"

static const unsigned EL = 256;
static unsigned GU_NT = 4;  // TCP_GU_NT of the loaded PTX (env GU_NT)  // local experts: global ids < 256

struct LPool {  // the local experts, served [N, K/2] and prefill [K/2, N]
    u64 *gp, *gs, *upk, *us, *dp, *ds, *gpT, *gsT, *upT, *usT, *dpT, *dsT;
    float *g2, *u2, *d2;
    explicit LPool(unsigned n) {
        std::vector<u64> t[12];
        for (auto& v : t) v.assign(NE, 0);
        std::vector<float> s[3];
        for (auto& v : s) v.assign(NE, 0.f);
        for (unsigned e = 0; e < EL; e++) {
            if (e >= n) {  // fewer distinct experts: cycle them
                for (int i = 0; i < 12; i++) t[i][e] = t[i][e % n];
                for (int j = 0; j < 3; j++) s[j][e] = s[j][e % n];
                continue;
            }
            Proj p[3] = {make_proj(I, H), make_proj(I, H), make_proj(H, I)};
            const unsigned nn[3] = {I, I, H}, kk[3] = {H, H, I};
            for (int j = 0; j < 3; j++) {
                t[2 * j][e] = (u64)p[j].packed; t[2 * j + 1][e] = (u64)p[j].scale; s[j][e] = p[j].s2;
                const auto pk = dget(p[j].packed, (size_t)nn[j] * kk[j] / 2), sc = dget(p[j].scale, (size_t)nn[j] * kk[j] / 16);
                std::vector<unsigned char> pT(pk.size()), sT(sc.size());
                for (unsigned r = 0; r < nn[j]; r++) {
                    for (unsigned c = 0; c < kk[j] / 2; c++) pT[(size_t)c * nn[j] + r] = pk[(size_t)r * (kk[j] / 2) + c];
                    for (unsigned c = 0; c < kk[j] / 16; c++) sT[(size_t)c * nn[j] + r] = sc[(size_t)r * (kk[j] / 16) + c];
                }
                t[6 + 2 * j][e] = (u64)dput(pT); t[7 + 2 * j][e] = (u64)dput(sT);
            }
        }
        u64** d[12] = {&gp, &gs, &upk, &us, &dp, &ds, &gpT, &gsT, &upT, &usT, &dpT, &dsT};
        for (int i = 0; i < 12; i++) *d[i] = dput(t[i]);
        g2 = dput(s[0]); u2 = dput(s[1]); d2 = dput(s[2]);
    }
};

// A chunk: tokens x top-10 of 512 (lognormal popularity), the local entries
// sorted by expert; pos[token * 10 + slot] = sorted row (or -1: remote).
struct Chunk {
    unsigned tokens;
    std::vector<unsigned> ids;
    std::vector<int> off, tok, pos;
    void* A;
    std::vector<unsigned short> a;
};
static Chunk make_chunk(unsigned tokens) {
    Chunk c{tokens, {}, std::vector<int>(EL + 1, 0), {}, std::vector<int>((size_t)tokens * TOPK, -1), nullptr, {}};
    std::lognormal_distribution<double> ln(0.0, 1.0);
    std::vector<double> pop(NE);
    for (auto& p : pop) p = ln(g_rng);
    std::discrete_distribution<unsigned> pick(pop.begin(), pop.end());
    std::vector<std::pair<unsigned, unsigned>> loc;  // (expert, token * 10 + slot)
    for (unsigned t = 0; t < tokens; t++) {
        std::vector<unsigned> row;
        while (row.size() < TOPK) {
            const unsigned e = pick(g_rng);
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        for (unsigned s = 0; s < TOPK; s++) if (row[s] < EL) loc.push_back({row[s], t * TOPK + s});
        c.ids.insert(c.ids.end(), row.begin(), row.end());
    }
    std::sort(loc.begin(), loc.end());
    for (size_t i = 0; i < loc.size(); i++) {
        c.off[loc[i].first + 1]++;
        c.tok.push_back((int)(loc[i].second / TOPK));
        c.pos[loc[i].second] = (int)i;
    }
    for (unsigned e = 0; e < EL; e++) c.off[e + 1] += c.off[e];
    // Activations: most rows N(0, 1), every 7th x30 so gate/up pass +-10.
    std::vector<unsigned short> a((size_t)tokens * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (unsigned t = 0; t < tokens; t++)
        for (unsigned k = 0; k < H; k++) a[(size_t)t * H + k] = tobf(nd(g_rng) * (t % 7 == 3 ? 30.f : 1.f));
    c.A = dput(a);
    c.a = std::move(a);
    return c;
}

struct Dev { void *off, *tok, *act, *C, *a8, *act8; unsigned gy, slots, gyg; };
static Dev dev_chunk(const Chunk& c) {
    int mx = 0;
    for (unsigned e = 0; e < EL; e++) mx = std::max(mx, c.off[e + 1] - c.off[e]);
    const size_t S = c.tok.size();
    return {dput(c.off), dput(c.tok), dzero(S * I * 2), dzero(S * H * 2), dzero((size_t)c.tokens * H),
            dzero(S * I), (unsigned)(mx + 32 * 2 - 1) / (32 * 2), (unsigned)S, (unsigned)(mx + 63) / 64};
}

static void tcp_chain(const LPool& p, const Chunk& c, Dev& d, bool nc, int part = 3) {
    const char* M = "qwen4exp_moe_tcp";
    unsigned ne = EL;
    if (part & 1)
        launch(load(M, nc ? "qwen4exp_moe_tcp_gate_up_nc" : "qwen4exp_moe_tcp_gate_up"), dim3(I / (8 * GU_NT), d.gyg, EL),
               dim3(128), {(void*)&c.A, (void*)&p.gp, (void*)&p.gs, (void*)&p.g2, (void*)&p.upk, (void*)&p.us,
                           (void*)&p.u2, &d.act, &d.off, &d.tok, &ne});
    if (part & 2)
        launch(load(M, "qwen4exp_moe_tcp_down"), dim3(H / 64, d.gy, EL), dim3(128),
               {&d.act, (void*)&p.dp, (void*)&p.ds, (void*)&p.d2, &d.C, &d.off, &ne});
}

static void q38_chain(const LPool& p, const Chunk& c, Dev& d, int part = 3) {
    const char* M = "moe_prefill_q38";
    unsigned n = c.tokens * H, ne = EL, ni = I, kh = H, nh = H, ki = I;
    if (part & 1) {
        launch(load(M, "moe_q38_a_to_e4m3"), dim3((n / 4 + 255) / 256), dim3(256), {(void*)&c.A, &d.a8, &n});
        launch(load(M, "moe_q38w_gate_up_silu"), dim3(I / 64, d.gy, EL), dim3(256),
               {&d.a8, (void*)&p.gpT, (void*)&p.gsT, (void*)&p.g2, (void*)&p.upT, (void*)&p.usT, (void*)&p.u2,
                &d.act8, &d.off, &d.tok, &ne, &ni, &kh});
    }
    if (part & 2)
        launch(load(M, "moe_q38w_down"), dim3(H / 128, d.gy, EL), dim3(256),
               {&d.act8, (void*)&p.dpT, (void*)&p.dsT, (void*)&p.d2, &d.C, &d.off, &ne, &nh, &ki});
}

// The TC decode chain on `toks` (<= 64 tokens) of the chunk; down rows
// [toks.size() * 10, 2560], (token, slot) order.
static std::vector<unsigned char> tc_decode(const LPool& p, const Chunk& c, const std::vector<unsigned>& toks, bool nc) {
    const unsigned R = toks.size(), S = R * TOPK;
    std::vector<unsigned> ids;
    std::vector<unsigned short> a;
    for (unsigned t : toks) {
        ids.insert(ids.end(), c.ids.begin() + t * TOPK, c.ids.begin() + (t + 1) * TOPK);
        const unsigned short* src = c.a.data() + (size_t)t * H;
        a.insert(a.end(), src, src + H);
    }
    void *A = dput(a), *di = dput(ids), *ws = dzero(1 << 20), *act = dzero((size_t)(S + R) * I * 4);
    void *down = dzero((size_t)S * H * 2), *shd = dzero((size_t)R * H * 2), *null = nullptr;
    float z = 0.f;
    unsigned topk = TOPK, rows = R;
    const unsigned units = S + (R + 15) / 16;
    const char* M = "qwen4exp_moe_c8_tc";
    launch(load(M, "qwen4exp_moe_c8_tc_plan"), dim3(1), dim3(1024), {&di, &ws, &topk, &rows});
    launch(load(M, nc ? "qwen4exp_moe_c8_tc_gate_up_nc" : "qwen4exp_moe_c8_tc_gate_up"), dim3(I / 8, units), dim3(256),
           {&A, (void*)&p.gp, (void*)&p.gs, (void*)&p.g2, (void*)&p.upk, (void*)&p.us, (void*)&p.u2, &null, &null, &z,
            &null, &null, &z, &ws, &null, &null, &null, &null, &act, &topk, &rows});
    launch(load(M, "qwen4exp_moe_c8_tc_down"), dim3(H / 64, units), dim3(256),
           {&act, (void*)&p.dp, (void*)&p.ds, (void*)&p.d2, &null, &null, &z, &ws, &down, &shd, &topk, &rows});
    auto out = dget(down, (size_t)S * H * 2);
    for (void* q : {A, di, ws, act, down, shd}) cudaFree(q);
    return out;
}

static int check(const LPool& p) {
    Chunk c = make_chunk(getenv("TOKENS") ? atoi(getenv("TOKENS")) : 2048);
    Dev d = dev_chunk(c);
    int bad = 0, rows = 0;
    std::vector<unsigned char> prev;
    for (bool nc : {false, true}) {
        CK(cudaMemset(d.C, 0x55, (size_t)d.slots * H * 2));
        tcp_chain(p, c, d, nc);
        const auto C = dget(d.C, (size_t)d.slots * H * 2);
        if (nc) {  // the switch must matter where the clamp bites
            size_t moved = 0;
            for (size_t r = 0; r < d.slots; r++) moved += memcmp(&C[r * H * 2], &prev[r * H * 2], H * 2) != 0;
            printf("  clamp vs no clamp: %zu of %u prefill rows differ\n", moved, d.slots);
        }
        prev = C;
        std::vector<unsigned> sample;
        for (unsigned t = 0; t < c.tokens && sample.size() < 96; t += 1 + c.tokens / 96) sample.push_back(t);
        for (size_t w0 = 0; w0 < sample.size(); w0 += 16) {
            const std::vector<unsigned> wave(sample.begin() + w0, sample.begin() + std::min(sample.size(), w0 + 16));
            for (int solo = 0; solo < 2; solo++) {
                for (size_t i = 0; i < wave.size(); i++) {
                    const std::vector<unsigned> toks = solo ? std::vector<unsigned>{wave[i]} : wave;
                    if (solo == 0 && i) break;
                    const auto D = tc_decode(p, c, toks, nc);
                    for (size_t r = 0; r < toks.size(); r++)
                        for (unsigned s = 0; s < TOPK; s++) {
                            const int at = c.pos[toks[r] * TOPK + s];
                            if (at < 0) continue;
                            rows++;
                            bad += memcmp(&C[(size_t)at * H * 2], &D[(r * TOPK + s) * (size_t)H * 2], H * 2) != 0;
                        }
                }
            }
        }
        printf("  %s  %s: %d of %d (token, local expert) down rows differ, prefill chunk of %u tokens vs TC decode "
               "(1-row and 16-row launches)\n", bad ? "BAD" : "ok ", nc ? "NO_CLAMP" : "clamp   ", bad, rows, c.tokens);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}

static void time_chunks(const LPool& p) {
    Timer tm;
    for (unsigned tokens : {4096u, 16000u}) {
        Chunk c = make_chunk(tokens);
        Dev d = dev_chunk(c);
        const int it = 5;
        const double q = tm.run(it, [&](int) { q38_chain(p, c, d); });
        const double b = tm.run(it, [&](int) { tcp_chain(p, c, d, false); });
        const double bn = tm.run(it, [&](int) { tcp_chain(p, c, d, true); });
        const double qg = tm.run(it, [&](int) { q38_chain(p, c, d, 1); });
        const double bg = tm.run(it, [&](int) { tcp_chain(p, c, d, false, 1); });
        printf("  %5u tokens (%u local rows): q38 W2 FP8 %8.2f ms   BF16 tcp %8.2f ms (%+.0f%%)   tcp no-clamp %8.2f ms"
               "   -> x48 layers: %.0f vs %.0f ms\n", tokens, d.slots, q / 1e3, b / 1e3, 100 * (b / q - 1), bn / 1e3,
               48 * q / 1e3, 48 * b / 1e3);
        printf("        gate/up(+a2e4m3) %.2f vs %.2f ms, down %.2f vs %.2f ms; tcp %.1f TFLOP/s\n", qg / 1e3, bg / 1e3,
               (q - qg) / 1e3, (b - bg) / 1e3, 2.0 * d.slots * 3.0 * H * I / b / 1e6);
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (getenv("GU_NT")) GU_NT = atoi(getenv("GU_NT"));
    const std::string mode = argc > 2 ? argv[2] : "time";
    CU(cuInit(0));
    CK(cudaFree(0));
    LPool p(mode == "check" ? 24 : EL);  // check: 24 distinct experts cycled (the bytes are per expert)
    if (mode == "check") return check(p);
    time_chunks(p);
    return 0;
}
