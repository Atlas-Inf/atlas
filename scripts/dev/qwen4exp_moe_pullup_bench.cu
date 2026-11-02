// SPDX-License-Identifier: AGPL-3.0-only
// Pull-up check: the MoE prefill GEMMs Atlas already vendors, at the qwen4_exp
// TP=EP=2 prefill shape (256 local experts, hidden 2560, intermediate 640,
// top-10 of 512 with a lognormal skew, one 16000-token chunk), against the
// q38 chain qwen4_exp runs today (ATLAS_QWEN4EXP_PREFILL_MOE):
//
//   q38      moe_q38_a_to_e4m3 -> moe_q38_gate_up_silu -> moe_q38_down
//            (W4A8: E4M3 activations, k32 FP8 MMAs, FP32 accumulate)
//   cutlass  atlas_cutlass_nvfp4_grouped_gate_up_fused -> moe_silu_mul ->
//            atlas_cutlass_nvfp4_grouped_down (ATLAS_MOE_GROUPED_CUTLASS;
//            W4A4: activations re-quantized to NVFP4 per 16, block-scaled
//            FP4 MMAs)
//   marlin   atlas_marlin_moe_nvfp4_m8 for gate_up (fused N = 2 x 640) and
//            down (W4A16: BF16 activations; the vendored instantiation has an
//            8-row M tile). Timed only, on random repacked bytes.
//
// For q38 and cutlass the down output of 64 sampled routed rows is compared
// with an FP64 reference (exact NVFP4 weights, BF16 input, no intermediate
// rounding): relative RMS error and max |err| / RMS(ref). This states the
// exactness CLASS of each path; none can be bit-identical to q38, whose
// operands differ.
//
// Build (repo root, inside atlas-release-builder: CUDA 13.0 + /opt/cutlass):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4; C=kernels/gb10/common
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/moe_prefill_q38.ptx $K/moe_prefill_q38.cu
//   nvcc $F -o $D/moe_silu_mul.ptx $C/moe_silu_mul.cu
//   nvcc $F --expt-relaxed-constexpr -std=c++17 -I$C -o $D/marlin_moe.ptx $C/marlin_moe_nvfp4.cu
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_121f \
//     -I/opt/cutlass/include -I/opt/cutlass/tools/util/include \
//     -o $D/bench scripts/dev/qwen4exp_moe_pullup_bench.cu \
//     crates/spark-runtime/cuda/cutlass_nvfp4_grouped_gemm.cu -lcuda
//   $D/bench $D [tokens=16000]
#include "qwen4exp_ptx_harness.h"
#include <cmath>
#include <random>

extern "C" int atlas_cutlass_pack_weight_sfb(const void*, void*, int, int, int, cudaStream_t);
extern "C" int atlas_cutlass_nvfp4_grouped_gate_up_fused(
    const void*, const int*, const unsigned long long*, const unsigned long long*, const float*,
    const unsigned long long*, const unsigned long long*, const float*, void*, void*, const int*,
    int, int, int, void*, size_t, cudaStream_t);
extern "C" int atlas_cutlass_nvfp4_grouped_down(
    const void*, const unsigned long long*, const unsigned long long*, const float*, void*,
    const int*, int, int, int, void*, size_t, cudaStream_t);

static const unsigned H = 2560, I = 640, E_ALL = 512, E = 256, TOPK = 10;
static const float LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

static unsigned char e4m3_pos(float f) {
    int ex;
    float m = frexpf(f, &ex);
    int e = ex - 1 + 7;
    int mant = (int)lrintf((m * 2.0f - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; ++e; }
    if (e < 1) e = 1;
    if (e > 15) e = 15;
    return (unsigned char)((e << 3) | mant);
}
static float e4m3_val(unsigned char b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6);
    return (b & 0x80) ? -v : v;
}

// One projection's experts: logical W[e][n][k] as nibbles + per-(k/16, n)
// scales + scale2, in q38's [K/2, N] / [K/16, N] tables and CUTLASS's
// [N, K/2] / swizzled-SFB tables.
struct Proj {
    unsigned K, N;
    std::vector<unsigned char> nib, sc;   // [E][N][K], [E][K/16][N]
    std::vector<float> s2;
    Buf<unsigned char> q_p, q_s, c_p, c_sfb;
    Buf<unsigned long long> q_pp, q_ps;
    Buf<float> d_s2;
    std::vector<unsigned long long> c_pp, c_ps;
    size_t sfb_bytes() const { return (size_t)((N + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4); }
    float w(unsigned e, unsigned n, unsigned k) const {
        return LUT[nib[((size_t)e * N + n) * K + k]] * e4m3_val(sc[((size_t)e * (K / 16) + k / 16) * N + n]) * s2[e];
    }
    void make(std::mt19937& rng, unsigned k_, unsigned n_) {
        K = k_; N = n_;
        nib.resize((size_t)E * N * K);
        sc.resize((size_t)E * (K / 16) * N);
        s2.resize(E);
        for (auto& x : nib) x = rng() & 15;
        std::uniform_real_distribution<float> u(0.004f, 0.06f), u2(0.5f, 2.0f);
        for (auto& x : sc) x = e4m3_pos(u(rng));
        for (auto& x : s2) x = u2(rng);
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N;
        std::vector<unsigned char> qp(pb * E), cp(pb * E);
        for (unsigned e = 0; e < E; ++e)
            for (unsigned n = 0; n < N; ++n)
                for (unsigned kp = 0; kp < K / 2; ++kp) {
                    const unsigned char* w2 = &nib[((size_t)e * N + n) * K + 2 * kp];
                    const unsigned char b = w2[0] | (w2[1] << 4);
                    qp[e * pb + (size_t)kp * N + n] = b;
                    cp[e * pb + (size_t)n * (K / 2) + kp] = b;
                }
        q_p.alloc(qp.size()); q_p.put(qp);
        c_p.alloc(cp.size()); c_p.put(cp);
        q_s.alloc(sc.size()); q_s.put(sc);
        d_s2.alloc(E); d_s2.put(s2);
        c_sfb.alloc(sfb_bytes() * E * 2);
        c_sfb.fill(0);
        std::vector<unsigned long long> pp(E), ps(E);
        c_pp.resize(E); c_ps.resize(E);
        for (unsigned e = 0; e < E; ++e) {
            pp[e] = (unsigned long long)(q_p.p + e * pb);
            ps[e] = (unsigned long long)(q_s.p + e * sb);
            c_pp[e] = (unsigned long long)(c_p.p + e * pb);
            c_ps[e] = (unsigned long long)(c_sfb.p + e * sfb_bytes() * 2);
            if (atlas_cutlass_pack_weight_sfb(q_s.p + e * sb, (void*)c_ps[e], N, K, 0, 0)) {
                fprintf(stderr, "pack_weight_sfb failed\n"); exit(1);
            }
        }
        q_pp.alloc(E); q_pp.put(pp);
        q_ps.alloc(E); q_ps.put(ps);
    }
};

static float bf(unsigned short h) { return bf2f(h); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [tokens]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 16000;
    init_driver();
    PtxModule m_q38, m_silu, m_marlin;
    m_q38.load(dir + "/moe_prefill_q38.ptx");
    m_silu.load(dir + "/moe_silu_mul.ptx");
    const bool have_marlin = m_marlin.try_load(dir + "/marlin_moe.ptx");

    // Routing: the qwen4exp_moe_prefill_bench distribution.
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<double> pop(E_ALL);
    for (auto& p : pop) p = exp(0.8 * nd(rng));
    std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::vector<std::vector<int>> rows_of(E);
    for (unsigned t = 0; t < T; ++t) {
        int chosen[TOPK];
        for (unsigned k = 0; k < TOPK; ++k) {
            int e;
            do { e = pick(rng); } while (std::find(chosen, chosen + k, e) != chosen + k);
            chosen[k] = e;
            if (e < (int)E) rows_of[e].push_back((int)(t * TOPK + k));
        }
    }
    std::vector<int> offsets(E + 1, 0), sorted_slot, sorted_tok;
    for (unsigned e = 0; e < E; ++e) {
        offsets[e + 1] = offsets[e] + (int)rows_of[e].size();
        for (int s : rows_of[e]) { sorted_slot.push_back(s); sorted_tok.push_back(s / TOPK); }
    }
    const unsigned R = offsets[E];
    printf("tokens %u, local routed rows %u\n", T, R);

    std::vector<unsigned short> a((size_t)T * H);
    for (auto& x : a) x = f2bf(nd(rng));
    Proj gate, up, down;
    gate.make(rng, H, I);
    up.make(rng, H, I);
    down.make(rng, I, H);
    CK(cudaDeviceSynchronize());

    Buf<unsigned short> d_a, d_g, d_u, d_act, d_out_q, d_out_c;
    Buf<unsigned char> d_a8, d_act8;
    Buf<int> d_off, d_sorted;
    d_a.alloc(a.size()); d_a.put(a);
    d_g.alloc((size_t)R * I); d_u.alloc((size_t)R * I); d_act.alloc((size_t)R * I);
    d_out_q.alloc((size_t)R * H); d_out_c.alloc((size_t)R * H);
    d_a8.alloc((size_t)T * H); d_act8.alloc((size_t)R * I);
    d_off.alloc(E + 1); d_off.put(offsets);
    d_sorted.alloc(R); d_sorted.put(sorted_tok);

    // ── q38 ──
    CUfunction k_a8 = m_q38.fn("moe_q38_a_to_e4m3"), k_gus = m_q38.fn("moe_q38_gate_up_silu");
    CUfunction k_dn = m_q38.fn("moe_q38_down"), k_silu = m_silu.fn("moe_silu_mul");
    const unsigned grid_m = ((R + E - 1) / E * 2 + 127) / 128;
    auto q38 = [&]() {
        Args x0; x0.add(d_a.p).add(d_a8.p).add(T * H);
        launch(k_a8, dim3((T * H / 4 + 255) / 256), dim3(256), 0, x0);
        Args x1;
        x1.add(d_a8.p).add(gate.q_pp.p).add(gate.q_ps.p).add(gate.d_s2.p)
          .add(up.q_pp.p).add(up.q_ps.p).add(up.d_s2.p).add(d_act8.p)
          .add(d_off.p).add(d_sorted.p).add(E).add(I).add(H);
        launch(k_gus, dim3(I / 64, grid_m, E), dim3(256), 0, x1);
        Args x2;
        x2.add(d_act8.p).add(down.q_pp.p).add(down.q_ps.p).add(down.d_s2.p).add(d_out_q.p)
          .add(d_off.p).add(E).add(H).add(I);
        launch(k_dn, dim3(H / 128, grid_m, E), dim3(256), 0, x2);
    };

    // ── CUTLASS grouped NVFP4 ──
    const size_t ws_bytes = (size_t)1 << 30;
    Buf<unsigned char> ws; ws.alloc(ws_bytes);
    int rc_gu = 0, rc_dn = 0;
    auto cutlass = [&]() {
        rc_gu = atlas_cutlass_nvfp4_grouped_gate_up_fused(
            d_a.p, d_sorted.p, gate.c_pp.data(), gate.c_ps.data(), gate.s2.data(),
            up.c_pp.data(), up.c_ps.data(), up.s2.data(), d_g.p, d_u.p, offsets.data(),
            E, I, H, ws.p, ws_bytes, 0);
        Args xs; xs.add(d_g.p).add(d_u.p).add(d_act.p).add(R * I);
        launch(k_silu, dim3((R * I + 255) / 256), dim3(256), 0, xs);
        rc_dn = atlas_cutlass_nvfp4_grouped_down(
            d_act.p, down.c_pp.data(), down.c_ps.data(), down.s2.data(), d_out_c.p,
            offsets.data(), E, H, I, ws.p, ws_bytes, 0);
    };

    q38();
    cutlass();
    CK(cudaDeviceSynchronize());
    if (rc_gu || rc_dn) { fprintf(stderr, "cutlass rc gate_up %d down %d\n", rc_gu, rc_dn); return 1; }

    // FP64 reference on sampled rows.
    const auto oq = d_out_q.get(), oc = d_out_c.get();
    double se_q = 0, se_c = 0, ss = 0, mx_q = 0, mx_c = 0;
    std::mt19937 pick_rng(7);
    for (int sidx = 0; sidx < 64; ++sidx) {
        const unsigned row = pick_rng() % R;
        unsigned e = 0;
        while ((unsigned)offsets[e + 1] <= row) ++e;
        const unsigned tok = sorted_tok[row];
        std::vector<double> act(I);
        for (unsigned n = 0; n < I; ++n) {
            double g = 0, u = 0;
            for (unsigned k = 0; k < H; ++k) {
                const double av = bf(a[(size_t)tok * H + k]);
                g += av * gate.w(e, n, k);
                u += av * up.w(e, n, k);
            }
            act[n] = g / (1.0 + exp(-g)) * u;
        }
        for (unsigned n = 0; n < H; ++n) {
            double y = 0;
            for (unsigned k = 0; k < I; ++k) y += act[k] * down.w(e, n, k);
            const double dq = bf(oq[(size_t)row * H + n]) - y, dc = bf(oc[(size_t)row * H + n]) - y;
            se_q += dq * dq; se_c += dc * dc; ss += y * y;
            mx_q = std::max(mx_q, fabs(dq)); mx_c = std::max(mx_c, fabs(dc));
        }
    }
    const double rms = sqrt(ss / (64.0 * H));
    printf("error vs FP64 reference (64 rows x %u): rel RMS  q38 %.4f  cutlass %.4f;  max|err|/RMS  q38 %.3f  cutlass %.3f\n",
           H, sqrt(se_q / ss), sqrt(se_c / ss), mx_q / rms, mx_c / rms);
    size_t same = 0;
    for (size_t i = 0; i < oq.size(); ++i) same += oq[i] == oc[i];
    printf("cutlass vs q38: %.2f%% of output elements byte-equal\n", 100.0 * same / oq.size());

    const double flop = 2.0 * R * H * 2.0 * I + 2.0 * R * I * H;
    const float t_q = time_ms(q38), t_c = time_ms(cutlass);
    printf("q38      %8.3f ms  %6.1f TFLOP/s\n", t_q, flop / t_q / 1e9);
    printf("cutlass  %8.3f ms  %6.1f TFLOP/s  (%.2fx vs q38; host-side problem setup included)\n",
           t_c, flop / t_c / 1e9, t_q / t_c);

    // ── Marlin MoE (timing only) ──
    if (have_marlin) {
        CUfunction k_m = m_marlin.fn("atlas_marlin_moe_nvfp4_m8", 96 * 1024);
        const int BLK = 8;
        std::vector<int> m_sorted, m_eids;
        for (unsigned e = 0; e < E; ++e) {
            const int c = (int)rows_of[e].size(), padded = (c + BLK - 1) / BLK * BLK;
            for (int i = 0; i < padded; ++i) m_sorted.push_back(i < c ? rows_of[e][i] : (int)(T * TOPK));
            for (int b = 0; b < padded / BLK; ++b) m_eids.push_back((int)e);
        }
        const int n_post = (int)m_sorted.size();
        Buf<int> dm_sorted, dm_eids, dm_npost, dm_locks;
        dm_sorted.alloc(m_sorted.size()); dm_sorted.put(m_sorted);
        dm_eids.alloc(m_eids.size()); dm_eids.put(m_eids);
        dm_npost.alloc(1); dm_npost.put(std::vector<int>{n_post});
        dm_locks.alloc(1 << 20);
        Buf<unsigned char> b_gu, b_dn, s_gu, s_dn;
        b_gu.alloc((size_t)E * H * 2 * I / 2); b_gu.fill(0x5A);
        b_dn.alloc((size_t)E * I * H / 2); b_dn.fill(0x5A);
        s_gu.alloc((size_t)E * H / 16 * 2 * I); s_gu.fill(0x30);
        s_dn.alloc((size_t)E * I / 16 * H); s_dn.fill(0x30);
        Buf<float> gs, c_tmp; gs.alloc(E); gs.fill(0); c_tmp.alloc((size_t)64 << 20);
        Buf<unsigned short> c_gu, c_dn;
        c_gu.alloc((size_t)T * TOPK * 2 * I); c_dn.alloc((size_t)T * TOPK * H);
        auto mm = [&](const void* A, const void* B, void* C, const void* S, int top_k, int M, int N, int K) {
            Args x;
            x.add(A).add(B).add(C).add((void*)c_tmp.p).add((void*)nullptr).add((void*)nullptr)
             .add(S).add((void*)gs.p).add((void*)nullptr).add((void*)nullptr)
             .add((void*)dm_sorted.p).add((void*)dm_eids.p).add((void*)dm_npost.p).add((void*)nullptr)
             .add(top_k).add(0).add(K / 16).add(M).add(N).add(K).add((void*)dm_locks.p)
             .add(0).add(0).add(1);
            launch(k_m, dim3(48), dim3(128), 96 * 1024, x);
        };
        auto marlin = [&]() {
            CK(cudaMemsetAsync(dm_locks.p, 0, 1 << 20, 0));
            mm(d_a.p, b_gu.p, c_gu.p, s_gu.p, TOPK, T, 2 * I, H);
            CK(cudaMemsetAsync(dm_locks.p, 0, 1 << 20, 0));
            mm(c_gu.p, b_dn.p, c_dn.p, s_dn.p, 1, T * TOPK, H, I);
        };
        marlin();
        CK(cudaDeviceSynchronize());
        const float t_m = time_ms(marlin, 2, 3);
        printf("marlin   %8.3f ms  %6.1f TFLOP/s  (%.2fx vs q38; gate_up + down, no SiLU)\n",
               t_m, flop / t_m / 1e9, t_q / t_m);
    }
    return 0;
}
