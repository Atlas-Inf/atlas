// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of the qwen4_exp routed-MoE PREFILL GEMMs at the
// TP=EP=2 shape: 256 local experts of hidden 2560 / intermediate 640, NVFP4
// weights (packed [K/2, N] bytes + per-16 FP8 scales + per-expert scale2),
// one 8192-token chunk routed top-10 over 512 experts with a lognormal
// popularity skew, this rank's half kept.
//
//   default   moe_w4a16_fused_gate_up_t_k64 -> moe_silu_mul ->
//             moe_w4a16_grouped_gemm_ptrtable_t_k64   (moe_w4a16_grouped_gemm.cu)
//   fast      moe_q38_gate_up_silu (gate+up+SiLU*mul, one kernel) ->
//             moe_q38_down                             (moe_prefill_q38.cu)
//   router    w4a16_gemm  vs  moe_q38_router_dequant + dense_gemm_bf16_pipelined
//             ([tokens, 2560] x 512 experts; logits must match byte for byte)
//
// The `act` (SiLU*up) and down outputs must match byte for byte.
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4; C=kernels/gb10/common
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/moe_w4a16.ptx $K/moe_w4a16_grouped_gemm.cu
//   nvcc $F -o $D/moe_silu_mul.ptx $C/moe_silu_mul.cu
//   nvcc $F -o $D/moe_prefill_q38.ptx $K/moe_prefill_q38.cu
//   nvcc $F -o $D/w4a16.ptx $K/w4a16_gemm.cu      # router: the default
//   nvcc $F -o $D/gemm.ptx $C/dense_gemm_bf16.cu  # router: the q38 arm
//   nvcc $F -o $D/moe.ptx $C/moe_permute.cu        # unpermute: the default
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_moe_prefill_bench.cu -lcuda
//   $D/bench $D [tokens=8192]
#include "qwen4exp_ptx_harness.h"
#include <cmath>
#include <random>

static const unsigned H = 2560, I = 640, E_ALL = 512, E = 256, TOPK = 10;

static unsigned char e4m3(float f) {
    // Positive normal values only (scales): round-to-nearest-even on the
    // 3-bit mantissa, bias 7.
    int ex;
    float m = frexpf(f, &ex);              // f = m * 2^ex, m in [0.5, 1)
    int e = ex - 1 + 7;                    // 1.mmm * 2^(ex-1)
    float frac = m * 2.0f - 1.0f;          // [0, 1)
    int mant = (int)lrintf(frac * 8.0f);
    if (mant == 8) { mant = 0; ++e; }
    if (e < 1) e = 1;
    if (e > 15) e = 15;
    return (unsigned char)((e << 3) | mant);
}

struct ExpertSet {
    Buf<unsigned char> packed, scales;
    Buf<unsigned long long> ptr_p, ptr_s;
    Buf<float> s2;
    void make(std::mt19937& rng, unsigned K, unsigned N) {
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N;
        std::vector<unsigned char> p(pb * E), s(sb * E);
        for (auto& x : p) x = (unsigned char)(rng() & 0xFF);
        std::uniform_real_distribution<float> u(0.004f, 0.06f);
        for (auto& x : s) x = e4m3(u(rng));
        packed.alloc(p.size()); packed.put(p);
        scales.alloc(s.size()); scales.put(s);
        std::vector<unsigned long long> pp(E), ps(E);
        for (unsigned e = 0; e < E; ++e) {
            pp[e] = (unsigned long long)(packed.p + e * pb);
            ps[e] = (unsigned long long)(scales.p + e * sb);
        }
        ptr_p.alloc(E); ptr_p.put(pp);
        ptr_s.alloc(E); ptr_s.put(ps);
        std::vector<float> v(E);
        std::uniform_real_distribution<float> u2(0.5f, 2.0f);
        for (auto& x : v) x = u2(rng);
        s2.alloc(E); s2.put(v);
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [tokens]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 8192;
    init_driver();
    PtxModule m_moe, m_silu, m_q38;
    m_moe.load(dir + "/moe_w4a16.ptx");
    m_silu.load(dir + "/moe_silu_mul.ptx");
    bool have_q38 = m_q38.try_load(dir + "/moe_prefill_q38.ptx");
    CUfunction k_gu = m_moe.fn("moe_w4a16_fused_gate_up_t_k64");
    CUfunction k_dn = m_moe.fn("moe_w4a16_grouped_gemm_ptrtable_t_k64");
    CUfunction k_silu = m_silu.fn("moe_silu_mul");

    // ── routing ──
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<double> pop(E_ALL);
    for (auto& p : pop) p = exp(0.8 * nd(rng));
    std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::vector<std::vector<int>> rows_of(E);
    std::vector<int> route_ids((size_t)T * TOPK);
    for (unsigned t = 0; t < T; ++t) {
        int chosen[TOPK];
        for (unsigned k = 0; k < TOPK; ++k) {
            int e;
            do {
                e = pick(rng);
            } while (std::find(chosen, chosen + k, e) != chosen + k);
            chosen[k] = e;
            route_ids[(size_t)t * TOPK + k] = e;
            if (e < (int)E) rows_of[e].push_back((int)t);
        }
    }
    std::vector<int> offsets(E + 1, 0), sorted;
    unsigned max_rows = 0;
    for (unsigned e = 0; e < E; ++e) {
        offsets[e + 1] = offsets[e] + (int)rows_of[e].size();
        max_rows = std::max<unsigned>(max_rows, rows_of[e].size());
        sorted.insert(sorted.end(), rows_of[e].begin(), rows_of[e].end());
    }
    const unsigned R = offsets[E];
    const unsigned avg = (R + E - 1) / E;
    const unsigned max_m_tiles = (max_rows + 63) / 64;
    const unsigned grid_m = std::min(max_m_tiles, (avg * 2 + 63) / 64);   // persistent default (F=2)
    printf("tokens %u, local routed rows %u (avg %u, max %u per expert), grid.y %u of %u m-tiles\n",
           T, R, avg, max_rows, grid_m, max_m_tiles);

    // ── data ──
    std::vector<unsigned short> a((size_t)T * H);
    for (auto& x : a) x = f2bf(nd(rng));
    Buf<unsigned short> d_a, d_g, d_u, d_act, d_act2, d_out, d_out2;
    d_a.alloc(a.size()); d_a.put(a);
    d_g.alloc((size_t)R * I); d_u.alloc((size_t)R * I);
    d_act.alloc((size_t)R * I); d_act2.alloc((size_t)R * I);
    d_out.alloc((size_t)R * H); d_out2.alloc((size_t)R * H);
    Buf<int> d_off, d_sorted;
    d_off.alloc(E + 1); d_off.put(offsets);
    d_sorted.alloc(R); d_sorted.put(sorted);
    ExpertSet gate, up, down;
    gate.make(rng, H, I);
    up.make(rng, H, I);
    down.make(rng, I, H);

    auto gate_up = [&]() {
        Args x;
        x.add(d_a.p).add(gate.ptr_p.p).add(gate.ptr_s.p).add(gate.s2.p)
         .add(up.ptr_p.p).add(up.ptr_s.p).add(up.s2.p).add(d_g.p).add(d_u.p)
         .add(d_off.p).add(d_sorted.p).add(E).add(I).add(H);
        launch(k_gu, dim3(2 * I / 128, grid_m, E), dim3(128), 0, x);
    };
    auto silu = [&]() {
        Args x;
        x.add(d_g.p).add(d_u.p).add(d_act.p).add(R * I);
        launch(k_silu, dim3((R * I + 255) / 256), dim3(256), 0, x);
    };
    auto down_k = [&](unsigned short* act, unsigned short* out) {
        Args x;
        x.add(act).add(down.ptr_p.p).add(down.ptr_s.p).add(down.s2.p).add(out)
         .add(d_off.p).add((int*)nullptr).add(E).add(H).add(I);
        launch(k_dn, dim3(H / 128, grid_m, E), dim3(128), 0, x);
    };
    gate_up();
    silu();
    down_k(d_act.p, d_out.p);
    CK(cudaDeviceSynchronize());
    const double gu_tf = 2.0 * R * H * 2.0 * I / 1e12, dn_tf = 2.0 * R * I * H / 1e12;
    float t_gu = time_ms(gate_up), t_si = time_ms(silu);
    float t_dn = time_ms([&] { down_k(d_act.p, d_out.p); });
    printf("default: gate_up %.3f ms (%.1f TFLOP/s)  silu %.3f ms  down %.3f ms (%.1f TFLOP/s)  total %.3f ms\n",
           t_gu, gu_tf / t_gu * 1e3, t_si, t_dn, dn_tf / t_dn * 1e3, t_gu + t_si + t_dn);

    bool ok = true;
    if (have_q38) {
        CUfunction k_a8 = m_q38.fn("moe_q38_a_to_e4m3");
        CUfunction k_gus = m_q38.fn("moe_q38_gate_up_silu");
        CUfunction k_dn2 = m_q38.fn("moe_q38_down");
        const unsigned bm = 128, grid_m2 = (avg * 2 + bm - 1) / bm;
        Buf<unsigned char> d_a8, d_act8, d_act8_ref;
        d_a8.alloc((size_t)T * H);
        d_act8.alloc((size_t)R * I);
        d_act8_ref.alloc((size_t)R * I);
        auto to_e4m3 = [&](const unsigned short* src, unsigned char* dst, unsigned n) {
            Args x;
            x.add(src).add(dst).add(n);
            launch(k_a8, dim3((n / 4 + 255) / 256), dim3(256), 0, x);
        };
        auto gate_up_silu = [&]() {
            Args x;
            x.add(d_a8.p).add(gate.ptr_p.p).add(gate.ptr_s.p).add(gate.s2.p)
             .add(up.ptr_p.p).add(up.ptr_s.p).add(up.s2.p).add(d_act8.p)
             .add(d_off.p).add(d_sorted.p).add(E).add(I).add(H);
            launch(k_gus, dim3(I / 64, grid_m2, E), dim3(256), 0, x);
        };
        auto down2 = [&]() {
            Args x;
            x.add(d_act8.p).add(down.ptr_p.p).add(down.ptr_s.p).add(down.s2.p).add(d_out2.p)
             .add(d_off.p).add(E).add(H).add(I);
            launch(k_dn2, dim3(H / 128, grid_m2, E), dim3(256), 0, x);
        };
        // The default's activation as the default down GEMM would see it:
        // the E4M3 image of the BF16 `moe_silu_mul` output.
        to_e4m3(d_act.p, d_act8_ref.p, R * I);
        to_e4m3(d_a.p, d_a8.p, T * H);
        d_act8.fill(0x77);
        gate_up_silu();
        d_out2.fill(0x66);
        down2();
        CK(cudaDeviceSynchronize());
        size_t d = diff_bytes(d_act8_ref.get(), d_act8.get());
        printf("bitwise act (E4M3 image of gate_up + silu vs moe_q38_gate_up_silu): %zu differing bytes\n", d);
        size_t dd = diff_bytes(d_out.get(), d_out2.get());
        printf("bitwise down output (default chain vs q38 chain): %zu differing bytes\n", dd);
        ok = ok && d == 0 && dd == 0;
        float t_a = time_ms([&] { to_e4m3(d_a.p, d_a8.p, T * H); });
        float t = time_ms(gate_up_silu);
        float t2 = time_ms(down2);
        printf("fast:    a->e4m3 %.3f ms  gate_up+silu %.3f ms (%.1f TFLOP/s)  down %.3f ms (%.1f TFLOP/s)\n",
               t_a, t, gu_tf / t * 1e3, t2, dn_tf / t2 * 1e3);
        printf("MoE GEMMs: default %.3f ms -> fast %.3f ms (%.2fx)\n", t_gu + t_si + t_dn, t_a + t + t2,
               (t_gu + t_si + t_dn) / (t_a + t + t2));
    }
    // ── router: w4a16_gemm vs moe_q38_router_dequant + dense_gemm_bf16_pipelined ──
    PtxModule m_w4, m_gemm;
    if (have_q38 && m_w4.try_load(dir + "/w4a16.ptx") && m_gemm.try_load(dir + "/gemm.ptx")) {
        CUfunction k_w4 = m_w4.fn("w4a16_gemm");
        CUfunction k_dq = m_q38.fn("moe_q38_router_dequant");
        CUfunction k_dg = m_gemm.fn("dense_gemm_bf16_pipelined");
        const unsigned NR = E_ALL;
        std::vector<unsigned char> rp((size_t)NR * H / 2), rs((size_t)NR * H / 16);
        for (auto& x : rp) x = (unsigned char)(rng() & 0xFF);
        std::uniform_real_distribution<float> us(0.004f, 0.06f);
        for (auto& x : rs) x = e4m3(us(rng));
        Buf<unsigned char> d_rp, d_rs;
        Buf<unsigned short> d_rw, d_l1, d_l2;
        d_rp.alloc(rp.size()); d_rp.put(rp);
        d_rs.alloc(rs.size()); d_rs.put(rs);
        d_rw.alloc((size_t)NR * H);
        d_l1.alloc((size_t)T * NR); d_l2.alloc((size_t)T * NR);
        const float s2 = 0.73f;
        auto router_old = [&]() {
            Args x;
            x.add(d_a.p).add(d_rp.p).add(d_rs.p).add(s2).add(d_l1.p).add(T).add(NR).add(H);
            launch(k_w4, dim3(NR / 64, (T + 63) / 64), dim3(128), 0, x);
        };
        auto router_new = [&]() {
            Args x;
            x.add(d_rp.p).add(d_rs.p).add(s2).add(d_rw.p).add(NR).add(H);
            launch(k_dq, dim3((NR * H / 2 + 255) / 256), dim3(256), 0, x);
            Args y;
            y.add(d_a.p).add(d_rw.p).add(d_l2.p).add(T).add(NR).add(H);
            launch(k_dg, dim3((NR + 127) / 128, (T + 127) / 128), dim3(256), 0, y);
        };
        d_l1.fill(0x12); d_l2.fill(0x34);
        router_old();
        router_new();
        CK(cudaDeviceSynchronize());
        size_t d = diff_bytes(d_l1.get(), d_l2.get());
        printf("bitwise router logits (w4a16_gemm vs dequant + dense_gemm_bf16_pipelined): %zu differing bytes\n", d);
        ok = ok && d == 0;
        const double r_tf = 2.0 * T * NR * H / 1e12;
        float t1 = time_ms(router_old), t2 = time_ms(router_new);
        printf("router: w4a16_gemm %.3f ms (%.1f TFLOP/s) -> dequant + dense_gemm %.3f ms (%.1f TFLOP/s), %.2fx\n",
               t1, r_tf / t1 * 1e3, t2, r_tf / t2 * 1e3, t1 / t2);

        // ── shared expert: w4a16_gemm_t x3 + moe_silu_mul vs the dense q38 pair ──
        CUfunction k_wt = m_w4.fn("w4a16_gemm_t");
        CUfunction k_a8 = m_q38.fn("moe_q38_a_to_e4m3");
        CUfunction k_sgu = m_q38.fn("moe_q38_dense_gate_up_silu");
        CUfunction k_sdn = m_q38.fn("moe_q38_dense_down");
        auto mk = [&](unsigned K, unsigned N, Buf<unsigned char>& p, Buf<unsigned char>& sc) {
            std::vector<unsigned char> hp((size_t)K / 2 * N), hs((size_t)K / 16 * N);
            for (auto& x : hp) x = (unsigned char)(rng() & 0xFF);
            for (auto& x : hs) x = e4m3(us(rng));
            p.alloc(hp.size()); p.put(hp);
            sc.alloc(hs.size()); sc.put(hs);
        };
        Buf<unsigned char> sgp, sgs, sup, sus, sdp, sds, d_sa8, d_sact8;
        mk(H, I, sgp, sgs); mk(H, I, sup, sus); mk(I, H, sdp, sds);
        const float s2g = 0.9f, s2u = 1.1f, s2d = 0.8f;
        Buf<unsigned short> d_sg, d_su, d_so1, d_so2;
        d_sg.alloc((size_t)T * I); d_su.alloc((size_t)T * I);
        d_so1.alloc((size_t)T * H); d_so2.alloc((size_t)T * H);
        d_sa8.alloc((size_t)T * H); d_sact8.alloc((size_t)T * I);
        auto wt = [&](const void* a, Buf<unsigned char>& p, Buf<unsigned char>& sc, float s2v,
                      unsigned short* c, unsigned N, unsigned K) {
            Args x;
            x.add(a).add(p.p).add(sc.p).add(s2v).add(c).add(T).add(N).add(K).add(N);
            launch(k_wt, dim3((N + 127) / 128, (T + 63) / 64), dim3(128), 0, x);
        };
        auto shared_old = [&]() {
            wt(d_a.p, sgp, sgs, s2g, d_sg.p, I, H);
            wt(d_a.p, sup, sus, s2u, d_su.p, I, H);
            Args x;
            x.add(d_sg.p).add(d_su.p).add(d_sg.p).add(T * I);
            launch(k_silu, dim3((T * I + 255) / 256), dim3(256), 0, x);
            wt(d_sg.p, sdp, sds, s2d, d_so1.p, H, I);
        };
        auto shared_new = [&]() {
            Args a;
            a.add(d_a.p).add(d_sa8.p).add(T * H);
            launch(k_a8, dim3((T * H / 4 + 255) / 256), dim3(256), 0, a);
            Args x;
            x.add(d_sa8.p).add(sgp.p).add(sgs.p).add(s2g).add(sup.p).add(sus.p).add(s2u)
             .add(d_sact8.p).add(T).add(I).add(H);
            launch(k_sgu, dim3(I / 64, (T + 127) / 128), dim3(256), 0, x);
            Args y;
            y.add(d_sact8.p).add(sdp.p).add(sds.p).add(s2d).add(d_so2.p).add(T).add(H).add(I);
            launch(k_sdn, dim3(H / 128, (T + 127) / 128), dim3(256), 0, y);
        };
        d_so1.fill(0x21); d_so2.fill(0x43);
        shared_old();
        shared_new();
        CK(cudaDeviceSynchronize());
        size_t ds = diff_bytes(d_so1.get(), d_so2.get());
        printf("bitwise shared expert output (w4a16_gemm_t chain vs q38 dense): %zu differing bytes\n", ds);
        ok = ok && ds == 0;
        const double s_tf = 2.0 * T * (2.0 * I * H + (double)I * H) / 1e12;
        float ts1 = time_ms(shared_old), ts2 = time_ms(shared_new);
        printf("shared expert: default %.3f ms (%.1f TFLOP/s) -> q38 %.3f ms (%.1f TFLOP/s), %.2fx\n",
               ts1, s_tf / ts1 * 1e3, ts2, s_tf / ts2 * 1e3, ts1 / ts2);
    }
    // ── unpermute: dense (remote rows zeroed) vs local-only (remote rows garbage) ──
    if (have_q38) {
        PtxModule m_perm;
        if (m_perm.try_load(dir + "/moe.ptx")) {
            CUfunction k_dense = m_perm.fn("moe_unpermute_reduce_indexed");
            CUfunction k_local = m_q38.fn("moe_q38_unpermute_local");
            const size_t rows = (size_t)T * TOPK;
            std::vector<unsigned short> eo(rows * H), eo_g(rows * H);
            std::vector<int> perm(rows);
            std::vector<float> wts(rows);
            std::uniform_real_distribution<float> uw(0.01f, 0.3f);
            for (size_t r = 0; r < rows; ++r) {
                perm[r] = (int)r;
                wts[r] = uw(rng);
                const bool local = route_ids[r] < (int)E;
                for (unsigned c = 0; c < H; ++c) {
                    unsigned short v = f2bf(nd(rng));
                    eo[r * H + c] = local ? v : 0;
                    eo_g[r * H + c] = local ? v : 0x7F7Fu;   // garbage where unread
                }
            }
            Buf<unsigned short> d_eo, d_eo_g, d_o1, d_o2;
            Buf<int> d_perm, d_ids;
            Buf<float> d_w;
            d_eo.alloc(eo.size()); d_eo.put(eo);
            d_eo_g.alloc(eo_g.size()); d_eo_g.put(eo_g);
            d_perm.alloc(rows); d_perm.put(perm);
            d_ids.alloc(rows); d_ids.put(route_ids);
            d_w.alloc(rows); d_w.put(wts);
            d_o1.alloc((size_t)T * H); d_o2.alloc((size_t)T * H);
            auto dense = [&]() {
                Args x;
                x.add(d_eo.p).add(d_o1.p).add(d_perm.p).add(d_w.p).add(H).add(T).add(TOPK);
                launch(k_dense, dim3(T), dim3(256), 0, x);
            };
            auto local = [&]() {
                Args x;
                x.add(d_eo_g.p).add(d_o2.p).add(d_perm.p).add(d_ids.p).add(d_w.p)
                 .add(H).add(T).add(TOPK).add(0u).add(E);
                launch(k_local, dim3(T), dim3(H / 8), 0, x);
            };
            dense();
            local();
            CK(cudaDeviceSynchronize());
            size_t d = diff_bytes(d_o1.get(), d_o2.get());
            printf("bitwise unpermute (dense over zeroed remote rows vs local-only over garbage): %zu differing bytes\n", d);
            ok = ok && d == 0;
            float t1 = time_ms(dense), t2 = time_ms(local);
            printf("unpermute: dense %.3f ms -> local-only %.3f ms (%.2fx); the dense arm also needs "
                   "%.0f MB of memsets a layer\n", t1, t2, t1 / t2,
                   (double)rows * (H + 2 * I) * 2 / 1e6);
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
