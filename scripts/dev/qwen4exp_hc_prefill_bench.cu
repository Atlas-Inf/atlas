// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of the qwen4_exp mHC PREFILL seam and collapse
// GEMMs at the real site shape (hidden 2560, hc 4, rank 320) over one
// 2048-token slab (hc_pre_gemm's SLAB):
//
//   seam   hc_post_vec + hc_pre_stage_bf16  vs  hc_post_stage_bf16
//          (highway and normed bytes must match)
//   gemm   down projection [T,10240] x [320,10240]^T on every
//          dense_gemm_bf16_pipelined build in the PTX dir (gemm_*.ptx, the
//          tile is parsed from the name: gemm_m<M>_n<N>_*), each against
//          gemm_base.ptx byte for byte
//   mix    hc_pre_mix, timed for reference
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4; C=kernels/gb10/common
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/hyper_connection.ptx $K/hyper_connection.cu
//   nvcc $F -o $D/gemm_base.ptx $C/dense_gemm_bf16.cu
//   nvcc $F -DDM_N_TILE=64 -o $D/gemm_m128_n64.ptx $C/dense_gemm_bf16.cu   # etc.
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_hc_prefill_bench.cu -lcuda -lcublasLt
//   $D/bench $D [T=2048] [HUM_BD=32] [all]
// Run it against the cuBLASLt the server links (the runtime image's CUDA
// 13.0 -- e.g. inside atlas-release-builder, not a newer host CUDA): the
// default up GEMM IS cuBLASLt, so the fused up+mix is exact only where that
// library's kernel is the plain in-order k-chain at every slab height
// (`qwen4exp_prefill_hc::UP_MIX_VERIFIED_LT` lists the versions checked).
#include "qwen4exp_ptx_harness.h"
#include <dirent.h>
#include <cublasLt.h>
#include <random>

static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H;
static const float EPS = 1e-6f;
static unsigned g_bd = 32;   // HUM_BD the hyper_connection PTX was built with (argv[3])
static bool g_all = false;   // argv[4] == "all": up+mix at every slab height T..1

// The production cuBLASLt BF16 call (crates/spark-runtime/src/cublaslt.rs
// gemm_bf16): FP32 compute, heuristic algorithm 0, 64 MiB workspace.
// out[M,N] = act[M,K] x W, W stored [N,K] (w_kn = false, opT) or [K,N] (opN).
// Returns the heuristic's split-K count.
static cublasLtHandle_t g_lt;
static void* g_ws;
static const size_t WS = 64ull << 20;
#define LT(x) do { int e_ = (int)(x); if (e_) { fprintf(stderr, "%s:%d cublasLt %d\n", __FILE__, __LINE__, e_); exit(1); } } while (0)
static int lt_gemm(const void* act, const void* w, void* out, int m, int n, int k, bool w_kn) {
    cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t la, lb, ld; cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = w_kn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    if (w_kn) LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, n, k, n));
    else LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, k));
    LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, m, n));
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res; int found = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, la, lb, ld, ld, pref, 1, &res, &found));
    int splitk = 1; size_t sz;
    cublasLtMatmulAlgoConfigGetAttribute(&res.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &sz);
    const float alpha = 1.f, beta = 0.f;
    LT(cublasLtMatmul(g_lt, desc, &alpha, w, la, act, lb, &beta, out, ld, out, ld, &res.algo, g_ws, WS, 0));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
    return splitk;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [T]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 2048;
    if (argc > 3) g_bd = atoi(argv[3]);
    g_all = argc > 4 && std::string(argv[4]) == "all";
    printf("cuBLASLt version %zu\n", cublasLtGetVersion());
    init_driver();
    PtxModule hc;
    hc.load(dir + "/hyper_connection.ptx");
    CUfunction k_post_vec = hc.fn("hc_post_vec");
    CUfunction k_post = hc.fn("hc_post");
    CUfunction k_stage = hc.fn("hc_pre_stage_bf16");
    CUfunction k_fused = hc.fn("hc_post_stage_bf16");
    CUfunction k_mix = hc.fn("hc_pre_mix");

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 2.0f);
    std::vector<float> streams((size_t)T * HCD), inj((size_t)T * HC);
    for (size_t i = 0; i < streams.size(); ++i) {
        // Per-token scale spread (1e-3 .. 1e2) so the RMS sums are not all alike,
        // plus exact zeros and negative zeros.
        float sc = powf(10.0f, (float)((i / HCD) % 6) - 3.0f);
        streams[i] = (i % 97 == 0) ? 0.0f : (i % 89 == 0) ? -0.0f : sc * nd(rng);
    }
    for (auto& x : inj) x = ud(rng);
    std::vector<unsigned short> block((size_t)T * H), norm_w(HCD), down((size_t)RANK * HCD);
    for (auto& x : block) x = f2bf(nd(rng));
    for (auto& x : norm_w) x = f2bf(0.1f * nd(rng));
    for (auto& x : down) x = f2bf(0.02f * nd(rng));

    Buf<float> d_s0, d_s1, d_inj;
    Buf<unsigned short> d_block, d_nw, d_n0, d_n1, d_down, d_up_pre, d_y;
    d_s0.alloc(streams.size()); d_s1.alloc(streams.size());
    d_inj.alloc(inj.size()); d_inj.put(inj);
    d_block.alloc(block.size()); d_block.put(block);
    d_nw.alloc(HCD); d_nw.put(norm_w);
    d_n0.alloc((size_t)T * HCD); d_n1.alloc((size_t)T * HCD);
    d_down.alloc(down.size()); d_down.put(down);
    d_up_pre.alloc((size_t)T * HCD); d_y.alloc((size_t)T * H);

    auto post_vec = [&](float* s) {
        Args a; a.add(d_block.p).add(s).add(d_inj.p).add(s).add(H).add(HC);
        launch(k_post_vec, dim3(T, (H / 4 + 63) / 64), dim3(64), 0, a);
    };
    auto post = [&](float* s) {
        Args a; a.add(d_block.p).add(s).add(d_inj.p).add(s).add(H).add(HC);
        launch(k_post, dim3(T), dim3(256), 0, a);
    };
    auto stage = [&](float* s, unsigned short* n) {
        Args a; a.add(s).add(d_nw.p).add(n).add(H).add(HC).add(EPS);
        launch(k_stage, dim3(T), dim3(1024), 0, a);
    };
    auto fused = [&](float* s, unsigned short* n) {
        Args a; a.add(d_block.p).add(s).add(d_inj.p).add(d_nw.p).add(n).add(H).add(HC).add(EPS);
        launch(k_fused, dim3(T), dim3(1024), 0, a);
    };
    auto mix = [&]() {
        Args a; a.add(d_n0.p).add(d_up_pre.p).add((unsigned short*)nullptr).add(d_y.p)
                 .add(d_inj.p).add(H).add(HC).add(1.0f / HC);
        launch(k_mix, dim3(T), dim3(256), 0, a);
    };

    bool ok = true;
    // ── seam ──
    for (int which = 0; which < 2; ++which) {
        d_s0.put(streams); d_s1.put(streams);
        d_n0.fill(0xAB); d_n1.fill(0xCD);
        if (which == 0) post_vec(d_s0.p); else post(d_s0.p);
        stage(d_s0.p, d_n0.p);
        fused(d_s1.p, d_n1.p);
        CK(cudaDeviceSynchronize());
        size_t ds = diff_bytes(d_s0.get(), d_s1.get());
        size_t dn = diff_bytes(d_n0.get(), d_n1.get());
        printf("bitwise seam (%s + hc_pre_stage_bf16 vs hc_post_stage_bf16) T=%u: highway %zu, normed %zu differing bytes\n",
               which == 0 ? "hc_post_vec" : "hc_post", T, ds, dn);
        ok = ok && ds == 0 && dn == 0;
    }
    d_s0.put(streams);
    const double seam_mb = (double)T * (H * 2 + HCD * 4 * 2 + HCD * 4 * 2 + HCD * 2) / 1e6;
    const double fused_mb = (double)T * (H * 2 + HCD * 4 * 2 + HCD * 2) / 1e6;
    float t_pv = time_ms([&] { post_vec(d_s0.p); });
    float t_st = time_ms([&] { stage(d_s0.p, d_n0.p); });
    float t_fu = time_ms([&] { fused(d_s0.p, d_n0.p); });
    float t_mx = time_ms([&] { mix(); });
    printf("T=%u seam: hc_post_vec %.3f ms + hc_pre_stage_bf16 %.3f ms = %.3f ms (%.0f GB/s)\n",
           T, t_pv, t_st, t_pv + t_st, seam_mb / (t_pv + t_st));
    printf("           hc_post_stage_bf16 %.3f ms (%.0f GB/s)  -> %.2fx\n",
           t_fu, fused_mb / t_fu, (t_pv + t_st) / t_fu);
    printf("     mix:  hc_pre_mix %.3f ms (%.0f GB/s)\n", t_mx,
           (double)T * (HCD * 2 * 2 + H * 2) / 1e6 / t_mx);

    // ── up projection + mix: cuBLASLt (opN on the stored [rank, hc*H]) +
    //    hc_pre_mix  vs  hc_up_mix_bf16, over several slab sizes ──
    LT(cublasLtCreate(&g_lt));
    CK(cudaMalloc(&g_ws, WS));
    CUfunction k_upmix_nt = hc.fn("hc_up_mix_bf16_nt");
    CUfunction k_tr = hc.fn("hc_transpose_bf16");
    {
        std::vector<unsigned short> upw((size_t)RANK * HCD), injw((size_t)HC * HCD);
        for (auto& x : upw) x = f2bf(0.05f * nd(rng));
        for (auto& x : injw) x = f2bf(0.02f * nd(rng));
        Buf<unsigned short> d_upw, d_injw, d_lowv, d_injpre, d_y2, d_upwt;
        d_upwt.alloc((size_t)RANK * HCD);
        Buf<float> d_inj2;
        d_upw.alloc(upw.size()); d_upw.put(upw);
        d_injw.alloc(injw.size()); d_injw.put(injw);
        d_lowv.alloc((size_t)T * RANK); d_injpre.alloc((size_t)T * HC);
        d_y2.alloc((size_t)T * H); d_inj2.alloc((size_t)T * HC);
        std::vector<unsigned short> lowv((size_t)T * RANK);
        for (auto& x : lowv) x = f2bf(0.3f * nd(rng));
        d_lowv.put(lowv);
        d_s0.put(streams);
        fused(d_s0.p, d_n0.p);   // realistic normed
        auto mix_m = [&](unsigned m) {
            Args a; a.add(d_n0.p).add(d_up_pre.p).add(d_injpre.p).add(d_y.p)
                     .add(d_inj.p).add(H).add(HC).add(1.0f / HC);
            launch(k_mix, dim3(m), dim3(256), 0, a);
        };
        auto transpose = [&]() {
            Args a; a.add(d_upw.p).add(d_upwt.p).add(RANK).add(HCD);
            launch(k_tr, dim3((HCD + 31) / 32, (RANK + 31) / 32), dim3(32, 32), 0, a);
        };
        auto upmix_nt = [&](unsigned m) {
            Args a; a.add(d_lowv.p).add(d_upwt.p).add(d_n0.p).add(d_injpre.p).add(d_y2.p)
                     .add(d_inj2.p).add(m).add(H).add(RANK).add(1.0f / HC);
            launch(k_upmix_nt, dim3(H / g_bd, (m + 127) / 128), dim3(256), 0, a);
        };
        transpose();
        // Every slab height T..1 (`all` as argv[4]), or a spread of them.
        std::vector<unsigned> ms;
        if (g_all) { for (unsigned m = T; m >= 1; --m) ms.push_back(m); }
        else for (unsigned m : {T, 2047u, 1838u, 1117u, 640u, 129u, 64u, 9u}) if (m <= T) ms.push_back(m);
        size_t bad_m = 0, split_m = 0;
        for (unsigned m : ms) {
            lt_gemm(d_n0.p, d_injw.p, d_injpre.p, m, HC, HCD, false);
            int sk = lt_gemm(d_lowv.p, d_upw.p, d_up_pre.p, m, HCD, RANK, true);
            d_y.fill(0x11); d_y2.fill(0x22); d_inj.fill(0x33); d_inj2.fill(0x44);
            mix_m(m);
            upmix_nt(m);
            CK(cudaDeviceSynchronize());
            auto y1 = d_y.get(), y2 = d_y2.get();
            auto i1 = d_inj.get(), i2 = d_inj2.get();
            y1.resize((size_t)m * H); y2.resize((size_t)m * H);
            i1.resize((size_t)m * HC); i2.resize((size_t)m * HC);
            size_t dy = diff_bytes(y1, y2), di = diff_bytes(i1, i2);
            if (!g_all || dy || di || sk != 1)
                printf("bitwise up+mix (cuBLASLt splitK=%d + hc_pre_mix vs hc_up_mix_bf16_nt) T=%u: y %zu, inj %zu differing bytes\n",
                       sk, m, dy, di);
            bad_m += dy || di;
            split_m += sk != 1;
        }
        printf("up+mix over %zu slab heights: %zu differ, %zu split-K\n", ms.size(), bad_m, split_m);
        ok = ok && bad_m == 0;
        float t_lt = time_ms([&] { lt_gemm(d_lowv.p, d_upw.p, d_up_pre.p, T, HCD, RANK, true); });
        float t_m = time_ms([&] { mix_m(T); });
        float t_f = time_ms([&] { upmix_nt(T); });
        float t_tr = time_ms([&] { transpose(); });
        float t_inj = time_ms([&] { lt_gemm(d_n0.p, d_injw.p, d_injpre.p, T, HC, HCD, false); });
        printf("T=%u up+mix: cuBLASLt up %.3f ms + hc_pre_mix %.3f ms = %.3f ms;  hc_up_mix_bf16_nt %.3f ms -> %.2fx"
               " (+ up_w transpose %.3f ms once per call)\n",
               T, t_lt, t_m, t_lt + t_m, t_f, (t_lt + t_m) / t_f, t_tr);
        printf("       (inject GEMM, cuBLASLt split-K, unchanged: %.3f ms)\n", t_inj);
    }

    // ── down GEMM tile variants ──
    std::vector<std::string> gemms;
    if (DIR* dp = opendir(dir.c_str())) {
        while (dirent* e = readdir(dp)) {
            std::string n = e->d_name;
            if (n.rfind("gemm_", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".ptx") gemms.push_back(n);
        }
        closedir(dp);
    }
    std::sort(gemms.begin(), gemms.end());
    std::vector<unsigned short> base_out;
    Buf<unsigned short> d_low;
    d_low.alloc((size_t)T * RANK);
    {
        // A = normed as staged above (realistic magnitudes).
        d_s0.put(streams);
        fused(d_s0.p, d_n0.p);
        CK(cudaDeviceSynchronize());
    }
    const double gflop = 2.0 * T * RANK * HCD / 1e9;
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& name : gemms) {
            bool is_base = name == "gemm_base.ptx";
            if ((pass == 0) != is_base) continue;
            unsigned bm = 128, bn = 128;
            sscanf(name.c_str(), "gemm_m%u_n%u", &bm, &bn);
            PtxModule m;
            if (!m.try_load(dir + "/" + name)) {
                printf("down GEMM %-28s does not load (shared memory?)\n", name.c_str());
                continue;
            }
            CUfunction k = m.fn("dense_gemm_bf16_pipelined");
            auto run = [&] {
                Args a; a.add(d_n0.p).add(d_down.p).add(d_low.p).add(T).add(RANK).add(HCD);
                launch(k, dim3((RANK + bn - 1) / bn, (T + bm - 1) / bm), dim3(256), 0, a);
            };
            d_low.fill(0x5A);
            run();
            CK(cudaDeviceSynchronize());
            auto out = d_low.get();
            if (is_base) base_out = out;
            size_t d = base_out.empty() ? 0 : diff_bytes(out, base_out);
            float t = time_ms(run);
            printf("down GEMM %-28s grid %3ux%-3u %.3f ms %6.2f TFLOP/s  bitwise vs base: %zu bytes differ\n",
                   name.c_str(), (RANK + bn - 1) / bn, (T + bm - 1) / bm, t, gflop / t, d);
            ok = ok && d == 0;
            CU(cuModuleUnload(m.mod));
        }
    }
    if (!base_out.empty()) {
        // Reference only -- the fast arm keeps the default's down GEMM: on
        // cuBLASLt 13.1.1 the heuristic picks a non-split kernel here, equal
        // to the tile kernel; on 13.0.0 (the runtime image) split-K 3, which
        // is not.
        d_low.fill(0x5A);
        int sk = lt_gemm(d_n0.p, d_down.p, d_low.p, T, RANK, HCD, false);
        CK(cudaDeviceSynchronize());
        size_t d = diff_bytes(d_low.get(), base_out);
        float t = time_ms([&] { lt_gemm(d_n0.p, d_down.p, d_low.p, T, RANK, HCD, false); });
        printf("down GEMM cuBLASLt (splitK=%d)          %.3f ms %6.2f TFLOP/s  bitwise vs base: %zu bytes differ (not used)\n",
               sk, t, gflop / t, d);
        // Every algorithm the heuristic offers for this shape: the non-split
        // ones are the candidates a pinned, exact down GEMM could use.
        cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t la, lb, ld; cublasLtMatmulPreference_t pref;
        LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, HCD, RANK, HCD));
        LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, HCD, T, HCD));
        LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, RANK, T, RANK));
        LT(cublasLtMatmulPreferenceCreate(&pref));
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
        cublasLtMatmulHeuristicResult_t res[16]; int found = 0;
        LT(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, la, lb, ld, ld, pref, 16, res, &found));
        for (int i = 0; i < found; ++i) {
            int id = 0, tile = 0, splitk = 1, red = 0, stages = 0; size_t sz;
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &sz);
            const float alpha = 1.f, beta = 0.f;
            auto go = [&] {
                LT(cublasLtMatmul(g_lt, desc, &alpha, d_down.p, la, d_n0.p, lb, &beta, d_low.p, ld, d_low.p, ld,
                                  &res[i].algo, g_ws, WS, 0));
            };
            d_low.fill(0x5A);
            go();
            CK(cudaDeviceSynchronize());
            size_t dd = diff_bytes(d_low.get(), base_out);
            float ta_ms = time_ms(go);
            printf("  heuristic #%d: algo %d tile %d stages %d splitK %d reduction %d  %.3f ms  bitwise vs base: %zu\n",
                   i, id, tile, stages, splitk, red, ta_ms, dd);
        }
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
