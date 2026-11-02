// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the Qwen3.8-Flash-Next GDN projections in BF16 (as
// shipped, the default decode path) against the opt-in FP8 copy
// (ATLAS_QWEN4EXP_FP8_GDN=1): 128x128 block-scaled E4M3 weights read by the
// existing W8A16 kernels. Real shapes, synthetic weights:
//
//   in_proj_qkvz  [8192 x 2560] (TP2 shard)  [16384 x 2560] (TP1)
//   out_proj      [2560 x 3072] (TP2 shard)  [2560 x 6144]  (TP1)
//
// Decode, T = 1, 2, 4 activation rows, the arms serving dispatches:
//   BF16  T=1 dense_gemv_bf16, T=2 dense_gemv_bf16_batch2, T=4 dense_gemv_bf16_batchm
//   FP8   T=1 w8a16_gemv,      T=2/4 w8a16_gemv_batch4 (one weight pass for all rows)
//   (ref) per-row FP8 dense_gemv_fp8w / dense_gemv_fp8w_batchm (ATLAS_GDN_FP8_DECODE's kernels)
//   (ref) NVFP4 T=1 w4a16_gemv_sw on the ATLAS_QWEN4EXP_BF16_GDN=0 requant
// Prefill (M = 512, 2048, 8192): cuBLASLt BF16 (the QKVZ arm),
//   dense_gemm_bf16_pipelined (the out_proj arm) and w8a16_gemm_pipelined
//   (what prefill would read if the BF16 copy were freed).
//
// Weights are the load path's: BF16 -> quantize_bf16_to_fp8_blockscaled, so
// the error is the error serving would see on weights of this distribution.
// Error is reported against the BF16 kernel's output (what the switch changes)
// and both against an FP64 CPU reference of the BF16 weights (the floor).
// Timing cycles a ring of weight copies (>= 256 MB per precision, ~10x the
// 24 MiB L2) so every launch streams from DRAM as serving's 72 distinct
// projections per token do; it is GPU time between events around 200
// back-to-back launches, so the inter-launch gap is included.
//
// Build: each kernel source is its own module (their macros and LUT symbols
// collide in one translation unit), loaded through the driver API as Atlas
// loads PTX:
//   K=kernels/gb10/common
//   for f in dense_gemv_bf16 dense_gemv_bf16_batch2 dense_gemv_bf16_batchm \
//            dense_gemv_fp8w dense_gemv_fp8w_batchm w8a16_gemv w8a16_gemv_batch4 \
//            w8a16_gemm_pipelined dense_gemm_bf16 quantize_bf16_to_fp8_blockscaled \
//            quantize_bf16_to_nvfp4 w4a16_gemv; do
//     nvcc -cubin -arch=sm_121a -O3 --fmad=false $K/$f.cu -o $f.cubin; done
//   nvcc -arch=sm_121a -O3 -std=c++17 scripts/dev/qwen4exp_gdn_fp8_bench.cu \
//        -o gdn_fp8_bench -lcuda -lcublasLt
//   ./gdn_fp8_bench [cubin_dir=.] [iters=200] [tp]   (`tp`: the TP exactness check only)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s = nullptr; \
    cuGetErrorString(r, &s); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s ? s : "?"); exit(1); } } while (0)
#define LT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d cublasLt status %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

static std::string g_dir = ".";
static int g_iters = 200;

static CUfunction load(const char* module, const char* fn) {
    static std::vector<std::pair<std::string, CUmodule>> mods;
    CUmodule m = nullptr;
    for (auto& p : mods) if (p.first == module) m = p.second;
    if (!m) {
        CU(cuModuleLoad(&m, (g_dir + "/" + module + ".cubin").c_str()));
        mods.push_back({module, m});
    }
    CUfunction f;
    CU(cuModuleGetFunction(&f, m, fn));
    return f;
}

static void launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx, std::vector<void*> args) {
    CU(cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, 0, args.data(), nullptr));
}

static unsigned short tobf(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}
static float frombf(unsigned short u) {
    unsigned int x = (unsigned int)u << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static void* dalloc(size_t bytes) {
    void* p;
    CK(cudaMalloc(&p, bytes));
    return p;
}

// GPU time per launch over `g_iters` back-to-back launches.
template <typename F>
static double time_us(F f) {
    for (int i = 0; i < 10; i++) f(i);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, 0));
    for (int i = 0; i < g_iters; i++) f(i);
    CK(cudaEventRecord(b, 0));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return 1000.0 * ms / g_iters;
}

struct Err {
    double rel_inf, rel_l2, max_rel_big, cos;
};
// rel_inf = max|y-r| / max|r|; rel_l2 = ||y-r|| / ||r||;
// max_rel_big = max |y-r|/|r| over outputs with |r| >= rms(r) (the elementwise
// relative error where it is meaningful, i.e. not near a zero crossing).
static Err compare(const std::vector<double>& y, const std::vector<double>& r) {
    double dmax = 0, rmax = 0, d2 = 0, r2 = 0, y2 = 0, yr = 0;
    for (size_t i = 0; i < r.size(); i++) {
        double d = y[i] - r[i];
        dmax = std::max(dmax, std::fabs(d));
        rmax = std::max(rmax, std::fabs(r[i]));
        d2 += d * d;
        r2 += r[i] * r[i];
        y2 += y[i] * y[i];
        yr += y[i] * r[i];
    }
    double rms = std::sqrt(r2 / r.size()), big = 0;
    for (size_t i = 0; i < r.size(); i++)
        if (std::fabs(r[i]) >= rms) big = std::max(big, std::fabs(y[i] - r[i]) / std::fabs(r[i]));
    return {dmax / rmax, std::sqrt(d2 / r2), big, yr / std::sqrt(y2 * r2)};
}

static std::vector<double> down_bf16(const void* d, size_t n) {
    std::vector<unsigned short> h(n);
    CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<double> o(n);
    for (size_t i = 0; i < n; i++) o[i] = frombf(h[i]);
    return o;
}

static cublasLtHandle_t g_lt;
static void* g_ws;
static const size_t WS = 64ull << 20;  // Atlas's cuBLASLt workspace

// Row-major out[M,N] = act[M,K] @ w[N,K]^T, BF16 in/out, FP32 compute, the
// heuristic's first algorithm -- spark_runtime::cublaslt::gemm_bf16's recipe.
static void lt_gemm(const void* act, const void* w, void* out, unsigned m, unsigned n, unsigned k) {
    cublasLtMatmulDesc_t desc;
    LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtMatrixLayout_t la, lb, ld;
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, m, k));
    LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, m, n));
    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof(WS)));
    cublasLtMatmulHeuristicResult_t res;
    int got = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, la, lb, ld, ld, pref, 1, &res, &got));
    if (got < 1) { fprintf(stderr, "no cuBLASLt algo\n"); exit(1); }
    float alpha = 1.f, beta = 0.f;
    LT(cublasLtMatmul(g_lt, desc, &alpha, w, la, act, lb, &beta, out, ld, out, ld, &res.algo, g_ws, WS, 0));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc);
}

static void run_shape(const char* name, unsigned N, unsigned K, std::mt19937_64& rng) {
    const unsigned MAXT = 4;
    printf("\n=== %s  [%u x %u] ===\n", name, N, K);

    // Weights: rows of varying norm, input channels of varying gain (the
    // structure a 128x128 block scale sees and a per-row scale does not), and
    // a heavy tail (0.1%% of entries x8).
    std::normal_distribution<float> g(0.f, 1.f);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> rs(N), cs(K);
    for (auto& v : rs) v = 0.02f * std::exp(0.3f * g(rng));
    for (auto& v : cs) v = std::exp(0.25f * g(rng));
    std::vector<unsigned short> wh((size_t)N * K);
    for (unsigned n = 0; n < N; n++)
        for (unsigned k = 0; k < K; k++) {
            float v = rs[n] * cs[k] * g(rng);
            if (u(rng) < 1e-3f) v *= 8.f;
            wh[(size_t)n * K + k] = tobf(v);
        }
    // Activations: unit scale with a few outlier channels (x10).
    std::vector<unsigned short> xh((size_t)MAXT * K);
    for (unsigned t = 0; t < MAXT; t++)
        for (unsigned k = 0; k < K; k++) {
            float v = g(rng);
            if (k % 509 == 7) v *= 10.f;
            xh[(size_t)t * K + k] = tobf(v);
        }

    // FP64 reference of the BF16 weights.
    std::vector<double> ref((size_t)MAXT * N);
    for (unsigned t = 0; t < MAXT; t++)
        for (unsigned n = 0; n < N; n++) {
            double s = 0;
            const unsigned short* w = &wh[(size_t)n * K];
            const unsigned short* x = &xh[(size_t)t * K];
            for (unsigned k = 0; k < K; k++) s += (double)frombf(w[k]) * (double)frombf(x[k]);
            ref[(size_t)t * N + n] = s;
        }

    const size_t wbytes16 = (size_t)N * K * 2, wbytes8 = (size_t)N * K;
    const size_t sbytes = (size_t)((N + 127) / 128) * ((K + 127) / 128) * 4;
    const size_t ring16 = std::max<size_t>(2, (256ull << 20) / wbytes16);
    const size_t ring8 = std::max<size_t>(2, (256ull << 20) / wbytes8);
    std::vector<void*> w16(ring16), w8(ring8), w8r(ring8);
    for (auto& p : w16) p = dalloc(wbytes16);
    for (auto& p : w8) p = dalloc(wbytes8);
    for (auto& p : w8r) p = dalloc(wbytes8);
    void* scale = dalloc(sbytes);
    void* rscale = dalloc((size_t)N * 4);
    CK(cudaMemcpy(w16[0], wh.data(), wbytes16, cudaMemcpyHostToDevice));
    for (size_t i = 1; i < ring16; i++) CK(cudaMemcpy(w16[i], w16[0], wbytes16, cudaMemcpyDeviceToDevice));

    // The load path: block-scaled quantizer (grid (k_blocks, n_blocks), 256).
    {
        CUfunction q = load("quantize_bf16_to_fp8_blockscaled", "quantize_bf16_to_fp8_blockscaled");
        void* a0 = w16[0]; void* a1 = w8[0]; void* a2 = scale; unsigned n = N, k = K;
        launch(q, (K + 127) / 128, (N + 127) / 128, 256, {&a0, &a1, &a2, &n, &k});
        CUfunction qr = load("dense_gemv_fp8w", "quantize_bf16_to_fp8");
        void* b1 = w8r[0]; void* b2 = rscale;
        launch(qr, N, 1, 256, {&a0, &b1, &b2, &n, &k});
        CK(cudaDeviceSynchronize());
    }
    for (size_t i = 1; i < ring8; i++) {
        CK(cudaMemcpy(w8[i], w8[0], wbytes8, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(w8r[i], w8r[0], wbytes8, cudaMemcpyDeviceToDevice));
    }
    printf("memory: BF16 %.1f MB, FP8 %.1f MB + block scales %.1f KB (%.1f MB saved / copy)\n",
           wbytes16 / 1e6, wbytes8 / 1e6, sbytes / 1e3, (wbytes16 - wbytes8 - sbytes) / 1e6);

    void* x = dalloc((size_t)MAXT * K * 2);
    CK(cudaMemcpy(x, xh.data(), (size_t)MAXT * K * 2, cudaMemcpyHostToDevice));
    void* y16 = dalloc((size_t)MAXT * N * 2);
    void* y8 = dalloc((size_t)MAXT * N * 2);
    void* y8r = dalloc((size_t)MAXT * N * 2);
    void* yt = dalloc((size_t)MAXT * N * 2);

    CUfunction k_bf16 = load("dense_gemv_bf16", "dense_gemv_bf16");
    CUfunction k_bf16_b2 = load("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2");
    CUfunction k_bf16_bm = load("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    CUfunction k_w8 = load("w8a16_gemv", "w8a16_gemv");
    CUfunction k_w8_b4 = load("w8a16_gemv_batch4", "w8a16_gemv_batch4");
    CUfunction k_fp8w = load("dense_gemv_fp8w", "dense_gemv_fp8w");
    CUfunction k_fp8w_bm = load("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm");

    // One launch of each arm at T rows, weight copy `i` of the ring.
    auto bf16_arm = [&](unsigned T, size_t i, void* out) {
        void* w = w16[i % ring16]; unsigned n = N, k = K, m = T, st = N;
        if (T == 1) launch(k_bf16, (N + 3) / 4, 1, 256, {&x, &w, &out, &n, &k});
        else if (T == 2) launch(k_bf16_b2, (N + 3) / 4, 1, 256, {&x, &w, &out, &n, &k, &st});
        else launch(k_bf16_bm, (N + 3) / 4, 1, 256, {&x, &w, &out, &m, &n, &k, &st});
    };
    auto fp8_arm = [&](unsigned T, size_t i, void* out) {
        void* w = w8[i % ring8]; unsigned n = N, k = K, m = T;
        if (T == 1) launch(k_w8, (N + 3) / 4, 1, 256, {&x, &w, &scale, &out, &n, &k});
        else launch(k_w8_b4, (N + 3) / 4, 1, 256, {&x, &w, &scale, &out, &m, &n, &k});
    };
    auto fp8r_arm = [&](unsigned T, size_t i, void* out) {
        void* w = w8r[i % ring8]; unsigned n = N, k = K, m = T, st = N;
        if (T == 1) launch(k_fp8w, (N + 3) / 4, 1, 256, {&x, &w, &rscale, &out, &n, &k});
        else launch(k_fp8w_bm, (N + 3) / 4, 1, 256, {&x, &w, &rscale, &out, &m, &n, &k, &st});
    };

    printf("%-3s %-26s %9s %8s | %-24s %9s %8s %7s | %-20s %9s %7s\n", "T", "BF16 arm", "us", "GB/s",
           "FP8 block arm", "us", "GB/s", "speedup", "FP8 per-row (ref)", "us", "speedup");
    for (unsigned T : {1u, 2u, 4u}) {
        double t16 = time_us([&](int i) { bf16_arm(T, i, y16); });
        double t8 = time_us([&](int i) { fp8_arm(T, i, y8); });
        double t8r = time_us([&](int i) { fp8r_arm(T, i, y8r); });
        const char* a16 = T == 1 ? "dense_gemv_bf16" : T == 2 ? "dense_gemv_bf16_batch2" : "dense_gemv_bf16_batchm";
        const char* a8 = T == 1 ? "w8a16_gemv" : "w8a16_gemv_batch4";
        const char* a8r = T == 1 ? "dense_gemv_fp8w" : "dense_gemv_fp8w_batchm";
        printf("%-3u %-26s %9.1f %8.1f | %-24s %9.1f %8.1f %6.2fx | %-20s %9.1f %6.2fx\n", T, a16, t16,
               wbytes16 / t16 / 1e3, a8, t8, (wbytes8 + sbytes) / t8 / 1e3, t16 / t8, a8r, t8r, t16 / t8r);
    }

    // Numerics on copy 0 at T=4 (every row computed by each batched arm).
    bf16_arm(MAXT, 0, y16);
    fp8_arm(MAXT, 0, y8);
    fp8r_arm(MAXT, 0, y8r);
    CK(cudaDeviceSynchronize());
    auto o16 = down_bf16(y16, (size_t)MAXT * N);
    auto o8 = down_bf16(y8, (size_t)MAXT * N);
    auto o8r = down_bf16(y8r, (size_t)MAXT * N);
    auto pr = [](const char* what, Err e) {
        printf("  %-34s rel_inf %.3e  rel_l2 %.3e  max_rel(|r|>=rms) %.3e  cos %.8f\n", what, e.rel_inf,
               e.rel_l2, e.max_rel_big, e.cos);
    };
    pr("BF16 GEMV   vs FP64(BF16 W)", compare(o16, ref));
    pr("FP8 block   vs BF16 GEMV", compare(o8, o16));
    pr("FP8 block   vs FP64(BF16 W)", compare(o8, ref));
    pr("FP8 per-row vs BF16 GEMV", compare(o8r, o16));

    // For scale: the lossy NVFP4 requant ATLAS_QWEN4EXP_BF16_GDN=0 serves
    // (per-tensor scale2 + 16-wide E4M3 group scales, w4a16_gemv_sw at T=1).
    {
        CUfunction amax = load("quantize_bf16_to_nvfp4", "nvfp4_global_absmax");
        CUfunction q4 = load("quantize_bf16_to_nvfp4", "quantize_bf16_to_nvfp4");
        CUfunction g4 = load("w4a16_gemv", "w4a16_gemv_sw");
        const size_t pb = (size_t)N * K / 2, sb = (size_t)N * K / 16;
        const size_t ring4 = std::max<size_t>(2, (256ull << 20) / (pb + sb));
        std::vector<void*> wp(ring4), ws(ring4);
        for (size_t i = 0; i < ring4; i++) { wp[i] = dalloc(pb); ws[i] = dalloc(sb); }
        float* dmax = (float*)dalloc(4);
        CK(cudaMemset(dmax, 0, 4));
        void* a0 = w16[0]; void* a1 = dmax; unsigned tot = N * K, n = N, k = K;
        launch(amax, std::min<unsigned>(1024, std::max<unsigned>(1, tot / 256)), 1, 256, {&a0, &a1, &tot});
        float gmax;
        CK(cudaMemcpy(&gmax, dmax, 4, cudaMemcpyDeviceToHost));
        float s2 = gmax > 0 ? gmax / (6.f * 448.f) : 1.f;
        void* p0 = wp[0]; void* p1 = ws[0];
        launch(q4, N, 1, 256, {&a0, &p0, &p1, &s2, &n, &k});
        CK(cudaDeviceSynchronize());
        for (size_t i = 1; i < ring4; i++) {
            CK(cudaMemcpy(wp[i], wp[0], pb, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(ws[i], ws[0], sb, cudaMemcpyDeviceToDevice));
        }
        auto fp4_arm = [&](size_t i, void* xin, void* out) {
            void* w = wp[i % ring4]; void* s = ws[i % ring4]; unsigned n2 = N, k2 = K; float sc = s2;
            launch(g4, (N + 7) / 8, 1, 256, {&xin, &w, &s, &sc, &out, &n2, &k2});
        };
        double t4 = time_us([&](int i) { fp4_arm(i, x, yt); });
        std::vector<double> o4;
        for (unsigned t = 0; t < MAXT; t++) {
            fp4_arm(0, (char*)x + (size_t)t * K * 2, yt);
            CK(cudaDeviceSynchronize());
            auto r = down_bf16(yt, N);
            o4.insert(o4.end(), r.begin(), r.end());
        }
        printf("  NVFP4 T=1 w4a16_gemv_sw %.1f us (%.2fx vs BF16 GEMV)\n", t4,
               time_us([&](int i) { bf16_arm(1, i, y16); }) / t4);
        pr("NVFP4       vs BF16 GEMV", compare(o4, o16));
        for (size_t i = 0; i < ring4; i++) { CK(cudaFree(wp[i])); CK(cudaFree(ws[i])); }
        CK(cudaFree(dmax));
    }

    // Batched rows must equal the T=1 kernel's (verify == decode).
    size_t bad8 = 0, bad16 = 0;
    for (unsigned t = 0; t < MAXT; t++) {
        void* xt = (char*)x + (size_t)t * K * 2;
        void* w = w8[0]; void* w2 = w16[0]; unsigned n = N, k = K;
        launch(k_w8, (N + 3) / 4, 1, 256, {&xt, &w, &scale, &yt, &n, &k});
        CK(cudaDeviceSynchronize());
        auto r1 = down_bf16(yt, N);
        for (unsigned i = 0; i < N; i++) bad8 += r1[i] != o8[(size_t)t * N + i];
        launch(k_bf16, (N + 3) / 4, 1, 256, {&xt, &w2, &yt, &n, &k});
        CK(cudaDeviceSynchronize());
        r1 = down_bf16(yt, N);
        for (unsigned i = 0; i < N; i++) bad16 += r1[i] != o16[(size_t)t * N + i];
    }
    printf("  batched-vs-T=1 bitwise: w8a16_gemv_batch4 %zu / %u differ, dense_gemv_bf16_batchm %zu / %u differ\n",
           bad8, MAXT * N, bad16, MAXT * N);

    // Prefill: what each projection's prefill GEMM costs at chunk widths.
    CUfunction k_pipe = load("w8a16_gemm_pipelined", "w8a16_gemm_pipelined");
    CUfunction k_dpipe = load("dense_gemm_bf16", "dense_gemm_bf16_pipelined");
    for (unsigned M : {512u, 2048u, 8192u}) {
        void* xa = dalloc((size_t)M * K * 2);
        void* ya = dalloc((size_t)M * N * 2);
        void* yb = dalloc((size_t)M * N * 2);
        std::vector<unsigned short> xm((size_t)M * K);
        for (auto& v : xm) v = tobf(g(rng));
        CK(cudaMemcpy(xa, xm.data(), xm.size() * 2, cudaMemcpyHostToDevice));
        double flop = 2.0 * M * N * K;
        double tl = time_us([&](int i) { lt_gemm(xa, w16[i % ring16], ya, M, N, K); });
        double td = time_us([&](int i) {
            void* w = w16[i % ring16]; unsigned m = M, n = N, k = K;
            launch(k_dpipe, (N + 127) / 128, (M + 127) / 128, 256, {&xa, &w, &ya, &m, &n, &k});
        });
        double tp = time_us([&](int i) {
            void* w = w8[i % ring8]; unsigned m = M, n = N, k = K;
            launch(k_pipe, (N + 31) / 32, (M + 127) / 128, 256, {&xa, &w, &scale, &yb, &m, &n, &k});
        });
        lt_gemm(xa, w16[0], ya, M, N, K);
        void* w = w8[0]; unsigned m = M, n = N, k = K;
        launch(k_pipe, (N + 31) / 32, (M + 127) / 128, 256, {&xa, &w, &scale, &yb, &m, &n, &k});
        CK(cudaDeviceSynchronize());
        Err e = compare(down_bf16(yb, (size_t)M * N), down_bf16(ya, (size_t)M * N));
        printf("prefill M=%-5u cuBLASLt BF16 %8.1f us (%5.1f TF) | dense_gemm_bf16_pipelined %8.1f us (%5.1f TF)"
               " | w8a16_gemm_pipelined %8.1f us (%5.1f TF, %.2fx vs cuBLASLt; rel_l2 vs BF16 %.2e)\n",
               M, tl, flop / tl / 1e6, td, flop / td / 1e6, tp, flop / tp / 1e6, tl / tp, e.rel_l2);
        CK(cudaFree(xa));
        CK(cudaFree(ya));
        CK(cudaFree(yb));
    }

    for (auto p : w16) CK(cudaFree(p));
    for (auto p : w8) CK(cudaFree(p));
    for (auto p : w8r) CK(cudaFree(p));
    for (void* p : {scale, rscale, x, y16, y8, y8r, yt}) CK(cudaFree(p));
}

// TP exactness: rank r's FP8 copy, quantized from its BF16 shard as the load
// path does, must be byte-for-byte the matching slice of the TP=1
// quantization -- weights AND block scales. QKVZ shards on rows by segment
// ([Q|K|V|Z], each sliced to the rank's heads: tp_shard/gdn.rs), out_proj on
// columns (row-parallel). True whenever every rank boundary is a multiple of
// 128 on the 128x128 scale grid, which is what the loader ensures.
static void check_tp_exact(std::mt19937_64& rng) {
    const unsigned H = 2560, KD = 128, VD = 128, NK = 16, NV = 48, TP = 2;
    const unsigned seg[4] = {NK * KD, NK * KD, NV * VD, NV * VD};  // Q K V Z
    const unsigned NQ = seg[0] + seg[1] + seg[2] + seg[3], VDIM = NV * VD;
    CUfunction q = load("quantize_bf16_to_fp8_blockscaled", "quantize_bf16_to_fp8_blockscaled");
    std::normal_distribution<float> g(0.f, 0.02f);
    auto quant = [&](void* w, unsigned n, unsigned k, std::vector<unsigned char>& b, std::vector<float>& s) {
        void* o = dalloc((size_t)n * k);
        void* sc = dalloc((size_t)((n + 127) / 128) * ((k + 127) / 128) * 4);
        launch(q, (k + 127) / 128, (n + 127) / 128, 256, {&w, &o, &sc, &n, &k});
        CK(cudaDeviceSynchronize());
        b.resize((size_t)n * k);
        s.resize((size_t)((n + 127) / 128) * ((k + 127) / 128));
        CK(cudaMemcpy(b.data(), o, b.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.data(), sc, s.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaFree(o));
        CK(cudaFree(sc));
    };
    auto upload = [&](const std::vector<unsigned short>& h) {
        void* d = dalloc(h.size() * 2);
        CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        return d;
    };
    std::vector<unsigned short> qf((size_t)NQ * H), of((size_t)H * VDIM);
    for (auto& v : qf) v = tobf(g(rng));
    for (auto& v : of) v = tobf(g(rng));
    void* dq = upload(qf);
    void* dof = upload(of);
    std::vector<unsigned char> fb, rb;
    std::vector<float> fs, rs;
    quant(dq, NQ, H, fb, fs);
    size_t bad_b = 0, bad_s = 0;
    for (unsigned r = 0; r < TP; r++) {
        // QKVZ: the rank's rows of each segment, packed in segment order.
        std::vector<unsigned short> sh;
        std::vector<unsigned> rows;  // full-tensor row of each local row
        unsigned base = 0;
        for (unsigned sgi = 0; sgi < 4; sgi++) {
            unsigned loc = seg[sgi] / TP;
            for (unsigned i = 0; i < loc; i++) rows.push_back(base + r * loc + i);
            base += seg[sgi];
        }
        for (unsigned row : rows) sh.insert(sh.end(), &qf[(size_t)row * H], &qf[(size_t)row * H] + H);
        void* ds = upload(sh);
        quant(ds, (unsigned)rows.size(), H, rb, rs);
        CK(cudaFree(ds));
        for (size_t i = 0; i < rows.size(); i++)
            bad_b += memcmp(&rb[i * H], &fb[(size_t)rows[i] * H], H) != 0;
        for (size_t i = 0; i < rows.size(); i += 128)
            for (unsigned kb = 0; kb < H / 128; kb++)
                bad_s += rs[(i / 128) * (H / 128) + kb] != fs[(rows[i] / 128) * (H / 128) + kb];
    }
    printf("TP%u exactness, in_proj_qkvz [%u x %u]: %zu shard rows differ, %zu block scales differ\n", TP, NQ, H,
           bad_b, bad_s);
    quant(dof, H, VDIM, fb, fs);
    bad_b = bad_s = 0;
    for (unsigned r = 0; r < TP; r++) {
        const unsigned loc = VDIM / TP;
        std::vector<unsigned short> sh;
        for (unsigned n = 0; n < H; n++)
            sh.insert(sh.end(), &of[(size_t)n * VDIM + r * loc], &of[(size_t)n * VDIM + r * loc] + loc);
        void* ds = upload(sh);
        quant(ds, H, loc, rb, rs);
        CK(cudaFree(ds));
        for (unsigned n = 0; n < H; n++)
            bad_b += memcmp(&rb[(size_t)n * loc], &fb[(size_t)n * VDIM + r * loc], loc) != 0;
        for (unsigned nb = 0; nb < H / 128; nb++)
            for (unsigned kb = 0; kb < loc / 128; kb++)
                bad_s += rs[nb * (loc / 128) + kb] != fs[nb * (VDIM / 128) + r * (loc / 128) + kb];
    }
    printf("TP%u exactness, out_proj [%u x %u]: %zu shard rows differ, %zu block scales differ\n", TP, H, VDIM,
           bad_b, bad_s);
    CK(cudaFree(dq));
    CK(cudaFree(dof));
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (argc > 2) g_iters = atoi(argv[2]);
    CK(cudaFree(0));  // primary context, shared with the driver-API modules
    LT(cublasLtCreate(&g_lt));
    g_ws = dalloc(WS);
    std::mt19937_64 rng(0x5eed);
    check_tp_exact(rng);
    if (argc > 3 && std::string(argv[3]) == "tp") return 0;
    run_shape("in_proj_qkvz TP2 shard", 8192, 2560, rng);
    run_shape("out_proj     TP2 shard", 2560, 3072, rng);
    run_shape("in_proj_qkvz TP1", 16384, 2560, rng);
    run_shape("out_proj     TP1", 2560, 6144, rng);
    return 0;
}
