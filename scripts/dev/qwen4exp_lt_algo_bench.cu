// SPDX-License-Identifier: AGPL-3.0-only
// cuBLASLt algorithm survey for the qwen4_exp TP2 prefill projections: for
// each shape, every algorithm the heuristic offers (the production call:
// BF16 in/out, FP32 compute, opT weight [N,K], 64 MiB workspace), its
// split-K, its time, and whether its output is byte-identical to the
// heuristic's first choice -- the one serving uses today. A non-split
// algorithm that is byte-identical and faster is a candidate for an exact
// pin.
//
// Run it against the cuBLASLt the server links (inside atlas-release-builder,
// CUDA 13.0): a newer host library picks differently.
//   nvcc -O3 -std=c++17 -o lt scripts/dev/qwen4exp_lt_algo_bench.cu -lcublasLt
//   ./lt M N K [M N K ...]        (default: the GDN in_proj at 16016 rows)
//   ./lt pin N K M0 M1 STEP       the k-chain pin (spark_runtime::cublaslt,
//                                 ATLAS_LT_KCHAIN_PIN) at every M in
//                                 [M0, M1] by STEP: its pick, its bytes
//                                 against the heuristic's first choice, times
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d cublasLt status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static unsigned short f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

template <typename F>
static float time_ms(F f, int iters = 5, int reps = 5) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(a)); for (int i = 0; i < iters; ++i) f(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms / iters);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int attr(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t which) {
    int v = 0; size_t sz;
    cublasLtMatmulAlgoConfigGetAttribute(&a, which, &v, sizeof v, &sz);
    return v;
}

// The pin's rule, as spark_runtime::cublaslt implements it: when the
// heuristic's first choice is the non-split CUTLASS sm80 kernel (algo 21,
// split-K 1, no reduction), take the first offered algo-21 non-split
// candidate with a preferred tile (24 = 256x128, then 23 = 128x256).
static int pin_pick(const cublasLtMatmulHeuristicResult_t* r, int found) {
    auto kchain = [&](int i) {
        return attr(r[i].algo, CUBLASLT_ALGO_CONFIG_ID) == 21 &&
               attr(r[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) <= 1 &&
               attr(r[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == 0;
    };
    if (found < 1 || !kchain(0)) return 0;
    for (int tile : {24, 23})
        for (int i = 0; i < found; ++i)
            if (kchain(i) && attr(r[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID) == tile) return i;
    return 0;
}

static int pin_sweep(cublasLtHandle_t lt, void* ws, size_t WS, int N, int K, int M0, int M1, int step) {
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> ha((size_t)M1 * K), hw((size_t)N * K);
    for (auto& x : ha) x = f2bf(nd(rng));
    for (auto& x : hw) x = f2bf(0.02f * nd(rng));
    void *a, *w, *o0, *o1;
    CK(cudaMalloc(&a, ha.size() * 2)); CK(cudaMalloc(&w, hw.size() * 2));
    CK(cudaMalloc(&o0, (size_t)M1 * N * 2)); CK(cudaMalloc(&o1, (size_t)M1 * N * 2));
    CK(cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
    int bad = 0;
    double t0s = 0, t1s = 0;
    for (int M = M0; M <= M1; M += step) {
        cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t la, lb, ld; cublasLtMatmulPreference_t pref;
        LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
        LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
        LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));
        LT(cublasLtMatmulPreferenceCreate(&pref));
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
        cublasLtMatmulHeuristicResult_t res[8]; int found = 0;
        LT(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 8, res, &found));
        const int p = pin_pick(res, found);
        const float alpha = 1.f, beta = 0.f;
        auto run = [&](int i, void* o) {
            LT(cublasLtMatmul(lt, desc, &alpha, w, la, a, lb, &beta, o, ld, o, ld, &res[i].algo, ws, WS, 0));
        };
        CK(cudaMemset(o0, 0x5A, (size_t)M * N * 2)); CK(cudaMemset(o1, 0xA5, (size_t)M * N * 2));
        run(0, o0); run(p, o1); CK(cudaDeviceSynchronize());
        std::vector<unsigned char> b0((size_t)M * N * 2), b1((size_t)M * N * 2);
        CK(cudaMemcpy(b0.data(), o0, b0.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b1.data(), o1, b1.size(), cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t j = 0; j < b0.size(); ++j) diff += b0[j] != b1[j];
        bad += diff != 0;
        const float t0 = time_ms([&] { run(0, o0); }, 2, 3), t1 = time_ms([&] { run(p, o1); }, 2, 3);
        t0s += t0; t1s += t1;
        printf("  M=%5d #0 tile %2d splitK %d -> pick #%d tile %2d: bytes differ %zu  %.3f -> %.3f ms\n", M,
               attr(res[0].algo, CUBLASLT_ALGO_CONFIG_TILE_ID), attr(res[0].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
               p, attr(res[p].algo, CUBLASLT_ALGO_CONFIG_TILE_ID), diff, t0, t1);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
    }
    printf("N=%d K=%d: %d M values with differing bytes; total %.2f -> %.2f ms\n", N, K, bad, t0s, t1s);
    printf("(sizeof(cublasLtMatmulHeuristicResult_t)=%zu) ", sizeof(cublasLtMatmulHeuristicResult_t));
    printf("(enum check: CONFIG_ID=%d TILE_ID=%d SPLITK_NUM=%d REDUCTION_SCHEME=%d)\n", (int)CUBLASLT_ALGO_CONFIG_ID,
           (int)CUBLASLT_ALGO_CONFIG_TILE_ID, (int)CUBLASLT_ALGO_CONFIG_SPLITK_NUM, (int)CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    CK(cudaFree(a)); CK(cudaFree(w)); CK(cudaFree(o0)); CK(cudaFree(o1));
    return bad;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "pin") {
        if (argc < 7) { fprintf(stderr, "usage: %s pin N K M0 M1 STEP\n", argv[0]); return 2; }
        cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
        const size_t WS = 64u << 20; void* ws; CK(cudaMalloc(&ws, WS));
        printf("cuBLASLt %zu\n", cublasLtGetVersion());
        return pin_sweep(lt, ws, WS, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6])) ? 1 : 0;
    }
    std::vector<int> dims;
    for (int i = 1; i < argc; ++i) dims.push_back(atoi(argv[i]));
    if (dims.empty()) dims = {16016, 8192, 2560};
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
    printf("cuBLASLt %zu\n", cublasLtGetVersion());
    const size_t WS = 64u << 20; void* ws; CK(cudaMalloc(&ws, WS));
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t s = 0; s + 2 < dims.size(); s += 3) {
        const int M = dims[s], N = dims[s + 1], K = dims[s + 2];
        std::vector<unsigned short> ha((size_t)M * K), hw((size_t)N * K);
        for (auto& x : ha) x = f2bf(nd(rng));
        for (auto& x : hw) x = f2bf(0.02f * nd(rng));
        void *a, *w, *o, *o0;
        CK(cudaMalloc(&a, ha.size() * 2)); CK(cudaMalloc(&w, hw.size() * 2));
        CK(cudaMalloc(&o, (size_t)M * N * 2)); CK(cudaMalloc(&o0, (size_t)M * N * 2));
        CK(cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t la, lb, ld; cublasLtMatmulPreference_t pref;
        LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
        LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
        LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
        LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
        LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));
        LT(cublasLtMatmulPreferenceCreate(&pref));
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
        cublasLtMatmulHeuristicResult_t res[24]; int found = 0;
        LT(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 24, res, &found));
        const double gf = 2.0 * M * N * K / 1e9;
        printf("M=%d N=%d K=%d: %d algorithms\n", M, N, K, found);
        std::vector<unsigned char> base((size_t)M * N * 2), got((size_t)M * N * 2);
        for (int i = 0; i < found; ++i) {
            int id = 0, tile = 0, splitk = 1, red = 0, stages = 0; size_t sz;
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &sz);
            const float alpha = 1.f, beta = 0.f;
            void* out = i == 0 ? o0 : o;
            auto go = [&] {
                LT(cublasLtMatmul(lt, desc, &alpha, w, la, a, lb, &beta, out, ld, out, ld, &res[i].algo, ws, WS, 0));
            };
            CK(cudaMemset(out, 0x5A, (size_t)M * N * 2));
            go();
            CK(cudaDeviceSynchronize());
            size_t diff = 0;
            if (i == 0) {
                CK(cudaMemcpy(base.data(), o0, base.size(), cudaMemcpyDeviceToHost));
            } else {
                CK(cudaMemcpy(got.data(), o, got.size(), cudaMemcpyDeviceToHost));
                for (size_t j = 0; j < got.size(); ++j) diff += got[j] != base[j];
            }
            const float t = time_ms(go);
            printf("  #%-2d algo %3d tile %3d stages %3d splitK %d red %d  %8.3f ms %6.1f TFLOP/s  bytes vs #0: %zu\n",
                   i, id, tile, stages, splitk, red, t, gf / t, diff);
        }
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(desc);
        CK(cudaFree(a)); CK(cudaFree(w)); CK(cudaFree(o)); CK(cudaFree(o0));
    }
    return 0;
}
