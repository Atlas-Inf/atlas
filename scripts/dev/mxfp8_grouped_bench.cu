// SPDX-License-Identifier: AGPL-3.0-only
// Standalone gate for the opt-in GLM MXFP8 twins at verify row counts:
//   W_uk  (ATLAS_GLM_MLA_KVB_MXFP8): 32 heads x [N=512, K=256]
//   W_uv  (ATLAS_GLM_MLA_KVB_MXFP8): 32 heads x [N=256, K=512]
//   wq_b  (ATLAS_GLM_INDEX_MXFP8):    1 head  x [N=4096, K=1536]
// Baseline: cuBLASLt strided-batched BF16 with the descriptors of
// spark_runtime::cublaslt::bf16_grouped_gemm_act_weight_t (built once, not
// per call). Candidate: mxfp8_gemv_tc{8,16}_grouped for the heads, the plain
// mxfp8_gemv_tc{8,16,32} for wq_b (as mla_prefill_dense runs it), timed
// without and with PDL. "saved" uses the non-PDL column: back-to-back PDL
// launches of one kernel overlap each other, which cuBLAS cannot.
// Checks that every grouped output bit equals per-head plain launches, and
// reports the MXFP8 relative L2 error against BF16. Weights cycle through
// `copies` sets so they stream from DRAM as in decode.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/mxfp8_grouped_bench.cu -lcublasLt -o mxfp8_grouped_bench
//   ./mxfp8_grouped_bench [copies=12] [iters=100] [reps=9]
// Device memory: ~0.35 GB at the defaults.
#include "mxfp8_gemv.cu"
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d cublas status %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)
typedef __nv_bfloat16 bf;
typedef void (*Plain)(const bf*, const unsigned char*, const unsigned char*, bf*, unsigned, unsigned, unsigned, unsigned);
typedef void (*Grouped)(const bf*, const unsigned char*, const unsigned char*, bf*, unsigned, unsigned, unsigned, unsigned, unsigned);
static const Plain kPlain[3] = {mxfp8_gemv_tc8, mxfp8_gemv_tc16, mxfp8_gemv_tc32};
static const Grouped kGrouped[2] = {mxfp8_gemv_tc8_grouped, mxfp8_gemv_tc16_grouped};
static int tier(unsigned m) { return m <= 8 ? 0 : m <= 16 ? 1 : 2; }
static float bf2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }

// C[t, h*n + j] = sum_k A[t, h*k + i] W[h, j, i] as one strided-batched matmul.
struct LtGrouped {
    cublasLtHandle_t h; void* ws; size_t ws_size = 64u << 20;
    cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t l[3]; cublasLtMatmulHeuristicResult_t r;
    LtGrouped() { CB(cublasLtCreate(&h)); CK(cudaMalloc(&ws, ws_size)); }
    void plan(unsigned m, unsigned g, unsigned n, unsigned k, unsigned a_stride, unsigned c_stride) {
        CB(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const int ta = CUBLAS_OP_T, tb = CUBLAS_OP_N, batches = g;
        CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, 4));
        CB(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, 4));
        const unsigned long long rows[3] = {k, k, n}, cols[3] = {n, m, m};
        const long long ld[3] = {k, a_stride, c_stride}, stride[3] = {(long long)n * k, k, n};
        for (int i = 0; i < 3; i++) {
            CB(cublasLtMatrixLayoutCreate(&l[i], CUDA_R_16BF, rows[i], cols[i], ld[i]));
            CB(cublasLtMatrixLayoutSetAttribute(l[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, 4));
            CB(cublasLtMatrixLayoutSetAttribute(l[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride[i], 8));
        }
        cublasLtMatmulPreference_t p;
        CB(cublasLtMatmulPreferenceCreate(&p));
        CB(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, 8));
        int ret = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(h, d, l[0], l[1], l[2], l[2], p, 1, &r, &ret));
        if (!ret) { fprintf(stderr, "no cuBLASLt algorithm\n"); exit(1); }
        cublasLtMatmulPreferenceDestroy(p);
    }
    void run(const bf* act, const bf* w, bf* out, cudaStream_t st) {
        const float one = 1.f, zero = 0.f;
        CB(cublasLtMatmul(h, d, &one, w, l[0], act, l[1], &zero, out, l[2], out, l[2], &r.algo, ws, ws_size, st));
    }
    void done() { for (auto x : l) cublasLtMatrixLayoutDestroy(x); cublasLtMatmulDescDestroy(d); }
};

int main(int argc, char** argv) {
    const int copies = argc > 1 ? atoi(argv[1]) : 12;
    const int iters = argc > 2 ? atoi(argv[2]) : 100;
    const int reps = argc > 3 ? atoi(argv[3]) : 9;
    const unsigned MMAX = 32;
    struct Shape { const char* name; unsigned G, N, K; } shapes[3] = {
        {"W_uk", 32, 512, 256}, {"W_uv", 32, 256, 512}, {"index_wq_b", 1, 4096, 1536}};
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    LtGrouped lt;
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int failures = 0;
    for (const Shape& sh : shapes) {
        const unsigned G = sh.G, N = sh.N, K = sh.K, lda = G * K, ldc = G * N;
        const size_t wel = (size_t)G * N * K, cel = (size_t)MMAX * ldc;
        std::vector<unsigned short> hw(wel * copies), ha((size_t)MMAX * lda);
        for (auto& x : hw) { bf b = __float2bfloat16(nd(rng) * 0.03f); memcpy(&x, &b, 2); }
        for (auto& x : ha) { bf b = __float2bfloat16(nd(rng)); memcpy(&x, &b, 2); }
        bf *dw, *da, *dah, *dc[3]; unsigned char *dq, *ds;
        CK(cudaMalloc(&dw, wel * copies * 2)); CK(cudaMalloc(&da, ha.size() * 2)); CK(cudaMalloc(&dah, MMAX * K * 2));
        CK(cudaMalloc(&dq, wel * copies)); CK(cudaMalloc(&ds, wel * copies / MX_BLOCK));
        for (auto& c : dc) CK(cudaMalloc(&c, cel * 2));
        CK(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
        const unsigned long long blocks = wel * copies / MX_BLOCK;
        mxfp8_quantize_bf16<<<(unsigned)((blocks + 255) / 256), 256>>>(dw, dq, ds, blocks);
        CK(cudaDeviceSynchronize());
        auto grouped = [&](int copy, unsigned M, bf* out, bool pdl = true) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3((N + 15) / 16, 1, G); cfg.blockDim = dim3(MX_WARPS * MX_WARP); cfg.stream = st;
            cudaLaunchAttribute at; at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at.val.programmaticStreamSerializationAllowed = 1; cfg.attrs = &at; cfg.numAttrs = pdl;
            const unsigned char* q = dq + copy * wel;
            const unsigned char* s = ds + copy * wel / MX_BLOCK;
            if (G == 1) CK(cudaLaunchKernelEx(&cfg, kPlain[tier(M)], (const bf*)da, q, s, out, M, N, K, ldc));
            else CK(cudaLaunchKernelEx(&cfg, kGrouped[tier(M)], (const bf*)da, q, s, out, M, N, K, lda, ldc));
        };
        auto cublas = [&](int copy, bf* out) { lt.run(da, dw + copy * wel, out, st); };

        // Bitwise: grouped == per-head plain launches on a compacted activation slice.
        std::vector<unsigned short> o[3] = {std::vector<unsigned short>(cel), std::vector<unsigned short>(cel),
                                            std::vector<unsigned short>(cel)};
        size_t diff = 0;
        for (unsigned M : {1u, 5u, 8u, 13u, 16u}) {
            if (G == 1) break;
            CK(cudaMemsetAsync(dc[0], 0xFF, cel * 2, st)); CK(cudaMemsetAsync(dc[1], 0xEE, cel * 2, st));
            grouped(1, M, dc[0]);
            for (unsigned h = 0; h < G; h++) {
                CK(cudaMemcpy2DAsync(dah, K * 2, da + h * K, lda * 2, K * 2, M, cudaMemcpyDeviceToDevice, st));
                kPlain[tier(M)]<<<(N + 15) / 16, MX_WARPS * MX_WARP, 0, st>>>(
                    dah, dq + wel + (size_t)h * N * K, ds + (wel + (size_t)h * N * K) / MX_BLOCK, dc[1] + h * N, M, N, K, ldc);
            }
            CK(cudaStreamSynchronize(st));
            CK(cudaMemcpy(o[0].data(), dc[0], cel * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(o[1].data(), dc[1], cel * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < (size_t)M * ldc; i++) diff += o[0][i] != o[1][i];
        }
        // Error of the MXFP8 twin against BF16 at M = 8.
        lt.plan(8, G, N, K, lda, ldc); cublas(0, dc[2]); lt.done(); grouped(0, 8, dc[0]);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(o[0].data(), dc[0], cel * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o[2].data(), dc[2], cel * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < (size_t)8 * ldc; i++) {
            const double a = bf2f(o[2][i]), b = bf2f(o[0][i]);
            num += (a - b) * (a - b); den += a * a;
        }
        printf("%s (G=%u N=%u K=%u): grouped vs per-head plain %zu differing; MXFP8 vs BF16 rel_l2 %.4f\n",
               sh.name, G, N, K, diff, sqrt(num / den));
        failures += diff != 0;

        printf("  us/call (median of %d reps x %d, %d weight sets)  cuBLAS-BF16  MXFP8  MXFP8+PDL  saved\n",
               reps, iters, copies);
        for (unsigned M : {1u, 2u, 4u, 5u, 8u, 16u, 32u}) {
            if (G > 1 && M > 16) break;
            std::vector<float> t[3];
            lt.plan(M, G, N, K, lda, ldc);
            for (int r = 0; r < reps; r++)
                for (int v = 0; v < 3; v++) {
                    auto one = [&](int it) {
                        if (v) grouped(it % copies, M, dc[0], v == 2); else cublas(it % copies, dc[2]);
                    };
                    for (int w = 0; w < 5; w++) one(w);
                    CK(cudaEventRecord(e0, st));
                    for (int it = 0; it < iters; it++) one(it);
                    CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[v].push_back(ms * 1000.f / iters);
                }
            lt.done();
            for (auto& v : t) std::sort(v.begin(), v.end());
            const float b = t[0][reps / 2], m = t[1][reps / 2], p = t[2][reps / 2];
            printf("  M=%-2u %8.2f %8.2f %8.2f %7.2f\n", M, b, m, p, b - m);
        }
        cudaFree(dw); cudaFree(da); cudaFree(dah); cudaFree(dq); cudaFree(ds);
        for (auto c : dc) cudaFree(c);
    }
    return failures ? 2 : 0;
}
