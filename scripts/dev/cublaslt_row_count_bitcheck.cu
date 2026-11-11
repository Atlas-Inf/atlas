// SPDX-License-Identifier: AGPL-3.0-only
// Does a row of a cuBLASLt BF16 GEMM keep its bits when the row count changes?
// Mirrors crates/spark-runtime/src/cublaslt.rs gemm_bf16 (opT weight) and
// cublaslt/grouped.rs bf16_grouped_gemm_act_weight_t: FP32 compute, BF16
// operands, heuristic algorithm 0 under a 64 MiB workspace. For M = 1..32 rows
// of the same 32 activation rows, counts the outputs that differ from the
// 32-row call. nvcc -O2 scripts/dev/cublaslt_row_count_bitcheck.cu -lcublasLt -lcublas -o lt_rows
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { int e = (int)(x); if (e) { fprintf(stderr, "%s:%d error %d\n", __FILE__, __LINE__, e); exit(1); } } while (0)
static unsigned short f2bf(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static const size_t WS = 64ull << 20;
static cublasLtHandle_t lt;
static void* ws;

// out[t*c_stride + h*n + j] = sum_k act[t*a_stride + h*k + i] * w[h*n*k + j*k + i]; g = 1 is the plain GEMM.
static int run(const void* act, const void* w, void* out, int m, int g, int n, int k, int a_stride, int c_stride) {
    cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t l[3]; cublasLtMatmulPreference_t pref;
    CK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    CK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    const long long dims[3][4] = {{k, n, k, (long long)n * k}, {k, m, a_stride, k}, {n, m, c_stride, n}};
    for (int i = 0; i < 3; i++) {
        CK(cublasLtMatrixLayoutCreate(&l[i], CUDA_R_16BF, dims[i][0], dims[i][1], dims[i][2]));
        if (g > 1) {
            int batches = g; long long stride = dims[i][3];
            CK(cublasLtMatrixLayoutSetAttribute(l[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof batches));
            CK(cublasLtMatrixLayoutSetAttribute(l[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride));
        }
    }
    CK(cublasLtMatmulPreferenceCreate(&pref));
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res; int found = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(lt, desc, l[0], l[1], l[2], l[2], pref, 1, &res, &found));
    if (!found) { fprintf(stderr, "no algorithm at m=%d\n", m); exit(1); }
    int id = -1; size_t sz;
    cublasLtMatmulAlgoConfigGetAttribute(&res.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &sz);
    const float alpha = 1.f, beta = 0.f;
    CK(cublasLtMatmul(lt, desc, &alpha, w, l[0], act, l[1], &beta, out, l[2], out, l[2], &res.algo, ws, WS, 0));
    CK(cudaDeviceSynchronize());
    cublasLtMatmulPreferenceDestroy(pref);
    for (int i = 0; i < 3; i++) cublasLtMatrixLayoutDestroy(l[i]);
    cublasLtMatmulDescDestroy(desc);
    return id;
}

static void shape(const char* name, int g, int n, int k) {
    const int M = 32, a_stride = g * k, c_stride = g * n;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> act((size_t)M * a_stride), w((size_t)g * n * k), ref((size_t)M * c_stride), got(ref.size());
    for (auto& x : act) x = f2bf(nd(rng));
    for (auto& x : w) x = f2bf(nd(rng) * 0.05f);
    void *da, *dw, *dc;
    CK(cudaMalloc(&da, act.size() * 2)); CK(cudaMalloc(&dw, w.size() * 2)); CK(cudaMalloc(&dc, ref.size() * 2));
    CK(cudaMemcpy(da, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    const int id32 = run(da, dw, dc, M, g, n, k, a_stride, c_stride);
    CK(cudaMemcpy(ref.data(), dc, ref.size() * 2, cudaMemcpyDeviceToHost));
    printf("%s  g=%d n=%d k=%d  (32 rows: algo %d)\n  rows:differing outputs(algo)", name, g, n, k, id32);
    int moved = 0;
    for (int m = 1; m < M; m++) {
        CK(cudaMemset(dc, 0, ref.size() * 2));
        const int id = run(da, dw, dc, m, g, n, k, a_stride, c_stride);
        CK(cudaMemcpy(got.data(), dc, (size_t)m * c_stride * 2, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < (size_t)m * c_stride; i++) diff += got[i] != ref[i];
        moved += diff != 0;
        printf(" %d:%zu(%d)", m, diff, id);
    }
    printf("\n  => %d of 31 row counts give rows that differ from the 32-row call\n", moved);
    // The same against a 16-row and an 8-row call: do the narrow row counts agree among themselves?
    for (int top : {16, 8}) {
        std::vector<unsigned short> ref2((size_t)top * c_stride);
        run(da, dw, dc, top, g, n, k, a_stride, c_stride);
        CK(cudaMemcpy(ref2.data(), dc, ref2.size() * 2, cudaMemcpyDeviceToHost));
        int moved2 = 0;
        printf("  vs the %d-row call:", top);
        for (int m = 1; m < top; m++) {
            run(da, dw, dc, m, g, n, k, a_stride, c_stride);
            CK(cudaMemcpy(got.data(), dc, (size_t)m * c_stride * 2, cudaMemcpyDeviceToHost));
            size_t diff = 0;
            for (size_t i = 0; i < (size_t)m * c_stride; i++) diff += got[i] != ref2[i];
            moved2 += diff != 0;
            printf(" %d:%zu", m, diff);
        }
        printf("  => %d differ\n", moved2);
    }
    cudaFree(da); cudaFree(dw); cudaFree(dc);
}

int main() {
    CK(cublasLtCreate(&lt));
    CK(cudaMalloc(&ws, WS));
    shape("MLA W_uk absorb (grouped)", 32, 512, 256);
    shape("MLA W_uv (grouped)       ", 32, 256, 512);
    shape("index wq_b (plain)       ", 1, 4096, 1536);
    shape("kv_a-sized (plain)       ", 1, 512, 4096);
    shape("weights_proj-sized (plain)", 1, 32, 4096);
    return 0;
}
