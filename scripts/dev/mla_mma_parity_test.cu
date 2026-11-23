// SPDX-License-Identifier: AGPL-3.0-only
// Parity + bench for hy4_mla_batch_partial_mma vs hy4_mla_batch_partial.
//   nvcc -gencode arch=compute_120a,code=sm_120a -o mla_par mla_mma_parity_test.cu
//   ./mla_par [rows heads page_size count splits window]
// Defaults model the TP2 prefill chunk: rows=8 heads=16 count=2048
// splits=16 indices=on. SWA layers use heads=32 and window=512.
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1);} } while (0)

int main(int argc, char** argv) {
    unsigned rows = argc > 1 ? atoi(argv[1]) : 8;
    unsigned heads = argc > 2 ? atoi(argv[2]) : 16;
    unsigned page_size = argc > 3 ? atoi(argv[3]) : 16;
    unsigned count = argc > 4 ? atoi(argv[4]) : 2048;
    unsigned splits = count / 128;
    unsigned window = argc > 5 ? atoi(argv[5]) : 0;   // 0 = DSA indices
    unsigned index_stride = 2048;
    int bench = argc > 6 ? atoi(argv[6]) : 0;

    unsigned max_row_cache = 65536;
    unsigned columns = 512;

    std::mt19937 rng(7);
    auto frand = [&]{ return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); };

    // q: [rows][heads][576] bf16
    size_t qn = (size_t)rows * heads * 576;
    std::vector<__nv_bfloat16> hq(qn);
    for (size_t i = 0; i < qn; ++i) hq[i] = __float2bfloat16_rn(frand());
    __nv_bfloat16 *dq, *dcache;
    CK(cudaMalloc(&dq, qn * 2));
    CK(cudaMemcpy(dq, hq.data(), qn * 2, cudaMemcpyHostToDevice));

    // cache: max_row_cache x 576 bf16
    size_t cn = (size_t)max_row_cache * 576;
    std::vector<__nv_bfloat16> hc(cn);
    for (size_t i = 0; i < cn; ++i) hc[i] = __float2bfloat16_rn(frand() * 0.3f);
    CK(cudaMalloc(&dcache, cn * 2));
    CK(cudaMemcpy(dcache, hc.data(), cn * 2, cudaMemcpyHostToDevice));

    // table: columns -> physical page ids (shuffle)
    std::vector<unsigned> ht(columns);
    for (unsigned i = 0; i < columns; ++i) ht[i] = i % (max_row_cache / page_size);
    unsigned *dtable;
    CK(cudaMalloc(&dtable, columns * 4));
    CK(cudaMemcpy(dtable, ht.data(), columns * 4, cudaMemcpyHostToDevice));

    // indices: [rows][index_stride] int — random causal-selected logical keys
    int* dindices = nullptr;
    std::vector<int> hi;
    if (!window) {
        hi.resize((size_t)rows * index_stride);
        for (unsigned r = 0; r < rows; ++r) {
            unsigned pos = 1500 + r;
            unsigned cnt = (pos / 4 < index_stride / 4 - 1 ? pos / 4 : index_stride / 4 - 1) * 4 + pos % 4 + 1;
            for (unsigned i = 0; i < count; ++i)
                hi[r * index_stride + i] = i < cnt ? (int)((i * 7 + r * 13) % cnt) : -1;
            for (unsigned i = count; i < index_stride; ++i) hi[r * index_stride + i] = -1;
        }
        CK(cudaMalloc(&dindices, hi.size() * 4));
        CK(cudaMemcpy(dindices, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
    }

    size_t pn = (size_t)rows * heads * splits * 514;
    float *dp_ref, *dp_mma;
    CK(cudaMalloc(&dp_ref, pn * 4));
    CK(cudaMalloc(&dp_mma, pn * 4));
    CK(cudaMemset(dp_ref, 0xCD, pn * 4));
    CK(cudaMemset(dp_mma, 0xCD, pn * 4));

    unsigned position = 1500;

    // Reference launch: grid [heads, splits, rows]
    dim3 g_ref(heads, splits, rows);
    hy4_mla_batch_partial<<<g_ref, 256>>>(
        dq, dcache, dtable, dindices, dp_ref, position, page_size, columns,
        window, count, splits, heads, index_stride);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // MMA launch: grid [splits, rows, 16-head tiles], dynamic shared
    CK(cudaFuncSetAttribute(hy4_mla_batch_partial_mma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MLM_SMEM_BYTES));
    dim3 g_mma(splits, rows, (heads + 15) / 16);
    hy4_mla_batch_partial_mma<<<g_mma, 256, MLM_SMEM_BYTES>>>(
        dq, dcache, dtable, dindices, dp_mma, position, page_size, columns,
        window, count, splits, heads, index_stride);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> href(pn), hmma(pn);
    CK(cudaMemcpy(href.data(), dp_ref, pn * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hmma.data(), dp_mma, pn * 4, cudaMemcpyDeviceToHost));

    double max_err = 0, sum_err = 0; size_t worst = 0;
    unsigned nan_ref = 0, nan_mma = 0;
    for (size_t i = 0; i < pn; ++i) {
        if (isnan(href[i])) nan_ref++;
        if (isnan(hmma[i])) nan_mma++;
        double d = fabs((double)href[i] - (double)hmma[i]);
        if (!isnan(d)) sum_err += d;
        if (d > max_err) { max_err = d; worst = i; }
    }
    printf("parity rows=%u heads=%u count=%u splits=%u window=%u: max_err=%.6f mean_err=%.6f nan_ref=%u nan_mma=%u (worst idx %zu ref=%.4f mma=%.4f)\n",
           rows, heads, count, splits, window, max_err, sum_err / pn,
           nan_ref, nan_mma, worst, href[worst], hmma[worst]);
    if (max_err > 0.05) { printf("FAIL\n"); return 1; }

    if (bench) {
        int iters = argc > 7 ? atoi(argv[7]) : 50;
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i)
            hy4_mla_batch_partial<<<g_ref, 256>>>(
                dq, dcache, dtable, dindices, dp_ref, position, page_size,
                columns, window, count, splits, heads, index_stride);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms_ref; CK(cudaEventElapsedTime(&ms_ref, t0, t1));
        ms_ref /= iters;

        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i)
            hy4_mla_batch_partial_mma<<<g_mma, 256, MLM_SMEM_BYTES>>>(
                dq, dcache, dtable, dindices, dp_mma, position, page_size,
                columns, window, count, splits, heads, index_stride);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms_mma; CK(cudaEventElapsedTime(&ms_mma, t0, t1));
        ms_mma /= iters;
        printf("bench ref=%.3f ms  mma=%.3f ms  speedup=%.1fx\n",
               ms_ref, ms_mma, ms_ref / ms_mma);
    }
    printf("PASS\n");
    return 0;
}
