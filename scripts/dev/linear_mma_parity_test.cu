// SPDX-License-Identifier: AGPL-3.0-only
// Parity harness: hy4_batch_linear (kind==1 fp8 GEMV) vs
// hy4_batch_linear_mma (bf16 tensor-core). Patterned bf16 activations,
// random e4m3 weights + per-column scales. Positional args:
//   rows k n heads [bench] [iter N]
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_ffn.cu"
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

static float frand(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return ((s >> 8) & 0xFFFFFF) / 8388608.f * 2.f - 1.f;
}

int main(int argc, char** argv) {
    unsigned rows = argc > 1 ? atoi(argv[1]) : 256;
    unsigned k = argc > 2 ? atoi(argv[2]) : 3072;
    unsigned n = argc > 3 ? atoi(argv[3]) : 3072;
    unsigned heads = argc > 4 ? atoi(argv[4]) : 1;
    bool bench = argc > 5 && !strcmp(argv[5], "bench");
    unsigned iters = argc > 6 ? atoi(argv[6]) : 50;
    unsigned xr = k * heads, yr = n * heads, xh = k, yh = n, wr = k;

    std::vector<__nv_bfloat16> hx(rows * xr);
    std::vector<unsigned char> hw(heads * n * wr);
    std::vector<float> hs(heads * n);
    unsigned seed = 12345;
    for (auto& v : hx) v = __float2bfloat16_rn(frand(seed) * 2.f);
    for (auto& v : hw) v = (unsigned char)(frand(seed) * 127.f);   // |e4m3| any non-NaN
    for (auto& v : hs) v = 0.5f + fabsf(frand(seed));

    __nv_bfloat16 *dx, *dout_g, *dout_m;
    unsigned char* dw;
    float* ds;
    cudaMalloc(&dx, hx.size() * 2);
    cudaMalloc(&dw, hw.size());
    cudaMalloc(&ds, hs.size() * 4);
    cudaMalloc(&dout_g, rows * yr * 2);
    cudaMalloc(&dout_m, rows * yr * 2);
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);

    auto run_gemv = [&] {
        hy4_batch_linear<<<dim3(n / 8, (rows + 7) / 8, heads), 256>>>(
            dx, dw, ds, dout_g, n, k, rows, heads, xh, yh, xr, yr, wr, 1, 0);
    };
    auto run_mma = [&] {
        hy4_batch_linear_mma<<<dim3((n + 127) / 128, (rows + 63) / 64, heads), 256>>>(
            dx, dw, ds, dout_m, n, k, rows, heads, xh, yh, xr, yr, wr);
    };

    run_gemv(); run_mma();
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<__nv_bfloat16> hg(rows * yr), hm(rows * yr);
    cudaMemcpy(hg.data(), dout_g, hg.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), dout_m, hm.size() * 2, cudaMemcpyDeviceToHost);

    double max_err = 0, sum_err = 0;
    size_t mism = 0, max_i = 0;
    for (size_t i = 0; i < hg.size(); ++i) {
        float g = (float)hg[i], m = (float)hm[i];
        double e = fabs(g - m);
        if (e > max_err) { max_err = e; max_i = i; }
        sum_err += e;
        if (e > 1e-3) mism++;
    }
    printf("max_err=%.6f (n=%zu, idx=%zu, mismatched=%zu, mean_err=%.6f)\n",
           max_err, hg.size(), max_i, mism, sum_err / hg.size());

    if (bench) {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (unsigned i = 0; i < 5; ++i) run_gemv();
        cudaEventRecord(t0);
        for (unsigned i = 0; i < iters; ++i) run_gemv();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float gms; cudaEventElapsedTime(&gms, t0, t1);
        for (unsigned i = 0; i < 5; ++i) run_mma();
        cudaEventRecord(t0);
        for (unsigned i = 0; i < iters; ++i) run_mma();
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float mms; cudaEventElapsedTime(&mms, t0, t1);
        printf("gemv=%.3fms mma=%.3fms speedup=%.1fx\n",
               gms / iters, mms / iters, gms / mms);
    }
    return max_err > 0.1 ? 1 : 0;
}
