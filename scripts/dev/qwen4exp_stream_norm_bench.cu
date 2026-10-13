// SPDX-License-Identifier: AGPL-3.0-only
// Bit parity and cost of `rms_norm_f32_grouped` (one launch over a drafter
// position's n x hc highway rows, ATLAS_QWEN4EXP_MTP_GROUPED_NORM) against
// the per-(row, stream) `rms_norm_f32` launches it replaces, at the real
// shape (hidden 2560, hc 4, n = 1..8 sequences).
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/common \
//        scripts/dev/qwen4exp_stream_norm_bench.cu -o qwen4exp_stream_norm_bench
//   ./qwen4exp_stream_norm_bench
#include "rms_norm.cu"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const unsigned H = 2560, HC = 4, MAXN = 8, ROWS = MAXN * HC;
static const float EPS = 1e-6f;

int main() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> w(HC * H);
    for (auto& v : w) v = __float2bfloat16(0.1f * nd(rng));
    float *x; __nv_bfloat16 *wd, *ref, *got;
    CK(cudaMalloc(&x, ROWS * H * 4)); CK(cudaMalloc(&wd, HC * H * 2));
    CK(cudaMalloc(&ref, ROWS * H * 2)); CK(cudaMalloc(&got, ROWS * H * 2));
    CK(cudaMemcpy(wd, w.data(), HC * H * 2, cudaMemcpyHostToDevice));
    int fail = 0;
    for (float scale : {1.f, 40.f, 1e-4f}) {
        std::vector<float> hx(ROWS * H);
        for (auto& v : hx) v = scale * nd(rng);
        hx[3] = 0.f; hx[5] = -0.f;
        CK(cudaMemcpy(x, hx.data(), ROWS * H * 4, cudaMemcpyHostToDevice));
        for (unsigned n = 1; n <= MAXN; n++) {
            const unsigned rows = n * HC;
            CK(cudaMemset(ref, 0x7F, ROWS * H * 2)); CK(cudaMemset(got, 0x7F, ROWS * H * 2));
            for (unsigned s = 0; s < rows; s++)
                rms_norm_f32<<<1, 1024>>>(x + s * H, wd + (s % HC) * H, ref + s * H, H, EPS);
            rms_norm_f32_grouped<<<rows, 1024>>>(x, wd, got, H, EPS, HC);
            CK(cudaDeviceSynchronize());
            std::vector<unsigned char> a(ROWS * H * 2), b(ROWS * H * 2);
            CK(cudaMemcpy(a.data(), ref, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), got, b.size(), cudaMemcpyDeviceToHost));
            if (memcmp(a.data(), b.data(), a.size())) { printf("  MISMATCH n=%u x%g\n", n, scale); fail++; }
        }
    }
    printf("%s  rms_norm_f32_grouped vs per-row rms_norm_f32, n = 1..8 x hc 4, 3 scales\n", fail ? "BAD" : "ok ");
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (unsigned n : {1u, 8u}) {
        const unsigned rows = n * HC;
        float ms[2];
        for (int arm = 0; arm < 2; arm++) {
            for (int rep = 0; rep < 2; rep++) {
                CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
                for (int i = 0; i < 200; i++) {
                    if (arm == 0)
                        for (unsigned s = 0; s < rows; s++)
                            rms_norm_f32<<<1, 1024>>>(x + s * H, wd + (s % HC) * H, ref + s * H, H, EPS);
                    else
                        rms_norm_f32_grouped<<<rows, 1024>>>(x, wd, got, H, EPS, HC);
                }
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[arm], e0, e1));
            }
        }
        printf("  n=%u: %2u launches %6.1f us -> one launch %5.1f us per draft position\n", n, rows,
               ms[0] * 1e3f / 200, ms[1] * 1e3f / 200);
    }
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
