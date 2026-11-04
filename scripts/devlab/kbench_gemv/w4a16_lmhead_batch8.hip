// SPDX-License-Identifier: AGPL-3.0-only
// kbench loop #5: w4a16_gemv_batch8 (gb10/common/w4a16_gemv.cu, compiled for strix-hip). At C8 decode it is the 27B
// lm_head, once per step: 12.7 ms/step on strix s210 for ~0.72 GB of NVFP4 weights + scales, ~56 GB/s, the least
// efficient decode kernel. Its __launch_bounds__(BLOCK_SIZE, 5) was tuned on GB10 (CUDA min-blocks-per-SM meaning).
// Candidates instantiate the SAME w4a16_gemv_batchm_impl<8> body under other launch bounds -> must be BIT-IDENTICAL.
#include "w4a16_gemv.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define LM_ARGS const __nv_bfloat16* A, const unsigned char* B, const unsigned char* S, const float s2, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K
extern "C" __global__ void cand_lm_free(LM_ARGS) { w4a16_gemv_batchm_impl<8>(A, B, S, s2, C, M, N, K, N); }
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2) void cand_lm_lb2(LM_ARGS) { w4a16_gemv_batchm_impl<8>(A, B, S, s2, C, M, N, K, N); }
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 8) void cand_lm_lb8(LM_ARGS) { w4a16_gemv_batchm_impl<8>(A, B, S, s2, C, M, N, K, N); }
extern "C" __global__ __launch_bounds__(BLOCK_SIZE) void cand_lm_lb1(LM_ARGS) { w4a16_gemv_batchm_impl<8>(A, B, S, s2, C, M, N, K, N); }

typedef void (*LFn)(LM_ARGS);
struct Cand { const char* name; LFn fn; };

int main() {
    const unsigned int shapes[][2] = {{248320, 5120}};
    const Cand cands[] = {{"batch8 (shipped,5)", w4a16_gemv_batch8}, {"free", cand_lm_free}, {"lb1", cand_lm_lb1},
                          {"lb2", cand_lm_lb2}, {"lb8", cand_lm_lb8}};
    const unsigned int M = 8;
    for (auto& sh : shapes) {
        const unsigned int N = sh[0], K = sh[1], G = K / 16;
        std::vector<unsigned char> hB((size_t)N * K / 2), hS((size_t)N * G); std::vector<unsigned short> hA((size_t)M * K);
        srand(99);
        for (auto& b : hB) b = rand() & 0xff;
        for (auto& s : hS) s = 0x30 + (rand() % 0x20);
        for (auto& a : hA) { float f = ((rand() % 2001) - 1000) * 1e-3f; unsigned int u; memcpy(&u, &f, 4); a = (unsigned short)(u >> 16); }
        unsigned char *dB, *dS; __nv_bfloat16 *dA, *dC;
        CHECK(hipMalloc(&dB, hB.size())); CHECK(hipMalloc(&dS, hS.size())); CHECK(hipMalloc(&dA, hA.size() * 2)); CHECK(hipMalloc(&dC, (size_t)M * N * 2));
        CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dS, hS.data(), hS.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        const double bytes = (double)N * K / 2 + (double)N * G;
        std::vector<unsigned short> ref((size_t)M * N), out(ref.size());
        printf("-- lm_head N=%u K=%u M=%u (weights+scales %.1f MB; larger than the MALL, no rotation needed)\n", N, K, M, bytes / 1e6);
        for (int ci = 0; ci < (int)(sizeof(cands) / sizeof(cands[0])); ++ci) {
            const Cand& c = cands[ci]; dim3 grid((N + N_PER_BLOCK - 1) / N_PER_BLOCK), block(BLOCK_SIZE);
            CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
            hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dB, dS, 1.0f, dC, M, N, K); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(ci ? out.data() : ref.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            for (int w = 0; w < 3; ++w) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dB, dS, 1.0f, dC, M, N, K);
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int iters = 20; CHECK(hipEventRecord(e0));
            for (int it = 0; it < iters; ++it) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dB, dS, 1.0f, dC, M, N, K);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            size_t diff = 0; if (ci) for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
            printf("   %-20s %9.1f us  %6.1f GB/s  %s\n", c.name, us, bytes / us / 1e3, ci == 0 ? "(reference)" : (diff == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
        }
    }
    return 0;
}
