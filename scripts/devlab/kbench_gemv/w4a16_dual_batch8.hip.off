// SPDX-License-Identifier: AGPL-3.0-only
// kbench: w4a16_gemv_dp4a_dual_batch8_d4_dyn_vl2 (27B MLP gate+up at C8 decode; 30% of a step on strix s210) against
// candidates where one warp computes NO adjacent outputs and shares each activation load (8 rows x 16 B int8 + scale)
// across them. Per output: the same logical lanes p / p+32, group walk, DP4A order, accumulate expression and
// 32-lane reduce + two-half add as the shipped VLANES=2 path, so every candidate must be BIT-IDENTICAL.
#include "w4a16_gemv_dp4a.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int NO>
__device__ __forceinline__ void dual8_multi(const signed char* __restrict__ a_q, const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B0p, const unsigned char* __restrict__ B0s, const float B0s2, __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1p, const unsigned char* __restrict__ B1s, const float B1s2, __nv_bfloat16* __restrict__ C1,
    unsigned int M, unsigned int N, unsigned int K) {
    const unsigned int lanes_per_out = 32;                 // == DP4A_BLOCK_SIZE / (DP4A_N_PER_BLOCK * 2) in the shipped vl2
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int nbase = blockIdx.x * (8 * NO) + warp * NO;
    const unsigned int half_k = K / 2, groups = K / DP4A_GROUP_SIZE, group_stride = 64;
    float a0[NO][8] = {}, a1[NO][8] = {}, a0b[NO][8] = {}, a1b[NO][8] = {};
    for (unsigned int group = lane; group < groups; group += group_stride) {
        const unsigned int g2 = group + lanes_per_out;
        const bool g2v = g2 < groups;
        int w0[NO][4], w1[NO][4], w0b[NO][4], w1b[NO][4];
        float s0[NO], s1[NO], s0b[NO], s1b[NO];
        #pragma unroll
        for (int o = 0; o < NO; ++o) {
            const unsigned int n = nbase + o;
            const bool vn = n < N;
            const unsigned long long p0 = vn ? *(const unsigned long long*)(B0p + (unsigned long long)n * half_k + group * 8) : 0ull;
            const unsigned long long p1 = vn ? *(const unsigned long long*)(B1p + (unsigned long long)n * half_k + group * 8) : 0ull;
            const unsigned long long p0b = (vn && g2v) ? *(const unsigned long long*)(B0p + (unsigned long long)n * half_k + g2 * 8) : 0ull;
            const unsigned long long p1b = (vn && g2v) ? *(const unsigned long long*)(B1p + (unsigned long long)n * half_k + g2 * 8) : 0ull;
            int2 f, s;
            f = dp4a_expand_codebook_d4((unsigned int)p0);  s = dp4a_expand_codebook_d4((unsigned int)(p0 >> 32));
            w0[o][0] = f.x; w0[o][1] = f.y; w0[o][2] = s.x; w0[o][3] = s.y;
            f = dp4a_expand_codebook_d4((unsigned int)p1);  s = dp4a_expand_codebook_d4((unsigned int)(p1 >> 32));
            w1[o][0] = f.x; w1[o][1] = f.y; w1[o][2] = s.x; w1[o][3] = s.y;
            f = dp4a_expand_codebook_d4((unsigned int)p0b); s = dp4a_expand_codebook_d4((unsigned int)(p0b >> 32));
            w0b[o][0] = f.x; w0b[o][1] = f.y; w0b[o][2] = s.x; w0b[o][3] = s.y;
            f = dp4a_expand_codebook_d4((unsigned int)p1b); s = dp4a_expand_codebook_d4((unsigned int)(p1b >> 32));
            w1b[o][0] = f.x; w1b[o][1] = f.y; w1b[o][2] = s.x; w1b[o][3] = s.y;
            s0[o] = vn ? dp4a_scl_fp8(B0s[(unsigned long long)n * groups + group]) * (0.5f * B0s2) : 0.0f;
            s1[o] = vn ? dp4a_scl_fp8(B1s[(unsigned long long)n * groups + group]) * (0.5f * B1s2) : 0.0f;
            s0b[o] = (vn && g2v) ? dp4a_scl_fp8(B0s[(unsigned long long)n * groups + g2]) * (0.5f * B0s2) : 0.0f;
            s1b[o] = (vn && g2v) ? dp4a_scl_fp8(B1s[(unsigned long long)n * groups + g2]) * (0.5f * B1s2) : 0.0f;
        }
        #pragma unroll
        for (int row = 0; row < 8; ++row) {
            if ((unsigned int)row >= M) continue;
            const int4 act = *(const int4*)(a_q + (unsigned long long)row * K + group * DP4A_GROUP_SIZE);
            const int v[4] = {act.x, act.y, act.z, act.w};
            const float as = a_scale[(unsigned long long)row * groups + group];
            #pragma unroll
            for (int o = 0; o < NO; ++o) {
                int d0 = 0, d1 = 0;
                #pragma unroll
                for (int i = 0; i < 4; ++i) { d0 = DP4A_DOT(v[i], w0[o][i], d0); d1 = DP4A_DOT(v[i], w1[o][i], d1); }
                a0[o][row] += (float)d0 * as * s0[o];
                a1[o][row] += (float)d1 * as * s1[o];
            }
        }
        if (g2v) {
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                if ((unsigned int)row >= M) continue;
                const int4 act = *(const int4*)(a_q + (unsigned long long)row * K + g2 * DP4A_GROUP_SIZE);
                const int v[4] = {act.x, act.y, act.z, act.w};
                const float as = a_scale[(unsigned long long)row * groups + g2];
                #pragma unroll
                for (int o = 0; o < NO; ++o) {
                    int d0 = 0, d1 = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) { d0 = DP4A_DOT(v[i], w0b[o][i], d0); d1 = DP4A_DOT(v[i], w1b[o][i], d1); }
                    a0b[o][row] += (float)d0 * as * s0b[o];
                    a1b[o][row] += (float)d1 * as * s1b[o];
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        if ((unsigned int)row >= M) continue;
        #pragma unroll
        for (int o = 0; o < NO; ++o) {
            float v0 = a0[o][row], v0b = a0b[o][row], v1 = a1[o][row], v1b = a1b[o][row];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v0 += __shfl_down_sync(0xffffffffu, v0, off); v0b += __shfl_down_sync(0xffffffffu, v0b, off);
                v1 += __shfl_down_sync(0xffffffffu, v1, off); v1b += __shfl_down_sync(0xffffffffu, v1b, off);
            }
            const unsigned int n = nbase + o;
            if (lane == 0 && n < N) {
                C0[(unsigned long long)row * N + n] = __float2bfloat16(v0 + v0b);
                C1[(unsigned long long)row * N + n] = __float2bfloat16(v1 + v1b);
            }
        }
    }
}

#define DUAL_ARGS const signed char* a, const float* as, const unsigned char* b0, const unsigned char* s0, const float s02, __nv_bfloat16* c0, \
                  const unsigned char* b1, const unsigned char* s1, const float s12, __nv_bfloat16* c1, unsigned int M, unsigned int N, unsigned int K
extern "C" __global__ void __launch_bounds__(256) cand_dual_n2(DUAL_ARGS) { dual8_multi<2>(a, as, b0, s0, s02, c0, b1, s1, s12, c1, M, N, K); }
extern "C" __global__ void __launch_bounds__(256) cand_dual_n4(DUAL_ARGS) { dual8_multi<4>(a, as, b0, s0, s02, c0, b1, s1, s12, c1, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 2) cand_dual_n1_lb2(DUAL_ARGS) { dual8_multi<1>(a, as, b0, s0, s02, c0, b1, s1, s12, c1, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 3) cand_dual_n1_lb3(DUAL_ARGS) { dual8_multi<1>(a, as, b0, s0, s02, c0, b1, s1, s12, c1, M, N, K); }
extern "C" __global__ void __launch_bounds__(256) cand_dual_n1(DUAL_ARGS) { dual8_multi<1>(a, as, b0, s0, s02, c0, b1, s1, s12, c1, M, N, K); }

typedef void (*DFn)(DUAL_ARGS);
struct Cand { const char* name; DFn fn; int outs_per_block; };

int main() {
    // 27B MLP gate/up: N = intermediate 17408, K = hidden 5120, M = 8 decode rows
    const unsigned int shapes[][2] = {{17408, 5120}};
    // Loop #2: weights rotated across >256 MB of copies (DRAM-realistic) and occupancy arms; loop #1's n2/n4 lost.
    const Cand cands[] = {{"vl2 (shipped)", w4a16_gemv_dp4a_dual_batch8_d4_dyn_vl2, 8}};
    const unsigned int M = 8;
    for (auto& sh : shapes) {
        const unsigned int N = sh[0], K = sh[1], G = K / 16;
        std::vector<unsigned char> hB0((size_t)N * K / 2), hB1(hB0.size()), hS0((size_t)N * G), hS1(hS0.size());
        std::vector<signed char> hA((size_t)M * K); std::vector<float> hAs((size_t)M * G);
        srand(4321);
        for (auto& b : hB0) b = rand() & 0xff;
        for (auto& b : hB1) b = rand() & 0xff;
        for (auto& s : hS0) s = 0x30 + (rand() % 0x20);   // finite e4m3 scales
        for (auto& s : hS1) s = 0x30 + (rand() % 0x20);
        for (auto& a : hA) a = (signed char)((rand() % 255) - 127);
        for (auto& s : hAs) s = 1e-3f + (rand() % 100) * 1e-5f;
        unsigned char *dB0, *dB1, *dS0, *dS1; signed char* dA; float* dAs; __nv_bfloat16 *dC0, *dC1;
        CHECK(hipMalloc(&dB0, hB0.size())); CHECK(hipMalloc(&dB1, hB1.size())); CHECK(hipMalloc(&dS0, hS0.size())); CHECK(hipMalloc(&dS1, hS1.size()));
        CHECK(hipMalloc(&dA, hA.size())); CHECK(hipMalloc(&dAs, hAs.size() * 4)); CHECK(hipMalloc(&dC0, (size_t)M * N * 2)); CHECK(hipMalloc(&dC1, (size_t)M * N * 2));
        CHECK(hipMemcpy(dB0, hB0.data(), hB0.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dB1, hB1.data(), hB1.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dS0, hS0.data(), hS0.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dS1, hS1.data(), hS1.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dAs, hAs.data(), hAs.size() * 4, hipMemcpyHostToDevice));
        const double bytes = 2.0 * ((double)N * K / 2 + (double)N * G);
        const int NROT = (int)(256e6 / bytes) + 2;
        std::vector<unsigned char*> r0b(NROT), r1b(NROT);
        for (int r = 0; r < NROT; ++r) {
            CHECK(hipMalloc(&r0b[r], hB0.size())); CHECK(hipMemcpy(r0b[r], hB0.data(), hB0.size(), hipMemcpyHostToDevice));
            CHECK(hipMalloc(&r1b[r], hB1.size())); CHECK(hipMemcpy(r1b[r], hB1.data(), hB1.size(), hipMemcpyHostToDevice));
        }
        std::vector<unsigned short> r0((size_t)M * N), r1(r0.size()), o0(r0.size()), o1(r0.size());
        printf("-- dual gate+up N=%u K=%u M=%u (weights+scales %.1f MB)\n", N, K, M, bytes / 1e6);
        for (int ci = 0; ci < (int)(sizeof(cands) / sizeof(cands[0])); ++ci) {
            const Cand& c = cands[ci]; dim3 grid((N + c.outs_per_block - 1) / c.outs_per_block), block(256);
            CHECK(hipMemset(dC0, 0, (size_t)M * N * 2)); CHECK(hipMemset(dC1, 0, (size_t)M * N * 2));
            hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, dB0, dS0, 1.0f, dC0, dB1, dS1, 1.0f, dC1, M, N, K);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(ci ? o0.data() : r0.data(), dC0, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(ci ? o1.data() : r1.data(), dC1, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            for (int w = 0; w < 5; ++w) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, dB0, dS0, 1.0f, dC0, dB1, dS1, 1.0f, dC1, M, N, K);
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int iters = 50; CHECK(hipEventRecord(e0));
            for (int it = 0; it < iters; ++it) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, r0b[it % NROT], dS0, 1.0f, dC0, r1b[it % NROT], dS1, 1.0f, dC1, M, N, K);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            size_t diff = 0; if (ci) for (size_t i = 0; i < o0.size(); ++i) diff += (o0[i] != r0[i]) + (o1[i] != r1[i]);
            printf("   %-14s %8.1f us  %6.1f GB/s  %s\n", c.name, us, bytes / us / 1e3, ci == 0 ? "(reference)" : (diff == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
            if (ci && diff) printf("      %zu of %zu outputs differ\n", diff, 2 * o0.size());
        }
        for (unsigned int m4 : {4u, 2u}) {   // Loop #7: the m 2..4 tier (dual_batch4_d4 vs 2x single batch4_d4)
            dim3 grid((N + DP4A_N_PER_BLOCK - 1) / DP4A_N_PER_BLOCK), block(256);
            std::vector<unsigned short> d0(r0.size()), d1(r0.size()), q0(r0.size()), q1(r0.size());
            auto dual4 = m4 == 4 ? w4a16_gemv_dp4a_dual_batch4_d4 : w4a16_gemv_dp4a_dual_batch4_d4_dyn;
            auto single4 = m4 == 4 ? w4a16_gemv_dp4a_batch4_d4 : w4a16_gemv_dp4a_batch4_d4_dyn;
            double us_d = 0, us_s = 0;
            for (int arm = 0; arm < 2; ++arm) {
                CHECK(hipMemset(dC0, 0, (size_t)M * N * 2)); CHECK(hipMemset(dC1, 0, (size_t)M * N * 2));
                auto run = [&](int it) {
                    unsigned char* b0 = it < 0 ? dB0 : r0b[it % NROT]; unsigned char* b1 = it < 0 ? dB1 : r1b[it % NROT];
                    if (arm == 0) hipLaunchKernelGGL(dual4, grid, block, 0, 0, dA, dAs, b0, dS0, 1.0f, dC0, b1, dS1, 1.0f, dC1, m4, N, K);
                    else { hipLaunchKernelGGL(single4, grid, block, 0, 0, dA, dAs, b0, dS0, 1.0f, dC0, m4, N, K);
                           hipLaunchKernelGGL(single4, grid, block, 0, 0, dA, dAs, b1, dS1, 1.0f, dC1, m4, N, K); }
                };
                run(-1); CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(arm ? q0.data() : d0.data(), dC0, (size_t)M * N * 2, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(arm ? q1.data() : d1.data(), dC1, (size_t)M * N * 2, hipMemcpyDeviceToHost));
                hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
                const int iters = 50; CHECK(hipEventRecord(e0));
                for (int it = 0; it < iters; ++it) run(it);
                CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
                (arm ? us_s : us_d) = ms * 1000.0 / iters;
            }
            size_t diff = 0; for (size_t i = 0; i < (size_t)m4 * N; ++i) diff += (q0[i] != d0[i]) + (q1[i] != d1[i]);
            printf("   m=%u dual_batch4 %8.1f us %6.1f GB/s | 2x single batch4 %8.1f us %6.1f GB/s  %s\n", m4, us_d, bytes / us_d / 1e3,
                   us_s, bytes / us_s / 1e3, diff == 0 ? "BIT-IDENTICAL" : "DIFFERS");
        }
        {   // Loop #6: two launches of the single-matrix w4a16_gemv_dp4a_batch8_d4 (gate, then up). Its 64 threads per
            // output split the groups across two warps exactly like the dual's two virtual lanes, with the same
            // accumulate expression and reduce -> expected BIT-IDENTICAL to the dual kernel, at the single's ~187 GB/s.
            dim3 grid((N + DP4A_N_PER_BLOCK - 1) / DP4A_N_PER_BLOCK), block(256);
            CHECK(hipMemset(dC0, 0, (size_t)M * N * 2)); CHECK(hipMemset(dC1, 0, (size_t)M * N * 2));
            hipLaunchKernelGGL(w4a16_gemv_dp4a_batch8_d4, grid, block, 0, 0, dA, dAs, dB0, dS0, 1.0f, dC0, M, N, K);
            hipLaunchKernelGGL(w4a16_gemv_dp4a_batch8_d4, grid, block, 0, 0, dA, dAs, dB1, dS1, 1.0f, dC1, M, N, K);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(o0.data(), dC0, (size_t)M * N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(o1.data(), dC1, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int iters = 50; CHECK(hipEventRecord(e0));
            for (int it = 0; it < iters; ++it) {
                hipLaunchKernelGGL(w4a16_gemv_dp4a_batch8_d4, grid, block, 0, 0, dA, dAs, r0b[it % NROT], dS0, 1.0f, dC0, M, N, K);
                hipLaunchKernelGGL(w4a16_gemv_dp4a_batch8_d4, grid, block, 0, 0, dA, dAs, r1b[it % NROT], dS1, 1.0f, dC1, M, N, K);
            }
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            size_t diff = 0; for (size_t i = 0; i < o0.size(); ++i) diff += (o0[i] != r0[i]) + (o1[i] != r1[i]);
            printf("   %-14s %8.1f us  %6.1f GB/s  %s\n", "2x single d4", us, bytes / us / 1e3, diff == 0 ? "BIT-IDENTICAL" : "DIFFERS");
            if (diff) printf("      %zu of %zu outputs differ\n", diff, 2 * o0.size());
        }
    }
    return 0;
}
