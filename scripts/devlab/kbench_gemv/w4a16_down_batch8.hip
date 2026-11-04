// SPDX-License-Identifier: AGPL-3.0-only
// kbench loop #4: w4a16_gemv_dp4a_batch8_d4 (27B MLP down at C8 decode; 11% of a step on strix s210, ~128 GB/s).
// Each of the 64 lanes per output issues ONE 8-byte weight load per loop iteration. Candidates unroll the group loop
// by U, issuing all U weight words + scales before any DP4A; each lane still adds its groups in the same order
// (g, g+64, g+128, ...) into the same accumulator, and the reduce tail is unchanged -> must be BIT-IDENTICAL.
// Weights rotate over >256 MB of copies so DRAM, not the 32 MB MALL, serves them.
#include "w4a16_gemv_dp4a.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int U>
__device__ __forceinline__ void down8_unrolled(const signed char* __restrict__ a_q, const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B_packed, const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    const unsigned int threads_per_out = DP4A_BLOCK_SIZE / DP4A_N_PER_BLOCK;   // 64
    const unsigned int local_out = threadIdx.x / threads_per_out, lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * DP4A_N_PER_BLOCK + local_out;
    const bool valid_n = n < N;
    const unsigned int half_k = K / 2, groups = K / DP4A_GROUP_SIZE;
    float acc[8] = {};
    if (valid_n) {
        for (unsigned int g0 = lane; g0 < groups; g0 += threads_per_out * U) {
            unsigned long long packed[U]; float wsc[U]; bool gv[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                const unsigned int g = g0 + u * threads_per_out;
                gv[u] = g < groups;
                packed[u] = gv[u] ? *(const unsigned long long*)(B_packed + (unsigned long long)n * half_k + g * 8) : 0ull;
                wsc[u] = gv[u] ? dp4a_scl_fp8(B_scale[(unsigned long long)n * groups + g]) * (0.5f * scale2) : 0.0f;
            }
            #pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!gv[u]) break;
                const unsigned int g = g0 + u * threads_per_out;
                const int2 first = dp4a_expand_codebook_d4((unsigned int)packed[u]);
                const int2 second = dp4a_expand_codebook_d4((unsigned int)(packed[u] >> 32));
                const int weight[4] = {first.x, first.y, second.x, second.y};
                #pragma unroll
                for (int row = 0; row < 8; ++row) {
                    const int4 activation = *(const int4*)(a_q + (unsigned long long)row * K + g * DP4A_GROUP_SIZE);
                    const int values[4] = {activation.x, activation.y, activation.z, activation.w};
                    int dot = 0;
                    #pragma unroll
                    for (int index = 0; index < 4; ++index) dot = DP4A_DOT(values[index], weight[index], dot);
                    acc[row] += (float)dot * a_scale[(unsigned long long)row * groups + g] * wsc[u];
                }
            }
        }
    }
    __shared__ float partial[8][DP4A_N_PER_BLOCK * 2];
    const unsigned int warp_lane = lane % DP4A_WARP_SIZE, warp_in_out = lane / DP4A_WARP_SIZE;
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        float result = acc[row];
        #pragma unroll
        for (int offset = DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1) result += __shfl_down_sync(0xffffffffu, result, offset);
        if (warp_lane == 0) partial[row][local_out * 2 + warp_in_out] = result;
    }
    __syncthreads();
    if (valid_n && lane == 0) {
        #pragma unroll
        for (int row = 0; row < 8; ++row)
            C[(unsigned long long)row * N + n] = __float2bfloat16(partial[row][local_out * 2] + partial[row][local_out * 2 + 1]);
    }
}

#define DOWN_ARGS const signed char* a, const float* as, const unsigned char* b, const unsigned char* s, const float s2, __nv_bfloat16* c, unsigned int M, unsigned int N, unsigned int K
extern "C" __global__ void __launch_bounds__(256) cand_down_u1(DOWN_ARGS) { down8_unrolled<1>(a, as, b, s, s2, c, M, N, K); }
extern "C" __global__ void __launch_bounds__(256) cand_down_u2(DOWN_ARGS) { down8_unrolled<2>(a, as, b, s, s2, c, M, N, K); }
extern "C" __global__ void __launch_bounds__(256) cand_down_u4(DOWN_ARGS) { down8_unrolled<4>(a, as, b, s, s2, c, M, N, K); }

typedef void (*DFn)(DOWN_ARGS);
struct Cand { const char* name; DFn fn; };

int main() {
    // 27B MLP down: N = hidden 5120, K = intermediate 17408, M = 8. Plus the 27B lm_head-like wide shape for reference.
    const unsigned int shapes[][2] = {{5120, 17408}};
    const Cand cands[] = {{"batch8_d4 (shipped)", w4a16_gemv_dp4a_batch8_d4}, {"u1 (rewrite)", cand_down_u1},
                          {"u2", cand_down_u2}, {"u4", cand_down_u4}};
    const unsigned int M = 8;
    for (auto& sh : shapes) {
        const unsigned int N = sh[0], K = sh[1], G = K / 16;
        std::vector<unsigned char> hB((size_t)N * K / 2), hS((size_t)N * G);
        std::vector<signed char> hA((size_t)M * K); std::vector<float> hAs((size_t)M * G);
        srand(777);
        for (auto& b : hB) b = rand() & 0xff;
        for (auto& s : hS) s = 0x30 + (rand() % 0x20);
        for (auto& a : hA) a = (signed char)((rand() % 255) - 127);
        for (auto& s : hAs) s = 1e-3f + (rand() % 100) * 1e-5f;
        const double bytes = (double)N * K / 2 + (double)N * G;
        const int NROT = (int)(256e6 / bytes) + 2;
        std::vector<unsigned char*> rb(NROT);
        for (int r = 0; r < NROT; ++r) { CHECK(hipMalloc(&rb[r], hB.size())); CHECK(hipMemcpy(rb[r], hB.data(), hB.size(), hipMemcpyHostToDevice)); }
        unsigned char* dS; signed char* dA; float* dAs; __nv_bfloat16* dC;
        CHECK(hipMalloc(&dS, hS.size())); CHECK(hipMalloc(&dA, hA.size())); CHECK(hipMalloc(&dAs, hAs.size() * 4)); CHECK(hipMalloc(&dC, (size_t)M * N * 2));
        CHECK(hipMemcpy(dS, hS.data(), hS.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dAs, hAs.data(), hAs.size() * 4, hipMemcpyHostToDevice));
        std::vector<unsigned short> ref((size_t)M * N), out(ref.size());
        printf("-- down N=%u K=%u M=%u (weights+scales %.1f MB, %d rotated copies)\n", N, K, M, bytes / 1e6, NROT);
        for (int ci = 0; ci < (int)(sizeof(cands) / sizeof(cands[0])); ++ci) {
            const Cand& c = cands[ci]; dim3 grid((N + DP4A_N_PER_BLOCK - 1) / DP4A_N_PER_BLOCK), block(256);
            CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
            hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, rb[0], dS, 1.0f, dC, M, N, K); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(ci ? out.data() : ref.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            for (int w = 0; w < 5; ++w) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, rb[w % NROT], dS, 1.0f, dC, M, N, K);
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int iters = 50; CHECK(hipEventRecord(e0));
            for (int it = 0; it < iters; ++it) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dAs, rb[it % NROT], dS, 1.0f, dC, M, N, K);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            size_t diff = 0; if (ci) for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
            printf("   %-20s %8.1f us  %6.1f GB/s  %s\n", c.name, us, bytes / us / 1e3, ci == 0 ? "(reference)" : (diff == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
            if (ci && diff) printf("      %zu of %zu outputs differ\n", diff, out.size());
        }
    }
    return 0;
}
