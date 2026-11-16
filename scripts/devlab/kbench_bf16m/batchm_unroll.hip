// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #159 (Flash-Next C8, user priority): dense_gemv_bf16_batchm is 14% of C8 decode (fn-g 43808029150). At the
// GDN out_proj shape (N 2560, K 6144, M 8) it takes 434 us (~72 GB/s), while qkvz (N 16384, K 2560) runs 383 us
// (~219 GB/s, the ceiling). Each thread holds one 16 B weight load in flight and every 64-uint4 k step costs two block
// barriers. Twin: U steps per barrier pair. Stage U x 64 activation uint4s per row, issue the U weight loads before the
// barrier, then run the same per-thread kv sequence and dot2 order -> bit-identical.
#include "dense_gemv_bf16_batchm.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template<int U>
__global__ void __launch_bounds__(256) batchm_u(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK, local_out = threadIdx.x / threads_per_out, lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out; const bool live = n < N;
    const unsigned int m = (M > MAX_M) ? MAX_M : M;
    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;
    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);
    __shared__ uint4 s_a[MAX_M][U * 64];
    const unsigned int steps = (K_VEC + threads_per_out - 1) / threads_per_out;
    for (unsigned int s0 = 0; s0 < steps; s0 += U) {
        uint4 bd[U];
        #pragma unroll
        for (int q = 0; q < U; q++) { const unsigned int kv = (s0 + q) * threads_per_out + lane; if (live && s0 + q < steps && kv < K_VEC) bd[q] = B_vec[kv]; }
        for (unsigned int i = threadIdx.x; i < m * U * threads_per_out; i += BLOCK_SIZE) {
            const unsigned int t = i / (U * threads_per_out), v = i % (U * threads_per_out), kv = s0 * threads_per_out + v;
            if (kv < K_VEC) s_a[t][v] = ((const uint4*)(A + (unsigned long long)t * K))[kv]; }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < U; q++) {
            const unsigned int kv = (s0 + q) * threads_per_out + lane;
            if (live && s0 + q < steps && kv < K_VEC) {
                const unsigned int b_raw[4] = {bd[q].x, bd[q].y, bd[q].z, bd[q].w};
                for (unsigned int t = 0; t < m; t++) {
                    const uint4 a_data = s_a[t][q * threads_per_out + lane];
                    const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                    float a = acc[t];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(a) : "v"(a_raw[i]), "v"(b_raw[i]));
                    acc[t] = a;
                }
            }
        }
        __syncthreads();
    }
    if (!live) return;
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    for (unsigned int t = 0; t < m; t++) { float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        acc[t] = a; }
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    if (warp_lane == 0) { const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE); for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t]; }
    __syncthreads();
    if (lane == 0) for (unsigned int t = 0; t < m; t++) C[(unsigned long long)t * out_stride + n] = __float2bfloat16(smem[t][local_out * 2] + smem[t][local_out * 2 + 1]);
}

int main() {
    struct S { unsigned N, K, M; const char* nm; } shapes[] = {{2560, 6144, 8, "gdn out_proj"}, {2560, 6144, 4, "gdn out_proj"}, {16384, 2560, 8, "gdn qkvz"}, {2560, 8192, 8, "K 8192"}};
    for (auto& sh : shapes) {
        const unsigned N = sh.N, K = sh.K, M = sh.M; const int NB = 6;   // rotate weights past the 32 MB MALL
        std::vector<__nv_bfloat16> hb((size_t)N * K), ha((size_t)M * K); srand(N + K + M);
        for (auto& x : hb) x = __float2bfloat16(((rand() % 2001) - 1000) / 4000.f); for (auto& x : ha) x = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *dB[NB], *dA, *dC[4];
        for (int i = 0; i < NB; i++) { CHECK(hipMalloc(&dB[i], hb.size() * 2)); CHECK(hipMemcpy(dB[i], hb.data(), hb.size() * 2, hipMemcpyHostToDevice)); }
        CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (int v = 0; v < 4; v++) CHECK(hipMalloc(&dC[v], (size_t)M * N * 2));
        const dim3 g((N + N_PER_BLOCK - 1) / N_PER_BLOCK), b(BLOCK_SIZE);
        auto launch = [&](int v, int i) {
            if (v == 0) hipLaunchKernelGGL(dense_gemv_bf16_batchm, g, b, 0, 0, dA, dB[i], dC[0], M, N, K, N);
            else if (v == 1) hipLaunchKernelGGL(batchm_u<2>, g, b, 0, 0, dA, dB[i], dC[1], M, N, K, N);
            else if (v == 2) hipLaunchKernelGGL(batchm_u<4>, g, b, 0, 0, dA, dB[i], dC[2], M, N, K, N);
            else hipLaunchKernelGGL(batchm_u<1>, g, b, 0, 0, dA, dB[i], dC[3], M, N, K, N); };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[4];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 4; v++) { for (int i = 0; i < 3; i++) launch(v, i % NB); CHECK(hipEventRecord(e0)); for (int i = 0; i < 30; i++) launch(v, i % NB); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 30; }
        for (int v = 0; v < 4; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> c0((size_t)M * N), c1((size_t)M * N); CHECK(hipMemcpy(c0.data(), dC[0], c0.size() * 2, hipMemcpyDeviceToHost));
        const double gb = (double)N * K * 2 / 1e3;
        printf("-- %-13s N=%5u K=%u M=%u: ship %6.1f us (%3.0f GB/s)", sh.nm, N, K, M, t[0], gb / t[0]);
        const char* nm[4] = {"", "U2", "U4", "U1"};
        for (int v = 1; v < 4; v++) { CHECK(hipMemcpy(c1.data(), dC[v], c1.size() * 2, hipMemcpyDeviceToHost));
            printf(" | %s %6.1f x%.2f %s", nm[v], t[v], t[0] / t[v], memcmp(c0.data(), c1.data(), c0.size() * 2) ? "DIFF" : "same"); }
        printf("\n");
        for (int i = 0; i < NB; i++) CHECK(hipFree(dB[i])); CHECK(hipFree(dA)); for (int v = 0; v < 4; v++) CHECK(hipFree(dC[v]));
    }
    return 0;
}
