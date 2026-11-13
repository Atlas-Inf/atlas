// SPDX-License-Identifier: AGPL-3.0-only
// kbench (KLOOP #87, /v1/score latency on 27B NVFP4): rocprof of a score pass (DevLab 43496724984) puts
// w8a16_gemm_n_m128 (the FP8 block-scaled [N,K] projections) at 39.7% of GPU time, 1.44 ms per call. A score pass
// has M ~ 56-140 rows, and n_m128 launches only N/128 CTAs per 128 rows (40 CTAs for N=5120), each with 8 warps of
// which most hold padding rows. Candidate w8m32<NT,KS>: 128 threads, 32 rows x NT cols per CTA, KS 32-K steps
// staged per barrier (the #354/#359 m32 deep-K shape). Same A bf16, same B dequant bf16(e4m3 * block_scale), and the
// same 16x16x16 bf16 WMMAs in the same k order per output, so it must be BIT-IDENTICAL to n_m128.
#include "w8a16_gemm_n_m128.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int NT, int KS>
__global__ __launch_bounds__(128) void w8m32(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    constexpr int NBW = NT / 64;            // 16-col tiles per warp
    constexpr int BPT = NT / 64;            // B uint4 per thread per step (NT x 32 bytes / 16 / 128)
    const unsigned int cta_n = blockIdx.x * NT, cta_m = blockIdx.y * 32;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;
    __shared__ __align__(16) __nv_bfloat16 sA[KS][32][W8N128_KSTEP + W8N128_APAD];
    __shared__ __align__(16) __nv_bfloat16 sB[KS][NT][W8N128_KSTEP + W8N128_BPAD];
    v8f acc[2][NBW];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int j = 0; j < NBW; j++) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int a_row = threadIdx.x >> 2, a_col = (threadIdx.x & 3) << 3;
    const unsigned int ksb = (K + W8N128_FP8B - 1) / W8N128_FP8B, n_block = cta_n >> 7;
    auto load = [&](unsigned int kb, uint4& ra, uint4 (&rb)[BPT], float& rs) {
        const unsigned int gr = cta_m + a_row, gc = kb + a_col;
        if (gr < M && gc + 7 < K && (K & 7) == 0) ra = *(const uint4*)&A[(unsigned long long)gr * K + gc];
        else { union { __nv_bfloat16 h[8]; uint4 v; } u;
            #pragma unroll
            for (int i = 0; i < 8; i++) u.h[i] = (gr < M && gc + i < K) ? A[(unsigned long long)gr * K + gc + i] : __float2bfloat16(0.0f);
            ra = u.v; }
        #pragma unroll
        for (int u4 = 0; u4 < BPT; u4++) {
            const unsigned int id = threadIdx.x + u4 * 128, gn = cta_n + (id >> 1), gk = kb + ((id & 1) << 4);
            if (gn < N && gk + 15 < K && (K & 15) == 0) rb[u4] = *(const uint4*)&B[(unsigned long long)gn * K + gk];
            else { union { unsigned char b[16]; uint4 v; } u;
                #pragma unroll
                for (int i = 0; i < 16; i++) u.b[i] = (gn < N && gk + i < K) ? B[(unsigned long long)gn * K + gk + i] : 0;
                rb[u4] = u.v; }
        }
        rs = (kb < K) ? block_scale[(unsigned long long)n_block * ksb + (kb >> 7)] : 0.0f;
    };
    auto store = [&](int s, const uint4& ra, const uint4 (&rb)[BPT], float rs) {
        *(uint4*)&sA[s][a_row][a_col] = ra;
        #pragma unroll
        for (int u4 = 0; u4 < BPT; u4++) {
            const unsigned int id = threadIdx.x + u4 * 128, n = id >> 1, ko = (id & 1) << 4;
            union { __nv_bfloat16 h[16]; uint4 v[2]; } u;
            const unsigned int rw[4] = {rb[u4].x, rb[u4].y, rb[u4].z, rb[u4].w};
            #pragma unroll
            for (int i = 0; i < 16; i++) u.h[i] = __float2bfloat16(w8n128_e4m3((unsigned char)((rw[i >> 2] >> ((i & 3) * 8)) & 0xFFu)) * rs);
            *(uint4*)&sB[s][n][ko] = u.v[0];
            *(uint4*)&sB[s][n][ko + 8] = u.v[1];
        }
    };
    auto compute = [&](int s) {
        const unsigned int r0 = lane_id & 15;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            v16bf a0, a1;
            memcpy(&a0, &sA[s][r0][h * 16], 32);
            memcpy(&a1, &sA[s][16 + r0][h * 16], 32);
            #pragma unroll
            for (int j = 0; j < NBW; j++) {
                const unsigned int nc = (warp_id * NBW + j) * 16 + (lane_id & 15);
                v16bf b;
                memcpy(&b, &sB[s][nc][h * 16], 32);
                acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a0, b, acc[0][j]);
                acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a1, b, acc[1][j]);
            }
        }
    };
    uint4 ra[KS], rb[KS][BPT]; float rs[KS];
    #pragma unroll
    for (int s = 0; s < KS; s++) load(s * W8N128_KSTEP, ra[s], rb[s], rs[s]);
    #pragma unroll
    for (int s = 0; s < KS; s++) store(s, ra[s], rb[s], rs[s]);
    __syncthreads();
    unsigned int k_cur = 0;
    for (unsigned int kb = KS * W8N128_KSTEP; kb < K; kb += KS * W8N128_KSTEP) {
        #pragma unroll
        for (int s = 0; s < KS; s++) load(kb + s * W8N128_KSTEP, ra[s], rb[s], rs[s]);
        #pragma unroll
        for (int s = 0; s < KS; s++) compute(s);
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < KS; s++) store(s, ra[s], rb[s], rs[s]);
        __syncthreads();
        k_cur = kb;
    }
    #pragma unroll
    for (int s = 0; s < KS; s++)
        if (k_cur + s * W8N128_KSTEP < K) compute(s);
    #pragma unroll
    for (int rt = 0; rt < 2; rt++)
        #pragma unroll
        for (int j = 0; j < NBW; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = cta_m + rt * 16 + 2 * e + (lane_id >> 4);
                const unsigned int c = cta_n + (warp_id * NBW + j) * 16 + (lane_id & 15);
                if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[rt][j][e]);
            }
}

int main() {
    struct S { unsigned N, K; } shapes[] = {{16384, 5120}, {12288, 5120}, {5120, 6144}, {5120, 5120}, {2048, 5120}, {1024, 5120}, {5120, 17408}};
    struct V { const char* name; int nt; void (*f)(const __nv_bfloat16*, const unsigned char*, const float*, __nv_bfloat16*, unsigned, unsigned, unsigned); };
    V vs[] = {{"m32 n64 KS1", 64, w8m32<64, 1>}, {"m32 n64 KS2", 64, w8m32<64, 2>}, {"m32 n64 KS4", 64, w8m32<64, 4>},
              {"m32 n128 KS1", 128, w8m32<128, 1>}, {"m32 n128 KS2", 128, w8m32<128, 2>}};
    for (auto& s : shapes) for (unsigned M : {17u, 56u, 140u}) {
        const unsigned N = s.N, K = s.K;
        const size_t wb = (size_t)N * K, sb = (size_t)(N / 128) * (K / 128);
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wb + 1);
        std::vector<unsigned char> hb(wb); std::vector<float> hs(sb);
        srand(3); for (auto& b : hb) { b = rand() & 0xFF; if ((b & 0x7F) == 0x7F) b ^= 1; }
        for (auto& x : hs) x = ldexpf(1.0f + (rand() % 1000) / 1000.f, -(rand() % 12) - 4);
        std::vector<unsigned char*> P(copies); std::vector<float*> Q(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], wb)); CHECK(hipMemcpy(P[c], hb.data(), wb, hipMemcpyHostToDevice));
            CHECK(hipMalloc(&Q[c], sb * 4)); CHECK(hipMemcpy(Q[c], hs.data(), sb * 4, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> ha((size_t)M * K); for (auto& a : ha) a = __float2bfloat16(((rand() % 2001) - 1000) / 500.f);
        __nv_bfloat16 *A, *C0, *C1; CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto timeit = [&](auto f) { for (int i = 0; i < 3; i++) f(i % copies); const int it = 30; CHECK(hipEventRecord(e0));
            for (int i = 0; i < it; i++) f(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / it; };
        auto ref = [&](int c) { hipLaunchKernelGGL(w8a16_gemm_n_m128, dim3((N + 127) / 128, (M + 127) / 128), dim3(256), 0, 0, A, P[c], Q[c], C0, M, N, K); };
        const float t0 = timeit(ref); ref(0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost));
        printf("-- N=%u K=%u M=%u (%.1f MB) | n_m128 %8.1f us %6.1f GB/s\n", N, K, M, wb / 1e6, t0, wb / 1e3 / t0);
        for (auto& v : vs) {
            auto l = [&](int c) { hipLaunchKernelGGL(v.f, dim3((N + v.nt - 1) / v.nt, (M + 31) / 32), dim3(128), 0, 0, A, P[c], Q[c], C1, M, N, K); };
            const float t = timeit(l); CHECK(hipMemset(C1, 0xFF, (size_t)M * N * 2)); l(0); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
            size_t neq = 0; for (size_t i = 0; i < r0.size(); i++) neq += memcmp(&r0[i], &r1[i], 2) != 0;
            printf("   %-13s %8.1f us %6.1f GB/s  x%.2f  (%zu/%zu differ)\n", v.name, t, wb / 1e3 / t, t0 / t, neq, r0.size());
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); } hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
