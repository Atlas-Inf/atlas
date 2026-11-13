// SPDX-License-Identifier: AGPL-3.0-only
// kbench (KLOOP #71, Ripple DFlash2 drafter): the drafter's gamma=16 projections and LM head run w4a16_gemv_batch16
// on the NATIVE [N, K/2] NVFP4 layout at 47-63 GB/s, 26.9% of DFlash2 decode GPU time. KLOOP #70 showed it is
// dot2-issue-bound (an LDS-staged A tile did not help), so the lever is WMMA. Candidate m32n<NT>: the #354 m32 twin
// with only the B load/store rewritten for the native layout (one 16-byte K-run + 2 scale bytes per output column per
// 32-K step). Its dequantized f16 B tile is the same as m32's on the transposed copy, so it must be BIT-IDENTICAL to
// m128_f16a / m32 on the transposed weights; timing is compared with batch16 from KLOOP #70 (q|gate 993, o 559,
// gate/up 1884, down 1263, fc 1305, lm_head 4571 us).
#include "../q27/w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
static inline float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int NT>
__global__ __launch_bounds__(128)
void w4a16_gemm_t_m32_f16a(const __nv_bfloat16* __restrict__ A_, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    constexpr int NBW = NT / 64;            // 16-col n-blocks per warp
    constexpr int TPR = NT / 16;            // threads per k-pair row of B (one uint4 = 16 columns each)
    const unsigned short* A = (const unsigned short*)A_;
    const unsigned int cta_n = blockIdx.x * NT;
    const unsigned int cta_m = blockIdx.y * 32;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;

    __shared__ __align__(16) _Float16 sA[32][F16_K_STEP + F16_PAD_T];
    #define M32_B_OFF(n) ((n) * 40 + ((n) >> 4) * 8)
    __shared__ __align__(16) _Float16 sB[M32_B_OFF(NT)];

    v8f acc[2][NBW];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int j = 0; j < NBW; j++) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int a_row = threadIdx.x >> 2;          // 0..31
    const unsigned int a_col = (threadIdx.x & 3) << 3;
    const bool b_on = threadIdx.x < 16 * TPR;
    const unsigned int b_kp = threadIdx.x / TPR;
    const unsigned int b_ns = (threadIdx.x % TPR) << 4;

    auto load = [&](unsigned int kb, uint4& ra, uint4& rb, uint4& rs) {
        const unsigned int gr = cta_m + a_row, gc = kb + a_col;
        ra = (gr < M && gc + 7 < K) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        rb = rs = uint4{0, 0, 0, 0};
        if (b_on) {
            const unsigned int gke = kb + (b_kp << 1), gns = cta_n + b_ns;
            if (gke + 1 <= K && gns + 15 < N) rb = *(const uint4*)&B_packed[(unsigned long long)(gke >> 1) * N + gns];
            const unsigned int sg = kb / F16_GROUP + (b_kp >> 3);
            if (gns + 15 < N) rs = *(const uint4*)&B_scale[(unsigned long long)sg * N + gns];
        }
    };
    auto store = [&](const uint4& ra, const uint4& rb, const uint4& rs) {
        *(uint4*)&sA[a_row][a_col] = bf16x8_to_f16x8(ra);
        if (!b_on) return;
        const unsigned int* pw = (const unsigned int*)&rb;
        const unsigned int* ps = (const unsigned int*)&rs;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            const unsigned int hl = e2m1x4_f16hi(pw[d] & 0x0F0F0F0Fu);
            const unsigned int hh = e2m1x4_f16hi((pw[d] >> 4) & 0x0F0F0F0Fu);
            const v2h s01 = e4m3x2_f16(ps[d] & 0xFFFFu);
            const v2h s23 = e4m3x2_f16(ps[d] >> 16);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int sel = 0x0Cu | ((unsigned)j << 8) | (0x0Cu << 16) | ((4u + j) << 24);
                const unsigned int wb = __builtin_amdgcn_perm(hh, hl, sel);
                v2h w; __builtin_memcpy(&w, &wb, 4);
                const _Float16 s = (j == 0) ? s01.x : (j == 1) ? s01.y : (j == 2) ? s23.x : s23.y;
                w = w * (v2h){s, s};
                *(v2h*)&sB[M32_B_OFF(b_ns + d * 4 + j) + b_kp * 2] = w;
            }
        }
    };
    auto compute = [&]() {
        const unsigned int r0 = lane_id & 15;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            v16h a0, a1;
            __builtin_memcpy(&a0, &sA[r0][h * 16], 32);
            __builtin_memcpy(&a1, &sA[16 + r0][h * 16], 32);
            #pragma unroll
            for (int j = 0; j < NBW; j++) {
                const unsigned int nc = (warp_id * NBW + j) * 16 + (lane_id & 15);
                v16h b;
                __builtin_memcpy(&b, &sB[M32_B_OFF(nc) + h * 16], 32);
                acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0][j]);
                acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1][j]);
            }
        }
    };

    uint4 ra, rb, rs;
    load(0, ra, rb, rs);
    store(ra, rb, rs);
    __syncthreads();
    for (unsigned int k_base = F16_K_STEP; k_base < K; k_base += F16_K_STEP) {
        load(k_base, ra, rb, rs);
        compute();
        __syncthreads();
        store(ra, rb, rs);
        __syncthreads();
    }
    compute();
    #undef M32_B_OFF

    #pragma unroll
    for (int rt = 0; rt < 2; rt++)
        #pragma unroll
        for (int j = 0; j < NBW; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = cta_m + rt * 16 + 2 * e + (lane_id >> 4);
                const unsigned int c = cta_n + (warp_id * NBW + j) * 16 + (lane_id & 15);
                if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[rt][j][e] * scale2);
            }
}

template <int NT>
__global__ __launch_bounds__(128)
void w4a16_gemm_m32_native(const __nv_bfloat16* __restrict__ A_, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    constexpr int NBW = NT / 64;
    const unsigned short* A = (const unsigned short*)A_;
    const unsigned int cta_n = blockIdx.x * NT;
    const unsigned int cta_m = blockIdx.y * 32;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    __shared__ __align__(16) _Float16 sA[32][F16_K_STEP + F16_PAD_T];
    #define M32_B_OFF(n) ((n) * 40 + ((n) >> 4) * 8)
    __shared__ __align__(16) _Float16 sB[M32_B_OFF(NT)];
    v8f acc[2][NBW];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int j = 0; j < NBW; j++) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int a_row = threadIdx.x >> 2;
    const unsigned int a_col = (threadIdx.x & 3) << 3;
    const bool b_on = threadIdx.x < NT;
    const unsigned int half_K = K / 2, K16 = K / 16;
    auto load = [&](unsigned int kb, uint4& ra, uint4& rb, unsigned int& rs) {
        const unsigned int gr = cta_m + a_row, gc = kb + a_col;
        ra = (gr < M && gc + 7 < K) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        rb = uint4{0, 0, 0, 0}; rs = 0;
        const unsigned int n = cta_n + threadIdx.x;
        if (b_on && n < N && kb + 31 < K) {
            rb = *(const uint4*)&B_packed[(unsigned long long)n * half_K + (kb >> 1)];
            rs = *(const unsigned short*)&B_scale[(unsigned long long)n * K16 + (kb >> 4)];
        }
    };
    auto store = [&](const uint4& ra, const uint4& rb, unsigned int rs) {
        *(uint4*)&sA[a_row][a_col] = bf16x8_to_f16x8(ra);
        if (!b_on) return;
        const unsigned int* pw = (const unsigned int*)&rb;
        const v2h s01 = e4m3x2_f16(rs & 0xFFFFu);
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            const unsigned int hl = e2m1x4_f16hi(pw[d] & 0x0F0F0F0Fu);
            const unsigned int hh = e2m1x4_f16hi((pw[d] >> 4) & 0x0F0F0F0Fu);
            const _Float16 s = d < 2 ? s01.x : s01.y;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int sel = 0x0Cu | ((unsigned)j << 8) | (0x0Cu << 16) | ((4u + j) << 24);
                const unsigned int wb = __builtin_amdgcn_perm(hh, hl, sel);
                v2h w; __builtin_memcpy(&w, &wb, 4);
                w = w * (v2h){s, s};
                *(v2h*)&sB[M32_B_OFF(threadIdx.x) + d * 8 + j * 2] = w;
            }
        }
    };
    auto compute = [&]() {
        const unsigned int r0 = lane_id & 15;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            v16h a0, a1;
            __builtin_memcpy(&a0, &sA[r0][h * 16], 32);
            __builtin_memcpy(&a1, &sA[16 + r0][h * 16], 32);
            #pragma unroll
            for (int j = 0; j < NBW; j++) {
                const unsigned int nc = (warp_id * NBW + j) * 16 + (lane_id & 15);
                v16h b;
                __builtin_memcpy(&b, &sB[M32_B_OFF(nc) + h * 16], 32);
                acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0][j]);
                acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1][j]);
            }
        }
    };
    uint4 ra, rb; unsigned int rs;
    load(0, ra, rb, rs);
    store(ra, rb, rs);
    __syncthreads();
    for (unsigned int k_base = F16_K_STEP; k_base < K; k_base += F16_K_STEP) {
        load(k_base, ra, rb, rs);
        compute();
        __syncthreads();
        store(ra, rb, rs);
        __syncthreads();
    }
    compute();
    #undef M32_B_OFF
    #pragma unroll
    for (int rt = 0; rt < 2; rt++)
        #pragma unroll
        for (int j = 0; j < NBW; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = cta_m + rt * 16 + 2 * e + (lane_id >> 4);
                const unsigned int c = cta_n + (warp_id * NBW + j) * 16 + (lane_id & 15);
                if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[rt][j][e] * scale2);
            }
}

int main() {
    struct S { const char* name; unsigned int M, N, K; } shapes[] = {
        {"q|gate", 16, 20480, 5120}, {"k|v", 16, 2048, 5120}, {"o", 16, 5120, 10240},
        {"gate/up", 16, 40960, 5120}, {"down", 16, 5120, 20480}, {"fc 5x", 16, 5120, 25600},
        {"lm_head", 16, 100352, 5120}, {"wide lm_head-like", 16 | 0x100, 100278, 5120}, {"wide down M17", 17 | 0x100, 5120, 40960}};
    for (auto& s : shapes) {
        const bool wide = s.M & 0x100; const unsigned M = s.M & 0xFF, N = s.N, K = s.K;
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N, wbytes = pb + sb;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char> hp(pb), hs(sb), tp(pb), ts(sb);   // native [N][K/2], [N][K/16]; transposed copies
        srand(7); for (auto& b : hp) b = wide ? (rand() & 0xFF) : (rand() & 0x77);
        for (auto& b : hs) { b = wide ? (rand() % 0x7F) : 0x30 + (rand() & 0x0F); if (wide && (rand() & 1)) b |= 0x80; }
        for (size_t n = 0; n < N; n++) {
            for (size_t kp = 0; kp < K / 2; kp++) tp[kp * N + n] = hp[n * (K / 2) + kp];
            for (size_t g = 0; g < K / 16; g++) ts[g * N + n] = hs[n * (K / 16) + g];
        }
        std::vector<unsigned char*> P(copies), Q(copies), TPc(copies), TSc(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&TPc[c], pb)); CHECK(hipMemcpy(TPc[c], tp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&TSc[c], sb)); CHECK(hipMemcpy(TSc[c], ts.data(), sb, hipMemcpyHostToDevice)); }
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], pb)); CHECK(hipMemcpy(P[c], hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&Q[c], sb)); CHECK(hipMemcpy(Q[c], hs.data(), sb, hipMemcpyHostToDevice)); }
        unsigned char *TP, *TS; CHECK(hipMalloc(&TP, pb)); CHECK(hipMemcpy(TP, tp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&TS, sb)); CHECK(hipMemcpy(TS, ts.data(), sb, hipMemcpyHostToDevice));
        std::vector<__nv_bfloat16> hA((size_t)M * K); for (auto& a : hA) a = __float2bfloat16(wide ? ((rand() % 2001) - 1000) / 1000.f * std::pow(2.f, (float)(rand() % 24 - 12)) : ((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *A, *C0, *C1; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        // reference: m128_f16a on the transposed copy
        hipLaunchKernelGGL(w4a16_gemm_t_m128_f16a, dim3((N + 127) / 128, (M + 127) / 128), dim3(128), 0, 0, A, TP, TS, 0.01f, C0, M, N, K);
        CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost));
        auto launch = [&](int v, int c, __nv_bfloat16* C) {
            if (v == 0) hipLaunchKernelGGL(w4a16_gemm_m32_native<128>, dim3((N + 127) / 128, (M + 31) / 32), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
            else if (v == 1) hipLaunchKernelGGL(w4a16_gemm_m32_native<64>, dim3((N + 63) / 64, (M + 31) / 32), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
            else hipLaunchKernelGGL(w4a16_gemm_t_m128_f16a, dim3((N + 127) / 128, (M + 127) / 128), dim3(128), 0, 0, A, TPc[c], TSc[c], 0.01f, C, M, N, K);
        };
        // v2: m128_f16a on the transposed layout (what the drafter would run if its weights were transposed in place at
        // load, like the main model's). batch16 baselines (KLOOP #16/#70): q|gate 993, o 559, gate/up 1884, down 1263, fc 1305, lm_head 4571 us.
        const char* names[] = {"m32n NT128", "m32n NT64", "m128t"};
        printf("-- %s M=%u N=%u K=%u wide=%d (%.1f MB)\n", s.name, M, N, K, (int)wide, wbytes / 1e6);
        for (int v = 0; v < 3; v++) {
            for (int i = 0; i < 3; i++) launch(v, i % copies, C1);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(v, i % copies, C1); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            CHECK(hipMemset(C1, 0xFF, (size_t)M * N * 2));
            launch(v, 0, C1); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
            size_t neq = 0; for (size_t i = 0; i < r0.size(); i++) neq += memcmp(&r0[i], &r1[i], 2) != 0 && !(std::isnan(bf(r0[i])) && std::isnan(bf(r1[i])));
            printf("   %-10s %8.1f us  %6.1f GB/s  (%zu/%zu differ from m128_f16a on the transposed copy)\n", names[v], 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000), neq, r0.size());
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); hipFree(TPc[c]); hipFree(TSc[c]); }
        hipFree(TP); hipFree(TS); hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
