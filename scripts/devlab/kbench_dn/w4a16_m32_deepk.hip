// SPDX-License-Identifier: AGPL-3.0-only
// kbench (KLOOP #75, Ripple DFlash2 verify): rocprof on m32 main (DevLab 43402666346) puts w4a16_gemm_t_m32_f16a at
// 49% of decode GPU time. The n128 launches (gate/up 40960x5120) run at ~170 GB/s, but the n64 launches (down
// 5120x40960) take 1277 us = ~92 GB/s: 80 CTAs, each walking K=40960 in 1280 serial 32-K steps with two barriers per
// step and one 16-byte B load per thread in flight. Candidate deepk<NT,KS>: the same m32 body, but each barrier
// interval stages KS consecutive 32-K steps (KS x the LDS, KS loads in flight per thread) and then runs the KS
// computes in ascending k. Every 16x16 tile sees the same WMMA operands in the same (k, h) order, so it must be
// BIT-IDENTICAL to m32 (KS=1, the #354 kernel reproduced here) and to m128_f16a.
#include "../q27/w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
static inline float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define DK_B_OFF(n) ((n) * 40 + ((n) >> 4) * 8)
template <int NT, int KS, int RQ = 1>
__global__ __launch_bounds__(128)
void m32_deepk(const __nv_bfloat16* __restrict__ A_, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    constexpr int NBW = NT / 64;
    constexpr int TPR = NT / 16;
    const unsigned short* A = (const unsigned short*)A_;
    const unsigned int cta_n = blockIdx.x * NT;
    const unsigned int cta_m = blockIdx.y * 32;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;

    __shared__ __align__(16) _Float16 sA[KS][32][F16_K_STEP + F16_PAD_T];
    __shared__ __align__(16) _Float16 sB[KS][DK_B_OFF(NT)];

    v8f acc[2][NBW];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int j = 0; j < NBW; j++) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int a_row = threadIdx.x >> 2;
    const unsigned int a_col = (threadIdx.x & 3) << 3;
    const bool b_on = threadIdx.x < 16 * TPR;
    const unsigned int b_kp = threadIdx.x / TPR;
    const unsigned int b_ns = (threadIdx.x % TPR) << 4;

    auto load = [&](unsigned int kb, uint4& ra, uint4& rb, uint4& rs) {
        const unsigned int gr = cta_m + a_row, gc = kb + a_col;
        ra = (gr < M && gc + 7 < K) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        rb = rs = uint4{0, 0, 0, 0};
        if (b_on && kb < K) {
            const unsigned int gke = kb + (b_kp << 1), gns = cta_n + b_ns;
            if (gke + 1 <= K && gns + 15 < N) rb = *(const uint4*)&B_packed[(unsigned long long)(gke >> 1) * N + gns];
            const unsigned int sg = kb / F16_GROUP + (b_kp >> 3);
            if (gns + 15 < N) rs = *(const uint4*)&B_scale[(unsigned long long)sg * N + gns];
        }
    };
    auto store = [&](int s, const uint4& ra, const uint4& rb, const uint4& rs) {
        *(uint4*)&sA[s][a_row][a_col] = bf16x8_to_f16x8(ra);
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
                const _Float16 sc = (j == 0) ? s01.x : (j == 1) ? s01.y : (j == 2) ? s23.x : s23.y;
                w = w * (v2h){sc, sc};
                *(v2h*)&sB[s][DK_B_OFF(b_ns + d * 4 + j) + b_kp * 2] = w;
            }
        }
    };
    auto compute = [&](int s) {
        const unsigned int r0 = lane_id & 15;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            v16h a0, a1;
            __builtin_memcpy(&a0, &sA[s][r0][h * 16], 32);
            __builtin_memcpy(&a1, &sA[s][16 + r0][h * 16], 32);
            #pragma unroll
            for (int j = 0; j < NBW; j++) {
                const unsigned int nc = (warp_id * NBW + j) * 16 + (lane_id & 15);
                v16h b;
                __builtin_memcpy(&b, &sB[s][DK_B_OFF(nc) + h * 16], 32);
                acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0][j]);
                acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1][j]);
            }
        }
    };

    if constexpr (RQ > 1) {
        // KLOOP #79 register queue: RQ-1 K-steps in flight in VGPRs, one LDS stage, same compute order.
        uint4 qa[RQ], qb[RQ], qs[RQ];
        load(0, qa[0], qb[0], qs[0]);
        store(0, qa[0], qb[0], qs[0]);
        #pragma unroll
        for (int i = 1; i < RQ; i++) load(i * F16_K_STEP, qa[i], qb[i], qs[i]);
        __syncthreads();
        for (unsigned int k_base = F16_K_STEP; k_base < K; k_base += F16_K_STEP) {
            load(k_base + (RQ - 1) * F16_K_STEP, qa[0], qb[0], qs[0]);
            compute(0);
            __syncthreads();
            store(0, qa[1], qb[1], qs[1]);
            __syncthreads();
            #pragma unroll
            for (int i = 1; i < RQ; i++) { const uint4 ta = qa[i], tb = qb[i], ts = qs[i]; qa[i] = qa[(i + 1) % RQ]; qb[i] = qb[(i + 1) % RQ]; qs[i] = qs[(i + 1) % RQ]; qa[(i + 1) % RQ] = ta; qb[(i + 1) % RQ] = tb; qs[(i + 1) % RQ] = ts; }
        }
        compute(0);
    } else {
    uint4 ra[KS], rb[KS], rs[KS];
    #pragma unroll
    for (int s = 0; s < KS; s++) load(s * F16_K_STEP, ra[s], rb[s], rs[s]);
    #pragma unroll
    for (int s = 0; s < KS; s++) store(s, ra[s], rb[s], rs[s]);
    __syncthreads();
    unsigned int k_cur = 0;
    for (unsigned int k_base = KS * F16_K_STEP; k_base < K; k_base += KS * F16_K_STEP) {
        #pragma unroll
        for (int s = 0; s < KS; s++) load(k_base + s * F16_K_STEP, ra[s], rb[s], rs[s]);
        #pragma unroll
        for (int s = 0; s < KS; s++) compute(s);
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < KS; s++) store(s, ra[s], rb[s], rs[s]);
        __syncthreads();
        k_cur = k_base;
    }
    #pragma unroll
    for (int s = 0; s < KS; s++)
        if (k_cur + s * F16_K_STEP < K) compute(s);
    }

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
    struct S { const char* name; unsigned M, N, K; } shapes[] = {
        {"down M17", 17, 5120, 40960}, {"o M17", 17, 5120, 10240}, {"gate/up M17", 17, 40960, 5120},
        {"q|gate M17", 17, 20480, 5120}, {"drafter down M16", 16, 5120, 40960}, {"fc M16", 16, 5120, 25600},
        {"down M9", 9, 5120, 40960}, {"odd K=5120+96 M17", 17, 5120, 5216}};
    struct V { const char* name; int nt; void (*f)(const __nv_bfloat16*, const unsigned char*, const unsigned char*, float, __nv_bfloat16*, unsigned, unsigned, unsigned); };
    V vs[] = {{"m32 n64 (KS1)", 64, m32_deepk<64, 1>}, {"deepk n64 KS2", 64, m32_deepk<64, 2>},
              {"deepk n64 KS4", 64, m32_deepk<64, 4>}, {"deepk n64 KS8", 64, m32_deepk<64, 8>},
              {"m32 n128 (KS1)", 128, m32_deepk<128, 1>}, {"deepk n128 KS2", 128, m32_deepk<128, 2>},
              {"deepk n128 KS4", 128, m32_deepk<128, 4>}, {"rq n128 RQ2", 128, m32_deepk<128, 1, 2>},
              {"rq n128 RQ3", 128, m32_deepk<128, 1, 3>}, {"rq n64 RQ2", 64, m32_deepk<64, 1, 2>},
              {"rq n64 RQ3", 64, m32_deepk<64, 1, 3>}};
    for (auto& s : shapes) {
        const unsigned M = s.M, N = s.N, K = s.K;
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N, wbytes = pb + sb;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char> hp(pb), hs(sb);
        srand(11); for (auto& b : hp) b = rand() & 0xFF;
        for (auto& b : hs) { b = rand() % 0x7F; if (rand() & 1) b |= 0x80; }
        std::vector<unsigned char*> P(copies), Q(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], pb)); CHECK(hipMemcpy(P[c], hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&Q[c], sb)); CHECK(hipMemcpy(Q[c], hs.data(), sb, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> hA((size_t)M * K); for (auto& a : hA) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f * std::pow(2.f, (float)(rand() % 16 - 8)));
        __nv_bfloat16 *A, *C0, *C1; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        hipLaunchKernelGGL(w4a16_gemm_t_m128_f16a, dim3((N + 127) / 128, (M + 127) / 128), dim3(128), 0, 0, A, P[0], Q[0], 0.01f, C0, M, N, K);
        CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost));
        printf("-- %s M=%u N=%u K=%u (%.1f MB)\n", s.name, M, N, K, wbytes / 1e6);
        for (auto& v : vs) {
            auto launch = [&](int c) { hipLaunchKernelGGL(v.f, dim3((N + v.nt - 1) / v.nt, (M + 31) / 32), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C1, M, N, K); };
            for (int i = 0; i < 3; i++) launch(i % copies);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            CHECK(hipMemset(C1, 0xFF, (size_t)M * N * 2));
            launch(0); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
            size_t neq = 0; for (size_t i = 0; i < r0.size(); i++) neq += memcmp(&r0[i], &r1[i], 2) != 0 && !(std::isnan(bf(r0[i])) && std::isnan(bf(r1[i])));
            printf("   %-15s %8.1f us  %6.1f GB/s  (%zu/%zu differ from m128_f16a)\n", v.name, 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000), neq, r0.size());
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); }
        hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
