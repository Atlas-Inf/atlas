// SPDX-License-Identifier: AGPL-3.0-only
// kbench (KLOOP #12): dequant-once row-GEMV for DFlash verify FFN (M = K+1 = 9 / 17 rows) on gfx1151.
// #11 showed w4a16_gemm_t_m128 is the best shipped kernel at these shapes (~58 GB/s of ~216) and that the
// small-M tile GEMMs are dequant/compute bound, not grid starved. m128 pads M to 128, so 7/8 of its WMMA work
// is on zero rows. Candidate: each lane owns 4 output columns, dequantizes each NVFP4 weight ONCE (same
// e2m1x4_off12 / e4m3_scale_half math as m128 -> identical bf16 weights) and applies it to all M rows with
// explicit fmaf; A is read with wave-uniform addresses (scalar loads). 4 waves split K inside the block (fixed
// LDS reduction order); KS > 1 splits K across blocks into an fp32 workspace + an ordered reduce kernel.
// Not bit-identical to the WMMA kernels (different fp32 summation order) — reports relL2 vs t_m128.
#include "../q27/w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define RG_KW 4
template <int MR, bool DOT2>
__global__ __launch_bounds__(32 * RG_KW) void rowgemv(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ Bp, const unsigned char* __restrict__ Bs,
    const float scale2, __nv_bfloat16* __restrict__ C, float* __restrict__ ws, unsigned int N, unsigned int K, unsigned int KS) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const unsigned int n0 = blockIdx.x * 128 + lane * 4;
    const unsigned int slice = (K / 2) / (RG_KW * KS);                 // K-pairs per wave, multiple of 8
    const unsigned int kp0 = __builtin_amdgcn_readfirstlane((blockIdx.y * RG_KW + w) * slice);
    const unsigned int* A32 = (const unsigned int*)A;                   // bf16 pairs (k, k+1)
    float acc[MR][4];
    #pragma unroll
    for (int m = 0; m < MR; m++) { acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f; }
    float svh[4];
    for (unsigned int kp = kp0; kp < kp0 + slice; kp++) {
        if ((kp & 7) == 0) {
            const unsigned int sb = *(const unsigned int*)&Bs[(unsigned long long)(kp >> 3) * N + n0];
            #pragma unroll
            for (int c = 0; c < 4; c++) svh[c] = e4m3_scale_half((sb >> (8 * c)) & 0xFF, scale2);
        }
        const unsigned int pw = *(const unsigned int*)&Bp[(unsigned long long)kp * N + n0];
        const unsigned int tl = e2m1x4_off12(pw & 0x0F0F0F0Fu), th = e2m1x4_off12((pw >> 4) & 0x0F0F0F0Fu);
        float wl[4], wh[4];
        unsigned int w2[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {   // bf16-round the weight exactly as m128's smem tile does
            wl[c] = __bfloat162float(__float2bfloat16((float)(int)(((tl >> (8 * c)) & 0xFF) - 12) * svh[c]));
            wh[c] = __bfloat162float(__float2bfloat16((float)(int)(((th >> (8 * c)) & 0xFF) - 12) * svh[c]));
            w2[c] = (__float_as_uint(wl[c]) >> 16) | (__float_as_uint(wh[c]) & 0xFFFF0000u);
        }
        #pragma unroll
        for (int m = 0; m < MR; m++) {
            const unsigned int ap = A32[((unsigned long long)m * K >> 1) + kp];
            if (DOT2) {   // v_dot2_f32_bf16: a.lo*w.lo + a.hi*w.hi + acc in one VALU op
                #pragma unroll
                for (int c = 0; c < 4; c++) asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc[m][c]) : "v"(ap), "v"(w2[c]));
            } else {
                const float al = __uint_as_float(ap << 16), ah = __uint_as_float(ap & 0xFFFF0000u);
                #pragma unroll
                for (int c = 0; c < 4; c++) acc[m][c] = fmaf(ah, wh[c], fmaf(al, wl[c], acc[m][c]));
            }
        }
    }
    __shared__ float red[RG_KW - 1][MR][128];
    if (w > 0) {
        #pragma unroll
        for (int m = 0; m < MR; m++)
            #pragma unroll
            for (int c = 0; c < 4; c++) red[w - 1][m][lane * 4 + c] = acc[m][c];
    }
    __syncthreads();
    if (w != 0) return;
    #pragma unroll
    for (int m = 0; m < MR; m++)
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float s = acc[m][c];
            #pragma unroll
            for (int j = 0; j < RG_KW - 1; j++) s += red[j][m][lane * 4 + c];
            if (KS == 1) C[(unsigned long long)m * N + n0 + c] = __float2bfloat16(s);
            else ws[((unsigned long long)blockIdx.y * MR + m) * N + n0 + c] = s;
        }
}
__global__ void rowgemv_reduce(const float* __restrict__ ws, __nv_bfloat16* __restrict__ C, unsigned int MN, unsigned int KS) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float s = 0.f;
    for (unsigned int k = 0; k < KS; k++) s += ws[(unsigned long long)k * MN + i];
    C[i] = __float2bfloat16(s);
}

static float bf(__nv_bfloat16 x) { unsigned short u; std::memcpy(&u, &x, 2); unsigned int v = (unsigned int)u << 16; float f; std::memcpy(&f, &v, 4); return f; }
int main() {
    struct S { const char* name; unsigned int M, N, K; } shapes[] = {
        {"ripple gate/up", 17, 40960, 5120}, {"ripple down", 17, 5120, 40960},
        {"27b gate/up K8", 9, 17408, 5120}, {"27b down K8", 9, 5120, 17408}};
    for (auto& s : shapes) {
        const unsigned M = s.M, N = s.N, K = s.K;
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N, wbytes = pb + sb;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char> hp(pb), hs(sb);
        srand(7); for (auto& b : hp) b = rand() & 0x77; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
        std::vector<unsigned char*> P(copies), Q(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], pb)); CHECK(hipMemcpy(P[c], hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&Q[c], sb)); CHECK(hipMemcpy(Q[c], hs.data(), sb, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> hA((size_t)M * K); for (auto& a : hA) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *A, *C0, *C1; float* ws; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2)); CHECK(hipMalloc(&ws, (size_t)16 * M * N * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        const unsigned int KSV[] = {0, 1, 2, 4, 8, 17, 18, 20, 24};   // 0 = t_m128 baseline; 16+ks = DOT2
        auto launch = [&](unsigned int ks, int c, __nv_bfloat16* C) {
            if (ks == 0) { hipLaunchKernelGGL(w4a16_gemm_t_m128, dim3((N + 127) / 128, (M + 127) / 128), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K); return; }
            const bool d2 = ks >= 16; ks = d2 ? ks - 16 : ks;
            if (M == 17) { if (d2) hipLaunchKernelGGL((rowgemv<17, true>), dim3(N / 128, ks), dim3(32 * RG_KW), 0, 0, A, P[c], Q[c], 0.01f, C, ws, N, K, ks);
                           else    hipLaunchKernelGGL((rowgemv<17, false>), dim3(N / 128, ks), dim3(32 * RG_KW), 0, 0, A, P[c], Q[c], 0.01f, C, ws, N, K, ks); }
            else         { if (d2) hipLaunchKernelGGL((rowgemv<9, true>), dim3(N / 128, ks), dim3(32 * RG_KW), 0, 0, A, P[c], Q[c], 0.01f, C, ws, N, K, ks);
                           else    hipLaunchKernelGGL((rowgemv<9, false>), dim3(N / 128, ks), dim3(32 * RG_KW), 0, 0, A, P[c], Q[c], 0.01f, C, ws, N, K, ks); }
            if (ks > 1) hipLaunchKernelGGL(rowgemv_reduce, dim3((M * N + 255) / 256), dim3(256), 0, 0, ws, C, M * N, ks);
        };
        printf("-- %s M=%u N=%u K=%u (%.1f MB)\n", s.name, M, N, K, wbytes / 1e6);
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        launch(0, 0, C0); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost));
        for (unsigned int ks : KSV) {
            for (int i = 0; i < 3; i++) launch(ks, i % copies, C1);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(ks, i % copies, C1); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            launch(ks, 0, C1); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
            double n = 0, d = 0; size_t neq = 0; for (size_t i = 0; i < r0.size(); i++) { double x = bf(r1[i]) - bf(r0[i]); n += x * x; d += (double)bf(r0[i]) * bf(r0[i]); neq += (x != 0); }
            char nm[32]; if (ks == 0) snprintf(nm, sizeof nm, "t_m128"); else if (ks >= 16) snprintf(nm, sizeof nm, "dot2 KS=%u", ks - 16); else snprintf(nm, sizeof nm, "rowgemv KS=%u", ks);
            printf("   %-14s %8.1f us  %6.1f GB/s  relL2 vs m128 %.2e  (%zu/%zu differ)\n", nm, 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000), std::sqrt(n / (d + 1e-30)), neq, r0.size());
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); }
        hipFree(A); hipFree(C0); hipFree(C1); hipFree(ws);
    }
    return 0;
}
