// SPDX-License-Identifier: AGPL-3.0-only
// kbench: which existing transposed NVFP4 GEMM to use for DFlash-verify rows (M = K+1) on gfx1151.
// dense_ffn::w4a16_prefill_gemm picks w4a16_gemm_t (M64 tile) / w4a16_gemm_t_k64 for M <= 64 from GB10
// measurements (w4a16_m17_bench). Times t / k64 / m128 at the Ripple-65B and Qwen3.8-27B verify FFN shapes.
// Each kernel is exact math with its own K order; reports relL2 vs w4a16_gemm_t.
#include "../q27/w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
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
        __nv_bfloat16 *A, *C0, *C1; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto launch = [&](int which, int c, __nv_bfloat16* C) {
            if (which == 0) hipLaunchKernelGGL(w4a16_gemm_t, dim3((N + 127) / 128, (M + 63) / 64), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K, N);
            else if (which == 1) hipLaunchKernelGGL(w4a16_gemm_t_k64, dim3((N + 127) / 128, (M + 63) / 64), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
            else hipLaunchKernelGGL(w4a16_gemm_t_m128, dim3((N + 127) / 128, (M + 127) / 128), dim3(128), 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
        };
        const char* names[] = {"t (M64)", "t_k64", "t_m128"};
        printf("-- %s M=%u N=%u K=%u (%.1f MB)\n", s.name, M, N, K, wbytes / 1e6);
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        launch(0, 0, C0); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost));
        for (int w = 0; w < 3; w++) {
            if (w == 1 && K % 64) continue;
            for (int i = 0; i < 3; i++) launch(w, i % copies, C1);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(w, i % copies, C1); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            launch(w, 0, C1); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
            double n = 0, d = 0; for (size_t i = 0; i < r0.size(); i++) { double x = bf(r1[i]) - bf(r0[i]); n += x * x; d += (double)bf(r0[i]) * bf(r0[i]); }
            printf("   %-10s %8.1f us  %6.1f GB/s  relL2 vs t %.2e\n", names[w], 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000), std::sqrt(n / (d + 1e-30)));
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); }
        hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
