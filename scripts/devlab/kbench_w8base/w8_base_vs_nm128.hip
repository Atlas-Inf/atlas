// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #112: is the base w8a16_gemm byte-identical to w8a16_gemm_n_m128 (and its small-M twins) at the batched-GDN
// row counts (n 17..72)? Decides whether perf/hip-ssm-batched-fp8-nm128 needs an accuracy gate. 27B GDN QKVZ
// 16384x5120 (12288?) and out_proj 5120x6144, block_scale [N/128, K/128].
#include "w8a16_gemm.cu"
#include "w8a16_gemm_n_m128.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
int main() {
    struct S { unsigned N, K; } shapes[] = {{16384, 5120}, {5120, 6144}};
    for (auto& s : shapes) for (unsigned M : {17u, 36u, 64u, 72u}) {
        const unsigned N = s.N, K = s.K; const size_t wb = (size_t)N * K, sb = (size_t)(N / 128) * (K / 128);
        std::vector<unsigned char> hb(wb); std::vector<float> hs(sb);
        srand(3); for (auto& b : hb) { b = rand() & 0xFF; if ((b & 0x7F) == 0x7F) b ^= 1; }
        for (auto& x : hs) x = ldexpf(1.0f + (rand() % 1000) / 1000.f, -(rand() % 12) - 4);
        std::vector<__nv_bfloat16> ha((size_t)M * K); for (auto& a : ha) a = __float2bfloat16(((rand() % 2001) - 1000) / 500.f);
        unsigned char* B; float* SC; __nv_bfloat16 *A, *C0, *C1;
        CHECK(hipMalloc(&B, wb)); CHECK(hipMemcpy(B, hb.data(), wb, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&SC, sb * 4)); CHECK(hipMemcpy(SC, hs.data(), sb * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto base = [&]() { hipLaunchKernelGGL(w8a16_gemm, dim3((N + 127) / 128, (M + 255) / 256), dim3(512), 0, 0, A, B, SC, C0, M, N, K); };
        auto nm = [&]() {
            if (M <= 32) hipLaunchKernelGGL(w8a16_gemm_n_m32, dim3((N + 63) / 64, (M + 31) / 32), dim3(128), 0, 0, A, B, SC, C1, M, N, K);
            else if (M <= 64) hipLaunchKernelGGL(w8a16_gemm_n_m64, dim3((N + 63) / 64, (M + 63) / 64), dim3(128), 0, 0, A, B, SC, C1, M, N, K);
            else hipLaunchKernelGGL(w8a16_gemm_n_m128, dim3((N + 127) / 128, (M + 127) / 128), dim3(256), 0, 0, A, B, SC, C1, M, N, K); };
        auto t = [&](auto f) { for (int i = 0; i < 3; i++) f(); CHECK(hipEventRecord(e0)); for (int i = 0; i < 20; i++) f(); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / 20; };
        const float tb = t(base), tn = t(nm); base(); nm(); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> r0((size_t)M * N), r1((size_t)M * N);
        CHECK(hipMemcpy(r0.data(), C0, r0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(r1.data(), C1, r1.size() * 2, hipMemcpyDeviceToHost));
        size_t neq = 0; double mx = 0, rr = 0; for (size_t i = 0; i < r0.size(); i++) { neq += memcmp(&r0[i], &r1[i], 2) != 0;
            double a = __bfloat162float(r0[i]), b = __bfloat162float(r1[i]); mx = std::max(mx, fabs(a - b)); rr += a * a; }
        printf("-- N=%u K=%u M=%u | base w8a16_gemm %8.1f us | n_m128 route %8.1f us | x%.2f | %zu/%zu differ, max|d| %.3e (rms %.3e)\n", N, K, M, tb, tn, tb / tn, neq, r0.size(), mx, sqrt(rr / r0.size()));
        hipFree(B); hipFree(SC); hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
