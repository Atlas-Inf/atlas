// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #140 (G3 Flash-Next C1 decode): w8a16_gemv_batch4 is 14.9% of single-user decode GPU time after #373/#374
// (rocprof 43709734119; FP8 GDN / attention projections at MTP verify widths M=1..3). Is it at the stream ceiling?
// Reports us and weight GB/s at the Flash-Next FP8 shapes, DRAM-rotated.
#include "w8a16_gemv_batch4.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
int main() {
    struct S { const char* nm; unsigned N, K; } shapes[] = {{"gdn qkvz", 16384, 2560}, {"gdn out", 2560, 6144}, {"attn q", 12288, 2560}, {"attn kv", 1024, 2560}, {"attn o", 2560, 8192}};
    for (auto& s : shapes) for (unsigned M : {1u, 2u, 3u, 4u}) {
        const size_t wb = (size_t)s.N * s.K, sb = (size_t)((s.N + 127) / 128) * ((s.K + 127) / 128);
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wb + 1);
        std::vector<unsigned char> hb(wb); std::vector<float> hs(sb); srand(1);
        for (auto& b : hb) { b = rand() & 0xFF; if ((b & 0x7F) == 0x7F) b ^= 1; } for (auto& x : hs) x = 0.01f;
        std::vector<unsigned char*> B(copies); std::vector<float*> SC(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&B[c], wb)); CHECK(hipMemcpy(B[c], hb.data(), wb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&SC[c], sb * 4)); CHECK(hipMemcpy(SC[c], hs.data(), sb * 4, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> ha((size_t)M * s.K, __float2bfloat16(0.5f)); __nv_bfloat16 *A, *C;
        CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice)); CHECK(hipMalloc(&C, (size_t)M * s.N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto f = [&](int c) { hipLaunchKernelGGL(w8a16_gemv_batch4, dim3((s.N + 3) / 4), dim3(256), 0, 0, A, B[c], SC[c], C, M, s.N, s.K); };
        for (int i = 0; i < 4; i++) f(i % copies); CHECK(hipEventRecord(e0)); for (int i = 0; i < 40; i++) f(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); const float us = 1000.f * ms / 40;
        printf("-- %-9s N=%5u K=%5u M=%u: %7.1f us  %6.1f GB/s\n", s.nm, s.N, s.K, M, us, wb / 1e3 / us);
        for (int c = 0; c < copies; c++) { hipFree(B[c]); hipFree(SC[c]); } hipFree(A); hipFree(C);
    }
    return 0;
}
