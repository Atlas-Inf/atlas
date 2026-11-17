// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #166 (Flash-Next C8, FP8 GDN =full): the batched GDN projections go through exact_rows -> w8a16_gemv_batch16
// for 5..16 rows. At M=8 in situ (43841189468) qkvz 16384x2560 takes 352 us (~120 GB/s) and out_proj 307 us, about
// half of what the BF16 batchm reached at the same N. batch16 carries a 16-row accumulator template for 8 live rows.
// The batch8 / batch8_dyn_vl2 / vl2_n2 tiers are the same template (per-row k16 walk + reduction unchanged), so their
// rows must be BYTE-IDENTICAL to batch16's. Which is fastest at M = 5..8 on the GDN shapes?
#include "w8a16_gemv_batch4.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    struct S { unsigned N, K; const char* nm; } shapes[] = {{5120, 6144, "27B 5120x6144"}, {16384, 5120, "27B 16384x5120"}, {12288, 5120, "27B 12288x5120"}, {1024, 5120, "27B 1024x5120"}, {5120, 4096, "27B 5120x4096"}};
    for (auto& sh : shapes) for (unsigned M : {8u}) {
        const unsigned N = sh.N, K = sh.K; const int NB = 6;
        const size_t nsc = (size_t)((N + 127) / 128) * ((K + 127) / 128);
        std::vector<unsigned char> hb((size_t)N * K); std::vector<float> hs(nsc); std::vector<__nv_bfloat16> ha((size_t)M * K); srand(N + K + M);
        for (auto& x : hb) { x = rand() & 0xFF; if ((x & 0x7F) == 0x7F) x = 0x3C; } for (auto& x : hs) x = 0.001f * (1 + rand() % 50);
        for (auto& x : ha) x = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        unsigned char* dB[NB]; float* dS; __nv_bfloat16 *dA, *dC[4];
        for (int i = 0; i < NB; i++) { CHECK(hipMalloc(&dB[i], hb.size())); CHECK(hipMemcpy(dB[i], hb.data(), hb.size(), hipMemcpyHostToDevice)); }
        CHECK(hipMalloc(&dS, nsc * 4)); CHECK(hipMemcpy(dS, hs.data(), nsc * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (int v = 0; v < 4; v++) CHECK(hipMalloc(&dC[v], (size_t)M * N * 2));
        auto launch = [&](int v, int i) {
            if (v == 0) hipLaunchKernelGGL(w8a16_gemv_batch16, dim3((N + 3) / 4), dim3(256), 0, 0, dA, dB[i], dS, dC[0], M, N, K);
            else if (v == 1) hipLaunchKernelGGL(w8a16_gemv_batch8, dim3((N + 3) / 4), dim3(256), 0, 0, dA, dB[i], dS, dC[1], M, N, K);
            else if (v == 2) hipLaunchKernelGGL(w8a16_gemv_batch8_dyn_vl2, dim3((N + 7) / 8), dim3(256), 0, 0, dA, dB[i], dS, dC[2], M, N, K);
            else hipLaunchKernelGGL(w8a16_gemv_batch8_dyn_vl2_n2, dim3((N + 15) / 16), dim3(256), 0, 0, dA, dB[i], dS, dC[3], M, N, K); };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[4];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 4; v++) { for (int i = 0; i < 3; i++) launch(v, i % NB); CHECK(hipEventRecord(e0)); for (int i = 0; i < 30; i++) launch(v, i % NB); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 30; }
        for (int v = 0; v < 4; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> c0((size_t)M * N), c1((size_t)M * N); CHECK(hipMemcpy(c0.data(), dC[0], c0.size() * 2, hipMemcpyDeviceToHost));
        const double gb = (double)N * K / 1e3;
        printf("-- %-9s N=%5u K=%u M=%u: batch16 %6.1f us (%3.0f GB/s)", sh.nm, N, K, M, t[0], gb / t[0]);
        const char* nm[4] = {"", "batch8", "vl2", "vl2_n2"};
        for (int v = 1; v < 4; v++) { CHECK(hipMemcpy(c1.data(), dC[v], c1.size() * 2, hipMemcpyDeviceToHost));
            printf(" | %s %6.1f x%.2f %s", nm[v], t[v], t[0] / t[v], memcmp(c0.data(), c1.data(), c0.size() * 2) ? "DIFF" : "same"); }
        printf("\n");
        for (int i = 0; i < NB; i++) CHECK(hipFree(dB[i])); CHECK(hipFree(dS)); CHECK(hipFree(dA)); for (int v = 0; v < 4; v++) CHECK(hipFree(dC[v]));
    }
    return 0;
}
