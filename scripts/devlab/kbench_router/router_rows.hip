// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #203 (Flash-Next C8): forward_rows_t still runs the MoE router per row (w4a16_gemv_sw x ~196k at C8, NVFP4
// gate 512 x 2560). forward_rows batches it through the w4a16 batch template, but on HIP that template uses v_dot2,
// whose 2-term grouping may not equal w4a16_gemv_sw's. Question 1: are batch2/4/8 rows BYTE-EQUAL to per-row
// w4a16_gemv_sw (and to w4a16_gemv)? Question 2: how much does one batched launch save over M per-row launches?
#include "w4a16_gemv.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    const unsigned N = 512, K = 2560; const size_t pb = (size_t)N * K / 2, sb = (size_t)N * K / 16;
    std::vector<unsigned char> hp(pb), hs(sb); srand(11); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    unsigned char *dp, *ds; CHECK(hipMalloc(&dp, pb)); CHECK(hipMalloc(&ds, sb)); CHECK(hipMemcpy(dp, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMemcpy(ds, hs.data(), sb, hipMemcpyHostToDevice));
    for (unsigned M : {2u, 4u, 8u}) {
        std::vector<__nv_bfloat16> ha((size_t)M * K); for (auto& x : ha) x = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *dA, *cs, *cg, *cb; CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (auto* p : {&cs, &cg, &cb}) CHECK(hipMalloc(p, (size_t)M * N * 2));
        auto per_row = [&](bool sw) { for (unsigned r = 0; r < M; r++) {
            if (sw) hipLaunchKernelGGL(w4a16_gemv_sw, dim3(N / N_PER_BLOCK_SW), dim3(256), 0, 0, dA + (size_t)r * K, dp, ds, 0.01f, cs + (size_t)r * N, N, K);
            else hipLaunchKernelGGL(w4a16_gemv, dim3(N / N_PER_BLOCK), dim3(256), 0, 0, dA + (size_t)r * K, dp, ds, 0.01f, cg + (size_t)r * N, N, K); } };
        auto batched = [&]() {
            if (M == 2) hipLaunchKernelGGL(w4a16_gemv_batch2, dim3(N / 4), dim3(256), 0, 0, dA, dp, ds, 0.01f, cb, N, K);
            else if (M == 4) hipLaunchKernelGGL(w4a16_gemv_batch4, dim3(N / 4), dim3(256), 0, 0, dA, dp, ds, 0.01f, cb, M, N, K);
            else hipLaunchKernelGGL(w4a16_gemv_batch8, dim3(N / 4), dim3(256), 0, 0, dA, dp, ds, 0.01f, cb, M, N, K); };
        per_row(true); per_row(false); batched(); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> s((size_t)M * N), g((size_t)M * N), b((size_t)M * N);
        CHECK(hipMemcpy(s.data(), cs, s.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(g.data(), cg, g.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), cb, b.size() * 2, hipMemcpyDeviceToHost));
        size_t dsb = 0, dgb = 0, dsg = 0; for (size_t i = 0; i < s.size(); i++) { dsb += memcmp(&s[i], &b[i], 2) != 0; dgb += memcmp(&g[i], &b[i], 2) != 0; dsg += memcmp(&s[i], &g[i], 2) != 0; }
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[2];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 2; v++) { CHECK(hipEventRecord(e0)); for (int i = 0; i < 100; i++) { if (v == 0) per_row(true); else batched(); } CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 100; }
        printf("-- router N=%u K=%u M=%u: per-row sw %6.1f us | batch%u %6.1f us x%.2f | values differing: sw vs batch %zu, gemv vs batch %zu, sw vs gemv %zu (of %zu)\n",
               N, K, M, t[0], M, t[1], t[0] / t[1], dsb, dgb, dsg, s.size());
        CHECK(hipFree(dA)); CHECK(hipFree(cs)); CHECK(hipFree(cg)); CHECK(hipFree(cb));
    }
    return 0;
}
