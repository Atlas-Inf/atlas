// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #171 (Flash-Next C4-C8): kbench_hc5 (43859265376) put hc_pre_down at 49 / 48 / 56 us for T = 1 / 2 / 4 but the
// T >= 5 path, hc_pre_down_tiled, at 159 us (T=5) and ~205 us in situ at T=8 (11.4% of C8 decode). The HIP cutover to
// tiled at T >= 5 came from kbench 42932386097 (tiled 189 vs pre_down 319 us at T=8), measured BEFORE #373 made
// hc_pre_down 2.7x faster. Both kernels are bit-identical to each other (each (t, r) is one lane-strided warp walk +
// the same shuffle tree), so this re-measures the cutover: hc_pre_down at T = 5..8 over its row-split grid.
#include "hyper_connection.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HCD = H * HC; const int NW = 8;
    std::vector<__nv_bfloat16> hw((size_t)RANK * HCD); srand(3); for (auto& x : hw) x = __float2bfloat16(((rand() % 2001) - 1000) / 8000.f);
    __nv_bfloat16* dw[NW]; for (int i = 0; i < NW; i++) { CHECK(hipMalloc(&dw[i], hw.size() * 2)); CHECK(hipMemcpy(dw[i], hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    const unsigned splits[] = {0, 4, 6, 8, 10, 16};   // 0 = host default clamp(48 / T, 1, 10)
    const int NS = 6;
    for (unsigned T : {4u, 5u, 6u, 8u}) {
        std::vector<float> hn((size_t)T * HCD); for (auto& x : hn) x = ((rand() % 2001) - 1000) / 1000.f;
        float* dn; CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        float* dout[NS + 1]; for (auto& p : dout) CHECK(hipMalloc(&p, (size_t)T * RANK * 4));
        const unsigned dsplit_t = std::min(16u, RANK / 32u) * 2u;
        auto launch = [&](int v, int i) {
            if (v == 0) hipLaunchKernelGGL(hc_pre_down_tiled, dim3(1, dsplit_t), dim3(1024), 0, 0, dn, dw[i], dout[0], H, HC, RANK, T);
            else { const unsigned s = splits[v - 1] ? splits[v - 1] : std::min(10u, std::max(1u, 48u / T));
                   hipLaunchKernelGGL(hc_pre_down, dim3(T, s), dim3(1024), HCD * 4, 0, dn, dw[i], dout[v], H, HC, RANK, T); } };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[NS + 1];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v <= NS; v++) { for (int i = 0; i < 3; i++) launch(v, i % NW); CHECK(hipEventRecord(e0)); for (int i = 0; i < 40; i++) launch(v, i % NW); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 40; }
        for (int v = 0; v <= NS; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> o0((size_t)T * RANK), o1((size_t)T * RANK); CHECK(hipMemcpy(o0.data(), dout[0], o0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- T=%u: tiled 1024x%u %6.1f us", T, dsplit_t, t[0]);
        for (int v = 1; v <= NS; v++) { CHECK(hipMemcpy(o1.data(), dout[v], o1.size() * 4, hipMemcpyDeviceToHost));
            const unsigned s = splits[v - 1] ? splits[v - 1] : std::min(10u, std::max(1u, 48u / T));
            printf(" | pre_down %ux%u%s %6.1f x%.2f %s", T, s, splits[v - 1] ? "" : "(dflt)", t[v], t[0] / t[v], memcmp(o0.data(), o1.data(), o0.size() * 4) ? "DIFF" : "same"); }
        printf("\n");
        CHECK(hipFree(dn)); for (auto p : dout) CHECK(hipFree(p));
    }
    return 0;
}
