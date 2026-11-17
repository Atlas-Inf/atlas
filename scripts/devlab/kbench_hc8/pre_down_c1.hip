// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #197 (Flash-Next C1): hc_pre_down is 8.6% of C1 decode (52.9 us x 40934 at T=2, grid T x 10 of 1024 threads,
// 43969950585 / 43961756148). dsplit = clamp(48 / T, 1, 10) leaves 10-20 blocks on strix's 40 CUs, one rank row per warp.
// Same kernel, other launch shapes (row splits x block size): every (t, r) is still one warp's lane-strided walk + the
// same shuffle tree, so outputs must be BIT-IDENTICAL. More blocks re-stage the 40 KB normed row more often.
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
    struct Sh { unsigned split, block; } sh[] = {{0, 1024}, {20, 512}, {40, 256}, {20, 1024}, {40, 512}, {80, 128}};
    const int NS = 6;
    for (unsigned T : {1u, 2u, 3u}) {
        std::vector<float> hn((size_t)T * HCD); for (auto& x : hn) x = ((rand() % 2001) - 1000) / 1000.f;
        float* dn; CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        float* dout[NS]; for (auto& p : dout) CHECK(hipMalloc(&p, (size_t)T * RANK * 4));
        const unsigned dflt = std::min(10u, std::max(1u, 48u / T));
        auto launch = [&](int v, int i) { const unsigned s = sh[v].split ? sh[v].split : dflt;
            hipLaunchKernelGGL(hc_pre_down, dim3(T, s), dim3(sh[v].block), HCD * 4, 0, dn, dw[i], dout[v], H, HC, RANK, T); };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[NS];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < NS; v++) { for (int i = 0; i < 3; i++) launch(v, i % NW); CHECK(hipEventRecord(e0)); for (int i = 0; i < 60; i++) launch(v, i % NW); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 60; }
        for (int v = 0; v < NS; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> o0((size_t)T * RANK), o1((size_t)T * RANK); CHECK(hipMemcpy(o0.data(), dout[0], o0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- hc_pre_down T=%u: ship %ux%u/1024 %6.1f us", T, T, dflt, t[0]);
        for (int v = 1; v < NS; v++) { CHECK(hipMemcpy(o1.data(), dout[v], o1.size() * 4, hipMemcpyDeviceToHost));
            printf(" | %ux%u/%u %6.1f x%.2f %s", T, sh[v].split, sh[v].block, t[v], t[0] / t[v], memcmp(o0.data(), o1.data(), o0.size() * 4) ? "DIFF" : "same"); }
        printf("\n");
        CHECK(hipFree(dn)); for (auto p : dout) CHECK(hipFree(p));
    }
    return 0;
}
