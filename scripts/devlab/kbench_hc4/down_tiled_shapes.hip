// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #160 (Flash-Next C8): hc_pre_down_tiled is 9.1% of C8 decode (193 us/call, grid 1 x 20 of 1024 threads,
// fn-g 43808029150). rank 320 / 20 splits = 16 rows a block, so 16 of the 32 warps never get a row, and 20 blocks
// cover half of strix's 40 CUs. Same kernel, other launch shapes (block x row splits). Every (t, r) is still one warp
// walking i = lane, lane+32, ... in order, then the same shfl reduction: outputs must be BIT-IDENTICAL.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HCD = H * HC; const int NW = 8;   // 8 weight copies x 6.55 MB rotate past the MALL
    std::vector<__nv_bfloat16> hw((size_t)RANK * HCD); srand(3); for (auto& x : hw) x = __float2bfloat16(((rand() % 2001) - 1000) / 8000.f);
    __nv_bfloat16* dw[NW]; for (int i = 0; i < NW; i++) { CHECK(hipMalloc(&dw[i], hw.size() * 2)); CHECK(hipMemcpy(dw[i], hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    struct Sh { unsigned block, split; } sh[] = {{1024, 20}, {512, 20}, {256, 40}, {128, 80}, {256, 20}, {512, 40}};
    const int NS = sizeof(sh) / sizeof(sh[0]);
    for (unsigned T : {5u, 8u}) {
        std::vector<float> hn((size_t)T * HCD); for (auto& x : hn) x = ((rand() % 2001) - 1000) / 1000.f;
        float* dn; CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        std::vector<float*> dout(NS); for (auto& p : dout) CHECK(hipMalloc(&p, (size_t)T * RANK * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); std::vector<float> t(NS);
        auto launch = [&](int v, int i) { hipLaunchKernelGGL(hc_pre_down_tiled, dim3((T + HC_TT - 1) / HC_TT, sh[v].split), dim3(sh[v].block), 0, 0, dn, dw[i], dout[v], H, HC, RANK, T); };
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < NS; v++) { for (int i = 0; i < 3; i++) launch(v, i % NW); CHECK(hipEventRecord(e0)); for (int i = 0; i < 40; i++) launch(v, i % NW); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 40; }
        for (int v = 0; v < NS; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> o0((size_t)T * RANK), o1((size_t)T * RANK); CHECK(hipMemcpy(o0.data(), dout[0], o0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- hc_pre_down_tiled T=%u: ship 1024x20 %6.1f us", T, t[0]);
        for (int v = 1; v < NS; v++) { CHECK(hipMemcpy(o1.data(), dout[v], o1.size() * 4, hipMemcpyDeviceToHost));
            printf(" | %ux%u %6.1f x%.2f %s", sh[v].block, sh[v].split, t[v], t[0] / t[v], memcmp(o0.data(), o1.data(), o0.size() * 4) ? "DIFF" : "same"); }
        printf("\n");
        CHECK(hipFree(dn)); for (auto p : dout) CHECK(hipFree(p));
    }
    return 0;
}
