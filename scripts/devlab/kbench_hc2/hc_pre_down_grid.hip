// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #129 (G3 Flash-Next C1 decode): hc_pre_down + hc_pre_finish_x4 are 27% of single-user decode GPU time
// (rocprof 43677480400; hc_pre_down ~114 us/call). At T=1 the call site launches dsplit = min(48/T, 10) = 10 blocks
// of 1024 threads, so 10 of 40 CUs stream the 6.5 MB down_w (rank 320 x hc*H 10240 BF16), ~57 GB/s. Same kernel,
// more and smaller blocks (rows still one warp each, same per-lane FMA order), so outputs must be BIT-IDENTICAL.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HD = H * HC;
    const int copies = 16;   // rotate 16 x 6.5 MB weights past L2
    std::vector<__nv_bfloat16> hw((size_t)RANK * HD); srand(5); for (auto& v : hw) v = __float2bfloat16(((rand() % 2001) - 1000) / 40000.f);
    std::vector<__nv_bfloat16*> W(copies); for (auto& p : W) { CHECK(hipMalloc(&p, hw.size() * 2)); CHECK(hipMemcpy(p, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    for (unsigned T : {1u, 2u, 4u}) {
        std::vector<float> hn((size_t)T * HD); for (auto& v : hn) v = ((rand() % 2001) - 1000) / 1000.f;
        float *N, *L0, *L1; CHECK(hipMalloc(&N, hn.size() * 4)); CHECK(hipMemcpy(N, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&L0, (size_t)T * RANK * 4)); CHECK(hipMalloc(&L1, (size_t)T * RANK * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto run = [&](unsigned ds, unsigned bs, float* out, int c) { hipLaunchKernelGGL(hc_pre_down, dim3(T, ds), dim3(bs), HD * 4, 0, N, W[c], out, H, HC, RANK, T); };
        auto tm = [&](unsigned ds, unsigned bs) { for (int i = 0; i < 4; i++) run(ds, bs, L1, i % copies); CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) run(ds, bs, L1, i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / 64; };
        const unsigned ship = (48 / T) < 10 ? (48 / T) : 10;
        run(ship, 1024, L0, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> r0((size_t)T * RANK), r1((size_t)T * RANK); CHECK(hipMemcpy(r0.data(), L0, r0.size() * 4, hipMemcpyDeviceToHost));
        const float tship = tm(ship, 1024);
        printf("-- T=%u ship dsplit %u x 1024: %7.1f us (%.0f GB/s)\n", T, ship, tship, hw.size() * 2 / 1e3 / tship);
        struct V { unsigned ds, bs; } vs[] = {{20, 512}, {40, 256}, {80, 128}, {40, 512}, {80, 256}, {160, 64}};
        for (auto v : vs) {
            const float t = tm(v.ds, v.bs); CHECK(hipMemset(L1, 0xFF, r1.size() * 4)); run(v.ds, v.bs, L1, 0); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(r1.data(), L1, r1.size() * 4, hipMemcpyDeviceToHost));
            printf("   dsplit %3u x %4u: %7.1f us (%.0f GB/s) x%.2f  %s\n", v.ds, v.bs, t, hw.size() * 2 / 1e3 / t, tship / t, memcmp(r0.data(), r1.data(), r0.size() * 4) ? "DIFFERS" : "bit-identical");
        }
        hipFree(N); hipFree(L0); hipFree(L1);
    }
    return 0;
}
