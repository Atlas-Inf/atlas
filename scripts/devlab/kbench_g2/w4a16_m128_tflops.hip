// SPDX-License-Identifier: AGPL-3.0-only
// kbench (G2 27B snapshot prefill): w4a16_gemm_t_m128_f16a is 38% of a 2k extension prefill on a cached 16k prefix
// (runner 42927951718). Baseline ceiling check before writing a candidate: TFLOP/s of the shipped f16a kernel and its
// bf16 twin w4a16_gemm_t_m128 at the 27B FFN shapes (gate/up N=17408 K=5120, down N=5120 K=17408) for M = 512, 1024,
// 2048, 4096, against gfx1151's ~59 TFLOP/s f16 WMMA peak and the ~33-44 measured for rocBLAS BF16 (11-10 kbench).
#include "w4a16_gemm.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    struct S { const char* nm; unsigned N, K; } shapes[] = {{"gate/up", 17408, 5120}, {"down", 5120, 17408}};
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    for (auto s : shapes) {
        const unsigned Mmax = 4096;
        std::vector<unsigned char> bp((size_t)s.K / 2 * s.N), bs((size_t)s.K / 16 * s.N);
        std::vector<uint16_t> ha((size_t)Mmax * s.K);
        srand(3);
        for (auto& v : bp) v = rand() & 0xff;
        for (auto& v : bs) v = 0x30 + (rand() % 16);
        for (auto& v : ha) v = 0x3c00 + (rand() % 0x200) - 0x100;
        unsigned char *dbp, *dbs; __nv_bfloat16 *dA, *dC;
        CHECK(hipMalloc(&dbp, bp.size())); CHECK(hipMalloc(&dbs, bs.size()));
        CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMalloc(&dC, (size_t)Mmax * s.N * 2));
        CHECK(hipMemcpy(dbp, bp.data(), bp.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dbs, bs.data(), bs.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (unsigned M : {512u, 1024u, 2048u, 4096u}) {
            dim3 grid(s.N / 128, (M + 127) / 128);
            for (int which = 0; which < 2; ++which) {
                auto go = [&] {
                    if (which == 0) hipLaunchKernelGGL(w4a16_gemm_t_m128_f16a, grid, dim3(128), 0, 0, dA, dbp, dbs, 1.0f, dC, M, s.N, s.K);
                    else hipLaunchKernelGGL(w4a16_gemm_t_m128, grid, dim3(128), 0, 0, dA, dbp, dbs, 1.0f, dC, M, s.N, s.K);
                };
                go(); CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
                const int N = 20; CHECK(hipEventRecord(a)); for (int i = 0; i < N; ++i) go(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
                float ms; CHECK(hipEventElapsedTime(&ms, a, b)); ms /= N;
                printf("  %-8s M=%-5u %-10s %8.3f ms  %6.1f TFLOP/s\n", s.nm, M, which ? "m128(bf16)" : "m128_f16a", ms, 2.0 * M * s.N * s.K / (ms * 1e9));
            }
        }
        hipFree(dbp); hipFree(dbs); hipFree(dA); hipFree(dC);
    }
    return 0;
}
