// KLOOP #16 (G1): the 27B DFlash verify FFN at M = 9..16 runs w4a16_gemv_batch16 on the NATIVE [N, K/2] layout
// (dense_ffn.rs: "else batch16"), not the m128 GEMM #15 baselined. Time it (and batch32 at Ripple's M = 17) at the
// shapes w4a16_rowgemv.hip times, so the rowgemv verdict for the 27B is against the kernel actually shipped.
// Timing only: random bytes, weights rotated over >= 512 MB so every call streams from DRAM.
#include "w4a16_gemv.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    struct S { const char* name; unsigned int M, N, K; } shapes[] = {
        {"ripple gate/up", 17, 40960, 5120}, {"ripple down", 17, 5120, 40960},
        {"27b gate/up K8", 9, 17408, 5120}, {"27b down K8", 9, 5120, 17408},
        {"27b gate/up M16", 16, 17408, 5120}, {"27b down M16", 16, 5120, 17408},
        {"27b gate/up M8", 8, 17408, 5120}, {"27b down M8", 8, 5120, 17408}};
    for (auto& s : shapes) {
        const unsigned M = s.M, N = s.N, K = s.K;
        const size_t pb = (size_t)K / 2 * N, sb = (size_t)K / 16 * N, wbytes = pb + sb;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char> hp(pb), hs(sb);
        srand(7); for (auto& b : hp) b = rand() & 0x77; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
        std::vector<unsigned char*> P(copies), Q(copies);
        for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&P[c], pb)); CHECK(hipMemcpy(P[c], hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&Q[c], sb)); CHECK(hipMemcpy(Q[c], hs.data(), sb, hipMemcpyHostToDevice)); }
        std::vector<__nv_bfloat16> hA((size_t)M * K); for (auto& a : hA) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *A, *C; CHECK(hipMalloc(&A, hA.size() * 2)); CHECK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C, (size_t)M * N * 2));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        printf("-- %s M=%u N=%u K=%u (%.1f MB)\n", s.name, M, N, K, wbytes / 1e6);
        for (int tier : {8, 16, 32}) {
            if ((unsigned)tier < M) continue;
            auto launch = [&](int c) {
                dim3 g((N + N_PER_BLOCK - 1) / N_PER_BLOCK), b(BLOCK_SIZE);
                if (tier == 8) hipLaunchKernelGGL(w4a16_gemv_batch8, g, b, 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
                else if (tier == 16) hipLaunchKernelGGL(w4a16_gemv_batch16, g, b, 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
                else hipLaunchKernelGGL(w4a16_gemv_batch32, g, b, 0, 0, A, P[c], Q[c], 0.01f, C, M, N, K);
            };
            for (int i = 0; i < 3; i++) launch(i % copies);
            const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) launch(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            printf("   batch%-8d %8.1f us  %6.1f GB/s\n", tier, 1000 * ms / it, wbytes / 1e9 / (ms / it / 1000));
        }
        for (int c = 0; c < copies; c++) { hipFree(P[c]); hipFree(Q[c]); }
        hipFree(A); hipFree(C);
    }
    return 0;
}
