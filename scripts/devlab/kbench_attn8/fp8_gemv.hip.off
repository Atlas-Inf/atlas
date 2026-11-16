// KLOOP #81: FP8 attention, M=1 (Ripple AR decode) and M=16 (DFlash gamma verify rows): w8a16_gemv /
// w8a16_gemv_batch16 on [N,K] e4m3 + 128x128 scales vs rocBLAS BF16. See common8.h.
#include "w8a16_gemv_batch4.cu"
#include "common8.h"
int main() {
    rocblas_handle h; rocblas_create_handle(&h);
    for (auto& s : kShapes) {
        const unsigned N = s.N, K = s.K; W w(N, K);
        const size_t bfb = w.bf.size() * 2, qb = w.q.size() + w.sc.size() * 4;
        const int cb = (int)std::max<size_t>(2, (512ull << 20) / bfb + 1), cq = (int)std::max<size_t>(2, (512ull << 20) / qb + 1);
        std::vector<__nv_bfloat16*> WB(cb); std::vector<unsigned char*> WQ(cq); std::vector<float*> WS(cq);
        for (auto& p : WB) { CHECK(hipMalloc(&p, bfb)); CHECK(hipMemcpy(p, w.bf.data(), bfb, hipMemcpyHostToDevice)); }
        for (int c = 0; c < cq; c++) { CHECK(hipMalloc(&WQ[c], w.q.size())); CHECK(hipMemcpy(WQ[c], w.q.data(), w.q.size(), hipMemcpyHostToDevice));
            CHECK(hipMalloc(&WS[c], w.sc.size() * 4)); CHECK(hipMemcpy(WS[c], w.sc.data(), w.sc.size() * 4, hipMemcpyHostToDevice)); }
        for (unsigned M : {1u, 8u, 16u}) {
            auto ha = make_act(M, K); __nv_bfloat16 *A, *C0, *C1;
            CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
            const float tb = time_us([&](int c) { roc_bf16(h, A, WB[c], C0, M, N, K); }, cb);
            roc_bf16(h, A, WB[0], C0, M, N, K);
            const float tq = time_us([&](int c) { hipLaunchKernelGGL(w8a16_gemv_batch16, dim3((N + 3) / 4), dim3(256), 0, 0, A, WQ[c], WS[c], C1, M, N, K); }, cq);
            hipLaunchKernelGGL(w8a16_gemv_batch16, dim3((N + 3) / 4), dim3(256), 0, 0, A, WQ[0], WS[0], C1, M, N, K);
            CHECK(hipDeviceSynchronize());
            std::vector<__nv_bfloat16> r((size_t)M * N), c((size_t)M * N);
            CHECK(hipMemcpy(r.data(), C0, r.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(c.data(), C1, c.size() * 2, hipMemcpyDeviceToHost));
            double mx, rm; err(r, c, mx, rm);
            printf("-- %-7s M=%-2u N=%u K=%u | rocBLAS bf16 %8.1f us %6.1f GB/s | fp8 batch16 %8.1f us %6.1f GB/s | x%.2f | err max %.2e rms %.2e\n",
                   s.name, M, N, K, tb, bfb / 1e3 / tb, tq, qb / 1e3 / tq, tb / tq, mx, rm);
            hipFree(A); hipFree(C0); hipFree(C1);
        }
        for (auto p : WB) hipFree(p); for (int c = 0; c < cq; c++) { hipFree(WQ[c]); hipFree(WS[c]); }
    }
    return 0;
}
