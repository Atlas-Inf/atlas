// KLOOP #81: FP8 attention, M=17 (Ripple DFlash2 verify block): w8a16_gemm_t_m128 (strix-hip common, B_t [K,N]
// e4m3 + [K/128,N/128] scales) vs rocBLAS BF16 at M 9/17/32. See common8.h.
#include "w8a16_gemm_t_m128.cu"
#include "common8.h"
int main() {
    rocblas_handle h; rocblas_create_handle(&h);
    for (auto& s : kShapes) {
        const unsigned N = s.N, K = s.K; W w(N, K);
        std::vector<unsigned char> qt((size_t)K * N); std::vector<float> st((size_t)(K / 128) * (N / 128));
        for (unsigned n = 0; n < N; n++) for (unsigned k = 0; k < K; k++) qt[(size_t)k * N + n] = w.q[(size_t)n * K + k];
        for (unsigned bn = 0; bn < N / 128; bn++) for (unsigned bk = 0; bk < K / 128; bk++) st[(size_t)bk * (N / 128) + bn] = w.sc[(size_t)bn * (K / 128) + bk];
        const size_t bfb = w.bf.size() * 2, qb = qt.size() + st.size() * 4;
        const int cb = (int)std::max<size_t>(2, (512ull << 20) / bfb + 1), cq = (int)std::max<size_t>(2, (512ull << 20) / qb + 1);
        std::vector<__nv_bfloat16*> WB(cb); std::vector<unsigned char*> WQ(cq); std::vector<float*> WS(cq);
        for (auto& p : WB) { CHECK(hipMalloc(&p, bfb)); CHECK(hipMemcpy(p, w.bf.data(), bfb, hipMemcpyHostToDevice)); }
        for (int c = 0; c < cq; c++) { CHECK(hipMalloc(&WQ[c], qt.size())); CHECK(hipMemcpy(WQ[c], qt.data(), qt.size(), hipMemcpyHostToDevice));
            CHECK(hipMalloc(&WS[c], st.size() * 4)); CHECK(hipMemcpy(WS[c], st.data(), st.size() * 4, hipMemcpyHostToDevice)); }
        for (unsigned M : {9u, 17u, 32u}) {
            auto ha = make_act(M, K); __nv_bfloat16 *A, *C0, *C1;
            CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
            const float tb = time_us([&](int c) { roc_bf16(h, A, WB[c], C0, M, N, K); }, cb);
            roc_bf16(h, A, WB[0], C0, M, N, K);
            auto lq = [&](int c) { hipLaunchKernelGGL(w8a16_gemm_t_m128, dim3((N + 127) / 128, (M + 127) / 128), dim3(256), 0, 0, A, WQ[c], WS[c], C1, M, N, K); };
            const float tq = time_us(lq, cq); lq(0); CHECK(hipDeviceSynchronize());
            std::vector<__nv_bfloat16> r((size_t)M * N), c((size_t)M * N);
            CHECK(hipMemcpy(r.data(), C0, r.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(c.data(), C1, c.size() * 2, hipMemcpyDeviceToHost));
            double mx, rm; err(r, c, mx, rm);
            printf("-- %-7s M=%-2u N=%u K=%u | rocBLAS bf16 %8.1f us %6.1f GB/s | fp8 m128 %8.1f us %6.1f GB/s | x%.2f | err max %.2e rms %.2e\n",
                   s.name, M, N, K, tb, bfb / 1e3 / tb, tq, qb / 1e3 / tq, tb / tq, mx, rm);
            hipFree(A); hipFree(C0); hipFree(C1);
        }
        for (auto p : WB) hipFree(p); for (int c = 0; c < cq; c++) { hipFree(WQ[c]); hipFree(WS[c]); }
    }
    return 0;
}
