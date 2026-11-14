// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #82: Ripple AR decode (M=1) FP8 attention via the existing dense_gemv_fp8w (per-row e4m3, quantized on the
// GPU by quantize_bf16_to_fp8, i.e. the exact load-time path) vs rocBLAS BF16. See common8.h / fp8_m32.hip.
#include "dense_gemv_fp8w.cu"
#include "common8.h"
int main() {
    rocblas_handle h; rocblas_create_handle(&h);
    for (auto& s : kShapes) {
        const unsigned N = s.N, K = s.K;
        std::vector<__nv_bfloat16> wb((size_t)N * K);
        { std::mt19937 g(1234); std::normal_distribution<float> d(0.f, 0.02f); std::uniform_real_distribution<float> u(0.f, 1.f);
          for (auto& v : wb) { float x = d(g); if (u(g) < 0.001f) x *= 8.f; v = __float2bfloat16(x); } }
        const size_t bfb = wb.size() * 2, qb = (size_t)N * K + N * 4;
        const int cb = (int)std::max<size_t>(2, (512ull << 20) / bfb + 1), cq = (int)std::max<size_t>(2, (512ull << 20) / qb + 1);
        std::vector<__nv_bfloat16*> WB(cb); std::vector<unsigned char*> Q(cq); std::vector<float*> RS(cq);
        for (auto& p : WB) { CHECK(hipMalloc(&p, bfb)); CHECK(hipMemcpy(p, wb.data(), bfb, hipMemcpyHostToDevice)); }
        for (int c = 0; c < cq; c++) { CHECK(hipMalloc(&Q[c], (size_t)N * K)); CHECK(hipMalloc(&RS[c], N * 4));
            hipLaunchKernelGGL(quantize_bf16_to_fp8, dim3(N), dim3(256), 0, 0, WB[0], Q[c], RS[c], N, K); }
        CHECK(hipDeviceSynchronize());
        auto ha = make_act(1, K); __nv_bfloat16 *A, *C0, *C1;
        CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&C0, N * 2)); CHECK(hipMalloc(&C1, N * 2));
        const float tb = time_us([&](int c) { roc_bf16(h, A, WB[c], C0, 1, N, K); }, cb);
        roc_bf16(h, A, WB[0], C0, 1, N, K);
        auto l = [&](int c) { hipLaunchKernelGGL(dense_gemv_fp8w, dim3((N + 3) / 4), dim3(256), 0, 0, A, Q[c], RS[c], C1, N, K); };
        const float tq = time_us(l, cq); l(0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> r(N), c(N);
        CHECK(hipMemcpy(r.data(), C0, N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(c.data(), C1, N * 2, hipMemcpyDeviceToHost));
        double mx, rm; err(r, c, mx, rm);
        printf("-- %-7s M=1 N=%u K=%u | rocBLAS bf16 %8.1f us %6.1f GB/s | dense_gemv_fp8w %8.1f us %6.1f GB/s | x%.2f | err max %.2e rms %.2e\n",
               s.name, N, K, tb, bfb / 1e3 / tb, tq, qb / 1e3 / tq, tb / tq, mx, rm);
        for (auto p : WB) hipFree(p); for (int cc = 0; cc < cq; cc++) { hipFree(Q[cc]); hipFree(RS[cc]); } hipFree(A); hipFree(C0); hipFree(C1);
    }
    return 0;
}
