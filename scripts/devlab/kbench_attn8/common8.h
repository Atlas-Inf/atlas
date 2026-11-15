// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #81 shared harness: Ripple-65B attention projections stored BF16 today (43454185121: dense_gemv_bf16 = 47% of
// AR decode, at ~bandwidth; rocBLAS Cijk = 33% of DFlash2 decode). Candidate: requant to FP8 E4M3 with 128x128 FP32
// block scales at load (half the bytes) and run the existing w8a16 kernels. Each bench reports us, weight GB/s and
// the FP8 output error vs the rocBLAS BF16 result on the same activations (max|d|/rms and rms(d)/rms).
#pragma once
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static float e4m3_val(unsigned b) {
    unsigned s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 7u; float v;
    if (e == 0) v = m * 0.001953125f; else if (e == 15 && m == 7) v = 0.f; else v = ldexpf(1.f + m / 8.f, (int)e - 7);
    return s ? -v : v;
}
// Round-to-nearest e4m3fn encode (|x| <= 448 after scaling).
static unsigned char e4m3_enc(float x) {
    static std::vector<float> pos; if (pos.empty()) for (unsigned b = 0; b < 0x7F; b++) pos.push_back(e4m3_val(b));
    const float a = std::min(fabsf(x), 448.f);
    auto it = std::lower_bound(pos.begin(), pos.end(), a); unsigned i = (unsigned)(it - pos.begin());
    if (i == pos.size()) i--; else if (i > 0 && a - pos[i - 1] <= pos[i] - a) i--;
    return (unsigned char)(i | (x < 0 ? 0x80u : 0u));
}
struct Shape { const char* name; unsigned N, K; };
static const Shape kShapes[] = {{"q|gate", 20480, 5120}, {"k|v", 2048, 5120}, {"o", 5120, 10240}};

// Weights: N(0, 0.02) with 0.1% outliers x8 (attention projections carry a few large channels).
struct W {
    unsigned N, K; std::vector<__nv_bfloat16> bf; std::vector<unsigned char> q;   // q: [N,K] e4m3
    std::vector<float> sc;                                                        // [N/128, K/128]
    W(unsigned N_, unsigned K_) : N(N_), K(K_), bf((size_t)N_ * K_), q((size_t)N_ * K_), sc((size_t)(N_ / 128) * (K_ / 128)) {
        std::mt19937 g(1234); std::normal_distribution<float> d(0.f, 0.02f); std::uniform_real_distribution<float> u(0.f, 1.f);
        for (auto& v : bf) { float x = d(g); if (u(g) < 0.001f) x *= 8.f; v = __float2bfloat16(x); }
        for (unsigned bn = 0; bn < N / 128; bn++) for (unsigned bk = 0; bk < K / 128; bk++) {
            float amax = 0; for (unsigned n = 0; n < 128; n++) for (unsigned k = 0; k < 128; k++) amax = std::max(amax, fabsf(__bfloat162float(bf[(size_t)(bn * 128 + n) * K + bk * 128 + k])));
            const float s = amax > 0 ? amax / 448.f : 1.f; sc[(size_t)bn * (K / 128) + bk] = s;
            for (unsigned n = 0; n < 128; n++) for (unsigned k = 0; k < 128; k++) { size_t i = (size_t)(bn * 128 + n) * K + bk * 128 + k; q[i] = e4m3_enc(__bfloat162float(bf[i]) / s); }
        }
    }
};
static std::vector<__nv_bfloat16> make_act(unsigned M, unsigned K) {
    std::mt19937 g(99); std::normal_distribution<float> d(0.f, 1.f); std::vector<__nv_bfloat16> a((size_t)M * K);
    for (auto& v : a) v = __float2bfloat16(d(g)); return a;
}
// rocBLAS BF16 reference / timing: C[M,N] = A[M,K] * W[N,K]^T (row-major).
static void roc_bf16(rocblas_handle h, const __nv_bfloat16* A, const __nv_bfloat16* Wd, __nv_bfloat16* C, unsigned M, unsigned N, unsigned K) {
    const float al = 1.f, be = 0.f;
    rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &al, Wd, rocblas_datatype_bf16_r, K, A, rocblas_datatype_bf16_r, K,
                    &be, C, rocblas_datatype_bf16_r, N, C, rocblas_datatype_bf16_r, N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}
static void err(const std::vector<__nv_bfloat16>& r, const std::vector<__nv_bfloat16>& c, double& maxrel, double& rmsrel) {
    double rr = 0, dd = 0, mx = 0;
    for (size_t i = 0; i < r.size(); i++) { double a = __bfloat162float(r[i]), b = __bfloat162float(c[i]); rr += a * a; dd += (a - b) * (a - b); mx = std::max(mx, fabs(a - b)); }
    const double rms = sqrt(rr / r.size()) + 1e-30; maxrel = mx / rms; rmsrel = sqrt(dd / r.size()) / rms;
}
template <class F> static float time_us(F f, int copies) {
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 3; i++) f(i % copies);
    const int it = 30; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) f(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
    float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / it;
}
