// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #82 (Ripple FP8 attention): the existing FP8 kernels lose at the verify M (w8a16_gemv_batch16 0.78x and
// w8a16_gemm_t_m128 0.73x rocBLAS on q|gate M16/M17, 43464561761) and dense_gemv_fp8w_batchm caps at M=8.
// Candidate fp8_m32<NT>: the #354 m32 WMMA verify twin with an FP8 B operand. B_t [K, N] e4m3 with a per-output-row
// f32 scale (the quantize_bf16_to_fp8 format, transposed). e4m3 -> f16 is a bit shift ((b&0x80)<<8 | (b&0x7F)<<7 is
// exactly value * 2^-8, subnormals included), and the 2^8 folds into the row scale, so the dequant costs no ALU. The
// rows are scaled once in the epilogue. Reports us, GB/s and error vs rocBLAS BF16 at M 1/9/17 (dense_gemv_fp8w M=1
// is in fp8_gemv1.hip: its scl_fp8 collides with w4a16_gemm.cu).
#include "../q27/w4a16_gemm.cu"
#include "common8.h"
#include <cstring>

#define F8_B_OFF(n) ((n) * 40 + ((n) >> 4) * 8)
__device__ __forceinline__ unsigned int e4m3x2_f16x2(unsigned int lo, unsigned int hi) {
    return (((lo & 0x80u) << 8) | ((lo & 0x7Fu) << 7)) | ((((hi & 0x80u) << 8) | ((hi & 0x7Fu) << 7)) << 16);
}
template <int NT>
__global__ __launch_bounds__(128) void fp8_m32(const __nv_bfloat16* __restrict__ A_, const unsigned char* __restrict__ B_t,
    const float* __restrict__ col_scale, __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    constexpr int NBW = NT / 64;
    constexpr int OCT = NT / 8;                 // n-octets per CTA
    constexpr int UPT = 16 * OCT / 128;         // (k-pair, n-octet) units per thread
    const unsigned short* A = (const unsigned short*)A_;
    const unsigned int cta_n = blockIdx.x * NT, cta_m = blockIdx.y * 32;
    if (cta_m >= M) return;
    const unsigned int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    __shared__ __align__(16) _Float16 sA[32][F16_K_STEP + F16_PAD_T];
    __shared__ __align__(16) _Float16 sB[F8_B_OFF(NT)];
    v8f acc[2][NBW];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int j = 0; j < NBW; j++) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int a_row = threadIdx.x >> 2, a_col = (threadIdx.x & 3) << 3;
    auto load = [&](unsigned int kb, uint4& ra, uint2 (&b0)[UPT], uint2 (&b1)[UPT]) {
        const unsigned int gr = cta_m + a_row, gc = kb + a_col;
        ra = (gr < M && gc + 7 < K) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        #pragma unroll
        for (int u = 0; u < UPT; u++) {
            const unsigned int id = threadIdx.x + u * 128, kp = id / OCT, gn = cta_n + (id % OCT) * 8, gk = kb + 2 * kp;
            const bool on = gn + 7 < N && gk + 1 < K;
            b0[u] = on ? *(const uint2*)&B_t[(unsigned long long)gk * N + gn] : uint2{0, 0};
            b1[u] = on ? *(const uint2*)&B_t[(unsigned long long)(gk + 1) * N + gn] : uint2{0, 0};
        }
    };
    auto store = [&](const uint4& ra, const uint2 (&b0)[UPT], const uint2 (&b1)[UPT]) {
        *(uint4*)&sA[a_row][a_col] = bf16x8_to_f16x8(ra);
        #pragma unroll
        for (int u = 0; u < UPT; u++) {
            const unsigned int id = threadIdx.x + u * 128, kp = id / OCT, nl = (id % OCT) * 8;
            const unsigned int lo[2] = {b0[u].x, b0[u].y}, hi[2] = {b1[u].x, b1[u].y};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned int w = e4m3x2_f16x2((lo[j >> 2] >> ((j & 3) * 8)) & 0xFFu, (hi[j >> 2] >> ((j & 3) * 8)) & 0xFFu);
                *(unsigned int*)&sB[F8_B_OFF(nl + j) + kp * 2] = w;
            }
        }
    };
    auto compute = [&]() {
        const unsigned int r0 = lane_id & 15;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            v16h a0, a1;
            __builtin_memcpy(&a0, &sA[r0][h * 16], 32);
            __builtin_memcpy(&a1, &sA[16 + r0][h * 16], 32);
            #pragma unroll
            for (int j = 0; j < NBW; j++) {
                const unsigned int nc = (warp_id * NBW + j) * 16 + (lane_id & 15);
                v16h b;
                __builtin_memcpy(&b, &sB[F8_B_OFF(nc) + h * 16], 32);
                acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0][j]);
                acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1][j]);
            }
        }
    };
    uint4 ra; uint2 b0[UPT], b1[UPT];
    load(0, ra, b0, b1); store(ra, b0, b1); __syncthreads();
    for (unsigned int kb = F16_K_STEP; kb < K; kb += F16_K_STEP) {
        load(kb, ra, b0, b1);
        compute();
        __syncthreads();
        store(ra, b0, b1);
        __syncthreads();
    }
    compute();
    #pragma unroll
    for (int rt = 0; rt < 2; rt++)
        #pragma unroll
        for (int j = 0; j < NBW; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const unsigned int r = cta_m + rt * 16 + 2 * e + (lane_id >> 4);
                const unsigned int c = cta_n + (warp_id * NBW + j) * 16 + (lane_id & 15);
                if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[rt][j][e] * col_scale[c]);
            }
}

int main() {
    rocblas_handle h; rocblas_create_handle(&h);
    for (auto& s : kShapes) {
        const unsigned N = s.N, K = s.K;
        // BF16 weights as in common8.h, per-output-row e4m3 (quantize_bf16_to_fp8 semantics: amax/448).
        std::vector<__nv_bfloat16> wb((size_t)N * K);
        { std::mt19937 g(1234); std::normal_distribution<float> d(0.f, 0.02f); std::uniform_real_distribution<float> u(0.f, 1.f);
          for (auto& v : wb) { float x = d(g); if (u(g) < 0.001f) x *= 8.f; v = __float2bfloat16(x); } }
        std::vector<unsigned char> qn((size_t)N * K), qt((size_t)K * N); std::vector<float> rs(N), rs256(N);
        for (unsigned n = 0; n < N; n++) {
            float amax = 0; for (unsigned k = 0; k < K; k++) amax = std::max(amax, fabsf(__bfloat162float(wb[(size_t)n * K + k])));
            const float sc = amax > 0 ? amax / 448.f : 1.f; rs[n] = sc; rs256[n] = sc * 256.f;
            for (unsigned k = 0; k < K; k++) { unsigned char b = e4m3_enc(__bfloat162float(wb[(size_t)n * K + k]) / sc); qn[(size_t)n * K + k] = b; qt[(size_t)k * N + n] = b; }
        }
        const size_t bfb = wb.size() * 2, qb = qn.size() + N * 4;
        const int cb = (int)std::max<size_t>(2, (512ull << 20) / bfb + 1), cq = (int)std::max<size_t>(2, (512ull << 20) / qb + 1);
        std::vector<__nv_bfloat16*> WB(cb); std::vector<unsigned char*> QN(cq), QT(cq); float *RS, *RS256;
        for (auto& p : WB) { CHECK(hipMalloc(&p, bfb)); CHECK(hipMemcpy(p, wb.data(), bfb, hipMemcpyHostToDevice)); }
        for (int c = 0; c < cq; c++) { CHECK(hipMalloc(&QN[c], qn.size())); CHECK(hipMemcpy(QN[c], qn.data(), qn.size(), hipMemcpyHostToDevice));
            CHECK(hipMalloc(&QT[c], qt.size())); CHECK(hipMemcpy(QT[c], qt.data(), qt.size(), hipMemcpyHostToDevice)); }
        CHECK(hipMalloc(&RS, N * 4)); CHECK(hipMemcpy(RS, rs.data(), N * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&RS256, N * 4)); CHECK(hipMemcpy(RS256, rs256.data(), N * 4, hipMemcpyHostToDevice));
        for (unsigned M : {1u, 9u, 17u}) {
            auto ha = make_act(M, K); __nv_bfloat16 *A, *C0, *C1;
            CHECK(hipMalloc(&A, ha.size() * 2)); CHECK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMalloc(&C0, (size_t)M * N * 2)); CHECK(hipMalloc(&C1, (size_t)M * N * 2));
            const float tb = time_us([&](int c) { roc_bf16(h, A, WB[c], C0, M, N, K); }, cb);
            roc_bf16(h, A, WB[0], C0, M, N, K); CHECK(hipDeviceSynchronize());
            std::vector<__nv_bfloat16> r((size_t)M * N), c((size_t)M * N);
            CHECK(hipMemcpy(r.data(), C0, r.size() * 2, hipMemcpyDeviceToHost));
            printf("-- %-7s M=%-2u N=%u K=%u | rocBLAS bf16 %8.1f us %6.1f GB/s\n", s.name, M, N, K, tb, bfb / 1e3 / tb);
            auto report = [&](const char* nm, float t) {
                CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(c.data(), C1, c.size() * 2, hipMemcpyDeviceToHost));
                double mx, rm; err(r, c, mx, rm);
                printf("   %-14s %8.1f us %6.1f GB/s  x%.2f  err max %.2e rms %.2e\n", nm, t, qb / 1e3 / t, tb / t, mx, rm);
            };
            auto l64 = [&](int cc) { hipLaunchKernelGGL(fp8_m32<64>, dim3((N + 63) / 64, (M + 31) / 32), dim3(128), 0, 0, A, QT[cc], RS256, C1, M, N, K); };
            auto l128 = [&](int cc) { hipLaunchKernelGGL(fp8_m32<128>, dim3((N + 127) / 128, (M + 31) / 32), dim3(128), 0, 0, A, QT[cc], RS256, C1, M, N, K); };
            { const float t = time_us(l64, cq); CHECK(hipMemset(C1, 0, (size_t)M * N * 2)); l64(0); report("fp8_m32 n64", t); }
            { const float t = time_us(l128, cq); CHECK(hipMemset(C1, 0, (size_t)M * N * 2)); l128(0); report("fp8_m32 n128", t); }
            hipFree(A); hipFree(C0); hipFree(C1);
        }
        for (auto p : WB) hipFree(p); for (int c = 0; c < cq; c++) { hipFree(QN[c]); hipFree(QT[c]); } hipFree(RS); hipFree(RS256);
    }
    return 0;
}
