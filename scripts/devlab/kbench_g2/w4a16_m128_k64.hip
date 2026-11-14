// SPDX-License-Identifier: AGPL-3.0-only
// kbench (G2 27B snapshot prefill): w4a16_gemm_t_m128_f16a runs 29-32 TFLOP/s against ~59 f16 WMMA peak and pays two
// __syncthreads per 32-K step (single LDS buffer; the double-buffered version was LDS-occupancy-bound, 5.35 waves/CU).
// Candidate: the same single-buffer kernel at a 64-K step (~37 KB LDS, still 3 CTAs per 128 KB WGP): half the barriers
// and loop overhead, twice the register prefetch. Every accumulator sees the same WMMA operands in the same k order, so
// the output must be BYTE-identical to the shipped kernel. Prints both timings and the bytewise compare per shape/M.
#include "../q27/w4a16_gemm.cu"   // the 27B copy (common/ has a w4a16_gemm.cu without the m128 kernels)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int KS>
__device__ __forceinline__ void m128_f16a_ks_body(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    constexpr int A_TPR = KS / 8;            // threads per A row (one uint4 = 8 halves each)
    constexpr int A_RPR = 128 / A_TPR;       // A rows per round
    constexpr int A_RND = 128 / A_RPR;       // rounds over the 128 rows
    constexpr int B_RND = KS / 32;           // rounds of 16 k-pairs
    constexpr int B_ROW = KS + 8;
    const unsigned int cta_n = blockIdx.x * F16_N_TILE;
    const unsigned int cta_m = blockIdx.y * (2 * F16_M_TILE);
    if (cta_m >= M) return;
    const unsigned int warp_m_offset = (threadIdx.x / 32) * 16;
    const unsigned int lane_id = threadIdx.x % 32;

    __shared__ __align__(16) _Float16 smem_A[2 * F16_M_TILE][KS + F16_PAD_T];
#define KB_OFF(n) ((n) * B_ROW + ((n) >> 4) * 8)
    __shared__ __align__(16) _Float16 smem_B[KB_OFF(F16_N_TILE)];

    v8f acc0[8], acc1[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { acc0[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0}; acc1[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0}; }

    const unsigned int a_row_base = threadIdx.x / A_TPR, a_col = (threadIdx.x % A_TPR) * 8;
    const unsigned int b_kp0 = threadIdx.x >> 3, b_ns = (threadIdx.x & 7) << 4;
    uint4 ra[A_RND], rb[B_RND], rs[B_RND];

    auto load = [&](unsigned int kb) {
#pragma unroll
        for (int r = 0; r < A_RND; r++) {
            unsigned int gr = cta_m + r * A_RPR + a_row_base, gc = kb + a_col;
            ra[r] = (gr < M && gc + 7 < K) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        }
#pragma unroll
        for (int r = 0; r < B_RND; r++) {
            unsigned int kp = b_kp0 + 16 * r, gke = kb + (kp << 1), gns = cta_n + b_ns;
            rb[r] = (gke + 1 <= K && gns + 15 < N) ? *(const uint4*)&B_packed[(unsigned long long)(gke >> 1) * N + gns] : uint4{0, 0, 0, 0};
            unsigned int sg = kb / F16_GROUP + (kp >> 3);
            rs[r] = (gns + 15 < N && sg < K / F16_GROUP) ? *(const uint4*)&B_scale[(unsigned long long)sg * N + gns] : uint4{0, 0, 0, 0};
        }
    };
    auto store = [&] {
#pragma unroll
        for (int r = 0; r < A_RND; r++) *(uint4*)&smem_A[r * A_RPR + a_row_base][a_col] = bf16x8_to_f16x8(ra[r]);
#pragma unroll
        for (int r = 0; r < B_RND; r++) {
            const unsigned int kp = b_kp0 + 16 * r;
            const unsigned int* pw = (const unsigned int*)&rb[r];
            const unsigned int* ps = (const unsigned int*)&rs[r];
#pragma unroll
            for (int d = 0; d < 4; d++) {
                const unsigned int hl = e2m1x4_f16hi(pw[d] & 0x0F0F0F0Fu), hh = e2m1x4_f16hi((pw[d] >> 4) & 0x0F0F0F0Fu);
                const v2h s01 = e4m3x2_f16(ps[d] & 0xFFFFu), s23 = e4m3x2_f16(ps[d] >> 16);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned int sel = 0x0Cu | ((unsigned)j << 8) | (0x0Cu << 16) | ((4u + j) << 24);
                    const unsigned int wb = __builtin_amdgcn_perm(hh, hl, sel);
                    v2h w; __builtin_memcpy(&w, &wb, 4);
                    const _Float16 s = (j == 0) ? s01.x : (j == 1) ? s01.y : (j == 2) ? s23.x : s23.y;
                    w = w * (v2h){s, s};
                    *(v2h*)&smem_B[KB_OFF(b_ns + d * 4 + j) + kp * 2] = w;
                }
            }
        }
    };
    auto compute = [&] {
        const unsigned int m_row0 = warp_m_offset + (lane_id & 15);
#pragma unroll
        for (int h = 0; h < KS / 16; h++) {
            v16h a0, a1;
            __builtin_memcpy(&a0, &smem_A[m_row0][h * 16], 32);
            __builtin_memcpy(&a1, &smem_A[F16_M_TILE + m_row0][h * 16], 32);
#pragma unroll
            for (int nb = 0; nb < 8; nb++) {
                v16h b; __builtin_memcpy(&b, &smem_B[KB_OFF(nb * 16 + (lane_id & 15)) + h * 16], 32);
                acc0[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc0[nb]);
                acc1[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc1[nb]);
            }
        }
    };
    load(0); store(); __syncthreads();
    for (unsigned int kb = KS; kb < K; kb += KS) { load(kb); compute(); __syncthreads(); store(); __syncthreads(); }
    compute();
#undef KB_OFF
#pragma unroll
    for (int ch = 0; ch < 2; ch++) {
        v8f* acc = ch ? acc1 : acc0;
#pragma unroll
        for (int nb = 0; nb < 8; nb++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int r = cta_m + ch * F16_M_TILE + warp_m_offset + 2 * e + (lane_id >> 4), c = cta_n + nb * 16 + (lane_id & 15);
                if (r < M && c < N) C[(unsigned long long)r * N + c] = __float2bfloat16(acc[nb][e] * scale2);
            }
    }
}

extern "C" __global__ __launch_bounds__(128, 3) void m128_f16a_k64(const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bp, const unsigned char* __restrict__ Bs, const float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    m128_f16a_ks_body<64>((const unsigned short*)A, Bp, Bs, s2, C, M, N, K);
}
// Control: the same template at the shipped 32-K step (must match the shipped kernel's time and bytes).
extern "C" __global__ __launch_bounds__(128, 3) void m128_f16a_k32(const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bp, const unsigned char* __restrict__ Bs, const float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K) {
    m128_f16a_ks_body<32>((const unsigned short*)A, Bp, Bs, s2, C, M, N, K);
}

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
        unsigned char *dbp, *dbs; __nv_bfloat16 *dA, *dC0, *dC1;
        const size_t cbytes = (size_t)Mmax * s.N * 2;
        CHECK(hipMalloc(&dbp, bp.size())); CHECK(hipMalloc(&dbs, bs.size()));
        CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMalloc(&dC0, cbytes)); CHECK(hipMalloc(&dC1, cbytes));
        CHECK(hipMemcpy(dbp, bp.data(), bp.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dbs, bs.data(), bs.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        std::vector<unsigned char> h0(cbytes), h1(cbytes);
        for (unsigned M : {512u, 1024u, 2048u, 4096u}) {
            dim3 grid(s.N / 128, (M + 127) / 128);
            float t[3];
            for (int which = 0; which < 3; ++which) {
                __nv_bfloat16* out = which ? dC1 : dC0;
                auto go = [&] {
                    if (which == 0) hipLaunchKernelGGL(w4a16_gemm_t_m128_f16a, grid, dim3(128), 0, 0, dA, dbp, dbs, 0.75f, out, M, s.N, s.K);
                    else if (which == 1) hipLaunchKernelGGL(m128_f16a_k32, grid, dim3(128), 0, 0, dA, dbp, dbs, 0.75f, out, M, s.N, s.K);
                    else hipLaunchKernelGGL(m128_f16a_k64, grid, dim3(128), 0, 0, dA, dbp, dbs, 0.75f, out, M, s.N, s.K);
                };
                CHECK(hipMemset(out, 0xA5, cbytes));
                go(); CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
                if (which) {
                    CHECK(hipMemcpy(h0.data(), dC0, (size_t)M * s.N * 2, hipMemcpyDeviceToHost));
                    CHECK(hipMemcpy(h1.data(), dC1, (size_t)M * s.N * 2, hipMemcpyDeviceToHost));
                    size_t diff = 0; for (size_t i = 0; i < (size_t)M * s.N * 2; ++i) diff += h0[i] != h1[i];
                    printf("  %-8s M=%-5u %-6s vs shipped: %s (%zu bytes differ)\n", s.nm, M, which == 1 ? "k32" : "k64", diff ? "DIFF" : "BIT-IDENTICAL", diff);
                }
                const int N = 20; CHECK(hipEventRecord(a)); for (int i = 0; i < N; ++i) go(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
                float ms; CHECK(hipEventElapsedTime(&ms, a, b)); t[which] = ms / N;
            }
            const double fl = 2.0 * M * s.N * s.K / 1e9;
            printf("  %-8s M=%-5u shipped %7.3f ms %5.1f TF | k32 %7.3f ms %5.1f TF | k64 %7.3f ms %5.1f TF  (k64/shipped %.3f)\n", s.nm, M,
                   t[0], fl / t[0], t[1], fl / t[1], t[2], fl / t[2], t[2] / t[0]);
        }
        hipFree(dbp); hipFree(dbs); hipFree(dA); hipFree(dC0); hipFree(dC1);
    }
    return 0;
}
