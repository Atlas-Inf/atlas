// SPDX-License-Identifier: AGPL-3.0-only
// kbench: deep-K DFlash-verify FFN down_proj (M = K+1 verify rows) on gfx1151.
// `w4a16_gemm_t_k64` tiles only N (N/128 CTAs): Ripple-65B down (N=5120, K=40960)
// and Qwen3.8-27B down (N=5120, K=17408) launch 40 CTAs on a 40-CU part —
// rocprofv3 (f236) 2.4 ms per Ripple call vs ~0.5 ms to stream the weights.
// Candidate: the same K64 tile loop over K-slices (blockIdx.z), fp32 partials,
// fixed-order reduce. NOT bit-identical (fp32 partial sums) — reports rel error.
// The 27B tree's copy (kbench stages it at <tmp>/q27; common/ has a different w4a16_gemm.cu).
#include "../q27/w4a16_gemm.cu"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define KS_K_STEP 64
#define KS_PAD 8

// w4a16_gemm_t_k64 body with K-slice [k_lo, k_hi) and fp32 partial output.
extern "C" __global__ void w4a16_gemm_t_k64_splitk(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, float* __restrict__ P,
    unsigned int M, unsigned int N, unsigned int K, unsigned int k_slice) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int k_lo = blockIdx.z * k_slice, k_hi = min(K, k_lo + k_slice);
    const unsigned int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32, warp_m_offset = warp_id * 16;
    __shared__ __nv_bfloat16 sA[2][M_TILE][KS_K_STEP + KS_PAD];
    __shared__ __nv_bfloat16 sB[2][N_TILE_LG][KS_K_STEP + 8];
    __shared__ float sLUT[16];
    if (threadIdx.x < 16) sLUT[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int a_row = threadIdx.x >> 1, a_col = (threadIdx.x & 1) << 5;
    const unsigned int b_kp = threadIdx.x >> 2, b_ns = (threadIdx.x & 3) << 5;
    auto load = [&](unsigned int kb, uint4* ra, uint4* rb, uint4* rs) {
        #pragma unroll
        for (int rnd = 0; rnd < 4; rnd++) {
            unsigned int gr = cta_m + a_row, gc = kb + a_col + rnd * 8;
            ra[rnd] = ((gr < M) && (gc + 7 < k_hi)) ? *(const uint4*)&A[(unsigned long long)gr * K + gc] : uint4{0, 0, 0, 0};
        }
        unsigned int gns = cta_n + b_ns, gke = kb + (b_kp << 1), sg = kb / GROUP_SIZE + (b_kp >> 3);
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            rb[half] = ((gke + 1 <= k_hi) && (gns + half * 16 + 15 < N)) ? *(const uint4*)&B_packed[(unsigned long long)(gke >> 1) * N + gns + half * 16] : uint4{0, 0, 0, 0};
            rs[half] = (gns + half * 16 + 15 < N) ? *(const uint4*)&B_scale[(unsigned long long)sg * N + gns + half * 16] : uint4{0, 0, 0, 0};
        }
    };
    auto store = [&](int buf, uint4* ra, uint4* rb, uint4* rs) {
        #pragma unroll
        for (int rnd = 0; rnd < 4; rnd++) *(uint4*)&sA[buf][a_row][a_col + rnd * 8] = ra[rnd];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const unsigned char* pk = (const unsigned char*)&rb[half];
            const unsigned char* sc = (const unsigned char*)&rs[half];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float sv = scl_fp8(sc[i]) * scale2;
                store_bf16_pair(&sB[buf][b_ns + half * 16 + i][b_kp * 2], sLUT[pk[i] & 0xF] * sv, sLUT[pk[i] >> 4] * sv);
            }
        }
    };
    auto mma = [&](int buf) {
        #pragma unroll
        for (int h = 0; h < 4; h++) {
            v16bf a; memcpy(&a, &sA[buf][warp_m_offset + (lane_id & 15)][h * 16], 32);
            #pragma unroll
            for (int nb = 0; nb < 8; nb++) {
                v16bf b; memcpy(&b, &sB[buf][nb * 16 + (lane_id & 15)][h * 16], 32);
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
            }
        }
    };
    uint4 ra[4], rb[2], rs[2];
    load(k_lo, ra, rb, rs); store(0, ra, rb, rs); __syncthreads();
    int cur = 0;
    for (unsigned int kb = k_lo + KS_K_STEP; kb < k_hi; kb += KS_K_STEP) {
        int nxt = 1 - cur;
        load(kb, ra, rb, rs); mma(cur); store(nxt, ra, rb, rs); __syncthreads(); cur = nxt;
    }
    mma(cur);
    float* Pz = P + (unsigned long long)blockIdx.z * M * N;
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4), c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) Pz[r * N + c] = acc[nb][e];
        }
}

extern "C" __global__ void splitk_reduce(const float* __restrict__ P, __nv_bfloat16* __restrict__ C, unsigned int MN, unsigned int S) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float s = 0.f;
    for (unsigned int z = 0; z < S; z++) s += P[(unsigned long long)z * MN + i];
    C[i] = __float2bfloat16(s);
}

static float bf(__nv_bfloat16 x) { unsigned short u; std::memcpy(&u, &x, 2); unsigned int v = (unsigned int)u << 16; float f; std::memcpy(&f, &v, 4); return f; }

int main() {
    struct Shape { const char* name; unsigned int M, N, K; } shapes[] = {
        {"ripple down", 17, 5120, 40960}, {"27b down", 9, 5120, 17408}, {"27b down K16", 17, 5120, 17408}};
    for (auto& sh : shapes) {
        const unsigned int M = sh.M, N = sh.N, K = sh.K;
        // Rotate several weight copies so the 32 MB MALL cannot hold them (DRAM-realistic).
        const size_t wbytes = (size_t)K / 2 * N + (size_t)K / 16 * N;
        const int copies = (int)std::max<size_t>(2, (512ull << 20) / wbytes + 1);
        std::vector<unsigned char*> Bp(copies), Bs(copies);
        std::vector<unsigned char> hp((size_t)K / 2 * N), hs((size_t)K / 16 * N);
        srand(1);
        for (auto& b : hp) b = rand() & 0xFF;
        for (auto& b : hs) b = 0x30 + (rand() & 0x0F);       // small positive e4m3 scales
        for (int c = 0; c < copies; c++) {
            CHECK(hipMalloc(&Bp[c], hp.size())); CHECK(hipMemcpy(Bp[c], hp.data(), hp.size(), hipMemcpyHostToDevice));
            CHECK(hipMalloc(&Bs[c], hs.size())); CHECK(hipMemcpy(Bs[c], hs.data(), hs.size(), hipMemcpyHostToDevice));
        }
        std::vector<__nv_bfloat16> hA((size_t)M * K);
        for (auto& a : hA) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *dA, *dC0, *dC1; float* dP;
        CHECK(hipMalloc(&dA, hA.size() * 2)); CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&dC0, (size_t)M * N * 2)); CHECK(hipMalloc(&dC1, (size_t)M * N * 2));
        CHECK(hipMalloc(&dP, (size_t)16 * M * N * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        const dim3 g0((N + N_TILE_LG - 1) / N_TILE_LG, (M + M_TILE - 1) / M_TILE, 1);
        auto run_base = [&](int c) { hipLaunchKernelGGL(w4a16_gemm_t_k64, g0, dim3(128), 0, 0, dA, Bp[c], Bs[c], 0.01f, dC0, M, N, K); };
        auto run_split = [&](int c, unsigned int S) {
            unsigned int slice = ((K / S + 63) / 64) * 64;
            unsigned int S_eff = (K + slice - 1) / slice;
            hipLaunchKernelGGL(w4a16_gemm_t_k64_splitk, dim3(g0.x, g0.y, S_eff), dim3(128), 0, 0, dA, Bp[c], Bs[c], 0.01f, dP, M, N, K, slice);
            hipLaunchKernelGGL(splitk_reduce, dim3((M * N + 255) / 256), dim3(256), 0, 0, dP, dC1, M * N, S_eff);
        };
        const int iters = 40;
        for (int i = 0; i < 3; i++) run_base(i % copies);
        CHECK(hipEventRecord(e0)); for (int i = 0; i < iters; i++) run_base(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
        float ms0; CHECK(hipEventElapsedTime(&ms0, e0, e1));
        const double gb = wbytes / 1e9;
        printf("-- %s M=%u N=%u K=%u (weights %.1f MB)\n   k64 (shipped)   %8.1f us  %6.1f GB/s  (reference)\n", sh.name, M, N, K, wbytes / 1e6, 1000 * ms0 / iters, gb / (ms0 / iters / 1000));
        std::vector<__nv_bfloat16> c0((size_t)M * N), c1((size_t)M * N);
        run_base(0); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(c0.data(), dC0, c0.size() * 2, hipMemcpyDeviceToHost));
        for (unsigned int S : {2u, 4u, 8u, 16u}) {
            for (int i = 0; i < 3; i++) run_split(i % copies, S);
            CHECK(hipEventRecord(e0)); for (int i = 0; i < iters; i++) run_split(i % copies, S); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            run_split(0, S); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(c1.data(), dC1, c1.size() * 2, hipMemcpyDeviceToHost));
            double num = 0, den = 0, mx = 0;
            for (size_t i = 0; i < c0.size(); i++) { double d = bf(c1[i]) - bf(c0[i]); num += d * d; den += (double)bf(c0[i]) * bf(c0[i]); mx = std::max(mx, std::fabs(d)); }
            printf("   splitK S=%-2u     %8.1f us  %6.1f GB/s  %.2fx  relL2 %.2e maxabs %.3e\n", S, 1000 * ms / iters, gb / (ms / iters / 1000), ms0 / ms, std::sqrt(num / (den + 1e-30)), mx);
        }
        for (int c = 0; c < copies; c++) { hipFree(Bp[c]); hipFree(Bs[c]); }
        hipFree(dA); hipFree(dC0); hipFree(dC1); hipFree(dP);
    }
    return 0;
}
