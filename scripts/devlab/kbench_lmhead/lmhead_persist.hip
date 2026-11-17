// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #196 (Flash-Next C8): the LM head (w4a16_gemv_batch8, N 248320, K 2560, M 8) takes 4.6 ms a step (~75 GB/s,
// 2.8% of C8). gb10's qwen4exp_w4_rows note: every 4-output CTA re-reads the M activation rows from L2 (here 62k CTAs
// x 40 KB = 2.5 GB a call). Twin: the SAME w4a16_gemv_batchm_impl body (generated from it), made persistent: a CTA walks
// output groups grid-stride, so A stays cache-resident. Per-output arithmetic unchanged -> BYTE-IDENTICAL.
#include "w4a16_gemv.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_persist(
    const __nv_bfloat16* __restrict__ A,         // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int C_stride
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
#if defined(__HIP_PLATFORM_AMD__)
    // Byte-indexed E2M1 pairs as packed bf16x2 (lo nibble -> .x, hi -> .y) so
    // the inner loop can feed `v_dot2_f32_bf16` (two MACs/instr) instead of
    // unpacking + two scalar FMAs per element — the M>=4 tiers are issue-bound
    // on that scalar chain, not bandwidth-bound. 256 entries = 1 KiB smem.
    // Truncating f32->bf16 is exact here: every E2M1 value fits bf16's
    // mantissa, so the products the dot2 computes are identical to the float
    // path's (only the 2-element summation grouping changes, which the verify
    // argmax tolerates).
    __shared__ unsigned int s_lut2[256];
    {
        unsigned int v = threadIdx.x;
        unsigned short lob = (unsigned short)(__float_as_uint(E2M1_LUT[v & 0xFu]) >> 16);
        unsigned short hib = (unsigned short)(__float_as_uint(E2M1_LUT[v >> 4]) >> 16);
        s_lut2[v] = ((unsigned int)hib << 16) | lob;
    }
#endif
    __syncthreads();

    for (unsigned int nb = blockIdx.x; nb * N_PER_BLOCK < N; nb += gridDim.x) {
    const unsigned int n = nb * N_PER_BLOCK + local_out;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // ONE accumulator array, reused by both phases — see the two-phase note in
    // the header. `phase` picks WHICH reference lane this thread is standing in
    // for; within a phase the thread runs a single reference accumulator chain.
    float acc[MAX_M];
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

        // Coalesced walk: at every step the 64 lanes read CONSECUTIVE chunks,
        // so a warp's 8-byte weight load covers 256 contiguous bytes and its
        // 16-byte activation loads are 32 B apart — the pre-fix access pattern.
        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {
            // 8 packed weight bytes (16 FP4) + 1 group scale → read and
            // unpacked ONCE for all M rows. This is the weight-DRAM saving
            // the tier exists for; it is independent of the FP order below.
            unsigned long long packed8 =
                *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            // UNSCALED E2M1 values. The group scale stays factored OUT of the
            // 16-FMA block and is applied once via fmaf(scale, part, acc),
            // exactly as `w4a16_gemv` does — pre-multiplying it into each
            // weight is a different FP32 expression.
#if defined(__HIP_PLATFORM_AMD__)
            // Byte-pair LUT: one LDS read per packed byte yields both weights as
            // a bf16x2, and the activation arrives already packed in ar[b] — so
            // the 16-element row product is 8 `v_dot2_f32_bf16` instead of 16
            // unpacks + 16 scalar FMAs. Product values identical to the float
            // path (bf16-exact E2M1 entries); only the intra-16 summation
            // grouping differs (2-term dots instead of a serial chain).
            // Inline asm, NOT __builtin_amdgcn_fdot2: the builtin resolves its
            // bf16x2 operands to llvm.amdgcn.fdot2(<2 x half>) — v_dot2_f32_F16 —
            // silently miscompiling bf16 pairs as fp16.
            unsigned int w2[8];
            #pragma unroll
            for (int b = 0; b < 8; b++)
                w2[b] = s_lut2[(unsigned char)(packed8 >> (b * 8))];

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0"
                                 : "+v"(part) : "v"(ar[b]), "v"(w2[b]));
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }
#else
            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];   // W[2j]   <-> act 2j
                wl[b * 2 + 1] = s_lut[byte_val >> 4];    // W[2j+1] <-> act 2j+1
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                // Per-chunk partial in the M=1 kernel's exact FMA order:
                // TWO separate fmaf per byte, never `part += x*w0 + y*w1`.
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    // BF16 -> FP32 is EXACT and is nothing but `bits << 16`
                    // (BF16 is the high half of the FP32 word), so this is a
                    // bit-for-bit substitution for
                    // `__bfloat1622float2(*(const __nv_bfloat162*)&ar[b])`,
                    // NOT a numeric change: the intrinsic lowers to
                    // `mov.b32 %f, {0, %h}`, which forces ptxas to first
                    // EXTRACT each 16-bit half into its own register (one
                    // PRMT apiece) and then re-assemble it. Doing the two
                    // shifts on the packed word the load already produced
                    // skips the extraction entirely: measured on sm_121f
                    // (`cuobjdump -sass`, --fmad=false) it removes all 64
                    // PRMT from batch8 (824 -> 760 static instructions,
                    // -7.8%) and all 32 from batch4 (472 -> 440, -6.8%) at
                    // identical registers (48), smem and occupancy. The
                    // template is issue-bound at M>=5, not DRAM-bound (see
                    // the tier table on `w4a16_gemv_batch5`), so instructions
                    // removed from this loop are the whole game.
                    const float ax = __uint_as_float(ar[b] << 16);
                    const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                    part = fmaf(ax, wl[b * 2], part);
                    part = fmaf(ay, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }
#endif
        }

        // Threads p and p^1 hold accumulator 0 and 1 of the SAME reference lane
        // (p/2 + 32*phase); fold them and land the result in virtual-lane order.
        // p adds acc0+acc1 and p^1 adds acc1+acc0 — the same FP32 value.
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];  // 2 warps/output, per row
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = s_vl[t][local_out][lane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * C_stride + n] = __float2bfloat16(r);
        }
    }
    __syncthreads();
    }
}

template <int MAX_M>
__global__ __launch_bounds__(BLOCK_SIZE, 5) void batch_persist(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2, __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {
    w4a16_gemv_batchm_persist<MAX_M>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

int main() {
    const unsigned N = 248320, K = 2560;
    const size_t pb = (size_t)N * K / 2, sb = (size_t)N * K / 16;
    std::vector<unsigned char> hp(pb), hs(sb); srand(7); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    unsigned char *dp, *ds; CHECK(hipMalloc(&dp, pb)); CHECK(hipMalloc(&ds, sb)); CHECK(hipMemcpy(dp, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMemcpy(ds, hs.data(), sb, hipMemcpyHostToDevice));
    for (unsigned M : {2u, 8u}) {
        std::vector<__nv_bfloat16> ha((size_t)M * K); for (auto& x : ha) x = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
        __nv_bfloat16 *dA, *dC[5]; CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (int v = 0; v < 5; v++) CHECK(hipMalloc(&dC[v], (size_t)M * N * 2));
        const unsigned grids[5] = {0, 160, 320, 640, 1280};
        auto launch = [&](int v) {
            if (v == 0) { if (M == 8) hipLaunchKernelGGL(w4a16_gemv_batch8, dim3(N / 4), dim3(256), 0, 0, dA, dp, ds, 0.01f, dC[0], M, N, K);
                          else hipLaunchKernelGGL(w4a16_gemv_batch2, dim3(N / 4), dim3(256), 0, 0, dA, dp, ds, 0.01f, dC[0], N, K); }
            else { if (M == 8) hipLaunchKernelGGL(batch_persist<8>, dim3(grids[v]), dim3(256), 0, 0, dA, dp, ds, 0.01f, dC[v], M, N, K);
                   else hipLaunchKernelGGL(batch_persist<2>, dim3(grids[v]), dim3(256), 0, 0, dA, dp, ds, 0.01f, dC[v], M, N, K); } };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[5];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 5; v++) { for (int i = 0; i < 2; i++) launch(v); CHECK(hipEventRecord(e0)); for (int i = 0; i < 10; i++) launch(v); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 10; }
        std::vector<__nv_bfloat16> c0((size_t)M * N), c1((size_t)M * N); CHECK(hipMemcpy(c0.data(), dC[0], c0.size() * 2, hipMemcpyDeviceToHost));
        const double gb = (double)(pb + sb) / 1e3;
        printf("-- LM head N=%u K=%u M=%u: ship %7.1f us (%.0f GB/s)", N, K, M, t[0], gb / t[0]);
        for (int v = 1; v < 5; v++) { CHECK(hipMemcpy(c1.data(), dC[v], c1.size() * 2, hipMemcpyDeviceToHost));
            printf(" | persist g%u %7.1f x%.2f %s", grids[v], t[v], t[0] / t[v], memcmp(c0.data(), c1.data(), c0.size() * 2) ? "DIFF" : "same"); }
        printf("\n");
        CHECK(hipFree(dA)); for (int v = 0; v < 5; v++) CHECK(hipFree(dC[v]));
    }
    return 0;
}
