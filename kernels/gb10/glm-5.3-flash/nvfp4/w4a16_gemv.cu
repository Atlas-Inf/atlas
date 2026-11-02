// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV — Fused NVFP4 weight dequant + BF16 GEMV for M=1 decode.
//
// out[n] = dot(A[0,:], dequant(B_fp4[n,:]))
//
// Specialized for M=1 decode: replaces w4a16_gemm which wastes ~98% of
// threads at M=1 with 64x64 tiles + MMA tensor cores (MMA requires M>=16).
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight format (HuggingFace/compressed-tensors):
//   B_packed: [N, K/2] uint8 — byte at [n, j] holds W[n, 2j] (low) and W[n, 2j+1] (high)
//   B_scale:  [N, K/GROUP_SIZE] FP8-E4M3 — one scale per group of 16 K-dim values
//   scale2:   scalar FP32 — per-tensor second-level scale
//
// K-dim packing: each byte holds 2 consecutive input features for the same output.
// Vectorized reads of 4 bytes = 8 weight values, coalesced across warps.
//
// 4 outputs per block, 64 threads (2 warps) per output. Cross-warp smem reduction.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format which mismatches the
// standard E4M3 scales written by the encoder -> corrupts every block scale.
// HIP/gfx1151 shares the same software path (no cvt.rn.satfinite.e4m3x2.f32 PTX).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

// E2M1 lookup table (same as w4a16_gemm.cu)
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Stage the E2M1 table in SHARED memory before any decode inner loop ──
// The dequant indexes the table with a DATA-DEPENDENT weight nibble. A read of
// __constant__ memory SERIALIZES across a warp whenever the lanes hit divergent
// entries: the constant cache is a BROADCAST cache and replays once per distinct
// address, and 32 lanes drawing NVFP4 nibbles cover 16*(1-(15/16)^32) ~= 14 of
// the 16 entries, so one LUT read costs ~14 transactions. Shared memory services
// divergent indices in parallel — the 16 entries sit in 16 distinct banks and
// same-index lanes multicast, so the read is one conflict-free transaction.
// Same lever, same reasoning as `w8a16_gemm_pipelined.cu` (measured 5x there).
//
// This is NOT a numerical change: `s_lut[i]` is a bit-exact FP32 copy of
// `E2M1_LUT[i]`, so every fmaf sees the identical operand it saw before.
//
// ── HIP / SCALE portability ──────────────────────────────────────────────────
// This file is SYMLINKED into kernels/strix-hip/common/ and kernels/strix/common/,
// so hipcc compiles it verbatim. `__syncwarp()` is a CUDA intrinsic that target
// does not accept: the strix-hip fork of w4a16_gemm.cu contains zero
// `__syncwarp` and the tree carries no shim. Introducing it unguarded broke the
// windows-x86_64-amd-hip release build (#519).
//
// NOTE the block-wide staging elsewhere in this file (`__shared__ s_lut` +
// `__syncthreads()`, 9 pre-existing sites) is FINE on HIP and is untouched.
// Only the WARP-scoped publish is CUDA-only.
//
// On AMD the warp-scoped path keeps the PRE-EXISTING behaviour: no staging, the
// partial indexes `__constant__ E2M1_LUT` directly, exactly as shipped before
// this change. Provably compilable (it is the shipped code) and provably
// correct; it forgoes the staging win on a target we cannot measure. Doing
// better needs a wave-scoped publish (`__builtin_amdgcn_wave_barrier()` plus
// the right fence) VALIDATED on gfx1151 — no such box exists here, so it is not
// guessed at, for the same reason the strix-hip batched-prefill argument shift
// was left to someone with the hardware.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define ATLAS_WARP_LUT_STAGED 0
#else
#define ATLAS_WARP_LUT_STAGED 1
#endif

// WARP-SCOPED variant, for the single-warp kernels: they early-return on a
// warp-uniform `n >= N`, so a block-wide __syncthreads() here would be a
// divergent barrier. Lanes 0..15 fill their own warp's copy; __syncwarp()
// publishes it. The reduction below stays barrier-free, as documented.
__device__ __forceinline__ void stage_e2m1_lut_warp(float* s_lut, unsigned int lane) {
#if ATLAS_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT[lane];
    __syncwarp();
#else
    (void)s_lut; (void)lane;   // AMD: nothing staged; callers pass the constant LUT
#endif
}

#if defined(__HIP_PLATFORM_AMD__)
// Block-level staging of the byte-indexed E2M1 bf16x2 pair table used by the
// AMD `v_dot2_f32_bf16` inner loops: `s_lut2[v]` packs (E2M1[v&0xF],
// E2M1[v>>4]) — low nibble in the low half — so one LDS read per packed byte
// feeds a dot2 covering BOTH nibbles. 256 entries = 1 KiB; every thread writes
// one entry, so the call must precede a `__syncthreads()` that covers all 256
// threads. Truncating f32->bf16 is exact: every E2M1 value fits bf16.
__device__ __forceinline__ void stage_e2m1_lut2_block(unsigned int* s_lut2) {
    unsigned int v = threadIdx.x;
    unsigned short lob = (unsigned short)(__float_as_uint(E2M1_LUT[v & 0xFu]) >> 16);
    unsigned short hib = (unsigned short)(__float_as_uint(E2M1_LUT[v >> 4]) >> 16);
    s_lut2[v] = ((unsigned int)hib << 16) | lob;
}
#endif

// W4A16 GEMV: C[n] = sum_k A[k] * dequant(B_fp4[n, k])
//
// Vectorized: 16 K-values per chunk (2× uint4 activation + uint64 weight),
// TWO chunks in flight per iteration with independent accumulators. Keeping 2
// outstanding weight loads per thread hides DRAM latency (ncu: 72% of warp
// stalls are long-scoreboard on GB10). The FP8 group scale is factored out of
// the inner 16-FMA block (exact regroup: sum(s*w*a) == s*sum(w*a)).
//
// Coalescing: within a warp, consecutive threads read consecutive weight and
// activation chunks. Perfectly coalesced.
//
// SSOT: `w4a16_gemv` and `w4a16_gemv_sw` share `w4a16_gemv_partial`. A previous
// SW copy used the older stride-64 sequential `acc += a*w` loop and was 1 ULP
// lossy vs this pipelined association (GB10 oracle: gdn in_proj / K-tail).

// One orig-lane's pipelined K16 partial. `orig_lane` is the 64-thread kernel's
// lane (0..63): k16 = orig_lane*2, stride 128, inner c={0,1} → kk pair.
__device__ __forceinline__ float w4a16_gemv_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut,   // shared-staged E2M1_LUT copy; see stage_e2m1_lut_warp
    const unsigned int* __restrict__ lut2)  // AMD-only: byte-indexed bf16x2 E2M1 pairs
{
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int stride2 = 128u; // threads_per_out (64) * 2
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            uint4 a_lo = ((const uint4*)A)[kk * 2];
            uint4 a_hi = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float part = 0.0f;
#if defined(__HIP_PLATFORM_AMD__)
            // v_dot2_f32_bf16: 8 dot2s replace 16 unpacks + 16 scalar FMAs.
            // lut2[byte] packs (E2M1[lo], E2M1[hi]) as bf16x2 — the products
            // match the float path exactly; only the intra-16 grouping
            // changes. Asm, not __builtin_amdgcn_fdot2 (miscompiles bf16->fp16).
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                const unsigned int w2 = lut2[(unsigned char)(packed8 >> (b * 8))];
                asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0"
                             : "+v"(part) : "v"(a_raw[b]), "v"(w2));
            }
#else
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, lut[byte_val & 0xF], part);
                part = fmaf(af.y, lut[byte_val >> 4], part);
            }
#endif
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    return acc0 + acc1;
}

extern "C" __global__ void w4a16_gemv(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];  // cross-warp reduction
#if defined(__HIP_PLATFORM_AMD__)
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
#else
    const unsigned int* s_lut2 = nullptr;
#endif
    // Block-staged here (all 256 threads reach this barrier — no early return).
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    // Do not return early: N%4!=0 tail warps must still hit the smem barrier.
    float acc = 0.0f;
    if (n < N) {
        acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, s_lut, s_lut2);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0 && n < N) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV — SINGLE-WARP-PER-OUTPUT variant (lossless; default ON, kill ATLAS_NO_GEMV_SW=1).
//
// Bit-identical to w4a16_gemv: same `w4a16_gemv_partial` per orig-lane. 32
// threads (1 warp) per output instead of 64 (2 warps). 8 outputs per 256-thread
// block (was 4). The cross-warp __syncthreads() + smem round-trip is gone —
// the final combine is one FP32 add of two warp-shuffle results.
//
// BIT-IDENTICALITY: orig warp A is lanes 0..31, warp B is 32..63. Each SW lane
// holds those two orig-lane partials:
//   acc_a[lane] == orig acc[lane]     (k16 = lane*2,     stride 128)
//   acc_b[lane] == orig acc[lane+32]  (k16 = (lane+32)*2, stride 128)
// Shuffle-reduce each in the same tree, then `reduced_a + reduced_b` ==
// smem[0]+smem[1]. Grid: (ceil(N / 8), 1, 1)   Block: (256, 1, 1)

#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_sw(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;       // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;            // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;

#if defined(__HIP_PLATFORM_AMD__)
    // s_lut2 staging + barrier MUST precede the warp-uniform early return:
    // shared by the whole block, so every thread writes one entry and all 256
    // reach the sync before any returns.
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
    __syncthreads();
#else
    const unsigned int* s_lut2 = nullptr;
#endif
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // One private copy per warp (8 warps x 16 floats = 512 B/block): the block
    // barrier is unavailable here (warp-uniform early return above).
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif

    // acc_a reproduces orig lane `lane` (warp A); acc_b reproduces orig lane
    // `lane+32` (warp B). Same operands, same order as the 64-thread kernel.
    float acc_a = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut, s_lut2);
    float acc_b = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut, s_lut2);

    // Reduce each accumulator within the warp in the SAME tree order as orig.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    // lane 0 holds reduced warp-A (acc_a) and warp-B (acc_b). Final combine ==
    // smem[0] + smem[1] in the 64-thread kernel. Bit-identical.
    if (lane == 0) {
        float result = acc_a + acc_b;
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with FP32 output (for LM head logits).
// Identical to w4a16_gemv but writes float instead of BF16.
// FP32 logits are critical for sampling quality — BF16 collapses
// similar logit values, making stochastic sampling random.
// ============================================================
extern "C" __global__ void w4a16_gemv_logits(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,  // FP32 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

#if defined(__HIP_PLATFORM_AMD__)
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
    __syncthreads();
#endif
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
#if defined(__HIP_PLATFORM_AMD__)
        // fdot2 with the group scale factored out of the 16-dot block
        // (acc = fmaf(scale, part, acc)) — replaces 16 unpacks + 16
        // scale-premultiplied adds. FP32 logits feed argmax; the regrouped
        // summation is within ~1e-4 and cannot flip a non-degenerate argmax.
        float part = 0.0f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const unsigned int w2 = s_lut2[(unsigned char)(packed8 >> (b * 8))];
            asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0"
                         : "+v"(part) : "v"(a_raw[b]), "v"(w2));
        }
        acc = fmaf(scale, part, acc);
#else
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
#endif
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        C[n] = smem[local_out * 2] + smem[local_out * 2 + 1]; // FP32 output!
    }
}

// ============================================================
// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs
// ============================================================
// For K=2 speculative verification: processes 2 input vectors through
// the same weight matrix in a single pass. Eliminates the GEMM M=2
// tile waste (64x64 tiles at 3% M-utilization).
//
// A: [2, K] BF16 contiguous (row 0 and row 1)
// B: [N, K/2] NVFP4 packed weights
// C: [2, N] BF16 contiguous (row 0 and row 1)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 2x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
// `w4a16_gemv_batch2` is defined below, next to batch4/8/16/32, because it is
// now a thin instantiation of the shared `w4a16_gemv_batchm_impl` template.
// Its former standalone body carried the same three bit-parity divergences
// from `w4a16_gemv` documented on that template — see the note there.

// ============================================================
// W4A16 batched GEMV (M<=MAX_M) — the NVFP4 sibling of w8a16_gemv_batch4/16.
// ============================================================
// At M-token batched decode the SSM QKVZ / out_proj projections share the same
// NVFP4 weight matrix across all M sequences, so a SINGLE DRAM pass over the
// packed 4-bit weight (dequantized E2M1*scale ONCE) serves all M rows — MAC'd
// into M independent FP32 accumulators. This is what lets FP4 amortize the
// weight read the way w8a16_gemv_batch4/16 does for FP8; without it the FP4
// multi-seq path capped at batch3 and re-streamed the weight ~3x at C=8.
//
// Per-row accumulation order is IDENTICAL to `w4a16_gemv` (M=1), so the output
// is bit-identical to running w4a16_gemv M times.
//
// That claim was FALSE until 2026-08-12 and is now enforced by
// `crates/spark-model/examples/w4a16_batch_bitparity_microtest.rs`. The body
// below had THREE independent divergences from `w4a16_gemv`, each of which
// alone breaks bit-parity under the dir's `--fmad=false`:
//
//   1. K-chunk → lane mapping. `w4a16_gemv` walks `k16 = lane*2` with stride
//      128 in PAIRS (c = 0,1) into two accumulators; this walked `k16 = lane`
//      with stride 64 into one. Different lane partitions ⇒ different
//      partial sums ⇒ a different shuffle-reduction tree.
//   2. Block scale placement. `w4a16_gemv` factors the FP8 group scale OUT of
//      the 16-FMA block (`acc = fmaf(scale, part, acc)`); this pre-multiplied
//      it into every unpacked weight (`lut * scale`). Exact in real
//      arithmetic, NOT in FP32.
//   3. Fused vs split accumulation — the same defect found in
//      `w8a16_gemv_batch4.cu`: `acc += x*w0 + y*w1` associates as
//      `acc + (x·w0 + y·w1)`, where the M=1 kernel computes
//      `(acc + x·w0) + y·w1`.
//
// Measured at the production Nemotron projection shapes before the fix
// ([10304x2688], [2688x4096], [18560x4096], [4096x8192], seeds 1/99/12345):
// 178 of 180 legs differed, 1..62 BF16 elements per launch, max|delta| 0.0625.
// The body now reproduces `w4a16_gemv` FMA-for-FMA; the weight DRAM read, the
// packed8 unpack and the scale decode are still done ONCE per chunk for all M
// rows, which is the entire point of the tier.
//
// ── VIRTUAL-LANE REMAP (why this does not walk K the way `w4a16_gemv` does) ──
//
// The literal transcription — physical thread `l` walks `k16 = l*2` with stride
// 128 in pairs, exactly like the M=1 kernel — is bit-exact but 45% SLOWER than
// the (wrong) body it replaced, and NOT because of the second accumulator
// array. It is coalescing. Under that mapping a warp's 16-byte activation load
// has its lanes 64 B apart, so one instruction touches 2048 B to use 512 — half
// the sector efficiency of the stride-32 walk, and the activation stream is
// M times the weight stream, which is why the cost grew with M (batch2 +13%,
// batch16 +49%). Priced directly: holding the pair mapping and both
// accumulators but addressing activations with the stride-32 pattern lands on
// baseline (-0.4%); widening the WEIGHT load to 16 B recovers nothing (+44%).
//
// So this walks K COALESCED — consecutive lanes read consecutive chunks, the
// pattern the previous body used — and recovers bit-parity by remapping which
// REFERENCE accumulator each physical thread stands in for. Write p = physical
// thread (0..63) and recall that the M=1 kernel's lane l owns chunks
// {2l + c + 128j} in accumulator c. Then, for either value of `phase`,
//
//   chunks {p + 64*phase + 128j} == 2*(p/2 + 32*phase) + (p%2) + 128j
//                                == ref lane (p/2 + 32*phase), accumulator p%2
//
// — the same chunk set in the same order, so the same fmaf chain, so a partial
// that is bit-identical to the reference one it stands for. The kernel runs
// that chain TWICE, phase 0 then phase 1, with ONE accumulator array reused;
// the two phases read DISJOINT halves of the weight row, so total DRAM traffic
// is unchanged (each pass is a stride-1024 B walk of contiguous 512 B runs) and
// the register cost stays at the pre-fix `acc[MAX_M]`. Carrying both chains at
// once instead costs a second array and ~20% at batch8/16, measured.
//
// `acc_l = acc0_l + acc1_l` is then one `__shfl_xor_sync(...,1)` between the
// adjacent threads p and p^1, which hold accumulator 0 and 1 of the SAME
// reference lane and are always in the same warp. p and p^1 add in opposite
// operand order, which is exact — FP32 addition is commutative.
//
// That leaves acc_l for l = 0..63 on the even threads, so `s_vl` re-lands it in
// virtual-lane order and the UNCHANGED two-warp shuffle tree below reduces
// lanes 0..31 and 32..63 exactly as `w4a16_gemv` does. Cost: one shfl per row
// per phase and a 64-float smem round trip per (row, output) — not per chunk.
//
// Register cost lands at or BELOW the pre-fix body — batch16 56 vs 70 and
// batch32 80 vs 106, so 4 and 3 CTA/SM where the pre-fix body got 3 and 2 —
// and no `__launch_bounds__` is wanted on the WIDE tiers (16/32): pinning them
// all measured 12.70 ms against 11.49 ms free. The narrow M=5..8 tiers are the
// exception and are pinned to 5 CTA/SM; the register table and the reasoning
// are on `w4a16_gemv_batch5`.
//
// A:[M,K] BF16, B_packed:[N,K/2], B_scale:[N,K/16] FP8-E4M3, scale2 FP32,
// C:[M,N] BF16. Grid: (ceil(N/4),1,1) Block: (256,1,1).
template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_impl(
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
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

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

    if (n >= N) return;

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
}

// M==2 (K=2 spec verify, and C=2 multi-seq decode: dense_ffn, qwen3 qkv/o_proj).
// Fixed-M signature (no `M` argument) — kept for its existing call sites.
//
// This used to be a hand-written standalone kernel that walked K in 8-value
// chunks with ONE accumulator per row and the group scale pre-multiplied into
// each weight. It therefore carried all three divergences documented on
// `w4a16_gemv_batchm_impl` above and was never byte-identical to 2 x
// `w4a16_gemv`. Routing it through the (now bit-exact) template fixes it by
// construction and deletes ~100 lines of duplicated dequant/reduce code. The
// template also reads 16 K-values per chunk with two chunks in flight, where
// the old body read 8 with one, so the weight loads get WIDER, not narrower.
extern "C" __global__ void w4a16_gemv_batch2(
    const __nv_bfloat16* __restrict__ A,          // [2, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [2, N]
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<2>(A, B_packed, B_scale, scale2, C, 2u, N, K, N);
}

// M==3 (K=3 spec verify, C=3 multi-seq decode, MoE forward_k3). Same story as
// batch2 above: formerly a standalone body with the same three divergences,
// plus the fused `acc += x*w_lo + y*w_hi` form verbatim.
extern "C" __global__ void w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ A,          // [3, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [3, N]
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<3>(A, B_packed, B_scale, scale2, C, 3u, N, K, N);
}

// M<=4 (common-path batched decode) — sibling of w8a16_gemv_batch4.
extern "C" __global__ void w4a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// ── EXACT-M TIERS 5/6/7 ─────────────────────────────────────────────────────
// Why they exist. `MAX_M` sizes `acc[]`, `s_vl[]` and `smem[]`, and — because
// the row loop is `#pragma unroll`ed — it also sizes the CODE. The `t >= M`
// guard skips the WORK of a dead row at run time but not its instructions:
// running M=5 on the MAX_M=8 tier still carries three dead unrolled row
// blocks through the hot loop. Measured on the real 27B qkv/o shape
// (N=5120 K=5120, cold-cycled weights, 273 GB/s peak) BEFORE this change:
//
//   tier    M    time      eff BW     % peak
//   batch4  4    70.5 us   209.2 GB/s  76.6%
//   batch8  4    74.8 us   197.3 GB/s  72.3%   <- same M, same work, +6.1%
//   batch8  5    89.0 us   165.7 GB/s  60.7%
//   batch8  6    91.5 us   161.2 GB/s  59.0%
//   batch8  8   106.4 us   138.5 GB/s  50.7%
//   batch16 8   113.3 us   130.1 GB/s  47.7%
//
// The batch8-at-M=4 row is the whole argument: identical rows, identical
// weight stream, +6.1%. It is NOT the `__launch_bounds__` — batch4 lands on
// 48 registers / 5 CTA with no pragma at all, exactly what the pragma pins
// batch8 to, so the two tiers run at the SAME occupancy. What differs is the
// unrolled body: 824 vs 472 static SASS instructions before the conversion
// fix above, 760 vs 440 after. Sizing the tier to the real row count is the
// only way to stop paying for rows that are not there.
//
// Occupancy. `__launch_bounds__(BLOCK_SIZE, 5)` on 5/6/7 for the same reason
// it is on 8, and it is free here — measured with `ptxas -v` on sm_121f:
//
//   MAX_M   free regs / CTA-per-SM      pinned to 5 CTA
//   5       48 / 5   (already 5)        48, no spill
//   6       55 / 4                      48, no spill
//   7       56 / 4                      48, no spill
//   8       56 / 4                      48, no spill  <- the existing pin
//
// i.e. 6 and 7 would otherwise DROP to 4 CTA/SM and lose the thing the tier
// was added to win. 48 is the ceiling at 5 CTA (5 x 256 x 48 = 61440 of the
// SM's 65536 registers; the next allocation step, 56, only fits 4 CTAs), and
// asking for 6 CTA spills 32 B at MAX_M=8 and 8 B at MAX_M=5.
//
// Dispatch picks the narrowest RESOLVED tier >= M
// (`spark-model/src/layers/w4a16_gemv_tiers.rs`); `ATLAS_NO_GEMV_EXACT_M_TIERS=1`
// hides 5/6/7 and restores the batch4/batch8-only decision.
//
// Bit-parity: same template, same `MAX_M`-independent FMA chain per row, so
// tier 5/6/7 output is bit-identical to tier 8 (and to M x `w4a16_gemv`) at
// the same M — enforced by `w4a16_batch_bitparity_microtest`.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// GLM KDA Q/K/V verifier projection. Grid Z chooses one of three independent
// native-NVFP4 weights while each plane executes the proven exact-M=5 body
// unchanged. This removes two graph nodes per KDA layer; it does not combine
// or reassociate any dot product.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5_qkv(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bq_packed,
    const unsigned char* __restrict__ Bq_scale,
    const float q_scale2,
    const unsigned char* __restrict__ Bk_packed,
    const unsigned char* __restrict__ Bk_scale,
    const float k_scale2,
    const unsigned char* __restrict__ Bv_packed,
    const unsigned char* __restrict__ Bv_scale,
    const float v_scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int plane = blockIdx.z;
    const unsigned char* B_packed = plane == 0u ? Bq_packed : (plane == 1u ? Bk_packed : Bv_packed);
    const unsigned char* B_scale = plane == 0u ? Bq_scale : (plane == 1u ? Bk_scale : Bv_scale);
    const float scale2 = plane == 0u ? q_scale2 : (plane == 1u ? k_scale2 : v_scale2);
    __nv_bfloat16* out = C + (unsigned long long)plane * M * N;
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, out, M, N, K, N);
}

// PDL twins of `w4a16_gemv_batch2/3` and `w4a16_gemv_batch5_qkv` with the
// pre-wait weight touch of atlas_pdl_touch.cuh (ATLAS_GLM_DECODE_GEMV_BATCH): the
// first `touch_ctas` CTAs (of plane 0) pull the first `touch_rows` rows of the
// (first) weight into L2 while the kernel waits on its predecessor. Each body
// is the template instance of the kernel it replaces (the batch2 twin keeps
// batch2's 40-register MAX_M=2 build), so rows are bit-identical to it. `M` is
// in the batch2/3 twins' ABI only to share one launcher; it must be 2 / 3.
extern "C" __global__ void w4a16_gemv_batch2_touch(
    const __nv_bfloat16* __restrict__ A,          // [2, K]
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [2, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int touch_rows,
    unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({B_packed, K / 2u, K / 2u}, {B_scale, K / GROUP_SIZE, K / GROUP_SIZE},
                          touch_rows, blockIdx.x, touch_ctas);
    w4a16_gemv_batchm_impl<2>(A, B_packed, B_scale, scale2, C, 2u, N, K, N);
}

extern "C" __global__ void w4a16_gemv_batch3_touch(
    const __nv_bfloat16* __restrict__ A,          // [3, K]
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [3, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int touch_rows,
    unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({B_packed, K / 2u, K / 2u}, {B_scale, K / GROUP_SIZE, K / GROUP_SIZE},
                          touch_rows, blockIdx.x, touch_ctas);
    w4a16_gemv_batchm_impl<3>(A, B_packed, B_scale, scale2, C, 3u, N, K, N);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5_qkv_touch(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ Bq_packed,
    const unsigned char* __restrict__ Bq_scale,
    const float q_scale2,
    const unsigned char* __restrict__ Bk_packed,
    const unsigned char* __restrict__ Bk_scale,
    const float k_scale2,
    const unsigned char* __restrict__ Bv_packed,
    const unsigned char* __restrict__ Bv_scale,
    const float v_scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int touch_rows,
    unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({Bq_packed, K / 2u, K / 2u}, {Bq_scale, K / GROUP_SIZE, K / GROUP_SIZE},
                          touch_rows, blockIdx.x, blockIdx.z == 0u ? touch_ctas : 0u);
    const unsigned int plane = blockIdx.z;
    const unsigned char* B_packed = plane == 0u ? Bq_packed : (plane == 1u ? Bk_packed : Bv_packed);
    const unsigned char* B_scale = plane == 0u ? Bq_scale : (plane == 1u ? Bk_scale : Bv_scale);
    const float scale2 = plane == 0u ? q_scale2 : (plane == 1u ? k_scale2 : v_scale2);
    __nv_bfloat16* out = C + (unsigned long long)plane * M * N;
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, out, M, N, K, N);
}

// GLM shared-expert gate/up verifier projections. Grid Z chooses one of two
// independent native-NVFP4 weights and output planes. Each plane executes the
// exact-M=5 implementation unchanged, so this removes one launch without
// changing a dot product, scale application, or BF16 rounding point.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5_dual(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B0_packed,
    const unsigned char* __restrict__ B0_scale,
    const float b0_scale2,
    const unsigned char* __restrict__ B1_packed,
    const unsigned char* __restrict__ B1_scale,
    const float b1_scale2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const bool second = blockIdx.z != 0u;
    const unsigned char* B_packed = second ? B1_packed : B0_packed;
    const unsigned char* B_scale = second ? B1_scale : B0_scale;
    const float scale2 = second ? b1_scale2 : b0_scale2;
    __nv_bfloat16* out = second ? C1 : C0;
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, out, M, N, K, N);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch6(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<6>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch7(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<7>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// M<=8 (chain-verify K=5..8) — keeps M=5..8 projections on the
// weight-streaming batched GEMV instead of falling to the tile GEMMs.
// batch16's acc[16] + smem[16][8] register/smem pressure is halved at
// MAX_M=8; per-row accumulation order is identical to batch4/batch16
// (same template, M-guarded), so output is bit-identical at matching M.
//
// The ONE tier that wants a `__launch_bounds__`: free allocation gives it 56
// registers / 4 CTA per SM, and pinning it to 48 / 5 measured 2.166 ms against
// 2.261 ms over the gate's M=5..8 legs (6 CTA is worse again, 2.437 ms). It is
// also the only tier still slower than the pre-fix body (1.969 ms, +10%);
// batch2/3/4/16/32 are all at or under it.
//
// DROPPING the pragma was re-examined 2026-08-17 and REJECTED: it does not
// reproduce batch4's codegen (batch4 is 48/5 for free), it produces the
// 56-register / 4-CTA variant the line above already measured as the slower
// of the two. The tiers above carry the same pin for the same reason.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<8>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// Strided-output variants (`_os`): same bit-exact `w4a16_gemv_batchm_impl`
// body, but output row `t` lands at `C[t*C_stride + n]` instead of
// `C[t*N + n]`. The K=4..8 attention QKV verify path writes each projection
// straight into its interleaved `qkv_buf` slice with these — removing the
// 3·M D2D scatter copies per attention layer. The contiguous entry points
// pass C_stride=N, so they are unchanged.
extern "C" __global__ void w4a16_gemv_batch4_os(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int C_stride
) {
    w4a16_gemv_batchm_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K, C_stride);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch8_os(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int C_stride
) {
    w4a16_gemv_batchm_impl<8>(A, B_packed, B_scale, scale2, C, M, N, K, C_stride);
}

// M<=16 (high-concurrency decode, n=5..16) — sibling of w8a16_gemv_batch16.
extern "C" __global__ void w4a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<16>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// M<=32 (native-bs32 batched propose, n=17..32) — same M-guarded template.
// Exists so the n=32 MTP re-propose reads the drafter LM head ONCE instead
// of two chunked batch16 sweeps (the measured 2 x 13.5 ms propose in the
// wave-10 K=2 step timing). acc[32] + smem[32][8] doubles batch16's
// per-thread state; the kernel is DRAM-bound on the shared weight read at
// grid ceil(N/4), so occupancy loss is secondary to halving weight traffic.
extern "C" __global__ void w4a16_gemv_batch32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<32>(A, B_packed, B_scale, scale2, C, M, N, K, N);
}

// ============================================================
// W4A16 GEMV with inline Q/Gate deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes Q and Gate to separate halves.
// Eliminates the separate deinterleave_qg kernel (saves 12 graph nodes).
//
// Input layout (interleaved per head): [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
// Output layout (deinterleaved): [Q_h0..Q_nh | G_h0..G_nh]
//
// N = num_heads * head_dim * 2  (total Q+Gate elements)
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q | G] deinterleaved
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
#if defined(__HIP_PLATFORM_AMD__)
    // Stage + barrier before the warp-uniform early return: s_lut2 is shared
    // by the whole block, so exited threads would leave unwritten entries.
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
    __syncthreads();
#endif
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

#if defined(__HIP_PLATFORM_AMD__)
    float acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K,
                                   num_groups, K / 16, lane, s_lut, s_lut2);
#else
    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }
#endif

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        // head = n / (2 * head_dim), is_gate = (n % (2 * head_dim)) >= head_dim
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;             // Q region
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);  // Gate region
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with inline QKVZ deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes to deinterleaved output locations.
// Eliminates the separate deinterleave_qkvz kernel (saves 36 graph nodes).
//
// QKVZ interleaved layout (N=12288, 16 groups of 768):
//   Group g: [Q_{g*128..128} | K_{g*128..128} | V_{g*256..256} | Z_{g*256..256}]
//
// Deinterleaved output: [Q_2048 | K_2048 | V_4096 | Z_4096]
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qkvz(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q|K|V|Z] deinterleaved
    unsigned int N,
    unsigned int K,
    // Deinterleave params:
    unsigned int num_groups,        // 16
    unsigned int head_k_dim,        // 128
    unsigned int vheads_per_group,  // 2
    unsigned int head_v_dim         // 128
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
#if defined(__HIP_PLATFORM_AMD__)
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
    __syncthreads();
#endif
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups_k = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

#if defined(__HIP_PLATFORM_AMD__)
    // Same pipelined partial as w4a16_gemv (scale factored out of the 16-FMA
    // block, two chunks in flight). gfx1151 measured ~5x over the k8 loop.
    float acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K,
                                   num_groups_k, K / 16, lane, s_lut, s_lut2);
#else
    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups_k + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }
#endif

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Compute deinterleaved output index
        unsigned int v_group_size = vheads_per_group * head_v_dim;
        unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
        unsigned int g = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_groups * head_k_dim;
        unsigned int k_total = num_groups * head_k_dim;

        unsigned int out_idx;
        if (idx < head_k_dim) {
            out_idx = g * head_k_dim + idx;
        } else if (idx < 2 * head_k_dim) {
            out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
        } else if (idx < 2 * head_k_dim + v_group_size) {
            out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
        } else {
            out_idx = q_total + k_total + num_groups * v_group_size
                    + g * v_group_size + (idx - 2 * head_k_dim - v_group_size);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV batch2 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch2 (2-input) with w4a16_gemv_qg (deinterleave).
// Reads Q+Gate weight matrix once for 2 input tokens, produces 2 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[2, K] BF16 (2 token hidden states)
// Output: C[2, N] BF16 (deinterleaved: C[0] = [Q0|G0], C[1] = [Q1|G1])
//
// FIFTH FAMILY, now fixed. The reference here is `w4a16_gemv_qg` (M=1), NOT
// `w4a16_gemv` — qg/qkvz have their own k8=lane chunking and pre-multiplied
// scale. Against that reference all four fused batched variants
// (`w4a16_gemv_qg_batch2`, `w4a16_gemv_dual_batch2`, `w4a16_gemv_qg_batch3`,
// `w4a16_gemv_dual_batch3`) diverged in exactly ONE way, defect 3 from the
// `w4a16_gemv_batchm_impl` note above: they computed
// `acc += x*w_lo + y*w_hi` where `w4a16_gemv_qg`/`w4a16_gemv_qkvz` do two
// separate `acc +=`. Chunking and scale placement already MATCHED, so the fix
// is a one-line split per accumulator — no lane remap, no register cost, and
// `ptxas -v` is unchanged at 40 registers / 6 CTAs per SM for all four.
// `w4a16_qg_batch_bitparity_microtest.rs` is the standing gate; the `dual`
// pair is compared against `w4a16_gemv_qg` driven with num_heads=1,
// head_dim=N, which makes its deinterleave the identity map.
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    __nv_bfloat16* __restrict__ C1 = C + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 GEMV dual batch2: K+V for 2 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 2 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 2 outputs.
//
// Input:  A[2, K_in] BF16 (2 token hidden states)
// Output: C[2, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch2(
    const __nv_bfloat16* __restrict__ A,         // [2, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [2, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [2, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    __nv_bfloat16* C_out1 = C_out + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs
// ============================================================
// For K=3 speculative verification: processes 3 input vectors through
// the same weight matrix in a single pass.
//
// A: [3, K] BF16 contiguous (row 0, 1, 2)
// B: [N, K/2] NVFP4 packed weights
// C: [3, N] BF16 contiguous (row 0, 1, 2)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 3x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
// `w4a16_gemv_batch3` is defined above, next to batch4/8/16/32, because it is
// now a thin instantiation of the shared `w4a16_gemv_batchm_impl` template.
// Its former standalone body carried the same three bit-parity divergences
// from `w4a16_gemv` documented on that template — see the note there.

// ============================================================
// W4A16 GEMV batch3 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch3 (3-input) with Q/Gate deinterleave.
// Reads Q+Gate weight matrix once for 3 input tokens, produces 3 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[3, K] BF16 (3 token hidden states)
// Output: C[3, N] BF16 (deinterleaved: C[i] = [Qi|Gi])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV dual batch3: K+V for 3 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 3 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 3 outputs.
//
// Input:  A[3, K_in] BF16 (3 token hidden states)
// Output: C[3, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch3(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [3, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [3, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}

#if !defined(__SCALE__) && !defined(__HIP_PLATFORM_AMD__)
// ============================================================
// W4A16 GEMV on tensor cores for M <= 8 rows (speculative verify blocks).
// ============================================================
// `w4a16_gemv_batchm_impl` keeps the M=1 kernel's scalar FMA chain so its rows
// are bit-identical to decode, which leaves it issue-bound at M >= 5 (batch8
// at M=8 reaches ~50% of DRAM peak). Where no M=1 decode must be matched (the
// GLM DFlash lane verifies every step as one block) the rows can go to the
// tensor cores instead: `mma.m16n8k16` takes 16 dequantized weight rows as A
// (16 outputs x 16 k) and the activation rows as B (16 k x 8 rows), so an
// 8-row block fills the MMA's N dimension with no padding and the kernel is
// DRAM-bound.
//
// Dequantization is exact: an E2M1 value times an E4M3 group scale has at most
// 7 significant bits and lands inside BF16's range, so the BF16 A operand is
// the true weight / scale2. Accumulation is FP32 and scale2 is applied once
// per output. Rows are independent of each other; the summation order differs
// from `w4a16_gemv`, so this is NOT bit-identical to it.
//
// Mapping: warp w of a CTA owns weight rows [16*blockIdx.x, +16) and the
// 128-wide K chunks w, w+TC_WARPS, ... Lane (g = lane/4, c = lane%4) loads 32
// consecutive K values (16 bytes) of rows g and g+8 and of activation row g,
// prefetching its next chunk while the current one is on the tensor cores.
// MMA step j (0..7) feeds k-slots {2c,2c+1} <- values 4j+{0,1} and
// {2c+8,2c+9} <- values 4j+{2,3} of the lane's run; A and B use the same slot
// -> K map, so each product pairs the right weight with the right activation.
//
// A:[M,K] BF16, B_packed:[N,K/2], B_scale:[N,K/16] E4M3, C:[M,N] BF16.
// K % 16 == 0. Grid: (ceil(N/16),1,1) Block: (TC_WARPS*32,1,1).
#define W4A16_TC_WARPS 8

struct W4a16TcChunk {
    uint4 w0, w1;          // 32 E2M1 values of rows g and g+8
    uint4 x[4];            // 32 BF16 activations of row g
    unsigned int s0, s1;   // group-scale bytes (two groups) of rows g, g+8
};

__device__ __forceinline__ void w4a16_tc_load(
    W4a16TcChunk& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p, const __nv_bfloat16* ap,
    unsigned int kb, unsigned int K, bool v0, bool v1, bool vm
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.x[0] = z; t.x[1] = z; t.x[2] = z; t.x[3] = z;
    t.s0 = 0u; t.s1 = 0u;
    if (kb >= K) return;
    const bool full = kb + 16u < K;  // second 16-value group present
    if (v0) {
        if (full) t.w0 = *(const uint4*)(w0p + kb / 2u);
        else { const uint2 h = *(const uint2*)(w0p + kb / 2u); t.w0.x = h.x; t.w0.y = h.y; }
        t.s0 = s0p[kb / GROUP_SIZE] | (full ? (unsigned int)s0p[kb / GROUP_SIZE + 1u] << 8 : 0u);
    }
    if (v1) {
        if (full) t.w1 = *(const uint4*)(w1p + kb / 2u);
        else { const uint2 h = *(const uint2*)(w1p + kb / 2u); t.w1.x = h.x; t.w1.y = h.y; }
        t.s1 = s1p[kb / GROUP_SIZE] | (full ? (unsigned int)s1p[kb / GROUP_SIZE + 1u] << 8 : 0u);
    }
    if (vm) {
        const uint4* xp = (const uint4*)(ap + kb);
        t.x[0] = xp[0]; t.x[1] = xp[1];
        if (full) { t.x[2] = xp[2]; t.x[3] = xp[3]; }
    }
}

__device__ __forceinline__ __nv_bfloat162 w4a16_tc_scale(unsigned int byte) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)byte;
    return __float2bfloat162_rn((float)f);
}

__device__ __forceinline__ void w4a16_tc_mma(
    float acc[4], const W4a16TcChunk& t, const unsigned int* s_pair
) {
    const unsigned int wb0[4] = {t.w0.x, t.w0.y, t.w0.z, t.w0.w};
    const unsigned int wb1[4] = {t.w1.x, t.w1.y, t.w1.z, t.w1.w};
    const unsigned int xw[16] = {t.x[0].x, t.x[0].y, t.x[0].z, t.x[0].w,
                                 t.x[1].x, t.x[1].y, t.x[1].z, t.x[1].w,
                                 t.x[2].x, t.x[2].y, t.x[2].z, t.x[2].w,
                                 t.x[3].x, t.x[3].y, t.x[3].z, t.x[3].w};
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const __nv_bfloat162 sc0 = w4a16_tc_scale(t.s0 >> (8 * half));
        const __nv_bfloat162 sc1 = w4a16_tc_scale(t.s1 >> (8 * half));
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            const int j = half * 4 + jj;
            // Bytes 2j, 2j+1 of each row's 16-byte run = values 4j..4j+3.
            const unsigned int word0 = wb0[j >> 1] >> ((j & 1) * 16);
            const unsigned int word1 = wb1[j >> 1] >> ((j & 1) * 16);
            unsigned int a[4];
            __nv_bfloat162 v;
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[word0 & 0xFFu]), sc0);
            a[0] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[word1 & 0xFFu]), sc1);
            a[1] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[(word0 >> 8) & 0xFFu]), sc0);
            a[2] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[(word1 >> 8) & 0xFFu]), sc1);
            a[3] = *reinterpret_cast<unsigned int*>(&v);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(xw[2 * j]), "r"(xw[2 * j + 1]));
        }
    }
}

// ATLAS_GLM_CANONICAL_VERIFY runs 9..32 rows through this 8-row tier as
// NT = ceil(rows / 8) activation tiles against ONE sweep of the CTA's weight
// block. Each tile sees exactly w4a16_gemv_tc8's chunk order, MMA sequence
// and cross-warp epilogue, so its rows come out bit-identical to an 8-row
// launch of them: a row's bits never depend on the rows beside it, whatever
// the width 1..32. The shared weight chunk is double-buffered; each tile's
// activations (L1/L2-resident, every CTA reads them) are loaded at use.
// On GB10 this beats the 64-wide-K tc16/tc32 at their own widths (cold
// 4096 x 4096: 53 vs 70 us at 16 rows, 66 vs 80 at 32).
template <int NT>
__device__ __forceinline__ void w4a16_tc8_sweep(
    const __nv_bfloat16* __restrict__ A, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p, const unsigned int* s_pair,
    float (*s_red)[WARP_SIZE][4], __nv_bfloat16* __restrict__ C, float scale2,
    unsigned int r0, unsigned int r1, unsigned int M, unsigned int N, unsigned int K,
    bool v0, bool v1
) {
    const unsigned int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int chunks = (K + 127u) / 128u;
    float acc[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    // Weights only (vm = false); a tile's activations join them at use.
    W4a16TcChunk cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks) w4a16_tc_load(cur, w0p, w1p, s0p, s1p, A, ch * 128u + 32u * c, K, v0, v1, false);
    for (; ch < chunks; ch += W4A16_TC_WARPS) {
        const unsigned int next = ch + W4A16_TC_WARPS;
        if (next < chunks)
            w4a16_tc_load(nxt, w0p, w1p, s0p, s1p, A, next * 128u + 32u * c, K, v0, v1, false);
        const unsigned int kb = ch * 128u + 32u * c;
        #pragma unroll
        for (int t = 0; t < NT; t++) {
            // The activation half of w4a16_tc_load for row 8t + g.
            const unsigned int row = 8u * t + g;
            W4a16TcChunk x = cur;
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            x.x[0] = z; x.x[1] = z; x.x[2] = z; x.x[3] = z;
            if (row < M && kb < K) {
                const uint4* xp = (const uint4*)(A + (unsigned long long)row * K + kb);
                x.x[0] = xp[0]; x.x[1] = xp[1];
                if (kb + 16u < K) { x.x[2] = xp[2]; x.x[3] = xp[3]; }
            }
            w4a16_tc_mma(acc[t], x, s_pair);
        }
        cur = nxt;
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        if (t > 0) __syncthreads();  // warp 0 has read tile t-1's partials
        #pragma unroll
        for (int i = 0; i < 4; i++) s_red[warp][lane][i] = acc[t][i];
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float v = 0.0f;
                #pragma unroll
                for (int w = 0; w < W4A16_TC_WARPS; w++) v += s_red[w][lane][i];
                const unsigned int n = (i < 2) ? r0 : r1;
                const unsigned int m = 8u * t + 2u * c + (i & 1u);
                if (n < N && m < M) C[(unsigned long long)m * N + n] = __float2bfloat16(v * scale2);
            }
        }
    }
}

// `ld_half` / `ld_groups`: bytes between weight rows and between scale rows
// (K/2 and K/16 for a whole weight; larger for a K-slice of a wider one).
__device__ __forceinline__ void w4a16_gemv_tc8_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned long long ld_half,
    unsigned long long ld_groups
) {
    // Byte -> packed BF16x2 {low nibble, high nibble} of unscaled E2M1.
    __shared__ unsigned int s_pair[256];
    __shared__ float s_red[W4A16_TC_WARPS][WARP_SIZE][4];
    for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(E2M1_LUT[i & 0xF], E2M1_LUT[i >> 4]);
        s_pair[i] = *reinterpret_cast<const unsigned int*>(&v);
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = blockIdx.x * 16u + g, r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N, vm = g < M;
    const unsigned char* w0p = B_packed + (unsigned long long)(v0 ? r0 : 0u) * ld_half;
    const unsigned char* w1p = B_packed + (unsigned long long)(v1 ? r1 : 0u) * ld_half;
    const unsigned char* s0p = B_scale + (unsigned long long)(v0 ? r0 : 0u) * ld_groups;
    const unsigned char* s1p = B_scale + (unsigned long long)(v1 ? r1 : 0u) * ld_groups;
    const __nv_bfloat16* ap = A + (unsigned long long)(vm ? g : 0u) * K;

    if (M > 8u) {  // ATLAS_GLM_CANONICAL_VERIFY: 9..32 rows (w4a16_tc8_sweep)
        switch ((M + 7u) / 8u) {
            case 2: w4a16_tc8_sweep<2>(A, w0p, w1p, s0p, s1p, s_pair, s_red, C, scale2, r0, r1, M, N, K, v0, v1); break;
            case 3: w4a16_tc8_sweep<3>(A, w0p, w1p, s0p, s1p, s_pair, s_red, C, scale2, r0, r1, M, N, K, v0, v1); break;
            default: w4a16_tc8_sweep<4>(A, w0p, w1p, s0p, s1p, s_pair, s_red, C, scale2, r0, r1, M, N, K, v0, v1); break;
        }
        return;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int chunks = (K + 127u) / 128u;
    W4a16TcChunk cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks) w4a16_tc_load(cur, w0p, w1p, s0p, s1p, ap, ch * 128u + 32u * c, K, v0, v1, vm);
    for (; ch < chunks; ch += W4A16_TC_WARPS) {
        const unsigned int next = ch + W4A16_TC_WARPS;
        if (next < chunks)
            w4a16_tc_load(nxt, w0p, w1p, s0p, s1p, ap, next * 128u + 32u * c, K, v0, v1, vm);
        w4a16_tc_mma(acc, cur, s_pair);
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) s_red[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < W4A16_TC_WARPS; w++) v += s_red[w][lane][i];
        // C fragment: acc[0..1] -> output row g, acc[2..3] -> row g+8;
        // columns (activation rows) 2c and 2c+1.
        const unsigned int n = (i < 2) ? r0 : r1;
        const unsigned int m = 2u * c + (i & 1u);
        if (n < N && m < M) C[(unsigned long long)m * N + n] = __float2bfloat16(v * scale2);
    }
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K
) {
    atlas_pdl_enter();
    w4a16_gemv_tc8_impl(A, B_packed, B_scale, scale2, C, M, N, K, K / 2u, K / GROUP_SIZE);
}

// K-slice of a wider weight: rows `ld_half` / `ld_groups` bytes apart.
extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc8_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups
) {
    atlas_pdl_enter();
    w4a16_gemv_tc8_impl(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

// `w4a16_gemv_tc8_ld` whose first `touch_ctas` CTAs pull the first
// `touch_rows` weight rows (packed bytes and scales) into L2 while the kernel
// waits on its PDL predecessor (atlas_pdl_touch.cuh). Same body: bit-identical.
extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc8_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups, unsigned int touch_rows, unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({B_packed, K / 2u, ld_half}, {B_scale, K / GROUP_SIZE, ld_groups},
                          touch_rows, blockIdx.x, touch_ctas);
    w4a16_gemv_tc8_impl(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

// ── Wider tensor-core tiers (9..32 rows): owner-batched verify blocks ──
// Same mapping as w4a16_gemv_tc8 with NT 8-row activation tiles sharing each
// dequantized weight fragment, so the weight is read once for up to NT*8 rows.
// Lanes cover 16 K values (one scale group) per 64-wide chunk to keep the
// NT activation tiles and the one-chunk prefetch in registers.
template <int NT>
struct W4a16TcnChunk {
    uint2 w0, w1;          // 16 E2M1 values of rows g and g+8
    uint4 x[NT][2];        // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;   // group-scale bytes of rows g, g+8
};

template <int NT>
__device__ __forceinline__ void w4a16_tcn_load(
    W4a16TcnChunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p, const __nv_bfloat16* A,
    unsigned int g, unsigned int M, unsigned int K, unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = make_uint2(0u, 0u); t.w1 = t.w0; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    if (v0) { t.w0 = *(const uint2*)(w0p + kb / 2u); t.s0 = s0p[kb / GROUP_SIZE]; }
    if (v1) { t.w1 = *(const uint2*)(w1p + kb / 2u); t.s1 = s1p[kb / GROUP_SIZE]; }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * K + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

template <int NT>
__device__ __forceinline__ void w4a16_gemv_tcn_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned long long ld_half,
    unsigned long long ld_groups
) {
    __shared__ unsigned int s_pair[256];
    __shared__ float s_red[W4A16_TC_WARPS][WARP_SIZE][NT * 4];
    for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(E2M1_LUT[i & 0xF], E2M1_LUT[i >> 4]);
        s_pair[i] = *reinterpret_cast<const unsigned int*>(&v);
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = blockIdx.x * 16u + g, r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned char* w0p = B_packed + (unsigned long long)(v0 ? r0 : 0u) * ld_half;
    const unsigned char* w1p = B_packed + (unsigned long long)(v1 ? r1 : 0u) * ld_half;
    const unsigned char* s0p = B_scale + (unsigned long long)(v0 ? r0 : 0u) * ld_groups;
    const unsigned char* s1p = B_scale + (unsigned long long)(v1 ? r1 : 0u) * ld_groups;

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    const unsigned int chunks = (K + 63u) / 64u;
    W4a16TcnChunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks) w4a16_tcn_load<NT>(cur, w0p, w1p, s0p, s1p, A, g, M, K, ch * 64u + 16u * c, v0, v1);
    for (; ch < chunks; ch += W4A16_TC_WARPS) {
        const unsigned int next = ch + W4A16_TC_WARPS;
        if (next < chunks)
            w4a16_tcn_load<NT>(nxt, w0p, w1p, s0p, s1p, A, g, M, K, next * 64u + 16u * c, v0, v1);
        const __nv_bfloat162 sc0 = w4a16_tc_scale(cur.s0);
        const __nv_bfloat162 sc1 = w4a16_tc_scale(cur.s1);
        const unsigned int wb0[2] = {cur.w0.x, cur.w0.y}, wb1[2] = {cur.w1.x, cur.w1.y};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned int word0 = wb0[j >> 1] >> ((j & 1) * 16);
            const unsigned int word1 = wb1[j >> 1] >> ((j & 1) * 16);
            unsigned int a[4];
            __nv_bfloat162 v;
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[word0 & 0xFFu]), sc0);
            a[0] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[word1 & 0xFFu]), sc1);
            a[1] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[(word0 >> 8) & 0xFFu]), sc0);
            a[2] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[(word1 >> 8) & 0xFFu]), sc1);
            a[3] = *reinterpret_cast<unsigned int*>(&v);
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                // Values 4j..4j+3 of the tile row's 16-value run = words 2j, 2j+1.
                const uint4 lo = cur.x[i][0], hi = cur.x[i][1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(xw[2 * j]), "r"(xw[2 * j + 1]));
            }
        }
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) s_red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < W4A16_TC_WARPS; w++) v += s_red[w][lane][i * 4 + q];
            const unsigned int n = (q < 2) ? r0 : r1;
            const unsigned int m = 8u * i + 2u * c + (q & 1u);
            if (n < N && m < M) C[(unsigned long long)m * N + n] = __float2bfloat16(v * scale2);
        }
    }
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemv_tcn_impl<2>(A, B_packed, B_scale, scale2, C, M, N, K, K / 2u, K / GROUP_SIZE);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc16_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups
) {
    w4a16_gemv_tcn_impl<2>(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K
) {
    w4a16_gemv_tcn_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K, K / 2u, K / GROUP_SIZE);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc32_ld(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups
) {
    w4a16_gemv_tcn_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

// Touch twins of the 16/32-row tiers (see `w4a16_gemv_tc8_touch`); like the
// tc8 twin they take explicit row strides and are launched with PDL.
extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc16_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups, unsigned int touch_rows, unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({B_packed, K / 2u, ld_half}, {B_scale, K / GROUP_SIZE, ld_groups},
                          touch_rows, blockIdx.x, touch_ctas);
    w4a16_gemv_tcn_impl<2>(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE)
w4a16_gemv_tc32_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale, const float scale2,
    __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,
    unsigned int ld_half, unsigned int ld_groups, unsigned int touch_rows, unsigned int touch_ctas
) {
    atlas_pdl_enter_touch({B_packed, K / 2u, ld_half}, {B_scale, K / GROUP_SIZE, ld_groups},
                          touch_rows, blockIdx.x, touch_ctas);
    w4a16_gemv_tcn_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K, ld_half, ld_groups);
}

// Two projections of one input `A` (the shared expert's gate and up: same N
// and K, whole weights) in ONE launch, grid z = plane, each plane the
// unchanged tc{8,16,32} body of its touch twin: bit-identical, and the up
// plane's CTAs run beside the gate plane's instead of waiting for the whole
// gate launch to complete (64 CTAs each at N = 1024: neither fills the GPU).
// Each plane touches its weight during the PDL wait (atlas_pdl_touch.cuh).
template <int NT>
__device__ __forceinline__ void w4a16_gemv_tc_pair(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B0, const unsigned char* __restrict__ S0, const float scale0,
    __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1, const float scale1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int M, unsigned int N, unsigned int K, unsigned int touch_rows, unsigned int touch_ctas
) {
    atlas_pdl_enter_touch_pair({B0, K / 2u, K / 2u}, {S0, K / GROUP_SIZE, K / GROUP_SIZE},
                               {B1, K / 2u, K / 2u}, {S1, K / GROUP_SIZE, K / GROUP_SIZE},
                               touch_rows, blockIdx.x, touch_ctas);
    const bool one = blockIdx.z != 0u;
    const unsigned char* B = one ? B1 : B0;
    const unsigned char* S = one ? S1 : S0;
    const float scale = one ? scale1 : scale0;
    __nv_bfloat16* C = one ? C1 : C0;
    if constexpr (NT == 1)
        w4a16_gemv_tc8_impl(A, B, S, scale, C, M, N, K, K / 2u, K / GROUP_SIZE);
    else
        w4a16_gemv_tcn_impl<NT>(A, B, S, scale, C, M, N, K, K / 2u, K / GROUP_SIZE);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE) w4a16_gemv_tc8_pair_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B0,
    const unsigned char* __restrict__ S0, const float scale0, __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1, const float scale1,
    __nv_bfloat16* __restrict__ C1, unsigned int M, unsigned int N, unsigned int K,
    unsigned int touch_rows, unsigned int touch_ctas) {
    w4a16_gemv_tc_pair<1>(A, B0, S0, scale0, C0, B1, S1, scale1, C1, M, N, K, touch_rows, touch_ctas);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE) w4a16_gemv_tc16_pair_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B0,
    const unsigned char* __restrict__ S0, const float scale0, __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1, const float scale1,
    __nv_bfloat16* __restrict__ C1, unsigned int M, unsigned int N, unsigned int K,
    unsigned int touch_rows, unsigned int touch_ctas) {
    w4a16_gemv_tc_pair<2>(A, B0, S0, scale0, C0, B1, S1, scale1, C1, M, N, K, touch_rows, touch_ctas);
}

extern "C" __global__ void __launch_bounds__(W4A16_TC_WARPS * WARP_SIZE) w4a16_gemv_tc32_pair_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B0,
    const unsigned char* __restrict__ S0, const float scale0, __nv_bfloat16* __restrict__ C0,
    const unsigned char* __restrict__ B1, const unsigned char* __restrict__ S1, const float scale1,
    __nv_bfloat16* __restrict__ C1, unsigned int M, unsigned int N, unsigned int K,
    unsigned int touch_rows, unsigned int touch_ctas) {
    w4a16_gemv_tc_pair<4>(A, B0, S0, scale0, C0, B1, S1, scale1, C1, M, N, K, touch_rows, touch_ctas);
}
#endif
