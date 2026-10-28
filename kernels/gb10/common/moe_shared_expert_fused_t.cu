// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE GEMV — Phase 8a unified-layout MoE.
//
// Same semantics as moe_expert_silu_down_shared / moe_expert_gate_up_shared,
// but reads the weight tensor in transposed `[K/2, N]` layout (input-major)
// instead of the current `[N, K/2]` layout (output-major). Goal: a single
// weight layout shared between prefill and decode so we can free the
// untransposed copies (~59 GB on MiniMax M2.7-NVFP4 EP=2) and either
// run the persistent down transpose or use the headroom for KV cache.
//
// Coalescence strategy is inverted vs the original kernel:
//   - original: each thread owns ONE output (sub-group reduces over K)
//     → reads B[n_lane * (K/2) + k_iter*8] = 8 bytes contiguous per lane
//   - transposed: each thread STILL owns one output, but threads in a
//     warp have ADJACENT n's (lane = n within warp). Read B_t[k_half * N + n]:
//     32 bytes contiguous per warp = 1 cache line per K iter. No warp
//     reduction needed (each lane has its own accumulator + own output).
//
// Block: (128, 1, 1)  Grid: (ceil(N/128), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ARM-2 Phase-K RIDER 1: the E8M0 scale primitive (mx_block_scale / atlas_dec_e4m3)
// lives in ONE shared header, included by both Family A (this file) and Family B
// (../qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu) — bit-identical across
// families, no second copy.
#include "mx_block_scale.cuh"

// Tuning note: 32-thread blocks (1 warp per block) outperform 128-thread blocks
// on GB10 for the transposed decode silu_down — more blocks → better SM
// occupancy, the s_act shared-mem precompute parallelism is unchanged because
// the K loop dominates (block_size irrelevant once each thread is 1 output).
#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// atlas_dec_e4m3 + mx_block_scale<E8M0> now live in mx_block_scale.cuh (RIDER 1).

// Transposed-layout fused gate+up decode kernel.
//
// Single-token GEMV: for each routed expert slot (top_k of them) plus the
// shared expert, compute gate_out = A @ W_gate^T and up_out = A @ W_up^T.
// blockIdx.z selects gate (0) or up (1). Same coalescence strategy as
// silu_down_t — each thread owns one output, lanes within warp adjacent
// in N.
// ARM-2 Phase-K RIDER A: DUAL-FORMAT. Routed experts (E8M0_R/GS_R) and the
// shared expert (E8M0_S/GS_S) can carry DIFFERENT quant formats — the native
// V4 ckpt is heterogeneous (routed MXFP4-E8M0, shared FP8→NVFP4). The format is
// keyed off the weight's `WeightQuantFormat` tag (Rust dispatch, asserted via
// `expect`), NOT positionally; `is_shared` only selects the region. The branch
// is BLOCK-UNIFORM (grid.y = expert_slot, one expert per block — RIDER A2).
template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void gate_up_shared_t_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed).
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed;
            B_scale = sh_gate_t_scale;
            s2 = sh_gate_s2;
            C = sh_gate_out;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed;
            B_scale = sh_up_t_scale;
            s2 = sh_up_s2;
            C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id];
            C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id];
            C = up_out;
        }
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // GEMV: C[n] = sum_k A[k] * W[n, k]. With transposed weight stored as
    // [K/2, N] packed: each byte at (k_half, n) holds two consecutive k
    // nibbles for output position n. Iterate by scale-group (16 K) to
    // cache the per-group scale.
    // Block-uniform dual-format accumulation. ONE parameterized macro so the
    // shared and routed paths run byte-identical logic differing only in
    // (GS, E8M0) — the shared NVFP4 branch stays bit-identical to the baseline
    // kernel (RIDER A3); the compile-time template keeps each fully unrolled.
    float acc = 0.0f;
    #define GATEUP_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                float a_lo = __bfloat162float(A[k_half * 2]); \
                float a_hi = __bfloat162float(A[k_half * 2 + 1]); \
                float w_lo = s_lut[byte & 0xFu] * sc; \
                float w_hi = s_lut[(byte >> 4) & 0xFu] * sc; \
                acc += a_lo * w_lo + a_hi * w_hi; \
            } \
        } \
    } while(0)
    // Same-format wrappers (NVFP4 default) collapse to a SINGLE loop — no
    // branch, PTX-identical to the baseline kernel (RIDER A3). Only the
    // heterogeneous e8m0 wrapper (routed≠shared) emits the block-uniform branch.
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        GATEUP_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { GATEUP_ACCUM(GS_S, E8M0_S); }
        else           { GATEUP_ACCUM(GS_R, E8M0_R); }
    }
    #undef GATEUP_ACCUM

    if (is_shared) {
        C[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void gate_up_shared_t_impl_smem(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k);

// NVFP4 (default): FP8-E4M3 per-16 scales × per-tensor global.
extern "C" __global__ void moe_expert_gate_up_shared_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl_smem<GROUP_SIZE, false, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// Native MXFP4 (ARM-2): ROUTED experts E8M0 per-32 (no global); SHARED expert
// stays NVFP4 (`<GROUP_SIZE,false>`) — the native V4 ckpt ships the shared
// expert FP8→NVFP4, NOT MXFP4. Routed buffers are the E8M0-tagged
// (`WeightQuantFormat::Mxfp4E8m0`) transcode-free loader output; sh_* are NVFP4.
extern "C" __global__ void moe_expert_gate_up_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    gate_up_shared_t_impl<32, true, GROUP_SIZE, false>(
        A, gate_packed_t_ptrs, gate_scale_t_ptrs, gate_scale2_vals, gate_out,
        up_packed_t_ptrs, up_scale_t_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_t_packed, sh_gate_t_scale, sh_gate_s2, sh_gate_out,
        sh_up_t_packed, sh_up_t_scale, sh_up_s2, sh_up_out, N, K, top_k);
}

// Transposed-layout silu_down decode kernel.
//
// Per-expert weight buffers `[K/2, N]` packed NVFP4 + `[K/16, N]` FP8 scales.
// Input gate_out / up_out: `[(top_k+1), K]` BF16 (per-slot). top_k slot is
// the shared-expert input. Output C: `[top_k, N]` BF16; shared-expert
// output goes to `sh_down_out: [N]`.
// ARM-2 Phase-K RIDER A: DUAL-FORMAT (see gate_up_shared_t_impl). Routed
// (GS_R/E8M0_R) vs shared (GS_S/E8M0_S), block-uniform on is_shared.
template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void silu_down_shared_t_impl(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,   // [num_experts] device ptrs to [K/2 * N] bytes
    const unsigned long long* __restrict__ scale_t_ptrs,    // [num_experts] device ptrs to [K/16 * N] bytes
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers (transposed layout)
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        if (sh_down_t_packed == 0) {
            // No shared expert — write zeros.
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed;
        B_scale = sh_down_t_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in;
        u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP remote expert: NULL pointer → write zero output and return.
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    // Phase 1: cooperatively precompute s_act[K] = SiLU(gate[K]) * up[K].
    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_T[threadIdx.x];
    __syncthreads();

    if (!valid) return;

    // Phase 2: per-thread accumulate over K_half iterations. Each thread
    // owns ONE output position `n`; lanes in a warp have adjacent n's so
    // `B_packed[k_half * N + n]` reads are coalesced (1 byte per lane,
    // 32 bytes contiguous per warp per iter).
    const unsigned int K_half = K / 2;
    float acc = 0.0f;

    // Block-uniform dual-format accumulation (RIDER A). Cache per-group scale
    // in a register; iterate GS/2 K_half iters per group.
    #define SILUDOWN_ACCUM(GS_, E8M0_) do { \
        const unsigned int num_groups = K / (GS_); \
        for (unsigned int sg = 0; sg < num_groups; sg++) { \
            unsigned char sb = B_scale[(unsigned long long)sg * N + n]; \
            float sc = mx_block_scale<(E8M0_)>(sb, s2); \
            const unsigned int kh_base = sg * ((GS_) / 2); \
            _Pragma("unroll") \
            for (unsigned int kh_off = 0; kh_off < ((GS_) / 2); kh_off++) { \
                unsigned int k_half = kh_base + kh_off; \
                unsigned char byte = B_packed[(unsigned long long)k_half * N + n]; \
                unsigned int nibble_lo = byte & 0xFu; \
                unsigned int nibble_hi = (byte >> 4) & 0xFu; \
                float w_lo = s_lut[nibble_lo] * sc; \
                float w_hi = s_lut[nibble_hi] * sc; \
                float a_lo = s_act[k_half * 2]; \
                float a_hi = s_act[k_half * 2 + 1]; \
                acc += a_lo * w_lo + a_hi * w_hi; \
            } \
            if (kh_base + ((GS_) / 2) > K_half) break; \
        } \
    } while(0)
    if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
        SILUDOWN_ACCUM(GS_R, E8M0_R);
    } else {
        if (is_shared) { SILUDOWN_ACCUM(GS_S, E8M0_S); }
        else           { SILUDOWN_ACCUM(GS_R, E8M0_R); }
    }
    #undef SILUDOWN_ACCUM

    // Output offset: routed → C[expert_slot * N + n]; shared → sh_down_out[n].
    if (is_shared) {
        sh_down_out[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void silu_down_shared_t_impl_smem(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k);

// NVFP4 (default): FP8-E4M3 per-16 scales × per-tensor global.
extern "C" __global__ void moe_expert_silu_down_shared_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl_smem<GROUP_SIZE, false, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// Native MXFP4 (ARM-2): ROUTED experts E8M0 per-32; SHARED expert stays NVFP4.
extern "C" __global__ void moe_expert_silu_down_shared_t_e8m0(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    silu_down_shared_t_impl<32, true, GROUP_SIZE, false>(
        gate_out, up_out, packed_t_ptrs, scale_t_ptrs, scale2_vals, C,
        expert_indices, sh_gate_in, sh_up_in, sh_down_t_packed, sh_down_t_scale,
        sh_down_s2, sh_down_out, N, K, top_k);
}

// ── Shared-memory tiled impls (S2d): production path ─────────────────────
// Stage each expert's packed tile into smem via 16 B cooperative loads
// (double-buffered), then every thread accumulates ITS column in the
// identical per-output FP32 order. Byte-identical to the scalar loop.
#define TILE_KH 64u
#define TILE_STRIDE 48u
template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void gate_up_shared_t_impl_smem(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2; __nv_bfloat16* C;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) { if (n < N) sh_gate_out[n] = __float2bfloat16(0.0f); return; }
            B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            if (sh_up_t_packed == 0) { if (n < N) sh_up_out[n] = __float2bfloat16(0.0f); return; }
            B_packed = sh_up_t_packed; B_scale = sh_up_t_scale; s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        if (B_packed == 0) { if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f); return; }
    }

    const unsigned int lane = threadIdx.x;
    const unsigned int ns0 = blockIdx.x * blockDim.x;
    const bool valid = (n < N);
    __shared__ float s_lut[16];
    if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ unsigned char s_pk[2][TILE_KH * TILE_STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * TILE_STRIDE];
    __syncthreads();

    const unsigned int r = lane / 2u;
    const unsigned int off = (lane % 2u) * 16u;
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        _Pragma("unroll")
        for (unsigned int w = 0; w < TILE_KH / 16u; w++) {
            unsigned int rr = r + w * 16u;
            *(uint4*)&s_pk[buf][rr * TILE_STRIDE + off] =
                *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + off];
        }
        const unsigned int ng = TILE_KH / 8u;
        if (lane < ng * 2u) {
            unsigned int sg = lane / 2u;
            unsigned int off2 = (lane % 2u) * 16u;
            *(uint4*)&s_sc[buf][sg * TILE_STRIDE + off2] =
                *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + off2];
        }
    };

    float acc = 0.0f;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (valid) {
            #define GU_TILE_ACC(GS_, E8M0_) do { \
                if constexpr ((GS_) == 16) { \
                    _Pragma("unroll 8") \
                    for (unsigned int rr = 0; rr < TILE_KH; rr++) { \
                        unsigned int k_half = tg * TILE_KH + rr; \
                        float sc = mx_block_scale<(E8M0_)>(s_sc[buf][(rr >> 3) * TILE_STRIDE + lane], s2); \
                        unsigned char byte = s_pk[buf][rr * TILE_STRIDE + lane]; \
                        float a_lo = __bfloat162float(A[k_half * 2]); \
                        float a_hi = __bfloat162float(A[k_half * 2 + 1]); \
                        acc += a_lo * (s_lut[byte & 0xFu] * sc) + a_hi * (s_lut[(byte >> 4) & 0xFu] * sc); \
                    } \
                } \
            } while(0)
            if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
                GU_TILE_ACC(GS_R, E8M0_R);
            } else {
                if (is_shared) { GU_TILE_ACC(GS_S, E8M0_S); }
                else           { GU_TILE_ACC(GS_R, E8M0_R); }
            }
            #undef GU_TILE_ACC
        }
    }
    __syncthreads();
    if (!valid) return;
    if (is_shared) C[n] = __float2bfloat16(acc);
    else C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
}

template<int GS_R, bool E8M0_R, int GS_S, bool E8M0_S>
__device__ __forceinline__ void silu_down_shared_t_impl_smem(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2;
    const __nv_bfloat16* g_ptr; const __nv_bfloat16* u_ptr;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_shared) {
        if (sh_down_t_packed == 0) { if (n < N) sh_down_out[n] = __float2bfloat16(0.0f); return; }
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        if (B_packed == 0) { if (n < N) C[expert_slot * N + n] = __float2bfloat16(0.0f); return; }
    }

    const unsigned int lane = threadIdx.x;
    const unsigned int ns0 = blockIdx.x * blockDim.x;
    const bool valid = (n < N);
    extern __shared__ float s_act[];
    for (unsigned int i = lane; i < K; i += blockDim.x) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __shared__ float s_lut[16];
    if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ unsigned char s_pk[2][TILE_KH * TILE_STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * TILE_STRIDE];
    __syncthreads();

    const unsigned int r = lane / 2u;
    const unsigned int off = (lane % 2u) * 16u;
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        _Pragma("unroll")
        for (unsigned int w = 0; w < TILE_KH / 16u; w++) {
            unsigned int rr = r + w * 16u;
            *(uint4*)&s_pk[buf][rr * TILE_STRIDE + off] =
                *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + off];
        }
        const unsigned int ng = TILE_KH / 8u;
        if (lane < ng * 2u) {
            unsigned int sg = lane / 2u;
            unsigned int off2 = (lane % 2u) * 16u;
            *(uint4*)&s_sc[buf][sg * TILE_STRIDE + off2] =
                *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + off2];
        }
    };

    float acc = 0.0f;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (valid) {
            #define SD_TILE_ACC(GS_, E8M0_) do { \
                if constexpr ((GS_) == 16) { \
                    _Pragma("unroll 8") \
                    for (unsigned int rr = 0; rr < TILE_KH; rr++) { \
                        unsigned int k_half = tg * TILE_KH + rr; \
                        float sc = mx_block_scale<(E8M0_)>(s_sc[buf][(rr >> 3) * TILE_STRIDE + lane], s2); \
                        unsigned char byte = s_pk[buf][rr * TILE_STRIDE + lane]; \
                        acc += s_act[k_half * 2] * (s_lut[byte & 0xFu] * sc) \
                             + s_act[k_half * 2 + 1] * (s_lut[(byte >> 4) & 0xFu] * sc); \
                    } \
                } \
            } while(0)
            if constexpr (GS_R == GS_S && E8M0_R == E8M0_S) {
                SD_TILE_ACC(GS_R, E8M0_R);
            } else {
                if (is_shared) { SD_TILE_ACC(GS_S, E8M0_S); }
                else           { SD_TILE_ACC(GS_R, E8M0_R); }
            }
            #undef SD_TILE_ACC
        }
    }
    __syncthreads();
    if (!valid) return;
    if (is_shared) sh_down_out[n] = __float2bfloat16(acc);
    else C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
}

// ── Multi-row dedup (S7): verify K rows share unique expert reads ────────
// Union of K rows' top-k ids: unique experts + per-row slot map. One warp.
#define MOE_MAX_K 8u
extern "C" __global__ void moe_expert_union_t(
    const unsigned int* __restrict__ expert_indices, // [K * top_k]
    unsigned int* __restrict__ u_experts,            // [K * top_k]
    unsigned int* __restrict__ u_rowmask,            // [K * top_k] bitmask of rows
    unsigned char* __restrict__ u_rowslot,           // [K * top_k][MOE_MAX_K] slot per row, 0xFF absent
    unsigned int* __restrict__ u_count,
    unsigned int K, unsigned int top_k
) {
    // Serial fill by lane 0: K*top_k <= 80 slots, trivially cheap on GPU.
    if (threadIdx.x == 0) {
        unsigned int U = 0;
        for (unsigned int t = 0; t < K * top_k; t++) {
            unsigned int id = expert_indices[t];
            unsigned int r = t / top_k;
            unsigned int slot = t % top_k;
            unsigned int u;
            for (u = 0; u < U; u++) if (u_experts[u] == id) break;
            if (u == U) {
                u_experts[u] = id;
                u_rowmask[u] = 0;
                for (unsigned int rr = 0; rr < MOE_MAX_K; rr++)
                    u_rowslot[u * MOE_MAX_K + rr] = 0xFF;
                U++;
            }
            u_rowmask[u] |= (1u << r);
            u_rowslot[u * MOE_MAX_K + r] = (unsigned char)slot;
        }
        *u_count = U;
    }
}

// gate_up over the UNIQUE routed experts of ROWS rows: each expert's packed
// tile is staged once (smem double-buffer, row stride BS*3/2 bytes), the A
// k-slice is staged per row, and each thread owns COLS output columns with a
// register acc[COLS][ROWS]. Every (row, column) keeps the single-row `_t`
// smem kernel's exact per-element expression and k order — bit-identical to
// ROWS single-row launches. Masks gate stores only; an unselected row still
// reads its own in-bounds A row and is never stored.
// Grid: (N/(BS*COLS), u_count + 1, 2); blockIdx.y == u_count -> shared.
template<int ROWS, int BS = 32, int COLS = 1, bool VEC = false>
__device__ __forceinline__ void gate_up_shared_t_rows_impl(
    const __nv_bfloat16* __restrict__ A,               // [ROWS, h]
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,              // [ROWS * top_k, N]
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ u_experts,
    const unsigned int* __restrict__ u_rowmask,
    const unsigned char* __restrict__ u_rowslot,
    const unsigned int* __restrict__ u_count,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,           // [ROWS, N]
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int h
) {
    constexpr unsigned int COLW = BS * COLS;           // columns per CTA
    constexpr unsigned int STRIDE = COLW + (COLW >= 256 ? 16 : COLW / 2);
    const unsigned int u = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    if (u > *u_count) return;
    const bool is_shared = (u == *u_count);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2;
    if (is_shared) {
        if (proj == 0) { B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2; }
        else           { B_packed = sh_up_t_packed;   B_scale = sh_up_t_scale;   s2 = sh_up_s2; }
    } else {
        const unsigned int eid = u_experts[u];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[eid];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[eid];
            s2 = gate_scale2_vals[eid];
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[eid];
            B_scale = (const unsigned char*)up_scale_t_ptrs[eid];
            s2 = up_scale2_vals[eid];
        }
    }

    const unsigned int lane = threadIdx.x;
    const unsigned int ns0 = blockIdx.x * (BS * COLS);
    unsigned int col[COLS]; bool vc[COLS]; bool any = false;
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++) { col[c] = ns0 + lane + c * BS; vc[c] = col[c] < N; any |= vc[c]; }
    __shared__ float s_lut[16];
    if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ unsigned char s_pk[2][TILE_KH * STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * STRIDE];
    __shared__ __nv_bfloat16 s_a[2][ROWS * TILE_KH * 2];
    __syncthreads();

    unsigned int mask = is_shared ? ((1u << ROWS) - 1u) : u_rowmask[u];
    if (B_packed == 0) {
        _Pragma("unroll")
        for (unsigned int c = 0; c < COLS; c++) {
            if (!vc[c]) continue;
            _Pragma("unroll")
            for (unsigned int r = 0; r < ROWS; r++) {
                if (!(mask & (1u << r))) continue;
                if (is_shared) { if (proj == 0) sh_gate_out[r * N + col[c]] = __float2bfloat16(0.0f);
                                 else           sh_up_out[r * N + col[c]] = __float2bfloat16(0.0f); }
                else {
                    unsigned int slot = u_rowslot[u * MOE_MAX_K + r];
                    __nv_bfloat16* C = (proj == 0) ? gate_out : up_out;
                    C[(unsigned long long)(r * top_k + slot) * N + col[c]] = __float2bfloat16(0.0f);
                }
            }
        }
        return;
    }

    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        _Pragma("unroll")
        for (unsigned int j = lane; j < TILE_KH * (COLW / 16u); j += BS) {
            const unsigned int rr = j / (COLW / 16u);
            const unsigned int cb = (j % (COLW / 16u)) * 16u;
            *(uint4*)&s_pk[buf][rr * STRIDE + cb] =
                *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + cb];
        }
        const unsigned int ng = TILE_KH / 8u;
        _Pragma("unroll")
        for (unsigned int j = lane; j < ng * (COLW / 16u); j += BS) {
            const unsigned int sg = j / (COLW / 16u);
            const unsigned int cb = (j % (COLW / 16u)) * 16u;
            *(uint4*)&s_sc[buf][sg * STRIDE + cb] =
                *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + cb];
        }
        _Pragma("unroll")
        for (unsigned int j = lane; j < ROWS * (TILE_KH * 2 / 8u); j += BS) {
            const unsigned int row = j / (TILE_KH * 2 / 8u);
            const unsigned int chunk = j % (TILE_KH * 2 / 8u);
            *(uint4*)&s_a[buf][row * TILE_KH * 2 + chunk * 8] =
                *(const uint4*)&A[(unsigned long long)row * h + tg * TILE_KH * 2 + chunk * 8];
        }
    };

    float acc[COLS][ROWS];
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++)
        _Pragma("unroll")
        for (unsigned int r = 0; r < ROWS; r++) acc[c][r] = 0.0f;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (any) {
            if constexpr (VEC) {
                _Pragma("unroll")
                for (unsigned int rr4 = 0; rr4 < TILE_KH / 4u; rr4++) {
                    __nv_bfloat16 av[ROWS][8];
                    _Pragma("unroll")
                    for (unsigned int r = 0; r < ROWS; r++)
                        *(uint4*)&av[r][0] = *(const uint4*)&s_a[buf][r * TILE_KH * 2 + rr4 * 8];
                    _Pragma("unroll")
                    for (unsigned int j = 0; j < 4; j++) {
                        const unsigned int rr = rr4 * 4 + j;
                        float w_lo[COLS], w_hi[COLS];
                        _Pragma("unroll")
                        for (unsigned int c = 0; c < COLS; c++) {
                            float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * STRIDE + lane + c * BS], s2);
                            unsigned char byte = s_pk[buf][rr * STRIDE + lane + c * BS];
                            w_lo[c] = s_lut[byte & 0xFu] * sc;
                            w_hi[c] = s_lut[(byte >> 4) & 0xFu] * sc;
                        }
                        _Pragma("unroll")
                        for (unsigned int r = 0; r < ROWS; r++) {
                            const float a_lo = __bfloat162float(av[r][2 * j]);
                            const float a_hi = __bfloat162float(av[r][2 * j + 1]);
                            _Pragma("unroll")
                            for (unsigned int c = 0; c < COLS; c++)
                                acc[c][r] += a_lo * w_lo[c] + a_hi * w_hi[c];
                        }
                    }
                }
            } else {
                _Pragma("unroll 8")
                for (unsigned int rr = 0; rr < TILE_KH; rr++) {
                    float w_lo[COLS], w_hi[COLS];
                    _Pragma("unroll")
                    for (unsigned int c = 0; c < COLS; c++) {
                        float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * STRIDE + lane + c * BS], s2);
                        unsigned char byte = s_pk[buf][rr * STRIDE + lane + c * BS];
                        w_lo[c] = s_lut[byte & 0xFu] * sc;
                        w_hi[c] = s_lut[(byte >> 4) & 0xFu] * sc;
                    }
                    _Pragma("unroll")
                    for (unsigned int r = 0; r < ROWS; r++) {
                        const float a_lo = __bfloat162float(s_a[buf][r * TILE_KH * 2 + rr * 2]);
                        const float a_hi = __bfloat162float(s_a[buf][r * TILE_KH * 2 + rr * 2 + 1]);
                        _Pragma("unroll")
                        for (unsigned int c = 0; c < COLS; c++)
                            acc[c][r] += a_lo * w_lo[c] + a_hi * w_hi[c];
                    }
                }
            }
        }
    }
    __syncthreads();
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++) {
        if (!vc[c]) continue;
        _Pragma("unroll")
        for (unsigned int r = 0; r < ROWS; r++) {
            if (!(mask & (1u << r))) continue;
            if (is_shared) { if (proj == 0) sh_gate_out[r * N + col[c]] = __float2bfloat16(acc[c][r]);
                             else           sh_up_out[r * N + col[c]] = __float2bfloat16(acc[c][r]); }
            else {
                unsigned int slot = u_rowslot[u * MOE_MAX_K + r];
                __nv_bfloat16* C = (proj == 0) ? gate_out : up_out;
                C[(unsigned long long)(r * top_k + slot) * N + col[c]] = __float2bfloat16(acc[c][r]);
            }
        }
    }
}

// silu_down over the unique routed experts of ROWS rows, COLS columns per
// thread. The per-(row, slot) gate/up k-slice is staged in smem beside the
// weight tile; the activation is the single-row s_act expression evaluated
// on the staged values. An unselected row computes on its slot-0 row and is
// never stored. Grid: (N/(BS*COLS), u_count + 1).
template<int ROWS, int BS = 32, int COLS = 1, bool VEC = false>
__device__ __forceinline__ void silu_down_shared_t_rows_impl(
    const __nv_bfloat16* __restrict__ gate_out,        // [ROWS * top_k, K]
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                     // [ROWS * top_k, N]
    const unsigned int* __restrict__ u_experts,
    const unsigned int* __restrict__ u_rowmask,
    const unsigned char* __restrict__ u_rowslot,
    const unsigned int* __restrict__ u_count,
    const __nv_bfloat16* __restrict__ sh_gate_in,      // [ROWS, K]
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,           // [ROWS, N]
    unsigned int N, unsigned int K, unsigned int top_k
) {
    constexpr unsigned int COLW = BS * COLS;
    constexpr unsigned int STRIDE = COLW + (COLW >= 256 ? 16 : COLW / 2);
    const unsigned int u = blockIdx.y;
    if (u > *u_count) return;
    const bool is_shared = (u == *u_count);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2;
    if (is_shared) {
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
    } else {
        const unsigned int eid = u_experts[u];
        B_packed = (const unsigned char*)packed_t_ptrs[eid];
        B_scale = (const unsigned char*)scale_t_ptrs[eid];
        s2 = scale2_vals[eid];
    }

    const unsigned int lane = threadIdx.x;
    const unsigned int ns0 = blockIdx.x * (BS * COLS);
    unsigned int col[COLS]; bool vc[COLS]; bool any = false;
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++) { col[c] = ns0 + lane + c * BS; vc[c] = col[c] < N; any |= vc[c]; }
    __shared__ float s_lut[16];
    if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ unsigned char s_pk[2][TILE_KH * STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * STRIDE];
    // Activation staged per k-tile as fp32, computed once per element per
    // CTA (the single-row kernel's s_act precompute, tiled) instead of per
    // element per column thread. Same bytes budget as the s_g/s_u pair.
    __shared__ float s_act[2][ROWS * TILE_KH * 2];
    __syncthreads();

    unsigned int mask = is_shared ? ((1u << ROWS) - 1u) : u_rowmask[u];
    if (B_packed == 0) {
        _Pragma("unroll")
        for (unsigned int c = 0; c < COLS; c++) {
            if (!vc[c]) continue;
            _Pragma("unroll")
            for (unsigned int r = 0; r < ROWS; r++) {
                if (!(mask & (1u << r))) continue;
                if (is_shared) sh_down_out[r * N + col[c]] = __float2bfloat16(0.0f);
                else {
                    unsigned int slot = u_rowslot[u * MOE_MAX_K + r];
                    C[(unsigned long long)(r * top_k + slot) * N + col[c]] = __float2bfloat16(0.0f);
                }
            }
        }
        return;
    }

    const __nv_bfloat16* g_base[ROWS];
    const __nv_bfloat16* u_base[ROWS];
    _Pragma("unroll")
    for (unsigned int r = 0; r < ROWS; r++) {
        if (is_shared) {
            g_base[r] = sh_gate_in + (unsigned long long)r * K;
            u_base[r] = sh_up_in + (unsigned long long)r * K;
        } else {
            unsigned int slot = u_rowslot[u * MOE_MAX_K + r];
            if (slot == 0xFF) slot = 0;
            const unsigned long long base = (unsigned long long)(r * top_k + slot) * K;
            g_base[r] = gate_out + base;
            u_base[r] = up_out + base;
        }
    }

    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        _Pragma("unroll")
        for (unsigned int j = lane; j < TILE_KH * (COLW / 16u); j += BS) {
            const unsigned int rr = j / (COLW / 16u);
            const unsigned int cb = (j % (COLW / 16u)) * 16u;
            *(uint4*)&s_pk[buf][rr * STRIDE + cb] =
                *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + cb];
        }
        const unsigned int ng = TILE_KH / 8u;
        _Pragma("unroll")
        for (unsigned int j = lane; j < ng * (COLW / 16u); j += BS) {
            const unsigned int sg = j / (COLW / 16u);
            const unsigned int cb = (j % (COLW / 16u)) * 16u;
            *(uint4*)&s_sc[buf][sg * STRIDE + cb] =
                *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + cb];
        }
        _Pragma("unroll")
        for (unsigned int j = lane; j < ROWS * (TILE_KH * 2 / 8u); j += BS) {
            const unsigned int row = j / (TILE_KH * 2 / 8u);
            const unsigned int chunk = j % (TILE_KH * 2 / 8u);
            const __nv_bfloat16* g8 =
                (const __nv_bfloat16*)&g_base[row][tg * TILE_KH * 2 + chunk * 8];
            const __nv_bfloat16* u8 =
                (const __nv_bfloat16*)&u_base[row][tg * TILE_KH * 2 + chunk * 8];
            float a8[8];
            _Pragma("unroll")
            for (unsigned int i = 0; i < 8; i++) {
                const float gf = __bfloat162float(g8[i]);
                const float uf = __bfloat162float(u8[i]);
                a8[i] = (gf / (1.0f + __expf(-gf))) * uf;
            }
            *(float4*)&s_act[buf][row * TILE_KH * 2 + chunk * 8] =
                make_float4(a8[0], a8[1], a8[2], a8[3]);
            *(float4*)&s_act[buf][row * TILE_KH * 2 + chunk * 8 + 4] =
                make_float4(a8[4], a8[5], a8[6], a8[7]);
        }
    };

    float acc[COLS][ROWS];
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++)
        _Pragma("unroll")
        for (unsigned int r = 0; r < ROWS; r++) acc[c][r] = 0.0f;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (any) {
            if constexpr (VEC) {
                _Pragma("unroll")
                for (unsigned int rr4 = 0; rr4 < TILE_KH / 4u; rr4++) {
                    float a8[ROWS][8];
                    _Pragma("unroll")
                    for (unsigned int r = 0; r < ROWS; r++) {
                        *(float4*)&a8[r][0] = *(const float4*)&s_act[buf][r * TILE_KH * 2 + rr4 * 8];
                        *(float4*)&a8[r][4] = *(const float4*)&s_act[buf][r * TILE_KH * 2 + rr4 * 8 + 4];
                    }
                    _Pragma("unroll")
                    for (unsigned int j = 0; j < 4; j++) {
                        const unsigned int rr = rr4 * 4 + j;
                        float w_lo[COLS], w_hi[COLS];
                        _Pragma("unroll")
                        for (unsigned int c = 0; c < COLS; c++) {
                            float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * STRIDE + lane + c * BS], s2);
                            unsigned char byte = s_pk[buf][rr * STRIDE + lane + c * BS];
                            w_lo[c] = s_lut[byte & 0xFu] * sc;
                            w_hi[c] = s_lut[(byte >> 4) & 0xFu] * sc;
                        }
                        _Pragma("unroll")
                        for (unsigned int r = 0; r < ROWS; r++) {
                            const float act0 = a8[r][2 * j];
                            const float act1 = a8[r][2 * j + 1];
                            _Pragma("unroll")
                            for (unsigned int c = 0; c < COLS; c++)
                                acc[c][r] += act0 * w_lo[c] + act1 * w_hi[c];
                        }
                    }
                }
            } else {
                _Pragma("unroll 8")
                for (unsigned int rr = 0; rr < TILE_KH; rr++) {
                    float w_lo[COLS], w_hi[COLS];
                    _Pragma("unroll")
                    for (unsigned int c = 0; c < COLS; c++) {
                        float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * STRIDE + lane + c * BS], s2);
                        unsigned char byte = s_pk[buf][rr * STRIDE + lane + c * BS];
                        w_lo[c] = s_lut[byte & 0xFu] * sc;
                        w_hi[c] = s_lut[(byte >> 4) & 0xFu] * sc;
                    }
                    _Pragma("unroll")
                    for (unsigned int r = 0; r < ROWS; r++) {
                        const float act0 = s_act[buf][r * TILE_KH * 2 + rr * 2];
                        const float act1 = s_act[buf][r * TILE_KH * 2 + rr * 2 + 1];
                        _Pragma("unroll")
                        for (unsigned int c = 0; c < COLS; c++)
                            acc[c][r] += act0 * w_lo[c] + act1 * w_hi[c];
                    }
                }
            }
        }
    }
    __syncthreads();
    _Pragma("unroll")
    for (unsigned int c = 0; c < COLS; c++) {
        if (!vc[c]) continue;
        _Pragma("unroll")
        for (unsigned int r = 0; r < ROWS; r++) {
            if (!(mask & (1u << r))) continue;
            if (is_shared) sh_down_out[r * N + col[c]] = __float2bfloat16(acc[c][r]);
            else {
                unsigned int slot = u_rowslot[u * MOE_MAX_K + r];
                C[(unsigned long long)(r * top_k + slot) * N + col[c]] = __float2bfloat16(acc[c][r]);
            }
        }
    }
}

#define T_ROWS_GU_ARGS \
    const __nv_bfloat16* A, const unsigned long long* gpk, \
    const unsigned long long* gsc, const float* gs2, __nv_bfloat16* gate_out, \
    const unsigned long long* upk, const unsigned long long* usc, \
    const float* us2, __nv_bfloat16* up_out, \
    const unsigned int* u_experts, const unsigned int* u_rowmask, \
    const unsigned char* u_rowslot, const unsigned int* u_count, \
    const unsigned char* sgp, const unsigned char* sgs, float sg2, \
    __nv_bfloat16* sh_gate_out, const unsigned char* sup, \
    const unsigned char* sus, float su2, __nv_bfloat16* sh_up_out, \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int h, \
    unsigned int num_rows
#define T_ROWS_SD_ARGS \
    const __nv_bfloat16* gate_out, const __nv_bfloat16* up_out, \
    const unsigned long long* pk, const unsigned long long* sc, \
    const float* s2, __nv_bfloat16* C, \
    const unsigned int* u_experts, const unsigned int* u_rowmask, \
    const unsigned char* u_rowslot, const unsigned int* u_count, \
    const __nv_bfloat16* sh_gate_in, const __nv_bfloat16* sh_up_in, \
    const unsigned char* sdp, const unsigned char* sds, float sd2, \
    __nv_bfloat16* sh_down_out, \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int num_rows

// Production shape: the measured winner — BS 256, one column per thread,
// 16 B smem activations loads (VEC), ROWS 2..8 (verify batch depths).
#define T_ROWS_SHIP(R) \
extern "C" __global__ void moe_expert_gate_up_shared_t_rows_v3_r##R##_b256(T_ROWS_GU_ARGS) { \
    (void)num_rows; \
    gate_up_shared_t_rows_impl<R,256,1,true>(A, gpk, gsc, gs2, gate_out, upk, usc, us2, \
        up_out, u_experts, u_rowmask, u_rowslot, u_count, sgp, sgs, sg2, \
        sh_gate_out, sup, sus, su2, sh_up_out, N, K, top_k, h); \
} \
extern "C" __global__ void moe_expert_silu_down_shared_t_rows_v3_r##R##_b256(T_ROWS_SD_ARGS) { \
    (void)num_rows; \
    silu_down_shared_t_rows_impl<R,256,1,true>(gate_out, up_out, pk, sc, s2, C, \
        u_experts, u_rowmask, u_rowslot, u_count, sh_gate_in, sh_up_in, \
        sdp, sds, sd2, sh_down_out, N, K, top_k); \
}

T_ROWS_SHIP(2)
T_ROWS_SHIP(3)
T_ROWS_SHIP(4)
T_ROWS_SHIP(5)
T_ROWS_SHIP(6)
T_ROWS_SHIP(7)
T_ROWS_SHIP(8)
