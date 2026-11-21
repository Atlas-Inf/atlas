// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 decode attention over the paged BF16 KV cache: common/paged_decode_attn.cu
// (one CTA per (q_head, seq), 8 warps over the KV range, online softmax) plus the two
// things Inkling adds to the logits:
//
//   - a relative-position bias: bias[d] = rel[h] . P[:, d] for backward distance
//     0 <= d < rel_extent (0 beyond), where rel [d_rel] is the query token's clipped
//     (and, on full layers, log-scaled) relative state and P [d_rel, rel_extent] the
//     layer's profile bank. Each CTA builds its row of rel_extent biases in shared
//     memory before the KV loop.
//   - a per-token sink: one extra logit per head (the sink projection of the query
//     token) with a zero value, which only absorbs probability mass.
//
// Q arrives clipped and scaled (q_l2_clip, log scaling on full layers), K clipped and
// V normed by the projection epilogue, so the score is plain q.k / sqrt(head_dim).
// Reference: transformers PR #51050 `Inkling2RelativePositionBias` and
// `eager_attention_forward` (Apache-2.0); numerics in fp32 here, the reference adds
// the bias in BF16.
//
// Grid: (num_q_heads, num_seqs, 1). Block: (256, 1, 1). HDIM = 128.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 128
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32 (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define MAX_REL_EXTENT 1024
#define MAX_D_REL 32

__device__ __forceinline__ void ink_unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

extern "C" __global__ void inkling2_decode_attn(
    const __nv_bfloat16* __restrict__ Q,        // [num_seqs, q_stride] (heads x HDIM per row)
    const __nv_bfloat16* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, HDIM]
    const __nv_bfloat16* __restrict__ V_cache,  // same layout
    __nv_bfloat16* __restrict__ O,              // [num_seqs, num_q_heads, HDIM]
    const int* __restrict__ block_tables,       // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs], including the current token
    const __nv_bfloat16* __restrict__ rel,      // [num_seqs, num_q_heads, d_rel]
    const __nv_bfloat16* __restrict__ rel_proj, // [d_rel, rel_extent]
    const __nv_bfloat16* __restrict__ sinks,    // [num_seqs, num_q_heads]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride,
    const unsigned int sliding_window,  // 0 = full attention
    const unsigned int d_rel,
    const unsigned int rel_extent
) {
    atlas_pdl_enter();
    const unsigned int q_head = blockIdx.x;
    const unsigned int seq_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    if (q_head >= num_q_heads) return;
    const unsigned int seq_len = (unsigned int)seq_lens[seq_idx];
    if (seq_len == 0) return;

    // bias[d] for d = 0 .. rel_extent - 1
    __shared__ float s_rel[MAX_D_REL];
    __shared__ float s_bias[MAX_REL_EXTENT];
    const __nv_bfloat16* my_rel = rel + ((unsigned long long)seq_idx * num_q_heads + q_head) * d_rel;
    if (tid < d_rel) s_rel[tid] = __bfloat162float(my_rel[tid]);
    __syncthreads();
    for (unsigned int d = tid; d < rel_extent; d += blockDim.x) {
        float b = 0.0f;
        for (unsigned int j = 0; j < d_rel; j++)
            b += s_rel[j] * __bfloat162float(rel_proj[(unsigned long long)j * rel_extent + d]);
        s_bias[d] = b;
    }
    __syncthreads();

    const unsigned int q_pos = seq_len - 1;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);
    const unsigned int vec_offset = lane_id * VEC_BF16;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)seq_idx * q_stride
                                                       + (unsigned long long)q_head * HDIM + vec_offset);
    float q_reg[VEC_BF16];
#pragma unroll
    for (int i = 0; i < VEC_U32; i++) ink_unpack2(q32[i], q_reg[2 * i], q_reg[2 * i + 1]);

    const unsigned int attended = seq_len - window_start;
    const unsigned int chunk = (attended + NUM_WARPS - 1) / NUM_WARPS;
    unsigned int my_start = window_start + warp_id * chunk;
    unsigned int my_end = my_start + chunk;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f, l = 0.0f;
    float o_reg[VEC_BF16];
#pragma unroll
    for (int i = 0; i < VEC_BF16; i++) o_reg[i] = 0.0f;

    const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * HDIM;
    const unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;
    for (unsigned int pos = my_start; pos < my_end; pos++) {
        const unsigned long long base =
            (unsigned long long)(unsigned int)my_block_table[pos / block_size] * page_stride
            + (unsigned long long)(pos % block_size) * head_stride_kv
            + (unsigned long long)kv_head * HDIM + vec_offset;
        const unsigned int* k32 = (const unsigned int*)(K_cache + base);
        const unsigned int* v32 = (const unsigned int*)(V_cache + base);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float k0, k1;
            ink_unpack2(k32[i], k0, k1);
            dot += q_reg[2 * i] * k0 + q_reg[2 * i + 1] * k1;
        }
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        const unsigned int dist = q_pos - pos;
        const float score = dot * inv_sqrt_d + (dist < rel_extent ? s_bias[dist] : 0.0f);
        const float m_new = fmaxf(m, score);
        const float e_old = __expf(m - m_new), e_new = __expf(score - m_new);
        l = l * e_old + e_new;
#pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            float v0, v1;
            ink_unpack2(v32[i], v0, v1);
            o_reg[2 * i] = o_reg[2 * i] * e_old + e_new * v0;
            o_reg[2 * i + 1] = o_reg[2 * i + 1] * e_old + e_new * v1;
        }
        m = m_new;
    }

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS][HDIM];
    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
#pragma unroll
    for (int i = 0; i < VEC_BF16; i++) smem_o[warp_id][vec_offset + i] = o_reg[i];
    __syncthreads();
#pragma unroll
    for (int stride = NUM_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            const unsigned int other = warp_id + stride;
            const float lw = smem_l[other];
            if (lw > 0.0f) {
                const float mw = smem_m[other], my_m = smem_m[warp_id];
                const float m_new = fmaxf(my_m, mw);
                const float s_me = __expf(my_m - m_new), s_w = __expf(mw - m_new);
                smem_l[warp_id] = smem_l[warp_id] * s_me + lw * s_w;
                smem_m[warp_id] = m_new;
#pragma unroll
                for (int i = 0; i < VEC_BF16; i++)
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * s_me + smem_o[other][vec_offset + i] * s_w;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        // the sink: an extra logit with a zero value joins only the normaliser
        const float sink = __bfloat162float(sinks[(unsigned long long)seq_idx * num_q_heads + q_head]);
        const float m_all = fmaxf(smem_m[0], sink);
        const float l_all = smem_l[0] * __expf(smem_m[0] - m_all) + __expf(sink - m_all);
        const float scale = __expf(smem_m[0] - m_all) / l_all;
        unsigned int* o32 = (unsigned int*)(O + ((unsigned long long)seq_idx * num_q_heads + q_head) * HDIM
                                              + vec_offset);
#pragma unroll
        for (int i = 0; i < VEC_U32; i++) {
            const float v0 = smem_o[0][vec_offset + 2 * i] * scale;
            const float v1 = smem_o[0][vec_offset + 2 * i + 1] * scale;
            o32[i] = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0))
                   | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1)) << 16);
        }
    }
}
