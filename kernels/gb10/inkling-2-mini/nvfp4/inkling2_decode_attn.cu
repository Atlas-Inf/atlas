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
// Every row is one query at positions[r] over keys 0 .. positions[r]: decode rows of
// several sequences (one block table each), or prefill rows of one sequence sharing one
// table (max_blocks_per_seq = 0).
//
// Two launches. inkling2_decode_attn: grid (num_kv_heads, splits, rows), 256 threads; a CTA
// takes one KV head's GQA group of query heads over 1/splits of the row's attended keys
// (sized from the device positions, so the grid is graph-static), one warp per key at a
// time, each K/V load shared by the group; the relative bias joins the same warp
// reduction as q.k. It writes per-split (max, sum, unnormalised output) partials.
// inkling2_decode_attn_combine: grid (num_q_heads, rows), 128 threads; merges the splits
// and the sink and writes BF16. Decode uses many splits; prefill rows one. HDIM = 128.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define WARP_SIZE 32
#ifndef HDIM
#define HDIM 128
#endif
#define VEC_BF16 (HDIM / WARP_SIZE)
#define VEC_U32 (HDIM / (WARP_SIZE * 2))
#define NUM_WARPS 8
#define MAX_GROUP 4   // query heads per KV head
#define MIN_SPLIT_KEYS 64u
#define MAX_REL_EXTENT 1024
#define MAX_D_REL 32

__device__ __forceinline__ void ink_unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// partials[((row * num_q_heads + head) * splits + split) * (HDIM + 2)]: m, l, o[HDIM]
extern "C" __global__ void inkling2_decode_attn(
    const __nv_bfloat16* __restrict__ Q,        // [rows, q_stride] (heads x HDIM per row)
    const __nv_bfloat16* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, HDIM]
    const __nv_bfloat16* __restrict__ V_cache,  // same layout
    float* __restrict__ partials,
    const int* __restrict__ block_tables,       // row r at r * max_blocks_per_seq (0: one shared table)
    const int* __restrict__ positions,          // [rows] query positions (u32 or i32)
    const __nv_bfloat16* __restrict__ rel,      // [rows, num_q_heads, d_rel]
    const __nv_bfloat16* __restrict__ rel_proj, // [d_rel, rel_extent]
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
    const unsigned int kv_head = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
    const unsigned long long row = blockIdx.z;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int group = num_q_heads / num_kv_heads;
    const unsigned int seq_len = (unsigned int)positions[row] + 1;
    const unsigned int q_pos = seq_len - 1;
    const unsigned int window_start =
        (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0u;
    // at least MIN_SPLIT_KEYS keys a split: a short context uses few splits, the rest
    // write an empty partial and leave
    const unsigned int per_split = max((seq_len - window_start + splits - 1) / splits, MIN_SPLIT_KEYS);
    const unsigned int s_start = min(window_start + split * per_split, seq_len);
    const unsigned int s_end = min(s_start + per_split, seq_len);
    if (s_start >= s_end) {
        const unsigned int g = threadIdx.x;
        if (g < group) {
            float* out = partials + ((row * num_q_heads + kv_head * group + g) * splits + split) * (HDIM + 2);
            out[0] = -1e30f;
            out[1] = 0.0f;
        }
        return;
    }

    const unsigned int vec_offset = lane_id * VEC_BF16;
    float q_reg[MAX_GROUP][VEC_BF16], rel_reg[MAX_GROUP];
#pragma unroll
    for (unsigned int g = 0; g < MAX_GROUP; g++) {
        if (g >= group) break;
        const unsigned int h = kv_head * group + g;
        const unsigned int* q32 = (const unsigned int*)(Q + row * q_stride + (unsigned long long)h * HDIM + vec_offset);
#pragma unroll
        for (int i = 0; i < VEC_U32; i++) ink_unpack2(q32[i], q_reg[g][2 * i], q_reg[g][2 * i + 1]);
        rel_reg[g] = lane_id < d_rel ? __bfloat162float(rel[(row * num_q_heads + h) * d_rel + lane_id]) : 0.0f;
    }

    float m[MAX_GROUP], l[MAX_GROUP], o[MAX_GROUP][VEC_BF16];
#pragma unroll
    for (unsigned int g = 0; g < MAX_GROUP; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC_BF16; i++) o[g][i] = 0.0f;
    }
    const int* table = block_tables + row * max_blocks_per_seq;
    const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * HDIM;
    const unsigned long long page_stride = (unsigned long long)block_size * head_stride_kv;
    for (unsigned int pos = s_start + warp_id; pos < s_end; pos += NUM_WARPS) {
        const unsigned long long base = (unsigned long long)(unsigned int)table[pos / block_size] * page_stride
                                      + (unsigned long long)(pos % block_size) * head_stride_kv
                                      + (unsigned long long)kv_head * HDIM + vec_offset;
        const uint2 kv = *(const uint2*)(K_cache + base);
        const uint2 vv = *(const uint2*)(V_cache + base);
        float k[VEC_BF16], v[VEC_BF16];
        ink_unpack2(kv.x, k[0], k[1]);
        ink_unpack2(kv.y, k[2], k[3]);
        ink_unpack2(vv.x, v[0], v[1]);
        ink_unpack2(vv.y, v[2], v[3]);
        const unsigned int dist = q_pos - pos;
        const float pj = (dist < rel_extent && lane_id < d_rel)
                             ? __bfloat162float(rel_proj[(unsigned long long)lane_id * rel_extent + dist]) : 0.0f;
#pragma unroll
        for (unsigned int g = 0; g < MAX_GROUP; g++) {
            if (g >= group) break;
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < VEC_BF16; i++) part += q_reg[g][i] * k[i];
            // q.k / sqrt(d) and the bias rel . rel_proj[:, dist] in one reduction
            float sc = part * inv_sqrt_d + rel_reg[g] * pj;
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sc += __shfl_xor_sync(0xffffffff, sc, off);
            const float m_new = fmaxf(m[g], sc);
            const float e_old = __expf(m[g] - m_new), e_new = __expf(sc - m_new);
            l[g] = l[g] * e_old + e_new;
#pragma unroll
            for (int i = 0; i < VEC_BF16; i++) o[g][i] = o[g][i] * e_old + e_new * v[i];
            m[g] = m_new;
        }
    }

    // merge the warps, then one partial per query head
    __shared__ float smem_m[NUM_WARPS][MAX_GROUP];
    __shared__ float smem_l[NUM_WARPS][MAX_GROUP];
    __shared__ float smem_o[NUM_WARPS][MAX_GROUP][HDIM];
#pragma unroll
    for (unsigned int g = 0; g < MAX_GROUP; g++) {
        if (g >= group) break;
        if (lane_id == 0) {
            smem_m[warp_id][g] = m[g];
            smem_l[warp_id][g] = l[g];
        }
#pragma unroll
        for (int i = 0; i < VEC_BF16; i++) smem_o[warp_id][g][vec_offset + i] = o[g][i];
    }
    __syncthreads();
    // thread t: query head t / HDIM of the group, dimension t % HDIM (256 threads: 2 heads a pass)
    for (unsigned int t = threadIdx.x; t < group * HDIM; t += blockDim.x) {
        const unsigned int g = t / HDIM, d = t % HDIM;
        float mm = -1e30f;
        for (unsigned int w = 0; w < NUM_WARPS; w++) mm = fmaxf(mm, smem_m[w][g]);
        float ll = 0.0f, oo = 0.0f;
        for (unsigned int w = 0; w < NUM_WARPS; w++) {
            const float sc = smem_l[w][g] > 0.0f ? __expf(smem_m[w][g] - mm) : 0.0f;
            ll += smem_l[w][g] * sc;
            oo += smem_o[w][g][d] * sc;
        }
        float* out = partials + ((row * num_q_heads + kv_head * group + g) * splits + split) * (HDIM + 2);
        if (d == 0) {
            out[0] = mm;
            out[1] = ll;
        }
        out[2 + d] = oo;
    }
}

extern "C" __global__ void inkling2_decode_attn_combine(
    const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ sinks,    // [rows, num_q_heads]
    __nv_bfloat16* __restrict__ O,              // [rows, num_q_heads, HDIM]
    const unsigned int num_q_heads,
    const unsigned int splits
) {
    atlas_pdl_enter();
    const unsigned int head = blockIdx.x, d = threadIdx.x;
    const unsigned long long row = blockIdx.y;
    const float* p = partials + (row * num_q_heads + head) * splits * (HDIM + 2);
    // the sink: an extra logit with a zero value joins only the normaliser
    const float sink = __bfloat162float(sinks[row * num_q_heads + head]);
    float mm = sink;
    for (unsigned int s = 0; s < splits; s++) mm = fmaxf(mm, p[s * (HDIM + 2)]);
    float ll = __expf(sink - mm), oo = 0.0f;
    for (unsigned int s = 0; s < splits; s++) {
        const float* q = p + s * (HDIM + 2);
        if (q[1] <= 0.0f) continue;
        const float sc = __expf(q[0] - mm);
        ll += q[1] * sc;
        oo += q[2 + d] * sc;
    }
    O[(row * num_q_heads + head) * HDIM + d] = __float2bfloat16(oo / ll);
}
