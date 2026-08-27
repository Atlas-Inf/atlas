// SPDX-License-Identifier: AGPL-3.0-only

// Single-token attention for qwen4_exp, over a contiguous K/V buffer.
//
// Atlas's production attention is paged; this is the decode-step arithmetic on
// a flat buffer, which is what a bring-up needs before paging is wired in.
//
// The gate is what makes this model's attention different: q_proj emits
// [query | gate] PER HEAD, so its head stride is 2*head_dim, and the gate is
// applied ELEMENTWISE to the attention output before o_proj -- not as a
// softmax bias, not as a per-head scalar.
//
// Reference: Qwen4ExpTextAttention in modeling_qwen4_exp.py, and the CPU oracle
// in atlas_core::qwen4exp_reference (checked against it at 8.0e-7).

#include <cuda_bf16.h>

__device__ __forceinline__ float q4e_attn_warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off));
    return v;
}

__device__ __forceinline__ float q4e_attn_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

// One block per query head. Threads span head_dim.
//
// Grid: (num_heads, 1, 1)  Block: (head_dim, 1, 1)
extern "C" __global__ void q4e_attn_decode(
    const __nv_bfloat16* __restrict__ query,   // [num_heads, head_dim] post-norm, post-rope
    const __nv_bfloat16* __restrict__ gate,    // [num_heads, head_dim] pre-sigmoid
    const __nv_bfloat16* __restrict__ k_cache, // [seq_len, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_cache, // [seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ out,            // [num_heads, head_dim]
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int seq_len
) {
    unsigned int head = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (head >= num_heads || tid >= head_dim) return;

    unsigned int group = num_heads / num_kv_heads;
    unsigned int kv_head = head / group;
    float scale = rsqrtf((float)head_dim);

    const __nv_bfloat16* q = query + (unsigned long long)head * head_dim;
    float q_i = __bfloat162float(q[tid]);

    extern __shared__ float smem[];
    float* scores = smem; // [seq_len]

    // Scores: one dot product per past position, reduced across the block.
    for (unsigned int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* k =
            k_cache + ((unsigned long long)t * num_kv_heads + kv_head) * head_dim;
        float partial = q_i * __bfloat162float(k[tid]);

        partial = q4e_attn_warp_sum(partial);
        __shared__ float warp_parts[32];
        unsigned int lane = tid & 31;
        unsigned int warp = tid >> 5;
        if (lane == 0) warp_parts[warp] = partial;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < (blockDim.x + 31) / 32) ? warp_parts[lane] : 0.0f;
            v = q4e_attn_warp_sum(v);
            if (lane == 0) scores[t] = v * scale;
        }
        __syncthreads();
    }

    // Softmax over the causal window, computed once per block. The running max
    // is a local: it is only needed to shift the exponents, and every consumer
    // of the result reads the already-shifted `scores` and their sum.
    __shared__ float soft_sum;
    if (tid == 0) {
        float m = -INFINITY;
        for (unsigned int t = 0; t < seq_len; ++t) m = fmaxf(m, scores[t]);
        float s = 0.0f;
        for (unsigned int t = 0; t < seq_len; ++t) {
            scores[t] = expf(scores[t] - m);
            s += scores[t];
        }
        soft_sum = s;
    }
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int t = 0; t < seq_len; ++t) {
        const __nv_bfloat16* v =
            v_cache + ((unsigned long long)t * num_kv_heads + kv_head) * head_dim;
        acc += scores[t] / soft_sum * __bfloat162float(v[tid]);
    }

    // Elementwise sigmoid gate, applied BEFORE o_proj.
    float g = __bfloat162float(gate[(unsigned long long)head * head_dim + tid]);
    out[(unsigned long long)head * head_dim + tid] =
        __float2bfloat16(acc * (1.0f / (1.0f + expf(-g))));
}
