// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next QSA indexer — the decode-side selection machinery.
//
// Reference: modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer. Per query, the
// visible prefix is grouped into `ratio`(=4)-token blocks; each block's key
// is the MEAN of its raw per-token indexer keys, then k_layernorm
// (offset-from-1 RMSNorm), then partial rope at the block's FIRST token
// position. Scores are sum_h relu(q_h . k_b) / sqrt(head_dim); the top
// `block_topk` blocks plus the incomplete tail are the visible set.
//
// Selection feeds the EXISTING paged decode attention: qsa_gather packs the
// selected tokens' K/V rows into a contiguous scratch laid out NHD
// ([page, slot, kv_head, dim]) so an identity block table over the scratch
// reproduces the reference mask semantics with zero new attention code.
//
// Rope here is computed INLINE in double precision (32 freq lanes,
// inv_freq_j = theta^(-2j/rot)) rather than read from the attention rope
// tables — the golden's cos/sin come from torch fp32 and double sincos
// keeps the parity comparison out of ulp territory. Text-only mrope with
// equal position grids reduces to exactly this.

#include <cuda_bf16.h>

__device__ __forceinline__ float qsa_block_reduce_sum(float v, float* red) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        for (unsigned int w = 0; w < warps; ++w) tot += red[w];
        red[0] = tot;
    }
    __syncthreads();
    return red[0];
}

// normed (already in smem, length hd) -> rope at `pos` -> out (bf16).
// Assumes hd threads; rot must be even, pairs are (j, j + rot/2).
__device__ __forceinline__ void qsa_rope_store(
    const float* normed, __nv_bfloat16* out,
    unsigned int d, unsigned int rot, unsigned int pos, float theta
) {
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = normed[j];
        const float x2 = normed[j + half];
        const float v = (d < half) ? (x1 * (float)c - x2 * (float)s)
                                   : (x2 * (float)c + x1 * (float)s);
        out[d] = __float2bfloat16(v);
    } else {
        out[d] = __float2bfloat16(normed[d]);
    }
}

// ── qsa_block_pool ──
// Pool `n_new` freshly COMPLETE blocks starting at `first_block`:
// mean(ratio raw keys) -> RMSNorm*(1+w) -> rope at pos = block*ratio.
// Appends into block_keys [*, hd]. Grid: (n_new,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_block_pool(
    const __nv_bfloat16* __restrict__ raw_keys,   // [S, hd]
    const __nv_bfloat16* __restrict__ k_norm_w,   // [hd]
    __nv_bfloat16* __restrict__ block_keys,       // [max_blocks, hd]
    const unsigned int first_block,
    const unsigned int ratio,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int b = first_block + blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];               // [hd] normed + red
    float* stage = smem;
    float* red = smem + hd;

    float v = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) {
        v += (float)raw_keys[(size_t)(b * ratio + r) * hd + d];
    }
    v /= (float)ratio;

    const float sq = qsa_block_reduce_sum(v * v, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = v * rms * (1.0f + (float)k_norm_w[d]);
    __syncthreads();

    qsa_rope_store(stage, block_keys + (size_t)b * hd, d, rot, b * ratio, theta);
}

// ── qsa_qprep ──
// One decode query: per head, RMSNorm*(1+w) then rope at `pos`.
// q_in is the head-concatenated slice of the qk projection row.
// Grid: (n_heads,1,1)  Block: (hd,1,1). Output FP32 (feeds the scorer).
extern "C" __global__ void qsa_qprep(
    const __nv_bfloat16* __restrict__ q_in,       // [n_heads, hd]
    const __nv_bfloat16* __restrict__ q_norm_w,   // [hd]
    float* __restrict__ q_out,                    // [n_heads, hd]
    const unsigned int hd,
    const unsigned int rot,
    const unsigned int pos,
    const float theta,
    const float eps
) {
    const unsigned int h = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)q_in[(size_t)h * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + (size_t)h * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// ── qsa_score ──
// scores[b] = sum_h relu(q_h . k_b) / sqrt(hd).
// Grid: (n_blocks,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_score(
    const float* __restrict__ q,                  // [n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys, // [*, hd]
    float* __restrict__ scores,                   // [n_blocks]
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_heads; ++h) {
        const float dot = qsa_block_reduce_sum(q[(size_t)h * hd + d] * k, red);
        if (threadIdx.x == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[b] = acc * rsqrtf((float)hd);
    }
}

// ── qsa_gather ──
// Pack the selected tokens' K/V rows (NHD paged layout) into contiguous
// scratch: dst slot i holds src position sel[i]. The scratch, viewed through
// an identity block table, IS a valid paged cache for the existing decode
// attention kernel. Grid: (n_sel,1,1)  Block: (256,1,1).
extern "C" __global__ void qsa_gather(
    const __nv_bfloat16* __restrict__ k_cache,    // [blocks, bs, nkv, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,          // logical -> physical
    const int* __restrict__ sel,                  // [n_sel] token positions
    __nv_bfloat16* __restrict__ k_out,            // [n_sel(padded), nkv, hd]
    __nv_bfloat16* __restrict__ v_out,
    const unsigned int block_size,
    const unsigned int nkv,
    const unsigned int hd
) {
    const unsigned int i = blockIdx.x;
    const unsigned int pos = (unsigned int)sel[i];
    const unsigned int row = nkv * hd;
    const unsigned long long page_stride =
        (unsigned long long)block_size * row;
    const unsigned long long src_off =
        (unsigned long long)(unsigned int)block_table[pos / block_size] * page_stride
        + (unsigned long long)(pos % block_size) * row;
    const unsigned long long dst_off = (unsigned long long)i * row;
    for (unsigned int e = threadIdx.x; e < row; e += blockDim.x) {
        k_out[dst_off + e] = k_cache[src_off + e];
        v_out[dst_off + e] = v_cache[src_off + e];
    }
}
