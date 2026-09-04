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


// ──────────────────── stage 2: per-query PREFILL selection ────────────────────
//
// Selectivity is monotone in position: every chunk row at global pos >= 2051
// needs its own top-512-block set. Rows are processed as a contiguous range
// [first_pos, first_pos + n_rows); per row the score matrix is masked at the
// row's own complete-block count, host top-k builds a 512-entry block list,
// and qsa_prefill_attn OVERWRITES that row's attention context (pre-gate,
// pre-o_proj) with attention over exactly the selected set — read straight
// from the paged KV cache, so the dense flash pass it replaces needs no
// changes.

// Per-row q prep: RMSNorm*(1+w) + partial rope at pos = first_pos + row.
// qk rows are the indexer projection [rows, (n_heads+1)*hd]; q is the head-
// concatenated prefix of each row. Grid: (rows, n_heads)  Block: (hd,1,1).
extern "C" __global__ void qsa_qprep_rows(
    const __nv_bfloat16* __restrict__ qk,       // [rows, qkw]
    const __nv_bfloat16* __restrict__ q_norm_w, // [hd]
    float* __restrict__ q_out,                  // [rows, n_heads, hd]
    const unsigned int first_pos,
    const unsigned int qkw,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int pos = first_pos + r;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)qk[(size_t)r * qkw + (size_t)hh * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + ((size_t)r * n_heads + hh) * hd;
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

// Per-row block scores. scores[r, b] = sum_h relu(q[r,h] . k_b)/sqrt(hd) for
// b < complete(row), -inf otherwise (host top-k then never picks it).
// Grid: (rows, n_blocks_max)  Block: (hd,1,1).
extern "C" __global__ void qsa_score_rows(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    float* out = scores + (size_t)r * score_stride + b;
    if (b >= complete) {
        if (d == 0) *out = -1e30f;
        return;
    }

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    const float* qr = q + (size_t)r * n_heads * hd;
    float acc = 0.0f;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float dot = qsa_block_reduce_sum(qr[(size_t)hh * hd + d] * k, red);
        if (d == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (d == 0) *out = acc * rsqrtf((float)hd);
}

// Attention over EXACTLY the selected set for one (row, q-head): the listed
// `topk` blocks (ratio tokens each) plus the incomplete tail
// [complete*ratio, pos]. K/V come straight from the paged cache; the output
// OVERWRITES that row's context in attn_out (pre-gate, pre-o_proj), so the
// surrounding dense path needs no other change. Softmax is order-invariant
// and rope is baked into cached K, so this equals the reference mask.
// Grid: (rows, nq)  Block: (256,1,1) = 8 warps, warp-striped online softmax.
#define QSA_PA_WARPS 8
extern "C" __global__ void qsa_prefill_attn(
    const __nv_bfloat16* __restrict__ q,        // [rows, nq, hd] (roped)
    const __nv_bfloat16* __restrict__ k_cache,  // paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // [rows, nq, hd]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int hd,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.x;
    const unsigned int qh = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int kvh = qh / (nq / nkv);
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int vec = hd / 32;           // elems per lane (8 at hd=256)

    extern __shared__ float smem[];
    // Per-warp partials: [warps][hd] acc, then [warps] m, [warps] l.
    float* acc_w = smem;                        // [QSA_PA_WARPS * hd]
    float* m_w = smem + QSA_PA_WARPS * hd;      // [QSA_PA_WARPS]
    float* l_w = m_w + QSA_PA_WARPS;            // [QSA_PA_WARPS]

    // q slice for this (row, head), staged per lane.
    const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh) * hd;
    float qreg[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        qreg[e] = (e < vec) ? (float)qrow[lane * vec + e] : 0.0f;
    }

    float m = -1e30f, l = 0.0f;
    float acc[8];
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) acc[e] = 0.0f;

    const int* my_list = lists + (size_t)r * topk;
    for (unsigned int t = warp; t < n_tok; t += QSA_PA_WARPS) {
        unsigned int tok;
        if (t < topk * ratio) {
            tok = (unsigned int)my_list[t / ratio] * ratio + (t % ratio);
        } else {
            tok = complete * ratio + (t - topk * ratio);
        }
        const unsigned long long off =
            (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
            + (unsigned long long)(tok % block_size) * row_elems
            + (unsigned long long)kvh * hd;
        const __nv_bfloat16* krow = k_cache + off;
        float dot = 0.0f;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) dot += qreg[e] * (float)krow[lane * vec + e];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, o);
        dot = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d;

        const float m_new = fmaxf(m, dot);
        const float scale = __expf(m - m_new);
        const float p = __expf(dot - m_new);
        l = l * scale + p;
        const __nv_bfloat16* vrow = v_cache + off;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) acc[e] = acc[e] * scale + p * (float)vrow[lane * vec + e];
        }
        m = m_new;
    }

    // Park warp partials, then warp 0 merges.
    #pragma unroll
    for (unsigned int e = 0; e < 8; ++e) {
        if (e < vec) acc_w[warp * hd + lane * vec + e] = acc[e];
    }
    if (lane == 0) { m_w[warp] = m; l_w[warp] = l; }
    __syncthreads();

    if (warp == 0) {
        float m_tot = -1e30f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) m_tot = fmaxf(m_tot, m_w[w]);
        float l_tot = 0.0f;
        float out[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            const float s = __expf(m_w[w] - m_tot);
            l_tot += l_w[w] * s;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) out[e] += acc_w[w * hd + lane * vec + e] * s;
            }
        }
        const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
        __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) orow[lane * vec + e] = __float2bfloat16(out[e] * inv_l);
        }
    }
}

// ── Device-side decode block selection ─────────────────────────────────
// The `block_topk` largest of `scores[0..complete)` with torch.topk's
// tie-break (equal scores: the LOWER index wins), emitted in ASCENDING block
// order and expanded to token ids (`b*ratio + r`), followed by the partial
// block's tail `tail_start..visible`. Replaces the per-layer D2H copy + host
// sort + H2D upload (`QsaIndexer::decode_select`); one block of threads,
// `complete <= QSA_SELECT_MAX_BLOCKS` (the host arm covers anything wider).
// Rank of block b = #{j : s_j > s_b or (s_j == s_b and j < b)}; selected iff
// rank < block_topk. NaN scores are never selected.
#define QSA_SELECT_MAX_BLOCKS 4096
extern "C" __global__ void qsa_select_topk(
    const float* __restrict__ scores,   // [complete]
    int* __restrict__ sel,              // [block_topk*ratio + (visible - tail_start)]
    int complete,
    int block_topk,
    int ratio,
    int tail_start,
    int visible)
{
    __shared__ unsigned char selected[QSA_SELECT_MAX_BLOCKS];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    for (int b = tid; b < complete; b += nt) {
        const float sb = scores[b];
        int rank = 0;
        if (!isnan(sb)) {
            for (int j = 0; j < complete; ++j) {
                const float sj = scores[j];
                if (isnan(sj)) continue;
                if (sj > sb || (sj == sb && j < b)) ++rank;
            }
        }
        selected[b] = (!isnan(sb) && rank < block_topk) ? 1 : 0;
    }
    __syncthreads();
    for (int b = tid; b < complete; b += nt) {
        if (!selected[b]) continue;
        int pos = 0;
        for (int j = 0; j < b; ++j) pos += selected[j];
        const int base = pos * ratio;
        for (int r = 0; r < ratio; ++r) sel[base + r] = b * ratio + r;
    }
    for (int t = tail_start + tid; t < visible; t += nt) {
        sel[block_topk * ratio + (t - tail_start)] = t;
    }
}
