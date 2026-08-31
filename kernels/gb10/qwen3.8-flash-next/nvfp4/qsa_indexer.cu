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

// Same scores, several per block.
//
// `qsa_score_rows` launches ONE 128-thread block per OUTPUT SCALAR. At 30k that
// is 1.397 BILLION blocks across the 12 full-attention layers, each computing
// `n_heads * hd` = 512 MACs and paying four block-wide reductions for them. The
// kernel measured 1.43 TFLOP in 4.860 s = **294 GFLOP/s**, against 28.0 TFLOP/s
// for the MoE grouped GEMM in the same prefill -- the same GPU, 95x apart.
//
// This variant gives each block QSA_SR_B consecutive `b` values and stages the
// row's `q` in shared once instead of re-reading it per block. Block count
// drops by QSA_SR_B; the arithmetic does not change at all.
//
// BIT-IDENTICAL, deliberately, because these scores feed a top-k and a shifted
// score can change WHICH blocks are attended -- not just by how much. Every
// output still contracts with the same `qsa_block_reduce_sum` over the same 128
// threads in the same tree, accumulates `fmaxf(dot, 0)` over `hh` in the same
// order, and scales by the same `rsqrtf(hd)`. Staging `q` through shared moves
// where the float is read from, never its value. A GEMM formulation would be
// far faster still, but it reassociates the contraction and so needs the
// precision gate; this does not.
#define QSA_SR_B 16
extern "C" __global__ void qsa_score_rows_b(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int n_blocks_max
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b0 = blockIdx.y * QSA_SR_B;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;

    extern __shared__ float smem[];
    float* red = smem;              // reduction scratch, as in qsa_score_rows
    float* qs = smem + 32;          // [n_heads, hd], staged once per block

    const float* qr = q + (size_t)r * n_heads * hd;
    for (unsigned int h = 0; h < n_heads; ++h) {
        qs[h * hd + d] = qr[(size_t)h * hd + d];
    }
    __syncthreads();

    float* srow = scores + (size_t)r * score_stride;
    // Both branch conditions below are block-UNIFORM (they depend only on
    // blockIdx and the loop counter), so every thread runs the same sequence of
    // `__syncthreads` inside the reduction.
    for (unsigned int i = 0; i < QSA_SR_B; ++i) {
        const unsigned int b = b0 + i;
        if (b >= n_blocks_max) {
            break;
        }
        if (b >= complete) {
            if (d == 0) {
                srow[b] = -1e30f;
            }
            continue;
        }
        const float k = (float)block_keys[(size_t)b * hd + d];
        float acc = 0.0f;
        for (unsigned int hh = 0; hh < n_heads; ++hh) {
            const float dot = qsa_block_reduce_sum(qs[hh * hd + d] * k, red);
            if (d == 0) {
                acc += fmaxf(dot, 0.0f);
            }
            __syncthreads();
        }
        if (d == 0) {
            srow[b] = acc * rsqrtf((float)hd);
        }
    }
}

// Attention over EXACTLY the selected set for one (row, q-head): the listed
// `topk` blocks (ratio tokens each) plus the incomplete tail
// [complete*ratio, pos]. K/V come straight from the paged cache; the output
// OVERWRITES that row's context in attn_out (pre-gate, pre-o_proj), so the
// surrounding dense path needs no other change. Softmax is order-invariant
// and rope is baked into cached K, so this equals the reference mask.
// Grid: (rows, nq)  Block: (256,1,1) = 8 warps, warp-striped online softmax.
// ── Per-row top-k block selection, on the GPU ───────────────────────────
//
// WHY THIS EXISTS. The selection used to run on the HOST: copy the whole
// score matrix D2H, sort each row, copy the list back. That is a full stream
// drain per attention layer per slab, and it is invisible to any kernel-time
// profile because while it runs no kernel is running. Measuring GPU IDLE
// instead (scripts/gaps.py), on a 30k prefill:
//
//   qsa_score_rows -> qsa_prefill_attn_g   7279 ms   179 gaps   40.7 ms each
//
// 19% of a 41.6 s window, and the largest single item in it. The cost is not
// the sort algorithm -- that is already O(n) and multi-threaded -- it is
// shipping `rows x stride` floats (60.8 MB per slab at 30k) across the bus and
// touching them again on the CPU.
//
// EXACTNESS. The list ORDER matters, not just its contents: `qsa_prefill_attn`
// walks it warp-striped and its online softmax accumulates in list order. So
// this must reproduce the host's `(score DESCENDING, index ASCENDING)` order
// exactly. It does so by construction rather than by argument: each element
// becomes ONE u64 (`qsa_rank_key`, the same monotone f32->u32 map the Rust
// side uses, inverted for descending, index in the low bits), the keys are
// DISTINCT because the index is in them, and any correct selection-and-sort of
// distinct integers has exactly one answer. There is no floating-point
// reassociation anywhere in here to get wrong.
//
// Bitonic top-K: keep a running ascending array of the K best keys, and for
// each chunk of K new keys sort them ascending, take `min(A[i], B[K-1-i])`
// (the K smallest of the union, and bitonic by the standard result), then a
// bitonic merge restores ascending order. K = blockDim.x.
#define QSA_TOPK_K 512

__device__ __forceinline__ unsigned long long qsa_rank_key(float s, unsigned int idx) {
    // -0.0f and +0.0f have different bit patterns but compare Equal in IEEE,
    // so canonicalise before the bit map or the order would differ from the
    // host comparator on exactly that value. Keep in sync with
    // `qsa_select.rs::rank_key`.
    const unsigned int b = (s == 0.0f) ? 0u : __float_as_uint(s);
    const unsigned int mono = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)(~mono) << 32) | (unsigned long long)idx;
}

extern "C" __global__ void qsa_topk_rows(
    const float* __restrict__ scores,   // [rows, score_stride]
    int* __restrict__ lists,            // [rows, topk]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int topk
) {
    __shared__ unsigned long long s[2 * QSA_TOPK_K];
    const unsigned int r = blockIdx.x;
    const unsigned int t = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    const float* row = scores + (size_t)r * score_stride;

    // Running best-K, seeded with the worst possible key.
    s[t] = 0xFFFFFFFFFFFFFFFFull;
    __syncthreads();

    for (unsigned int base = 0; base < complete; base += QSA_TOPK_K) {
        const unsigned int j = base + t;
        s[QSA_TOPK_K + t] = (j < complete) ? qsa_rank_key(row[j], j)
                                           : 0xFFFFFFFFFFFFFFFFull;
        __syncthreads();

        // Bitonic sort of the incoming chunk, ascending, in place.
        for (unsigned int k = 2; k <= QSA_TOPK_K; k <<= 1) {
            for (unsigned int j2 = k >> 1; j2 > 0; j2 >>= 1) {
                const unsigned int ixj = t ^ j2;
                if (ixj > t) {
                    const bool up = ((t & k) == 0);
                    const unsigned long long a = s[QSA_TOPK_K + t];
                    const unsigned long long b = s[QSA_TOPK_K + ixj];
                    if ((a > b) == up) {
                        s[QSA_TOPK_K + t] = b;
                        s[QSA_TOPK_K + ixj] = a;
                    }
                }
                __syncthreads();
            }
        }

        // K smallest of the union of two ascending runs: min(A[i], B[K-1-i]).
        // The result is bitonic, so one bitonic merge restores ascending.
        const unsigned long long a = s[t];
        const unsigned long long b = s[2 * QSA_TOPK_K - 1 - t];
        __syncthreads();
        s[t] = (a < b) ? a : b;
        __syncthreads();
        for (unsigned int j2 = QSA_TOPK_K >> 1; j2 > 0; j2 >>= 1) {
            const unsigned int ixj = t ^ j2;
            if (ixj > t) {
                const unsigned long long x = s[t];
                const unsigned long long y = s[ixj];
                if (x > y) {
                    s[t] = y;
                    s[ixj] = x;
                }
            }
            __syncthreads();
        }
    }

    if (t < topk) {
        lists[(size_t)r * topk + t] = (int)(unsigned int)(s[t] & 0xFFFFFFFFull);
    }
}

#define QSA_PA_WARPS 8

// ── G q-heads per block ──────────────────────────────────────────────────
//
// The selected block list is per ROW (`lists + r * topk`), NOT per head, so
// every q head of a row attends over the IDENTICAL key set. `qsa_prefill_attn`
// launches grid=[rows, nq] and therefore re-reads those K/V rows once per q
// head: nq=24 over nkv=2 means twelve reads of every byte.
//
// nsys, 11066-token prefill (2026-08-30): qsa_prefill_attn was 3.70 s, 32.6%
// of a 12.3 s window and the largest kernel in it by a factor of three, moving
// ~51 GB of L2 traffic per launch at ~1 TB/s. Bandwidth, not arithmetic --
// the whole selected K/V set for a layer is only a few MB.
//
// Serving QSA_PA_G heads per block divides that traffic by QSA_PA_G, and gives
// each warp G independent dot-product chains per loaded key instead of one.
// The ceiling is the merge buffer, [QSA_PA_WARPS][G][hd] floats: at hd=128,
// G=12 is 49 KB and busts the 48 KB block limit; G=4 is 16 KB.
//
// BIT-IDENTICAL. Each (row, head) still walks the same warp-striped `t`
// sequence in the same order, keeps its own online-softmax state, and merges
// across the same 8 warps in the same order. Only the LOADS are shared, and
// the K/V values are converted to float before use exactly as before. The
// launcher falls back to the one-head kernel when the head geometry does not
// divide evenly or the merge buffer would not fit.
#define QSA_PA_G 4
extern "C" __global__ void qsa_prefill_attn_g(
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
    const unsigned int qh0 = blockIdx.y * QSA_PA_G;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    // Every head in the group maps to the same kv head; the launcher only
    // dispatches here when QSA_PA_G divides nq / nkv, which guarantees it.
    const unsigned int kvh = qh0 / (nq / nkv);
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int vec = hd / 32;

    extern __shared__ float smem[];
    float* acc_w = smem;                                    // [WARPS][G][hd]
    float* m_w = smem + QSA_PA_WARPS * QSA_PA_G * hd;       // [WARPS][G]
    float* l_w = m_w + QSA_PA_WARPS * QSA_PA_G;             // [WARPS][G]

    float qreg[QSA_PA_G][8];
    #pragma unroll
    for (unsigned int g = 0; g < QSA_PA_G; ++g) {
        const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh0 + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            qreg[g][e] = (e < vec) ? (float)qrow[lane * vec + e] : 0.0f;
        }
    }

    float m[QSA_PA_G], l[QSA_PA_G], acc[QSA_PA_G][8];
    #pragma unroll
    for (unsigned int g = 0; g < QSA_PA_G; ++g) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) acc[g][e] = 0.0f;
    }

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
        // ONE K row and ONE V row for all QSA_PA_G heads -- this is the point.
        const __nv_bfloat16* krow = k_cache + off;
        const __nv_bfloat16* vrow = v_cache + off;
        float kreg[8], vreg[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            kreg[e] = (e < vec) ? (float)krow[lane * vec + e] : 0.0f;
            vreg[e] = (e < vec) ? (float)vrow[lane * vec + e] : 0.0f;
        }
        #pragma unroll
        for (unsigned int g = 0; g < QSA_PA_G; ++g) {
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) dot += qreg[g][e] * kreg[e];
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xFFFFFFFFu, dot, o);
            dot = __shfl_sync(0xFFFFFFFFu, dot, 0) * inv_sqrt_d;

            const float m_new = fmaxf(m[g], dot);
            const float scale = __expf(m[g] - m_new);
            const float p = __expf(dot - m_new);
            l[g] = l[g] * scale + p;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) acc[g][e] = acc[g][e] * scale + p * vreg[e];
            }
            m[g] = m_new;
        }
    }

    #pragma unroll
    for (unsigned int g = 0; g < QSA_PA_G; ++g) {
        float* dst = acc_w + ((size_t)warp * QSA_PA_G + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) dst[lane * vec + e] = acc[g][e];
        }
        if (lane == 0) {
            m_w[warp * QSA_PA_G + g] = m[g];
            l_w[warp * QSA_PA_G + g] = l[g];
        }
    }
    __syncthreads();

    // One warp per head merges its own partials -- the per-head `w` order is
    // the same 0..WARPS-1 the single-head kernel used.
    if (warp < QSA_PA_G) {
        const unsigned int g = warp;
        float m_tot = -1e30f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            m_tot = fmaxf(m_tot, m_w[w * QSA_PA_G + g]);
        }
        float l_tot = 0.0f;
        float out[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
        for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
            const float s = __expf(m_w[w * QSA_PA_G + g] - m_tot);
            l_tot += l_w[w * QSA_PA_G + g] * s;
            const float* srcw = acc_w + ((size_t)w * QSA_PA_G + g) * hd;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) out[e] += srcw[lane * vec + e] * s;
            }
        }
        const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
        __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh0 + g) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) orow[lane * vec + e] = __float2bfloat16(out[e] * inv_l);
        }
    }
}

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
