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

// Block scores, one thread per score, BIT-IDENTICAL to `qsa_score_rows`.
//
// The tiled-GEMM variant below is fast but reassociates: it sums `d` serially
// per thread, which is a different FP addition tree, and these scores feed a
// top-k so near-ties around the 512th block flip (measured: 4.2e-7 relative
// score drift -> top-1 agreement 69.3%).
//
// That tension is avoidable. Bit-identity does not require a 128-thread
// block-wide reduction — it requires evaluating the same FP addition DAG. One
// thread can evaluate the reference's `__shfl_down_sync` tree locally, with no
// reductions and no `__syncthreads` at all.
//
// The reference, at blockDim.x = hd = 128 (4 warps), lane-0 of each warp after
// offsets 16,8,4,2,1 holds:
//
//   b[i] = (x[i] + x[i+16]) + (x[i+8] + x[i+24])   i = 0..7
//   c[i] = b[i] + b[i+4]                           i = 0..3
//   e[i] = c[i] + c[i+2]                           i = 0..1
//   p    = e[0] + e[1]
//
// then thread 0 folds the four warp partials starting from 0.0f, and the caller
// folds `fmaxf(dot, 0)` over `hh` starting from 0.0f. All of that is reproduced
// below in order.
//
// `__fmul_rn` / `__fadd_rn` are load-bearing, not decoration: with plain `*`
// and `+` nvcc contracts `q*k + t` into an FMA, which skips the rounding of the
// product and breaks bit-identity. The reference cannot contract because its
// addend comes from a shuffle of the rounded product.
#define QSA_SE_BM 8
#define QSA_SE_BN 32
extern "C" __global__ void qsa_score_rows_exact(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rows,
    const unsigned int n_blocks_max
) {
    const unsigned int r0 = blockIdx.x * QSA_SE_BM;
    const unsigned int b0 = blockIdx.y * QSA_SE_BN;
    const unsigned int tid = threadIdx.x;
    const unsigned int ldk = hd + 1u;            // pad: see the GEMM variant

    extern __shared__ float smem[];
    float* qs = smem;                                     // [BM][n_heads][hd]
    float* ks = smem + (size_t)QSA_SE_BM * n_heads * hd;  // [BN][hd + 1]

    const unsigned int qn = QSA_SE_BM * n_heads * hd;
    for (unsigned int i = tid; i < qn; i += blockDim.x) {
        const unsigned int rr = i / (n_heads * hd);
        const unsigned int rest = i - rr * n_heads * hd;
        const unsigned int r = r0 + rr;
        qs[i] = (r < rows) ? q[(size_t)r * n_heads * hd + rest] : 0.0f;
    }
    const unsigned int kn = QSA_SE_BN * hd;
    for (unsigned int i = tid; i < kn; i += blockDim.x) {
        const unsigned int bb = i / hd;
        const unsigned int d = i - bb * hd;
        const unsigned int b = b0 + bb;
        ks[bb * ldk + d] =
            (b < n_blocks_max) ? (float)block_keys[(size_t)b * hd + d] : 0.0f;
    }
    __syncthreads();

    const unsigned int ii = tid / QSA_SE_BN;
    const unsigned int jj = tid - ii * QSA_SE_BN;
    const unsigned int r = r0 + ii;
    const unsigned int b = b0 + jj;
    if (r >= rows || b >= n_blocks_max) {
        return;
    }
    float* out = scores + (size_t)r * score_stride + b;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    if (b >= complete) {
        *out = -1e30f;
        return;
    }

    const float* qrow = qs + (size_t)ii * n_heads * hd;
    const float* krow = ks + (size_t)jj * ldk;
    const unsigned int groups = hd >> 5;         // one per warp of the reference

    float acc = 0.0f;                            // reference starts at 0.0f
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float* qh = qrow + (size_t)hh * hd;
        float dot = 0.0f;                        // reference starts at 0.0f
        for (unsigned int g = 0; g < groups; ++g) {
            const float* qg = qh + g * 32u;
            const float* kg = krow + g * 32u;
            float t[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                // offsets 16 then 8, in the reference's associativity
                const float x0 = __fmul_rn(qg[i], kg[i]);
                const float x1 = __fmul_rn(qg[i + 16], kg[i + 16]);
                const float x2 = __fmul_rn(qg[i + 8], kg[i + 8]);
                const float x3 = __fmul_rn(qg[i + 24], kg[i + 24]);
                t[i] = __fadd_rn(__fadd_rn(x0, x1), __fadd_rn(x2, x3));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) t[i] = __fadd_rn(t[i], t[i + 4]);
            #pragma unroll
            for (int i = 0; i < 2; ++i) t[i] = __fadd_rn(t[i], t[i + 2]);
            // thread 0 of the reference folds warp partials from 0.0f, in order
            dot = __fadd_rn(dot, __fadd_rn(t[0], t[1]));
        }
        acc = __fadd_rn(acc, fmaxf(dot, 0.0f));
    }
    *out = acc * rsqrtf((float)hd);
}

// Block scores as a TILED GEMM — one thread per score, no block reductions.
//
// `qsa_score_rows_b` still spends four block-wide reductions per output, and
// 8d595d5b measured that those are ~4/5 of its cost (tiling 16 b-values per CTA
// removed the other 1/5). This removes them: a CTA owns a QSA_SG_BM x QSA_SG_BN
// tile of scores, stages that tile's `q` rows and `k` blocks in shared once,
// and each thread contracts ONE score serially over `hd`. Total MACs are
// unchanged; what goes away is the reduction machinery around them.
//
//   qsa_score_rows    1.43 TFLOP in 4.860 s =  294 GFLOP/s   (1 CTA per score)
//   qsa_score_rows_b                3.795 s =  377 GFLOP/s   (16 scores per CTA)
//   MoE, same prefill 142.7 TFLOP in 5.089 s = 28.0 TFLOP/s  (a real tiled GEMM)
//
// NUMERICS — this is NOT bit-identical, deliberately. The `d` contraction
// becomes a sequential per-thread sum instead of a block-wide tree, which
// reassociates it. That matters more here than in most kernels: these scores
// feed a top-k, so a changed score can change WHICH blocks a query attends to,
// not merely by how much. Gate with `lc_subset.py` + `kl_drift.py
// --precision-change` AND `lc_check.py` needle recall — a KL check alone cannot
// see a model that has quietly stopped retrieving. Everything else is held
// fixed: `k` is still widened bf16 -> float before the multiply, the `hh` sum is
// still `fmaxf(dot, 0)` in `hh` order, and the scale is still `rsqrtf(hd)`.
//
// SHARED LAYOUT. `ks` is padded to `hd + 1` floats per block. Without the pad,
// threads of a warp vary `j` while sharing `d`, and a stride of hd=128 floats
// puts every `j` in the same bank — a 32-way conflict on the hot load.
#define QSA_SG_BM 8
#define QSA_SG_BN 32
extern "C" __global__ void qsa_score_rows_gemm(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rows,
    const unsigned int n_blocks_max
) {
    const unsigned int r0 = blockIdx.x * QSA_SG_BM;
    const unsigned int b0 = blockIdx.y * QSA_SG_BN;
    const unsigned int tid = threadIdx.x;
    const unsigned int ldk = hd + 1u;

    extern __shared__ float smem[];
    float* qs = smem;                                   // [BM][n_heads][hd]
    float* ks = smem + (size_t)QSA_SG_BM * n_heads * hd; // [BN][hd + 1]

    const unsigned int qn = QSA_SG_BM * n_heads * hd;
    for (unsigned int i = tid; i < qn; i += blockDim.x) {
        const unsigned int rr = i / (n_heads * hd);
        const unsigned int rest = i - rr * n_heads * hd;
        const unsigned int r = r0 + rr;
        qs[i] = (r < rows) ? q[(size_t)r * n_heads * hd + rest] : 0.0f;
    }
    const unsigned int kn = QSA_SG_BN * hd;
    for (unsigned int i = tid; i < kn; i += blockDim.x) {
        const unsigned int bb = i / hd;
        const unsigned int d = i - bb * hd;
        const unsigned int b = b0 + bb;
        ks[bb * ldk + d] =
            (b < n_blocks_max) ? (float)block_keys[(size_t)b * hd + d] : 0.0f;
    }
    __syncthreads();

    // One score per thread: QSA_SG_BM * QSA_SG_BN == blockDim.x.
    const unsigned int ii = tid / QSA_SG_BN;
    const unsigned int jj = tid - ii * QSA_SG_BN;
    const unsigned int r = r0 + ii;
    const unsigned int b = b0 + jj;
    if (r >= rows || b >= n_blocks_max) {
        return;
    }
    float* out = scores + (size_t)r * score_stride + b;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    if (b >= complete) {
        *out = -1e30f;
        return;
    }

    const float* qrow = qs + (size_t)ii * n_heads * hd;
    const float* krow = ks + (size_t)jj * ldk;
    float acc = 0.0f;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float* qh = qrow + (size_t)hh * hd;
        float dot = 0.0f;
        for (unsigned int d = 0; d < hd; ++d) {
            dot = __fmaf_rn(qh[d], krow[d], dot);
        }
        acc += fmaxf(dot, 0.0f);
    }
    *out = acc * rsqrtf((float)hd);
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

// Physical offset of selected key `t` for one row. Factored out so the
// software pipeline in `qsa_prefill_attn_g` can compute the NEXT key's address
// before consuming the current one's data.
__device__ __forceinline__ unsigned long long qsa_key_off(
    unsigned int t, unsigned int topk_ratio, unsigned int ratio, bool ratio_p2,
    unsigned int ratio_sh, unsigned int ratio_mask, unsigned int complete,
    unsigned int block_size, bool bs_p2, unsigned int bs_sh, unsigned int bs_mask,
    const int* __restrict__ my_list, const int* __restrict__ block_table,
    unsigned long long page_stride, unsigned int row_elems,
    unsigned int kvh, unsigned int hd)
{
    unsigned int tok;
    if (t < topk_ratio) {
        const unsigned int li = ratio_p2 ? (t >> ratio_sh) : (t / ratio);
        const unsigned int lo = ratio_p2 ? (t & ratio_mask) : (t % ratio);
        tok = (unsigned int)my_list[li] * ratio + lo;
    } else {
        tok = complete * ratio + (t - topk_ratio);
    }
    const unsigned int pg = bs_p2 ? (tok >> bs_sh) : (tok / block_size);
    const unsigned int inb = bs_p2 ? (tok & bs_mask) : (tok % block_size);
    const unsigned long long off =
           (unsigned long long)(unsigned int)block_table[pg] * page_stride
         + (unsigned long long)inb * row_elems
         + (unsigned long long)kvh * hd;
#ifdef QSA_PA_PROBE_LOCALITY
    // MEASUREMENT PROBE ONLY -- PRODUCES WRONG OUTPUT, NEVER SHIP.
    // Collapses the K/V working set into a 128 KB window so every load hits
    // cache. Every instruction above still executes (block_table and my_list
    // are still read, the index math is unchanged); only WHICH bytes are
    // fetched changes. The delta against the unprobed build is therefore the
    // cost of K/V memory traffic alone, with occupancy held fixed -- which the
    // G=2 vs G=4 comparison could not isolate, because G moves registers too.
    // 0xFF00 preserves the 256-element alignment the uint4 loads require.
    return off & 0xFF00ull;
#else
    return off;
#endif
}

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
    // `ratio` and `block_size` are runtime arguments, so nvcc emits a full
    // integer-division sequence for each of the four divides below -- and they
    // run once per lane per key, ~2048 keys per (row, head-group). Both are
    // powers of two in every shipped configuration (4 and 16), so decompose
    // once here and shift in the loop. Exact integer arithmetic either way; the
    // `_gen` fallback keeps a non-power-of-two configuration correct.
    const bool ratio_p2 = ratio != 0u && (ratio & (ratio - 1u)) == 0u;
    const bool bs_p2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
    const unsigned int ratio_sh = ratio_p2 ? (unsigned int)(__ffs((int)ratio) - 1) : 0u;
    const unsigned int ratio_mask = ratio_p2 ? (ratio - 1u) : 0u;
    const unsigned int bs_sh = bs_p2 ? (unsigned int)(__ffs((int)block_size) - 1) : 0u;
    const unsigned int bs_mask = bs_p2 ? (block_size - 1u) : 0u;
    const unsigned int topk_ratio = topk * ratio;

    // Software pipeline. Bandwidth is not the limiter here (a 4x K/V traffic
    // cut bought only 1.47x), nor SFU, nor principally shuffles -- all measured
    // -- which leaves load LATENCY, and after vectorisation there are exactly
    // two loads per key to hide. Issue key t+WARPS's K/V while key t is being
    // reduced. The online softmax stays strictly sequential in t, so the
    // arithmetic and its order are untouched; only the loads move earlier.
    //
    // The raw uint4s are carried, not the widened floats: 8 registers instead
    // of 16, and the bf16->float converts stay next to their use.
    unsigned int t = warp;
    uint4 kraw_n, vraw_n;
    unsigned long long off_n = 0ull;
    if (t < n_tok) {
        off_n = qsa_key_off(t, topk_ratio, ratio, ratio_p2, ratio_sh, ratio_mask,
                            complete, block_size, bs_p2, bs_sh, bs_mask,
                            my_list, block_table, page_stride, row_elems, kvh, hd);
        if (vec == 8) {
            kraw_n = *reinterpret_cast<const uint4*>(k_cache + off_n + lane * 8);
            vraw_n = *reinterpret_cast<const uint4*>(v_cache + off_n + lane * 8);
        }
    }
    for (; t < n_tok; t += QSA_PA_WARPS) {
        const unsigned long long off = off_n;
        const uint4 kraw = kraw_n;
        const uint4 vraw = vraw_n;
        // Issue the NEXT key's loads before touching this one's data.
        const unsigned int t_n = t + QSA_PA_WARPS;
        if (t_n < n_tok) {
            off_n = qsa_key_off(t_n, topk_ratio, ratio, ratio_p2, ratio_sh, ratio_mask,
                                complete, block_size, bs_p2, bs_sh, bs_mask,
                                my_list, block_table, page_stride, row_elems, kvh, hd);
            if (vec == 8) {
                kraw_n = *reinterpret_cast<const uint4*>(k_cache + off_n + lane * 8);
                vraw_n = *reinterpret_cast<const uint4*>(v_cache + off_n + lane * 8);
            }
        }
        const __nv_bfloat16* krow = k_cache + off;
        const __nv_bfloat16* vrow = v_cache + off;
        float kreg[8], vreg[8];
        // Lane `l` wants elements [l*vec, l*vec+vec) -- contiguous. At vec == 8
        // that is 16 bytes, and `off` is a multiple of `hd`, so the address is
        // 16-byte aligned: one 128-bit load instead of EIGHT 2-byte ones, for
        // each of K and V, on a loop that runs ~2048 times per (row, group).
        // Bit-identical -- same bytes, same `(float)` widening, same order.
        if (vec == 8) {
            const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&kraw);
            const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&vraw);
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                kreg[e] = (float)kp[e];
                vreg[e] = (float)vp[e];
            }
        } else {
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                kreg[e] = (e < vec) ? (float)krow[lane * vec + e] : 0.0f;
                vreg[e] = (e < vec) ? (float)vrow[lane * vec + e] : 0.0f;
            }
        }
        #pragma unroll
        for (unsigned int g = 0; g < QSA_PA_G; ++g) {
            float dot = 0.0f;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) dot += qreg[g][e] * kreg[e];
            }
            // Butterfly instead of down-shift + broadcast. For lane 0 the two
            // are the SAME value at every step -- 0^16 == 16, 0^8 == 8, ... --
            // so the addition order is unchanged, and xor leaves the total in
            // every lane, which retires the separate broadcast. Six shuffles
            // per dot become five. `qsa_prefill_attn_g_matches_single_head_
            // bitwise` compares against the one-head kernel, which still uses
            // the down-shift, so a mistake here fails loudly.
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, o);
            dot *= inv_sqrt_d;

            // All 32 lanes compute the same two `__expf` here, because `dot`
            // is broadcast and `m[g]` is warp-uniform. Computing them once in
            // lane 0 and broadcasting was tried and is SLOWER -- 26.201 s ->
            // 26.846 s at 30k. The redundant SFU work hides under the rest of
            // the loop; two extra shuffles do not. This kernel is not
            // SFU-bound, so leave it.
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

// -- 8 lanes per head: a SHORTER reduction, not a smaller one ---------------
//
// `qsa_prefill_attn_g` gives every lane 8 elements of hd=256 and reduces each
// head across all 32 lanes: five butterfly levels, and four heads reduced
// separately, so TWENTY shuffle instructions per lane per warp-key on a chain
// five deep.
//
// This variant splits the warp into four 8-lane groups. Group `lane>>3` owns
// ONE head; lane `lane&7` owns 32 contiguous elements of it. The reduction is
// then three levels inside an 8-lane group -- and because `__shfl_xor_sync`
// with an offset below 8 never crosses a group, all four heads reduce in the
// SAME three warp instructions. Twenty shuffles become three, and the
// dependency chain goes from five levels to three.
//
// WHY THIS SHAPE, given the measurements in TTFT_GAP.md 22. Free K/V memory is
// worth 2.4% here and cutting 24% of the ALU stream is worth 0%, so this kernel
// is bound by neither traffic nor instruction count. The only two changes that
// have ever moved it -- removing one shuffle (-0.28 s) and making loads hit
// cache (-0.61 s) -- both shortened a latency chain. So the target is the
// chain, and this trades MORE instructions (~175 vs ~150 per lane per
// warp-key: 32 dot FMAs and 32 accumulator updates instead of 8 and 8x4, four
// uint4 loads per operand instead of one) for far less serial latency. Under
// an instruction-count model that is a losing trade; under the latency model
// the evidence actually supports, it is the right one. The measurement decides.
//
// K/V TRAFFIC IS UNCHANGED, which is the part that is easy to get wrong. All
// four groups read the same 512-byte K row, so each 16-byte segment is
// requested by four lanes of the same instruction and the coalescer merges
// them: 512 unique bytes per warp-key, exactly as before. Only the number of
// load INSTRUCTIONS rises.
//
// NOT BIT-IDENTICAL -- and this is a different kind of reassociation from the
// one that sank the GEMM scorer (16a). There, reassociated scores fed a top-k
// over thousands of near-tied blocks and flipped the SELECTION. Here the drift
// lands in the attention output, which feeds the residual stream, so it is
// ordinary numerical drift. It still needs `scripts/ppl.py` and `kl_drift.py`,
// because a later layer's indexer top-k is downstream of it.
#define QSA_L8_LANES 8
#define QSA_L8_EV    32          // hd / QSA_L8_LANES at hd = 256; the cap

extern "C" __global__ void qsa_prefill_attn_l8(
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
    const unsigned int gsub = lane >> 3;         // head within the group
    const unsigned int sub = lane & 7u;          // slice of hd
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int kvh = qh0 / (nq / nkv);
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int ev = hd / QSA_L8_LANES;   // elements per lane
    const unsigned int nch = ev >> 3;            // 16-byte chunks per lane
    const unsigned int vec = hd / 32;            // epilogue only

    extern __shared__ float smem[];
    float* acc_w = smem;                                    // [WARPS][G][hd]
    float* m_w = smem + QSA_PA_WARPS * QSA_PA_G * hd;       // [WARPS][G]
    float* l_w = m_w + QSA_PA_WARPS * QSA_PA_G;             // [WARPS][G]

    float qreg[QSA_L8_EV];
    {
        const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh0 + gsub) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < QSA_L8_EV; ++e) {
            qreg[e] = (e < ev) ? (float)qrow[sub * ev + e] : 0.0f;
        }
    }

    float m = -1e30f;
    float l = 0.0f;
    float acc[QSA_L8_EV];
    #pragma unroll
    for (unsigned int e = 0; e < QSA_L8_EV; ++e) acc[e] = 0.0f;

    const int* my_list = lists + (size_t)r * topk;
    const bool ratio_p2 = ratio != 0u && (ratio & (ratio - 1u)) == 0u;
    const bool bs_p2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
    const unsigned int ratio_sh = ratio_p2 ? (unsigned int)(__ffs((int)ratio) - 1) : 0u;
    const unsigned int ratio_mask = ratio_p2 ? (ratio - 1u) : 0u;
    const unsigned int bs_sh = bs_p2 ? (unsigned int)(__ffs((int)block_size) - 1) : 0u;
    const unsigned int bs_mask = bs_p2 ? (block_size - 1u) : 0u;
    const unsigned int topk_ratio = topk * ratio;

    for (unsigned int t = warp; t < n_tok; t += QSA_PA_WARPS) {
        const unsigned long long off =
            qsa_key_off(t, topk_ratio, ratio, ratio_p2, ratio_sh, ratio_mask,
                        complete, block_size, bs_p2, bs_sh, bs_mask,
                        my_list, block_table, page_stride, row_elems, kvh, hd);
        // `off` is a multiple of hd and `sub * ev` a multiple of 32 elements,
        // so every `+ c * 8` address below is 16-byte aligned.
        const __nv_bfloat16* krow = k_cache + off + sub * ev;
        const __nv_bfloat16* vrow = v_cache + off + sub * ev;

        float dot = 0.0f;
        #pragma unroll
        for (unsigned int c = 0; c < (QSA_L8_EV >> 3); ++c) {
            if (c < nch) {
                const uint4 kraw = *reinterpret_cast<const uint4*>(krow + c * 8);
                const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(&kraw);
                #pragma unroll
                for (unsigned int e = 0; e < 8; ++e) {
                    dot += qreg[c * 8 + e] * (float)kp[e];
                }
            }
        }
        // Three levels, not five -- and offsets below 8 never leave the 8-lane
        // group, so all four heads reduce in these same three instructions.
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, o);
        dot *= inv_sqrt_d;

        const float m_new = fmaxf(m, dot);
        const float scale = __expf(m - m_new);
        const float p = __expf(dot - m_new);
        l = l * scale + p;
        #pragma unroll
        for (unsigned int c = 0; c < (QSA_L8_EV >> 3); ++c) {
            if (c < nch) {
                const uint4 vraw = *reinterpret_cast<const uint4*>(vrow + c * 8);
                const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&vraw);
                #pragma unroll
                for (unsigned int e = 0; e < 8; ++e) {
                    acc[c * 8 + e] = acc[c * 8 + e] * scale + p * (float)vp[e];
                }
            }
        }
        m = m_new;
    }

    {
        float* dst = acc_w + ((size_t)warp * QSA_PA_G + gsub) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < QSA_L8_EV; ++e) {
            if (e < ev) dst[sub * ev + e] = acc[e];
        }
        if (sub == 0) {
            m_w[warp * QSA_PA_G + gsub] = m;
            l_w[warp * QSA_PA_G + gsub] = l;
        }
    }
    __syncthreads();

    // Merge is UNCHANGED from `qsa_prefill_attn_g`: `acc_w` has the identical
    // [WARPS][G][hd] layout, so the same warp-per-head, same `w` order.
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
            const float sc = __expf(m_w[w * QSA_PA_G + g] - m_tot);
            l_tot += l_w[w * QSA_PA_G + g] * sc;
            const float* srcw = acc_w + ((size_t)w * QSA_PA_G + g) * hd;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) out[e] += srcw[lane * vec + e] * sc;
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

// ── Device-side decode block selection ─────────────────────────────────
// The `block_topk` largest of `scores[0..complete)` with torch.topk's
// tie-break (equal scores: the LOWER index wins), emitted in ASCENDING block
// order and expanded to token ids (`b*ratio + r`), followed by the partial
// block's tail `tail_start..visible`. Replaces the per-layer D2H copy + host
// sort + H2D upload (`QsaIndexer::decode_select`); one block of threads,
// `complete <= QSA_SELECT_MAX_BLOCKS` (the host arm covers anything wider).
// Rank of block b = #{j : s_j > s_b or (s_j == s_b and j < b)}; selected iff
// rank < block_topk. A NaN score is staged as -inf, i.e. it ranks BELOW every
// real score (scores are >= 0) and ties among themselves go by index. The
// order is therefore total for ANY input, so exactly block_topk blocks are
// selected whenever complete > block_topk (the only case this runs in). That
// is a memory-safety property, not a quality one: `n_sel` is derived from the
// position alone and `qsa_gather` reads all of it, so an entry this launch
// did not write would be a stale id from another step — or another sequence,
// `sel` being layer-owned scratch. NaN is not reachable today (`qsa_score`
// takes the relu as fmaxf, which returns the non-NaN operand); this keeps the
// kernel safe if that ever changes. Same rule as the host arm (`rank_cmp`).
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
    __shared__ float sc[QSA_SELECT_MAX_BLOCKS];   // scores staged once (16 KB)
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    // Stage the scores in shared memory: the rank loop below reads every
    // score `complete` times, and nsys had the global-memory version at
    // ~71 us per launch (12 per token) for complete <= 512.
    const float neg_inf = __int_as_float(0xff800000u);
    for (int b = tid; b < complete; b += nt) {
        const float v = scores[b];
        sc[b] = isnan(v) ? neg_inf : v;     // NaN ranks below every real score
    }
    __syncthreads();
    for (int b = tid; b < complete; b += nt) {
        const float sb = sc[b];
        int rank = 0;
        for (int j = 0; j < complete; ++j) {
            const float sj = sc[j];
            if (sj > sb || (sj == sb && j < b)) ++rank;
        }
        selected[b] = (rank < block_topk) ? 1 : 0;
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
