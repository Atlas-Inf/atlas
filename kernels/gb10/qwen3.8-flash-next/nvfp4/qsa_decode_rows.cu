// SPDX-License-Identifier: AGPL-3.0-only
//
// QSA decode, several rows of ONE sequence per launch
// (ATLAS_QWEN4EXP_QSA_DECODE_ROWS=1; `layers/qsa_decode_rows.rs`).
//
// A speculative verify past the QSA inert bound attends R consecutive rows of
// one sequence. Each row must see exactly what a serial decode step at its
// position sees, so the serial machinery ran once per row and layer: score,
// top-k, gather, attention, each a launch over one row. At 77K that is ~19K
// scored blocks and a 12-CTA attention per row; the rows queue one behind the
// other on a GPU that one row leaves mostly idle.
//
// These kernels do the same per-row work for all rows in one launch each,
// with the per-row arithmetic unchanged:
//   - `qsa_select_topk_radix_rows`: one CTA per row running the shared
//     `qsa_radix_select` (qsa_radix.cuh) on that row's scores at that row's
//     geometry — the code the single-row `qsa_select_topk_radix` runs.
//   - `qsa_sparse_decode_attn`: the selected-set attention WITHOUT the gather.
//     It is `paged_decode_attn` (common/paged_decode_attn.cu) over the
//     gathered scratch, re-derived: the control flow (warp chunks, BC=4
//     batches cut at `block_size` boundaries of the VIRTUAL, gathered
//     position) and every float operation are that kernel's, in its order;
//     only the address of virtual position v changes, from scratch row v to
//     the paged-cache row of token sel[v] — the row qsa_gather would have
//     copied there. Same values, same operations: bit-identical outputs
//     (`scripts/dev/qwen4exp_qsa_decode_bench.cu` checks it byte for byte).
// Scores and q prep reuse the prefill row kernels in qsa_indexer.cu
// (`qsa_qprep_rows`, `qsa_score_rows_exact`), bit-identical to the decode
// `qsa_qprep` / `qsa_score` (same bench).
//
// Row r sits at position first_pos + r; every row is ACTIVE (more complete
// blocks than block_topk). Built with --fmad=false like the rest of this
// target (KERNEL.toml), so no product is contracted differently here than in
// paged_decode_attn.

#include <cuda_bf16.h>
#include "qsa_radix.cuh"

// Row r's selection geometry, as `qsa_decode_select::select_geometry`.
struct QsaRowGeo {
    int complete, tail_start, visible, n_sel;
};

__device__ __forceinline__ QsaRowGeo qsa_row_geo(unsigned int pos, int ratio, int block_topk) {
    QsaRowGeo g;
    g.visible = (int)pos + 1;
    g.complete = g.visible / ratio;
    g.tail_start = g.complete * ratio;
    g.n_sel = block_topk * ratio + (g.visible - g.tail_start);
    return g;
}

// Grid: (rows,1,1)  Block: (QSA_RADIX_THREADS,1,1).
extern "C" __global__ void __launch_bounds__(QSA_RADIX_THREADS) qsa_select_topk_radix_rows(
    const float* __restrict__ scores,   // [rows, score_stride]
    int* __restrict__ sel,              // [rows, sel_stride]
    unsigned int score_stride,
    unsigned int sel_stride,
    unsigned int first_pos,
    int block_topk,
    int ratio)
{
    const unsigned int r = blockIdx.x;
    const QsaRowGeo g = qsa_row_geo(first_pos + r, ratio, block_topk);
    qsa_radix_select(scores + (size_t)r * score_stride, sel + (size_t)r * sel_stride,
                     g.complete, block_topk, ratio, g.tail_start, g.visible);
}

// ── qsa_score_rows_dec ─────────────────────────────────────────────────
// `qsa_score_rows_exact` (qsa_indexer.cu) for a few decode rows: the same
// per-score arithmetic -- the reference `qsa_block_reduce_sum` tree replayed
// in one thread with __fmul_rn/__fadd_rn, `fmaxf(dot, 0)` folded over heads
// from 0.0f, then `* rsqrtf(hd)` -- so the same bytes. What changes is data
// movement: a thread owns ONE block and walks every row, keeping that
// block's 32-wide key group in registers while each row's query group comes
// from shared memory as a warp-wide broadcast. The tiled exact scorer
// re-read both operands from shared memory per product (load-bound), and
// its 8-row tile left half the threads idle at 4 rows.
//
// Grid: (ceil(n_blocks / QSA_SD_THREADS),1,1)  Block: (QSA_SD_THREADS,1,1).
// Dynamic shared: rows * n_heads * hd floats. n_heads <= QSA_SD_MAXH,
// rows <= QSA_SD_MAXR, hd % 32 == 0.
#define QSA_SD_THREADS 128
#define QSA_SD_MAXR 16
#define QSA_SD_MAXH 4
extern "C" __global__ void __launch_bounds__(QSA_SD_THREADS) qsa_score_rows_dec(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rows,
    const unsigned int n_blocks_max)
{
    extern __shared__ float4 qsd_smem[];
    const float* qs = reinterpret_cast<const float*>(qsd_smem);
    {
        float* w = reinterpret_cast<float*>(qsd_smem);
        const unsigned int qn = rows * n_heads * hd;
        for (unsigned int i = threadIdx.x; i < qn; i += blockDim.x) w[i] = q[i];
    }
    __syncthreads();
    const unsigned int b = blockIdx.x * QSA_SD_THREADS + threadIdx.x;
    if (b >= n_blocks_max) return;
    const __nv_bfloat16* krow = block_keys + (size_t)b * hd;
    const unsigned int groups = hd >> 5;

    float dot[QSA_SD_MAXR][QSA_SD_MAXH];
    #pragma unroll
    for (int r = 0; r < QSA_SD_MAXR; ++r) {
        #pragma unroll
        for (int h = 0; h < QSA_SD_MAXH; ++h) dot[r][h] = 0.0f;   // reference starts at 0.0f
    }
    for (unsigned int g = 0; g < groups; ++g) {
        float kg[32];
        const uint4* k4 = reinterpret_cast<const uint4*>(krow + g * 32u);
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 u = k4[v];
            const unsigned int w[4] = {u.x, u.y, u.z, u.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kg[v * 8 + 2 * j] = __uint_as_float(w[j] << 16);
                kg[v * 8 + 2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
            }
        }
        #pragma unroll
        for (int r = 0; r < QSA_SD_MAXR; ++r) {
            if ((unsigned int)r >= rows) break;
            #pragma unroll
            for (int h = 0; h < QSA_SD_MAXH; ++h) {
                if ((unsigned int)h >= n_heads) break;
                const float4* q4 = reinterpret_cast<const float4*>(qs + ((size_t)r * n_heads + h) * hd + g * 32u);
                float qg[32];
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const float4 x = q4[v];
                    qg[4 * v] = x.x; qg[4 * v + 1] = x.y; qg[4 * v + 2] = x.z; qg[4 * v + 3] = x.w;
                }
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
                dot[r][h] = __fadd_rn(dot[r][h], __fadd_rn(t[0], t[1]));
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < QSA_SD_MAXR; ++r) {
        if ((unsigned int)r >= rows) break;
        const unsigned int complete = (first_pos + r + 1) / ratio;
        float acc = 0.0f;                        // reference starts at 0.0f
        #pragma unroll
        for (int h = 0; h < QSA_SD_MAXH; ++h) {
            if ((unsigned int)h >= n_heads) break;
            acc = __fadd_rn(acc, fmaxf(dot[r][h], 0.0f));
        }
        scores[(size_t)r * score_stride + b] = (b >= complete) ? -1e30f : acc * rsqrtf((float)hd);
    }
}

// ── qsa_sparse_decode_attn ──────────────────────────────────────────────
#define QSD_WARP 32
#define QSD_HDIM 256
#define QSD_VEC_BF16 (QSD_HDIM / QSD_WARP)
#define QSD_VEC_U32  (QSD_HDIM / (QSD_WARP * 2))
#define QSD_WARPS 8
#define QSD_BC 4
// Largest selection a row can hold (budget + ratio - 1 at the published card
// is 2051); the host refuses wider geometry.
#define QSD_MAX_SEL 4096

__device__ __forceinline__ void qsd_unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

// Row of token `t` for `kv_head` in the NHD paged cache.
__device__ __forceinline__ const __nv_bfloat16* qsd_tok(
    const __nv_bfloat16* __restrict__ cache, const int* __restrict__ table,
    unsigned int t, unsigned int block_size, unsigned long long page_stride,
    unsigned long long head_stride_kv, unsigned int kv_head, unsigned int head_dim)
{
    return cache + (unsigned long long)(unsigned int)table[t / block_size] * page_stride
                 + (unsigned long long)(t % block_size) * head_stride_kv
                 + (unsigned long long)kv_head * head_dim;
}

// Grid: (num_q_heads, rows, 1)  Block: (256,1,1). head_dim == QSD_HDIM.
extern "C" __global__ void qsa_sparse_decode_attn(
    const __nv_bfloat16* __restrict__ Q,          // row r at Q + r*q_stride
    const __nv_bfloat16* __restrict__ K_cache,    // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,                // [rows, num_q_heads, head_dim]
    const int* __restrict__ block_table,          // the sequence's real table
    const int* __restrict__ sel,                  // [rows, sel_stride] token ids
    const unsigned int sel_stride,
    const unsigned int first_pos,
    const int ratio,
    const int block_topk,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,                // the gathered view's page size
    const float inv_sqrt_d,
    const unsigned int q_stride)
{
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / QSD_WARP;
    const unsigned int lane_id = tid % QSD_WARP;

    if (q_head >= num_q_heads) return;
    const QsaRowGeo g = qsa_row_geo(first_pos + row, ratio, block_topk);
    const unsigned int seq_len = (unsigned int)g.n_sel;   // the gathered length
    // The host sizes the selection at <= QSD_MAX_SEL (QSA_SPARSE_MAX_SEL).
    if (seq_len > QSD_MAX_SEL) __trap();

    __shared__ int s_sel[QSD_MAX_SEL];
    const int* my_sel = sel + (size_t)row * sel_stride;
    for (unsigned int i = tid; i < seq_len; i += blockDim.x) s_sel[i] = my_sel[i];
    __syncthreads();

    const unsigned int gqa_ratio = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa_ratio;
    const unsigned int vec_offset = lane_id * QSD_VEC_BF16;
    const unsigned long long page_stride = (unsigned long long)block_size * num_kv_heads * head_dim;
    const unsigned long long head_stride_kv = (unsigned long long)num_kv_heads * head_dim;

    const unsigned int* q32 = (const unsigned int*)(Q + (unsigned long long)row * q_stride
                                                       + (unsigned long long)q_head * head_dim + vec_offset);
    float q_reg[QSD_VEC_BF16];
    #pragma unroll
    for (int i = 0; i < QSD_VEC_U32; i++) {
        qsd_unpack2(q32[i], q_reg[2*i], q_reg[2*i+1]);
    }

    // paged_decode_attn's split of [0, seq_len) (window_start == 0).
    const unsigned int attended = seq_len;
    unsigned int chunk_size = (attended + QSD_WARPS - 1) / QSD_WARPS;
    unsigned int my_start = warp_id * chunk_size;
    unsigned int my_end = my_start + chunk_size;
    if (my_end > seq_len) my_end = seq_len;
    if (my_start > seq_len) my_start = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[QSD_VEC_BF16];
    #pragma unroll
    for (int i = 0; i < QSD_VEC_BF16; i++) o_reg[i] = 0.0f;

    // The warp's work is paged_decode_attn's sequence of UNITS over virtual
    // positions: per segment (cut at block_size boundaries), BC-wide batches
    // over its aligned part, then single positions. Same units, same
    // arithmetic per unit, in the same order. The only change is that the
    // next unit's K/V rows are loaded before the current unit is computed
    // (software pipelining): the loads were a dependent chain of
    // LDS(sel) -> LDG(table) -> LDG(row) per unit, and the serial unit chain
    // made this kernel latency-bound.
    unsigned int seg_pos = my_start, seg_cnt = 0, seg_aligned = 0, seg_done = 0;
    auto seg_start = [&]() {
        const unsigned int block_offset = seg_pos % block_size;
        const unsigned int remaining_in_block = block_size - block_offset;
        const unsigned int remaining_total = my_end - seg_pos;
        seg_cnt = remaining_in_block < remaining_total ? remaining_in_block : remaining_total;
        seg_aligned = (seg_cnt / QSD_BC) * QSD_BC;
        seg_done = 0;
    };
    // Advance to the next unit; false when the warp's range is done.
    auto next_unit = [&](unsigned int& upos, unsigned int& ucnt) -> bool {
        if (seg_done == seg_cnt) {
            seg_pos += seg_cnt;
            if (seg_pos >= my_end) return false;
            seg_start();
        }
        upos = seg_pos + seg_done;
        ucnt = (seg_done < seg_aligned) ? QSD_BC : 1u;
        seg_done += ucnt;
        return true;
    };
    auto load_unit = [&](unsigned int upos, unsigned int ucnt,
                         unsigned int (&kk)[QSD_BC][QSD_VEC_U32],
                         unsigned int (&vv)[QSD_BC][QSD_VEC_U32]) {
        #pragma unroll
        for (int b = 0; b < QSD_BC; b++) {
            if ((unsigned int)b < ucnt) {
                const unsigned int t = (unsigned int)s_sel[upos + b];
                const unsigned int* k32 = (const unsigned int*)(qsd_tok(K_cache, block_table, t,
                    block_size, page_stride, head_stride_kv, kv_head, head_dim) + vec_offset);
                const unsigned int* v32 = (const unsigned int*)(qsd_tok(V_cache, block_table, t,
                    block_size, page_stride, head_stride_kv, kv_head, head_dim) + vec_offset);
                #pragma unroll
                for (int i = 0; i < QSD_VEC_U32; i++) { kk[b][i] = k32[i]; vv[b][i] = v32[i]; }
            }
        }
    };

    unsigned int k_packed[QSD_BC][QSD_VEC_U32], v_packed[QSD_BC][QSD_VEC_U32];
    unsigned int k_next[QSD_BC][QSD_VEC_U32], v_next[QSD_BC][QSD_VEC_U32];
    unsigned int cur_pos = 0, cur_cnt = 0;
    bool have = false;
    if (my_start < my_end) {
        seg_start();
        have = next_unit(cur_pos, cur_cnt);
        if (have) load_unit(cur_pos, cur_cnt, k_packed, v_packed);
    }
    while (have) {
        unsigned int nxt_pos = 0, nxt_cnt = 0;
        const bool have_next = next_unit(nxt_pos, nxt_cnt);
        if (have_next) load_unit(nxt_pos, nxt_cnt, k_next, v_next);

        if (cur_cnt == QSD_BC) {
            float scores[QSD_BC];
            #pragma unroll
            for (int b = 0; b < QSD_BC; b++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < QSD_VEC_U32; i++) {
                    float k0, k1;
                    qsd_unpack2(k_packed[b][i], k0, k1);
                    dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
                }
                #pragma unroll
                for (int offset = QSD_WARP / 2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[b] = dot * inv_sqrt_d;
            }

            float m_new = m;
            #pragma unroll
            for (int b = 0; b < QSD_BC; b++)
                m_new = fmaxf(m_new, scores[b]);

            float exp_old = __expf(m - m_new);
            #pragma unroll
            for (int i = 0; i < QSD_VEC_BF16; i++)
                o_reg[i] *= exp_old;
            l *= exp_old;

            float exp_factors[QSD_BC];
            #pragma unroll
            for (int b = 0; b < QSD_BC; b++) {
                exp_factors[b] = __expf(scores[b] - m_new);
                l += exp_factors[b];
            }
            m = m_new;

            #pragma unroll
            for (int b = 0; b < QSD_BC; b++) {
                float ef = exp_factors[b];
                #pragma unroll
                for (int i = 0; i < QSD_VEC_U32; i++) {
                    float v0, v1;
                    qsd_unpack2(v_packed[b][i], v0, v1);
                    o_reg[2*i]   += ef * v0;
                    o_reg[2*i+1] += ef * v1;
                }
            }
        } else {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < QSD_VEC_U32; i++) {
                float k0, k1;
                qsd_unpack2(k_packed[0][i], k0, k1);
                dot += q_reg[2*i] * k0 + q_reg[2*i+1] * k1;
            }
            #pragma unroll
            for (int offset = QSD_WARP / 2; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);

            float score = dot * inv_sqrt_d;
            float m_new = fmaxf(m, score);
            float exp_old = __expf(m - m_new);
            float exp_new = __expf(score - m_new);
            l = l * exp_old + exp_new;

            #pragma unroll
            for (int i = 0; i < QSD_VEC_U32; i++) {
                float v0, v1;
                qsd_unpack2(v_packed[0][i], v0, v1);
                o_reg[2*i]   = o_reg[2*i]   * exp_old + exp_new * v0;
                o_reg[2*i+1] = o_reg[2*i+1] * exp_old + exp_new * v1;
            }
            m = m_new;
        }

        #pragma unroll
        for (int b = 0; b < QSD_BC; b++) {
            #pragma unroll
            for (int i = 0; i < QSD_VEC_U32; i++) { k_packed[b][i] = k_next[b][i]; v_packed[b][i] = v_next[b][i]; }
        }
        cur_cnt = nxt_cnt; have = have_next;
    }

    __shared__ float smem_m[QSD_WARPS];
    __shared__ float smem_l[QSD_WARPS];
    __shared__ float smem_o[QSD_WARPS][QSD_HDIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m;
        smem_l[warp_id] = l;
    }
    #pragma unroll
    for (int i = 0; i < QSD_VEC_BF16; i++) {
        smem_o[warp_id][vec_offset + i] = o_reg[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = QSD_WARPS / 2; stride > 0; stride >>= 1) {
        if (warp_id < (unsigned int)stride) {
            unsigned int other = warp_id + stride;
            float lw = smem_l[other];
            if (lw > 0.0f) {
                float mw = smem_m[other];
                float my_m = smem_m[warp_id];
                float my_l = smem_l[warp_id];
                float m_new = fmaxf(my_m, mw);
                float scale_me = __expf(my_m - m_new);
                float scale_w = __expf(mw - m_new);
                smem_l[warp_id] = my_l * scale_me + lw * scale_w;
                smem_m[warp_id] = m_new;
                #pragma unroll
                for (int i = 0; i < QSD_VEC_BF16; i++) {
                    smem_o[warp_id][vec_offset + i] =
                        smem_o[warp_id][vec_offset + i] * scale_me +
                        smem_o[other][vec_offset + i] * scale_w;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        float final_l = smem_l[0];
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        unsigned int* o32 = (unsigned int*)(O + (unsigned long long)row * num_q_heads * head_dim
                                              + (unsigned long long)q_head * head_dim + vec_offset);
        #pragma unroll
        for (int i = 0; i < QSD_VEC_U32; i++) {
            float v0 = smem_o[0][vec_offset + 2*i]     * inv_l;
            float v1 = smem_o[0][vec_offset + 2*i + 1] * inv_l;
            unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
            unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
            o32[i] = lo | (hi << 16);
        }
    }
}
