// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 on-device bilinear candidate selector — full checkpoint geometry.
//
// Eliminates the logits D2H sync and CPU float loop by running the whole
// greedy chain on-device:
// 1. Per-thread → per-warp → per-block top-k reduction of each row's unary
//    logits, where k is the checkpoint's selector_top_k (runtime parameter,
//    capped at DF2_SEL_MAX_TOP_K).
// 2. On-device context vector over the checkpoint's full selector_rank
//    (capped at DF2_SEL_MAX_RANK):
//      Score(c) = Unary(c) + sum_{r=0}^{rank-1} (pred[prev, r] * H_proj[row, r]) * succ[c, r]
// 3. One warp per candidate scores across the whole rank; the best-scoring
//    candidate (strict >, so ties resolve to the higher-unary entry)
//    advances the chain via s_prev_token.
// 4. All gamma draft tokens are written directly into device memory.
//
// Candidate ordering is a TOTAL ORDER: unary value descending, then vocab
// index ascending — the lower index wins ties, matching the engine's
// first-index-wins argmax contract. Empty slots carry the sentinel
// (-INFINITY, 0xFFFFFFFF): a real -inf logit still beats a sentinel via the
// index rule.
//
// Grid: (1, 1, 1)   Block: (1024, 1, 1) — exactly 32 warps.
// 1024 threads is a HARD requirement: phase 3 maps warp slot w to lane w of
// warp 0 (s_warp_*[32]), so fewer threads would drop warp lists and more
// would write out of bounds.

#define DF2_SEL_MAX_TOP_K 16
#define DF2_SEL_MAX_RANK 256

// Insert (v, idx) into a sorted-desc local list of length top_k.
__device__ __forceinline__ void insert_topk(
    float* __restrict__ vals,
    unsigned int* __restrict__ idxs,
    float v,
    unsigned int idx,
    unsigned int top_k
) {
    // Strictly worse than the current k-th entry (value desc, index asc).
    if (v < vals[top_k - 1] || (v == vals[top_k - 1] && idx > idxs[top_k - 1])) return;
    int pos = (int)top_k - 2;
    while (pos >= 0 && (v > vals[pos] || (v == vals[pos] && idx < idxs[pos]))) {
        vals[pos + 1] = vals[pos];
        idxs[pos + 1] = idxs[pos];
        pos--;
    }
    vals[pos + 1] = v;
    idxs[pos + 1] = idx;
}

// Merge two sorted-desc lists of length top_k into out (length top_k).
__device__ __forceinline__ void merge_topk(
    const float* __restrict__ a_v,
    const unsigned int* __restrict__ a_i,
    const float* __restrict__ b_v,
    const unsigned int* __restrict__ b_i,
    float* __restrict__ out_v,
    unsigned int* __restrict__ out_i,
    unsigned int top_k
) {
    int i = 0, j = 0;
    for (unsigned int k = 0; k < top_k; k++) {
        bool take_a;
        if (i < (int)top_k && j < (int)top_k) {
            take_a = (a_v[i] > b_v[j]) || (a_v[i] == b_v[j] && a_i[i] < b_i[j]);
        } else {
            take_a = (i < (int)top_k);
        }
        if (take_a) {
            out_v[k] = a_v[i];
            out_i[k] = a_i[i];
            i++;
        } else {
            out_v[k] = b_v[j];
            out_i[k] = b_i[j];
            j++;
        }
    }
}

extern "C" __global__ void dflash2_candidate_selector(
    const __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int* __restrict__ out_tokens,
    unsigned int last_token,
    unsigned int gamma,
    unsigned int vocab_size,
    unsigned int rank,
    unsigned int top_k
) {
    __shared__ float s_warp_v[32][DF2_SEL_MAX_TOP_K];
    __shared__ unsigned int s_warp_i[32][DF2_SEL_MAX_TOP_K];
    __shared__ float s_final_v[DF2_SEL_MAX_TOP_K];
    __shared__ unsigned int s_final_i[DF2_SEL_MAX_TOP_K];
    __shared__ float s_score[DF2_SEL_MAX_TOP_K];
    __shared__ float s_context[DF2_SEL_MAX_RANK];
    __shared__ unsigned int s_prev_token;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp_id = tid / 32;

    // Host guarantees 1 <= top_k <= DF2_SEL_MAX_TOP_K and
    // 1 <= rank <= DF2_SEL_MAX_RANK; guard anyway — a violated contract
    // would otherwise write past the shared arrays.
    if (top_k > DF2_SEL_MAX_TOP_K || top_k == 0) return;
    if (rank > DF2_SEL_MAX_RANK || rank == 0) return;

    if (tid == 0) {
        s_prev_token = last_token;
    }
    __syncthreads();

    // Iterate through all gamma rows.
    // Row 0: anchor token (unary argmax).
    // Rows 1..gamma-1: mask draft tokens conditioned on predecessor chain.
    for (unsigned int row = 0; row < gamma; row++) {
        const __nv_bfloat16* row_logits = logits + (unsigned long long)row * (unsigned long long)vocab_size;

        // Phase 1: per-thread local top-k. All DF2_SEL_MAX_TOP_K slots are
        // carried through the reductions; only the first top_k are meaningful.
        float local_v[DF2_SEL_MAX_TOP_K];
        unsigned int local_i[DF2_SEL_MAX_TOP_K];
        #pragma unroll
        for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
            local_v[k] = -INFINITY;
            local_i[k] = 0xFFFFFFFFu;
        }

        for (unsigned int i = tid; i < vocab_size; i += blockDim.x) {
            float v = __bfloat162float(row_logits[i]);
            insert_topk(local_v, local_i, v, i, top_k);
        }

        // Phase 2: warp-level top-k reduction (5 steps, all slots shuffled).
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_v[DF2_SEL_MAX_TOP_K];
            unsigned int other_i[DF2_SEL_MAX_TOP_K];
            #pragma unroll
            for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                other_v[k] = __shfl_down_sync(0xffffffff, local_v[k], offset);
                other_i[k] = __shfl_down_sync(0xffffffff, local_i[k], offset);
            }
            float merged_v[DF2_SEL_MAX_TOP_K];
            unsigned int merged_i[DF2_SEL_MAX_TOP_K];
            merge_topk(local_v, local_i, other_v, other_i, merged_v, merged_i, top_k);
            #pragma unroll
            for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                local_v[k] = merged_v[k];
                local_i[k] = merged_i[k];
            }
        }

        // Leader of each warp writes its list to shared memory.
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                s_warp_v[warp_id][k] = local_v[k];
                s_warp_i[warp_id][k] = local_i[k];
            }
        }
        __syncthreads();

        // Phase 3: warp 0 reduces the 32 warp lists — lane w holds warp w's
        // list (this is why the block must be exactly 1024 threads / 32 warps).
        if (warp_id == 0) {
            float warp0_v[DF2_SEL_MAX_TOP_K];
            unsigned int warp0_i[DF2_SEL_MAX_TOP_K];
            #pragma unroll
            for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                warp0_v[k] = s_warp_v[lane][k];
                warp0_i[k] = s_warp_i[lane][k];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_v[DF2_SEL_MAX_TOP_K];
                unsigned int other_i[DF2_SEL_MAX_TOP_K];
                #pragma unroll
                for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                    other_v[k] = __shfl_down_sync(0xffffffff, warp0_v[k], offset);
                    other_i[k] = __shfl_down_sync(0xffffffff, warp0_i[k], offset);
                }
                float merged_v[DF2_SEL_MAX_TOP_K];
                unsigned int merged_i[DF2_SEL_MAX_TOP_K];
                merge_topk(warp0_v, warp0_i, other_v, other_i, merged_v, merged_i, top_k);
                #pragma unroll
                for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                    warp0_v[k] = merged_v[k];
                    warp0_i[k] = merged_i[k];
                }
            }

            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
                    s_final_v[k] = warp0_v[k];
                    s_final_i[k] = warp0_i[k];
                }
            }
        }
        __syncthreads();

        if (row == 0) {
            // Anchor row: top-1 unary argmax (first-index-wins by the order above).
            if (tid == 0) {
                out_tokens[0] = s_final_i[0];
            }
            __syncthreads();
            continue;
        }

        // Mask row > 0: context vector over the full declared rank.
        if (tid < rank) {
            unsigned int prev = s_prev_token;
            if (prev >= vocab_size) prev = 0;
            float pred_val = __bfloat162float(pred_codebook[(unsigned long long)prev * (unsigned long long)rank + tid]);
            float h_val = __bfloat162float(projected_hidden[(unsigned long long)row * (unsigned long long)rank + tid]);
            s_context[tid] = pred_val * h_val;
        }
        __syncthreads();

        // Score candidates: one warp each, dot product over the full rank.
        if (warp_id < top_k) {
            unsigned int cand = s_final_i[warp_id];
            bool valid = cand < vocab_size && s_final_v[warp_id] > -INFINITY;
            if (!valid) cand = 0;
            float dot = 0.0f;
            for (unsigned int r = lane; r < rank; r += 32) {
                dot += s_context[r] * __bfloat162float(succ_codebook[(unsigned long long)cand * (unsigned long long)rank + r]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                s_score[warp_id] = valid ? s_final_v[warp_id] + dot : -INFINITY;
            }
        }
        __syncthreads();

        // Pick the best-scoring candidate. Strict `>` in list order, so a
        // score tie resolves to the earlier (higher-unary) candidate.
        if (tid == 0) {
            unsigned int best = s_final_i[0];
            float best_score = s_score[0];
            for (unsigned int c = 1; c < top_k; c++) {
                if (s_score[c] > best_score) {
                    best_score = s_score[c];
                    best = s_final_i[c];
                }
            }
            out_tokens[row] = best;
            s_prev_token = best;
        }
        __syncthreads();
    }
}
