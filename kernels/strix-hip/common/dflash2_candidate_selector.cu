// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 on-device bilinear candidate selector — full checkpoint geometry.
//
// Eliminates the logits D2H sync and CPU float loop by running the whole
// greedy chain on-device, in one launch:
// 1. Every block takes one (vocab slice, row) and reduces it to its exact
//    top DF2_SEL_MAX_TOP_K candidates in the total order below, written to
//    `scratch`. The whole grid streams the gamma × vocab logits at once
//    instead of one block walking every row in turn.
// 2. The last block to finish (atomic ticket, reset before it returns so a
//    captured graph replays cleanly) merges each row's slice lists into the
//    row's top-k, where k is the checkpoint's selector_top_k (runtime
//    parameter, capped at DF2_SEL_MAX_TOP_K).
// 3. On-device context vector over the checkpoint's full selector_rank
//    (capped at DF2_SEL_MAX_RANK):
//      Score(c) = Unary(c) + sum_{r=0}^{rank-1} (pred[prev, r] * H_proj[row, r]) * succ[c, r]
// 4. One warp per candidate scores across the whole rank; the best-scoring
//    candidate (strict >, so ties resolve to the higher-unary entry)
//    advances the chain. All gamma draft tokens are written directly into
//    device memory.
//
// The anchor (row 1's predecessor) and the banned depth are read from device
// memory, not kernel arguments, so a CUDA graph that captured this launch
// replays with the current step's values (the host writes both before the
// captured tail). Rows 1..=*ban_depth never pick end0..end3: below a
// request's min_tokens floor the verifier may not end the turn, so an end
// token there would only truncate an otherwise acceptable draft chain.
// Unused end slots carry 0xFFFFFFFF; a null ban_depth bans nothing.
//
// Candidate ordering is a TOTAL ORDER: unary value descending, then vocab
// index ascending — the lower index wins ties, matching the engine's
// first-index-wins argmax contract. Empty slots carry the sentinel
// (-INFINITY, 0xFFFFFFFF): a real -inf logit still beats a sentinel via the
// index rule. Every slice list is exact in that order, so the merged prefix
// is the row's exact top-k.
//
// Grid: (splits, gamma, 1), splits <= DF2_SEL_MAX_SPLITS. Block:
// (DF2_SEL_THREADS, 1, 1) — the walk scores candidate c on warp c, so the
// block needs at least DF2_SEL_MAX_TOP_K warps, and phase 1 needs rank <=
// blockDim.x.
// scratch: u32 ticket (zero before the first launch; the kernel leaves it
// zero), then f32 vals and u32 ids, each [gamma, splits, DF2_SEL_MAX_TOP_K].
//
// DF2_SEL_CONF (defined by a twin that includes this file under another
// entry name): out_tokens is [2 * gamma] words and word gamma + r receives,
// as an f32, the confidence of row r's pick: the log of its softmax
// probability over the row's scored candidates (0 for the anchor row; NaN
// when every candidate was banned). The picks themselves are unchanged.
// The signal is knapcio's GLM_DRAFT_TRUNC confidence (log max softmax of the
// selector scores; github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4
// overlay/glm_draft_trunc.py; idea only, see docs/glm-prior-art.md).

#define DF2_SEL_MAX_TOP_K 16
#define DF2_SEL_MAX_RANK 256
#define DF2_SEL_MAX_SPLITS 16
#define DF2_SEL_THREADS 512

// (va, ia) precedes (vb, ib) in the total order.
__device__ __forceinline__ bool df2_sel_before(float va, unsigned int ia, float vb, unsigned int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Block-wide argmax of (v, i) in the total order; every thread gets the
// winner and the winning thread's index. `s_*` hold one entry per warp.
__device__ __forceinline__ void df2_sel_block_best(
    float& v, unsigned int& i, unsigned int& owner,
    float* s_v, unsigned int* s_i, unsigned int* s_owner
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, offset);
        const unsigned int oi = __shfl_down_sync(0xffffffffu, i, offset);
        const unsigned int oo = __shfl_down_sync(0xffffffffu, owner, offset);
        if (df2_sel_before(ov, oi, v, i)) {
            v = ov;
            i = oi;
            owner = oo;
        }
    }
    if (lane == 0) {
        s_v[warp] = v;
        s_i[warp] = i;
        s_owner[warp] = owner;
    }
    __syncthreads();
    if (warp == 0) {
        const bool live = lane < (blockDim.x >> 5);
        v = live ? s_v[lane] : -INFINITY;
        i = live ? s_i[lane] : 0xFFFFFFFFu;
        owner = live ? s_owner[lane] : 0xFFFFFFFFu;
        #pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, offset);
            const unsigned int oi = __shfl_down_sync(0xffffffffu, i, offset);
            const unsigned int oo = __shfl_down_sync(0xffffffffu, owner, offset);
            if (df2_sel_before(ov, oi, v, i)) {
                v = ov;
                i = oi;
                owner = oo;
            }
        }
        if (lane == 0) {
            s_v[0] = v;
            s_i[0] = i;
            s_owner[0] = owner;
        }
    }
    __syncthreads();
    v = s_v[0];
    i = s_i[0];
    owner = s_owner[0];
    __syncthreads();
}

// Streaming single-sequence entry, matching the sixteen-argument launcher.
extern "C" __global__ void dflash2_candidate_selector(
    const __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int* __restrict__ out_tokens,
    const unsigned int* __restrict__ anchor,
    const unsigned int* __restrict__ ban_depth,
    unsigned int gamma,
    unsigned int vocab_size,
    unsigned int rank,
    unsigned int top_k,
    unsigned int end0,
    unsigned int end1,
    unsigned int end2,
    unsigned int end3,
    unsigned int* __restrict__ scratch
) {
    __shared__ float s_v[DF2_SEL_THREADS / 32];
    __shared__ unsigned int s_i[DF2_SEL_THREADS / 32];
    __shared__ unsigned int s_owner[DF2_SEL_THREADS / 32];
    __shared__ float s_list_v[DF2_SEL_MAX_TOP_K];
    __shared__ unsigned int s_list_i[DF2_SEL_MAX_TOP_K];
    __shared__ float s_score[DF2_SEL_MAX_TOP_K];
    __shared__ float s_context[DF2_SEL_MAX_RANK];
    __shared__ unsigned int s_prev_token;
    __shared__ bool s_last;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const unsigned int splits = gridDim.x;
    const unsigned int split = blockIdx.x;
    const unsigned int row = blockIdx.y;

    // Host guarantees these; guard anyway — a violated contract would
    // otherwise write past the shared arrays.
    if (top_k > DF2_SEL_MAX_TOP_K || top_k == 0) return;
    if (rank > DF2_SEL_MAX_RANK || rank == 0 || rank > blockDim.x) return;
    if (splits > DF2_SEL_MAX_SPLITS || blockDim.x != DF2_SEL_THREADS) return;

    unsigned int* ticket = scratch;
    float* cand_v = reinterpret_cast<float*>(scratch + 4);
    unsigned int* cand_i = scratch + 4 + gamma * splits * DF2_SEL_MAX_TOP_K;

    // Phase 1: this slice's exact top-K. Each thread keeps a sorted register
    // list (fixed size: no dynamic indexing, no local-memory spill), then K
    // block-wide argmax rounds pop the heads.
    {
        const unsigned int chunk = (vocab_size + splits - 1) / splits;
        const unsigned int lo = split * chunk;
        const unsigned int hi = min(vocab_size, lo + chunk);
        const __nv_bfloat16* row_logits = logits + (unsigned long long)row * vocab_size;
        float vals[DF2_SEL_MAX_TOP_K];
        unsigned int ids[DF2_SEL_MAX_TOP_K];
        #pragma unroll
        for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
            vals[k] = -INFINITY;
            ids[k] = 0xFFFFFFFFu;
        }
        for (unsigned int i = lo + tid; i < hi; i += blockDim.x) {
            const float v = __bfloat162float(row_logits[i]);
            if (!df2_sel_before(v, i, vals[DF2_SEL_MAX_TOP_K - 1], ids[DF2_SEL_MAX_TOP_K - 1])) continue;
            vals[DF2_SEL_MAX_TOP_K - 1] = v;
            ids[DF2_SEL_MAX_TOP_K - 1] = i;
            #pragma unroll
            for (int k = DF2_SEL_MAX_TOP_K - 1; k > 0; k--) {
                if (df2_sel_before(vals[k], ids[k], vals[k - 1], ids[k - 1])) {
                    const float tv = vals[k];
                    vals[k] = vals[k - 1];
                    vals[k - 1] = tv;
                    const unsigned int ti = ids[k];
                    ids[k] = ids[k - 1];
                    ids[k - 1] = ti;
                }
            }
        }
        const unsigned long long base = ((unsigned long long)row * splits + split) * DF2_SEL_MAX_TOP_K;
        for (unsigned int out = 0; out < DF2_SEL_MAX_TOP_K; out++) {
            // The head is always vals[0]: the winner shifts its list up.
            float v = vals[0];
            unsigned int i = ids[0];
            unsigned int owner = tid;
            df2_sel_block_best(v, i, owner, s_v, s_i, s_owner);
            if (tid == 0) {
                cand_v[base + out] = v;
                cand_i[base + out] = i;
            }
            if (tid == owner) {
                #pragma unroll
                for (int k = 0; k < DF2_SEL_MAX_TOP_K - 1; k++) {
                    vals[k] = vals[k + 1];
                    ids[k] = ids[k + 1];
                }
                vals[DF2_SEL_MAX_TOP_K - 1] = -INFINITY;
                ids[DF2_SEL_MAX_TOP_K - 1] = 0xFFFFFFFFu;
            }
        }
    }

    // Publish, then take a ticket: the last block runs the walk.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int total = gridDim.x * gridDim.y;
        s_last = atomicAdd(ticket, 1u) == total - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid == 0) {
        *ticket = 0u; // ready for the next launch / graph replay
        s_prev_token = *anchor;
    }
    const unsigned int banned_to = ban_depth ? *ban_depth : 0u;
    __syncthreads();

    for (unsigned int r = 0; r < gamma; r++) {
        // Phase 2: merge the row's slice lists (splits × K, each sorted) into
        // its top-k by k block-wide argmax rounds over the list heads.
        {
            const unsigned long long base = (unsigned long long)r * splits * DF2_SEL_MAX_TOP_K;
            const unsigned int n = splits * DF2_SEL_MAX_TOP_K;
            float v = -INFINITY;
            unsigned int i = 0xFFFFFFFFu;
            if (tid < n) {
                v = __ldcg(cand_v + base + tid);
                i = __ldcg(cand_i + base + tid);
            }
            for (unsigned int out = 0; out < top_k; out++) {
                float bv = v;
                unsigned int bi = i;
                unsigned int owner = tid;
                df2_sel_block_best(bv, bi, owner, s_v, s_i, s_owner);
                if (tid == 0) {
                    s_list_v[out] = bv;
                    s_list_i[out] = bi;
                }
                if (tid == owner) {
                    v = -INFINITY;
                    i = 0xFFFFFFFFu;
                }
            }
        }
        __syncthreads();

        if (r == 0) {
            // Anchor row: top-1 unary argmax (first-index-wins by the order above).
            if (tid == 0) {
                out_tokens[0] = s_list_i[0];
#ifdef DF2_SEL_CONF
                reinterpret_cast<float*>(out_tokens)[gamma] = 0.0f;
#endif
            }
            __syncthreads();
            continue;
        }

        // Phase 3: context vector over the full declared rank.
        if (tid < rank) {
            unsigned int prev = s_prev_token;
            if (prev >= vocab_size) prev = 0;
            const float pred_val = __bfloat162float(pred_codebook[(unsigned long long)prev * rank + tid]);
            const float h_val = __bfloat162float(projected_hidden[(unsigned long long)r * rank + tid]);
            s_context[tid] = pred_val * h_val;
        }
        __syncthreads();

        // Phase 4: score candidates, one warp each, dot product over the rank.
        if (warp_id < top_k) {
            unsigned int cand = s_list_i[warp_id];
            const bool valid = cand < vocab_size && s_list_v[warp_id] > -INFINITY;
            if (!valid) cand = 0;
            float dot = 0.0f;
            for (unsigned int k = lane; k < rank; k += 32) {
                dot += s_context[k] * __bfloat162float(succ_codebook[(unsigned long long)cand * rank + k]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                const bool banned = r <= banned_to
                    && (cand == end0 || cand == end1 || cand == end2 || cand == end3);
                s_score[warp_id] = valid && !banned ? s_list_v[warp_id] + dot : -INFINITY;
            }
        }
        __syncthreads();

        // Pick the best-scoring candidate. Strict `>` in list order, so a
        // score tie resolves to the earlier (higher-unary) candidate.
        if (tid == 0) {
            unsigned int best = s_list_i[0];
            float best_score = s_score[0];
            for (unsigned int c = 1; c < top_k; c++) {
                if (s_score[c] > best_score) {
                    best_score = s_score[c];
                    best = s_list_i[c];
                }
            }
            out_tokens[r] = best;
            s_prev_token = best;
#ifdef DF2_SEL_CONF
            float mass = 0.0f;
            for (unsigned int c = 0; c < top_k; c++) {
                mass += expf(s_score[c] - best_score);
            }
            reinterpret_cast<float*>(out_tokens)[gamma + r] = -logf(mass);
#endif
        }
        __syncthreads();
    }
}

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

// Preserve upstream's 1024-thread, one-CTA-per-sequence batched selector.
// Its ten-argument launcher does not use the streaming selector scratch.
__device__ __forceinline__ void df2_selector_batched_body(
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


// Batched variant (job 596): one CTA per sequence. The serial-per-seq loop
// launched this kernel grid (1,1,1) n times on one stream — 4.70 ms each,
// ~75 ms/step at bs16 on ONE SM at a time. Per-sequence pointers are plain
// base + seq*stride offsets of the contiguous [seq][gamma][width] staging
// the batched tail already produces; `last_tokens` is the device array
// (batch_markov_prev) the prologue already uploads.
// Grid: (n_seqs, 1, 1)   Block: (1024, 1, 1) — same hard requirement.
extern "C" __global__ void dflash2_candidate_selector_batched(
    const __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int* __restrict__ out_tokens,
    const unsigned int* __restrict__ last_tokens,
    unsigned int gamma,
    unsigned int vocab_size,
    unsigned int rank,
    unsigned int top_k
) {
    unsigned int s = blockIdx.x;
    df2_selector_batched_body(
        logits + (unsigned long long)s * (unsigned long long)gamma * (unsigned long long)vocab_size,
        projected_hidden + (unsigned long long)s * (unsigned long long)gamma * (unsigned long long)rank,
        pred_codebook,
        succ_codebook,
        out_tokens + (unsigned long long)s * (unsigned long long)gamma,
        last_tokens[s],
        gamma,
        vocab_size,
        rank,
        top_k
    );
}
