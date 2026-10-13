// SPDX-License-Identifier: AGPL-3.0-only
//
// QSA decode block selection by radix select: the body of
// `qsa_select_topk_radix` (qsa_indexer.cu) and of the per-row
// `qsa_select_topk_radix_rows` (qsa_decode_rows.cu). One definition, so the
// single-row and the batched-row selections are the same code.
//
// The `block_topk` largest of `scores[0..complete)` with torch.topk's
// tie-break (equal scores: the LOWER index wins), emitted in ASCENDING block
// order and expanded to token ids (`b*ratio + r`), followed by the partial
// block's tail `tail_start..visible`:
//   key  = order-preserving u32 of rank_key(score) (NaN -> -inf, -0 -> +0, the
//          host `rank_key`), so "larger key" == "larger score";
//   T    = the block_topk-th largest key, found by four 8-bit MSD histogram
//          passes; every key > T is selected, plus the `need` lowest-index
//          keys == T (ties to the lower index, the host `rank_cmp`);
//   out  = selected blocks in ascending index order, expanded by `ratio`, then
//          the incomplete tail — the layout qsa_gather reads.
// One CTA of QSA_RADIX_THREADS threads; each thread owns a contiguous index
// chunk so the two block-wide prefix scans preserve index order. Nothing in
// here bounds `complete`: scores are re-read from global memory each pass (no
// staging), and the per-thread chunk is ceil(complete / QSA_RADIX_THREADS).
//
// `static` for internal linkage per translation unit (as gdn_reduce.cuh).

#ifndef ATLAS_QSA_RADIX_CUH
#define ATLAS_QSA_RADIX_CUH

#define QSA_RADIX_THREADS 1024

static __device__ __forceinline__ unsigned int qsa_rank_key_u32(float s) {
    if (isnan(s)) s = __int_as_float(0xff800000);   // NaN ranks below every real score
    if (s == 0.0f) s = 0.0f;                         // -0 folds into +0
    const unsigned int u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive block scan of one value per thread (blockDim == QSA_RADIX_THREADS).
static __device__ __forceinline__ int qsa_block_exclusive_scan(int v, int* tmp, int* total) {
    const int tid = threadIdx.x;
    tmp[tid] = v;
    __syncthreads();
    for (int off = 1; off < QSA_RADIX_THREADS; off <<= 1) {
        const int add = (tid >= off) ? tmp[tid - off] : 0;
        __syncthreads();
        tmp[tid] += add;
        __syncthreads();
    }
    const int incl = tmp[tid];
    if (total) *total = tmp[QSA_RADIX_THREADS - 1];
    __syncthreads();
    return incl - v;
}

// The whole selection for one query. Every thread of the CTA must call it.
static __device__ __forceinline__ void qsa_radix_select(
    const float* __restrict__ scores,   // [complete]
    int* __restrict__ sel,              // [block_topk*ratio + (visible - tail_start)]
    int complete,
    int block_topk,
    int ratio,
    int tail_start,
    int visible)
{
    __shared__ unsigned int hist[256];
    __shared__ int scan_tmp[QSA_RADIX_THREADS];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_need;
    const int tid = threadIdx.x;

    // ---- 1. threshold key T via four MSD 8-bit histogram passes ----
    if (tid == 0) { s_prefix = 0u; s_mask = 0u; s_need = block_topk; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix, mask = s_mask;
        for (int b = tid; b < complete; b += QSA_RADIX_THREADS) {
            const unsigned int k = qsa_rank_key_u32(scores[b]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need;
            int bucket = 255;
            for (; bucket > 0; --bucket) {            // largest keys first
                const int c = (int)hist[bucket];
                if (c >= need) break;
                need -= c;
            }
            s_need = need;                            // still needed inside `bucket`
            s_prefix = prefix | ((unsigned int)bucket << shift);
            s_mask = mask | (0xFFu << shift);
        }
        __syncthreads();
    }
    const unsigned int T = s_prefix;                  // exact block_topk-th largest key
    const int need_eq = s_need;                       // how many keys == T to take

    // ---- 2. per-thread contiguous chunk: count ties, then selections ----
    const int chunk = (complete + QSA_RADIX_THREADS - 1) / QSA_RADIX_THREADS;
    const int lo = min(tid * chunk, complete), hi = min(lo + chunk, complete);
    int n_eq = 0;
    for (int b = lo; b < hi; ++b) n_eq += (qsa_rank_key_u32(scores[b]) == T);
    const int eq_before = qsa_block_exclusive_scan(n_eq, scan_tmp, nullptr);
    int n_sel = 0;
    {
        int eq_rank = eq_before;
        for (int b = lo; b < hi; ++b) {
            const unsigned int k = qsa_rank_key_u32(scores[b]);
            if (k > T) ++n_sel;
            else if (k == T) { if (eq_rank < need_eq) ++n_sel; ++eq_rank; }
        }
    }
    const int sel_before = qsa_block_exclusive_scan(n_sel, scan_tmp, nullptr);

    // ---- 3. write the selected blocks (ascending) expanded by ratio ----
    {
        int eq_rank = eq_before, pos = sel_before;
        for (int b = lo; b < hi; ++b) {
            const unsigned int k = qsa_rank_key_u32(scores[b]);
            bool take = false;
            if (k > T) take = true;
            else if (k == T) { take = eq_rank < need_eq; ++eq_rank; }
            if (take) {
                const int base = pos * ratio;
                for (int r = 0; r < ratio; ++r) sel[base + r] = b * ratio + r;
                ++pos;
            }
        }
    }
    for (int t = tail_start + tid; t < visible; t += QSA_RADIX_THREADS) {
        sel[block_topk * ratio + (t - tail_start)] = t;
    }
}

#endif // ATLAS_QSA_RADIX_CUH
