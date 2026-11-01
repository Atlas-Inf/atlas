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
// 2. The last block of each row to finish (per-row atomic ticket) merges the
//    row's slice lists into the row's exact top DF2_SEL_MAX_TOP_K, so the
//    rows merge in parallel.
// 3. The last row to finish (atomic ticket; every ticket is reset before its
//    block returns, so a captured graph replays cleanly) walks the chain over
//    each row's top-k prefix, where k is the checkpoint's selector_top_k
//    (runtime parameter, capped at DF2_SEL_MAX_TOP_K). On-device context
//    vector over the checkpoint's full selector_rank (capped at
//    DF2_SEL_MAX_RANK):
//      Score(c) = Unary(c) + sum_{r=0}^{rank-1} (pred[prev, r] * H_proj[row, r]) * succ[c, r]
// 4. One warp per candidate scores across the whole rank; the best-scoring
//    candidate (strict >, so ties resolve to the higher-unary entry)
//    advances the chain. All gamma draft tokens are written directly into
//    device memory.
//
// Step 1 reads its slice twice instead of keeping a sorted list per thread:
// the best candidate of each of the block's K warps, then every candidate that
// precedes or equals the last of those (at most a few dozen on real logits)
// into a shared pool. Each merge (the pool, a row's slice lists, the rank
// split's two lists of a row) ranks every candidate by counting the ones that
// precede it, so a list is exact and sorted without the K serial block-wide
// argmax rounds of a merge by repeated extraction (the walk's block ran 8
// rows × 16 of them, most of the ~0.16 ms the selector took at GLM-5.3's
// geometry; it now takes ~0.03 ms, ennspark03 2026-10-11).
//
// The rank split of the vocabulary head (`rank_split`, ATLAS_GLM_DRAFT_TP)
// runs the same steps in two launches: `dflash2_selector_topk` on each rank
// reduces that rank's share of the vocabulary to every row's exact top
// DF2_SEL_MAX_TOP_K (steps 1-2), the ranks swap those lists instead of their
// logits, and `dflash2_selector_walk` merges the two lists of each row and
// walks (steps 2-4). Every list is exact in the total order, so the picks are
// the fused kernel's.
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
// index rule. A NaN logit never enters a list. Every slice list is exact in
// that order, so the merged prefix is the row's exact top-k.
//
// Grid: (splits, gamma, 1), splits <= DF2_SEL_MAX_SPLITS. Block:
// (DF2_SEL_THREADS, 1, 1) — the walk scores candidate c on warp c, so the
// block needs at least DF2_SEL_MAX_TOP_K warps, and phase 1 needs rank <=
// blockDim.x.
// scratch (u32 words, zero before the first launch; the kernel leaves every
// ticket zero; launches sharing one share gamma): the walk ticket padded to 4
// words; f32 vals and u32 ids, each [gamma, DF2_SEL_MAX_SPLITS,
// DF2_SEL_MAX_TOP_K] (the first `splits` lists of each row used); the row
// tickets [gamma]; then f32 vals and u32 ids of each row's merged list, each
// [gamma, DF2_SEL_MAX_TOP_K].
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

// Shared candidates a block's threshold pass may collect (`df2_sel_slice`).
#define DF2_SEL_POOL 512
// `df2_sel_slice`'s threshold needs a best candidate from K warps.
static_assert(DF2_SEL_THREADS / 32 >= DF2_SEL_MAX_TOP_K, "the threshold pass needs K warps");

// A block's shared scratch for steps 1-2.
struct Df2SelShared {
    float v[DF2_SEL_POOL];
    unsigned int i[DF2_SEL_POOL];
    unsigned int count;
    float tau_v;
    unsigned int tau_i;
    bool last;
};

// The exact top DF2_SEL_MAX_TOP_K of the `n` candidates in `s_v`/`s_i`
// (shared, complete), sorted into `dst_*`: each candidate counts those that
// precede it (equal ones — sentinels — by slot) and lands at that rank, so
// the ranks are a permutation. `parts` adjacent lanes share a candidate's
// count, so a small `n` still spreads over the block. Slots past `n` get the
// sentinel. The caller syncs before reading `dst_*`.
__device__ __forceinline__ void df2_sel_rank_merge(
    const float* s_v, const unsigned int* s_i, unsigned int n,
    float* dst_v, unsigned int* dst_i
) {
    unsigned int parts = 1;
    while (parts < 32 && n * parts * 2 <= blockDim.x) parts *= 2;
    for (unsigned int base = 0; base < n * parts; base += blockDim.x) {
        const unsigned int g = base + threadIdx.x;
        const unsigned int t = g / parts;
        const bool live = t < n;
        float v = 0.0f;
        unsigned int i = 0, rank = 0;
        if (live) {
            v = s_v[t];
            i = s_i[t];
            for (unsigned int s = g % parts; s < n; s += parts) {
                const float sv = s_v[s];
                const unsigned int si = s_i[s];
                rank += (df2_sel_before(sv, si, v, i) || (s < t && sv == v && si == i)) ? 1u : 0u;
            }
        }
        for (unsigned int offset = parts >> 1; offset > 0; offset >>= 1) {
            rank += __shfl_xor_sync(0xffffffffu, rank, offset);
        }
        if (live && g % parts == 0 && rank < DF2_SEL_MAX_TOP_K) {
            dst_v[rank] = v;
            dst_i[rank] = i;
        }
    }
    for (unsigned int k = n + threadIdx.x; k < DF2_SEL_MAX_TOP_K; k += blockDim.x) {
        dst_v[k] = -INFINITY;
        dst_i[k] = 0xFFFFFFFFu;
    }
}

// The exact top-K of `row_logits[lo, hi)` (candidate ids `id_base + column`)
// when the threshold pass overflowed its pool: each thread keeps a sorted
// register list (fixed size: no dynamic indexing, no local-memory spill);
// each warp pops its K best heads by shuffles alone into `s_v`/`s_i`
// ([warps, K]); the block merges those by rank.
__device__ __noinline__ void df2_sel_slice_lists(
    const __nv_bfloat16* __restrict__ row_logits, unsigned int lo, unsigned int hi,
    unsigned int id_base, float* out_v, unsigned int* out_i,
    float* s_v, unsigned int* s_i
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    float vals[DF2_SEL_MAX_TOP_K];
    unsigned int ids[DF2_SEL_MAX_TOP_K];
    #pragma unroll
    for (int k = 0; k < DF2_SEL_MAX_TOP_K; k++) {
        vals[k] = -INFINITY;
        ids[k] = 0xFFFFFFFFu;
    }
    for (unsigned int c = lo + tid; c < hi; c += blockDim.x) {
        const float v = __bfloat162float(row_logits[c]);
        const unsigned int i = id_base + c;
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
    // The warp's K best: K rounds of a butterfly argmax over the lanes' heads
    // (ties between equal sentinels go to the lower lane, so every lane
    // agrees on one owner, which shifts its list up).
    for (unsigned int out = 0; out < DF2_SEL_MAX_TOP_K; out++) {
        float bv = vals[0];
        unsigned int bi = ids[0];
        unsigned int bl = lane;
        #pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, offset);
            const unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, offset);
            const unsigned int ol = __shfl_xor_sync(0xffffffffu, bl, offset);
            if (df2_sel_before(ov, oi, bv, bi) || (ov == bv && oi == bi && ol < bl)) {
                bv = ov;
                bi = oi;
                bl = ol;
            }
        }
        if (lane == 0) {
            s_v[warp * DF2_SEL_MAX_TOP_K + out] = bv;
            s_i[warp * DF2_SEL_MAX_TOP_K + out] = bi;
        }
        if (lane == bl) {
            #pragma unroll
            for (int k = 0; k < DF2_SEL_MAX_TOP_K - 1; k++) {
                vals[k] = vals[k + 1];
                ids[k] = ids[k + 1];
            }
            vals[DF2_SEL_MAX_TOP_K - 1] = -INFINITY;
            ids[DF2_SEL_MAX_TOP_K - 1] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();
    df2_sel_rank_merge(s_v, s_i, (blockDim.x >> 5) * DF2_SEL_MAX_TOP_K, out_v, out_i);
}

// Calls `f(column, value)` for every column of `row[lo, hi)`: 16-byte loads
// over the aligned body, so a thread's loads are in flight together. The
// order is unspecified; every caller is order-free.
template <typename F>
__device__ __forceinline__ void df2_sel_scan(
    const __nv_bfloat16* __restrict__ row, unsigned int lo, unsigned int hi, F f
) {
    const unsigned short* bits = reinterpret_cast<const unsigned short*>(row);
    unsigned int vlo = hi, vhi = hi;
    if ((reinterpret_cast<unsigned long long>(row) & 15) == 0) {
        vlo = min(hi, (lo + 7) & ~7u);
        vhi = max(vlo, hi & ~7u);
    }
    for (unsigned int c = lo + threadIdx.x; c < vlo; c += blockDim.x) {
        f(c, __uint_as_float((unsigned int)bits[c] << 16));
    }
    for (unsigned int c = vhi + threadIdx.x; c < hi; c += blockDim.x) {
        f(c, __uint_as_float((unsigned int)bits[c] << 16));
    }
    const uint4* body = reinterpret_cast<const uint4*>(row);
    for (unsigned int q = vlo / 8 + threadIdx.x; q < vhi / 8; q += blockDim.x) {
        const uint4 w = body[q];
        const unsigned int words[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            f(q * 8 + 2 * k, __uint_as_float(words[k] << 16));
            f(q * 8 + 2 * k + 1, __uint_as_float(words[k] & 0xFFFF0000u));
        }
    }
}

// Phase 1: this block's exact top-K of `row_logits[lo, hi)`, candidate ids
// `id_base + column`, sorted into `out_*`. The block has K warps (512
// threads, K = 16), whose best candidates are K distinct candidates, so the
// last of them (`tau`) is preceded or equalled by at least K candidates and
// every one of the block's top-K precedes or equals it (a warp without a
// candidate makes `tau` the sentinel, which every candidate precedes). A
// second pass collects those into the shared pool — a few dozen on real
// logits — and ranks them. A pool that overflows (a slice of ties at the
// top) takes the register-list path instead; both are exact.
__device__ __forceinline__ void df2_sel_slice(
    const __nv_bfloat16* __restrict__ row_logits, unsigned int lo, unsigned int hi,
    unsigned int id_base, float* out_v, unsigned int* out_i, Df2SelShared& sh
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    // Pass 1: each warp's best candidate.
    float bv = -INFINITY;
    unsigned int bi = 0xFFFFFFFFu;
    df2_sel_scan(row_logits, lo, hi, [&](unsigned int c, float v) {
        const unsigned int i = id_base + c;
        if (df2_sel_before(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    });
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, offset);
        const unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, offset);
        if (df2_sel_before(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sh.v[warp] = bv;
        sh.i[warp] = bi;
    }
    __syncthreads();
    // `tau`: the last of the warps' bests.
    if (tid == 0) {
        float tv = sh.v[0];
        unsigned int ti = sh.i[0];
        for (unsigned int w = 1; w < (blockDim.x >> 5); w++) {
            if (df2_sel_before(tv, ti, sh.v[w], sh.i[w])) {
                tv = sh.v[w];
                ti = sh.i[w];
            }
        }
        sh.tau_v = tv;
        sh.tau_i = ti;
        sh.count = 0;
    }
    __syncthreads();
    const float tv = sh.tau_v;
    const unsigned int ti = sh.tau_i;
    // Pass 2: every candidate that precedes or equals `tau` into the pool.
    df2_sel_scan(row_logits, lo, hi, [&](unsigned int c, float v) {
        const unsigned int i = id_base + c;
        if (df2_sel_before(v, i, tv, ti) || (v == tv && i == ti)) {
            const unsigned int at = atomicAdd(&sh.count, 1u);
            if (at < DF2_SEL_POOL) {
                sh.v[at] = v;
                sh.i[at] = i;
            }
        }
    });
    __syncthreads();
    const unsigned int n = sh.count;
    if (n <= DF2_SEL_POOL) {
        df2_sel_rank_merge(sh.v, sh.i, n, out_v, out_i);
    } else {
        df2_sel_slice_lists(row_logits, lo, hi, id_base, out_v, out_i, sh.v, sh.i);
    }
}

// The last of `count` blocks to reach `ticket` (after publishing its global
// writes) gets true, and resets the ticket for the next launch.
__device__ __forceinline__ bool df2_sel_last(unsigned int* ticket, unsigned int count, bool* s_last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        *s_last = atomicAdd(ticket, 1u) == count - 1;
    }
    __syncthreads();
    if (!*s_last) return false;
    __threadfence();
    if (threadIdx.x == 0) {
        *ticket = 0u;
    }
    return true;
}

// Scratch words past the walk ticket: slice lists, row tickets, merged lists.
struct Df2SelScratch {
    float* cand_v;
    unsigned int* cand_i;
    unsigned int* row_ticket;
    float* merged_v;
    unsigned int* merged_i;
};

// The layout depends on `gamma` alone (the slice lists take room for
// DF2_SEL_MAX_SPLITS), so every launch on a scratch finds the row tickets
// where the last one reset them, whatever its slice count.
__device__ __forceinline__ Df2SelScratch df2_sel_scratch(unsigned int* scratch, unsigned int gamma) {
    const unsigned int lists = gamma * DF2_SEL_MAX_SPLITS * DF2_SEL_MAX_TOP_K;
    unsigned int* cand_i = scratch + 4 + lists;
    unsigned int* row_ticket = cand_i + lists;
    float* merged_v = reinterpret_cast<float*>(row_ticket + gamma);
    return {
        reinterpret_cast<float*>(scratch + 4),
        cand_i,
        row_ticket,
        merged_v,
        reinterpret_cast<unsigned int*>(merged_v + gamma * DF2_SEL_MAX_TOP_K),
    };
}

// Steps 1-2 for this block's (slice, row): the slice's list, then, in the
// row's last block, the row's merged list in `row_v`/`row_i`. False in every
// other block.
__device__ __forceinline__ bool df2_sel_row_topk(
    const __nv_bfloat16* __restrict__ row_logits, unsigned int columns, unsigned int id_base,
    const Df2SelScratch& sc, float* row_v, unsigned int* row_i, Df2SelShared& sh
) {
    const unsigned int splits = gridDim.x;
    const unsigned int row = blockIdx.y;
    // Slices start on 8-column boundaries (`df2_sel_scan`'s 16-byte loads).
    const unsigned int chunk = ((columns + splits - 1) / splits + 7) & ~7u;
    const unsigned int lo = min(columns, blockIdx.x * chunk);
    const unsigned int hi = min(columns, lo + chunk);
    const unsigned long long rows_base = (unsigned long long)row * splits * DF2_SEL_MAX_TOP_K;
    const unsigned long long base = rows_base + blockIdx.x * DF2_SEL_MAX_TOP_K;
    df2_sel_slice(row_logits, lo, hi, id_base, sc.cand_v + base, sc.cand_i + base, sh);
    if (!df2_sel_last(sc.row_ticket + row, splits, &sh.last)) return false;
    const unsigned int n = splits * DF2_SEL_MAX_TOP_K;
    for (unsigned int t = threadIdx.x; t < n; t += blockDim.x) {
        sh.v[t] = __ldcg(sc.cand_v + rows_base + t);
        sh.i[t] = __ldcg(sc.cand_i + rows_base + t);
    }
    __syncthreads();
    df2_sel_rank_merge(sh.v, sh.i, n, row_v, row_i);
    return true;
}

// Steps 3-4 over rows 0..gamma: `load(r)` leaves row r's top-k prefix in
// `s_list_*` and syncs. Row 0 (the anchor) takes its top-1.
// Pull one cache line toward L2 (the walk's rows are scattered DRAM reads
// whose latency the chain otherwise pays row after row).
__device__ __forceinline__ void df2_sel_prefetch(const void* p) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    (void)p;
#else
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#endif
}

// The codebook rows of the staged rows' candidates `ids` (shared; every pick
// and predecessor is one of them, or the anchor), pulled toward L2 for the
// rows the staging below does not cover yet.
__device__ __forceinline__ void df2_sel_prefetch_rows(
    const unsigned int* ids, unsigned int count, unsigned int anchor,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int vocab_size, unsigned int rank
) {
    const unsigned int lines = (rank * 2 + 127) / 128;
    for (unsigned int k = threadIdx.x; k <= count * lines; k += blockDim.x) {
        const unsigned int slot = k / lines;
        const unsigned int id = slot < count ? ids[slot] : anchor;
        if (id >= vocab_size) continue;
        const unsigned long long at = (unsigned long long)id * rank + (k % lines) * 64;
        df2_sel_prefetch(pred_codebook + at);
        df2_sel_prefetch(succ_codebook + at);
    }
}

// Rows of top-k lists the walk stages in shared memory at a time.
#define DF2_SEL_WALK_ROWS 16
// 16-byte words of one staged block of codebook rows: K rows of up to
// DF2_SEL_MAX_RANK BF16.
#define DF2_SEL_BLOCK_WORDS (DF2_SEL_MAX_TOP_K * DF2_SEL_MAX_RANK / 8)

// The walk's shared state.
struct Df2SelWalkShared {
    // The successor rows of the current row's candidates, and the
    // predecessor rows of the previous row's (one of which is the current
    // row's predecessor), slot by slot as the lists hold them.
    uint4 succ[DF2_SEL_BLOCK_WORDS];
    uint4 pred[DF2_SEL_BLOCK_WORDS];
    float rows_v[DF2_SEL_WALK_ROWS * DF2_SEL_MAX_TOP_K];
    unsigned int rows_i[DF2_SEL_WALK_ROWS * DF2_SEL_MAX_TOP_K];
    // The staged rows' projected hidden rows.
    __nv_bfloat16 hidden[DF2_SEL_WALK_ROWS * DF2_SEL_MAX_RANK];
    float score[DF2_SEL_MAX_TOP_K];
    float context[DF2_SEL_MAX_RANK];
    unsigned int prev_token;
    // The predecessor's slot in `pred`, or DF2_SEL_MAX_TOP_K (read the
    // codebook).
    unsigned int prev_slot;
};

// This thread's 16 bytes of the block of `table` rows of the candidates
// `ids` (a row's top-K list): slot `tid / (rank / 8)`. False when the slot's
// id is no row (it is never read).
__device__ __forceinline__ bool df2_sel_stage_load(
    const unsigned int* ids, const __nv_bfloat16* __restrict__ table,
    unsigned int vocab_size, unsigned int rank, uint4& word
) {
    const unsigned int per = rank / 8;
    if (threadIdx.x >= DF2_SEL_MAX_TOP_K * per) return false;
    const unsigned int id = ids[threadIdx.x / per];
    if (id >= vocab_size) return false;
    word = reinterpret_cast<const uint4*>(table + (unsigned long long)id * rank)[threadIdx.x % per];
    return true;
}

// Steps 3-4 over rows 0..gamma, DF2_SEL_WALK_ROWS at a time: `load(r0, rows)`
// leaves rows `r0..r0 + rows` top-k lists in `w.rows_*` ([rows, K]) and
// syncs. Row 0 (the anchor) takes its top-1. The codebook rows a row reads
// are staged in shared memory while the row before it runs (its candidates'
// successor rows, the previous row's candidates' predecessor rows), so a row
// waits on no global load; the values, and so the arithmetic, are the
// codebooks'.
template <typename LoadRows>
__device__ __forceinline__ void df2_sel_walk(
    LoadRows load,
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int* __restrict__ out_tokens,
    unsigned int prev_token,
    unsigned int banned_to,
    unsigned int gamma,
    unsigned int vocab_size,
    unsigned int rank,
    unsigned int top_k,
    unsigned int end0,
    unsigned int end1,
    unsigned int end2,
    unsigned int end3,
    Df2SelWalkShared& w
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp_id = tid >> 5;
    const __nv_bfloat16* s_succ = reinterpret_cast<const __nv_bfloat16*>(w.succ);
    const __nv_bfloat16* s_pred = reinterpret_cast<const __nv_bfloat16*>(w.pred);
    const bool staged = rank % 8 == 0
        && DF2_SEL_MAX_TOP_K * (rank / 8) <= blockDim.x
        && (reinterpret_cast<unsigned long long>(pred_codebook) & 15) == 0
        && (reinterpret_cast<unsigned long long>(succ_codebook) & 15) == 0;
    // The row whose candidates' successor rows `w.succ` holds (gamma: none).
    unsigned int succ_row = gamma;
    if (tid == 0) {
        w.prev_token = prev_token;
        w.prev_slot = DF2_SEL_MAX_TOP_K;
    }
    __syncthreads();

    for (unsigned int r = 0; r < gamma; r++) {
        const unsigned int r0 = r / DF2_SEL_WALK_ROWS * DF2_SEL_WALK_ROWS;
        const unsigned int staged_rows = min(DF2_SEL_WALK_ROWS, gamma - r0);
        if (r == r0) {
            for (unsigned int k = tid; k < staged_rows * rank; k += blockDim.x) {
                w.hidden[k] = projected_hidden[(unsigned long long)r0 * rank + k];
            }
            load(r0, staged_rows);
            df2_sel_prefetch_rows(w.rows_i, staged_rows * DF2_SEL_MAX_TOP_K, prev_token,
                                  pred_codebook, succ_codebook, vocab_size, rank);
        }
        const float* s_list_v = w.rows_v + (r - r0) * DF2_SEL_MAX_TOP_K;
        const unsigned int* s_list_i = w.rows_i + (r - r0) * DF2_SEL_MAX_TOP_K;

        if (r == 0) {
            // Anchor row: top-1 unary argmax (first-index-wins by the order above).
            if (tid == 0) {
                out_tokens[0] = s_list_i[0];
#ifdef DF2_SEL_CONF
                reinterpret_cast<float*>(out_tokens)[gamma] = 0.0f;
#endif
            }
            // Row 1's successor rows, before its turn.
            uint4 word;
            if (staged && gamma > 1 && staged_rows > 1
                && df2_sel_stage_load(s_list_i + DF2_SEL_MAX_TOP_K, succ_codebook, vocab_size, rank, word)) {
                w.succ[tid] = word;
            }
            succ_row = staged && gamma > 1 && staged_rows > 1 ? 1 : gamma;
            __syncthreads();
            continue;
        }

        // The next row's blocks, in flight while this row runs: its
        // candidates' successor rows and this row's predecessor rows.
        const bool next_succ = staged && r + 1 < gamma && r + 1 < r0 + staged_rows;
        const bool next_pred = staged && r + 1 < gamma;
        uint4 succ_word, pred_word;
        const bool have_succ = next_succ
            && df2_sel_stage_load(s_list_i + DF2_SEL_MAX_TOP_K, succ_codebook, vocab_size, rank, succ_word);
        const bool have_pred = next_pred
            && df2_sel_stage_load(s_list_i, pred_codebook, vocab_size, rank, pred_word);

        // Phase 3: context vector over the full declared rank.
        if (tid < rank) {
            unsigned int prev = w.prev_token;
            if (prev >= vocab_size) prev = 0;
            const unsigned int slot = w.prev_slot;
            const float pred_val = __bfloat162float(slot < DF2_SEL_MAX_TOP_K
                ? s_pred[slot * rank + tid]
                : pred_codebook[(unsigned long long)prev * rank + tid]);
            const float h_val = __bfloat162float(w.hidden[(r - r0) * rank + tid]);
            w.context[tid] = pred_val * h_val;
        }
        __syncthreads();

        // Phase 4: score candidates, one warp each, dot product over the rank.
        if (warp_id < top_k) {
            unsigned int cand = s_list_i[warp_id];
            const bool valid = cand < vocab_size && s_list_v[warp_id] > -INFINITY;
            if (!valid) cand = 0;
            const __nv_bfloat16* succ_row_vals = succ_row == r && valid
                ? s_succ + warp_id * rank
                : succ_codebook + (unsigned long long)cand * rank;
            float dot = 0.0f;
            for (unsigned int k = lane; k < rank; k += 32) {
                dot += w.context[k] * __bfloat162float(succ_row_vals[k]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                const bool banned = r <= banned_to
                    && (cand == end0 || cand == end1 || cand == end2 || cand == end3);
#ifdef DF2_SEL_EDGE
                // Edge-weight twin (`glm_dflash2_selector_conf_e*`): the
                // transition term scaled by DF2_SEL_EDGE.
                w.score[warp_id] = valid && !banned ? s_list_v[warp_id] + DF2_SEL_EDGE * dot : -INFINITY;
#else
                w.score[warp_id] = valid && !banned ? s_list_v[warp_id] + dot : -INFINITY;
#endif
            }
        }
        __syncthreads();

        // This row's blocks are read; the next row's replace them.
        if (have_succ) w.succ[tid] = succ_word;
        if (have_pred) w.pred[tid] = pred_word;
        succ_row = next_succ ? r + 1 : gamma;

        // Pick the best-scoring candidate. Strict `>` in list order, so a
        // score tie resolves to the earlier (higher-unary) candidate: the
        // first maximum, where a NaN never wins and a NaN first score keeps
        // slot 0 — one warp's butterfly.
        if (warp_id == 0) {
            const bool live = lane < top_k;
            const float mine = live ? w.score[lane] : -INFINITY;
            float bv = live && !isnan(mine) ? mine : -INFINITY;
            unsigned int bc = live ? lane : 32u;
            #pragma unroll
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, offset);
                const unsigned int oc = __shfl_xor_sync(0xffffffffu, bc, offset);
                if (ov > bv || (ov == bv && oc < bc)) {
                    bv = ov;
                    bc = oc;
                }
            }
            const unsigned int best_slot = isnan(w.score[0]) ? 0u : bc;
            const unsigned int best = s_list_i[best_slot];
#ifdef DF2_SEL_CONF
            // Each candidate's term in parallel, summed in list order.
            const float best_score = w.score[best_slot];
            const float term = live ? expf(mine - best_score) : 0.0f;
            float mass = 0.0f;
            for (unsigned int c = 0; c < top_k; c++) {
                mass += __shfl_sync(0xffffffffu, term, c);
            }
#endif
            if (lane == 0) {
                out_tokens[r] = best;
                w.prev_token = best;
                w.prev_slot = next_pred && best < vocab_size ? best_slot : DF2_SEL_MAX_TOP_K;
#ifdef DF2_SEL_CONF
                reinterpret_cast<float*>(out_tokens)[gamma + r] = -logf(mass);
#endif
            }
        }
        __syncthreads();
    }
}

// The launch contract the host guarantees; guarded anyway — a violated
// contract would otherwise write past the shared arrays.
__device__ __forceinline__ bool df2_sel_shape_ok(unsigned int rank, unsigned int top_k) {
    return top_k <= DF2_SEL_MAX_TOP_K && top_k != 0
        && rank <= DF2_SEL_MAX_RANK && rank != 0 && rank <= blockDim.x
        && gridDim.x <= DF2_SEL_MAX_SPLITS && blockDim.x == DF2_SEL_THREADS;
}

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
    __shared__ Df2SelShared sh;
    __shared__ Df2SelWalkShared w;

    if (!df2_sel_shape_ok(rank, top_k)) return;

    const unsigned int row = blockIdx.y;
    const Df2SelScratch sc = df2_sel_scratch(scratch, gamma);
    if (!df2_sel_row_topk(
            logits + (unsigned long long)row * vocab_size, vocab_size, 0, sc,
            sc.merged_v + row * DF2_SEL_MAX_TOP_K, sc.merged_i + row * DF2_SEL_MAX_TOP_K, sh)) {
        return;
    }
    // The last row's block walks.
    if (!df2_sel_last(scratch, gamma, &sh.last)) return;
    const unsigned int banned_to = ban_depth ? *ban_depth : 0u;
    auto load = [&](unsigned int r0, unsigned int rows) {
        for (unsigned int k = threadIdx.x; k < rows * DF2_SEL_MAX_TOP_K; k += blockDim.x) {
            w.rows_v[k] = __ldcg(sc.merged_v + r0 * DF2_SEL_MAX_TOP_K + k);
            w.rows_i[k] = __ldcg(sc.merged_i + r0 * DF2_SEL_MAX_TOP_K + k);
        }
        __syncthreads();
    };
    df2_sel_walk(load, projected_hidden, pred_codebook, succ_codebook, out_tokens, *anchor,
                 banned_to, gamma, vocab_size, rank, top_k, end0, end1, end2, end3,
                 w);
}

// Steps 1-2 over one rank's share of the vocabulary: `logits` rows of
// `columns` values `row_stride` apart, candidate ids `id_base + column`.
// `out_lists` receives every row's exact top DF2_SEL_MAX_TOP_K: f32 values
// [gamma, K], then u32 ids [gamma, K]. Grid (splits, gamma); scratch as the
// fused kernel's (its walk ticket unused).
extern "C" __global__ void dflash2_selector_topk(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int row_stride,
    unsigned int columns,
    unsigned int id_base,
    unsigned int gamma,
    float* __restrict__ out_lists,
    unsigned int* __restrict__ scratch
) {
    __shared__ Df2SelShared sh;

    if (!df2_sel_shape_ok(1, 1)) return;
    const unsigned int row = blockIdx.y;
    const Df2SelScratch sc = df2_sel_scratch(scratch, gamma);
    unsigned int* out_ids = reinterpret_cast<unsigned int*>(out_lists + gamma * DF2_SEL_MAX_TOP_K);
    df2_sel_row_topk(
        logits + (unsigned long long)row * row_stride, columns, id_base, sc,
        out_lists + row * DF2_SEL_MAX_TOP_K, out_ids + row * DF2_SEL_MAX_TOP_K, sh);
}

// Steps 2-4 over the lists two `dflash2_selector_topk` launches wrote for
// complementary shares of the vocabulary (`lists_a`, `lists_b`): each row's
// two lists merge by rank into its top-k, then the walk. One block.
extern "C" __global__ void dflash2_selector_walk(
    const float* __restrict__ lists_a,
    const float* __restrict__ lists_b,
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
    unsigned int end3
) {
    // A row's two lists, back to back, then its merged top-k.
    __shared__ float s_v[DF2_SEL_WALK_ROWS * 2 * DF2_SEL_MAX_TOP_K];
    __shared__ unsigned int s_i[DF2_SEL_WALK_ROWS * 2 * DF2_SEL_MAX_TOP_K];
    __shared__ Df2SelWalkShared w;

    if (!df2_sel_shape_ok(rank, top_k) || gridDim.x != 1) return;
    const unsigned int banned_to = ban_depth ? *ban_depth : 0u;
    const unsigned int n = gamma * DF2_SEL_MAX_TOP_K;
    // Every staged row's two lists merge at once: a candidate counts the 2K
    // of its row that precede it (equal sentinels by slot).
    constexpr unsigned int pair = 2 * DF2_SEL_MAX_TOP_K;
    auto load = [&](unsigned int r0, unsigned int rows) {
        for (unsigned int g = threadIdx.x; g < rows * pair; g += blockDim.x) {
            const unsigned int t = g % pair;
            const float* lists = t < DF2_SEL_MAX_TOP_K ? lists_a : lists_b;
            const unsigned int at = (r0 + g / pair) * DF2_SEL_MAX_TOP_K + t % DF2_SEL_MAX_TOP_K;
            s_v[g] = lists[at];
            s_i[g] = reinterpret_cast<const unsigned int*>(lists + n)[at];
        }
        __syncthreads();
        for (unsigned int g = threadIdx.x; g < rows * pair; g += blockDim.x) {
            const unsigned int t = g % pair, base = g - t;
            const float v = s_v[g];
            const unsigned int i = s_i[g];
            unsigned int at = 0;
            for (unsigned int s = 0; s < pair; s++) {
                const float sv = s_v[base + s];
                const unsigned int si = s_i[base + s];
                at += (df2_sel_before(sv, si, v, i) || (s < t && sv == v && si == i)) ? 1u : 0u;
            }
            if (at < DF2_SEL_MAX_TOP_K) {
                w.rows_v[g / pair * DF2_SEL_MAX_TOP_K + at] = v;
                w.rows_i[g / pair * DF2_SEL_MAX_TOP_K + at] = i;
            }
        }
        __syncthreads();
    };
    df2_sel_walk(load, projected_hidden, pred_codebook, succ_codebook, out_tokens, *anchor,
                 banned_to, gamma, vocab_size, rank, top_k, end0, end1, end2, end3,
                 w);
}

#ifndef DF2_SEL_CONF
// Insert (v, idx) into a sorted-desc local list of length top_k.
__device__ __forceinline__ void df2_batch_insert_topk(
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
__device__ __forceinline__ void df2_batch_merge_topk(
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

// Upstream per-sequence body for the batched entry: one 1024-thread CTA
// owns every row of one sequence. Unlike the grid-wide single selector,
// this path needs no cross-block ticket or scratch and supports neither
// confidence output nor end-token banning (the Rust caller gates those).
__device__ __forceinline__ void df2_batch_selector_body(
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
            df2_batch_insert_topk(local_v, local_i, v, i, top_k);
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
            df2_batch_merge_topk(local_v, local_i, other_v, other_i, merged_v, merged_i, top_k);
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
                df2_batch_merge_topk(warp0_v, warp0_i, other_v, other_i, merged_v, merged_i, top_k);
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
    df2_batch_selector_body(
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
#endif // !DF2_SEL_CONF
