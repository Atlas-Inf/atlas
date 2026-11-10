// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared-scan paged decode attention for a BF16 KV cache
// (ATLAS_ATTN_SHARED_SCAN=1, `crates/spark-model/src/layers/ops/shared_scan.rs`),
// qwen3.6-27b tree (so the qwen3.6-27b and qwen3.8-27b targets).
//
// WHY. `paged_decode_attn` runs one CTA per (query head, row) and each one
// walks the whole K and V of its KV head in scalar FP32. With 24 query heads
// over 4 KV heads and the 8 rows of a speculative verify, every cached key
// and value is fetched 48 times a layer; at 200K tokens that scan is three
// quarters of the decode step. Here a CTA owns one KV head over one range of
// positions, reads those keys and values ONCE, and scores every query that
// attends them (the KV head's query heads, for every row that shares the
// blocks) on tensor cores.
//
// EXACT VERIFY. A row's output must not depend on the rows launched beside it:
// `ATLAS_EXACT_VERIFY` compares a verify row with the serial decode step at
// the same position, bit for bit. Everything a row's arithmetic is keyed on
// is its own:
//
//   * Positions are cut into SUPERBLOCKS of `ss_sb_positions(seq_len)`
//     positions, aligned at absolute position 0. The size is a function of
//     the row's own length; rows of other size classes are never grouped
//     with it.
//   * Inside a superblock the walk is in STEPS of 16 positions, aligned at
//     absolute position 0, in order. A step's scores are one `m16n8k16`
//     chain over the 256 dims per (query, position); each output element of
//     an MMA is its own dot product, so a query's score does not depend on
//     the row slot it sits in or on the other queries of the fragment.
//   * The online softmax (running max, sum, value accumulator) is per query,
//     folded step by step; the superblock partials are folded in superblock
//     order by `attn_shared_scan_reduce`.
//   * Positions at or past a row's length are masked by SELECT (score -inf,
//     weight exactly 0), never by arithmetic on what the slot holds. Slots at
//     or past the group's longest row are staged as zeros, so a serial step
//     never multiplies a stale or poisoned slot; the shorter rows of a verify
//     add exact zeros for the slots the longer rows wrote. The partial is
//     written through `+ 0.0f` so an all-zero accumulator has one sign.
//
// GROUPS are found on the device, per superblock, so no caller describes
// them. A CTA owns a WINDOW of SS_ROWS_MAX consecutive rows and walks it:
// adjacent rows join a group while their lengths do not decrease, they share
// the size class, and their block tables agree over the superblock's blocks
// up to the shorter row. The last row of a group has the longest length and
// its table addresses the KV for all of them. The CTA scans once per group,
// one group after another. Grouping is a traffic decision only: by the
// points above no row's bits depend on it.
//
// VALUE WEIGHTS. The V product takes BF16 operands, so a weight goes in as
// two BF16 terms, hi = bf16(w) and lo = bf16(w - hi) (16 mantissa bits), and
// the sum l accumulates exactly those two terms.
//
// Layout (NHD, as `paged_decode_attn`): K/V [num_blocks, block_size,
// num_kv_heads, 256] BF16; Q row `r`, head `h` at Q + r*q_stride + h*256.
// Partials: WS[((row*num_q_heads + q_head)*SS_MAX_SB + sb)*SS_REC] =
// {m, l, o[256]} FP32.
//
// Host-checked (`shared_scan.rs`): head_dim 256, block_size a power of two in
// 16..=256, num_q_heads a multiple of num_kv_heads with ratio <= 48.
//
// Neither kernel is on a PDL list (`cuda_backend/gpu_impl/pdl.rs`), so neither
// waits on PDL entry; a PDL-launched successor still waits for the reduce.
//
// attn_shared_scan         Grid: (num_kv_heads*SS_MAX_SB, ceil(num_seqs/SS_ROWS_MAX))  Block: 96
// attn_shared_scan_reduce  Grid: (num_q_heads, num_seqs)             Block: 256

#include <cuda_bf16.h>

#define SS_HD 256u
#define SS_STEP 16u    // positions a step: the V product's K extent
#define SS_MAX_SB 32u  // superblocks a row
#define SS_MIN_SB 64u  // smallest superblock, in positions
#define SS_WARPS 3u
#define SS_THREADS (SS_WARPS * 32u)
#define SS_QMAX (SS_WARPS * 16u) // queries a CTA
#define SS_ROWS_MAX 8u // rows a window, and the most a group holds
#define SS_REC (SS_HD + 2u)
#define SS_ROW_WORDS 132u // a staged Q or K row: 128 words + 4 (bank skew)
#define SS_VT_WORDS 12u   // a staged V^T dim row: 8 words + 4 (bank skew)
#define SS_STAGE_ROWS ((SS_STEP + SS_WARPS - 1u) / SS_WARPS) // rows a warp stages a step

// Superblock size for a row of length L: the next power of two at or above
// L, over SS_MAX_SB, at least SS_MIN_SB. ceil(L / size) <= SS_MAX_SB.
__device__ __forceinline__ unsigned int ss_sb_positions(unsigned int L) {
    const unsigned int p = (L <= 1u) ? 1u : (L > 0x80000000u ? 0x80000000u : (1u << (32 - __clz(L - 1u))));
    const unsigned int s = p / SS_MAX_SB;
    return s < SS_MIN_SB ? SS_MIN_SB : s;
}

__device__ __forceinline__ void ss_mma(float c[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                       unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One weight as its two BF16 terms; returns the value the terms carry.
__device__ __forceinline__ float ss_split(float w, unsigned int& hi, unsigned int& lo) {
    const __nv_bfloat16 h = __float2bfloat16(w);
    const float hf = __bfloat162float(h);
    const __nv_bfloat16 l = __float2bfloat16(w - hf);
    hi = (unsigned int)__bfloat16_as_ushort(h);
    lo = (unsigned int)__bfloat16_as_ushort(l);
    return hf + __bfloat162float(l);
}

extern "C" __global__ void __launch_bounds__(SS_THREADS) attn_shared_scan(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    float* __restrict__ WS,
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,     // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int num_seqs,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int q_stride // elements between rows of Q
) {
    __shared__ unsigned int sQ[SS_QMAX * SS_ROW_WORDS];
    __shared__ unsigned int sK[SS_STEP * SS_ROW_WORDS];
    __shared__ unsigned int sV[SS_HD * SS_VT_WORDS];
    __shared__ int sFlag;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int grp = lane >> 2; // MMA row / column group
    const unsigned int t = lane & 3u;   // thread in group

    const unsigned int kv_head = blockIdx.x % num_kv_heads;
    const unsigned int sb = blockIdx.x / num_kv_heads;
    const unsigned int win0 = blockIdx.y * SS_ROWS_MAX;
    if (sb >= SS_MAX_SB || win0 >= num_seqs) return;
    const unsigned int win1 = (win0 + SS_ROWS_MAX < num_seqs) ? (win0 + SS_ROWS_MAX) : num_seqs;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int rows_cap = (SS_QMAX / gqa) < SS_ROWS_MAX ? (SS_QMAX / gqa) : SS_ROWS_MAX;
    const unsigned long long row_elems = (unsigned long long)num_kv_heads * SS_HD;
    const float neg_inf = -__int_as_float(0x7f800000);
    // This thread's two queries: MMA rows grp and grp + 8 of its warp.
    const unsigned int qa = 16u * warp + grp;
    const unsigned int qb = qa + 8u;
    const unsigned int* qa_p = sQ + qa * SS_ROW_WORDS;
    const unsigned int* qb_p = sQ + qb * SS_ROW_WORDS;

    // The window's lengths; nothing to do if none of its rows reaches this
    // superblock. Every thread takes the same branches (shared inputs only).
    int Lw[SS_ROWS_MAX];
    #pragma unroll
    for (unsigned int j = 0; j < SS_ROWS_MAX; j++) Lw[j] = (win0 + j < win1) ? seq_lens[win0 + j] : 0;
    bool any = false;
    #pragma unroll
    for (unsigned int j = 0; j < SS_ROWS_MAX; j++)
        any |= Lw[j] > 0 && sb * ss_sb_positions((unsigned int)Lw[j]) < (unsigned int)Lw[j];
    if (!any) return;

    // Which adjacent rows share this superblock's scan (GROUPS above), for
    // the whole window in one round. `ok` is the length tests, the same on
    // every thread; `diff` is where this thread saw the tables differ.
    unsigned int ok = 0u, diff = 0u;
    unsigned int nbs[SS_ROWS_MAX - 1u], fbs[SS_ROWS_MAX - 1u];
    int xa[SS_ROWS_MAX - 1u], xb[SS_ROWS_MAX - 1u];
    #pragma unroll
    for (unsigned int j = 0; j + 1u < SS_ROWS_MAX; j++) {
        const int La = Lw[j];
        const int Lb = Lw[j + 1u];
        unsigned int nb = 0u, fb = 0u;
        if (La > 0 && Lb >= La) {
            const unsigned int Sa = ss_sb_positions((unsigned int)La);
            const unsigned int st = sb * Sa;
            if (st < (unsigned int)La && ss_sb_positions((unsigned int)Lb) == Sa) {
                const unsigned int lim = (unsigned int)La < st + Sa ? (unsigned int)La : st + Sa;
                nb = (lim - st + block_size - 1u) / block_size;
                fb = st / block_size;
                ok |= 1u << j;
            }
        }
        nbs[j] = nb;
        fbs[j] = fb;
        xa[j] = 0;
        xb[j] = 0;
        if (tid < nb) {
            xa[j] = block_tables[(unsigned long long)(win0 + j) * max_blocks_per_seq + fb + tid];
            xb[j] = block_tables[(unsigned long long)(win0 + j + 1u) * max_blocks_per_seq + fb + tid];
        }
    }
    #pragma unroll
    for (unsigned int j = 0; j + 1u < SS_ROWS_MAX; j++) {
        if (xa[j] != xb[j]) diff |= 1u << j;
        const int* ta = block_tables + (unsigned long long)(win0 + j) * max_blocks_per_seq + fbs[j];
        for (unsigned int i = tid + SS_THREADS; i < nbs[j]; i += SS_THREADS)
            if (ta[i] != ta[max_blocks_per_seq + i]) diff |= 1u << j;
    }
    if (tid == 0u) sFlag = 0;
    __syncthreads();
    if (diff) atomicOr(&sFlag, (int)diff);
    __syncthreads();
    const unsigned int share = ok & ~(unsigned int)sFlag;

    unsigned int row = win0;
    while (row < win1) {
        const unsigned int w0 = row - win0;
        const int L_row = Lw[w0];
        if (L_row <= 0) { row++; continue; }
        const unsigned int S = ss_sb_positions((unsigned int)L_row);
        const unsigned int start = sb * S;
        if (start >= (unsigned int)L_row) { row++; continue; }
        const unsigned int end = start + S;
        unsigned int g = 1;
        while (g < rows_cap && row + g < win1 && ((share >> (w0 + g - 1u)) & 1u)) g++;

        const unsigned int owner = row + g - 1u;
        const unsigned int max_len = (unsigned int)Lw[w0 + g - 1u];
        const unsigned int lim = max_len < end ? max_len : end;
        const int* tbl = block_tables + (unsigned long long)owner * max_blocks_per_seq;
        const unsigned int nq = g * gqa;
        const unsigned int q_words = 16u * ((nq + 15u) / 16u) * 128u; // the active warps' rows

        // Stage Q: sixteen loads issued, then sixteen stores.
        __syncthreads();
        for (unsigned int i0 = tid; i0 < q_words; i0 += SS_THREADS * 16u) {
            unsigned int v[16];
            #pragma unroll
            for (unsigned int u = 0; u < 16u; u++) {
                const unsigned int i = i0 + u * SS_THREADS;
                const unsigned int q = i >> 7;
                v[u] = 0u;
                if (i < q_words && q < nq) {
                    const unsigned int* src = (const unsigned int*)(Q + (unsigned long long)(row + q / gqa) * q_stride
                                                                    + (unsigned long long)(kv_head * gqa + q % gqa) * SS_HD);
                    v[u] = src[i & 127u];
                }
            }
            #pragma unroll
            for (unsigned int u = 0; u < 16u; u++) {
                const unsigned int i = i0 + u * SS_THREADS;
                if (i < q_words) sQ[(i >> 7) * SS_ROW_WORDS + (i & 127u)] = v[u];
            }
        }
        __syncthreads();

        const bool active = 16u * warp < nq;
        const unsigned int La = (qa < nq) ? (unsigned int)Lw[w0 + qa / gqa] : 0u;
        const unsigned int Lb = (qb < nq) ? (unsigned int)Lw[w0 + qb / gqa] : 0u;

        float ma = -1e30f, mb = -1e30f, la = 0.0f, lb = 0.0f;
        float o[SS_HD / 8u][4];
        #pragma unroll
        for (unsigned int dc = 0; dc < SS_HD / 8u; dc++) {
            o[dc][0] = 0.0f; o[dc][1] = 0.0f; o[dc][2] = 0.0f; o[dc][3] = 0.0f;
        }

        // A step's 16 keys and values of this KV head, in registers. A warp takes
        // whole rows (row = warp + 3j, 16 bytes a lane); all of a thread's loads
        // are issued before any is used, so their latencies overlap, and the
        // next step's are issued before this step's arithmetic.
        uint4 kx[SS_STAGE_ROWS], vx[SS_STAGE_ROWS];
        auto fetch = [&](unsigned int p0) {
            const unsigned int lblk = p0 / block_size;
            const unsigned long long slot0 = (unsigned long long)(unsigned int)tbl[lblk] * block_size + (p0 % block_size);
            const __nv_bfloat16* kb = K_cache + slot0 * row_elems + (unsigned long long)kv_head * SS_HD;
            const __nv_bfloat16* vb = V_cache + slot0 * row_elems + (unsigned long long)kv_head * SS_HD;
            #pragma unroll
            for (unsigned int j = 0; j < SS_STAGE_ROWS; j++) {
                const unsigned int r = warp + SS_WARPS * j;
                kx[j] = make_uint4(0u, 0u, 0u, 0u);
                vx[j] = make_uint4(0u, 0u, 0u, 0u);
                if (r < SS_STEP && p0 + r < max_len) {
                    kx[j] = *(const uint4*)(kb + (unsigned long long)r * row_elems + lane * 8u);
                    vx[j] = *(const uint4*)(vb + (unsigned long long)r * row_elems + lane * 8u);
                }
            }
        };
        fetch(start);

        for (unsigned int p0 = start; p0 < lim; p0 += SS_STEP) {
            // ── Stage the fetched step ──
            __syncthreads();
            #pragma unroll
            for (unsigned int j = 0; j < SS_STAGE_ROWS; j++) {
                const unsigned int r = warp + SS_WARPS * j;
                if (r >= SS_STEP) continue;
                *(uint4*)&sK[r * SS_ROW_WORDS + lane * 4u] = kx[j];
                // V transposed: dim row d holds the 16 positions, so a thread's
                // (position 2t, 2t+1) pair is one word.
                unsigned short* vt = (unsigned short*)sV;
                const unsigned int d0 = lane * 8u;
                vt[(d0 + 0u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].x & 0xFFFFu);
                vt[(d0 + 1u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].x >> 16);
                vt[(d0 + 2u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].y & 0xFFFFu);
                vt[(d0 + 3u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].y >> 16);
                vt[(d0 + 4u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].z & 0xFFFFu);
                vt[(d0 + 5u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].z >> 16);
                vt[(d0 + 6u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].w & 0xFFFFu);
                vt[(d0 + 7u) * (SS_VT_WORDS * 2u) + r] = (unsigned short)(vx[j].w >> 16);
            }
            __syncthreads();
            if (p0 + SS_STEP < lim) fetch(p0 + SS_STEP);
            if (!active) continue;

            // ── Scores: Q [16 queries x 256] . K^T, positions p0..+7 and p0+8..+15 ──
            float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned int* k0 = sK + grp * SS_ROW_WORDS;
            const unsigned int* k1 = sK + (grp + 8u) * SS_ROW_WORDS;
            #pragma unroll
            for (unsigned int c = 0; c < SS_HD / 16u; c++) {
                const unsigned int a0 = qa_p[c * 8u + t];
                const unsigned int a1 = qb_p[c * 8u + t];
                const unsigned int a2 = qa_p[c * 8u + 4u + t];
                const unsigned int a3 = qb_p[c * 8u + 4u + t];
                ss_mma(c0, a0, a1, a2, a3, k0[c * 8u + t], k0[c * 8u + 4u + t]);
                ss_mma(c1, a0, a1, a2, a3, k1[c * 8u + t], k1[c * 8u + 4u + t]);
            }

            // This thread's positions: p0 + 2t + {0,1} and p0 + 8 + 2t + {0,1}.
            const unsigned int pp = p0 + 2u * t;
            float sa[4], sbv[4];
            sa[0] = (pp < La) ? c0[0] * inv_sqrt_d : neg_inf;
            sa[1] = (pp + 1u < La) ? c0[1] * inv_sqrt_d : neg_inf;
            sa[2] = (pp + 8u < La) ? c1[0] * inv_sqrt_d : neg_inf;
            sa[3] = (pp + 9u < La) ? c1[1] * inv_sqrt_d : neg_inf;
            sbv[0] = (pp < Lb) ? c0[2] * inv_sqrt_d : neg_inf;
            sbv[1] = (pp + 1u < Lb) ? c0[3] * inv_sqrt_d : neg_inf;
            sbv[2] = (pp + 8u < Lb) ? c1[2] * inv_sqrt_d : neg_inf;
            sbv[3] = (pp + 9u < Lb) ? c1[3] * inv_sqrt_d : neg_inf;

            // Step max per query, over the group's four threads.
            float ta_ = fmaxf(fmaxf(sa[0], sa[1]), fmaxf(sa[2], sa[3]));
            float tb_ = fmaxf(fmaxf(sbv[0], sbv[1]), fmaxf(sbv[2], sbv[3]));
            ta_ = fmaxf(ta_, __shfl_xor_sync(0xffffffffu, ta_, 1));
            tb_ = fmaxf(tb_, __shfl_xor_sync(0xffffffffu, tb_, 1));
            ta_ = fmaxf(ta_, __shfl_xor_sync(0xffffffffu, ta_, 2));
            tb_ = fmaxf(tb_, __shfl_xor_sync(0xffffffffu, tb_, 2));
            const float na = fmaxf(ma, ta_);
            const float nb_ = fmaxf(mb, tb_);
            const float sca = __expf(ma - na);
            const float scb = __expf(mb - nb_);
            ma = na;
            mb = nb_;

            unsigned int ha[4], lo_a[4], hb[4], lo_b[4];
            float wa[4], wb[4];
            #pragma unroll
            for (unsigned int i = 0; i < 4u; i++) {
                const float xa = (sa[i] == neg_inf) ? 0.0f : __expf(sa[i] - na);
                const float xb = (sbv[i] == neg_inf) ? 0.0f : __expf(sbv[i] - nb_);
                wa[i] = ss_split(xa, ha[i], lo_a[i]);
                wb[i] = ss_split(xb, hb[i], lo_b[i]);
            }
            float sum_a = (wa[0] + wa[1]) + (wa[2] + wa[3]);
            float sum_b = (wb[0] + wb[1]) + (wb[2] + wb[3]);
            sum_a = sum_a + __shfl_xor_sync(0xffffffffu, sum_a, 1);
            sum_b = sum_b + __shfl_xor_sync(0xffffffffu, sum_b, 1);
            sum_a = sum_a + __shfl_xor_sync(0xffffffffu, sum_a, 2);
            sum_b = sum_b + __shfl_xor_sync(0xffffffffu, sum_b, 2);
            la = la * sca + sum_a;
            lb = lb * scb + sum_b;

            // x * 1.0f is x: skipping the rescale when both are 1 changes no bit.
            if (sca != 1.0f || scb != 1.0f) {
                #pragma unroll
                for (unsigned int dc = 0; dc < SS_HD / 8u; dc++) {
                    o[dc][0] *= sca; o[dc][1] *= sca; o[dc][2] *= scb; o[dc][3] *= scb;
                }
            }

            // ── Values: W [16 queries x 16 positions] . V [16 positions x 256] ──
            const unsigned int ah0 = ha[0] | (ha[1] << 16), ah1 = hb[0] | (hb[1] << 16);
            const unsigned int ah2 = ha[2] | (ha[3] << 16), ah3 = hb[2] | (hb[3] << 16);
            const unsigned int al0 = lo_a[0] | (lo_a[1] << 16), al1 = lo_b[0] | (lo_b[1] << 16);
            const unsigned int al2 = lo_a[2] | (lo_a[3] << 16), al3 = lo_b[2] | (lo_b[3] << 16);
            #pragma unroll
            for (unsigned int dc = 0; dc < SS_HD / 8u; dc++) {
                const unsigned int* v = sV + (8u * dc + grp) * SS_VT_WORDS;
                const unsigned int b0 = v[t];
                const unsigned int b1 = v[4u + t];
                ss_mma(o[dc], ah0, ah1, ah2, ah3, b0, b1);
                ss_mma(o[dc], al0, al1, al2, al3, b0, b1);
            }
        }

        // ── Partials: this thread holds dims 8*dc + 2t + {0,1} of its two queries ──
        if (active && qa < nq) {
            float* rec = WS + ((unsigned long long)((row + qa / gqa) * num_q_heads + kv_head * gqa + qa % gqa) * SS_MAX_SB + sb) * SS_REC;
            if (t == 0u) { rec[0] = ma; rec[1] = la; }
            #pragma unroll
            for (unsigned int dc = 0; dc < SS_HD / 8u; dc++) {
                rec[2u + 8u * dc + 2u * t] = o[dc][0] + 0.0f;
                rec[2u + 8u * dc + 2u * t + 1u] = o[dc][1] + 0.0f;
            }
        }
        if (active && qb < nq) {
            float* rec = WS + ((unsigned long long)((row + qb / gqa) * num_q_heads + kv_head * gqa + qb % gqa) * SS_MAX_SB + sb) * SS_REC;
            if (t == 0u) { rec[0] = mb; rec[1] = lb; }
            #pragma unroll
            for (unsigned int dc = 0; dc < SS_HD / 8u; dc++) {
                rec[2u + 8u * dc + 2u * t] = o[dc][2] + 0.0f;
                rec[2u + 8u * dc + 2u * t + 1u] = o[dc][3] + 0.0f;
            }
        }
        row += g;
    }
}

// Fold a row's superblock partials in superblock order and normalize.
extern "C" __global__ void __launch_bounds__(SS_HD) attn_shared_scan_reduce(
    const float* __restrict__ WS,
    __nv_bfloat16* __restrict__ O, // [num_seqs, num_q_heads, 256]
    const int* __restrict__ seq_lens,
    const unsigned int num_q_heads
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const int L = seq_lens[row];
    if (L <= 0) return;
    const unsigned int S = ss_sb_positions((unsigned int)L);
    const unsigned int n = ((unsigned int)L + S - 1u) / S;
    const float* rec = WS + (unsigned long long)(row * num_q_heads + q_head) * SS_MAX_SB * SS_REC;
    float m = rec[0];
    float l = rec[1];
    float o = rec[2u + d];
    for (unsigned int s = 1; s < n; s++) {
        const float* r = rec + (unsigned long long)s * SS_REC;
        const float ms = r[0];
        const float mn = fmaxf(m, ms);
        const float a = __expf(m - mn);
        const float b = __expf(ms - mn);
        l = l * a + r[1] * b;
        o = o * a + r[2u + d] * b;
        m = mn;
    }
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[(unsigned long long)(row * num_q_heads + q_head) * SS_HD + d] = __float2bfloat16(o * inv_l);
}
