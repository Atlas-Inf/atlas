// SPDX-License-Identifier: AGPL-3.0-only

// Batched paged prefill attention, BF16 KV cache, HDIM 256: 128 query rows a
// CTA (lever ATLAS_ATTN_ROWS128=1). Same arguments, masks (causal, sliding
// window, PREFILL_STATE_END prefix-bidirectional) and arithmetic as
// `inferspark_prefill_paged_batched_64` (common/prefill_paged_compute.cuh),
// laid out FlashAttention-2 style:
//
//   * every warp owns 16 query rows end to end: Q as MMA A-fragments in
//     registers, the S tile, the softmax and the output accumulator; P goes
//     from the S accumulators straight to PV A-fragments, so there is no P
//     tile, no row-stat exchange and one barrier per KV tile;
//   * K and V tiles are both double buffered and shared by 128 rows (the BR=64
//     kernel shares them by 64 and runs QK^T on half its warps).
//
// Per row the KV tiles, the order of every MMA, the online-softmax updates
// and every rounding are the BR=64 kernel's. A taller tile can only add KV
// tiles that are fully masked for a row (its causal or state bound ends
// earlier); such a tile leaves m, l and O unchanged (exp(-1e30 - m) = 0), so
// the output is bit-identical.
//
// Grid (num_q_heads, ceil(max_len / 128), batch), block 256. The `_gqa`
// entries (score passes) pack a KV head's query heads into each tile and can
// split a long KV range across a 2-CTA cluster; see the end of the file.

#include <cooperative_groups.h>
#include <cuda.h>   // CUtensorMap
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace rows128 {
constexpr int HD = 256;           // head dim
constexpr int BC = 32;            // KV tile
constexpr int BR = 128;           // query rows a CTA
constexpr int HP = HD + 8;        // smem row pitch (as HDIM_PAD)
constexpr int NT_O = HD / 8;      // output n-tiles a warp
constexpr int KV_CHUNKS = BC * (HD / 8);
// In the split entries, a stream with more than SPLIT_MIN_KEYS keys and at
// least SPLIT_KEYS_PER_ROW keys a row (a knowledge pack under a short state)
// shares each tile's KV range across the cluster; any other stream gives
// each rank whole tiles of its own, computed exactly as the unsplit entry
// does. The rule reads the stream alone, so a merged pass scores as each
// request alone. A split stream's tiles each read over 3,072 keys (its state
// ends at least 3/4 of the way through its KV), so no chunk is empty.
constexpr unsigned SPLIT_MIN_KEYS = 4096, SPLIT_KEYS_PER_ROW = 4;
__device__ __forceinline__ bool stream_splits(unsigned kv_len, unsigned rows) {
    return kv_len > SPLIT_MIN_KEYS && kv_len >= SPLIT_KEYS_PER_ROW * rows;
}
static_assert(KV_CHUNKS % 256 == 0, "load_kv: whole rounds of 256 threads");

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group 0;"); }

// As prefill_paged_compute.cuh: bf16 pair -> fp16 pair, through float.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    __half2 h2 = __floats2half2_rn(__bfloat162float(lo), __bfloat162float(hi));
    return *reinterpret_cast<const unsigned*>(&h2);
}
__device__ __forceinline__ unsigned f16x2(float lo, float hi) {
    const unsigned l = __half_as_ushort(__float2half_rn(lo));
    const unsigned h = __half_as_ushort(__float2half_rn(hi));
    return l | (h << 16);
}
__device__ __forceinline__ float softmax_exp(float x) { return __expf(x); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// A BF16 pair (as loaded) to the FP16 pair bf16x2_to_f16x2 makes of it.
__device__ __forceinline__ unsigned bf16w_to_f16x2(uint32_t w) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w);
    return bf16x2_to_f16x2(__low2bfloat16(v), __high2bfloat16(v));
}

// One KV tile [kv_s, kv_s + BC) of one head into a pitch-HP smem tile (the
// BF16 LOAD_KV_TILE of inferspark_prefill_paged_batched.cu). Thread t copies
// 16 bytes of rows t / 32 + 8 j; with a power-of-two page of at least 8 rows
// its rows j and j + 1 share a page, so each pair costs one table read.
__device__ __forceinline__ void load_kv(__nv_bfloat16 (*dst)[HP], const __nv_bfloat16* cache, const int* bt,
                                        unsigned kv_s, unsigned kv_l, unsigned kvh, unsigned cbs,
                                        unsigned num_kv_heads, unsigned tid) {
    const unsigned long long ps = (unsigned long long)cbs * num_kv_heads * HD;
    const unsigned long long rs = (unsigned long long)num_kv_heads * HD;
    const unsigned r0 = tid / (HD / 8), col = (tid % (HD / 8)) * 8;
    const bool pow2 = (cbs & (cbs - 1)) == 0 && cbs >= 16 && (kv_s & 7) == 0;
    const unsigned sh = __ffs(cbs) - 1;
    const __nv_bfloat16* base = cache + kvh * HD + col;
    if (pow2) {
#pragma unroll
        for (unsigned j = 0; j < 4; j += 2) {
            // rows r0 + 8 j and r0 + 8 j + 8 lie in one 16-row-aligned window
            const unsigned pos = kv_s + r0 + 8 * j;
            // Past kv_l both rows are zero-filled: do not read the table
            // past its last page (the base LOAD_KV_TILE guards likewise).
            const unsigned pb = pos < kv_l ? (unsigned)bt[pos >> sh] : 0u;
#pragma unroll
            for (unsigned h = 0; h < 2; h++) {
                const unsigned p = pos + 8 * h, row = r0 + 8 * (j + h);
                if (p < kv_l) {
                    const unsigned pg = (p >> sh) == (pos >> sh) ? pb : (unsigned)bt[p >> sh];
                    cp16(&dst[row][col], base + pg * ps + (p & (cbs - 1)) * rs);
                } else {
                    *reinterpret_cast<uint4*>(&dst[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
    } else {
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const unsigned row = r0 + 8 * j, p = kv_s + row;
            if (p < kv_l) {
                cp16(&dst[row][col], base + (unsigned)bt[p / cbs] * ps + (p % cbs) * rs);
            } else {
                *reinterpret_cast<uint4*>(&dst[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}
// ── TMA (the _tma entry): a KV tile as 4 column chunks of 64 x 32 rows,
// each written by the copy engine with the 128-byte swizzle (16-byte unit u
// of row r at unit u ^ (r & 7)), one 16-row box per page and chunk. ──
constexpr int TILE_TMA = BC * HD * 2;   // 16 KB, no padding
constexpr int TILE_PAD = BC * HP * 2;   // the cp.async tile
__device__ __forceinline__ unsigned swz(unsigned row, unsigned col) {
    return (col / 64) * (BC * 128) + row * 128 + ((((col % 64) / 8) ^ (row & 7)) * 16) + (col % 8) * 2;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(bar)),
                 "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile(
        "{\n.reg .pred P;\nW_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra D_%=;\nbra W_%=;\nD_%=:\n}\n" ::"r"((unsigned)__cvta_generic_to_shared(bar)),
        "r"(parity) : "memory");
}
// This thread's generic smem accesses before a later bulk copy into them.
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tma_2d(const CUtensorMap* map, uint64_t* bar, void* dst, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::
            "r"((unsigned)__cvta_generic_to_shared(dst)), "l"((const void*)map),
        "r"((unsigned)__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}
// Thread 0: tile [kv_s, kv_s + BC) of kv head kvh, K and V, completing on
// `bar`. Pages are 16 rows; a page wholly past kv_l is not loaded (its V
// rows are zeroed by v_to_f16_tma, its K rows masked before they are read).
__device__ __forceinline__ void issue_tma(unsigned char* k, unsigned char* v, uint64_t* bar, const CUtensorMap* km,
                                          const CUtensorMap* vm, const int* bt, unsigned kv_s, unsigned kv_l,
                                          unsigned kvh) {
    const unsigned pages = kv_s + 16 < kv_l ? 2 : 1;
    mbar_expect_tx(bar, pages * 4 * 16 * 128 * 2);
    for (unsigned i = 0; i < pages; i++) {
        const int row = bt[kv_s / 16 + i] * 16;
#pragma unroll
        for (unsigned c = 0; c < 4; c++) {
            const unsigned off = c * (BC * 128) + i * 16 * 128;
            tma_2d(km, bar, k + off, kvh * HD + c * 64, row);
            tma_2d(vm, bar, v + off, kvh * HD + c * 64, row);
        }
    }
}
// v_to_f16 for the swizzled tile: rows of 16-byte units, zeroed past `rows`.
__device__ __forceinline__ void v_to_f16_tma(unsigned char* v, unsigned rows, unsigned tid) {
#pragma unroll
    for (unsigned j = 0; j < TILE_TMA / 16 / 256; j++) {
        const unsigned u = tid + 256 * j, r = (u * 16 % (BC * 128)) / 128;
        uint4* c = reinterpret_cast<uint4*>(v + u * 16);
        uint4 w = make_uint4(0, 0, 0, 0);
        if (r < rows) {
            w = *c;
            w.x = bf16w_to_f16x2(w.x); w.y = bf16w_to_f16x2(w.y);
            w.z = bf16w_to_f16x2(w.z); w.w = bf16w_to_f16x2(w.w);
        }
        *c = w;
    }
}

// V to FP16 in place, each thread over exactly the chunks load_kv gave it
// (its own cp.async writes are visible to it after its own wait, no barrier).
// The BR=64 kernel converts each pair as it builds a fragment; same
// conversion, same bits, once for all warps.
__device__ __forceinline__ void v_to_f16(__nv_bfloat16 (*v)[HP], unsigned tid) {
#pragma unroll
    for (unsigned j = 0; j < KV_CHUNKS / 256; j++) {
        const unsigned i = tid + 256 * j;
        uint4* c = reinterpret_cast<uint4*>(&v[i / (HD / 8)][(i % (HD / 8)) * 8]);
        uint4 w = *c;
        w.x = bf16w_to_f16x2(w.x); w.y = bf16w_to_f16x2(w.y);
        w.z = bf16w_to_f16x2(w.z); w.w = bf16w_to_f16x2(w.w);
        *c = w;
    }
}
}  // namespace rows128

// The CS CTAs of a split tile hold O, m and l of their KV chunks for the same
// 128 rows. Each publishes m, l and (in two 128-column halves) O in its own
// smem; rank r reads every rank's rows [r 128 / CS, (r + 1) 128 / CS) through
// distributed shared memory and writes them, combined in rank order:
// O = sum_s exp(m_s - M) O_s / sum_s exp(m_s - M) l_s, M = max_s m_s. Keys
// are masked only from above here (no sliding window in a split), so a chunk
// either starts on a key the row reads or reads none of them.
template <unsigned CS, typename RowOut>
__device__ __forceinline__ void split_combine(unsigned char* smem, const float (&acc_o)[rows128::NT_O][4], float m0,
                                              float m1, float l0, float l1, unsigned rank, unsigned warp, unsigned lane,
                                              unsigned rows, RowOut row_out) {
    using namespace rows128;
    namespace cg = cooperative_groups;
    static_assert(BR * 2 * 4 + BR * 128 * 4 <= 4 * BC * HP * 2, "m, l and half of O fit the KV tiles");
    cg::cluster_group cl = cg::this_cluster();
    float* const ml = reinterpret_cast<float*>(smem);   // [BR][2]
    float* const so = ml + 2 * BR;                     // [BR][128], 8-float groups XOR-swizzled by row
    const auto at = [](unsigned r, unsigned c) { return r * 128 + (c ^ ((r & 7) << 3)); };
    const unsigned grp = lane >> 2, tq = lane & 3, row0 = warp * 16 + grp, row1 = row0 + 8;
    __syncthreads();   // every warp is done with the last K/V tile
    if (tq == 0) {
        ml[2 * row0] = m0; ml[2 * row0 + 1] = l0;
        ml[2 * row1] = m1; ml[2 * row1 + 1] = l1;
    }
    constexpr unsigned SLICE = BR / CS;
    const unsigned tid = warp * 32 + lane;
#pragma unroll
    for (unsigned half = 0; half < 2; half++) {
#pragma unroll
        for (int i = 0; i < NT_O / 2; i++) {
            const unsigned c0 = i * 8 + tq * 2;
            const int nt = half * (NT_O / 2) + i;
            *reinterpret_cast<float2*>(&so[at(row0, c0)]) = make_float2(acc_o[nt][0], acc_o[nt][1]);
            *reinterpret_cast<float2*>(&so[at(row1, c0)]) = make_float2(acc_o[nt][2], acc_o[nt][3]);
        }
        cl.sync();   // every rank's m, l and this half of O are published
        for (unsigned e = tid; e < SLICE * 64; e += 256) {
            const unsigned r = rank * SLICE + e / 64, c = (e % 64) * 2;
            if (r >= rows) continue;
            float mx = -1e30f;
#pragma unroll
            for (unsigned s = 0; s < CS; s++) mx = fmaxf(mx, cl.map_shared_rank(ml, s)[2 * r]);
            float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
            for (unsigned s = 0; s < CS; s++) {
                const float* pml = cl.map_shared_rank(ml, s);
                // A chunk wholly past the row's last key never left -1e30
                // (its l and O are not a softmax): it weighs nothing.
                const float w = pml[2 * r] == -1e30f ? 0.f : softmax_exp(pml[2 * r] - mx);
                const float2 v = *reinterpret_cast<const float2*>(&cl.map_shared_rank(so, s)[at(r, c)]);
                l += w * pml[2 * r + 1];
                o0 += w * v.x;
                o1 += w * v.y;
            }
            const float il = (l > 0) ? (1.f / l) : 0;
            const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(o0 * il));
            const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(o1 * il));
            *reinterpret_cast<unsigned*>(&row_out(r)[half * 128 + c]) = lo | (hi << 16);
        }
        cl.sync();   // no rank still reads this half (or, at the end, this CTA's smem)
    }
}

template <bool TMA, bool GQA, unsigned CS = 1>
__device__ __forceinline__ void rows128_body(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* const* __restrict__ block_table_ptrs,
    const unsigned int batch_size,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ kv_lens,
    const int* __restrict__ state_end,
    const unsigned int q_len,
    unsigned int kv_len,
    unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal_mask_enabled,
    const float inv_sqrt_d,
    const CUtensorMap* kmap, const CUtensorMap* vmap
) {
    using namespace rows128;
    // GQA: blockIdx.x is the KV head and the CTA's rows are its G query
    // heads' rows interleaved, flat index f = row * G + head-in-group, so a
    // short stream fills one tile instead of G mostly empty ones.
    const unsigned G = num_q_heads / num_kv_heads;
    const unsigned b = blockIdx.z;
    static_assert(CS == 1 || GQA, "split entries are GQA-packed");
    const unsigned q_head = GQA ? 0u : blockIdx.x;
    const unsigned kv_head = GQA ? blockIdx.x / CS : q_head / G;
    if (b >= batch_size || (GQA ? kv_head >= num_kv_heads : q_head >= num_q_heads)) return;
    const int* const __restrict__ block_table = block_table_ptrs[b];
    unsigned q_base_b, q_len_eff = q_len;
    if (cu_seqlens != nullptr) {
        q_base_b = (unsigned)cu_seqlens[b];
        q_len_eff = (unsigned)(cu_seqlens[b + 1] - cu_seqlens[b]);
    } else {
        q_base_b = b * q_len;
    }
    if (kv_lens != nullptr) {
        kv_len = (unsigned)kv_lens[b];
        if (cu_seqlens != nullptr && kv_len >= q_len_eff) q_offset = kv_len - q_len_eff;
    }
    const int s_end = state_end != nullptr ? state_end[b] : -1;
    // Split entries: rank r of the cluster takes chunk r of a split stream's
    // tile, or whole tile CS y + r of another stream.
    [[maybe_unused]] const unsigned rank = blockIdx.x % CS;
    const bool split = CS > 1 && stream_splits(kv_len, q_len_eff);
    const unsigned q_block = CS > 1 && !split ? blockIdx.y * CS + rank : blockIdx.y;
    const unsigned tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned grp = lane >> 2, tq = lane & 3;
    // Tile rows are flat indices; pos() is a flat index's row in the stream
    // and head_of() its query head (both the identity without GQA).
    const unsigned n_flat = GQA ? q_len_eff * G : q_len_eff;
    const auto pos = [&](unsigned f) { return GQA ? f / G : f; };
    const auto head_of = [&](unsigned f) { return GQA ? kv_head * G + f % G : q_head; };
    const unsigned q_start = q_block * BR;
    if (q_start >= n_flat) return;
    const unsigned q_tile_end = min(q_start + BR, n_flat);
    const unsigned q_tile_len = q_tile_end - q_start;
    // First and one-past-last stream row the tile touches.
    const unsigned r_start = pos(q_start), r_end = pos(q_tile_end - 1) + 1;
    const unsigned q_seq_stride = num_q_heads * head_dim;
    const unsigned long long q_batch_off = (unsigned long long)q_base_b * q_seq_stride;

    // K tiles then V tiles, two each: padded rows (cp.async) or swizzled chunks (TMA).
    constexpr unsigned TB = TMA ? TILE_TMA : TILE_PAD;
    __shared__ __align__(1024) unsigned char smem_kv[4 * TILE_PAD];
    __shared__ __align__(8) uint64_t bars[2];
    unsigned char* const kt[2] = {smem_kv, smem_kv + TB};
    unsigned char* const vt[2] = {smem_kv + 2 * TB, smem_kv + 3 * TB};
    const auto pad = [](unsigned char* t) { return reinterpret_cast<__nv_bfloat16(*)[HP]>(t); };
    // Byte offset of (row, col) in a tile.
    const auto at = [](unsigned row, unsigned col) { return TMA ? swz(row, col) : (row * HP + col) * 2; };
    const auto rows_of = [&](unsigned blk) { return min((unsigned)BC, kv_len - blk * BC); };

    // KV extent of the tile, as the BR=64 kernel derives it for its own rows.
    unsigned num_kv_blocks = (kv_len + BC - 1) / BC;
    if (s_end < 0) {
        num_kv_blocks = min(num_kv_blocks, (q_offset + r_end - 1) / BC + 1);
    } else if (q_offset + r_end <= (unsigned)s_end) {
        num_kv_blocks = min(num_kv_blocks, ((unsigned)s_end + BC - 1) / BC);
    }
    unsigned kv_block_lo = 0;
    if (causal_mask_enabled && sliding_window > 0) {
        const unsigned q_abs_lo = q_offset + r_start;
        if (q_abs_lo + 1 > sliding_window) kv_block_lo = (q_abs_lo + 1 - sliding_window) / BC;
    }
    if (kv_block_lo >= num_kv_blocks && num_kv_blocks > 0) kv_block_lo = num_kv_blocks - 1;
    if (split) {
        const unsigned chunk = (num_kv_blocks - kv_block_lo + CS - 1) / CS;
        kv_block_lo += rank * chunk;
        num_kv_blocks = min(num_kv_blocks, kv_block_lo + chunk);
    }

    // This warp's rows; a warp with none still loads tiles and syncs.
    const unsigned wm = warp * 16, row0 = wm + grp, row1 = row0 + 8;
    const bool active = wm < q_tile_len;

    if (TMA) {
        if (tid == 0) {
            mbar_init(&bars[0]);
            mbar_init(&bars[1]);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
        if (num_kv_blocks > 0 && tid == 0)
            issue_tma(kt[0], vt[0], &bars[0], kmap, vmap, block_table, kv_block_lo * BC, kv_len, kv_head);
    } else {
        if (num_kv_blocks > 0) {
            load_kv(pad(kt[0]), K_cache, block_table, kv_block_lo * BC, kv_len, kv_head, cache_block_size,
                    num_kv_heads, tid);
            load_kv(pad(vt[0]), V_cache, block_table, kv_block_lo * BC, kv_len, kv_head, cache_block_size,
                    num_kv_heads, tid);
        }
        cp_commit();
    }

    // Q A-fragments for the warp's 16 rows (rows past the stream read as 0).
    uint32_t qa[HD / 16][4];
    {
        const unsigned f0 = q_start + row0, f1 = q_start + row1;
        const bool v0 = f0 < n_flat, v1 = f1 < n_flat;
        const __nv_bfloat16* qb = Q + q_batch_off + 2 * tq;
        const __nv_bfloat16* q0 = qb + (unsigned long long)pos(f0) * q_seq_stride + head_of(f0) * head_dim;
        const __nv_bfloat16* q1 = qb + (unsigned long long)pos(f1) * q_seq_stride + head_of(f1) * head_dim;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + ks * 16) : 0u;
            qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + ks * 16) : 0u;
            qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + ks * 16 + 8) : 0u;
            qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + ks * 16 + 8) : 0u;
        }
    }

    if (TMA) {
        if (num_kv_blocks > 0) {
            mbar_wait(&bars[0], 0);
            v_to_f16_tma(vt[0], rows_of(kv_block_lo), tid);
        }
    } else {
        cp_wait();
        v_to_f16(pad(vt[0]), tid);
    }

    float acc_o[NT_O][4];
#pragma unroll
    for (int i = 0; i < NT_O; i++) acc_o[i][0] = acc_o[i][1] = acc_o[i][2] = acc_o[i][3] = 0.0f;
    float m_r0 = -1e30f, m_r1 = -1e30f, l_r0 = 0.0f, l_r1 = 0.0f;

    for (unsigned kv_block = kv_block_lo; kv_block < num_kv_blocks; kv_block++) {
        const unsigned buf = (kv_block - kv_block_lo) & 1;
        const unsigned kv_start = kv_block * BC;
        const unsigned kv_tile_len = min(kv_start + BC, kv_len) - kv_start;
        if (TMA) fence_async();   // this thread's accesses of buf ^ 1 precede its next fill
        __syncthreads();   // tile kv_block is in buf (V as FP16); buf ^ 1 is free
        const bool next = kv_block + 1 < num_kv_blocks;
        if (TMA) {
            if (next && tid == 0)
                issue_tma(kt[buf ^ 1], vt[buf ^ 1], &bars[buf ^ 1], kmap, vmap, block_table, (kv_block + 1) * BC,
                          kv_len, kv_head);
        } else {
            if (next) {
                load_kv(pad(kt[buf ^ 1]), K_cache, block_table, (kv_block + 1) * BC, kv_len, kv_head,
                        cache_block_size, num_kv_heads, tid);
                load_kv(pad(vt[buf ^ 1]), V_cache, block_table, (kv_block + 1) * BC, kv_len, kv_head,
                        cache_block_size, num_kv_heads, tid);
            }
            cp_commit();
        }
        if (active) {

        // ── S = Q K^T (16 rows x 32 keys) ──
        float acc_s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) acc_s[i][0] = acc_s[i][1] = acc_s[i][2] = acc_s[i][3] = 0.0f;
        // B fragments by ldmatrix: matrix j of an x4 is (keys 8 (j >> 1) .., dims 8 (j & 1) ..)
        // of an n-tile pair, i.e. b0 / b1 of n-tile 2 np and 2 np + 1.
        const unsigned lrow = (lane & 7) + ((lane >> 4) << 3), lcol = ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const unsigned kb = ks * 16;
            uint32_t kf[2][4];
#pragma unroll
            for (int np = 0; np < 2; np++) ldsm_x4(kf[np], kt[buf] + at(np * 16 + lrow, kb + lcol));
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const unsigned b0 = kf[nt >> 1][(nt & 1) * 2], b1 = kf[nt >> 1][(nt & 1) * 2 + 1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]), "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                    : "r"(qa[ks][0]), "r"(qa[ks][1]), "r"(qa[ks][2]), "r"(qa[ks][3]), "r"(b0), "r"(b1),
                      "f"(acc_s[nt][0]), "f"(acc_s[nt][1]), "f"(acc_s[nt][2]), "f"(acc_s[nt][3]));
            }
        }

        // ── masks and online softmax (the BR=64 kernel's, row for row) ──
        // A tile no mask can touch for any of the warp's rows skips them.
        const unsigned qw_lo = q_offset + pos(q_start + wm), qw_hi = q_offset + pos(q_start + wm + 15);
        const bool full = kv_start + BC <= kv_len && wm + 16 <= q_tile_len && sliding_window == 0 &&
            (s_end >= 0 ? (qw_hi < (unsigned)s_end || qw_lo >= (unsigned)s_end
                               ? kv_start + BC <= (qw_lo < (unsigned)s_end ? (unsigned)s_end : kv_len)
                               : kv_start + BC <= (unsigned)s_end)
                        : (!causal_mask_enabled || kv_start + BC - 1 <= qw_lo));
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
            acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
            if (full) continue;
            const unsigned c0 = nt * 8 + tq * 2, c1 = c0 + 1;
            const unsigned qr0 = q_offset + pos(q_start + row0), qr1 = q_offset + pos(q_start + row1);
            if (s_end >= 0) {
                const unsigned lim0 = (qr0 < (unsigned)s_end) ? (unsigned)s_end : kv_len;
                const unsigned lim1 = (qr1 < (unsigned)s_end) ? (unsigned)s_end : kv_len;
                if (kv_start + c0 >= lim0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 >= lim0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 >= lim1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 >= lim1) acc_s[nt][3] = -1e30f;
            } else if (causal_mask_enabled) {
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
            }
            if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
            if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
            if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
            if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            if (sliding_window > 0) {
                if (qr0 >= kv_start + c0 && qr0 - (kv_start + c0) >= sliding_window) acc_s[nt][0] = -1e30f;
                if (qr0 >= kv_start + c1 && qr0 - (kv_start + c1) >= sliding_window) acc_s[nt][1] = -1e30f;
                if (qr1 >= kv_start + c0 && qr1 - (kv_start + c0) >= sliding_window) acc_s[nt][2] = -1e30f;
                if (qr1 >= kv_start + c1 && qr1 - (kv_start + c1) >= sliding_window) acc_s[nt][3] = -1e30f;
            }
        }
        float rmax0 = -1e30f, rmax1 = -1e30f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
            rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
        }
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));
        const float mn0 = fmaxf(m_r0, rmax0);
        if (mn0 != m_r0) {
            const float eo0 = softmax_exp(m_r0 - mn0);
            l_r0 *= eo0;
#pragma unroll
            for (int i = 0; i < NT_O; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        const float mn1 = fmaxf(m_r1, rmax1);
        if (mn1 != m_r1) {
            const float eo1 = softmax_exp(m_r1 - mn1);
            l_r1 *= eo1;
#pragma unroll
            for (int i = 0; i < NT_O; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
            m_r1 = mn1;
        }
        float sum0 = 0, sum1 = 0;
        uint32_t pa[4][2];   // P as FP16 pairs: [n-tile][row0 | row1]
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const float p00 = softmax_exp(acc_s[nt][0] - m_r0), p01 = softmax_exp(acc_s[nt][1] - m_r0);
            const float p10 = softmax_exp(acc_s[nt][2] - m_r1), p11 = softmax_exp(acc_s[nt][3] - m_r1);
            sum0 += p00 + p01;
            sum1 += p10 + p11;
            pa[nt][0] = f16x2(p00, p01);
            pa[nt][1] = f16x2(p10, p11);
        }
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1); sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1); sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
        l_r0 += sum0;
        l_r1 += sum1;

        // ── O += P V (FP16) ──
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            const uint32_t a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
            // ldmatrix.trans: matrix j is (keys 16 ks + 8 (j & 1) .., dims 8 (nt + (j >> 1)) ..),
            // so the x4 holds b0 / b1 of n-tiles nt and nt + 1.
            const unsigned vrow = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), vcol = (lane >> 4) * 8;
#pragma unroll
            for (int np = 0; np < NT_O / 2; np++) {
              uint32_t vf[4];
              ldsm_x4_t(vf, vt[buf] + at(vrow, np * 16 + vcol));
#pragma unroll
              for (int h = 0; h < 2; h++) {
                const int nt = 2 * np + h;
                const unsigned b0 = vf[2 * h], b1 = vf[2 * h + 1];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                    : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]), "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc_o[nt][0]), "f"(acc_o[nt][1]), "f"(acc_o[nt][2]), "f"(acc_o[nt][3]));
              }
            }
        }
        }   // active
        if (next) {   // the next tile's V, converted before the barrier that publishes it
            if (TMA) {
                mbar_wait(&bars[buf ^ 1], ((kv_block + 1 - kv_block_lo) >> 1) & 1);
                v_to_f16_tma(vt[buf ^ 1], rows_of(kv_block + 1), tid);
            } else {
                cp_wait();
                v_to_f16(pad(vt[buf ^ 1]), tid);
            }
        }
    }

    if constexpr (CS > 1) {
        if (split) {
            split_combine<CS>(smem_kv, acc_o, m_r0, m_r1, l_r0, l_r1, rank, warp, lane, q_tile_len, [&](unsigned r) {
                const unsigned f = q_start + r;
                return O + q_batch_off + (unsigned long long)pos(f) * q_seq_stride + head_of(f) * head_dim;
            });
            return;
        }
    }
    if (!active) return;
    const float il0 = (l_r0 > 0) ? (1.f / l_r0) : 0, il1 = (l_r1 > 0) ? (1.f / l_r1) : 0;
    const unsigned f0 = q_start + row0, f1 = q_start + row1;
    const unsigned gr0 = pos(f0), gr1 = pos(f1);
    __nv_bfloat16* ob0 = O + q_batch_off + (unsigned long long)gr0 * q_seq_stride + head_of(f0) * head_dim;
    __nv_bfloat16* ob1 = O + q_batch_off + (unsigned long long)gr1 * q_seq_stride + head_of(f1) * head_dim;
#pragma unroll
    for (int nt = 0; nt < NT_O; nt++) {
        const unsigned c0 = nt * 8 + tq * 2;
        if (gr0 < q_len && row0 < q_tile_len && c0 < head_dim) {
            const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * il0));
            const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * il0));
            *reinterpret_cast<unsigned*>(&ob0[c0]) = lo | (hi << 16);
        }
        if (gr1 < q_len && row1 < q_tile_len && c0 < head_dim) {
            const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * il1));
            const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * il1));
            *reinterpret_cast<unsigned*>(&ob1[c0]) = lo | (hi << 16);
        }
    }
}

#define ROWS128_PARAMS                                                                              \
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K_cache,                 \
        const __nv_bfloat16* __restrict__ V_cache, __nv_bfloat16* __restrict__ O,                   \
        const int* const* __restrict__ block_table_ptrs, const unsigned int batch_size,             \
        const int* __restrict__ cu_seqlens, const int* __restrict__ kv_lens,                        \
        const int* __restrict__ state_end, const unsigned int q_len, unsigned int kv_len,           \
        unsigned int q_offset, const unsigned int num_q_heads, const unsigned int num_kv_heads,     \
        const unsigned int head_dim, const unsigned int cache_block_size,                           \
        const unsigned int sliding_window, const unsigned int causal_mask_enabled, const float inv_sqrt_d
#define ROWS128_ARGS                                                                                \
    Q, K_cache, V_cache, O, block_table_ptrs, batch_size, cu_seqlens, kv_lens, state_end, q_len,   \
        kv_len, q_offset, num_q_heads, num_kv_heads, head_dim, cache_block_size, sliding_window,    \
        causal_mask_enabled, inv_sqrt_d

extern "C" __global__ void __launch_bounds__(256, 1) prefill_paged_bf16_rows128(ROWS128_PARAMS) {
    rows128_body<false, false>(ROWS128_ARGS, nullptr, nullptr);
}

// K and V by TMA: `kmap` / `vmap` view a layer's pool as [blocks * 16 rows]
// x [num_kv_heads * 256] BF16 with 16 x 64 boxes and the 128-byte swizzle.
// Needs 16-row pages (the caller checks).
extern "C" __global__ void __launch_bounds__(256, 1)
prefill_paged_bf16_rows128_tma(ROWS128_PARAMS, const __grid_constant__ CUtensorMap kmap,
                               const __grid_constant__ CUtensorMap vmap) {
    rows128_body<true, false>(ROWS128_ARGS, &kmap, &vmap);
}

// GQA-packed entries (score passes), grid (num_kv_heads * CS, ceil(max_len
// * G / 128), batch): each CTA's 128 rows are one KV head's G query heads
// over ceil(128 / G) stream rows, sharing every K/V tile. Per row the
// unsplit entry's arithmetic is the per-head entry's (a taller row span only
// adds tiles fully masked for the rows that end earlier), so its output is
// bit-identical. The split entry (clusters of 2 along x, rank = blockIdx.x %
// 2) splits the streams `stream_splits` picks, within bf16 rounding of the
// unsplit entry, and computes every other stream exactly as that entry.
// Clusters of four were 5-10% faster for one long-pack request but 13%
// slower than unsplit with eight merged; two stay within 2% there.
extern "C" __global__ void __launch_bounds__(256, 1) prefill_paged_bf16_rows128_gqa(ROWS128_PARAMS) {
    rows128_body<false, true>(ROWS128_ARGS, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(256, 1)
prefill_paged_bf16_rows128_gqa_tma(ROWS128_PARAMS, const __grid_constant__ CUtensorMap kmap,
                                   const __grid_constant__ CUtensorMap vmap) {
    rows128_body<true, true>(ROWS128_ARGS, &kmap, &vmap);
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
prefill_paged_bf16_rows128_gqa_split2(ROWS128_PARAMS) {
    rows128_body<false, true, 2>(ROWS128_ARGS, nullptr, nullptr);
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1)
prefill_paged_bf16_rows128_gqa_split2_tma(ROWS128_PARAMS, const __grid_constant__ CUtensorMap kmap,
                                          const __grid_constant__ CUtensorMap vmap) {
    rows128_body<true, true, 2>(ROWS128_ARGS, &kmap, &vmap);
}
