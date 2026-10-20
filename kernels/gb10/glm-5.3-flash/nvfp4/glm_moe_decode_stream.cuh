// SPDX-License-Identifier: AGPL-3.0-only

// Stream-loaded twins of the M16 verify-decode kernels
// (ATLAS_GLM_MOE_DECODE_STREAM). Included by moe_w4a16_grouped_gemm.cu after
// glm_moe_decode_m16.cuh, whose tiles, MMA stage and epilogue they call.
//
// The M16 kernels are bound by DRAM, and on GB10 a weight byte read through
// an L1-allocating load (cp.async.ca, or a plain ld) costs about 8% more
// DRAM time than one read with a streaming load: the M16 cp.async pipeline
// streams the routed experts' tables at 203-220 GB/s, the same tables read
// in 64 KB chunks with streaming loads at 245-255 GB/s, and the M16 kernel
// with its cp.async replaced by plain loads is no faster than M16, with
// __ldcs (or ld.global.nc.L1::no_allocate) 7-16% faster
// (scripts/moe-decode-bench). So here the block's 256 threads read the next
// K stage with streaming loads into registers, all issued before any store,
// and store them where pqd_issue's cp.async would have left them (rows past
// the expert's, and ZSKIP's dead weight rows, as zeros), then run
// pqd_mma_stage on the stage that landed. Two stage buffers (40 KB) in place
// of M16's four, and two blocks per SM, which __launch_bounds__ holds (one
// block per SM, one stage in flight, is slower).
//
// Shared memory holds the same bytes before each MMA, and the MMAs and the
// epilogue are the M16 functions, so the outputs are those of the M16
// kernels bit for bit (moe_decode_bench gates every variant against
// production).
//
// MI row slabs of 16: MI = 2 takes up to 32 rows per expert (owner batches
// of two to four streams, which M16 leaves to the M64 K128W grid), each slab
// its own pqd_mma_stage over the same weight tile.
//
// Grid and arguments as the M16 kernels; block (PQS_THREADS, 1, 1).
//
// L2 prefetch (ATLAS_GLM_MOE_DECODE_L2PF, the *_l2pf twins). A CTA's tile
// is 128 columns of the gate table and the same of the up table (down: 256
// columns), so each K stage it reads 64 rows x 128 B of each at an N byte
// stride (1 KB at the rank's I/2 = 1024; down 256 B of 4 KB): its own reads
// never touch a whole row. The m16s gate/up reads its tables at about
// 225 GB/s on ennspark03 (4 rows, 22 experts: 104 MB in 459-465 us) where a
// pass over the same bytes in 64 KB chunks reaches 244-250 (commit a74552a4's
// moe_decode_bench runs); in serving (2026-10-03 nsys) both kernels run at
// about 210-220 GB/s if the expert counts are the bench's. The grid's
// gridDim.x CTAs of an expert together read every column of the stage's
// rows, which are one contiguous block of the table (64 N bytes, and 8 N of
// scales). So each CTA also asks for its 1/gridDim.x of that block by rows,
// whole rows, PQS_PF_DIST stages ahead of the stage it loads
// (prefetch.global.L2 per 128-byte line; ZSKIP's dead rows are not asked
// for): DRAM then serves an expert's stage as a few contiguous runs, and the
// CTAs' strided streaming loads find the bytes in L2. Only immutable weights are read, after the
// PDL wait, and no prefetch writes anything: every load, store, MMA and the
// epilogue are m16s's, so the outputs are the stream kernels' bit for bit.
// A CTA that lags its expert's siblings finds its slice in L2 already, and
// its own prefetch of rows they have read hits L2 while the lines live, so
// each byte still crosses DRAM about once.
//
// Persistent twins (ATLAS_GLM_MOE_DECODE_PERSIST, the *_l2pf_p kernels: the
// prefetching gate/up and zskip down). DRAM serves a read of few tables at
// once faster than one of many: on ennspark03 a sweep of 2 MB tables in 8 KB
// chunks reads 248-253 GB/s one table at a time, 240 two at a time and
// 230-234 with 12-48 (scripts/moe-decode-bench rooflines, 2026-10-09). The
// grid twins hold two CTAs per SM, 12 experts' gate and up tables at once
// (6 for the down); one CTA per SM halves that, and a persistent CTA strides
// over the worklist's (expert, column tile) items, expert-major, asking its
// next item's first stages (and building its zskip mask) during the last
// ones, so an item's start is hidden. Measured against the grid twins
// (moe_decode_bench, 3-32 rows): gate/up 1.5-2% and down 2.5-4.5% faster.
// Each item is computed exactly as by the grid kernels, so the same bytes,
// and with any CTA count (the GPU test runs 48, 7 and 1).
//
// Prior art (docs/glm-prior-art.md): pulling the weights the next kernels
// read into L2 is jayleaton's L2 prefetch
// (github.com/jayleaton/glm53-tensorfold-spark patches/0460, Apache-2.0), as
// carried in Mia's TensorFold recipe patch 0046-glm-l2-prefetch (Apache-2.0),
// which prefetches dense weights from a side stream during the all-gathers;
// TensorFold 0047's weight loads issued a step ahead are the same family.
// Ours runs inside the routed expert kernels, once the experts are known, and
// splits each stage's rows among an expert's CTAs. No code copied.
#pragma once

#define PQS_THREADS 256
// 16-byte pieces of a stage per thread: 1024 weight, 128 scale and up to 128
// activation pieces.
#define PQS_PIECES 5
// Stages ahead of the one being loaded that the L2 prefetch asks for (a stage
// is about 7 us of a CTA's time at two CTAs per SM and DRAM speed).
#ifndef PQS_PF_DIST
#define PQS_PF_DIST 2
#endif

// Thread t's pieces of K stage kb: the shared bytes of pqd_issue.
template<bool GATE_UP, int MI>
__device__ __forceinline__ void pqs_load(
    unsigned int t, unsigned char (*sA)[PQ2_AP], unsigned char (*sAs)[PQ2_KS / GROUP_SIZE],
    PqwB& sB, PqwS& sS, const int* sTok,
    const unsigned char* __restrict__ A_packed, const unsigned char* __restrict__ A_scale,
    const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    unsigned int M_eff, unsigned int cta_n, unsigned int N, unsigned int K, unsigned int kb,
    const unsigned char* live
) {
    constexpr unsigned int NC = PQW_NT / 16;                      // 16-byte chunks per row
    constexpr unsigned int NB = PQ2_KP * NC;                      // weight pieces
    constexpr unsigned int NS = (PQ2_KS / GROUP_SIZE) * NC;       // block-scale pieces
    constexpr unsigned int NA = PQD_M * MI * (PQ2_KP / 16);       // activation pieces
    static_assert(NB + NS + NA <= PQS_PIECES * PQS_THREADS, "stage pieces exceed the block");
    const uint4 zero = make_uint4(0, 0, 0, 0);
    uint4 v[PQS_PIECES];
    uint4* dst[PQS_PIECES];
    #pragma unroll
    for (int r = 0; r < PQS_PIECES; ++r) {
        const unsigned int p = t + r * PQS_THREADS;
        v[r] = zero;
        dst[r] = nullptr;
        if (p < NB) {
            const unsigned int kp = p / NC, c = p % NC;
            dst[r] = (uint4*)&sB[kp][(c ^ (kp & 7)) << 4];
            if (!live || live[kb / 2 + kp])
                v[r] = __ldcs((const uint4*)&(pqw_chunk_up<GATE_UP>(c) ? U_expert : B_expert)[
                    (unsigned long long)(kb / 2 + kp) * N + pqw_chunk_col<GATE_UP>(cta_n, c)]);
        } else if (p < NB + NS) {
            const unsigned int g = (p - NB) / NC, c = (p - NB) % NC;
            dst[r] = (uint4*)&sS[g][(c ^ g) << 4];
            v[r] = __ldcs((const uint4*)&(pqw_chunk_up<GATE_UP>(c) ? US_expert : S_expert)[
                (unsigned long long)(kb / GROUP_SIZE + g) * N + pqw_chunk_col<GATE_UP>(cta_n, c)]);
        } else if (p < NB + NS + NA) {
            const unsigned int row = (p - NB - NS) >> 2, col = ((p - NB - NS) & 3) << 4;
            dst[r] = (uint4*)&sA[row][col];
            if (row < M_eff)
                v[r] = *(const uint4*)&A_packed[(unsigned long long)(unsigned int)sTok[row] * (K / 2) + kb / 2 + col];
        }
    }
    #pragma unroll
    for (int r = 0; r < PQS_PIECES; ++r)
        if (dst[r]) *dst[r] = v[r];
    if (t < PQD_M * MI)
        *(unsigned long long*)&sAs[t][0] = t < M_eff
            ? *(const unsigned long long*)&A_scale[(unsigned long long)(unsigned int)sTok[t] * (K / GROUP_SIZE) + kb / GROUP_SIZE]
            : 0ull;
}

// This CTA's share of K stage kb of the expert's weight and scale tables,
// asked into L2: of the stage's PQ2_KP packed rows and PQ2_KS / GROUP_SIZE
// scale rows, the ones in this CTA's 1/gridDim.x by blockIdx.x, each whole
// (every column of the grid), less the dead weight rows of `live`; `tile` is
// the CTA's column tile of the expert's `ctas`.
// Consecutive threads take consecutive lines of a row.
template<bool GATE_UP>
__device__ __forceinline__ void pqs_prefetch(
    unsigned int t, const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    unsigned int N, unsigned int kb, const unsigned char* live, unsigned int tile, unsigned int ctas
) {
    constexpr unsigned int TABLES = GATE_UP ? 2 : 1;
    constexpr unsigned int GROUPS = PQ2_KS / GROUP_SIZE;
    const unsigned int lines = N / 128;
    const unsigned int rows = (PQ2_KP + ctas - 1) / ctas, groups = (GROUPS + ctas - 1) / ctas;
    const unsigned int nw = TABLES * rows * lines, total = nw + TABLES * groups * lines;
    for (unsigned int i = t; i < total; i += PQS_THREADS) {
        const bool w = i < nw;
        const unsigned int k = w ? i : i - nw, span = w ? rows : groups;
        const unsigned int tab = k / (span * lines), r = tile * span + k / lines % span;
        if (r >= (w ? PQ2_KP : GROUPS) || (w && live && !live[kb / 2 + r])) continue;
        const unsigned char* base = w ? (tab ? U_expert : B_expert) : (tab ? US_expert : S_expert);
        const unsigned char* p = base + (unsigned long long)(w ? kb / 2 + r : kb / GROUP_SIZE + r) * N
            + (k % lines) * 128;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
    }
}

// The routed tables of work item (`slot`, `tile`): the worklist's slot-th
// local expert and its tile-th column tile. `ok` is false for an item no CTA
// computes (a padding slot, a second M64 row tile, an expert with no rows or
// no weights); `full` for an expert with more rows than the slabs.
struct PqsItem {
    unsigned int expert_id, cta_m;
    int M_expert;
    const unsigned char *B, *S, *U, *US;
    bool ok, full;
};

template<bool GATE_UP, int MI>
__device__ __forceinline__ PqsItem pqs_item(
    const unsigned int* __restrict__ worklist, const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_packed_ptrs, const unsigned long long* __restrict__ B_scale_ptrs,
    const PqwGateUp& up, unsigned int num_experts, unsigned int slot
) {
    PqsItem it{};
    it.expert_id = worklist[4 + slot * 2];
    // One row tile per expert: a second M64 tile means more than 64 rows.
    if (it.expert_id >= num_experts || worklist[5 + slot * 2] != 0) return it;
    it.cta_m = expert_offsets[it.expert_id];
    it.M_expert = expert_offsets[it.expert_id + 1] - (int)it.cta_m;
    it.B = (const unsigned char*)B_packed_ptrs[it.expert_id];
    it.S = (const unsigned char*)B_scale_ptrs[it.expert_id];
    if (it.M_expert <= 0 || it.B == 0) return it;
    it.full = it.M_expert > PQD_M * MI;
    it.ok = true;
    if (GATE_UP) {
        it.U = (const unsigned char*)up.packed_ptrs[it.expert_id];
        it.US = (const unsigned char*)up.scale_ptrs[it.expert_id];
    }
    return it;
}

// NBUF stage buffers (2; PERSIST takes 3, whose 60 KB holds one CTA per SM).
// PERSIST: a static stride over the work items, `gridDim.x` CTAs (one per SM)
// taking items blockIdx.x, + gridDim.x, ... of the worklist's experts x
// column tiles, expert-major; while it runs a stage loop's last PQS_PF_DIST + 1
// stages, a CTA asks its next item's first stages into L2 (PF only). Each
// item's loads, MMAs and epilogue are the grid kernel's, so the same bytes.
template<bool GATE_UP, bool ZSKIP, int MI, bool PF = false, bool PERSIST = false>
__device__ __forceinline__ void pqs_impl(
    PQ2_ARGS,
    const unsigned int* __restrict__ worklist,
    const PqwGateUp up
) {
    constexpr int NBUF = PERSIST ? 3 : 2;
    atlas_pdl_enter();
    const unsigned int tiles = PERSIST ? N / (GATE_UP ? PQW_NT / 2 : PQW_NT) : gridDim.x;
    const unsigned int items = PERSIST ? worklist[0] * tiles : 0;
    if (!PERSIST && (int)blockIdx.y >= (int)worklist[0]) return;
    unsigned int item = PERSIST ? blockIdx.x : blockIdx.y * tiles + blockIdx.x;
    if (PERSIST && item >= items) return;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) unsigned char sA[NBUF][PQD_M * MI][PQ2_AP];
    __shared__ __align__(16) unsigned char sAs[NBUF][PQD_M * MI][PQ2_KS / GROUP_SIZE];
    __shared__ __align__(16) PqwB sB[NBUF];
    __shared__ __align__(16) PqwS sS[NBUF];
    __shared__ int sTok[PQD_M * MI];
    // ZSKIP: the M16 zskip kernel's mask of the weight rows to load, of this
    // item and (PERSIST) of the next one, which its prefetch skips too.
    __shared__ unsigned char sLive[PERSIST ? 2 : 1][ZSKIP ? PQD_ZSKIP_KP : 1];
    const bool zskip = ZSKIP && K / 2 <= PQD_ZSKIP_KP;
    const unsigned int stages = K / PQ2_KS;
    auto build_mask = [&](unsigned char* live, const PqsItem& m) {
        for (unsigned int kp = t; kp < K / 2; kp += PQS_THREADS) {
            unsigned int any = 0;
            for (int r = 0; r < m.M_expert; ++r) {
                const unsigned int tok = sorted_token_ids ? (unsigned int)sorted_token_ids[m.cta_m + r] : m.cta_m + r;
                any |= A_packed[(unsigned long long)tok * (K / 2) + kp] & 0x77u;
            }
            live[kp] = any != 0;
        }
    };
    unsigned int cur = 0;   // sLive buffer of this item

    PqsItem it = pqs_item<GATE_UP, MI>(worklist, expert_offsets, B_packed_ptrs, B_scale_ptrs, up, num_experts,
        item / tiles);
    // Whether this item's first stages were asked for, and its mask built,
    // during the last one.
    bool asked = false;
    for (;;) {
        const unsigned int tile = item % tiles;
        const unsigned int next = item + (PERSIST ? gridDim.x : 0);
        const bool has_next = PERSIST && next < items;
        const PqsItem nx = has_next
            ? pqs_item<GATE_UP, MI>(worklist, expert_offsets, B_packed_ptrs, B_scale_ptrs, up, num_experts, next / tiles)
            : PqsItem{};
        if (it.ok && it.full) {
            // More rows than the slabs the host selects these kernels for: NaN
            // over them from the downs, as the M16 downs.
            if constexpr (!GATE_UP) {
                const __nv_bfloat16 nan = __float2bfloat16(__int_as_float(0x7fc00000));
                for (int r = 0; r < it.M_expert; ++r)
                    C[(unsigned long long)(it.cta_m + r) * N + tile * PQW_NT + t] = nan;
            }
        } else if (it.ok) {
            const float scale2 = scale2_vals[it.expert_id];
            const unsigned int cta_n = tile * (GATE_UP ? PQW_NT / 2 : PQW_NT);

            if (t < PQD_M * MI)
                sTok[t] = (sorted_token_ids && (int)t < it.M_expert) ? sorted_token_ids[it.cta_m + t] : (int)(it.cta_m + t);
            const bool pf_next = PF && has_next && nx.ok && !nx.full;
            if (zskip && !asked) build_mask(sLive[cur], it);
            if (zskip && pf_next) build_mask(sLive[cur ^ 1], nx);
            __syncthreads();
            const unsigned char* live = zskip ? sLive[cur] : nullptr;

            auto load = [&](int buf, unsigned int kb) {
                pqs_load<GATE_UP, MI>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
                    it.B, it.S, it.U, it.US, (unsigned int)it.M_expert, cta_n, N, K, kb, live);
            };

            PqdAcc acc[MI];
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
                #pragma unroll
                for (int i = 0; i < 4; i++) acc[mi][i][0] = acc[mi][i][1] = acc[mi][i][2] = acc[mi][i][3] = 0.0f;

            // L2 prefetch of stage s (PF): asked for PQS_PF_DIST stages before it
            // is loaded, the first ones before stage 0's loads; past the last
            // stage, the next item's.
            auto prefetch = [&](unsigned int s) {
                if constexpr (PF) {
                    if (s < stages)
                        pqs_prefetch<GATE_UP>(t, it.B, it.S, it.U, it.US, N, s * PQ2_KS, live, tile, tiles);
                    else if (pf_next && s - stages <= PQS_PF_DIST)
                        pqs_prefetch<GATE_UP>(t, nx.B, nx.S, nx.U, nx.US, N, (s - stages) * PQ2_KS,
                            zskip ? sLive[cur ^ 1] : nullptr, next % tiles, tiles);
                }
            };
            if (!asked)
                for (unsigned int s = 0; s <= PQS_PF_DIST; ++s) prefetch(s);
            load(0, 0);
            __syncthreads();
            for (unsigned int st = 0; st < stages; ++st) {
                const int buf = st % NBUF;
                // Stage st is in buffer buf; the one stage st + 1 loads into
                // ended its MMAs before the last barrier.
                prefetch(st + 1 + PQS_PF_DIST);
                if (st + 1 < stages) load((st + 1) % NBUF, (st + 1) * PQ2_KS);
                #pragma unroll
                for (int mi = 0; mi < MI; mi++)
                    pqd_mma_stage<GATE_UP>(acc[mi], *(const PqdA*)&sA[buf][mi * PQD_M], *(const PqdAs*)&sAs[buf][mi * PQD_M],
                        sB[buf], sS[buf], warp_id, lane_id);
                __syncthreads();
            }
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
                pqd_epilogue<GATE_UP>(acc[mi], warp_id, lane_id, it.expert_id, scale2, it.cta_m + mi * PQD_M,
                    it.M_expert - mi * PQD_M, cta_n, N, C, up);
        }
        if (!has_next) return;
        // The loop above asked for the next item's first stages and built its
        // mask in the other buffer.
        asked = PF && it.ok && !it.full && nx.ok && !nx.full && stages > 0;
        if (asked) cur ^= 1;
        item = next;
        it = nx;
    }
}

// Down, zero-skipping down and fused gate/up over one row slab (up to 16 rows
// per expert) and over two (up to 32): arguments as the M16 kernels.
#define PQS_DOWN_ARGS PQ2_ARGS, const unsigned int* __restrict__ worklist
#define PQS_GATE_UP_ARGS \
    PQ2_ARGS, const unsigned int* __restrict__ worklist, \
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs, \
    const float* __restrict__ up_scale2_vals, unsigned char* __restrict__ out_packed, unsigned char* __restrict__ out_scale
#define PQS_TABLES A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, \
    expert_offsets, sorted_token_ids, num_experts, N, K, worklist
#define PQS_UP PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale}

extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_k128w(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 1>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_k128w_zskip(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 1>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_gate_up_silu_k128w(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 1>(PQS_TABLES, PQS_UP);
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_k128w(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 2>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_k128w_zskip(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 2>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_gate_up_silu_k128w(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 2>(PQS_TABLES, PQS_UP);
}
// The same six with the L2 prefetch of each stage's rows
// (ATLAS_GLM_MOE_DECODE_L2PF).
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_k128w_l2pf(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 1, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_k128w_zskip_l2pf(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 1, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m16s_gate_up_silu_k128w_l2pf(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 1, true>(PQS_TABLES, PQS_UP);
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_k128w_l2pf(PQS_DOWN_ARGS) {
    pqs_impl<false, false, 2, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_k128w_zskip_l2pf(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 2, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 2) glm_moe_decode_m32s_gate_up_silu_k128w_l2pf(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 2, true>(PQS_TABLES, PQS_UP);
}

// The prefetching twins as persistent kernels, one CTA per SM
// (ATLAS_GLM_MOE_DECODE_PERSIST): grid (SMs, 1, 1), arguments as above.
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 1) glm_moe_decode_m16s_k128w_zskip_l2pf_p(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 1, true, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 1) glm_moe_decode_m16s_gate_up_silu_k128w_l2pf_p(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 1, true, true>(PQS_TABLES, PQS_UP);
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 1) glm_moe_decode_m32s_k128w_zskip_l2pf_p(PQS_DOWN_ARGS) {
    pqs_impl<false, true, 2, true, true>(PQS_TABLES, PqwGateUp{});
}
extern "C" __global__ void __launch_bounds__(PQS_THREADS, 1) glm_moe_decode_m32s_gate_up_silu_k128w_l2pf_p(PQS_GATE_UP_ARGS) {
    pqs_impl<true, false, 2, true, true>(PQS_TABLES, PQS_UP);
}

#undef PQS_DOWN_ARGS
#undef PQS_GATE_UP_ARGS
#undef PQS_TABLES
#undef PQS_UP
