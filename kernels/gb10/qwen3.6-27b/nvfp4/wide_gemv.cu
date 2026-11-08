// SPDX-License-Identifier: AGPL-3.0-only
//
// Wide batched GEMVs for the Qwen3.6/3.8-27B exact verify (ATLAS_WIDE_GEMV=1,
// `crates/spark-model/src/layers/ops/wide_gemv.rs`): 5..16 rows of one weight
// in ONE pass, each (row, output) bit for bit what the single-row kernel that
// serial decode runs computes on that row.
//
//   w4a16_gemv_wide{8,9,12,16}_nb{2,4}     row t == `w4a16_gemv`    on row t
//   w4a16_gemv_qg_wide{8,9,12,16}_nb{2,4}  row t == `w4a16_gemv_qg` on row t
//   w8a16_gemv_wide{8,9,12,16}_nb{2,4}     row t == `w8a16_gemv`    on row t
//
// (so also == `w4a16_gemv_batch4..16`, `w4a16_gemv_qg_batch2..4` and
// `w8a16_gemv_batch4/16`, which are gated against the same single-row kernels).
// The number is the row capacity (MAX_M, any M <= it is served), `nb` the
// outputs a thread: nb2 = 8 output groups (512 threads), nb4 = 4 groups (256
// threads), 16 outputs a CTA tile either way. Gate:
// `examples/wide_gemv_bitparity_microtest.rs`; bench: `examples/wide_gemv_bench.rs`.
//
// ════════════════════════════════════════════════════════════════════════════
// THE ARITHMETIC EACH (row, output) MUST REPRODUCE — read off the references
// (kernels/gb10/common/w4a16_gemv.cu, w8a16_gemv.cu). The whole GB10 tree builds
// with --fmad=false, so the references' `acc += a * w` is a separate multiply
// and add; this file spells every operation as an explicit intrinsic
// (fmaf / __fmul_rn / __fadd_rn), so it does not depend on that flag.
// BF16 -> FP32 is exact and is `bits << 16` (low half) / `bits & 0xFFFF0000`
// (high half) of the packed activation word.
//
// W4 (`w4a16_gemv`, its `w4a16_gemv_partial`): K16 = K/16 chunks of 16 values.
//   Reference lane l in 0..63 runs two chains c = 0, 1 over the chunks
//   kk = 2l + c + 128j in increasing j. A chunk's partial:
//     part = 0; for byte b = 0..7 of the 8 packed bytes (little-endian):
//       part = fmaf(a[2b],   E2M1[byte & 0xF], part)
//       part = fmaf(a[2b+1], E2M1[byte >> 4],  part)
//     acc_c = fmaf(e4m3(scale[kk]) * scale2, part, acc_c)       (scale2 mul.rn)
//   lane value v_l = acc_0 + acc_1. w0 = shfl_down tree 16/8/4/2/1 over
//   v_0..v_31, w1 the same over v_32..v_63; out = bf16_rn(w0 + w1).
//
// QG (`w4a16_gemv_qg`): K8 = K/8 chunks of 8 values. Lane l in 0..63 runs ONE
//   accumulator over kk = l + 64j, increasing j; per chunk, scale =
//   e4m3(scale[kk/2]) * scale2, and for byte b = 0..3 of the 4 packed bytes:
//     acc = acc + a[2b]   * (E2M1[byte & 0xF] * scale)
//     acc = acc + a[2b+1] * (E2M1[byte >> 4]  * scale)
//   w0 / w1 = shfl_down trees over lanes 0..31 / 32..63; out = bf16_rn(w0 + w1)
//   written at the deinterleaved [Q | G] column of n.
//
// W8 (`w8a16_gemv`): K16 chunks of 16 FP8 bytes. Lane l in 0..63 runs ONE
//   accumulator over kk = l + 64j; per chunk scale = block_scale[n/128,
//   kk*16/128] (FP32), and for element e = 0..15 in order:
//     acc = acc + a[e] * (E4M3_LUT[byte e] * scale)
//   then the same two trees and bf16_rn(w0 + w1).
//
// Tree identity used below: an xor butterfly (lane ^ 16, 8, 4, 2, 1) pairs
// exactly the operands the shfl_down tree pairs at lane 0, partners compute
// a + b and b + a (FP32 addition commutes), so every lane ends with the
// shfl_down tree's lane-0 value.
//
// ════════════════════════════════════════════════════════════════════════════
// WHAT CHANGES (launch shape only — no operation, operand or order above):
//
// 1. Register blocking over N. A 64-thread output group owns NB consecutive
//    outputs; thread p stands in for the SAME reference lane of all NB of
//    them. One activation load and its BF16 widening feeds NB weight rows,
//    one unpacked weight chunk feeds every row: per (row, output, chunk) the
//    W4 loop issues ~17 FMA + (16 widen + 2 loads) / NB instead of ~37.
//
// 2. Persistent CTAs over 16-output tiles (grid = min(tiles, 2 x SMs)) with
//    the next chunk's weights — across phase and tile boundaries — in flight
//    in registers while this chunk computes; the narrow tiers start every
//    CTA with an exposed DRAM round trip and keep no load in flight during
//    compute.
//
// 3. W4 keeps the narrow tiers' coalesced two-phase walk (their "virtual-lane
//    remap", w4a16_gemv.cu `w4a16_gemv_batchm_impl`): in phase φ thread p
//    walks kk = p + 64φ + 128j, i.e. reference lane p/2 + 32φ, chain p % 2,
//    chunks in increasing j. After the phase, the shfl_xor 1 fold gives lanes
//    p, p^1 that lane's v. Phase φ's 32 lanes ARE reference warp φ: its tree
//    level 16 pairs v_m (warp 0 of the group, lane 2m) with v_{m+16} (warp 1,
//    the same lane) — one 16-float shared-memory exchange per (row, output),
//    rows split between the two warps by parity — and levels 8/4/2/1 are
//    lane xor 16/8/4/2 in the owning warp. w0 waits in shared memory for w1.
//    QG and W8 already walk coalesced with lane = reference lane, so a warp
//    IS a reference warp: a 5-level butterfly, then w0 + w1 through shared
//    memory.
//
// Limits (host-checked, `wide_gemv.rs`): K a multiple of 16; K >= 2048 (W4:
// every thread owns a chunk in both phases), >= 512 (QG), >= 1024 (W8), so the
// prefetch chain never meets an empty walk. Rows A[t] at A + t*K, contiguous.
// Rows past M are neither computed nor written; tail outputs (n >= N) load
// zeros and are never written; every thread reaches every barrier.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "../../common/e4m3_lut.cuh"  // E4M3_LUT, numerically w8a16_gemv.cu's table

#define WG_WARP 32u
#define WG_TPG 64u          // threads an output group: the reference's two warps
#define WG_TILE 16u         // outputs a CTA tile (GROUPS x NB)
#define WG_FULL 0xFFFFFFFFu

__device__ __constant__ float WG_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float wg_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float wg_hi(unsigned w) { return __uint_as_float(w & 0xFFFF0000u); }

// The references' `(float)fp8 * scale2`.
__device__ __forceinline__ float wg_e4m3_scale(unsigned char b, float scale2) {
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = b;
    return __fmul_rn((float)fp8, scale2);
}

// Lane xor 16 .. LAST: the shfl_down tree levels 16 .. LAST, as lane 0 sees them.
template <unsigned LAST>
__device__ __forceinline__ float wg_butterfly(float x) {
#pragma unroll
    for (unsigned off = 16u; off >= LAST; off >>= 1) x = __fadd_rn(x, __shfl_xor_sync(WG_FULL, x, off));
    return x;
}

// ── W4: `w4a16_gemv` ────────────────────────────────────────────────────────

template <int NB>
struct WgW4Chunk {
    uint2 packed[NB];          // 16 E2M1 codes an output, bytes 0..3 in .x
    unsigned char scale[NB];   // the chunk's E4M3 group scale
};

// One chunk kk of NB outputs into acc[o][t] (the reference chain of thread p).
template <int MAX_M, int NB>
__device__ __forceinline__ void wg_w4_chunk(const __nv_bfloat16* __restrict__ A, const WgW4Chunk<NB>& w,
                                            unsigned kk, unsigned row_vec, const float* s_lut,
                                            float scale2, unsigned M, float (&acc)[NB][MAX_M]) {
    float wl[NB][16], sc[NB];
#pragma unroll
    for (int o = 0; o < NB; o++) {
        const unsigned pw[2] = {w.packed[o].x, w.packed[o].y};
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const unsigned byte_val = (pw[b >> 2] >> ((b & 3) * 8)) & 0xFFu;
            wl[o][2 * b] = s_lut[byte_val & 0xFu];
            wl[o][2 * b + 1] = s_lut[byte_val >> 4];
        }
        sc[o] = wg_e4m3_scale(w.scale[o], scale2);
    }
    const uint4* a = (const uint4*)A + kk * 2u;
#pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned)t >= M) continue;
        const uint4 lo = a[t * row_vec], hi = a[t * row_vec + 1u];
        const unsigned ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float part[NB];
#pragma unroll
        for (int o = 0; o < NB; o++) part[o] = 0.0f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            const float ax = wg_lo(ar[b]), ay = wg_hi(ar[b]);
#pragma unroll
            for (int o = 0; o < NB; o++) {
                part[o] = fmaf(ax, wl[o][2 * b], part[o]);
                part[o] = fmaf(ay, wl[o][2 * b + 1], part[o]);
            }
        }
#pragma unroll
        for (int o = 0; o < NB; o++) acc[o][t] = fmaf(sc[o], part[o], acc[o][t]);
    }
}

template <int MAX_M, int NB, int GROUPS>
__device__ __forceinline__ void wg_w4a16_rows(const __nv_bfloat16* __restrict__ A,
                                              const unsigned char* __restrict__ Bp,
                                              const unsigned char* __restrict__ Bs, const float scale2,
                                              __nv_bfloat16* __restrict__ C, unsigned M, const unsigned N,
                                              const unsigned K, const unsigned C_stride) {
    constexpr int V = NB * MAX_M;
    constexpr unsigned NPB = GROUPS * NB;
    static_assert(NPB == WG_TILE, "a CTA tile is 16 outputs (wide_gemv.rs TILE_OUTPUTS)");
    if (M > (unsigned)MAX_M) M = MAX_M;
    const unsigned g = threadIdx.x / WG_TPG, p = threadIdx.x % WG_TPG;
    const unsigned wig = p / WG_WARP, lane = threadIdx.x % WG_WARP;
    const unsigned half_K = K / 2u, K16 = K / 16u, row_vec = K / 8u;
    const unsigned tiles = (N + NPB - 1u) / NPB;

    __shared__ float s_lut[16];
    __shared__ float s_x[2][GROUPS][V][16];  // level-16 exchange, a buffer a phase
    __shared__ float s_w0[GROUPS][V];        // reference warp 0's result
    if (threadIdx.x < 16u) s_lut[threadIdx.x] = WG_E2M1_LUT[threadIdx.x];

    auto fetch = [&](WgW4Chunk<NB>& w, unsigned tile, unsigned kk) {
#pragma unroll
        for (int o = 0; o < NB; o++) {
            const unsigned n = tile * NPB + g * NB + o;
            w.packed[o] = make_uint2(0u, 0u);
            w.scale[o] = 0;
            if (tile < tiles && n < N && kk < K16) {
                w.packed[o] = *(const uint2*)(Bp + (unsigned long long)n * half_K + kk * 8u);
                w.scale[o] = Bs[(unsigned long long)n * K16 + kk];
            }
        }
    };

    WgW4Chunk<NB> cur, nxt;
    fetch(cur, blockIdx.x, p);
    __syncthreads();  // s_lut

    for (unsigned tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const unsigned n0 = tile * NPB + g * NB;
#pragma unroll 1
        for (unsigned phase = 0; phase < 2u; phase++) {
            float acc[NB][MAX_M];
#pragma unroll
            for (int o = 0; o < NB; o++)
#pragma unroll
                for (int t = 0; t < MAX_M; t++) acc[o][t] = 0.0f;
            for (unsigned kk = p + phase * WG_TPG; kk < K16; kk += 2u * WG_TPG) {
                // The next visit: this phase's next chunk, else phase 1's
                // first, else the next tile's first (phase 0).
                unsigned nk = kk + 2u * WG_TPG, nt = tile;
                if (nk >= K16) {
                    nk = phase == 0u ? p + WG_TPG : p;
                    nt = phase == 0u ? tile : tile + gridDim.x;
                }
                fetch(nxt, nt, nk);
                wg_w4_chunk<MAX_M, NB>(A, cur, kk, row_vec, s_lut, scale2, M, acc);
                cur = nxt;
            }
            // Fold chains 0/1 (lanes p, p^1) into reference lane p/2 + 32φ's
            // value; the non-owning warp hands its half of the level-16 pairs
            // to the owner (rows split by parity).
#pragma unroll
            for (int o = 0; o < NB; o++)
#pragma unroll
                for (int t = 0; t < MAX_M; t++) {
                    if ((unsigned)t >= M) continue;
                    acc[o][t] = __fadd_rn(acc[o][t], __shfl_xor_sync(WG_FULL, acc[o][t], 1));
                    if (wig != (unsigned)(t & 1) && (lane & 1u) == 0u)
                        s_x[phase][g][o * MAX_M + t][lane >> 1] = acc[o][t];
                }
            __syncthreads();
            // Reference warp φ's tree: level 16 through shared memory, then
            // levels 8/4/2/1 as lane xor 16/8/4/2. All lanes end with w_φ.
#pragma unroll
            for (int o = 0; o < NB; o++)
#pragma unroll
                for (int t = 0; t < MAX_M; t++) {
                    if ((unsigned)t >= M || wig != (unsigned)(t & 1)) continue;
                    const int i = o * MAX_M + t;
                    const float w = wg_butterfly<2u>(__fadd_rn(acc[o][t], s_x[phase][g][i][lane >> 1]));
                    if (lane == 0u) {
                        if (phase == 0u) {
                            s_w0[g][i] = w;  // read back by this same thread in phase 1
                        } else if (n0 + o < N) {
                            C[(unsigned long long)t * C_stride + n0 + o] =
                                __float2bfloat16(__fadd_rn(s_w0[g][i], w));
                        }
                    }
                }
        }
    }
}

// ── QG and W8: lane = reference lane, one walk ──────────────────────────────
//
// A policy supplies the weight chunk (fetch), its multiply-accumulate into
// acc[o][t] (mac) and the output column; `wg_direct_rows` is the persistent
// walk kk = p + 64j with the next chunk in flight, and the two-warp reduction.

template <int NB>
struct WgQg {
    struct Chunk {
        unsigned packed[NB];       // 8 E2M1 codes an output
        unsigned char scale[NB];
    };
    static constexpr unsigned LUT = 16;
    const unsigned char* __restrict__ Bp;
    const unsigned char* __restrict__ Bs;
    float scale2;
    unsigned half_K, groups, heads, head_dim;

    __device__ __forceinline__ static float lut_entry(unsigned i) { return WG_E2M1_LUT[i]; }
    __device__ __forceinline__ static unsigned chunks(unsigned K) { return K / 8u; }

    __device__ __forceinline__ void fetch(Chunk& w, unsigned n0, unsigned N, bool live, unsigned kk) const {
#pragma unroll
        for (int o = 0; o < NB; o++) {
            const unsigned n = n0 + o;
            w.packed[o] = 0u;
            w.scale[o] = 0;
            if (live && n < N) {
                w.packed[o] = *(const unsigned*)(Bp + (unsigned long long)n * half_K + kk * 4u);
                w.scale[o] = Bs[(unsigned long long)n * groups + kk / 2u];  // base_k / 16
            }
        }
    }

    template <int MAX_M>
    __device__ __forceinline__ void mac(const __nv_bfloat16* __restrict__ A, unsigned row_vec, const Chunk& w,
                                        unsigned kk, const float* s_lut, unsigned M,
                                        float (&acc)[NB][MAX_M]) const {
        float wv[NB][8];
#pragma unroll
        for (int o = 0; o < NB; o++) {
            const float scale = wg_e4m3_scale(w.scale[o], scale2);
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const unsigned byte_val = (w.packed[o] >> (b * 8)) & 0xFFu;
                wv[o][2 * b] = __fmul_rn(s_lut[byte_val & 0xFu], scale);
                wv[o][2 * b + 1] = __fmul_rn(s_lut[byte_val >> 4], scale);
            }
        }
        const uint4* a = (const uint4*)A + kk;
#pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned)t >= M) continue;
            const uint4 v = a[t * row_vec];
            const unsigned ar[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const float lo = wg_lo(ar[b]), hi = wg_hi(ar[b]);
#pragma unroll
                for (int o = 0; o < NB; o++) {
                    acc[o][t] = __fadd_rn(acc[o][t], __fmul_rn(lo, wv[o][2 * b]));
                    acc[o][t] = __fadd_rn(acc[o][t], __fmul_rn(hi, wv[o][2 * b + 1]));
                }
            }
        }
    }

    // Interleaved [Q_h(hd) | G_h(hd)] row n -> deinterleaved [Q | G] column.
    __device__ __forceinline__ unsigned col(unsigned n) const {
        const unsigned group_dim = 2u * head_dim, h = n / group_dim, idx = n % group_dim;
        return idx < head_dim ? h * head_dim + idx : heads * head_dim + h * head_dim + (idx - head_dim);
    }
};

template <int NB>
struct WgW8 {
    struct Chunk {
        uint4 b[NB];   // 16 E4M3 bytes an output
        float scale;   // the block scale: the NB outputs share a 128-row block
    };
    static constexpr unsigned LUT = 256;
    const unsigned char* __restrict__ B;
    const float* __restrict__ block_scale;
    unsigned K, k_blocks;

    __device__ __forceinline__ static float lut_entry(unsigned i) { return E4M3_LUT[i]; }
    __device__ __forceinline__ static unsigned chunks(unsigned K) { return K / 16u; }

    __device__ __forceinline__ void fetch(Chunk& w, unsigned n0, unsigned N, bool live, unsigned kk) const {
#pragma unroll
        for (int o = 0; o < NB; o++) {
            const unsigned n = n0 + o;
            w.b[o] = make_uint4(0u, 0u, 0u, 0u);
            if (live && n < N) w.b[o] = ((const uint4*)(B + (unsigned long long)n * K))[kk];
        }
        // n0 is a multiple of NB, which divides 128: n0 .. n0 + NB - 1 share n / 128.
        w.scale = live && n0 < N ? block_scale[(n0 / 128u) * k_blocks + kk * 16u / 128u] : 0.0f;
    }

    template <int MAX_M>
    __device__ __forceinline__ void mac(const __nv_bfloat16* __restrict__ A, unsigned row_vec, const Chunk& w,
                                        unsigned kk, const float* s_lut, unsigned M,
                                        float (&acc)[NB][MAX_M]) const {
        float wf[NB][16];
#pragma unroll
        for (int o = 0; o < NB; o++) {
            const unsigned br[4] = {w.b[o].x, w.b[o].y, w.b[o].z, w.b[o].w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) wf[o][4 * i + j] = __fmul_rn(s_lut[(br[i] >> (8 * j)) & 0xFFu], w.scale);
        }
        const uint4* a = (const uint4*)A + kk * 2u;
#pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned)t >= M) continue;
            const uint4 lo = a[t * row_vec], hi = a[t * row_vec + 1u];
            const unsigned ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float x = wg_lo(ar[j]), y = wg_hi(ar[j]);
#pragma unroll
                for (int o = 0; o < NB; o++) {
                    acc[o][t] = __fadd_rn(acc[o][t], __fmul_rn(x, wf[o][2 * j]));
                    acc[o][t] = __fadd_rn(acc[o][t], __fmul_rn(y, wf[o][2 * j + 1]));
                }
            }
        }
    }

    __device__ __forceinline__ unsigned col(unsigned n) const { return n; }
};

template <int MAX_M, int NB, int GROUPS, typename P>
__device__ __forceinline__ void wg_direct_rows(const P pol, const __nv_bfloat16* __restrict__ A,
                                               __nv_bfloat16* __restrict__ C, unsigned M, const unsigned N,
                                               const unsigned K, const unsigned C_stride) {
    constexpr int V = NB * MAX_M;
    constexpr unsigned NPB = GROUPS * NB;
    static_assert(NPB == WG_TILE, "a CTA tile is 16 outputs (wide_gemv.rs TILE_OUTPUTS)");
    static_assert(V <= (int)WG_TPG, "one finishing thread a (row, output)");
    if (M > (unsigned)MAX_M) M = MAX_M;
    const unsigned g = threadIdx.x / WG_TPG, p = threadIdx.x % WG_TPG;
    const unsigned wig = p / WG_WARP, lane = threadIdx.x % WG_WARP;
    const unsigned KC = P::chunks(K), row_vec = K / 8u;
    const unsigned tiles = (N + NPB - 1u) / NPB;

    __shared__ float s_lut[P::LUT];
    __shared__ float s_red[2][GROUPS][V][2];  // [tile parity][group][row, output][warp]
    for (unsigned i = threadIdx.x; i < P::LUT; i += blockDim.x) s_lut[i] = P::lut_entry(i);

    typename P::Chunk cur, nxt;
    pol.fetch(cur, blockIdx.x * NPB + g * NB, N, blockIdx.x < tiles && p < KC, p);
    __syncthreads();  // s_lut

    unsigned parity = 0;
    for (unsigned tile = blockIdx.x; tile < tiles; tile += gridDim.x, parity ^= 1u) {
        const unsigned n0 = tile * NPB + g * NB;
        float acc[NB][MAX_M];
#pragma unroll
        for (int o = 0; o < NB; o++)
#pragma unroll
            for (int t = 0; t < MAX_M; t++) acc[o][t] = 0.0f;
        for (unsigned kk = p; kk < KC; kk += WG_TPG) {
            // The next visit: this walk's next chunk, else the next tile's first.
            unsigned nk = kk + WG_TPG, nt = tile;
            if (nk >= KC) {
                nk = p;
                nt = tile + gridDim.x;
            }
            pol.fetch(nxt, nt * NPB + g * NB, N, nt < tiles && nk < KC, nk);
            pol.template mac<MAX_M>(A, row_vec, cur, kk, s_lut, M, acc);
            cur = nxt;
        }
        // This warp is reference warp `wig` of every output it holds.
#pragma unroll
        for (int o = 0; o < NB; o++)
#pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned)t >= M) continue;
                const float w = wg_butterfly<1u>(acc[o][t]);
                if (lane == 0u) s_red[parity][g][o * MAX_M + t][wig] = w;
            }
        __syncthreads();
        if (p < (unsigned)V) {
            const unsigned o = p / MAX_M, t = p % MAX_M;
            if (t < M && n0 + o < N)
                C[(unsigned long long)t * C_stride + pol.col(n0 + o)] =
                    __float2bfloat16(__fadd_rn(s_red[parity][g][p][0], s_red[parity][g][p][1]));
        }
    }
}

// ── Entry points (spelled out so a name grep finds them) ────────────────────

#define WG_W4_PARAMS                                                                      \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,      \
        const unsigned char *__restrict__ B_scale, const float scale2,                    \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K, unsigned C_stride
#define WG_W4_ARGS A, B_packed, B_scale, scale2, C, M, N, K, C_stride

extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_wide8_nb2(WG_W4_PARAMS) {
    wg_w4a16_rows<8, 2, 8>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_wide9_nb2(WG_W4_PARAMS) {
    wg_w4a16_rows<9, 2, 8>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_wide12_nb2(WG_W4_PARAMS) {
    wg_w4a16_rows<12, 2, 8>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_wide16_nb2(WG_W4_PARAMS) {
    wg_w4a16_rows<16, 2, 8>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_wide8_nb4(WG_W4_PARAMS) {
    wg_w4a16_rows<8, 4, 4>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_wide9_nb4(WG_W4_PARAMS) {
    wg_w4a16_rows<9, 4, 4>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_wide12_nb4(WG_W4_PARAMS) {
    wg_w4a16_rows<12, 4, 4>(WG_W4_ARGS);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_wide16_nb4(WG_W4_PARAMS) {
    wg_w4a16_rows<16, 4, 4>(WG_W4_ARGS);
}

// C row t at C + t * N, deinterleaved [Q | G] (as `w4a16_gemv_qg_batch4`).
#define WG_QG_PARAMS                                                                      \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,      \
        const unsigned char *__restrict__ B_scale, const float scale2,                    \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K,                \
        unsigned num_heads, unsigned head_dim
#define WG_QG_POL(NB) WgQg<NB>{B_packed, B_scale, scale2, K / 2u, K / 16u, num_heads, head_dim}

extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_qg_wide8_nb2(WG_QG_PARAMS) {
    wg_direct_rows<8, 2, 8>(WG_QG_POL(2), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_qg_wide9_nb2(WG_QG_PARAMS) {
    wg_direct_rows<9, 2, 8>(WG_QG_POL(2), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_qg_wide12_nb2(WG_QG_PARAMS) {
    wg_direct_rows<12, 2, 8>(WG_QG_POL(2), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(512, 1) w4a16_gemv_qg_wide16_nb2(WG_QG_PARAMS) {
    wg_direct_rows<16, 2, 8>(WG_QG_POL(2), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_qg_wide8_nb4(WG_QG_PARAMS) {
    wg_direct_rows<8, 4, 4>(WG_QG_POL(4), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_qg_wide9_nb4(WG_QG_PARAMS) {
    wg_direct_rows<9, 4, 4>(WG_QG_POL(4), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_qg_wide12_nb4(WG_QG_PARAMS) {
    wg_direct_rows<12, 4, 4>(WG_QG_POL(4), A, C, M, N, K, N);
}
extern "C" __global__ void __launch_bounds__(256, 1) w4a16_gemv_qg_wide16_nb4(WG_QG_PARAMS) {
    wg_direct_rows<16, 4, 4>(WG_QG_POL(4), A, C, M, N, K, N);
}

#define WG_W8_PARAMS                                                                      \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B,             \
        const float *__restrict__ block_scale, __nv_bfloat16 *__restrict__ C,             \
        unsigned M, unsigned N, unsigned K, unsigned C_stride
#define WG_W8_POL(NB) WgW8<NB>{B, block_scale, K, (K + 127u) / 128u}

extern "C" __global__ void __launch_bounds__(512, 1) w8a16_gemv_wide8_nb2(WG_W8_PARAMS) {
    wg_direct_rows<8, 2, 8>(WG_W8_POL(2), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(512, 1) w8a16_gemv_wide9_nb2(WG_W8_PARAMS) {
    wg_direct_rows<9, 2, 8>(WG_W8_POL(2), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(512, 1) w8a16_gemv_wide12_nb2(WG_W8_PARAMS) {
    wg_direct_rows<12, 2, 8>(WG_W8_POL(2), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(512, 1) w8a16_gemv_wide16_nb2(WG_W8_PARAMS) {
    wg_direct_rows<16, 2, 8>(WG_W8_POL(2), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(256, 1) w8a16_gemv_wide8_nb4(WG_W8_PARAMS) {
    wg_direct_rows<8, 4, 4>(WG_W8_POL(4), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(256, 1) w8a16_gemv_wide9_nb4(WG_W8_PARAMS) {
    wg_direct_rows<9, 4, 4>(WG_W8_POL(4), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(256, 1) w8a16_gemv_wide12_nb4(WG_W8_PARAMS) {
    wg_direct_rows<12, 4, 4>(WG_W8_POL(4), A, C, M, N, K, C_stride);
}
extern "C" __global__ void __launch_bounds__(256, 1) w8a16_gemv_wide16_nb4(WG_W8_PARAMS) {
    wg_direct_rows<16, 4, 4>(WG_W8_POL(4), A, C, M, N, K, C_stride);
}
