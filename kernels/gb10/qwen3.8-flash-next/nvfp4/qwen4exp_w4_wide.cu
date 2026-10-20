// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) exact NVFP4 GEMV over the rows of a batched
// verify (C8 x K=4: 24..36 rows): MoE router, attention K/V and o_proj
// (ATLAS_QWEN4EXP_W4_ROWS_WIDE=1, `layers/ops/qwen4exp_w4_wide.rs`), in place
// of 16-row `w4a16_gemv_batch16` launches.
//
//   qwen4exp_w4_rows_wide   row t byte-identical to `w4a16_gemv` on row t
//
// SAME ARITHMETIC as `w4a16_gemv` (`kernels/gb10/common/w4a16_gemv.cu`,
// `w4a16_gemv_partial`): reference lane l (0..63) of output n runs two chains,
// c = 0, 1, over the 16-value chunks kk = 2l + c + 128j in increasing j; a
// chunk's partial is the fmaf chain over its 8 packed bytes, low nibble then
// high, `part = fmaf(a, E2M1_LUT[nibble], part)`, folded in as
// `acc_c = fmaf((float)e4m3(scale) * scale2, part, acc_c)`; the lane's value
// is `acc0 + acc1`, lanes 0..31 and 32..63 each reduce by the shfl_down
// 16/8/4/2/1 tree, and the output is `lane0 + lane32` rounded once to BF16.
// (This directory builds with --fmad=false; every fmaf here is explicit.)
//
// WHAT CHANGES, 1: who runs which chain. The reference's chain p = 2l + c owns
// chunks p + 128j, so at K = 2560 (160 chunks) chains 0..31 run two chunks
// and 32..127 one: a 64-lane output group idles 37% of its slots. Here ONE
// warp owns an output and lane i walks chunks kk = i + 32q, q = 0, 1, ...:
// chunk kk belongs to chain p = kk % 128 = i + 32s with s = q % 4, so lane i
// holds four chains (slots s = 0..3, reference lane l = i/2 + 16s, c = i % 2)
// and visits each slot's chunks in increasing q, i.e. increasing j. Every
// lane runs ceil(K/512) chunks, balanced at any K.
//
// The reduction then IS the reference's: the slot's lane value acc0 + acc1
// sits on lanes i, i^1 (one shfl_xor 1; FP32 addition commutes); the warp-A
// tree's first level (x + 16) pairs slots 0 and 1 of the SAME lane, so it is
// an in-thread add, and its levels 8/4/2/1 are lane xor 16/8/4/2; warp B is
// slots 2 and 3 likewise; the output is A + B. The xor levels run for all of
// a warp's (output, row) values at once by recursive halving (`qw4w_tree`).
//
// WHAT CHANGES, 2: what is done once. A CTA owns a group of MT rows (grid y)
// and walks output tiles (grid x, persistent). Its rows are widened to FP32
// ONCE, for the whole K, into shared memory (the reference's own `bits << 16`
// / `bits & 0xFFFF0000`, permuted so a lane's four 16-byte reads are
// consecutive across the warp), so the chunk loop does no conversion; a
// decoded weight feeds MT rows and a staged activation NT outputs; the codes
// unpack in hardware (`qw4w_e2m1x2`, the same floats as the table); the next
// chunk's weights are in flight while this one computes and the next tile's
// are prefetched into L2. Rows past M are staged as zeros and never written.
//
// Layouts: A [M, K] BF16; B_packed [N, K/2] E2M1; B_scale [N, K/16] E4M3;
// scale2 FP32; C row t at C + t * N. K a multiple of 16. Grid (persistent x,
// ceil(M/MT)), block 32 x W, dynamic shared memory MT x ceil(K/512) x 2 KiB,
// the block's only shared memory (so the host's > 48 KB opt-in covers it all).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Two E2M1 codes (one packed byte, low nibble first) to FP32 through the
// hardware unpack (`cvt.rn.f16x2.e2m1x2`, then f16 -> f32): every E2M1 value,
// -0 included, is exact in f16 and f32, so .x/.y are bit for bit the
// reference's E2M1_LUT[byte & 0xF] / E2M1_LUT[byte >> 4].
__device__ __forceinline__ float2 qw4w_e2m1x2(unsigned byte) {
    unsigned h2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(h2) : "r"(byte));
    return __half22float2(*(const __half2*)&h2);
}

// The reference tree's levels 8/4/2/1 (lane xor 16/8/4/2) for all V of a
// warp's values: recursive halving while a lane holds more than one value.
// On return v[0 .. V/16) hold values idx .. idx + V/16 - 1 (lanes i and i^1
// alike).
template <int V>
__device__ __forceinline__ void qw4w_tree(float (&v)[V], unsigned lane, unsigned& idx) {
    static_assert(V >= 16 && (V & (V - 1)) == 0, "a power of two, at least 16");
    idx = 0;
#pragma unroll
    for (int lvl = 0; lvl < 4; lvl++) {
        const int o = 16 >> lvl, c = V >> (lvl + 1);
        const bool upper = lane & o;
#pragma unroll
        for (int k = 0; k < c; k++) {
            const float send = upper ? v[k] : v[k + c];
            const float keep = upper ? v[k + c] : v[k];
            v[k] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, o);
        }
        if (upper) idx += c;
    }
}

// One chunk's weights a lane: NT outputs' 8 packed bytes and E4M3 scale.
template <int NT>
struct Qw4wChunk {
    uint2 packed[NT];
    unsigned char scale[NT];
};

template <int NT, int W, int MT>
__device__ __forceinline__ void qw4w_rows(const __nv_bfloat16* __restrict__ A,
                                          const unsigned char* __restrict__ Bp,
                                          const unsigned char* __restrict__ Bs, const float scale2,
                                          __nv_bfloat16* __restrict__ C, unsigned M, unsigned N,
                                          unsigned K) {
    constexpr unsigned NPB = NT * W, NTH = 32 * W;
    constexpr int V = NT * MT, VP = V < 16 ? 16 : V;  // tree width
    static_assert((V & (V - 1)) == 0, "NT x MT a power of two");
    extern __shared__ __align__(16) float4 s_a[];  // [MT][Q][quarter][32]
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned K16 = K / 16, Q = (K16 + 31) / 32, half_K = K / 2;
    const unsigned row0 = blockIdx.y * MT;
    const unsigned tiles = (N + NPB - 1) / NPB;

    // A tile's weights into L2 ahead of their loads: this warp's NT outputs,
    // per chunk step q two 128-byte lines of packed codes and the 32 scales.
    auto prefetch_tile = [&](unsigned tile) {
        if (tile >= tiles) return;
        for (unsigned e = lane; e < NT * Q * 3; e += 32) {
            const unsigned o = e / (Q * 3), q = e % (Q * 3) / 3, part = e % 3;
            const unsigned n = tile * NPB + warp * NT + o, kk = q * 32 + (part % 2) * 16;
            if (n >= N || kk >= K16) continue;
            const unsigned char* p = part < 2 ? Bp + (size_t)n * half_K + kk * 8 : Bs + (size_t)n * K16 + kk;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
    };
    prefetch_tile(blockIdx.x);

    // Stage the MT rows as FP32: granule g (16 bytes, 8 values) of row t is
    // half h = g % 2 of chunk kk = g / 2 = i + 32q, landing at quarters 2h,
    // 2h + 1 of [t][q][.][i]. Rows past M and the chunk tail past K16 are
    // zeros (and never feed an output). SB granules a thread are in flight at
    // once (a dependent load per granule would serialize ~10 L2 round trips).
    constexpr unsigned SB = 4;
    for (unsigned e0 = threadIdx.x; e0 < MT * Q * 64; e0 += SB * NTH) {
        uint4 r[SB];
#pragma unroll
        for (unsigned b = 0; b < SB; b++) {
            const unsigned e = e0 + b * NTH, t = e / (Q * 64), g = e % (Q * 64);
            r[b] = e < MT * Q * 64 && row0 + t < M && g / 2 < K16
                       ? ((const uint4*)(A + (size_t)(row0 + t) * K))[g]
                       : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (unsigned b = 0; b < SB; b++) {
            const unsigned e = e0 + b * NTH, t = e / (Q * 64), g = e % (Q * 64), kk = g / 2, h = g % 2;
            if (e >= MT * Q * 64) break;
            float4* dst = s_a + ((t * Q + kk / 32) * 4 + 2 * h) * 32 + kk % 32;
            dst[0] = make_float4(__uint_as_float(r[b].x << 16), __uint_as_float(r[b].x & 0xFFFF0000u),
                                 __uint_as_float(r[b].y << 16), __uint_as_float(r[b].y & 0xFFFF0000u));
            dst[32] = make_float4(__uint_as_float(r[b].z << 16), __uint_as_float(r[b].z & 0xFFFF0000u),
                                  __uint_as_float(r[b].w << 16), __uint_as_float(r[b].w & 0xFFFF0000u));
        }
    }

    auto fetch = [&](Qw4wChunk<NT>& w, unsigned tile, unsigned q) {
        const unsigned kk = lane + 32 * q;
#pragma unroll
        for (int o = 0; o < NT; o++) {
            const unsigned n = tile * NPB + warp * NT + o;
            w.packed[o] = make_uint2(0, 0);
            w.scale[o] = 0;
            if (tile < tiles && n < N && kk < K16) {
                w.packed[o] = *(const uint2*)(Bp + (size_t)n * half_K + kk * 8);
                w.scale[o] = Bs[(size_t)n * K16 + kk];
            }
        }
    };

    Qw4wChunk<NT> cur, nxt;
    unsigned tile = blockIdx.x;
    fetch(cur, tile, 0);
    __syncthreads();

    for (; tile < tiles; tile += gridDim.x) {
        prefetch_tile(tile + gridDim.x);
        float ua[VP], ub[VP];
#pragma unroll
        for (int s = 0; s < 4; s++) {
            float acc[NT][MT];
#pragma unroll
            for (int o = 0; o < NT; o++)
#pragma unroll
                for (int t = 0; t < MT; t++) acc[o][t] = 0.0f;
            for (unsigned q = s; q < Q; q += 4) {
                // The next visit: this slot's next chunk, else the next
                // slot's first, else the next tile's chunk 0.
                unsigned nq = q + 4, nt = tile;
                if (nq >= Q) {
                    nq = s + 1;
                    if (nq >= 4 || nq >= Q) nq = 0, nt = tile + gridDim.x;
                }
                fetch(nxt, nt, nq);
                if (lane + 32 * q < K16) {  // the reference lane's `kk >= K16` break
                    float wl[NT][16], scale[NT];
#pragma unroll
                    for (int o = 0; o < NT; o++) {
                        const unsigned pw[2] = {cur.packed[o].x, cur.packed[o].y};
#pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const float2 w2 = qw4w_e2m1x2(pw[b / 4] >> ((b % 4) * 8));
                            wl[o][2 * b] = w2.x;
                            wl[o][2 * b + 1] = w2.y;
                        }
                        __nv_fp8_e4m3 fp8;
                        *(unsigned char*)&fp8 = cur.scale[o];
                        scale[o] = (float)fp8 * scale2;
                    }
#pragma unroll
                    for (int t = 0; t < MT; t++) {
                        const float4* at = s_a + (t * Q + q) * 128 + lane;
                        float a[16];
#pragma unroll
                        for (int v = 0; v < 4; v++) {
                            const float4 f = at[32 * v];
                            a[4 * v] = f.x, a[4 * v + 1] = f.y, a[4 * v + 2] = f.z, a[4 * v + 3] = f.w;
                        }
                        float part[NT];
#pragma unroll
                        for (int o = 0; o < NT; o++) part[o] = 0.0f;
#pragma unroll
                        for (int b = 0; b < 8; b++)
#pragma unroll
                            for (int o = 0; o < NT; o++) {
                                part[o] = fmaf(a[2 * b], wl[o][2 * b], part[o]);
                                part[o] = fmaf(a[2 * b + 1], wl[o][2 * b + 1], part[o]);
                            }
#pragma unroll
                        for (int o = 0; o < NT; o++) acc[o][t] = fmaf(scale[o], part[o], acc[o][t]);
                    }
                }
                cur = nxt;
            }
            // Slot s done: the reference lane value acc0 + acc1 (lanes i, i^1),
            // then the tree's x + 16 level in-thread (slot 0 + slot 1, slot 2
            // + slot 3).
#pragma unroll
            for (int o = 0; o < NT; o++)
#pragma unroll
                for (int t = 0; t < MT; t++) {
                    const float v = acc[o][t] + __shfl_xor_sync(0xFFFFFFFFu, acc[o][t], 1);
                    float& u = s < 2 ? ua[o * MT + t] : ub[o * MT + t];
                    u = s % 2 == 0 ? v : u + v;
                }
        }
#pragma unroll
        for (int k = V; k < VP; k++) ua[k] = ub[k] = 0.0f;
        unsigned ia, ib;
        qw4w_tree<VP>(ua, lane, ia);
        qw4w_tree<VP>(ub, lane, ib);
        // ia == ib: the same halving on the same lane. Lanes i, i^1 agree;
        // the even one writes lane0 + lane32.
        if ((lane & 1) == 0) {
#pragma unroll
            for (int k = 0; k < VP / 16; k++) {
                const unsigned id = ia + k, o = id / MT, t = id % MT;
                const unsigned n = tile * NPB + warp * NT + o;
                if (id < (unsigned)V && n < N && row0 + t < M)
                    C[(size_t)(row0 + t) * N + n] = __float2bfloat16(ua[k] + ub[k]);
            }
        }
    }
}

#define QW4W_PARAMS                                                                     \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,    \
        const unsigned char *__restrict__ B_scale, const float scale2,                  \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K
#define QW4W_ARGS A, B_packed, B_scale, scale2, C, M, N, K

// Production entry point: 4 outputs a warp, 8 warps (32 outputs a tile), 8
// rows a CTA, one CTA an SM (80 / 96 KB staged at K = 2560 / 3072).
extern "C" __global__ void __launch_bounds__(256, 1) qwen4exp_w4_rows_wide(QW4W_PARAMS) {
    qw4w_rows<4, 8, 8>(QW4W_ARGS);
}

#ifdef QW4W_SWEEP
// Bench-only shapes: outputs a warp, warps, rows a CTA, CTAs an SM.
#define QW4W_SWEEP_ENTRY(name, NT, W, MT, MINB)                                        \
    extern "C" __global__ void __launch_bounds__(32 * W, MINB) name(QW4W_PARAMS) {      \
        qw4w_rows<NT, W, MT>(QW4W_ARGS);                                                \
    }
QW4W_SWEEP_ENTRY(qw4w_n4_w12_m8, 4, 12, 8, 1)
QW4W_SWEEP_ENTRY(qw4w_n2_w16_m8, 2, 16, 8, 1)
QW4W_SWEEP_ENTRY(qw4w_n4_w16_m8, 4, 16, 8, 1)
QW4W_SWEEP_ENTRY(qw4w_n4_w4_m8, 4, 4, 8, 1)
QW4W_SWEEP_ENTRY(qw4w_n4_w4_m4, 4, 4, 4, 2)
QW4W_SWEEP_ENTRY(qw4w_n2_w8_m8, 2, 8, 8, 1)
#endif
