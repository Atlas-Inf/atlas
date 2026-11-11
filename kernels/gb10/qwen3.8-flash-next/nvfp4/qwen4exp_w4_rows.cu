// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) exact NVFP4 GEMV over 1..8 rows with a
// wide N: the MTP drafter's 100k-row NVFP4 draft head (ATLAS_QWEN4EXP_W4_ROWS=1,
// `layers/qwen4exp_draft_head.rs`), in place of `w4a16_gemv_batch{M}`.
//
//   qwen4exp_w4_rows{M}   row t byte-identical to `w4a16_gemv` on row t
//
// SAME ARITHMETIC as `w4a16_gemv` (`kernels/gb10/common/w4a16_gemv.cu`,
// `w4a16_gemv_partial`): reference lane l (0..63) of output n walks the
// 16-value chunks kk = 2l + c + 128j, c = 0, 1 into two accumulators; a
// chunk's partial is the fmaf chain over its 8 packed bytes, low nibble then
// high nibble, `part = fmaf(a, E2M1_LUT[nibble], part)`, folded in as
// `acc_c = fmaf((float)e4m3(scale) * scale2, part, acc_c)`; the lane's value
// is `acc0 + acc1`, lanes 0..31 and 32..63 each reduce by the shfl_down
// 16/8/4/2/1 tree, and the output is `lane0 + lane32` rounded once to BF16.
// (This directory builds with --fmad=false; every fmaf here is explicit, as
// in the reference.)
//
// WHAT CHANGES. `w4a16_gemv_batch8` at the draft head (50k rows a rank under
// ATLAS_QWEN4EXP_MTP_DRAFT_TP, M = 8) runs at ~126 GB/s: a CTA owns 4
// outputs, so every CTA re-reads the M activation rows from L2 (12.5k CTAs x
// 40 KB = 0.5 GB a call against 72 MB of weights), and the activation is
// widened once an output. Here a persistent CTA stages the M rows once, for
// the whole K, in shared memory (permuted so a lane's 16-byte reads are
// consecutive across the warp), then walks output tiles: a thread is
// reference lane l of NT outputs, so each staged activation it widens feeds
// NT outputs, and the next tile's weights are in flight while this one
// computes. The reduction is the reference tree done for all of a warp's
// values at once by recursive halving (`qw4_tree`; FP addition is
// commutative, so the pairs and the tree are the reference's).
//
// Layouts: A [M, K] BF16; B_packed [N, K/2] E2M1; B_scale [N, K/16] E4M3;
// scale2 FP32; C row t at C + t * N. K a multiple of 32, at most 4096 (the
// host refuses others). Grid: persistent (`ops::Qwen4ExpW4Rows`), block
// 64 x OB, dynamic shared memory (and no static) of
//   M x 2 x 4 KiB (the rows) + 2 x M x NPB x 2 x 4 B (reduction) + 64 B (LUT).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QW4_LANES 64

__device__ __constant__ float QW4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void qw4_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

// The reference tree of every one of a warp's V values: recursive halving
// while a lane holds more than one value, then plain pairwise adds. Returns
// slot 0; `idx` is the value index it holds.
template <int V>
__device__ __forceinline__ float qw4_tree(float (&v)[V], unsigned lane, unsigned& idx) {
    idx = 0;
#pragma unroll
    for (int lvl = 0; lvl < 5; lvl++) {
        const int o = 16 >> lvl, c = V >> (lvl + 1);
        const bool upper = lane & o;
        if (c >= 1) {
#pragma unroll
            for (int k = 0; k < c; k++) {
                const float send = upper ? v[k] : v[k + c];
                const float keep = upper ? v[k + c] : v[k];
                v[k] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, o);
            }
            if (upper) idx += c;
        } else {
            v[0] += __shfl_xor_sync(0xFFFFFFFFu, v[0], o);
        }
    }
    return v[0];
}

// One tile's weights a thread: chunks 2l, 2l + 1 of step j for NT outputs.
template <int NT, int J>
struct Qw4Tile {
    uint4 packed[J][NT];
    unsigned short scale[J][NT];
};

template <int M, int NT, int OB>
__device__ __forceinline__ void qw4_rows(const __nv_bfloat16* __restrict__ A,
                                         const unsigned char* __restrict__ Bp,
                                         const unsigned char* __restrict__ Bs, const float scale2,
                                         __nv_bfloat16* __restrict__ C, unsigned N, unsigned K) {
    constexpr int J = 2;  // ceil(K16 / 128) for K <= 4096
    constexpr unsigned NPB = NT * OB, NTH = QW4_LANES * OB;
    static_assert(OB <= 15, "a named barrier a group (ids 1..15)");
    // ALL shared memory is dynamic (`qw4_smem_bytes`): the launcher opts in
    // past 48 KB from the dynamic size alone, so a static remainder on top of
    // a dynamic size at or just under 48 KB (M = 6: 48 KB of rows) would
    // overflow the default limit without the opt-in and fail the launch.
    extern __shared__ __align__(16) uint4 s_a[];  // [M][J][c][half][64]
    float* s_red_base = (float*)(s_a + M * J * 4 * QW4_LANES);  // [2][M * NPB * 2]
    float* s_lut = s_red_base + 2 * M * NPB * 2;                 // [16]
    const unsigned lane = threadIdx.x % QW4_LANES, og = threadIdx.x / QW4_LANES;
    const unsigned K8 = K / 8, K16 = K / 16, half_K = K / 2;
    const unsigned tiles = (N + NPB - 1) / NPB;
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QW4_E2M1_LUT[threadIdx.x];

    // Stage the M rows once: granule g (16 bytes) of row t is half g % 2 of
    // chunk kk = g / 2 = 2l + c + 128j, landing at [t][j][c][half][l].
    for (unsigned i = threadIdx.x; i < M * K8; i += NTH) {
        const unsigned t = i / K8, g = i % K8, kk = g / 2, h = g % 2;
        const unsigned j = kk / 128, l = (kk % 128) / 2, c = kk % 2;
        qw4_cp16(&s_a[(((t * J + j) * 2 + c) * 2 + h) * QW4_LANES + l],
                 (const uint4*)(A + (size_t)t * K) + g);
    }
    asm volatile("cp.async.commit_group;");

    auto fetch = [&](Qw4Tile<NT, J>& w, unsigned tile) {
#pragma unroll
        for (int j = 0; j < J; j++)
#pragma unroll
            for (int o = 0; o < NT; o++) {
                const unsigned n = tile * NPB + og * NT + o, kk = j * 128 + 2 * lane;
                w.packed[j][o] = make_uint4(0, 0, 0, 0);
                w.scale[j][o] = 0;
                if (tile < tiles && n < N && kk < K16) {
                    w.packed[j][o] = *(const uint4*)(Bp + (size_t)n * half_K + kk * 8);
                    w.scale[j][o] = *(const unsigned short*)(Bs + (size_t)n * K16 + kk);
                }
            }
    };

    Qw4Tile<NT, J> cur, nxt;
    unsigned tile = blockIdx.x;
    fetch(cur, tile);
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    for (unsigned it = 0; tile < tiles; it++, tile += gridDim.x) {
        fetch(nxt, tile + gridDim.x);
        float acc[2][NT][M];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int o = 0; o < NT; o++)
#pragma unroll
                for (int t = 0; t < M; t++) acc[c][o][t] = 0.0f;
#pragma unroll
        for (int j = 0; j < J; j++) {
            if (j * 128 + 2 * lane >= K16) break;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float wl[NT][16], scale[NT];
#pragma unroll
                for (int o = 0; o < NT; o++) {
                    const uint4 p = cur.packed[j][o];
                    const unsigned pw[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                    for (int b = 0; b < 8; b++) {
                        const unsigned byte = (pw[2 * c + b / 4] >> ((b % 4) * 8)) & 0xFFu;
                        wl[o][2 * b] = s_lut[byte & 0xF];
                        wl[o][2 * b + 1] = s_lut[byte >> 4];
                    }
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = (unsigned char)(cur.scale[j][o] >> (8 * c));
                    scale[o] = (float)fp8 * scale2;
                }
#pragma unroll
                for (int t = 0; t < M; t++) {
                    const uint4* at = s_a + ((t * J + j) * 2 + c) * 2 * QW4_LANES + lane;
                    const uint4 lo = at[0], hi = at[QW4_LANES];
                    const unsigned ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                    float part[NT];
#pragma unroll
                    for (int o = 0; o < NT; o++) part[o] = 0.0f;
#pragma unroll
                    for (int b = 0; b < 8; b++) {
                        const float ax = __uint_as_float(ar[b] << 16);
                        const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
#pragma unroll
                        for (int o = 0; o < NT; o++) {
                            part[o] = fmaf(ax, wl[o][2 * b], part[o]);
                            part[o] = fmaf(ay, wl[o][2 * b + 1], part[o]);
                        }
                    }
#pragma unroll
                    for (int o = 0; o < NT; o++) acc[c][o][t] = fmaf(scale[o], part[o], acc[c][o][t]);
                }
            }
        }
        // The lane's value acc0 + acc1, then the tree a warp (padded to a power
        // of two with zeros that are never written), then lane0 + lane32.
        constexpr int V = NT * M, VP = V <= 1 ? 1 : V <= 2 ? 2 : V <= 4 ? 4 : V <= 8 ? 8 : V <= 16 ? 16 : 32;
        static_assert(V <= 32, "a warp's values fit the tree");
        float v[VP];
#pragma unroll
        for (int k = 0; k < VP; k++) v[k] = 0.0f;
#pragma unroll
        for (int o = 0; o < NT; o++)
#pragma unroll
            for (int t = 0; t < M; t++) v[o * M + t] = acc[0][o][t] + acc[1][o][t];
        float* red = s_red_base + (it & 1) * (M * NPB * 2);
        unsigned idx;
        const float r = qw4_tree<VP>(v, lane % 32, idx);
        // Lanes sharing idx hold the same value; each writes it.
        if (idx < (unsigned)V) red[((idx % M) * NPB + og * NT + idx / M) * 2 + lane / 32] = r;
        // Only this output group's two warps meet (named barrier og + 1), so
        // the groups of a CTA never wait on each other.
        asm volatile("bar.sync %0, %1;" ::"r"(og + 1), "r"(QW4_LANES) : "memory");
        for (unsigned i = lane; i < M * NT; i += QW4_LANES) {
            const unsigned t = i / NT, o = og * NT + i % NT, n = tile * NPB + o;
            const unsigned at = (t * NPB + o) * 2;
            if (n < N) C[(size_t)t * N + n] = __float2bfloat16(red[at] + red[at + 1]);
        }
        cur = nxt;
    }
}

#define QW4_PARAMS                                                                      \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,    \
        const unsigned char *__restrict__ B_scale, const float scale2,                  \
        __nv_bfloat16 *__restrict__ C, unsigned N, unsigned K
#define QW4_ARGS A, B_packed, B_scale, scale2, C, N, K

// Production entry points (rows 1..8), spelled out for the kernel-name check:
// 4 outputs a thread, 4 output groups (256 threads, 16 outputs a tile).
#define QW4_ENTRY(name, M)                                                                 \
    extern "C" __global__ void __launch_bounds__(QW4_LANES * 4, 1) name(QW4_PARAMS) {       \
        qw4_rows<M, 4, 4>(QW4_ARGS);                                                        \
    }
QW4_ENTRY(qwen4exp_w4_rows1, 1)
QW4_ENTRY(qwen4exp_w4_rows2, 2)
QW4_ENTRY(qwen4exp_w4_rows3, 3)
QW4_ENTRY(qwen4exp_w4_rows4, 4)
QW4_ENTRY(qwen4exp_w4_rows5, 5)
QW4_ENTRY(qwen4exp_w4_rows6, 6)
QW4_ENTRY(qwen4exp_w4_rows7, 7)
QW4_ENTRY(qwen4exp_w4_rows8, 8)

#ifdef QW4_SWEEP
// Bench-only shapes at M = 8: outputs a thread, output groups, min CTAs.
#define QW4_SWEEP_ENTRY(name, M, NT, OB, MINB)                                             \
    extern "C" __global__ void __launch_bounds__(QW4_LANES * OB, MINB) name(QW4_PARAMS) {   \
        qw4_rows<M, NT, OB>(QW4_ARGS);                                                      \
    }
QW4_SWEEP_ENTRY(qw4_m8_n1_o8, 8, 1, 8, 1)
QW4_SWEEP_ENTRY(qw4_m8_n2_o8, 8, 2, 8, 1)
QW4_SWEEP_ENTRY(qw4_m8_n2_o4, 8, 2, 4, 2)
#endif
