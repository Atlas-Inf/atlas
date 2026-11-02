// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) exact BF16 GEMV over 17..32 rows in one
// weight pass, register-tiled (ATLAS_QWEN4EXP_ROWS32_TILE=1,
// `layers/ops/qwen4exp_wide_rows.rs`): the batched verify's GDN qkvz /
// out_proj and the LM head, in place of the pair tier
// `qwen4exp_bf16_rows32` (`qwen4exp_wide_rows.cu`).
//
//   qwen4exp_bf16_rows32t   row t byte-identical to `dense_gemv_bf16` on row t
//
// SAME ARITHMETIC. Every (row t, output n) is the single-row kernel's: 64
// "virtual lanes" l, lane l walking the 8-element vectors kv = l, l + 64, ...
// < K/8 in order, each vector's elements in memory order, then lanes 0..31
// and 32..63 each reduced by the shfl_down 16/8/4/2/1 tree and the output
// `lane0 + lane32` rounded once to BF16. Rows and outputs only add
// independent accumulators, so a row's bits depend neither on M nor on its
// neighbours (scripts/dev/qwen4exp_rows32_tile_bench.cu checks M = 1..32).
//
// WHAT CHANGES is the tiling. The pair tier gives a thread one lane of 2
// outputs x 32 rows; each activation vector it reads from shared memory
// feeds 2 outputs, and the kernel is bound by shared-memory issue and
// latency (ncu, LM head half at M = 32: 4 warps a scheduler, 0.59 issued a
// cycle, mio_throttle + short_scoreboard the top stalls). Here a thread
// owns one lane of NT outputs x MT rows: an activation vector feeds NT
// outputs, a weight vector (widened to FP32 once a K-step) MT rows. A K-step
// of every row (M x 1 KiB) is staged with cp.async, double-buffered, so the
// next step streams while this one computes.
//
// FUSED MULTIPLY-ADD, same bits. A BF16 x BF16 product has at most 16
// significant bits, so it is exactly representable in FP32 whenever it is
// zero or a normal number below 2^128. Then the reference's rounded product
// RN(a * w) is a * w, and its rounded sum RN(acc + RN(a * w)) equals the
// FMA's single rounding RN(acc + a * w) for every acc (inf and NaN
// included). A step fuses only where every operand is provably in that
// regime: each activation is non-zero with magnitude in [2^-63, 2^65), each
// weight in [2^-63, 2) (`QtActWindow`, `QtWeightWindow`), so every product
// lies in [2^-126, 2^66).
// The CTA checks the activation vectors it staged (`__syncthreads_or`), a
// thread its own weight vectors. Anything else -- +-0, subnormals, tiny or
// huge values, inf, NaN -- runs that step with the separate multiply and add
// this directory's --fmad=false gives `acc += a * w`: the reference itself.
// Real activations and weights sit in the window, so the fallback is a
// guard, not a path that runs.
//
// Layouts: A [M, K] BF16 rows K apart; B [N, K] BF16; C row t at
// C + t * out_stride. K a multiple of 8 (the host refuses others).
// Grid (ceil(N / NPB)), block 64 x RB x OB threads, dynamic shared memory
// 2 x 32 x 64 x 16 B = 64 KiB (`qwen4exp_wide_rows.rs` TILE_*).

#include <cuda_bf16.h>

#define QT_LANES 64  // the reference's virtual lanes: two warps an output
#define QT_ROWS 32   // rows a launch at most

// The fused windows, a few integer ops a 32-bit word (two BF16 halves):
//   activations  biased exponent 64..191, |a| in [2^-63, 2^65): x + 0x2000 a
//                half moves the window to 128..255, i.e. bit 14 set. An
//                exponent >= 192 carries out of its half and reads bit 14
//                clear (a carry into the next half only ever comes from a
//                half that is itself flagged).
//   weights      biased exponent 64..127, |w| in [2^-63, 2): bits 14:13 = 01.
// Products of in-window operands lie in [2^-126, 2^66): normal, exact.
// Real weights reach 2^-72 and 2^0 (Qwen3.8-Flash-Next GDN in/out
// projections, LM head), so the weight window starts at 2^-63, not higher.
struct QtActWindow {
    unsigned all = 0xFFFFFFFFu;
    __device__ __forceinline__ void add(const uint4& v) {
        all &= (v.x + 0x20002000u) & (v.y + 0x20002000u) & (v.z + 0x20002000u) &
               (v.w + 0x20002000u);
    }
    __device__ __forceinline__ bool inside() const { return (all & 0x40004000u) == 0x40004000u; }
};
struct QtWeightWindow {
    unsigned all = 0xFFFFFFFFu, any = 0;
    __device__ __forceinline__ void add(const uint4& v) {
        all &= v.x & v.y & v.z & v.w;
        any |= v.x | v.y | v.z | v.w;
    }
    __device__ __forceinline__ bool inside() const {
        return (all & 0x20002000u) == 0x20002000u && (any & 0x40004000u) == 0;
    }
};

__device__ __forceinline__ void qt_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void qt_widen(const uint4& v, float f[8]) {
    const unsigned r[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = __uint_as_float(r[i] << 16);              // exact BF16 -> FP32
        f[2 * i + 1] = __uint_as_float(r[i] & 0xFFFF0000u);
    }
}

// One K-step of `live` (<= MT) staged rows into a thread's accumulators:
// `acc[j][t] += a_t[i] * w_j[i]`, i = 0..7 in order, fused or as the
// reference's separate multiply and add (the tier's note).
template <int MT, int NT, bool FUSE>
__device__ __forceinline__ void qt_rows_step(const uint4* row, unsigned live,
                                             const float (&wf)[NT][8], float (&acc)[NT][MT]) {
#pragma unroll
    for (int t = 0; t < MT; t++) {
        if ((unsigned)t >= live) break;
        float a[8];
        qt_widen(row[t * QT_LANES], a);
#pragma unroll
        for (int j = 0; j < NT; j++) {
            float y = acc[j][t];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (FUSE)
                    y = __fmaf_rn(a[i], wf[j][i], y);
                else
                    y += a[i] * wf[j][i];
            }
            acc[j][t] = y;
        }
    }
}

// The shfl_down 16/8/4/2/1 tree of every one of a warp's V accumulators at
// once, by recursive halving: at offset o a lane keeps one half of its live
// values and swaps the other with lane ^ o, adding what it receives. Each
// value is still summed over the same pairs in the same tree (lane i with
// i ^ 16, those sums with i ^ 8, ...; FP addition is commutative), so the
// totals are the tree's bits, for V / 32 shuffles a level instead of V.
// After it, slot k of lane L holds accumulator k + (V / 32) * L.
template <int V>
__device__ __forceinline__ void qt_tree(float (&v)[V], unsigned lane) {
    static_assert(V >= 32 && (V & (V - 1)) == 0, "a power of two >= 32 values");
#pragma unroll
    for (int lvl = 0; lvl < 5; lvl++) {
        const int o = 16 >> lvl, c = V >> (lvl + 1);
        const bool upper = lane & o;
#pragma unroll
        for (int k = 0; k < c; k++) {
            const float send = upper ? v[k] : v[k + c];
            const float keep = upper ? v[k + c] : v[k];
            v[k] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, o);
        }
    }
}

// MT rows x NT outputs a thread; RB row blocks x OB output blocks of 64
// lanes a CTA (MT * RB == 32, NPB = NT * OB outputs a CTA).
template <int MT, int NT, int RB, int OB>
__device__ __forceinline__ void qt_rows32(const __nv_bfloat16* __restrict__ A,
                                          const __nv_bfloat16* __restrict__ B,
                                          __nv_bfloat16* __restrict__ C, unsigned M_, unsigned N,
                                          unsigned K, unsigned out_stride) {
    static_assert(MT * RB == QT_ROWS, "row blocks cover 32 rows");
    constexpr unsigned NPB = NT * OB, NTH = QT_LANES * RB * OB;
    extern __shared__ __align__(16) uint4 s_a[];  // [2][QT_ROWS][QT_LANES]
    const unsigned lane = threadIdx.x % QT_LANES, grp = threadIdx.x / QT_LANES;
    const unsigned rb = grp % RB, ob = grp / RB;
    const unsigned n0 = blockIdx.x * NPB + ob * NT, r0 = rb * MT;
    const unsigned M = M_ < QT_ROWS ? M_ : QT_ROWS;
    const unsigned K_VEC = K / 8, steps = (K_VEC + QT_LANES - 1) / QT_LANES;
    const uint4* A4 = (const uint4*)A;
    const uint4* B4 = (const uint4*)B;

    auto stage = [&](unsigned step, unsigned buf) {
        for (unsigned i = threadIdx.x; i < M * QT_LANES; i += NTH) {
            const unsigned t = i / QT_LANES, v = i % QT_LANES, kv = step * QT_LANES + v;
            if (kv < K_VEC)
                qt_cp16(&s_a[(buf * QT_ROWS + t) * QT_LANES + v], A4 + (size_t)t * K_VEC + kv);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto wload = [&](uint4 (&w)[NT], unsigned step) {
        const unsigned kv = step * QT_LANES + lane;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            w[j] = make_uint4(0, 0, 0, 0);
            if (n0 + j < N && kv < K_VEC) w[j] = B4[(size_t)(n0 + j) * K_VEC + kv];
        }
    };

    float acc[NT][MT];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) acc[j][t] = 0.0f;

    uint4 wnext[NT];
    wload(wnext, 0);
    stage(0, 0);
    for (unsigned s = 0; s < steps; s++) {
        const unsigned buf = s & 1;
        if (s + 1 < steps) {
            stage(s + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        // This thread's own copies of step s have landed: window-check them,
        // then the CTA agrees (and every copy is visible) at the barrier.
        QtActWindow aw;
        for (unsigned i = threadIdx.x; i < M * QT_LANES; i += NTH)
            if (s * QT_LANES + i % QT_LANES < K_VEC) aw.add(s_a[buf * QT_ROWS * QT_LANES + i]);
        QtWeightWindow ww;
        float wf[NT][8];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            ww.add(wnext[j]);
            qt_widen(wnext[j], wf[j]);
        }
        const bool fuse = !__syncthreads_or(!aw.inside()) && ww.inside();
        if (s + 1 < steps) wload(wnext, s + 1);
        if (s * QT_LANES + lane < K_VEC) {
            const uint4* row = s_a + (buf * QT_ROWS + r0) * QT_LANES + lane;
            // Branch-free row loops (a full row block, the fused or the
            // reference arithmetic) so the rows' independent chains interleave.
            if (r0 + MT <= M) {
                if (fuse)
                    qt_rows_step<MT, NT, true>(row, MT, wf, acc);
                else
                    qt_rows_step<MT, NT, false>(row, MT, wf, acc);
            } else if (r0 < M) {
                if (fuse)
                    qt_rows_step<MT, NT, true>(row, M - r0, wf, acc);
                else
                    qt_rows_step<MT, NT, false>(row, M - r0, wf, acc);
            }
        }
        __syncthreads();  // step s's readers are done before stage(s + 2) refills buf
    }

    // The reference's tree a warp (`qt_tree`), then lane0 + lane32 through
    // shared memory (the staging buffer, idle since the last barrier).
    static_assert(sizeof(float) * QT_ROWS * NPB * 2 <= 2 * QT_ROWS * QT_LANES * 16, "red fits");
    constexpr int V = MT * NT, PER = V / 32;
    float* red = (float*)s_a;
    float v[V];
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) v[j * MT + t] = acc[j][t];
    qt_tree<V>(v, lane % 32);
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const unsigned idx = k + PER * (lane % 32), j = idx / MT, t = idx % MT;
        red[((r0 + t) * NPB + ob * NT + j) * 2 + lane / 32] = v[k];
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < M * NPB; i += NTH) {
        const unsigned t = i / NPB, n = blockIdx.x * NPB + i % NPB;
        if (n < N) C[(size_t)t * out_stride + n] = __float2bfloat16(red[2 * i] + red[2 * i + 1]);
    }
}

#define QT_PARAMS                                                                     \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B,         \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K, unsigned out_stride
#define QT_ARGS A, B, C, M, N, K, out_stride

// Production entry point, spelled out for the kernel-name check.
extern "C" __global__ void __launch_bounds__(QT_LANES * 2 * 4, 1) qwen4exp_bf16_rows32t(QT_PARAMS) {
    qt_rows32<16, 4, 2, 4>(QT_ARGS);
}

#ifdef QT_SWEEP
// Bench-only shapes: rows a thread, outputs a thread, row blocks, output
// blocks, min CTAs an SM.
#define QT_SWEEP_TILE(name, MT, NT, RB, OB, MINB)                                          \
    extern "C" __global__ void __launch_bounds__(QT_LANES * RB * OB, MINB) name(QT_PARAMS) { \
        qt_rows32<MT, NT, RB, OB>(QT_ARGS);                                                 \
    }
QT_SWEEP_TILE(qt_m8_n4_r4_o2, 8, 4, 4, 2, 1)
QT_SWEEP_TILE(qt_m16_n2_r2_o4, 16, 2, 2, 4, 1)
QT_SWEEP_TILE(qt_m8_n4_r4_o1, 8, 4, 4, 1, 2)
QT_SWEEP_TILE(qt_m4_n8_r8_o1, 4, 8, 8, 1, 1)
#endif
