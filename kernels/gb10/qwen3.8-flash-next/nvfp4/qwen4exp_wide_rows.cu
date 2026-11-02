// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) exact GEMVs over 9..32 rows in ONE weight
// pass (ATLAS_QWEN4EXP_BATCH_FAST=1, `layers/ops/qwen4exp_wide_rows.rs`):
// the 16- and 32-row tiers above the 8-row `dense_gemv_bf16_batchm` and the
// 4-row `w4a16_gemv_qg_batch4`, for a batched speculative verify of C4..C8
// sequences x K=4 rows.
//
//   qwen4exp_bf16_rows16/32  row t byte-identical to `dense_gemv_bf16` on row t
//   qwen4exp_qg_rows16/32    row t byte-identical to `w4a16_gemv_qg` on row t
//
// THE INVARIANT. Every (row t, output n) is one FP32 accumulator walked
// exactly as the single-row kernel walks it, and nothing else touches it:
// "virtual lane" l in 0..63 of the output takes the 8-element vectors
// kv = l, l + 64, l + 128, ... < K/8 in order, each vector's elements in the
// order lo, hi of words 0..3, `acc += a * w` as a separate multiply and add
// (this directory builds with --fmad=false), then lanes 0..31 and 32..63
// each reduce by the shfl_down 16/8/4/2/1 tree and the output is
// `lane0 + lane32` rounded once to BF16. The rows of a launch only add
// independent accumulators, so a row's bits depend neither on M nor on which
// rows share the launch (tested at every M = 1..32,
// scripts/dev/qwen4exp_wide_rows_bench.cu).
//
// WHAT CHANGES vs the narrow tiers is only where a row's activation comes
// from: a K-step (64 vectors, 1 KiB a row) of every row is staged in shared
// memory, so the threads of a CTA read the activation rows once per CTA from
// L2 instead of once per output from L1 (at 32 rows the activation block is
// 160 KiB, past L1). The weight vector of the next step is loaded into
// registers before the barrier, so a CTA keeps a weight load in flight while
// it computes. The 32-row tiers are "pair" tiers: a thread runs the same
// virtual lane of TWO outputs, so each staged activation vector is read from
// shared memory and widened to FP32 once for two outputs' accumulators.
//
//   tier     CTA                     staging     regs  shared  bound at full width
//   rows16   4 outputs, 256 threads  2 x 16 rows   56  32 KiB  DRAM (16 rows cost 8)
//   rows32   16 outputs, 512 thr.    16-row pass  124  16 KiB  FP32 issue (1 CTA/SM)
//   qg32     8 outputs, 256 threads  16-row pass  128  16 KiB  FP32 issue (2 CTA/SM)
//
// The 32-row bound (ncu, LM head half, M = 32): the separate FP32 multiply
// and add are 682M of 1.08G warp instructions (2 a multiply-accumulate,
// fixed by --fmad=false and the bit-identity), the BF16 widening 170M (half
// of the one-output-a-thread kernel's 340M). Staging the rows as FP32 instead
// (no widening in the loop) was measured slower at every shape: it doubles
// the shared-memory bytes a multiply-accumulate reads (MIO throttle, short
// scoreboard stalls at 16 warps an SM); so was splitting a CTA's rows over
// more threads (registers spill or occupancy falls).
//
// A GB10, us per call, weights streamed from DRAM, against the
// chunks the lane ran before (scripts/dev/qwen4exp_wide_rows_bench.cu):
//
//   rows                          8      16      24      32
//   GDN qkvz TP2   8-row chunks  182     397     575     764
//                  rows16/32     172     178     244     278  (one output a thread: 259, 310)
//   GDN out_proj   8-row chunks   69     129     195     207
//                  rows16/32      69      72     108     119  (one output a thread: 100, 121)
//   LM head half   8-row chunks  2.59ms  5.09ms  7.63ms  10.17ms
//                  rows16/32     2.50ms  2.62ms  3.34ms  3.85ms  (one output a thread: 3.76, 4.49)
//   Q+gate TP2     4-row chunks   83     160     241     316
//                  rows16/32       -     130     178     208  (one output a thread: 202, 243)
//
// At <= 8 rows the narrow kernels stay (they are as fast there).
//
// Layouts: A [M, K] BF16 rows K apart; C row t at C + t * out_stride.
//   BF16  B [N, K] BF16.
//   QG    B [N, K/2] packed E2M1, S [N, K/16] E4M3 block scales, s2 FP32;
//         n indexes the interleaved [Q_h(hd) | G_h(hd)] rows and lands
//         deinterleaved [Q | G], as `w4a16_gemv_qg` writes it.
// K must be a multiple of 8 (BF16) / 16 (QG): no scalar tail (the host
// refuses other K). Grid: (ceil(N / NPB), 1, 1); block 64 threads an output,
// or an output pair in the pair tiers (`qwen4exp_wide_rows.rs` BF16_CTA, QG_CTA).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QW_TPO 64   // threads per output: the reference's two warps
#define QW_WARP 32

__device__ __constant__ float QW_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

struct QwArgs {
    const __nv_bfloat16* A;
    const unsigned char* B;   // BF16 or packed E2M1 weight rows
    const unsigned char* S;   // E4M3 scales (QG)
    float s2;                 // second-level scale (QG)
    __nv_bfloat16* C;
    unsigned M, N, K, out_stride, heads, head_dim;
};

// BF16 weights: one 16-byte vector a step, as `dense_gemv_bf16` loads it.
struct QwBf16 {
    typedef uint4 Raw;
    static __device__ __forceinline__ Raw load(const QwArgs& p, unsigned n, unsigned kv) {
        return ((const uint4*)(p.B + (unsigned long long)n * p.K * 2))[kv];
    }
    static __device__ __forceinline__ void decode(const QwArgs&, const Raw& r, const float*,
                                                  float w[8]) {
        const unsigned br[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            w[2 * i] = __uint_as_float(br[i] << 16);              // exact BF16 -> FP32
            w[2 * i + 1] = __uint_as_float(br[i] & 0xFFFF0000u);
        }
    }
    static __device__ __forceinline__ unsigned out_col(const QwArgs&, unsigned n) { return n; }
};

// NVFP4 Q+gate: 4 packed bytes and their group scale a step, each weight
// `lut[nibble] * (e4m3(scale) * s2)`, as `w4a16_gemv_qg` decodes it.
struct QwQg {
    struct Raw { unsigned packed; unsigned char scale; };
    static __device__ __forceinline__ Raw load(const QwArgs& p, unsigned n, unsigned kv) {
        Raw r;
        r.packed = *(const unsigned*)(p.B + (unsigned long long)n * (p.K / 2) + kv * 4);
        r.scale = p.S[(unsigned long long)n * (p.K / 16) + kv * 8 / 16];
        return r;
    }
    static __device__ __forceinline__ void decode(const QwArgs& p, const Raw& r, const float* lut,
                                                  float w[8]) {
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = r.scale;
        const float scale = (float)fp8 * p.s2;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const unsigned byte_val = (r.packed >> (b * 8)) & 0xFFu;
            w[2 * b] = lut[byte_val & 0xFu] * scale;
            w[2 * b + 1] = lut[byte_val >> 4] * scale;
        }
    }
    // Interleaved [Q_h | G_h] row n -> deinterleaved [Q | G] column.
    static __device__ __forceinline__ unsigned out_col(const QwArgs& p, unsigned n) {
        const unsigned group = 2 * p.head_dim, h = n / group, idx = n % group;
        return idx < p.head_dim ? h * p.head_dim + idx
                                : p.heads * p.head_dim + h * p.head_dim + (idx - p.head_dim);
    }
};

__device__ __forceinline__ void qw_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

template <int MAX_M, int NPB, int STAGES, typename W>
__device__ __forceinline__ void qw_rows_impl(const QwArgs p) {
    constexpr unsigned NT = NPB * QW_TPO;
    const unsigned local_out = threadIdx.x / QW_TPO;
    const unsigned lane = threadIdx.x % QW_TPO;
    const unsigned n = blockIdx.x * NPB + local_out;
    const bool live = n < p.N;
    const unsigned M = p.M < MAX_M ? p.M : MAX_M;
    const unsigned K_VEC = p.K / 8;
    const unsigned steps = (K_VEC + QW_TPO - 1) / QW_TPO;

    __shared__ __align__(16) uint4 s_a[STAGES][MAX_M][QW_TPO];
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QW_E2M1_LUT[threadIdx.x];

    // Stage step `step`'s activation vectors of every row into buffer `buf`.
    auto stage = [&](unsigned step, unsigned buf) {
        for (unsigned i = threadIdx.x; i < M * QW_TPO; i += NT) {
            const unsigned t = i / QW_TPO, v = i % QW_TPO, kv = step * QW_TPO + v;
            if (kv < K_VEC)
                qw_cp16(&s_a[buf][t][v], (const uint4*)(p.A + (unsigned long long)t * p.K) + kv);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[MAX_M];
#pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    typename W::Raw next{};
    if (live && lane < K_VEC) next = W::load(p, n, lane);
    stage(0, 0);
    for (unsigned s = 0; s < steps; s++) {
        const typename W::Raw cur = next;
        const unsigned kv = s * QW_TPO + lane;
        if (live && kv + QW_TPO < K_VEC) next = W::load(p, n, kv + QW_TPO);
        if (STAGES == 2 && s + 1 < steps) {
            stage(s + 1, (s + 1) % STAGES);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        if (live && kv < K_VEC) {
            float w[8];
            W::decode(p, cur, s_lut, w);
            const uint4(*buf)[QW_TPO] = s_a[s % STAGES];
#pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned)t >= M) break;
                const uint4 a = buf[t][lane];
                const unsigned ar[4] = {a.x, a.y, a.z, a.w};
                float x = acc[t];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    x += __uint_as_float(ar[i] << 16) * w[2 * i];
                    x += __uint_as_float(ar[i] & 0xFFFF0000u) * w[2 * i + 1];
                }
                acc[t] = x;
            }
        }
        __syncthreads();
        if (STAGES == 1 && s + 1 < steps) stage(s + 1, 0);
    }

#pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        float a = acc[t];
#pragma unroll
        for (int off = QW_WARP / 2; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, off);
        acc[t] = a;
    }
    // The warp partials of every (row, output), in the activation buffer
    // (free since the loop's last barrier).
    static_assert(sizeof(float) * MAX_M * NPB * 2 <= sizeof(s_a), "reduction fits s_a");
    float* red = (float*)s_a;
    if (threadIdx.x % QW_WARP == 0) {
#pragma unroll
        for (int t = 0; t < MAX_M; t++)
            if ((unsigned)t < M) red[(t * NPB + local_out) * 2 + lane / QW_WARP] = acc[t];
    }
    __syncthreads();
    // Thread `lane` of the output writes row `lane`: lane0 + lane32, as the
    // single-row kernel's `smem[w0] + smem[w1]`.
    if (live && lane < M) {
        const float r = red[(lane * NPB + local_out) * 2] + red[(lane * NPB + local_out) * 2 + 1];
        p.C[(unsigned long long)lane * p.out_stride + W::out_col(p, n)] = __float2bfloat16(r);
    }
}

// The pair tiers: OPT outputs a thread. Thread `lane` of output group g
// walks lane `lane` of outputs g*OPT .. g*OPT+OPT-1, each output's own
// accumulator taking exactly the products `qw_rows_impl` gives it, in the
// same order; one shared-memory load and BF16 widening of an activation
// vector then feeds OPT outputs instead of one. A K-step's rows are staged
// RP at a time (a "pass"), the next pass fetched into registers while this
// one computes.
template <int MAX_M, int NPB, int OPT, int RP, typename W>
__device__ __forceinline__ void qw_rows_pair_impl(const QwArgs p) {
    static_assert(NPB % OPT == 0 && MAX_M % RP == 0, "tile shape");
    constexpr unsigned NT = NPB / OPT * QW_TPO;
    constexpr unsigned PER = (RP * QW_TPO + NT - 1) / NT;  // staged vectors a thread
    constexpr int PASSES = MAX_M / RP;
    const unsigned lane = threadIdx.x % QW_TPO;
    const unsigned local0 = threadIdx.x / QW_TPO * OPT;
    const unsigned n0 = blockIdx.x * NPB + local0;
    const unsigned M = p.M < MAX_M ? p.M : MAX_M;
    const unsigned K_VEC = p.K / 8;
    const unsigned steps = (K_VEC + QW_TPO - 1) / QW_TPO;
    const unsigned passes = (M + RP - 1) / RP;

    __shared__ __align__(16) uint4 s_a[RP * QW_TPO];
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QW_E2M1_LUT[threadIdx.x];
    __syncthreads();  // the first step decodes before the walk's first barrier

    // Pass `u` (step u / passes, rows (u % passes) * RP ..) into registers;
    // zero past M or K.
    uint4 pre[PER];
    auto fetch = [&](unsigned u) {
        const unsigned s = u / passes, r0 = u % passes * RP;
#pragma unroll
        for (unsigned j = 0; j < PER; j++) {
            const unsigned i = threadIdx.x + j * NT, t = i / QW_TPO, kv = s * QW_TPO + i % QW_TPO;
            pre[j] = make_uint4(0, 0, 0, 0);
            if (i < RP * QW_TPO && r0 + t < M && kv < K_VEC)
                pre[j] = ((const uint4*)(p.A + (unsigned long long)(r0 + t) * p.K))[kv];
        }
    };

    float acc[OPT][MAX_M];
#pragma unroll
    for (int o = 0; o < OPT; o++)
#pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[o][t] = 0.0f;

    typename W::Raw next[OPT];
#pragma unroll
    for (int o = 0; o < OPT; o++) {
        next[o] = typename W::Raw{};
        if (n0 + o < p.N && lane < K_VEC) next[o] = W::load(p, n0 + o, lane);
    }
    fetch(0);
    unsigned u = 0;
    for (unsigned s = 0; s < steps; s++) {
        const unsigned kv = s * QW_TPO + lane;
        float w[OPT][8];
#pragma unroll
        for (int o = 0; o < OPT; o++) {
            W::decode(p, next[o], s_lut, w[o]);
            if (n0 + o < p.N && kv + QW_TPO < K_VEC) next[o] = W::load(p, n0 + o, kv + QW_TPO);
        }
#pragma unroll
        for (int pass = 0; pass < PASSES; pass++) {
            if ((unsigned)pass >= passes) break;
            __syncthreads();  // the previous pass's readers are done
#pragma unroll
            for (unsigned j = 0; j < PER; j++)
                if (threadIdx.x + j * NT < RP * QW_TPO) s_a[threadIdx.x + j * NT] = pre[j];
            __syncthreads();
            if (++u < steps * passes) fetch(u);
            if (kv < K_VEC) {
#pragma unroll
                for (int r = 0; r < RP; r++) {
                    const int t = pass * RP + r;
                    if ((unsigned)t >= M) break;
                    const uint4 x = s_a[r * QW_TPO + lane];
                    const unsigned ar[4] = {x.x, x.y, x.z, x.w};
                    float a[8];
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        a[2 * k] = __uint_as_float(ar[k] << 16);
                        a[2 * k + 1] = __uint_as_float(ar[k] & 0xFFFF0000u);
                    }
#pragma unroll
                    for (int o = 0; o < OPT; o++) {
                        float y = acc[o][t];
#pragma unroll
                        for (int i = 0; i < 8; i++) y += a[i] * w[o][i];
                        acc[o][t] = y;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int o = 0; o < OPT; o++)
#pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            float a = acc[o][t];
#pragma unroll
            for (int off = QW_WARP / 2; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, off);
            acc[o][t] = a;
        }
    static_assert(sizeof(float) * MAX_M * NPB * 2 <= sizeof(s_a), "reduction fits s_a");
    float* red = (float*)s_a;
    __syncthreads();  // the last pass's readers are done with s_a
    if (threadIdx.x % QW_WARP == 0) {
#pragma unroll
        for (int o = 0; o < OPT; o++)
#pragma unroll
            for (int t = 0; t < MAX_M; t++)
                if ((unsigned)t < M) red[(t * NPB + local0 + o) * 2 + lane / QW_WARP] = acc[o][t];
    }
    __syncthreads();
    if (lane < M) {
#pragma unroll
        for (int o = 0; o < OPT; o++) {
            if (n0 + o >= p.N) break;
            const unsigned at = (lane * NPB + local0 + o) * 2;
            p.C[(unsigned long long)lane * p.out_stride + W::out_col(p, n0 + o)] =
                __float2bfloat16(red[at] + red[at + 1]);
        }
    }
}

// Production entry points, spelled out so the kernel-name check
// (scripts/dev/check_qwen4exp_kernel_names.py) sees them.
#define QW_BF16_PARAMS                                                                 \
    const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B,          \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K, unsigned out_stride
#define QW_BF16_ARGS QwArgs{A, (const unsigned char*)B, nullptr, 0.0f, C, M, N, K, out_stride, 0, 0}
#define QW_QG_PARAMS                                                                   \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B_packed,   \
        const unsigned char *__restrict__ B_scale, const float scale2,                 \
        __nv_bfloat16 *__restrict__ C, unsigned M, unsigned N, unsigned K,             \
        unsigned out_stride, unsigned num_heads, unsigned head_dim
#define QW_QG_ARGS QwArgs{A, B_packed, B_scale, scale2, C, M, N, K, out_stride, num_heads, head_dim}

extern "C" __global__ void __launch_bounds__(4 * QW_TPO, 1) qwen4exp_bf16_rows16(QW_BF16_PARAMS) {
    qw_rows_impl<16, 4, 2, QwBf16>(QW_BF16_ARGS);
}
extern "C" __global__ void __launch_bounds__(8 * QW_TPO, 1) qwen4exp_bf16_rows32(QW_BF16_PARAMS) {
    qw_rows_pair_impl<32, 16, 2, 16, QwBf16>(QW_BF16_ARGS);
}
extern "C" __global__ void __launch_bounds__(4 * QW_TPO, 1) qwen4exp_qg_rows16(QW_QG_PARAMS) {
    qw_rows_impl<16, 4, 2, QwQg>(QW_QG_ARGS);
}
extern "C" __global__ void __launch_bounds__(4 * QW_TPO, 2) qwen4exp_qg_rows32(QW_QG_PARAMS) {
    qw_rows_pair_impl<32, 8, 2, 16, QwQg>(QW_QG_ARGS);
}

#ifdef QW_SWEEP
// Bench-only shapes (scripts/dev/qwen4exp_wide_rows_bench.cu sweep):
// rows, outputs per CTA, stages, min CTAs per SM.
#define QW_SWEEP_BF16(name, MAX_M, NPB, STAGES, MINB)                                   \
    extern "C" __global__ void __launch_bounds__(NPB * QW_TPO, MINB) name(QW_BF16_PARAMS) { \
        qw_rows_impl<MAX_M, NPB, STAGES, QwBf16>(QW_BF16_ARGS);                         \
    }
#define QW_SWEEP_QG(name, MAX_M, NPB, STAGES, MINB)                                     \
    extern "C" __global__ void __launch_bounds__(NPB * QW_TPO, MINB) name(QW_QG_PARAMS) { \
        qw_rows_impl<MAX_M, NPB, STAGES, QwQg>(QW_QG_ARGS);                             \
    }
QW_SWEEP_BF16(qw_sweep_bf16_m16_n8_s2, 16, 8, 2, 1)
QW_SWEEP_BF16(qw_sweep_bf16_m32_n16_s2, 32, 16, 2, 1)
QW_SWEEP_BF16(qw_sweep_bf16_m32_n4_s1, 32, 4, 1, 3)
QW_SWEEP_QG(qw_sweep_qg_m16_n8_s2, 16, 8, 2, 1)
QW_SWEEP_QG(qw_sweep_qg_m32_n16_s2, 32, 16, 2, 1)
QW_SWEEP_QG(qw_sweep_qg_m8_n4_s2, 8, 4, 2, 1)
// The pair tiers: rows, outputs per CTA, outputs a thread, rows a pass,
// min CTAs per SM; the 1-output-a-thread shapes of `qw_rows_impl` for
// comparison (rows32 before the pair tier).
#define QW_SWEEP_PAIR(name, Wt, PARAMS, ARGS, MAX_M, NPB, OPT, RP, MINB)                    \
    extern "C" __global__ void __launch_bounds__(NPB / OPT * QW_TPO, MINB) name(PARAMS) { \
        qw_rows_pair_impl<MAX_M, NPB, OPT, RP, Wt>(ARGS);                                 \
    }
QW_SWEEP_BF16(qw_sweep_bf16_m32_n8_s1, 32, 8, 1, 2)
QW_SWEEP_QG(qw_sweep_qg_m32_n8_s1, 32, 8, 1, 2)
QW_SWEEP_PAIR(qw_pair_bf16_m32_n8_o2_r16, QwBf16, QW_BF16_PARAMS, QW_BF16_ARGS, 32, 8, 2, 16, 2)
QW_SWEEP_PAIR(qw_pair_bf16_m32_n8_o2_r8, QwBf16, QW_BF16_PARAMS, QW_BF16_ARGS, 32, 8, 2, 8, 2)
QW_SWEEP_PAIR(qw_pair_bf16_m32_n16_o2_r8, QwBf16, QW_BF16_PARAMS, QW_BF16_ARGS, 32, 16, 2, 8, 1)
QW_SWEEP_PAIR(qw_pair_qg_m32_n16_o2_r16, QwQg, QW_QG_PARAMS, QW_QG_ARGS, 32, 16, 2, 16, 1)
#endif
