// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) routed + shared MoE decode GEMVs on the
// unified [K/2, N] expert layout, 1..4 rows in one launch
// (ATLAS_QWEN4EXP_MOE_FAST=1). NVFP4 experts: E2M1 nibbles, FP8-E4M3 per-16
// scales, per-tensor scale2. Shapes are this model's and nothing else's:
// hidden 2560, routed and shared intermediate 640.
//
// qwen4exp_moe_gate_up_t / qwen4exp_moe_silu_down_t replace, BIT FOR BIT,
// moe_expert_{gate_up,silu_down}_shared_t (one row; forward_batched runs it
// once per row) and its _batch2_t / _batch3_t twins (common/), which share
// one per-output chain: for each output n, sequentially over K,
//   sc = dec_e4m3(scale[k/16][n]) * s2;  w = lut[nibble] * sc;
//   acc += a_lo * w_lo + a_hi * w_hi;
// and for down, a = (g / (1 + __expf(-g))) * u of the bf16 gate/up rows. The
// target builds with --fmad=false, so nothing contracts and each output here
// is the same IEEE operation sequence. Only the work layout and the weight
// fetch change.
//
// Why the old kernels run at ~105 (gate/up) and ~145 (down) of GB10's ~245
// GB/s: one byte per lane per K step from 32-thread CTAs, so 20 one-warp CTAs
// per expert matrix each walk a 32-byte column strip of rows 640 bytes apart,
// and the K loop is only unrolled within a scale group, so a warp has 9 byte
// loads in flight. Rows (T > 1) each re-read every expert, the shared one
// included.
//
// What these do instead:
//   * V = 2 adjacent outputs per thread (2-byte loads), all K and N extents
//     compile-time, and a register ring of D = 4 scale groups (8 packed rows
//     + 1 scale row each) whose loads are all unconditional (a predicated
//     ring load made ptxas wait for zero outstanding loads on GB10 -- 4x
//     slower in the mHC campaign). The first groups' loads go out before the
//     activation staging and its barrier.
//   * one thread of each CTA asks L2 (cp.async.bulk.prefetch.L2) for scale
//     group g + PF's WHOLE rows (contiguous in [K/2, N]), the groups split
//     over the expert's CTAs, so DRAM streams each expert as long runs while
//     the column-strip loads hit L2. Idea credit: the GLM stream twins' L2PF
//     (8b3972a1), itself after jayleaton's glm53-tensorfold-spark 0460 as
//     carried in Mia's TensorFold recipe (docs/glm-prior-art.md). No code
//     copied.
//   * rows: one launch covers every row. A routed expert picked by several
//     rows is read ONCE: the CTA of its first slot computes every row that
//     picked it (one accumulator per row, so each row's chain is unchanged)
//     and the CTAs of the later duplicate slots exit. The shared expert
//     serves all rows from one read. (Weight reuse across rows routed to the
//     same expert, and one launch for all routed experts, follow SparkGLM's
//     GLM MoE decode campaign in this tree: layers/moe/decode_m16.rs.)
//   * each warp resolves its CTA's (expert, rows) unit from one ballot over
//     the slot indices -- no barrier, no shared memory.
//
// scripts/dev/qwen4exp_moe_decode_bench.cu checks every output byte against
// the replaced kernels and times both; see crates/spark-model/src/layers/
// ops/qwen4exp_moe.rs for the launch shapes.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QX_ROWS_MAX 4
#define QX_SLOTS_MAX 64
#define QX_HIDDEN 2560
#define QX_INTER 640

__device__ __constant__ float QX_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// The verbatim NVIDIA cast of the replaced kernels' atlas_dec_e4m3 (exact).
__device__ __forceinline__ float qx_dec_e4m3(unsigned int b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)b;
    return (float)f;
}

__device__ __forceinline__ float qx_bf16(unsigned short u) {
    __nv_bfloat16 h;
    *(unsigned short*)&h = u;
    return __bfloat162float(h);
}

// ── Row/slot resolution ─────────────────────────────────────────────────
//
// blockIdx.y < rows*top_k is routed slot q (row q / top_k); blockIdx.y ==
// rows*top_k is the shared expert. A routed CTA whose expert also appears at
// an EARLIER slot exits: that slot's CTA owns the expert and computes every
// row that picked it. Top-k picks are distinct within a row, so an expert
// serves at most one slot per row and `n <= rows`.
//
// Every warp resolves its CTA's unit on its own from one ballot over the
// (at most 64) slot indices: no barrier, no shared memory, and the per-row
// fields live packed in two registers (8 bits a row) rather than in an array
// a runtime index would push to local memory.
struct QxUnit {
    int n;              // rows served (0: this CTA has nothing to do)
    unsigned in_pack;   // input row (token) of served row r at bits 8r..8r+7
    unsigned out_pack;  // output row (routed slot, or token for shared), same
    unsigned expert;    // routed expert id (unused for shared)
    bool shared;
    __device__ __forceinline__ unsigned in(int r) const { return (in_pack >> (8 * r)) & 0xFFu; }
    __device__ __forceinline__ unsigned out(int r) const { return (out_pack >> (8 * r)) & 0xFFu; }
};

__device__ __forceinline__ QxUnit qx_resolve(
    const unsigned* __restrict__ expert_indices, unsigned rows, unsigned top_k
) {
    QxUnit u;
    const unsigned slots = rows * top_k;
    const unsigned y = blockIdx.y;
    u.n = 0;
    u.in_pack = 0;
    u.out_pack = 0;
    u.expert = 0;
    u.shared = (y >= slots);
    if (u.shared) {
        for (unsigned t = 0; t < rows; t++) {
            u.in_pack |= t << (8 * t);
            u.out_pack |= t << (8 * t);
        }
        u.n = (int)rows;
        return u;
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned v0 = lane < slots ? __ldg(expert_indices + lane) : 0xFFFFFFFFu;
    const unsigned v1 = lane + 32 < slots ? __ldg(expert_indices + lane + 32) : 0xFFFFFFFFu;
    const unsigned e = y < 32 ? __shfl_sync(0xFFFFFFFFu, v0, y) : __shfl_sync(0xFFFFFFFFu, v1, y - 32);
    const unsigned long long hits = (unsigned long long)__ballot_sync(0xFFFFFFFFu, v0 == e)
                                  | ((unsigned long long)__ballot_sync(0xFFFFFFFFu, v1 == e) << 32);
    u.expert = e;
    if (hits & ((1ull << y) - 1)) return u;  // a duplicate slot: the first one owns it
    unsigned long long m = hits;
    while (m && u.n < QX_ROWS_MAX) {
        const unsigned q = (unsigned)__ffsll((long long)m) - 1;
        m &= m - 1;
        u.in_pack |= (q / top_k) << (8 * u.n);
        u.out_pack |= q << (8 * u.n);
        u.n++;
    }
    return u;
}

// ═════════════════════════════════════════════════════════════════════════
// _t: unified [K/2, N] layout
// ═════════════════════════════════════════════════════════════════════════

template <int V> struct QxWord;
template <> struct QxWord<1> { typedef unsigned char T; };
template <> struct QxWord<2> { typedef unsigned short T; };
template <> struct QxWord<4> { typedef unsigned int T; };
template <> struct QxWord<8> { typedef uint2 T; };

__device__ __forceinline__ unsigned qx_byte(unsigned char w, int) { return (unsigned)w; }
__device__ __forceinline__ unsigned qx_byte(unsigned short w, int v) { return ((unsigned)w >> (8 * v)) & 0xFFu; }
__device__ __forceinline__ unsigned qx_byte(unsigned int w, int v) { return (w >> (8 * v)) & 0xFFu; }
__device__ __forceinline__ unsigned qx_byte(uint2 w, int v) { return ((v < 4 ? w.x : w.y) >> (8 * (v & 3))) & 0xFFu; }

// Bulk L2 prefetch (no destination register, so it never holds a scoreboard).
__device__ __forceinline__ void qx_l2_prefetch(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// Ask L2 for scale group g's WHOLE rows (8 packed rows + 1 scale row, both
// contiguous in [K/2, N]), each group by one of the unit's gridDim.x CTAs, so
// DRAM serves an expert as long contiguous runs while the CTAs' column-strip
// loads hit L2. The GLM stream twins' L2PF does the same per stage
// (8b3972a1). Only immutable weights are touched.
template <int PF>
__device__ __forceinline__ void qx_t_prefetch(
    const unsigned char* packed, const unsigned char* scale, unsigned N, int g, int G
) {
    // A hint only: skipped for a base that is not 16-byte aligned (the
    // bulk prefetch requires it; per-tensor cuMemAlloc and 16-byte-multiple
    // slab offsets always satisfy it).
    if (PF > 0 && threadIdx.x == 0 && g < G && (unsigned)g % gridDim.x == blockIdx.x
        && (((size_t)packed | (size_t)scale) & 15) == 0) {
        qx_l2_prefetch(packed + (size_t)g * 8 * N, 8 * N);
        qx_l2_prefetch(scale + (size_t)g * N, N);
    }
}

// One scale group (16 K = 8 rows of the [K/2, N] matrix) for V outputs and
// R rows. `a[r]` points at row r's fp32 activations; per output the chain is
// the replaced kernel's: sc = dec(scale) * s2; w = lut[nib] * sc;
// acc += a_lo * w_lo + a_hi * w_hi, row by row in K order.
template <int V, int R>
__device__ __forceinline__ void qx_t_group(
    int g,
    const typename QxWord<V>::T (&w)[8],
    typename QxWord<V>::T s,
    float s2,
    const float* const (&a)[R],
    const float* __restrict__ s_lut,
    float (&acc)[R][V]
) {
    float sc[V];
    #pragma unroll
    for (int v = 0; v < V; v++) sc[v] = qx_dec_e4m3(qx_byte(s, v)) * s2;
    #pragma unroll
    for (int jj = 0; jj < 8; jj += 2) {
        float4 av[R];
        #pragma unroll
        for (int r = 0; r < R; r++) av[r] = *(const float4*)(a[r] + g * 16 + jj * 2);
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int j = jj + h;
            #pragma unroll
            for (int v = 0; v < V; v++) {
                const unsigned b = qx_byte(w[j], v);
                const float w_lo = s_lut[b & 0xFu] * sc[v];
                const float w_hi = s_lut[(b >> 4) & 0xFu] * sc[v];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const float a_lo = h ? av[r].z : av[r].x;
                    const float a_hi = h ? av[r].w : av[r].y;
                    acc[r][v] += a_lo * w_lo + a_hi * w_hi;
                }
            }
        }
    }
}

// The whole K walk for V adjacent outputs starting at column n0: a ring of D
// scale groups in registers, every load unconditional. G % D == 0, so the
// steady loop refills each slot right after consuming it and the last D
// groups drain without loads. `stage()` (the smem activation fill and its
// barrier) runs after the first D groups' loads are in flight.
template <int V, int R, int D, int PF, int KH, typename Stage>
__device__ __forceinline__ void qx_t_walk(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale,
    float s2,
    unsigned N,
    unsigned n0,
    const float* const (&a)[R],
    const float* __restrict__ s_lut,
    float (&acc)[R][V],
    Stage&& stage
) {
    typedef typename QxWord<V>::T W;
    constexpr int G = KH / 8;
    static_assert(KH % 8 == 0 && G % D == 0 && G >= D, "ring must tile the scale groups");
    #pragma unroll
    for (int g = 0; g < PF; g++) qx_t_prefetch<PF>(packed, scale, N, g, G);
    const unsigned char* pp = packed + n0;
    const unsigned char* sp = scale + n0;
    const size_t row = N;
    W w[D][8];
    W s[D];
    #pragma unroll
    for (int d = 0; d < D; d++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) w[d][j] = __ldg((const W*)(pp + (size_t)(d * 8 + j) * row));
        s[d] = __ldg((const W*)(sp + (size_t)d * row));
    }
    stage();
    #pragma unroll 1
    for (int g0 = 0; g0 < G - D; g0 += D) {
        const unsigned char* pn = pp + (size_t)(g0 + D) * 8 * row;
        const unsigned char* sn = sp + (size_t)(g0 + D) * row;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            qx_t_prefetch<PF>(packed, scale, N, g0 + d + PF, G);
            qx_t_group<V, R>(g0 + d, w[d], s[d], s2, a, s_lut, acc);
            #pragma unroll
            for (int j = 0; j < 8; j++) w[d][j] = __ldg((const W*)(pn + (size_t)(d * 8 + j) * row));
            s[d] = __ldg((const W*)(sn + (size_t)d * row));
        }
    }
    #pragma unroll
    for (int d = 0; d < D; d++) qx_t_group<V, R>(G - D + d, w[d], s[d], s2, a, s_lut, acc);
}

__device__ __forceinline__ unsigned qx_pack_bf16x2(float lo, float hi) {
    return (unsigned)__bfloat16_as_ushort(__float2bfloat16(lo))
         | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16);
}

template <int V>
__device__ __forceinline__ void qx_store_bf16(__nv_bfloat16* __restrict__ dst, const float (&x)[V]) {
    if constexpr (V == 1) {
        *dst = __float2bfloat16(x[0]);
    } else if constexpr (V == 2) {
        *(unsigned*)dst = qx_pack_bf16x2(x[0], x[1]);
    } else if constexpr (V == 4) {
        *(uint2*)dst = make_uint2(qx_pack_bf16x2(x[0], x[1]), qx_pack_bf16x2(x[2], x[3]));
    } else {
        *(uint4*)dst = make_uint4(qx_pack_bf16x2(x[0], x[1]), qx_pack_bf16x2(x[2], x[3]),
                                  qx_pack_bf16x2(x[4], x[5]), qx_pack_bf16x2(x[6], x[7]));
    }
}

template <int V>
__device__ __forceinline__ void qx_store_zero(__nv_bfloat16* __restrict__ dst) {
    float z[V];
    #pragma unroll
    for (int v = 0; v < V; v++) z[v] = 0.0f;
    qx_store_bf16<V>(dst, z);
}

// Run the unit's R rows (staged by `stage` at s_act[r * 2*KH]) through the
// walk and store row r's V outputs at out_base + u.out(r) * N + n0.
template <int V, int D, int PF, int KH, int R, typename Stage>
__device__ __forceinline__ void qx_t_rows(
    const QxUnit& u,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale,
    float s2,
    unsigned N,
    unsigned n0,
    const float* __restrict__ s_act,
    const float* __restrict__ s_lut,
    __nv_bfloat16* __restrict__ out_base,
    Stage&& stage
) {
    const float* a[R];
    #pragma unroll
    for (int r = 0; r < R; r++) a[r] = s_act + r * (2 * KH);
    float acc[R][V];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int v = 0; v < V; v++) acc[r][v] = 0.0f;
    }
    qx_t_walk<V, R, D, PF, KH>(packed, scale, s2, N, n0, a, s_lut, acc, stage);
    #pragma unroll
    for (int r = 0; r < R; r++) qx_store_bf16<V>(out_base + (size_t)u.out(r) * N + n0, acc[r]);
}

template <int V, int D, int PF, int KH, typename Stage>
__device__ __forceinline__ void qx_t_dispatch(
    const QxUnit& u,
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale,
    float s2,
    unsigned N,
    unsigned n0,
    const float* __restrict__ s_act,
    const float* __restrict__ s_lut,
    __nv_bfloat16* __restrict__ out_base,
    Stage&& stage
) {
    switch (u.n) {
    case 1: qx_t_rows<V, D, PF, KH, 1>(u, packed, scale, s2, N, n0, s_act, s_lut, out_base, stage); break;
    case 2: qx_t_rows<V, D, PF, KH, 2>(u, packed, scale, s2, N, n0, s_act, s_lut, out_base, stage); break;
    case 3: qx_t_rows<V, D, PF, KH, 3>(u, packed, scale, s2, N, n0, s_act, s_lut, out_base, stage); break;
    default: qx_t_rows<V, D, PF, KH, 4>(u, packed, scale, s2, N, n0, s_act, s_lut, out_base, stage); break;
    }
}

// gate+up, unified layout. Grid (N / (blockDim.x * V), rows*top_k + 1, 2),
// dynamic smem rows * K * 4 bytes (the unit's input rows as fp32).
template <int V, int D, int PF>
__device__ __forceinline__ void qx_gate_up_t_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows
) {
    constexpr unsigned KK = QX_HIDDEN;
    if (K != KK || N % (blockDim.x * V) != 0 || rows == 0 || rows > QX_ROWS_MAX
        || rows * top_k > QX_SLOTS_MAX) {
        __trap();
    }
    extern __shared__ float s_act[];  // [u.n][K]
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QX_E2M1_LUT[threadIdx.x];
    const QxUnit u = qx_resolve(expert_indices, rows, top_k);
    if (u.n == 0) return;
    const unsigned proj = blockIdx.z;
    const unsigned char* packed;
    const unsigned char* scale;
    float s2;
    __nv_bfloat16* out;
    if (u.shared) {
        packed = proj == 0 ? sh_gate_packed : sh_up_packed;
        scale = proj == 0 ? sh_gate_scale : sh_up_scale;
        s2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        out = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        packed = (const unsigned char*)(proj == 0 ? gate_packed_ptrs : up_packed_ptrs)[u.expert];
        scale = (const unsigned char*)(proj == 0 ? gate_scale_ptrs : up_scale_ptrs)[u.expert];
        s2 = (proj == 0 ? gate_scale2_vals : up_scale2_vals)[u.expert];
        out = proj == 0 ? gate_out : up_out;
    }
    const unsigned n0 = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (packed == 0) {
        // NULL: no shared expert, or (EP) a routed expert another rank holds.
        for (int r = 0; r < u.n; r++) qx_store_zero<V>(out + (size_t)u.out(r) * N + n0);
        return;
    }
    // Stage the unit's input rows as fp32 (bf16 -> fp32 is exact).
    auto stage = [&]() {
        for (unsigned i = threadIdx.x; i < (unsigned)u.n * (KK / 8); i += blockDim.x) {
            const unsigned r = i / (KK / 8), c = i % (KK / 8);
            const uint4 raw = __ldg((const uint4*)(A + (size_t)u.in(r) * KK) + c);
            const unsigned wds[4] = {raw.x, raw.y, raw.z, raw.w};
            float f[8];
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                f[2 * q] = qx_bf16((unsigned short)(wds[q] & 0xFFFFu));
                f[2 * q + 1] = qx_bf16((unsigned short)(wds[q] >> 16));
            }
            float4* dst = (float4*)(s_act + r * KK) + 2 * c;
            dst[0] = make_float4(f[0], f[1], f[2], f[3]);
            dst[1] = make_float4(f[4], f[5], f[6], f[7]);
        }
        __syncthreads();
    };
    qx_t_dispatch<V, D, PF, KK / 2>(u, packed, scale, s2, N, n0, s_act, s_lut, out, stage);
}

// SiLU*up then down, unified layout. Grid (N / (blockDim.x * V),
// rows*top_k + 1), dynamic smem rows * K * 4 bytes.
template <int V, int D, int PF>
__device__ __forceinline__ void qx_silu_down_t_impl(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows
) {
    constexpr unsigned KK = QX_INTER;
    if (K != KK || N % (blockDim.x * V) != 0 || rows == 0 || rows > QX_ROWS_MAX
        || rows * top_k > QX_SLOTS_MAX) {
        __trap();
    }
    extern __shared__ float s_act[];  // [u.n][K]
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QX_E2M1_LUT[threadIdx.x];
    const QxUnit u = qx_resolve(expert_indices, rows, top_k);
    if (u.n == 0) return;

    const unsigned char* packed;
    const unsigned char* scale;
    float s2;
    __nv_bfloat16* out;
    const __nv_bfloat16* g_base;
    const __nv_bfloat16* u_base;
    if (u.shared) {
        packed = sh_down_packed; scale = sh_down_scale; s2 = sh_down_s2;
        out = sh_down_out; g_base = sh_gate_in; u_base = sh_up_in;
    } else {
        packed = (const unsigned char*)packed_ptrs[u.expert];
        scale = (const unsigned char*)scale_ptrs[u.expert];
        s2 = scale2_vals[u.expert];
        out = C; g_base = gate_out; u_base = up_out;
    }
    const unsigned n0 = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (packed == 0) {
        for (int r = 0; r < u.n; r++) qx_store_zero<V>(out + (size_t)u.out(r) * N + n0);
        return;
    }
    // A unit row reads the gate/up row of its OUTPUT row: the routed slot
    // (gate_out is slot-major) or the token (shared inputs are token-major).
    auto stage = [&]() {
        for (unsigned i = threadIdx.x; i < (unsigned)u.n * KK; i += blockDim.x) {
            const unsigned r = i / KK, k = i % KK;
            const float gf = __bfloat162float(g_base[(size_t)u.out(r) * KK + k]);
            const float uf = __bfloat162float(u_base[(size_t)u.out(r) * KK + k]);
            s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
        }
        __syncthreads();
    };
    qx_t_dispatch<V, D, PF, KK / 2>(u, packed, scale, s2, N, n0, s_act, s_lut, out, stage);
}

// ── Entry points ─────────────────────────────────────────────────────────

#define QX_GATE_UP_ARGS                                                        \
    const __nv_bfloat16* __restrict__ A,                                       \
    const unsigned long long* __restrict__ gate_packed_ptrs,                   \
    const unsigned long long* __restrict__ gate_scale_ptrs,                    \
    const float* __restrict__ gate_scale2_vals,                                \
    __nv_bfloat16* __restrict__ gate_out,                                      \
    const unsigned long long* __restrict__ up_packed_ptrs,                     \
    const unsigned long long* __restrict__ up_scale_ptrs,                      \
    const float* __restrict__ up_scale2_vals,                                  \
    __nv_bfloat16* __restrict__ up_out,                                        \
    const unsigned int* __restrict__ expert_indices,                           \
    const unsigned char* __restrict__ sh_gate_packed,                          \
    const unsigned char* __restrict__ sh_gate_scale,                           \
    float sh_gate_s2,                                                          \
    __nv_bfloat16* __restrict__ sh_gate_out,                                   \
    const unsigned char* __restrict__ sh_up_packed,                            \
    const unsigned char* __restrict__ sh_up_scale,                             \
    float sh_up_s2,                                                            \
    __nv_bfloat16* __restrict__ sh_up_out,                                     \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows

#define QX_GATE_UP_PASS                                                        \
    A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, gate_out,          \
    up_packed_ptrs, up_scale_ptrs, up_scale2_vals, up_out, expert_indices,     \
    sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_gate_out,                    \
    sh_up_packed, sh_up_scale, sh_up_s2, sh_up_out, N, K, top_k, rows

#define QX_SILU_DOWN_ARGS                                                      \
    const __nv_bfloat16* __restrict__ gate_out,                                \
    const __nv_bfloat16* __restrict__ up_out,                                  \
    const unsigned long long* __restrict__ packed_ptrs,                        \
    const unsigned long long* __restrict__ scale_ptrs,                         \
    const float* __restrict__ scale2_vals,                                     \
    __nv_bfloat16* __restrict__ C,                                             \
    const unsigned int* __restrict__ expert_indices,                           \
    const __nv_bfloat16* __restrict__ sh_gate_in,                              \
    const __nv_bfloat16* __restrict__ sh_up_in,                                \
    const unsigned char* __restrict__ sh_down_packed,                          \
    const unsigned char* __restrict__ sh_down_scale,                           \
    float sh_down_s2,                                                          \
    __nv_bfloat16* __restrict__ sh_down_out,                                   \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows

#define QX_SILU_DOWN_PASS                                                      \
    gate_out, up_out, packed_ptrs, scale_ptrs, scale2_vals, C, expert_indices, \
    sh_gate_in, sh_up_in, sh_down_packed, sh_down_scale, sh_down_s2,           \
    sh_down_out, N, K, top_k, rows

// Launch shape (crates/spark-model/src/layers/ops/qwen4exp_moe.rs mirrors
// it): 160-thread CTAs, V = 2, so 320 output columns a CTA; grid
// (N / 320, rows*top_k + 1, 2) for gate/up and (N / 320, rows*top_k + 1) for
// down; dynamic smem rows * K * 4 bytes.
extern "C" __global__ void __launch_bounds__(160) qwen4exp_moe_gate_up_t(QX_GATE_UP_ARGS) {
    qx_gate_up_t_impl<2, 4, 8>(QX_GATE_UP_PASS);
}
extern "C" __global__ void __launch_bounds__(160) qwen4exp_moe_silu_down_t(QX_SILU_DOWN_ARGS) {
    qx_silu_down_t_impl<2, 4, 8>(QX_SILU_DOWN_PASS);
}

// Sweep instantiations for scripts/dev/qwen4exp_moe_decode_bench.cu only.
#ifdef QX_SWEEP
#define QX_T_VARIANT(V_, D_, PF_)                                                                 \
    extern "C" __global__ void qx_sweep_gate_up_t_v##V_##_d##D_##_pf##PF_(QX_GATE_UP_ARGS) {     \
        qx_gate_up_t_impl<V_, D_, PF_>(QX_GATE_UP_PASS);                                          \
    }                                                                                              \
    extern "C" __global__ void qx_sweep_silu_down_t_v##V_##_d##D_##_pf##PF_(QX_SILU_DOWN_ARGS) { \
        qx_silu_down_t_impl<V_, D_, PF_>(QX_SILU_DOWN_PASS);                                      \
    }
QX_T_VARIANT(1, 4, 0)
QX_T_VARIANT(1, 4, 8)
QX_T_VARIANT(1, 4, 16)
QX_T_VARIANT(1, 8, 16)
QX_T_VARIANT(2, 4, 0)
QX_T_VARIANT(2, 4, 8)
QX_T_VARIANT(2, 4, 16)
QX_T_VARIANT(4, 2, 8)
QX_T_VARIANT(4, 4, 0)
QX_T_VARIANT(4, 4, 8)
QX_T_VARIANT(4, 4, 16)
#endif
