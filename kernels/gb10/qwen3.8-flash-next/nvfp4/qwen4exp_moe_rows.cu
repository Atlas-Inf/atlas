// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) routed + shared MoE GEMVs for several rows
// in ONE launch over the union of their experts, on the ORIGINALS expert
// layout ([N, K/2] packed, [N, K/16] scales)
// (ATLAS_QWEN4EXP_BATCH_FAST=1, `layers/moe/forward_rows.rs`).
//
// qwen4exp_moe_rows_gate_up / qwen4exp_moe_rows_silu_down replace, BIT FOR
// BIT, one launch per row of moe_expert_gate_up_shared /
// moe_expert_silu_down_shared (common/moe_shared_expert_fused.cu), the
// kernels single-sequence decode runs (`MoeLayer::forward`; TP1, and TP2's
// hybrid layout). For every output of every row the IEEE operation sequence
// is the single-row kernel's (the target builds with --fmad=false): lane l
// of the output's warp accumulates k16 = l, l + 32, ... < K/16 in order,
// each step's 8 packed bytes in order, `acc += a_lo * w_lo + a_hi * w_hi`
// with w = lut[nibble] * (dec_e4m3(scale) * s2); then the shfl_down
// 16/8/4/2/1 tree into lane 0 and one BF16 rounding. Remote (NULL) experts
// write zeros, as the single-row kernels do.
//
// `qwen4exp_moe_rows_plan` sorts the launch's (row, slot) entries by expert
// id and tags each UNIT -- up to QU_RMAX consecutive entries of one expert --
// at its first entry: order[rank] = q | count << 16 (count 0: not a head).
//
// gate/up: every CTA is the single-row kernel's CTA for one (row, slot) --
// the body is the same, statement for statement, only the row's activation
// and output pointers and the expert id come from the table -- launched in
// expert order (x, the output tile, fastest), so the rows that picked one
// expert, and all rows' shared expert, run back to back: the expert streams
// from DRAM once and the co-resident duplicates read it from L2. It runs at
// GB10's streaming bandwidth (~215 GB/s of unique expert bytes at 8 rows,
// 245 at one); a unit form of it (one CTA per expert tile, the tile staged in
// shared, rows looped) measured the same or slower, and an L2 look-ahead
// prefetch gained nothing (scripts/dev/qwen4exp_batch_exact_bench.cu).
//
// silu/down: the per-(row, slot) CTA spent most of its time outside the
// weight stream -- each CTA of 8 outputs computes the row's 640 SiLU
// activations (__expf, a divide) behind a barrier before its first weight
// load, then a lane has 1-2 eight-byte loads. Here one CTA serves a
// TILE-output tile of a UNIT: it issues the tile's weight bytes (contiguous,
// TILE * 360 B) as 16-byte cp.async, computes the unit rows' activations into
// shared meanwhile, and then each warp runs its outputs for every row of the
// unit against the shared tile, decoding each step's weights once for the
// rows of a chunk. Shared-expert units (QU_RMAX rows each) come first in the
// grid. 1.4-1.7x the per-(row, slot) form at 4-8 rows on GB10 (EP2 tables).
//
// (A one-CTA-per-expert form that keeps an accumulator pair per row in
// registers was built first and measured 0.50-0.85x of the per-row loop on
// GB10: the extra registers cost more occupancy than the shared reads saved.)
//
// Layouts (rows r < R, slot s < top_k, q = r * top_k + s):
//   A            [R, K]           BF16 activation rows
//   expert_ids   [R * top_k]      u32, each row's own top-k
//   order        [R * top_k]      u32 scratch, written by the plan kernel
//   gate/up_out  [R * top_k, N]   row q = row r's slot s (the per-row kernel's
//                                 [top_k, N] block of row r, stacked)
//   sh_*_out     [R, N]
//   down C       [R * top_k, N], sh_down_out [R, N]
//
// Grids: plan (1), (256); gate_up (ceil(N/8), R*top_k + R, 2), block 128;
// silu_down (N / QU_SD_TILE, ceil(R / QU_RMAX) + R*top_k, 1), block
// QU_SD_WARPS * 32, dynamic shared QU_SD_TILE * 360 + QU_SD_RC * 640 * 4
// bytes.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QR_BLOCK 128
#define QR_N_PER_BLOCK 4
#define QR_WARP 32
#define QR_GROUP 16

#ifndef QU_RMAX
#define QU_RMAX 8u
#endif
#define QU_SLOTS_MAX 1024u
#define QU_INTER 640u

__device__ __constant__ float QR_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// The replaced kernels' atlas_dec_e4m3 on NVIDIA: the verbatim cast.
__device__ __forceinline__ float qr_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}

// One block (any width). order[rank] = q | count << 16 over the entries
// ranked by (expert id, q); count = the rows of the unit headed at `rank`.
extern "C" __global__ void qwen4exp_moe_rows_plan(
    const unsigned int* __restrict__ expert_indices,
    unsigned int* __restrict__ order,
    unsigned int slots
) {
    __shared__ unsigned int s_idx[QU_SLOTS_MAX];
    __shared__ unsigned int s_q[QU_SLOTS_MAX];
    __shared__ unsigned int s_e[QU_SLOTS_MAX];
    if (slots > QU_SLOTS_MAX) __trap();
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) s_idx[t] = expert_indices[t];
    __syncthreads();
    for (unsigned t = threadIdx.x; t < slots; t += blockDim.x) {
        const unsigned e = s_idx[t];
        unsigned rank = 0;
        for (unsigned j = 0; j < slots; j++) {
            const unsigned f = s_idx[j];
            rank += (f < e) || (f == e && j < t);
        }
        s_q[rank] = t;
        s_e[rank] = e;
    }
    __syncthreads();
    for (unsigned p = threadIdx.x; p < slots; p += blockDim.x) {
        const unsigned e = s_e[p];
        unsigned start = p;
        while (start > 0 && s_e[start - 1] == e) start--;
        unsigned end = p + 1;
        while (end < slots && s_e[end] == e) end++;
        const unsigned count = (p - start) % QU_RMAX == 0 ? min(QU_RMAX, end - p) : 0u;
        order[p] = s_q[p] | (count << 16);
    }
}

extern "C" __global__ void qwen4exp_moe_rows_gate_up(
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
    const unsigned int* __restrict__ order,
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
    const unsigned slots = rows * top_k;
    const bool is_shared = blockIdx.y >= slots;
    const unsigned q = is_shared ? 0u : order[blockIdx.y] & 0xFFFFu;
    const unsigned row = is_shared ? blockIdx.y - slots : q / top_k;
    const unsigned proj = blockIdx.z;
    A += (unsigned long long)row * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    if (is_shared) {
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = ((proj == 0) ? sh_gate_out : sh_up_out) + (unsigned long long)row * N;
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) { B_packed = sh_gate_packed; B_scale = sh_gate_scale; s2 = sh_gate_s2; C = sh_gate_out; }
        else           { B_packed = sh_up_packed;   B_scale = sh_up_scale;   s2 = sh_up_s2;   C = sh_up_out; }
        C += (unsigned long long)row * N;
    } else {
        const unsigned expert_id = expert_indices[q];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        C += (unsigned long long)q * N;
        if (B_packed == 0) {
            const unsigned n_base = blockIdx.x * (QR_N_PER_BLOCK * 2);
            for (unsigned i = threadIdx.x; i < QR_N_PER_BLOCK * 2 && n_base + i < N; i += QR_BLOCK)
                C[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned threads_per_out = QR_BLOCK / QR_N_PER_BLOCK;
    const unsigned local_out = threadIdx.x / threads_per_out;
    const unsigned lane = threadIdx.x % threads_per_out;
    const unsigned n1 = blockIdx.x * (QR_N_PER_BLOCK * 2) + local_out * 2;
    const unsigned n2 = n1 + 1;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QR_E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    const unsigned half_K = K / 2;
    const unsigned num_groups = K / QR_GROUP;
    const unsigned K16 = K / 16;

    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        const unsigned sg = base_k / QR_GROUP;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = qr_dec_e4m3(sb1) * s2;
        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? qr_dec_e4m3(sb2) * s2 : 0.0f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }
#pragma unroll
    for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);
    if (have_n2) {
#pragma unroll
        for (int offset = QR_WARP / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// The CTA's unit: `n` rows (0: nothing to do). Routed: row i is entry
// q(i) = order[y0 + i] & 0xFFFF (input row q / top_k, output row q); shared:
// row i is token y0 + i for input and output. Shared-expert units come first
// in the grid (y < ceil(rows / QU_RMAX)): they carry the most rows.
struct QuUnit {
    unsigned n;
    bool shared;
    unsigned expert;
    unsigned y0;
    __device__ __forceinline__ unsigned q(const unsigned* order, unsigned i) const {
        return order[y0 + i] & 0xFFFFu;
    }
};

__device__ __forceinline__ QuUnit qu_resolve(
    const unsigned* __restrict__ expert_indices, const unsigned* __restrict__ order,
    unsigned rows
) {
    QuUnit u;
    const unsigned sh_units = (rows + QU_RMAX - 1) / QU_RMAX;
    const unsigned y = blockIdx.y;
    u.expert = 0;
    if (y < sh_units) {
        u.shared = true;
        u.y0 = y * QU_RMAX;
        u.n = min(QU_RMAX, rows - u.y0);
    } else {
        u.shared = false;
        u.y0 = y - sh_units;
        const unsigned ent = order[u.y0];
        u.n = ent >> 16;
        if (u.n) u.expert = expert_indices[ent & 0xFFFFu];
    }
    return u;
}

// ── Asynchronous tile staging ──
//
// A CTA's TILE output rows are contiguous in the [N, K/2] packed and
// [N, K/16] scale layouts. Every thread issues its share of both as 16-byte
// cp.async (L2 -> shared, no L1, no registers held), so the whole tile is in
// flight at once while the CTA computes its activation rows.
__device__ __forceinline__ void qu_cp16(void* dst, const void* src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src) : "memory");
}
// This thread's copies have landed (a __syncthreads publishes everyone's).
__device__ __forceinline__ void qu_cp_wait() {
    asm volatile("cp.async.commit_group;\n\tcp.async.wait_all;" ::: "memory");
}

// A warp reads one tile row at a time, its lanes 32 consecutive 8-byte words
// (and scale bytes): conflict-free at any row stride, so the tile is the
// global bytes as they are: [TILE][K/2] packed, then [TILE][K/16] scales.
template <unsigned K> struct QuTile {
    static constexpr unsigned ROW = K / 2, K16 = K / 16;
};

// Dynamic shared bytes of the weight tile (the Rust launcher mirrors this).
template <unsigned K, unsigned TILE>
__host__ __device__ constexpr unsigned qu_tile_bytes() {
    return TILE * (QuTile<K>::ROW + QuTile<K>::K16);
}

template <unsigned K, unsigned TILE>
__device__ __forceinline__ void qu_stage_tile(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale,
    unsigned n0, unsigned char* __restrict__ s_tile
) {
    typedef QuTile<K> T;
    static_assert(T::ROW % 16 == 0 && (TILE * T::K16) % 16 == 0, "16-byte tile rows");
    const unsigned char* gw = packed + (size_t)n0 * T::ROW;
    const unsigned char* gs = scale + (size_t)n0 * T::K16;
    unsigned char* s_s = s_tile + TILE * T::ROW;
    if ((((size_t)gw | (size_t)gs) & 15) == 0) {
        for (unsigned i = threadIdx.x; i < TILE * T::ROW / 16; i += blockDim.x)
            qu_cp16(s_tile + i * 16, gw + i * 16);
        for (unsigned i = threadIdx.x; i < TILE * T::K16 / 16; i += blockDim.x)
            qu_cp16(s_s + i * 16, gs + i * 16);
    } else {
        // Not expected (per-tensor allocations, 16-byte slab offsets); correct
        // for any alignment.
        for (unsigned i = threadIdx.x; i < TILE * T::ROW; i += blockDim.x) s_tile[i] = gw[i];
        for (unsigned i = threadIdx.x; i < TILE * T::K16; i += blockDim.x) s_s[i] = gs[i];
    }
}

template <unsigned OPW>
__device__ __forceinline__ void qu_store(__nv_bfloat16* __restrict__ dst, const float (&v)[OPW]) {
    if constexpr (OPW == 1) {
        *dst = __float2bfloat16(v[0]);
    } else {
        unsigned w[OPW / 2];
#pragma unroll
        for (unsigned o = 0; o < OPW; o += 2)
            w[o / 2] = (unsigned)__bfloat16_as_ushort(__float2bfloat16(v[o]))
                     | ((unsigned)__bfloat16_as_ushort(__float2bfloat16(v[o + 1])) << 16);
        if constexpr (OPW == 2) {
            *(unsigned*)dst = w[0];
        } else if constexpr (OPW == 4) {
            *(uint2*)dst = make_uint2(w[0], w[1]);
        } else {
#pragma unroll
            for (unsigned o = 0; o < OPW / 8; o++)
                ((uint4*)dst)[o] = make_uint4(w[4 * o], w[4 * o + 1], w[4 * o + 2], w[4 * o + 3]);
        }
    }
}

// R rows (R compile-time, so every accumulator stays in a register) x the
// warp's OPW consecutive tile rows. Lane l walks k16 = l, l + 32, ... < K/16
// -- the single-row kernel's lane chain -- as a rolled loop; per step each
// output's packed word and scale byte come from shared and are decoded once
// for the R rows (the same w = lut[nibble] * sc every row's chain uses), each
// row's 16 activations from `act(r, k16, f)`. Then the shfl_down
// 16/8/4/2/1 tree per (row, output); lane 0 stores row r's OPW outputs at
// out(r).
template <unsigned K, unsigned OPW, unsigned R, typename Act, typename Out>
__device__ __forceinline__ void qu_chunk(
    const unsigned char* __restrict__ s_w, const unsigned char* __restrict__ s_s,
    float s2, const float* __restrict__ s_lut, unsigned lane, Act&& act, Out&& out
) {
    typedef QuTile<K> T;
    float part[R][OPW];
#pragma unroll
    for (unsigned r = 0; r < R; r++) {
#pragma unroll
        for (unsigned o = 0; o < OPW; o++) part[r][o] = 0.0f;
    }
#pragma unroll 1
    for (unsigned k16 = lane; k16 < T::K16; k16 += 32) {
        float f[R][16];
#pragma unroll
        for (unsigned r = 0; r < R; r++) act(r, k16, f[r]);
#pragma unroll
        for (unsigned o = 0; o < OPW; o++) {
            const unsigned long long wq = *(const unsigned long long*)(s_w + o * T::ROW + k16 * 8);
            const float sc = qr_dec_e4m3(s_s[o * T::K16 + k16]) * s2;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                const unsigned char bv = (unsigned char)(wq >> (b * 8));
                const float wl = s_lut[bv & 0xF] * sc, wh = s_lut[bv >> 4] * sc;
#pragma unroll
                for (unsigned r = 0; r < R; r++) part[r][o] += f[r][2 * b] * wl + f[r][2 * b + 1] * wh;
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; r++) {
        float red[OPW];
#pragma unroll
        for (unsigned o = 0; o < OPW; o++) {
            float v = part[r][o];
#pragma unroll
            for (int off = QR_WARP / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
            red[o] = v;
        }
        if (lane == 0) qu_store<OPW>(out(r), red);
    }
}

// Rows c0 .. c0 + n - 1 (1 <= n <= RC) through `qu_chunk`.
template <unsigned K, unsigned OPW, unsigned RC, typename Act, typename Out>
__device__ __forceinline__ void qu_chunk_n(
    unsigned n, const unsigned char* s_w, const unsigned char* s_s, float s2,
    const float* s_lut, unsigned lane, Act&& act, Out&& out
) {
    static_assert(RC >= 1 && RC <= 8, "row chunk");
    if (n == 1) { qu_chunk<K, OPW, 1>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 2) if (n == 2) { qu_chunk<K, OPW, 2>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 3) if (n == 3) { qu_chunk<K, OPW, 3>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 4) if (n == 4) { qu_chunk<K, OPW, 4>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 5) if (n == 5) { qu_chunk<K, OPW, 5>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 6) if (n == 6) { qu_chunk<K, OPW, 6>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 7) if (n == 7) { qu_chunk<K, OPW, 7>(s_w, s_s, s2, s_lut, lane, act, out); return; }
    if constexpr (RC >= 8) qu_chunk<K, OPW, 8>(s_w, s_s, s2, s_lut, lane, act, out);
}

#define QU_SILU_DOWN_ARGS                                                      \
    const __nv_bfloat16* __restrict__ gate_out,                                \
    const __nv_bfloat16* __restrict__ up_out,                                  \
    const unsigned long long* __restrict__ packed_ptrs,                        \
    const unsigned long long* __restrict__ scale_ptrs,                         \
    const float* __restrict__ scale2_vals,                                     \
    __nv_bfloat16* __restrict__ C,                                             \
    const unsigned int* __restrict__ expert_indices,                           \
    const unsigned int* __restrict__ order,                                    \
    const __nv_bfloat16* __restrict__ sh_gate_in,                              \
    const __nv_bfloat16* __restrict__ sh_up_in,                                \
    const unsigned char* __restrict__ sh_down_packed,                          \
    const unsigned char* __restrict__ sh_down_scale,                           \
    float sh_down_s2,                                                          \
    __nv_bfloat16* __restrict__ sh_down_out,                                   \
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int rows
#define QU_SILU_DOWN_PASS                                                      \
    gate_out, up_out, packed_ptrs, scale_ptrs, scale2_vals, C, expert_indices, \
    order, sh_gate_in, sh_up_in, sh_down_packed, sh_down_scale, sh_down_s2,    \
    sh_down_out, N, K, top_k, rows

// silu/down, K = intermediate: grid (N / TILE, ceil(rows / QU_RMAX) +
// rows * top_k), block WARPS * 32, dynamic shared qu_tile_bytes<640, TILE>()
// + RC * 640 * 4. The activation rows of each chunk of RC unit rows are
// computed by the whole CTA into shared (as the single-row kernel stages its
// one row) while the tile copy is in flight.
template <unsigned TILE, unsigned WARPS, unsigned RC>
__device__ __forceinline__ void qu_silu_down(QU_SILU_DOWN_ARGS) {
    constexpr unsigned KK = QU_INTER;
    constexpr unsigned OPW = TILE / WARPS;
    static_assert(TILE % WARPS == 0, "tile rows per warp");
    if (K != KK || N % TILE != 0 || blockDim.x != WARPS * 32 || rows * top_k > QU_SLOTS_MAX)
        __trap();
    const QuUnit u = qu_resolve(expert_indices, order, rows);
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
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned n0 = blockIdx.x * TILE;
    // Routed gate/up rows are slot-major (row q), shared ones token-major:
    // input row == output row either way.
    auto row_of = [&](unsigned i) { return u.shared ? u.y0 + i : u.q(order, i); };
    if (packed == 0) {
        for (unsigned i = threadIdx.x; i < u.n * TILE; i += blockDim.x)
            out[(size_t)row_of(i / TILE) * N + n0 + i % TILE] = __float2bfloat16(0.0f);
        return;
    }
    extern __shared__ __align__(16) unsigned char qu_smem[];
    unsigned char* s_tile = qu_smem;
    float* s_act = (float*)(qu_smem + qu_tile_bytes<KK, TILE>());  // [RC][K]
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QR_E2M1_LUT[threadIdx.x];
    qu_stage_tile<KK, TILE>(packed, scale, n0, s_tile);
    const bool clamp = !u.shared;
    const unsigned char* w_w = s_tile + warp * OPW * QuTile<KK>::ROW;
    const unsigned char* w_s = s_tile + TILE * QuTile<KK>::ROW + warp * OPW * QuTile<KK>::K16;
    out += n0 + warp * OPW;
#pragma unroll 1
    for (unsigned c0 = 0; c0 < u.n; c0 += RC) {
        const unsigned n = min(RC, u.n - c0);
        if (c0) __syncthreads();  // the previous chunk's reads are done
        for (unsigned i = threadIdx.x; i < n * (KK / 8); i += blockDim.x) {
            const unsigned r = i / (KK / 8), c = i % (KK / 8);
            const size_t row = row_of(c0 + r);
            const uint4 gv = ((const uint4*)(g_base + row * KK))[c];
            const uint4 uv = ((const uint4*)(u_base + row * KK))[c];
            const unsigned gr[4] = {gv.x, gv.y, gv.z, gv.w};
            const unsigned ur[4] = {uv.x, uv.y, uv.z, uv.w};
            float a[8];
            // The single-row kernel's expression and clamp verbatim.
            const float SWIGLU_LIMIT = 10.0f;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                __nv_bfloat16 gb, ub;
                *(unsigned short*)&gb = (unsigned short)(gr[b / 2] >> (16 * (b & 1)));
                *(unsigned short*)&ub = (unsigned short)(ur[b / 2] >> (16 * (b & 1)));
                float gf = __bfloat162float(gb);
                float uf = __bfloat162float(ub);
                if (clamp) {
                    gf = fminf(gf, SWIGLU_LIMIT);
                    uf = fminf(fmaxf(uf, -SWIGLU_LIMIT), SWIGLU_LIMIT);
                }
                a[b] = (gf / (1.0f + __expf(-gf))) * uf;
            }
            float4* dst = (float4*)(s_act + r * KK) + 2 * c;
            dst[0] = make_float4(a[0], a[1], a[2], a[3]);
            dst[1] = make_float4(a[4], a[5], a[6], a[7]);
        }
        if (c0 == 0) qu_cp_wait();
        __syncthreads();
        qu_chunk_n<KK, OPW, RC>(n, w_w, w_s, s2, s_lut, lane,
            [&](unsigned r, unsigned k16, float (&f)[16]) {
                const float4* s = (const float4*)(s_act + r * KK + k16 * 16);
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float4 v = s[q];
                    f[4 * q] = v.x; f[4 * q + 1] = v.y; f[4 * q + 2] = v.z; f[4 * q + 3] = v.w;
                }
            },
            [&](unsigned r) { return out + (size_t)row_of(c0 + r) * N; });
    }
}

// Production shape (scripts/dev/qwen4exp_batch_exact_bench.cu `sweep`;
// crates/spark-model/src/layers/ops/qwen4exp_moe_rows.rs mirrors it): output
// rows per CTA, warps per CTA, unit rows per chunk.
#ifndef QU_SD_TILE
#define QU_SD_TILE 64
#endif
#ifndef QU_SD_WARPS
#define QU_SD_WARPS 8
#endif
#ifndef QU_SD_RC
#define QU_SD_RC 2
#endif

extern "C" __global__ void __launch_bounds__(QU_SD_WARPS * 32)
qwen4exp_moe_rows_silu_down(QU_SILU_DOWN_ARGS) {
    qu_silu_down<QU_SD_TILE, QU_SD_WARPS, QU_SD_RC>(QU_SILU_DOWN_PASS);
}

// Sweep shapes for scripts/dev/qwen4exp_batch_exact_bench.cu only (its
// SD_SHAPES table lists the same ones).
#ifdef QU_SWEEP
#define QU_SD_VARIANT(T_, W_, R_)                                              \
    extern "C" __global__ void __launch_bounds__(W_ * 32)                      \
    qu_sweep_silu_down_t##T_##_w##W_##_r##R_(QU_SILU_DOWN_ARGS) {              \
        qu_silu_down<T_, W_, R_>(QU_SILU_DOWN_PASS);                           \
    }
QU_SD_VARIANT(16, 4, 2)
QU_SD_VARIANT(32, 4, 2)
QU_SD_VARIANT(64, 4, 2)
QU_SD_VARIANT(32, 8, 1)
QU_SD_VARIANT(32, 8, 4)
QU_SD_VARIANT(32, 8, 2)
#endif
