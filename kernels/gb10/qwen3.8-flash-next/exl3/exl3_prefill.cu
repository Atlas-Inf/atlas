// SPDX-License-Identifier: AGPL-3.0-only
//
// Grouped packed-EXL3 MoE prefill (ATLAS_EXL3_PREFILL_GROUPED=1, off by default).
//
// The interim path runs every (token, selected expert) pair on its own, and
// CoopMK's prefill loops rows through its bsz-1 decode launch. Here one prefill
// chunk of one layer is routed once, its (token, slot) rows are sorted by
// expert, and each selected expert's trellis is decoded once per 64-row group
// of that expert's tokens, straight into tensor-core B fragments:
//
//   exl3_pf_gemv_rows       router logits for N rows. Body = common/
//                           dense_gemv_bf16.cu dense_gemv_bf16 (one row per
//                           blockIdx.y), so routing is bit-identical to the
//                           per-token paths.
//   exl3_pf_topk_rows       = common/moe_topk.cu moe_topk_softmax (the single-
//                           token kernel, lower-index-wins ties; NOT
//                           moe_topk_softmax_batched, which breaks ties
//                           differently), one token per block.
//   exl3_pf_gather_had      xh[r] = had_r_128(f16(x[row_src[r]]) * suh_e(r)):
//                           exl3_bf16_to_f16 + exl3_had_r128_pre of the
//                           interim, fused, with a row gather and the scale
//                           vector of the row's expert (pointer table).
//   exl3_pf_mma_k{2..8}     y = xh * W_inner for one (expert, <= 64 rows) item
//                           per blockIdx.y and 128 output columns per
//                           blockIdx.x. Each warp decodes its 16x16 trellis
//                           tile with the vendored dq_dispatch (the same
//                           decode exl3_reconstruct uses) into FragB, which is
//                           exactly the mma.m16n8k16 B operand (see
//                           reconstruct_tile.cuh's shuffle), and runs
//                           mma.sync against the rows staged in shared
//                           memory. No fp16 W_inner is written to global
//                           memory; fp32 accumulation, fp16 store (as the
//                           interim hgemm, different summation order).
//   exl3_pf_had_post_bf16   out[row_dst[r]] = bf16(had_r_128(y[r]) * svh_e(r)):
//                           exl3_had_r128_post + exl3_f16_to_bf16, fused, with
//                           a row scatter.
//
// Pointer tables hold one fp16 scale-vector address per local expert (routed
// experts, then the shared expert); row_exp[r] is the local expert of row r.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "exl3_vendor/hadamard_inner.cuh"
#include "exl3_vendor/exl3_dq.cuh"
#include "exl3_vendor/ptx_frag.cuh"

// ---------------------------------------------------------------------------
// Router logits for N rows. Body = dense_gemv_bf16 verbatim; only the per-row
// pointer offsets are added.
// Grid: (ceil(N / 4), rows, 1)  Block: (256, 1, 1)

#define PF_GEMV_BLOCK 256
#define PF_GEMV_N_PER_BLOCK 4
#define PF_WARP 32
#define PF_VEC 8

extern "C" __global__ void exl3_pf_gemv_rows(
    const __nv_bfloat16* __restrict__ A,  // [rows, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // [rows, c_stride]
    unsigned int N,
    unsigned int K,
    unsigned int c_stride)
{
    A += (unsigned long long) blockIdx.y * K;
    C += (unsigned long long) blockIdx.y * c_stride;

    const unsigned int threads_per_out = PF_GEMV_BLOCK / PF_GEMV_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * PF_GEMV_N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc = 0.0f;

    const unsigned int K_VEC = K / PF_VEC;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];

        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    {
        const unsigned int tail_start = K_VEC * PF_VEC;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            acc += __bfloat162float(A[k]) * __bfloat162float(B_row[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % PF_WARP;

    #pragma unroll
    for (int offset = PF_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[PF_GEMV_N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / PF_WARP);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ---------------------------------------------------------------------------
// Top-K softmax for N tokens. Body = moe_topk_softmax verbatim; only the
// per-token pointer offsets are added.
// Grid: (rows, 1, 1)  Block: (256, 1, 1)

#define PF_TOPK_BLOCK 256
#define PF_MAX_EXPERTS 512
#define PF_MAX_TOP_K 32

extern "C" __global__ void exl3_pf_topk_rows(
    const __nv_bfloat16* __restrict__ gate_logits,  // [rows, logit_stride]
    unsigned int* __restrict__ expert_indices,       // [rows, top_k]
    float* __restrict__ expert_weights,              // [rows, top_k]
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    unsigned int logit_stride)
{
    gate_logits += (unsigned long long) blockIdx.x * logit_stride;
    expert_indices += (unsigned long long) blockIdx.x * top_k;
    expert_weights += (unsigned long long) blockIdx.x * top_k;

    __shared__ float s_vals[PF_MAX_EXPERTS];
    __shared__ float s_top_vals[PF_MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[PF_MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = PF_TOPK_BLOCK / 32;

    unsigned int actual_n = num_experts < PF_MAX_EXPERTS ? num_experts : PF_MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += PF_TOPK_BLOCK) {
        s_vals[i] = __bfloat162float(gate_logits[i]);
    }
    for (unsigned int i = actual_n + tid; i < PF_MAX_EXPERTS; i += PF_TOPK_BLOCK) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += PF_TOPK_BLOCK) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val || (s_warp_val[w] == best_val && s_warp_idx[w] < best_idx)) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    float global_max = s_top_vals[0];

    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += PF_TOPK_BLOCK) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];

    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;

            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }

        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One 128-element Hadamard block held as one half4 per lane: the arithmetic of
// the vendored had_hf_r_128_inner (hadamard_inner.cuh), with the scale-vector
// chunk passed explicitly instead of read from blockIdx.y.

template <bool pre_scale, bool post_scale>
__device__ __forceinline__ half4 pf_had128(half4 v, const half* scale, int chunk, float r_scale, int t)
{
    if constexpr (pre_scale)
    {
        half4 s = ((const half4*) scale)[chunk * 32 + t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    if constexpr (post_scale)
    {
        half4 s = ((const half4*) scale)[chunk * 32 + t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }
    return v;
}

// ---------------------------------------------------------------------------
// out_z[r] = had_r_128(f16(src[row_src[r]]) * scale_z) for z = blockIdx.z in
// {0, 1} (gate and up share the gather, each has its own suh). scale_z =
// tab_z[row_exp[r]]. row_src NULL = identity. One warp per 128-column chunk.
// In place (src == out0, nsel 1) is safe: a lane reads its 4 elements before
// it writes the same 8 bytes.
// Grid: (rows, ceil(cols / 512), nsel)  Block: (32, 4)

extern "C" __global__ __launch_bounds__(128) void exl3_pf_gather_had(
    const __nv_bfloat16* src,
    const unsigned int* __restrict__ row_src,
    const unsigned int* __restrict__ row_exp,
    half* out0,
    half* out1,
    const unsigned long long* __restrict__ tab0,
    const unsigned long long* __restrict__ tab1,
    unsigned int cols,
    float r_scale)
{
    const unsigned int r = blockIdx.x;
    const unsigned int chunk = blockIdx.y * 4 + threadIdx.y;
    if (chunk * 128 >= cols) return;  // whole warp
    const unsigned int z = blockIdx.z;
    const unsigned int sr = row_src ? row_src[r] : r;
    const unsigned int e = row_exp[r];
    const half* scale = (const half*) (z ? tab1 : tab0)[e];
    const int t = threadIdx.x;

    const __nv_bfloat16* in = src + (unsigned long long) sr * cols + chunk * 128 + t * 4;
    half4 v;
    v.x = __halves2half2(__float2half_rn(__bfloat162float(in[0])), __float2half_rn(__bfloat162float(in[1])));
    v.y = __halves2half2(__float2half_rn(__bfloat162float(in[2])), __float2half_rn(__bfloat162float(in[3])));
    __syncwarp();

    v = pf_had128<true, false>(v, scale, chunk, r_scale, t);
    half* out = (z ? out1 : out0) + (unsigned long long) r * cols + chunk * 128;
    ((half4*) out)[t] = v;
}

// ---------------------------------------------------------------------------
// out_z[row_dst[r]] = bf16(had_r_128(y_z[r]) * scale_z), scale_z =
// tab_z[row_exp[r]]. row_dst NULL = identity; then out_z may equal y_z (a lane
// reads its half4 before it writes the same 8 bytes).
// Grid: (rows, ceil(cols / 512), nsel)  Block: (32, 4)

extern "C" __global__ __launch_bounds__(128) void exl3_pf_had_post_bf16(
    const half* y0,
    const half* y1,
    __nv_bfloat16* out0,
    __nv_bfloat16* out1,
    const unsigned int* __restrict__ row_dst,
    const unsigned int* __restrict__ row_exp,
    const unsigned long long* __restrict__ tab0,
    const unsigned long long* __restrict__ tab1,
    unsigned int cols,
    float r_scale)
{
    const unsigned int r = blockIdx.x;
    const unsigned int chunk = blockIdx.y * 4 + threadIdx.y;
    if (chunk * 128 >= cols) return;
    const unsigned int z = blockIdx.z;
    const unsigned int e = row_exp[r];
    const half* scale = (const half*) (z ? tab1 : tab0)[e];
    const int t = threadIdx.x;

    const half* y = (z ? y1 : y0) + (unsigned long long) r * cols + chunk * 128;
    half4 v = ((const half4*) y)[t];
    __syncwarp();
    v = pf_had128<false, true>(v, scale, chunk, r_scale, t);

    const unsigned int dr = row_dst ? row_dst[r] : r;
    __nv_bfloat16* o = (z ? out1 : out0) + (unsigned long long) dr * cols + chunk * 128 + t * 4;
    __nv_bfloat162 lo, hi;
    lo.x = __float2bfloat16_rn(__half2float(__low2half(v.x)));
    lo.y = __float2bfloat16_rn(__half2float(__high2half(v.x)));
    hi.x = __float2bfloat16_rn(__half2float(__low2half(v.y)));
    hi.y = __float2bfloat16_rn(__half2float(__high2half(v.y)));
    ((__nv_bfloat162*) o)[0] = lo;
    ((__nv_bfloat162*) o)[1] = hi;
}

// ---------------------------------------------------------------------------
// Fused decode + tensor-core GEMM for grouped experts.
//
// items[blockIdx.y] = { trellis, a, c, nrows <= 64 }: c[m, :] = a[m, :] *
// W_inner(trellis) for m < nrows, a row-major [nrows, lda] fp16 (already
// Hadamard-rotated), c row-major [nrows, ldc] fp16, W_inner [kt_count * 16,
// ntiles * 16]. blockIdx.x picks 128 output columns = 8 trellis tiles, one
// per warp. Per 16-row k slice each warp decodes its tile with dq_dispatch
// into FragB (B operand of mma.m16n8k16 for its two n8 halves) and multiplies
// up to four 16-row m-tiles staged in shared memory.
// Grid: (ntiles / 8, n_items, 1)  Block: 256

namespace exl3_pf {

struct Item
{
    unsigned long long trellis;
    unsigned long long a;
    unsigned long long c;
    unsigned int nrows;
    unsigned int pad;
};

constexpr int ROWS = 64;   // rows per item
constexpr int KCT = 8;     // k-tiles (of 16) staged per barrier pair
constexpr int AS_LD = 128 + 8;

__device__ __forceinline__ void mma_16816(float* c, const uint32_t* a, const FragB& b)
{
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&b.elems[0]);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&b.elems[1]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int K>
__device__ __forceinline__ void mma_item(const Item* __restrict__ items, int lda, int ldc, int ntiles, int kt_count)
{
    constexpr int TILE_U16 = 16 * K;                 // uint16 per 16x16 tile
    constexpr int TILE_U32 = 8 * K;
    constexpr int SLAB_INT4 = 8 * TILE_U16 * 2 / 16; // 8 tiles of one k slice

    __shared__ __align__(16) half As[ROWS][AS_LD];
    __shared__ __align__(16) uint32_t Ps[KCT][8][TILE_U32];

    const Item it = items[blockIdx.y];
    const int nrows = (int) it.nrows;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int nt0 = blockIdx.x * 8;
    const uint16_t* trellis = (const uint16_t*) it.trellis;
    const half* A = (const half*) it.a;
    half* C = (half*) it.c;
    const int mtiles = (nrows + 15) / 16;

    float acc[4][2][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int q = 0; q < 4; q++)
                acc[i][j][q] = 0.0f;

    for (int kc = 0; kc < kt_count; kc += KCT)
    {
        const int nk = min(KCT, kt_count - kc);
        __syncthreads();
        // Rows of this item, k columns [kc * 16, kc * 16 + nk * 16)
        for (int i = t; i < ROWS * 16; i += 256)
        {
            const int r = i >> 4;
            const int c = (i & 15) * 8;
            int4 v = make_int4(0, 0, 0, 0);
            if (r < nrows && c < nk * 16)
                v = *(const int4*) (A + (size_t) r * lda + kc * 16 + c);
            *(int4*) &As[r][c] = v;
        }
        // Packed tiles (kc + kk, nt0 .. nt0 + 7): contiguous per k slice
        for (int i = t; i < nk * SLAB_INT4; i += 256)
        {
            const int kk = i / SLAB_INT4;
            const int j = i - kk * SLAB_INT4;
            const int4* src = (const int4*) (trellis + ((size_t) (kc + kk) * ntiles + nt0) * TILE_U16);
            ((int4*) &Ps[kk][0][0])[j] = src[j];
        }
        __syncthreads();

        for (int kk = 0; kk < nk; kk++)
        {
            FragB f0, f1;
            dq_dispatch<K, 2, false>(Ps[kk][warp], lane * 8, f0, f1);
            #pragma unroll
            for (int mt = 0; mt < 4; mt++)
            {
                if (mt < mtiles)
                {
                    const int r = mt * 16 + (lane >> 2);
                    const int c = kk * 16 + (lane & 3) * 2;
                    uint32_t a[4];
                    a[0] = *(const uint32_t*) &As[r][c];
                    a[1] = *(const uint32_t*) &As[r + 8][c];
                    a[2] = *(const uint32_t*) &As[r][c + 8];
                    a[3] = *(const uint32_t*) &As[r + 8][c + 8];
                    mma_16816(acc[mt][0], a, f0);
                    mma_16816(acc[mt][1], a, f1);
                }
            }
        }
    }

    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
    {
        if (mt < mtiles)
        {
            #pragma unroll
            for (int nt = 0; nt < 2; nt++)
            {
                const int r = mt * 16 + (lane >> 2);
                const int col = (nt0 + warp) * 16 + nt * 8 + (lane & 3) * 2;
                if (r < nrows)
                    *(half2*) (C + (size_t) r * ldc + col) = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                if (r + 8 < nrows)
                    *(half2*) (C + (size_t) (r + 8) * ldc + col) = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
            }
        }
    }
}

}  // namespace exl3_pf

#define EXL3_PF_MMA(KB)                                                        \
    extern "C" __global__ __launch_bounds__(256) void exl3_pf_mma_k##KB(       \
        const exl3_pf::Item* __restrict__ items, int lda, int ldc, int ntiles, \
        int kt_count)                                                          \
    {                                                                          \
        exl3_pf::mma_item<KB>(items, lda, ldc, ntiles, kt_count);              \
    }

EXL3_PF_MMA(2)
EXL3_PF_MMA(3)
EXL3_PF_MMA(4)
EXL3_PF_MMA(5)
EXL3_PF_MMA(6)
EXL3_PF_MMA(7)
EXL3_PF_MMA(8)
