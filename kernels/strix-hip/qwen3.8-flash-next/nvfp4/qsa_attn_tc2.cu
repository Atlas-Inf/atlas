// SPDX-License-Identifier: AGPL-3.0-only
// QSA prefill attention on gfx1151 WMMA -- BOTH kv heads per CTA. HIP twin of
// kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_attn_tc2.cu: same entry name, same
// arguments, same launch geometry (grid [1, rows], block 128), so the existing
// `qsa_prefill_attn_tc2` launcher and `ATLAS_QSA_ATTN_TC2` gate drive it as-is.
//
// The body is the strix-hip `inferspark_prefill_paged_64` WMMA kernel
// (common/prefill_paged_compute.cuh: wave32 16x16x16 bf16 WMMA for Q.K^T, the
// smem-staged per-row online softmax, FP16 P.V) with only the addressing changed:
//
//   * The M tile is HEADS of one row, not query rows. Rows 0..gqa-1 are kv head
//     0's q-heads and rows 16..16+gqa-1 kv head 1's -- the split the inherited
//     warp mapping already makes at 16 -- because every q-head of a row attends
//     the IDENTICAL selected set (see the gb10 file for why rows are not tiled).
//   * Each BC=32 key tile is GATHERED through the row's selection list: token t
//     of the selected set lives at qsa_key_off(t). The two kv heads differ only
//     by `kvh * head_dim` within each cached token, so the per-token offsets are
//     computed once per tile into shared and both heads' loads reuse them.
//   * K and V are staged per kv head ([2][BC][HDIM_PAD], one buffer, K then V as
//     in the base kernel); the QK warp for half h and the PV warps for half h read
//     buffer h.
//   * No causal mask: every selected token is at or below the row's position by
//     construction. Only the tile tail (c >= tile_len) is masked.
//
// LDS: Q 16.5 KB + K/V 33 KB + S 4 KB + P 2 KB = ~56 KB (gfx1151 cap 64 KB).
//
// NOT BIT-IDENTICAL to `qsa_prefill_attn_g` (different summation tree, FP16 P):
// the same numerics class as gb10's tc2, whose gate is ppl + needles.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef __bf16 qtc_v16bf __attribute__((ext_vector_type(16)));
typedef __fp16 qtc_v16h  __attribute__((ext_vector_type(16)));
typedef float  qtc_v8f   __attribute__((ext_vector_type(8)));

#define QTC_BR 32
#define QTC_BC 32
#define QTC_HDIM 256
#define QTC_HDIM_PAD (QTC_HDIM + 2)          // row stride 129 dwords: distinct LDS banks
#define QTC_K16 16
#define QTC_QK_K_STEPS (QTC_HDIM / QTC_K16)   // 16
#define QTC_QK_N_TILES (QTC_BC / QTC_K16)     // 2
#define QTC_PV_K_STEPS (QTC_BC / QTC_K16)     // 2
#define QTC_PV_N_TILES ((QTC_HDIM / QTC_K16) / 2)  // 8 d-tiles per warp pair
#define QTC_CPR (QTC_HDIM / 8)               // 16-byte chunks per row (32)

__device__ __forceinline__ unsigned long long qsa_tc2_hip_key_off(
    unsigned int t, unsigned int topk_ratio, unsigned int ratio, bool ratio_p2,
    unsigned int ratio_sh, unsigned int ratio_mask, unsigned int complete,
    unsigned int block_size, bool bs_p2, unsigned int bs_sh, unsigned int bs_mask,
    const int* __restrict__ my_list, const int* __restrict__ block_table,
    unsigned long long page_stride, unsigned int row_elems)
{
    unsigned int tok;
    if (t < topk_ratio) {
        const unsigned int li = ratio_p2 ? (t >> ratio_sh) : (t / ratio);
        const unsigned int lo = ratio_p2 ? (t & ratio_mask) : (t % ratio);
        tok = (unsigned int)my_list[li] * ratio + lo;
    } else {
        tok = complete * ratio + (t - topk_ratio);
    }
    const unsigned int pg = bs_p2 ? (tok >> bs_sh) : (tok / block_size);
    const unsigned int inb = bs_p2 ? (tok & bs_mask) : (tok % block_size);
    return (unsigned long long)(unsigned int)block_table[pg] * page_stride
         + (unsigned long long)inb * row_elems;
}

extern "C" __global__ __launch_bounds__(128) void qsa_prefill_attn_tc2(
    const __nv_bfloat16* __restrict__ Q,          // [rows, nq, hd]
    const __nv_bfloat16* __restrict__ K,          // paged NHD
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,                // [rows, nq, hd]
    const int* __restrict__ block_table,
    const int* __restrict__ lists,                // [rows, topk]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const float inv_sqrt_d)
{
    const unsigned int r = blockIdx.y;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    if (num_kv_heads != 2u || gqa > 16u || head_dim != QTC_HDIM) return;

    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int seq_len = topk * ratio + tail;
    const int* my_list = lists + (size_t)r * topk;

    const unsigned int row_elems = num_kv_heads * head_dim;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const bool ratio_p2 = ratio != 0u && (ratio & (ratio - 1u)) == 0u;
    const bool bs_p2 = block_size != 0u && (block_size & (block_size - 1u)) == 0u;
    const unsigned int ratio_sh = ratio_p2 ? (unsigned int)(__ffs((int)ratio) - 1) : 0u;
    const unsigned int ratio_mask = ratio_p2 ? (ratio - 1u) : 0u;
    const unsigned int bs_sh = bs_p2 ? (unsigned int)(__ffs((int)block_size) - 1) : 0u;
    const unsigned int bs_mask = bs_p2 ? (block_size - 1u) : 0u;
    const unsigned int topk_ratio = topk * ratio;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int lane_lo = lane_id & 15;
    const unsigned int lane_hi = lane_id >> 4;
    const unsigned int q_seq_stride = num_q_heads * head_dim;

    __shared__ __nv_bfloat16 smem_Q[QTC_BR][QTC_HDIM_PAD];
    __shared__ __nv_bfloat16 smem_KV[2][QTC_BC][QTC_HDIM_PAD];   // [kv head][token][d]; K, then V
    __shared__ __half smem_P[QTC_BR][QTC_BC];
    __shared__ float smem_S[QTC_BR][QTC_BC];
    __shared__ float smem_ml[QTC_BR][2];
    __shared__ float smem_resc[QTC_BR];
    __shared__ unsigned long long smem_off[QTC_BC];             // per-token offset, kv head 0

    const unsigned int pv_warp_m  = (warp_id & 1) * 16;
    const unsigned int pv_n_start = (warp_id >> 1) * QTC_PV_N_TILES;
    const unsigned int pv_kvh = warp_id & 1;

    qtc_v8f acc_o[QTC_PV_N_TILES];
    #pragma unroll
    for (int i = 0; i < QTC_PV_N_TILES; i++) acc_o[i] = qtc_v8f{0,0,0,0,0,0,0,0};

    if (tid < QTC_BR) { smem_ml[tid][0] = -1e30f; smem_ml[tid][1] = 0.0f; }

    // Q: M-tile row `row` is head (row & 15) of kv head (row >> 4); rows past gqa are zero.
    for (unsigned int idx = tid; idx < QTC_BR * QTC_CPR; idx += 128) {
        const unsigned int row = idx / QTC_CPR, col = (idx % QTC_CPR) * 8;
        const unsigned int i = row & 15u, h = row >> 4;
        if (i < gqa) {
            *((uint4*)&smem_Q[row][col]) =
                *((const uint4*)&Q[(size_t)r * q_seq_stride + (size_t)(h * gqa + i) * head_dim + col]);
        } else {
            *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
        }
    }

    const unsigned int num_kv_blocks = (seq_len + QTC_BC - 1) / QTC_BC;
    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const unsigned int kv_start = kv_block * QTC_BC;
        const unsigned int tile_len = min(kv_start + QTC_BC, seq_len) - kv_start;

        if (tid < QTC_BC) {
            const unsigned int t = kv_start + tid;
            smem_off[tid] = (tid < tile_len)
                ? qsa_tc2_hip_key_off(t, topk_ratio, ratio, ratio_p2, ratio_sh, ratio_mask,
                                      complete, block_size, bs_p2, bs_sh, bs_mask,
                                      my_list, block_table, page_stride, row_elems)
                : 0ull;
        }
        __syncthreads();   // also orders the previous tile's PV reads before this K overwrite

        // K gather: [kvh][c][col], kv head h at +h*head_dim inside each token.
        for (unsigned int idx = tid; idx < 2u * QTC_BC * QTC_CPR; idx += 128) {
            const unsigned int h = idx / (QTC_BC * QTC_CPR);
            const unsigned int rem = idx % (QTC_BC * QTC_CPR);
            const unsigned int c = rem / QTC_CPR, col = (rem % QTC_CPR) * 8;
            *((uint4*)&smem_KV[h][c][col]) = (c < tile_len)
                ? *((const uint4*)&K[smem_off[c] + (unsigned long long)h * head_dim + col])
                : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // S = Q K^T: warp h computes M rows h*16..h*16+15 against kv head h's K.
        if (warp_id < 2) {
            const unsigned int qk_m = warp_id * 16;
            qtc_v8f acc_s[QTC_QK_N_TILES];
            #pragma unroll
            for (int n = 0; n < QTC_QK_N_TILES; n++) acc_s[n] = qtc_v8f{0,0,0,0,0,0,0,0};
            #pragma unroll
            for (unsigned int ks = 0; ks < QTC_QK_K_STEPS; ks++) {
                const unsigned int k_off = ks * QTC_K16;
                qtc_v16bf a;
                #pragma unroll
                for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_Q[qk_m + lane_lo][k_off + i];
                #pragma unroll
                for (int nt = 0; nt < QTC_QK_N_TILES; nt++) {
                    const unsigned int key_row = nt * 16 + lane_lo;
                    qtc_v16bf bb;
                    #pragma unroll
                    for (int k = 0; k < 16; k++) bb[k] = (__bf16)(float)smem_KV[warp_id][key_row][k_off + k];
                    acc_s[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, bb, acc_s[nt]);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < QTC_QK_N_TILES; nt++) {
                const unsigned int col = nt * 16 + lane_lo;
                #pragma unroll
                for (int e = 0; e < 8; e++) smem_S[qk_m + 2 * e + lane_hi][col] = acc_s[nt][e];
            }
        }
        __syncthreads();

        // Per-row online softmax (one thread per M row), exactly the base kernel's.
        if (tid < QTC_BR) {
            const unsigned int rr = tid;
            const bool row_valid = (rr & 15u) < gqa;
            float rmax = -1e30f;
            #pragma unroll
            for (unsigned int c = 0; c < QTC_BC; c++) {
                float s = smem_S[rr][c] * inv_sqrt_d;
                if (c >= tile_len || !row_valid) s = -1e30f;
                smem_S[rr][c] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = smem_ml[rr][0];
            const float l_old = smem_ml[rr][1];
            const float m_new = fmaxf(m_old, rmax);
            const float resc = __expf(m_old - m_new);
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int c = 0; c < QTC_BC; c++) {
                const float p = __expf(smem_S[rr][c] - m_new);
                smem_P[rr][c] = __float2half(p);
                sum += p;
            }
            smem_ml[rr][0] = m_new;
            smem_ml[rr][1] = l_old * resc + sum;
            smem_resc[rr] = resc;
        }

        // V gather into the same buffer (QK is done: the barrier above covers WAR).
        for (unsigned int idx = tid; idx < 2u * QTC_BC * QTC_CPR; idx += 128) {
            const unsigned int h = idx / (QTC_BC * QTC_CPR);
            const unsigned int rem = idx % (QTC_BC * QTC_CPR);
            const unsigned int c = rem / QTC_CPR, col = (rem % QTC_CPR) * 8;
            *((uint4*)&smem_KV[h][c][col]) = (c < tile_len)
                ? *((const uint4*)&V[smem_off[c] + (unsigned long long)h * head_dim + col])
                : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        {
            float resc_e[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) resc_e[e] = smem_resc[pv_warp_m + 2 * e + lane_hi];
            #pragma unroll
            for (int nt = 0; nt < QTC_PV_N_TILES; nt++)
                #pragma unroll
                for (int e = 0; e < 8; e++) acc_o[nt][e] *= resc_e[e];
        }
        #pragma unroll
        for (unsigned int ks = 0; ks < QTC_PV_K_STEPS; ks++) {
            const unsigned int k_off = ks * QTC_K16;
            qtc_v16h a;
            #pragma unroll
            for (int i = 0; i < 16; i++) a[i] = (__fp16)__half2float(smem_P[pv_warp_m + lane_lo][k_off + i]);
            #pragma unroll
            for (int nt = 0; nt < QTC_PV_N_TILES; nt++) {
                const unsigned int d_col = (pv_n_start + nt) * 16 + lane_lo;
                qtc_v16h bb;
                #pragma unroll
                for (int k = 0; k < 16; k++) bb[k] = (__fp16)(float)smem_KV[pv_kvh][k_off + k][d_col];
                acc_o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bb, acc_o[nt]);
            }
        }
        // The next iteration's first barrier orders these reads before the next K gather.
    }
    __syncthreads();

    #pragma unroll
    for (int nt = 0; nt < QTC_PV_N_TILES; nt++) {
        const unsigned int col = (pv_n_start + nt) * 16 + lane_lo;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const unsigned int row = pv_warp_m + 2 * e + lane_hi;
            const unsigned int i = row & 15u, h = row >> 4;
            if (i < gqa) {
                const float l = smem_ml[row][1];
                const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                O[(size_t)r * q_seq_stride + (size_t)(h * gqa + i) * head_dim + col] =
                    __float2bfloat16(acc_o[nt][e] * inv_l);
            }
        }
    }
}
