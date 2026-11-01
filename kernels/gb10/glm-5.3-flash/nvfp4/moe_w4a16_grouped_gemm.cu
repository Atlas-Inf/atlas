// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the deepseek-v4-flash `moe_w4a16_grouped_gemm.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../common/atlas_pdl.cuh"
#include "../../deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"

#include "silu_nvfp4_quant.cuh"

__device__ __forceinline__ void moe_cp_async_pred_4(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

// M=32 specialization for sparse routed-expert down projections.  GLM-5.3
// Flash assigns only ~28 rows/local expert on average for a 1K-token prefill,
// so the M=64 kernel spends close to half of each CTA on predicated rows.  Two
// warps cover exactly 32 rows while preserving the proven N=128/K=64 MMA and
// dequantization layout above.  This is a separate opt-in entry so other model
// families and decode remain on the established M=64 path.
__device__ __forceinline__ void moe_w4a16_grouped_gemm_ptrtable_t_k64_m32_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    constexpr unsigned int M_TILE_32 = 32;
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE_32;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned char* B_expert =
        (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert =
        (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_m32[2][M_TILE_32][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_m32[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_m32[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_m32[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_m32[16];
    __shared__ int smem_tok_m32[M_TILE_32];

    if (threadIdx.x < 16) smem_LUT_m32[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE_32) {
        const int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_m32[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_m32[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast_m32 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    // 64 threads: four rounds cover 32 A rows and all 32 packed-B K rows.
    #define M32_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 8 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_m32[row]; \
                moe_cp_async_pred_16(&smem_A_m32[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int kp_cur = rnd * 8 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&smem_Bp_m32[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    moe_cp_async_pred_16(&smem_Bs_m32[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // Each of 64 threads dequantizes two N columns to retain the exact M=64
    // B tile and MMA layout.
    #define M32_DEQUANT(buf) do { \
        _Pragma("unroll") \
        for (int nr = 0; nr < 2; nr++) { \
            unsigned int my_n = threadIdx.x + nr * 64; \
            float sv[K_STEP_T64 / GROUP_SIZE]; \
            _Pragma("unroll") \
            for (int g = 0; g < K_STEP_T64 / GROUP_SIZE; g++) \
                sv[g] = mx_block_scale<false>(smem_Bs_m32[(buf)][g][my_n], scale2); \
            _Pragma("unroll") \
            for (int kp = 0; kp < K_STEP_T64 / 2; kp++) { \
                float s = sv[kp / (GROUP_SIZE / 2)]; \
                unsigned char packed = smem_Bp_m32[(buf)][kp][my_n]; \
                float lo = smem_LUT_m32[packed & 0xF] * s; \
                float hi = smem_LUT_m32[packed >> 4] * s; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8_m32[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    #define M32_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_m32[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m32 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m32 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m32 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m32 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_m32[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_m32[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m32 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m32 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_m32 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_m32 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_m32[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_m32[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    M32_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    M32_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        M32_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        M32_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        M32_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M32_COMPUTE_MMA(cur);

    #undef M32_ISSUE_LOADS
    #undef M32_DEQUANT
    #undef M32_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t_k64_m32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K
) {
    moe_w4a16_grouped_gemm_ptrtable_t_k64_m32_impl(
        A, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K);
}

// Pre-quantized native-FP4 grouped GEMM for sparse MoE prefill.
//
// Unlike the older experimental W4A4 MoE arm, A is quantized exactly once by
// the common `quantize_bf16_to_nvfp4` kernel.  Every output-column CTA then
// loads the compact E2M1 values/scales instead of repeating BF16->FP4
// quantization.  B remains in Atlas's shared transposed pointer-table layout,
// so this adds no persistent weight copy.  Gate and up use two launches with
// the same pre-quantized A; down reuses the entry after quantizing SiLU output.
template<bool PQ_VEC_SCALES>
__device__ __forceinline__ void moe_w4a4_grouped_gemm_prequant_t_k64_impl(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    unsigned int work_expert_id,
    unsigned int work_m_tile,
    unsigned int work_n_tile
) {
    const unsigned int expert_id = work_expert_id;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = work_m_tile * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = work_n_tile * N_TILE_LG;
    const unsigned char* B_expert =
        (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert =
        (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is already compact FP4. Keep each row 16-byte aligned because the
    // loader issues 16-byte cp.async operations for both halves of the row.
    // (A 36-byte stride faults as CUDA_ERROR_MISALIGNED_ADDRESS on row 1.)
    __shared__ unsigned char smem_Ap_pq[2][M_TILE][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char smem_As_pq[2][M_TILE][K_STEP_T64 / GROUP_SIZE];
    // B checkpoint layout is [K/2,N]; load it coalesced, then transpose the
    // 32-byte K run required by the native block-scaled MMA.
    __shared__ unsigned char smem_BpT_pq[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bp_pq[N_TILE_LG][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char smem_Bs_pq[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG];
    __shared__ int smem_tok_pq[M_TILE];

    if (threadIdx.x < M_TILE) {
        const int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_pq[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_pq[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    #define PQ4_ISSUE_LOADS(buf, kb) do { \
        { \
            /* 128 threads x 16 B = 64 rows x 32 packed K bytes. */ \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int col = (threadIdx.x & 1) << 4; \
            bool valid = (cta_m_local + row) < M_eff && ((kb) + col * 2 + 31 < K); \
            unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
            moe_cp_async_pred_16(&smem_Ap_pq[(buf)][row][col], \
                &A_packed[(unsigned long long)a_row * (K / 2) + (kb) / 2 + col], \
                valid); \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* One aligned 4-byte transaction loads all four A scales/row. */ \
            if (threadIdx.x < M_TILE) { \
                unsigned int row = threadIdx.x; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
                moe_cp_async_pred_4(&smem_As_pq[(buf)][row][0], \
                    &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE], valid); \
            } \
        } else { \
            /* Conservative scalar scale loader. */ \
            _Pragma("unroll") \
            for (int job = 0; job < 2; job++) { \
                unsigned int jid = threadIdx.x + job * 128; \
                unsigned int row = jid >> 2; \
                unsigned int grp = jid & 3; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)smem_tok_pq[row]; \
                smem_As_pq[(buf)][row][grp] = valid \
                    ? A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE + grp] \
                    : 0; \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&smem_BpT_pq[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
            } \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* 32 aligned 16-byte transactions replace 512 scalar copies. */ \
            if (threadIdx.x < 32) { \
                unsigned int g = threadIdx.x >> 3; \
                unsigned int ns = (threadIdx.x & 7) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + g; \
                unsigned int gns = cta_n + ns; \
                moe_cp_async_pred_16(&smem_Bs_pq[(buf)][g][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N) && (sg < num_groups)); \
            } \
        } else { \
            unsigned int g = threadIdx.x >> 5; \
            unsigned int nn = threadIdx.x & 31; \
            unsigned int sg = (kb) / GROUP_SIZE + g; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int n_cur = rnd * 32 + nn; \
                unsigned int gns = cta_n + n_cur; \
                bool valid = (gns < N) && (sg < num_groups); \
                smem_Bs_pq[(buf)][g][n_cur] = valid \
                    ? S_expert[(unsigned long long)sg * N + gns] : 0; \
            } \
        } \
    } while(0)

    #define PQ4_TRANSPOSE(buf) do { \
        unsigned int my_n = threadIdx.x; \
        _Pragma("unroll") \
        for (int q = 0; q < (K_STEP_T64 / 2) / 4; q++) { \
            unsigned int w = (unsigned int)smem_BpT_pq[(buf)][q * 4 + 0][my_n] \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 1][my_n] << 8) \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 2][my_n] << 16) \
                | ((unsigned int)smem_BpT_pq[(buf)][q * 4 + 3][my_n] << 24); \
            *(unsigned int*)&smem_Bp_pq[my_n][q * 4] = w; \
        } \
    } while(0)

    #define PQ4_FRAG(P, ROW, KK) (*(const unsigned int*)&(P)[(ROW)][(KK) / 2])
    #define PQ4_COMPUTE_MMA(a_buf, b_buf) do { \
        unsigned int ra = warp_m_offset + group_id; \
        unsigned int a0 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra,     tid * 8); \
        unsigned int a1 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra + 8, tid * 8); \
        unsigned int a2 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra,     32 + tid * 8); \
        unsigned int a3 = PQ4_FRAG(smem_Ap_pq[(a_buf)], ra + 8, 32 + tid * 8); \
        unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2); \
        unsigned int sfa = (unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][0] \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][1] << 8) \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][2] << 16) \
            | ((unsigned int)smem_As_pq[(a_buf)][warp_m_offset + sfa_m][3] << 24); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = PQ4_FRAG(smem_Bp_pq, nc, tid * 8); \
            unsigned int b1 = PQ4_FRAG(smem_Bp_pq, nc, 32 + tid * 8); \
            unsigned int sfn = nt * 8 + (lane_id >> 2); \
            unsigned int sfb = (unsigned int)smem_Bs_pq[(b_buf)][0][sfn] \
                | ((unsigned int)smem_Bs_pq[(b_buf)][1][sfn] << 8) \
                | ((unsigned int)smem_Bs_pq[(b_buf)][2][sfn] << 16) \
                | ((unsigned int)smem_Bs_pq[(b_buf)][3][sfn] << 24); \
            unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]), \
                 "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_)); \
        } \
    } while(0)

    PQ4_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    PQ4_TRANSPOSE(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        PQ4_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        PQ4_COMPUTE_MMA(cur, cur);
        moe_cp_async_wait_all();
        __syncthreads();
        PQ4_TRANSPOSE(nxt);
        __syncthreads();
        cur = nxt;
    }
    PQ4_COMPUTE_MMA(cur, cur);

    #undef PQ4_ISSUE_LOADS
    #undef PQ4_TRANSPOSE
    #undef PQ4_FRAG
    #undef PQ4_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}

#define PQ4_PREQUANT_ARGS \
    const unsigned char* A_packed, const unsigned char* A_scale, \
    const unsigned long long* B_packed_ptrs, \
    const unsigned long long* B_scale_ptrs, const float* scale2_vals, \
    __nv_bfloat16* C, const int* expert_offsets, const int* sorted_token_ids, \
    unsigned int num_experts, unsigned int N, unsigned int K

#define PQ4_PREQUANT_CALL(VEC) \
    moe_w4a4_grouped_gemm_prequant_t_k64_impl<VEC>( \
        A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C, \
        expert_offsets, sorted_token_ids, num_experts, N, K, \
        blockIdx.z, blockIdx.y, blockIdx.x)

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64(
    PQ4_PREQUANT_ARGS
) {
    PQ4_PREQUANT_CALL(false);
}

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64_vecscale(
    PQ4_PREQUANT_ARGS
) {
    PQ4_PREQUANT_CALL(true);
}

// Compact-worklist variants of the exact same native-FP4 MMA.
// Each work item is (expert, m_tile, n_tile), packed by
// moe_build_tile_worklist. This removes thousands of empty expert CTAs for
// decode-sized verifier batches without changing tile arithmetic or stores.
template<bool PQ_VEC_SCALES>
__device__ __forceinline__ void moe_w4a4_grouped_gemm_prequant_compact_impl(
    PQ4_PREQUANT_ARGS,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles,
    unsigned int max_tiles
) {
    const int raw_total = *total_tiles;
    const unsigned int total = raw_total > 0
        ? min((unsigned int)raw_total, max_tiles) : 0u;
    const unsigned int wid = blockIdx.x;
    if (wid >= total) return;
    const unsigned int expert_id = worklist[wid * 2];
    const unsigned int packed = worklist[wid * 2 + 1];
    const unsigned int m_tile = packed >> 6;
    const unsigned int n_tile = packed & 0x3fu;
    if (expert_id >= num_experts || n_tile >= (N + N_TILE_LG - 1) / N_TILE_LG)
        return;
    moe_w4a4_grouped_gemm_prequant_t_k64_impl<PQ_VEC_SCALES>(
        A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K,
        expert_id, m_tile, n_tile);
}

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64_compact(
    PQ4_PREQUANT_ARGS,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles,
    unsigned int max_tiles
) {
    moe_w4a4_grouped_gemm_prequant_compact_impl<false>(
        A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K,
        worklist, total_tiles, max_tiles);
}

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact(
    PQ4_PREQUANT_ARGS,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles,
    unsigned int max_tiles
) {
    moe_w4a4_grouped_gemm_prequant_compact_impl<true>(
        A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K,
        worklist, total_tiles, max_tiles);
}

// ── Prequant native-FP4 grouped GEMM, K128 stages (prefill) ───────────────
// Same grid (N/128, m_tiles, experts), 64 x 128 tile, MMA sequence and stores
// as moe_w4a4_grouped_gemm_prequant_t_k64_vecscale, so outputs match bit for
// bit (every element accumulates the same k64 MMAs in the same order); only
// the data movement around the MMAs changes (moe_fp4_prefill_bench: 53 -> 65
// TFLOPS gate, 51 -> 59 down at a 4K-chunk EP2 routing):
//   * 256 threads: 8 warps of 16 rows x 64 columns (half the accumulators);
//   * K stages of 128 (two k64 MMA slices) halve the barriers per K, and the
//     next stage's cp.async overlaps this stage's transpose + MMAs;
//   * the [K/2, N] weight tile is transposed with 32-bit loads and a 4x4
//     byte PRMT transpose per thread instead of per-byte shared loads;
//   * A and B fragments come from ldmatrix.x4 (the m16n8k64 FP4 fragments
//     are exactly its 8x16-byte tiles), and B block scales are transposed
//     once per stage so each MMA reads its four scales with one load.
// Requires K % 128 == 0 and N % 128 == 0.
#define PQ2_KS 128                       // K per stage
#define PQ2_KP (PQ2_KS / 2)              // packed bytes per row per stage
#define PQ2_AP (PQ2_KP + 16)             // A row pitch (16-byte aligned)
#define PQ2_BP (PQ2_KP + 16)             // transposed B row pitch
#define PQ2_BTP (N_TILE_LG + 16)         // raw [kp][n] B row pitch

__device__ __forceinline__ void moe_cp_async_pred_8(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ unsigned int pq2_prmt(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

extern "C" __global__ void __launch_bounds__(256) moe_w4a4_grouped_gemm_prequant_t_k128(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    atlas_pdl_enter();
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const int m_start = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - m_start;
    if (M_expert <= 0) return;
    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    const float scale2 = scale2_vals[expert_id];
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;
    const unsigned int warp_m_offset = (warp_id & 3) * 16;
    const unsigned int warp_n_offset = (warp_id >> 2) * 64;
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;

    __shared__ __align__(16) unsigned char sA[2][M_TILE][PQ2_AP];
    __shared__ __align__(16) unsigned char sAs[2][M_TILE][PQ2_KS / GROUP_SIZE];
    __shared__ __align__(16) unsigned char sBraw[2][PQ2_KP][PQ2_BTP];
    __shared__ __align__(16) unsigned char sSraw[2][PQ2_KS / GROUP_SIZE][N_TILE_LG];
    __shared__ __align__(16) unsigned char sBt[N_TILE_LG][PQ2_BP];
    __shared__ __align__(16) unsigned char sSt[N_TILE_LG][PQ2_KS / GROUP_SIZE];
    __shared__ int sTok[M_TILE];

    if (t < M_TILE) {
        const bool live = (cta_m_local + (int)t) < M_expert;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    }
    __syncthreads();

    const unsigned int M_eff = (unsigned int)M_expert;
    auto issue = [&](int buf, unsigned int kb) {
        // A: 64 rows x 64 bytes = 256 x 16 B (two per thread).
        {
            const unsigned int j = t;
            const unsigned int row = j >> 2, col = (j & 3) << 4;
            const bool valid = (cta_m_local + row) < M_eff;
            const unsigned int a_row = (unsigned int)sTok[row];
            moe_cp_async_pred_16(&sA[buf][row][col],
                &A_packed[(unsigned long long)a_row * (K / 2) + kb / 2 + col], valid);
        }
        // A scales: 64 rows x 8 bytes.
        if (t < M_TILE) {
            const bool valid = (cta_m_local + t) < M_eff;
            const unsigned int a_row = (unsigned int)sTok[t];
            moe_cp_async_pred_8(&sAs[buf][t][0],
                &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) + kb / GROUP_SIZE], valid);
        }
        // B: 64 kp rows x 128 bytes = 512 x 16 B (four per thread).
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned int j = t + r * 256;
            const unsigned int kp = j >> 3, ns = (j & 7) << 4;
            moe_cp_async_pred_16(&sBraw[buf][kp][ns],
                &B_expert[(unsigned long long)(kb / 2 + kp) * N + cta_n + ns], true);
        }
        // B scales: 8 groups x 128 bytes = 64 x 16 B.
        if (t < 64) {
            const unsigned int g = t >> 3, ns = (t & 7) << 4;
            moe_cp_async_pred_16(&sSraw[buf][g][ns],
                &S_expert[(unsigned long long)(kb / GROUP_SIZE + g) * N + cta_n + ns], true);
        }
    };

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

    const unsigned int stages = K / PQ2_KS;
    issue(0, 0);
    moe_cp_async_commit();
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st & 1;
        moe_cp_async_wait_all();
        __syncthreads();   // stage `st` landed; every warp finished stage st-1
        if (st + 1 < stages) {
            issue(buf ^ 1, (st + 1) * PQ2_KS);
            moe_cp_async_commit();
        }
        // Transpose B: thread -> columns 4*cb..+3, kp 8*kb..+7. Each lane
        // rotates which of its four columns it stores first so an 8-lane
        // store phase hits eight distinct 16-byte bank slots (pitch 80).
        {
            const unsigned int cb = t & 31, kb8 = t >> 5;
            unsigned int col[4][2];
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const unsigned int kp0 = kb8 * 8 + q * 4;
                const unsigned int w0 = *(const unsigned int*)&sBraw[buf][kp0 + 0][cb * 4];
                const unsigned int w1 = *(const unsigned int*)&sBraw[buf][kp0 + 1][cb * 4];
                const unsigned int w2 = *(const unsigned int*)&sBraw[buf][kp0 + 2][cb * 4];
                const unsigned int w3 = *(const unsigned int*)&sBraw[buf][kp0 + 3][cb * 4];
                const unsigned int t0 = pq2_prmt(w0, w1, 0x5140), t1 = pq2_prmt(w2, w3, 0x5140);
                const unsigned int t2 = pq2_prmt(w0, w1, 0x7362), t3 = pq2_prmt(w2, w3, 0x7362);
                col[0][q] = pq2_prmt(t0, t1, 0x5410);
                col[1][q] = pq2_prmt(t0, t1, 0x7632);
                col[2][q] = pq2_prmt(t2, t3, 0x5410);
                col[3][q] = pq2_prmt(t2, t3, 0x7632);
            }
            const unsigned int rot = cb >> 1;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int cc = (c + rot) & 3;
                const unsigned int lo = cc == 0 ? col[0][0] : cc == 1 ? col[1][0] : cc == 2 ? col[2][0] : col[3][0];
                const unsigned int hi = cc == 0 ? col[0][1] : cc == 1 ? col[1][1] : cc == 2 ? col[2][1] : col[3][1];
                *(uint2*)&sBt[cb * 4 + cc][kb8 * 8] = make_uint2(lo, hi);
            }
            // Scales: threads 0..127 -> column t, 8 groups.
            if (t < N_TILE_LG) {
                unsigned int lo = 0, hi = 0;
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    lo |= (unsigned int)sSraw[buf][g][t] << (8 * g);
                    hi |= (unsigned int)sSraw[buf][g + 4][t] << (8 * g);
                }
                *(uint2*)&sSt[t][0] = make_uint2(lo, hi);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int sl = 0; sl < 2; ++sl) {
            const unsigned int ko = sl * 32;   // packed-byte offset of this k64 slice
            // A fragment via one ldmatrix.x4: matrices (rows 0-7 | 8-15) x
            // (bytes 0-15 | 16-31) give exactly a0..a3 of the m16n8k64 layout.
            unsigned int a0, a1, a2, a3;
            {
                const unsigned int j = lane_id >> 3, r = lane_id & 7;
                const unsigned int addr = __cvta_generic_to_shared(
                    &sA[buf][warp_m_offset + r + (j & 1) * 8][ko + (j >> 1) * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
            }
            const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
            const unsigned int sfa = *(const unsigned int*)&sAs[buf][warp_m_offset + sfa_m][sl * 4];
            #pragma unroll
            for (int np = 0; np < 4; np++) {
                // B fragments of n-subtiles 2np and 2np+1 via one ldmatrix.x4.
                unsigned int b[4];
                {
                    const unsigned int j = lane_id >> 3, r = lane_id & 7;
                    const unsigned int addr = __cvta_generic_to_shared(
                        &sBt[warp_n_offset + np * 16 + (j >> 1) * 8 + r][ko + (j & 1) * 16]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int nt = np * 2 + h;
                    const unsigned int b0 = b[h * 2], b1 = b[h * 2 + 1];
                    const unsigned int sfb = *(const unsigned int*)&sSt[warp_n_offset + nt * 8 + (lane_id >> 2)][sl * 4];
                    unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};"
                        :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]),
                         "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_));
                }
            }
        }
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2, c1 = c0 + 1;
        const unsigned int r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        const bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        const bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}

// Per-expert M64-tile prefix for the K128W grid below: prefix[e] = number of
// row tiles of the local experts before e (remote experts, NULL weights, have
// none); prefix[num_experts] = the total. One block; num_experts <= 1024.
extern "C" __global__ void __launch_bounds__(1024) moe_mtile_prefix(
    const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_packed_ptrs,
    int* __restrict__ prefix,
    unsigned int num_experts
) {
    __shared__ int s[1024];
    const unsigned int t = threadIdx.x;
    int v = 0;
    if (t < num_experts && B_packed_ptrs[t] != 0) {
        const int rows = expert_offsets[t + 1] - expert_offsets[t];
        v = rows > 0 ? (rows + M_TILE - 1) / M_TILE : 0;
    }
    s[t] = v;
    __syncthreads();
    for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
        const int add = t >= d ? s[t - d] : 0;
        __syncthreads();
        s[t] += add;
        __syncthreads();
    }
    if (t < num_experts) prefix[t + 1] = s[t];
    if (t == 0) prefix[0] = 0;
}

// ── Prequant native-FP4 grouped GEMM, 64 x 256 tiles (prefill) ───────────
// The MMA sequence of moe_w4a4_grouped_gemm_prequant_t_k128 per output
// element (same m16n8k64 operands and scales, k64 slices in K order), so the
// outputs match it bit for bit, with 40% of its instructions (ncu: 153M vs
// 375M on a 4K-chunk gate; at ~97% of GB10's measured read+write DRAM
// bandwidth in moe_fp4_prefill_bench):
//   * 8 warps of 32 rows x 64 columns: each A/B fragment feeds two MMAs;
//   * the [K/2, N] weight tile needs no transpose pass: ldmatrix.m16n16.trans
//     .b8 reads a 16 kp x 16 n byte block straight into the b0/b1 fragments
//     of two n8 subtiles, and the same load over the [K/16, N] scale rows
//     leaves slice s's four scales in quad thread s (the MMA's B scale
//     thread selector), so shared memory holds only the raw tiles;
//   * weight and scale rows are XOR-swizzled in 16-byte chunks (no padding);
//   * CTAs cover only the local experts' row tiles: grid (N/256, bound, 1)
//     with bound >= prefix[num_experts] (moe_mtile_prefix), each CTA y a
//     global row tile, so the dense grid's empty expert CTAs never launch.
// Two stages at two CTAs/SM measured faster than 3-4 stages at one CTA/SM
// and than 64 x 128 tiles with 3 stages. Requires K % 128 == 0, N % 256 == 0.
#define PQ2_ARGS \
    const unsigned char* __restrict__ A_packed, \
    const unsigned char* __restrict__ A_scale, \
    const unsigned long long* __restrict__ B_packed_ptrs, \
    const unsigned long long* __restrict__ B_scale_ptrs, \
    const float* __restrict__ scale2_vals, \
    __nv_bfloat16* __restrict__ C, \
    const int* __restrict__ expert_offsets, \
    const int* __restrict__ sorted_token_ids, \
    unsigned int num_experts, unsigned int N, unsigned int K

__device__ __forceinline__ void pqw_ldsm_t2(unsigned int (&d)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}

__device__ __forceinline__ void pqw_ldsm_t1(unsigned int (&d)[2], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(addr));
}

// The up projection and NVFP4 outputs of the fused gate/up variant.
struct PqwGateUp {
    const unsigned long long* __restrict__ packed_ptrs;
    const unsigned long long* __restrict__ scale_ptrs;
    const float* __restrict__ scale2_vals;
    unsigned char* __restrict__ out_packed;   // [rows, N/2] E2M1
    unsigned char* __restrict__ out_scale;    // [rows, N/16] E4M3
};

// One K128W tile is 64 rows x 256 B columns (GATE_UP: 128 gate columns and
// the same 128 up columns, with silu_mul_quant_nvfp4 applied in registers),
// computed by 256 threads in 8 warps of 32 rows x 64 columns. The pieces
// below are shared by the one-tile-per-CTA grid (pqw_impl) and the
// persistent grid (pqp_impl).
#define PQW_NT 256
#define PQW_NB (PQW_NT / 4 / 16)   // 16-column ldmatrix blocks per warp
// [kp][n] and [group][n] shared tiles: 16-byte chunk c of row r at c ^ (r & 7).
typedef unsigned char PqwA[M_TILE][PQ2_AP];
typedef unsigned char PqwAs[M_TILE][PQ2_KS / GROUP_SIZE];
typedef unsigned char PqwB[PQ2_KP][PQW_NT];
typedef unsigned char PqwS[PQ2_KS / GROUP_SIZE][PQW_NT];
typedef float PqwAcc[2][2 * PQW_NB][4];

// The local expert owning global row tile `tile` of mtile_prefix.
__device__ __forceinline__ unsigned int pqw_tile_expert(
    int tile, const int* __restrict__ mtile_prefix, unsigned int num_experts
) {
    unsigned int lo = 0, hi = num_experts;
    while (hi - lo > 1) {
        const unsigned int mid = (lo + hi) >> 1;
        if (mtile_prefix[mid] <= tile) lo = mid; else hi = mid;
    }
    return lo;
}

// Threads t < M_TILE: the source row of the tile's row t (its sorted token,
// or the row itself).
__device__ __forceinline__ void pqw_gather_rows(
    int* sTok, unsigned int t, const int* __restrict__ sorted_token_ids,
    unsigned int cta_m, int cta_m_local, int M_expert
) {
    if (t < M_TILE) {
        const bool live = (cta_m_local + (int)t) < M_expert;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    }
}

__device__ __forceinline__ void pqw_zero_acc(PqwAcc& acc) {
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int i = 0; i < 2 * PQW_NB; i++) acc[mi][i][0] = acc[mi][i][1] = acc[mi][i][2] = acc[mi][i][3] = 0.0f;
}

// Tile 16-byte column chunk c: whether it reads the up matrix, and its column.
template<bool GATE_UP>
__device__ __forceinline__ bool pqw_chunk_up(unsigned int c) { return GATE_UP && c >= PQW_NT / 32; }
template<bool GATE_UP>
__device__ __forceinline__ unsigned int pqw_chunk_col(unsigned int cta_n, unsigned int c) {
    return cta_n + ((GATE_UP ? c % (PQW_NT / 32) : c) << 4);
}

// Thread t's cp.async loads of K stage kb into one set of shared tiles.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_issue(
    unsigned int t, PqwA& sA, PqwAs& sAs, PqwB& sB, PqwS& sS, const int* sTok,
    const unsigned char* __restrict__ A_packed, const unsigned char* __restrict__ A_scale,
    const unsigned char* B_expert, const unsigned char* S_expert,
    const unsigned char* U_expert, const unsigned char* US_expert,
    int cta_m_local, unsigned int M_eff, unsigned int cta_n, unsigned int N, unsigned int K, unsigned int kb
) {
    {
        const unsigned int row = t >> 2, col = (t & 3) << 4;
        const bool valid = (cta_m_local + row) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[row];
        moe_cp_async_pred_16(&sA[row][col],
            &A_packed[(unsigned long long)a_row * (K / 2) + kb / 2 + col], valid);
    }
    if (t < M_TILE) {
        const bool valid = (cta_m_local + t) < M_eff;
        const unsigned int a_row = (unsigned int)sTok[t];
        moe_cp_async_pred_8(&sAs[t][0],
            &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) + kb / GROUP_SIZE], valid);
    }
    // B: 64 kp rows x NT bytes, NT/64 16-byte chunks per thread.
    #pragma unroll
    for (int r = 0; r < PQW_NT / 64; ++r) {
        const unsigned int j = t + r * 256;
        const unsigned int kp = j / (PQW_NT / 16), c = j % (PQW_NT / 16);
        moe_cp_async_pred_16(&sB[kp][(c ^ (kp & 7)) << 4],
            &(pqw_chunk_up<GATE_UP>(c) ? U_expert : B_expert)[(unsigned long long)(kb / 2 + kp) * N + pqw_chunk_col<GATE_UP>(cta_n, c)], true);
    }
    // B scales: 8 groups x NT bytes.
    if (t < PQW_NT / 2) {
        const unsigned int g = t / (PQW_NT / 16), c = t % (PQW_NT / 16);
        moe_cp_async_pred_16(&sS[g][(c ^ g) << 4],
            &(pqw_chunk_up<GATE_UP>(c) ? US_expert : S_expert)[(unsigned long long)(kb / GROUP_SIZE + g) * N + pqw_chunk_col<GATE_UP>(cta_n, c)], true);
    }
}

// Warp warp_id's MMAs of one K128 stage held in shared memory.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_mma_stage(
    PqwAcc& acc, const PqwA& sA, const PqwAs& sAs, const PqwB& sB, const PqwS& sS,
    unsigned int warp_id, unsigned int lane_id
) {
    constexpr int NB = PQW_NB;
    const unsigned int warp_m_offset = (warp_id & 1) * 32;
    const unsigned int warp_n_offset = (warp_id >> 1) * (PQW_NT / 4);
    // This warp's nb-th chunk (GATE_UP: gate chunks 2wn, 2wn+1, then up's).
    auto warp_chunk = [&](int nb) -> unsigned int {
        return GATE_UP ? (nb >> 1) * (PQW_NT / 32) + (warp_id >> 1) * 2 + (nb & 1)
                       : (warp_n_offset >> 4) + nb;
    };
    // Per-lane ldmatrix row: B (x2) rows kp = lane (matrix 0: k bytes 0-15 of
    // a k64 slice, matrix 1: bytes 16-31); scales (x1) rows g = lane & 7.
    const unsigned int kp_l = lane_id, g_l = lane_id & 7;
    const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
    const unsigned short tid_a = 0, bid_a = 0, bid_b = 0;
    // B scales of this stage: quad thread s holds slice s's four for
    // column group_id of each n8 subtile.
    unsigned int sfb[2 * NB];
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const unsigned int c = warp_chunk(nb);
        unsigned int d[2];
        pqw_ldsm_t1(d, __cvta_generic_to_shared(&sS[g_l][(c ^ g_l) << 4]));
        sfb[nb * 2] = d[0];
        sfb[nb * 2 + 1] = d[1];
    }
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const unsigned int ko = sl * 32;
        const unsigned short tid_b = sl;
        unsigned int a[2][4], sfa[2];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const unsigned int j = lane_id >> 3, r = lane_id & 7;
            const unsigned int addr = __cvta_generic_to_shared(
                &sA[warp_m_offset + mi * 16 + r + (j & 1) * 8][ko + (j >> 1) * 16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
            sfa[mi] = *(const unsigned int*)&sAs[warp_m_offset + mi * 16 + sfa_m][sl * 4];
        }
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const unsigned int c = warp_chunk(nb);
            const unsigned int kp = ko + kp_l;
            unsigned int b[4];
            pqw_ldsm_t2(b, __cvta_generic_to_shared(&sB[kp][(c ^ (kp & 7)) << 4]));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = nb * 2 + h;
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};"
                        :"=f"(acc[mi][nt][0]),"=f"(acc[mi][nt][1]),"=f"(acc[mi][nt][2]),"=f"(acc[mi][nt][3])
                        :"r"(a[mi][0]),"r"(a[mi][1]),"r"(a[mi][2]),"r"(a[mi][3]),"r"(b[h]),"r"(b[2 + h]),
                         "f"(acc[mi][nt][0]),"f"(acc[mi][nt][1]),"f"(acc[mi][nt][2]),"f"(acc[mi][nt][3]),
                         "r"(sfa[mi]),"h"(bid_a),"h"(tid_a),"r"(sfb[nt]),"h"(bid_b),"h"(tid_b));
                }
            }
        }
    }
}

// Warp warp_id's stores of a finished tile: BF16 C, or (GATE_UP) the packed
// NVFP4 SiLU·mul outputs.
template<bool GATE_UP>
__device__ __forceinline__ void pqw_epilogue(
    const PqwAcc& acc, unsigned int warp_id, unsigned int lane_id,
    unsigned int expert_id, float scale2, unsigned int cta_m, int cta_m_local, int M_expert,
    unsigned int cta_n, unsigned int N, __nv_bfloat16* __restrict__ C, const PqwGateUp& up
) {
    constexpr int NB = PQW_NB;
    const unsigned int warp_m_offset = (warp_id & 1) * 32;
    const unsigned int warp_n_offset = (warp_id >> 1) * (PQW_NT / 4);
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    if constexpr (GATE_UP) {
        // Gate subtile j pairs with up subtile j + 4 (same columns); each
        // 16-column group spans two subtiles, i.e. one quad of threads.
        const float scale2_up = up.scale2_vals[expert_id];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const unsigned int rl = warp_m_offset + mi * 16 + group_id;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const unsigned int row = cta_m + rl + half * 8;
                const bool valid = (int)(rl + half * 8 + cta_m_local) < M_expert;
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    float v[4], group_max = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int j = q * 2 + (i >> 1), e = half * 2 + (i & 1);
                        v[i] = silu_nvfp4_act(
                            __bfloat162float(__float2bfloat16(acc[mi][j][e] * scale2)),
                            __bfloat162float(__float2bfloat16(acc[mi][j + 4][e] * scale2_up)));
                        group_max = fmaxf(group_max, fabsf(v[i]));
                    }
                    group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 1));
                    group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 2));
                    float inv;
                    const unsigned char sc = silu_nvfp4_group_scale(group_max, &inv);
                    const unsigned int col = cta_n + (warp_id >> 1) * 32 + q * 16;
                    if (!valid) continue;
                    if (tid == 0) up.out_scale[(unsigned long long)row * (N / 16) + col / 16] = sc;
                    #pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const unsigned int q0 = silu_nvfp4_quantize_e2m1(v[hh * 2] * inv);
                        const unsigned int q1 = silu_nvfp4_quantize_e2m1(v[hh * 2 + 1] * inv);
                        up.out_packed[(unsigned long long)row * (N / 2) + (col + hh * 8) / 2 + tid] =
                            (unsigned char)((q1 << 4) | q0);
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            const unsigned int rl = warp_m_offset + mi * 16 + group_id;
            const bool r0v = (int)(rl + cta_m_local) < M_expert;
            const bool r1v = (int)(rl + 8 + cta_m_local) < M_expert;
            const unsigned int r0 = cta_m + rl, r1 = r0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2 * NB; nt++) {
                const unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2;
                if (r0v)
                    *(__nv_bfloat162*)&C[r0 * N + c0] = __floats2bfloat162_rn(
                        acc[mi][nt][0] * scale2, acc[mi][nt][1] * scale2);
                if (r1v)
                    *(__nv_bfloat162*)&C[r1 * N + c0] = __floats2bfloat162_rn(
                        acc[mi][nt][2] * scale2, acc[mi][nt][3] * scale2);
            }
        }
    }
}

template<bool GATE_UP>
__device__ __forceinline__ void pqw_impl(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const PqwGateUp up
) {
    constexpr int STAGES = 2;
    const int tile = blockIdx.y;
    if (tile >= mtile_prefix[num_experts]) return;
    const unsigned int expert_id = pqw_tile_expert(tile, mtile_prefix, num_experts);
    const int m_start = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - m_start;
    const int cta_m_local = (tile - mtile_prefix[expert_id]) * M_TILE;
    if (cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    const float scale2 = scale2_vals[expert_id];
    const unsigned char* U_expert = GATE_UP ? (const unsigned char*)up.packed_ptrs[expert_id] : nullptr;
    const unsigned char* US_expert = GATE_UP ? (const unsigned char*)up.scale_ptrs[expert_id] : nullptr;
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? PQW_NT / 2 : PQW_NT);

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqwA sA[STAGES];
    __shared__ __align__(16) PqwAs sAs[STAGES];
    __shared__ __align__(16) PqwB sB[STAGES];
    __shared__ __align__(16) PqwS sS[STAGES];
    __shared__ int sTok[M_TILE];

    pqw_gather_rows(sTok, t, sorted_token_ids, cta_m, cta_m_local, M_expert);
    __syncthreads();

    const unsigned int M_eff = (unsigned int)M_expert;
    auto issue = [&](int buf, unsigned int kb) {
        pqw_issue<GATE_UP>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            B_expert, S_expert, U_expert, US_expert, cta_m_local, M_eff, cta_n, N, K, kb);
    };

    PqwAcc acc;
    pqw_zero_acc(acc);

    const unsigned int stages = K / PQ2_KS;
    #pragma unroll
    for (int s0 = 0; s0 < STAGES - 1; ++s0) {
        if (s0 < stages) issue(s0, s0 * PQ2_KS);
        moe_cp_async_commit();
    }
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st % STAGES;
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        __syncthreads();   // stage `st` landed; every warp finished stage st-1
        const unsigned int nxt = st + STAGES - 1;
        if (nxt < stages) issue(nxt % STAGES, nxt * PQ2_KS);
        moe_cp_async_commit();
        pqw_mma_stage<GATE_UP>(acc, sA[buf], sAs[buf], sB[buf], sS[buf], warp_id, lane_id);
    }
    pqw_epilogue<GATE_UP>(acc, warp_id, lane_id, expert_id, scale2, cta_m, cta_m_local, M_expert,
        cta_n, N, C, up);
}

extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_t_k128w_compact(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix
) {
    pqw_impl<false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, PqwGateUp{});
}

// Gate (B_*) and up projections of the same 128 intermediate columns per CTA,
// grid (N/128, bound, 1), with silu_mul_quant_nvfp4 applied in the epilogue:
// writes its packed E2M1 [rows, N/2] + E4M3 [rows, N/16] bytes for the local
// experts' rows exactly (C unused), skipping both BF16 intermediates.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed,
    unsigned char* __restrict__ out_scale
) {
    pqw_impl<true>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix,
        PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale});
}

// ── Persistent K128W (ATLAS_GLM_MOE_PREFILL_PERSIST) ─────────────────────
// The K128W tiles, loads, MMA sequence and epilogue above, so the output
// bytes are identical; only the scheduling changes. Two CTAs per SM take
// (row tile, N tile) work items w = n + N_tiles * row, in order, from the
// zeroed counter `next_work`, which keeps the resident tiles as close
// together as the K128W grid's in-order dispatch does (the row tiles of one
// expert share its weights in L2; a static stride lets the CTAs drift apart
// and re-read them from DRAM). Nothing launches past the live tiles (the
// K128W grid launches the host's bound, most of it exiting at once), and
// each CTA gathers its next tile's token rows and issues its first K stage
// before the current tile's epilogue. Every tile below
// mtile_prefix[num_experts] holds rows of a local expert. An item whose
// weight table entry is null (the prefix came from another table) loads
// and stores nothing, as the grid kernel's CTA returns, but still passes
// the K loop's barriers with the CTA, its MMAs reading stale shared memory.
template<bool GATE_UP>
__device__ __forceinline__ void pqp_impl(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    int* __restrict__ next_work,
    const PqwGateUp up
) {
    const unsigned int n_tiles = N / (GATE_UP ? PQW_NT / 2 : PQW_NT);
    const unsigned int work = (unsigned int)mtile_prefix[num_experts] * n_tiles;
    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;

    __shared__ __align__(16) PqwA sA[2];
    __shared__ __align__(16) PqwAs sAs[2];
    __shared__ __align__(16) PqwB sB[2];
    __shared__ __align__(16) PqwS sS[2];
    __shared__ int sTok[M_TILE];
    __shared__ unsigned int sNext;

    // One work item: its expert, rows, column tile and weight tables.
    struct Tile {
        unsigned int expert_id, cta_m, cta_n;
        int M_expert, cta_m_local;
        const unsigned char *B, *S, *U, *US;
    };
    auto decode = [&](unsigned int w) {
        Tile d;
        const int tl = (int)(w / n_tiles);
        d.cta_n = (w % n_tiles) * (GATE_UP ? PQW_NT / 2 : PQW_NT);
        d.expert_id = pqw_tile_expert(tl, mtile_prefix, num_experts);
        const int m_start = expert_offsets[d.expert_id];
        d.M_expert = expert_offsets[d.expert_id + 1] - m_start;
        d.cta_m_local = (tl - mtile_prefix[d.expert_id]) * M_TILE;
        d.cta_m = m_start + d.cta_m_local;
        d.B = (const unsigned char*)B_packed_ptrs[d.expert_id];
        d.S = (const unsigned char*)B_scale_ptrs[d.expert_id];
        d.U = GATE_UP ? (const unsigned char*)up.packed_ptrs[d.expert_id] : nullptr;
        d.US = GATE_UP ? (const unsigned char*)up.scale_ptrs[d.expert_id] : nullptr;
        return d;
    };
    auto issue = [&](const Tile& d, int buf, unsigned int kb) {
        if (d.B == 0) return;
        pqw_issue<GATE_UP>(t, sA[buf], sAs[buf], sB[buf], sS[buf], sTok, A_packed, A_scale,
            d.B, d.S, d.U, d.US, d.cta_m_local, (unsigned int)d.M_expert, d.cta_n, N, K, kb);
    };
    // Gather the item's token rows and issue its first K stage. Every
    // thread's reads of sTok and sNext and of stage buffer 0 for the
    // previous item precede the last __syncthreads of its K loop or this one.
    auto start = [&](const Tile& d) {
        pqw_gather_rows(sTok, t, sorted_token_ids, d.cta_m, d.cta_m_local, d.M_expert);
        __syncthreads();
        issue(d, 0, 0);
        moe_cp_async_commit();
    };

    if (t == 0) sNext = atomicAdd(next_work, 1);
    __syncthreads();
    unsigned int w = sNext;
    if (w >= work) return;
    const unsigned int stages = K / PQ2_KS;
    Tile cur = decode(w);
    start(cur);
    while (true) {
        // Claim the next item now; the K loop's barriers publish it.
        if (t == 0) sNext = atomicAdd(next_work, 1);
        PqwAcc acc;
        pqw_zero_acc(acc);
        for (unsigned int st = 0; st < stages; ++st) {
            const int buf = st % 2;
            moe_cp_async_wait_all();
            __syncthreads();   // stage `st` landed; every warp finished stage st-1
            if (st + 1 < stages) issue(cur, buf ^ 1, (st + 1) * PQ2_KS);
            moe_cp_async_commit();
            pqw_mma_stage<GATE_UP>(acc, sA[buf], sAs[buf], sB[buf], sS[buf], warp_id, lane_id);
        }
        w = sNext;
        const Tile done = cur;
        const bool more = w < work;
        if (more) {
            cur = decode(w);
            start(cur);
        }
        if (done.B != 0)
            pqw_epilogue<GATE_UP>(acc, warp_id, lane_id, done.expert_id, scale2_vals[done.expert_id],
                done.cta_m, done.cta_m_local, done.M_expert, done.cta_n, N, C, up);
        if (!more) break;
    }
}

// Grid (2 x SMs, 1, 1), 256 threads; arguments as
// moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w plus the zeroed work
// counter.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w_persist(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    unsigned char* __restrict__ out_packed,
    unsigned char* __restrict__ out_scale,
    int* __restrict__ next_work
) {
    pqp_impl<true>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, next_work,
        PqwGateUp{up_packed_ptrs, up_scale_ptrs, up_scale2_vals, out_packed, out_scale});
}

// Grid (2 x SMs, 1, 1), 256 threads; arguments as
// moe_w4a4_grouped_gemm_prequant_t_k128w_compact plus the zeroed work counter.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_grouped_gemm_prequant_t_k128w_compact_persist(
    PQ2_ARGS,
    const int* __restrict__ mtile_prefix,
    int* __restrict__ next_work
) {
    pqp_impl<false>(A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
        expert_offsets, sorted_token_ids, num_experts, N, K, mtile_prefix, next_work, PqwGateUp{});
}

// K-major-weight variant (checkpoint-native [N, K/2] + [N, K/16] scales):
// the same MMA sequence with no on-chip transpose. Benchmark-only probe of
// what the N-major prefill layout costs.
extern "C" __global__ void __launch_bounds__(256) moe_w4a4_grouped_gemm_prequant_nk_k128(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;
    const int m_start = expert_offsets[expert_id];
    const int M_expert = expert_offsets[expert_id + 1] - m_start;
    if (M_expert <= 0) return;
    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;
    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    if (B_expert == 0) return;
    const float scale2 = scale2_vals[expert_id];
    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32, lane_id = t % 32;
    const unsigned int warp_m_offset = (warp_id & 3) * 16;
    const unsigned int warp_n_offset = (warp_id >> 2) * 64;
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;

    __shared__ __align__(16) unsigned char sA[2][M_TILE][PQ2_AP];
    __shared__ __align__(16) unsigned char sAs[2][M_TILE][PQ2_KS / GROUP_SIZE];
    __shared__ __align__(16) unsigned char sBt[2][N_TILE_LG][PQ2_BP];
    __shared__ __align__(16) unsigned char sSt[2][N_TILE_LG][PQ2_KS / GROUP_SIZE];
    __shared__ int sTok[M_TILE];

    if (t < M_TILE) {
        const bool live = (cta_m_local + (int)t) < M_expert;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t] : (int)(cta_m + t);
    }
    __syncthreads();

    const unsigned int M_eff = (unsigned int)M_expert;
    auto issue = [&](int buf, unsigned int kb) {
        // A: 64 rows x 64 bytes = 256 x 16 B (two per thread).
        {
            const unsigned int j = t;
            const unsigned int row = j >> 2, col = (j & 3) << 4;
            const bool valid = (cta_m_local + row) < M_eff;
            const unsigned int a_row = (unsigned int)sTok[row];
            moe_cp_async_pred_16(&sA[buf][row][col],
                &A_packed[(unsigned long long)a_row * (K / 2) + kb / 2 + col], valid);
        }
        // A scales: 64 rows x 8 bytes.
        if (t < M_TILE) {
            const bool valid = (cta_m_local + t) < M_eff;
            const unsigned int a_row = (unsigned int)sTok[t];
            moe_cp_async_pred_8(&sAs[buf][t][0],
                &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) + kb / GROUP_SIZE], valid);
        }
        // B (K-major [N, K/2]): 128 rows x 64 bytes = 512 x 16 B, straight
        // into the MMA-ready layout.
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned int j = t + r * 256;
            const unsigned int n = j >> 2, col = (j & 3) << 4;
            moe_cp_async_pred_16(&sBt[buf][n][col],
                &B_expert[(unsigned long long)(cta_n + n) * (K / 2) + kb / 2 + col], true);
        }
        // B scales ([N, K/16]): 8 contiguous bytes per column.
        if (t < N_TILE_LG) {
            moe_cp_async_pred_8(&sSt[buf][t][0],
                &S_expert[(unsigned long long)(cta_n + t) * (K / GROUP_SIZE) + kb / GROUP_SIZE], true);
        }
    };

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

    const unsigned int stages = K / PQ2_KS;
    issue(0, 0);
    moe_cp_async_commit();
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st & 1;
        moe_cp_async_wait_all();
        __syncthreads();   // stage `st` landed; every warp finished stage st-1
        if (st + 1 < stages) {
            issue(buf ^ 1, (st + 1) * PQ2_KS);
            moe_cp_async_commit();
        }
        #pragma unroll
        for (int sl = 0; sl < 2; ++sl) {
            const unsigned int ko = sl * 32;   // packed-byte offset of this k64 slice
            // A fragment via one ldmatrix.x4: matrices (rows 0-7 | 8-15) x
            // (bytes 0-15 | 16-31) give exactly a0..a3 of the m16n8k64 layout.
            unsigned int a0, a1, a2, a3;
            {
                const unsigned int j = lane_id >> 3, r = lane_id & 7;
                const unsigned int addr = __cvta_generic_to_shared(
                    &sA[buf][warp_m_offset + r + (j & 1) * 8][ko + (j >> 1) * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
            }
            const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
            const unsigned int sfa = *(const unsigned int*)&sAs[buf][warp_m_offset + sfa_m][sl * 4];
            #pragma unroll
            for (int np = 0; np < 4; np++) {
                // B fragments of n-subtiles 2np and 2np+1 via one ldmatrix.x4.
                unsigned int b[4];
                {
                    const unsigned int j = lane_id >> 3, r = lane_id & 7;
                    const unsigned int addr = __cvta_generic_to_shared(
                        &sBt[buf][warp_n_offset + np * 16 + (j >> 1) * 8 + r][ko + (j & 1) * 16]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int nt = np * 2 + h;
                    const unsigned int b0 = b[h * 2], b1 = b[h * 2 + 1];
                    const unsigned int sfb = *(const unsigned int*)&sSt[buf][warp_n_offset + nt * 8 + (lane_id >> 2)][sl * 4];
                    unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                        "{%14},{%15,%16},{%17},{%18,%19};"
                        :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]),
                         "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_));
                }
            }
        }
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2, c1 = c0 + 1;
        const unsigned int r0 = cta_m + warp_m_offset + group_id, r1 = r0 + 8;
        const bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        const bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}

// Projection-multiplexed compact gate/up dispatch.  The y grid selects the
// pointer table and destination while x retains the proven compact work item.
// This removes one host submission per MoE layer without changing the native
// FP4 MMA body, its K accumulation order, or either output tensor's layout.
template<bool PQ_VEC_SCALES>
__device__ __forceinline__ void moe_w4a4_grouped_gemm_prequant_compact_gate_up_impl(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ C_gate,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles,
    unsigned int max_tiles
) {
    if (blockIdx.y > 1) return;
    const bool is_up = blockIdx.y != 0;
    moe_w4a4_grouped_gemm_prequant_compact_impl<PQ_VEC_SCALES>(
        A_packed, A_scale,
        is_up ? up_packed_ptrs : gate_packed_ptrs,
        is_up ? up_scale_ptrs : gate_scale_ptrs,
        is_up ? up_scale2_vals : gate_scale2_vals,
        is_up ? C_up : C_gate,
        expert_offsets, sorted_token_ids, num_experts, N, K,
        worklist, total_tiles, max_tiles);
}

#define PQ4_COMPACT_GATE_UP_ARGS \
    const unsigned char* A_packed, const unsigned char* A_scale, \
    const unsigned long long* gate_packed_ptrs, \
    const unsigned long long* gate_scale_ptrs, const float* gate_scale2_vals, \
    __nv_bfloat16* C_gate, const unsigned long long* up_packed_ptrs, \
    const unsigned long long* up_scale_ptrs, const float* up_scale2_vals, \
    __nv_bfloat16* C_up, const int* expert_offsets, const int* sorted_token_ids, \
    unsigned int num_experts, unsigned int N, unsigned int K, \
    const unsigned int* worklist, const int* total_tiles, unsigned int max_tiles

#define PQ4_COMPACT_GATE_UP_CALL(VEC) \
    moe_w4a4_grouped_gemm_prequant_compact_gate_up_impl<VEC>( \
        A_packed, A_scale, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, \
        C_gate, up_packed_ptrs, up_scale_ptrs, up_scale2_vals, C_up, \
        expert_offsets, sorted_token_ids, num_experts, N, K, \
        worklist, total_tiles, max_tiles)

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up(
    PQ4_COMPACT_GATE_UP_ARGS
) {
    PQ4_COMPACT_GATE_UP_CALL(false);
}

extern "C" __global__ void moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up(
    PQ4_COMPACT_GATE_UP_ARGS
) {
    atlas_pdl_enter();
    PQ4_COMPACT_GATE_UP_CALL(true);
}

#undef PQ4_COMPACT_GATE_UP_CALL
#undef PQ4_COMPACT_GATE_UP_ARGS

#undef PQ4_PREQUANT_CALL
#undef PQ4_PREQUANT_ARGS

// GLM-only, default-off small-M gate/up A/B. Existing down is unchanged.
#include "glm_moe_gate_up_m16.cuh"

// GLM verify-decode M16 twins of the K128W kernels (ATLAS_GLM_MOE_DECODE_M16).
#include "glm_moe_decode_m16.cuh"

// Their stream-loaded twins, one and two row slabs (ATLAS_GLM_MOE_DECODE_STREAM).
#include "glm_moe_decode_stream.cuh"

// Staged GLM B-tile exports. No loader/serving selection is enabled here.
#include "glm_moe_btile.cuh"
#include "glm_moe_btile_m64.cuh"
