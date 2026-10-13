// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#ifdef __CUDACC__
#define GLM_SHARED_HD __host__ __device__
#else
#define GLM_SHARED_HD
#endif

// One helper serves both actual CUDA addressing and exhaustive CPU ownership.
GLM_SHARED_HD constexpr unsigned glm_shared_m16_col(unsigned warp, unsigned nt, unsigned offset) {
    return warp * 32 + nt * 8 + offset;
}
GLM_SHARED_HD constexpr unsigned glm_shared_m16_row(unsigned group, unsigned half) {
    return group + half * 8;
}
GLM_SHARED_HD constexpr bool glm_shared_m16_shape(unsigned m, unsigned n, unsigned k, unsigned ldb) {
    return m <= 16 && n > 0 && k > 0 && k <= 4096 && k % 32 == 0
        && ldb >= n && ldb <= 4096 && ldb % 128 == 0;
}
#undef GLM_SHARED_HD

#ifdef __CUDACC__
// Included at the end of w4a16_gemm.cu, after its shared primitive definitions.
// Same loaders/dequant/MMA instructions; only the A extent and warp ownership differ.
extern "C" __global__ void glm_shared_w4a16_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int ldb
) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || blockIdx.y || blockIdx.z
        || !glm_shared_m16_shape(M, N, K, ldb) || M == 0) return;
    const unsigned int LDB = ldb;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = 0;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][16][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    #define ISSUE_LOADS(buf, kb) do { \
        if (threadIdx.x < 64) { \
            unsigned int row = threadIdx.x >> 2; \
            unsigned int col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + col; \
            /* Predicated-off source is a valid allocation address too. */ \
            unsigned int gr = row < M ? row : 0; \
            cp_async_pred_16(&smem_A[(buf)][row][col], \
                &A[gr * K + gc], (row < M) && (gc + 7 < K)); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * LDB + gns], \
                (gke + 1 <= K) && (gns + 15 < LDB)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * LDB + gns], \
                    (gns + 15 < LDB)); \
            } \
        } \
    } while(0)

    // Dequant B: FP4 → FP8 E4M3 (cvt.rn.satfinite.e4m3x2.f32)
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: convert A BF16→E4M3 in registers, single m16n8k32 per N-tile
    #define COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = glm_shared_m16_row(group_id, 0), fr1 = glm_shared_m16_row(group_id, 1); \
        unsigned int a0 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 4; nt++) { \
            unsigned int nc = glm_shared_m16_col(warp_id, nt, group_id); \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    ISSUE_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        cp_async_commit();
        COMPUTE_MMA(cur);
        cp_async_wait_all();
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }

    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        unsigned int c0 = cta_n + glm_shared_m16_col(warp_id, nt, tid*2);
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + glm_shared_m16_row(group_id, 0);
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

#endif
