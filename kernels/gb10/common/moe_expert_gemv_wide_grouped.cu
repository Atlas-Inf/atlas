// SPDX-License-Identifier: AGPL-3.0-only
//
// Same FMA as moe_expert_gemv_wide, but grid is (ceil(N/32), num_experts).
// One CTA owns one expert N-tile and applies it to every token that routed
// here. B is streamed once per match-batch of 4 (not once per token).
// Empty experts exit after a cheap index scan.
//
// Do not fuse UP+DOWN. Do not use moe_w4a16_grouped_gemm (M_TILE=64).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8_grp(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_TILE 32
#define N_PER_WARP 4
#define WARP_SIZE 32
#define GROUP_SIZE 16
#define BATCH_M 4
#define MAX_MATCH 32

__device__ __constant__ float E2M1_LUT_GRP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_expert_gemv_wide_grouped(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride,
    unsigned int num_tokens,
    unsigned int relu2_input
) {
    const unsigned int expert_id = blockIdx.y;
    const unsigned int n_base = blockIdx.x * N_TILE;
    if (n_base >= N) return;

    __shared__ unsigned int s_m_tok[MAX_MATCH];
    __shared__ unsigned int s_m_slot[MAX_MATCH];
    __shared__ unsigned int s_nmatch;
    if (threadIdx.x == 0) {
        unsigned int count = 0;
        for (unsigned int tok = 0; tok < num_tokens && count < MAX_MATCH; tok++) {
            for (unsigned int slot = 0; slot < top_k && count < MAX_MATCH; slot++) {
                if (expert_indices[(unsigned long long)tok * top_k + slot] == expert_id) {
                    s_m_tok[count] = tok;
                    s_m_slot[count] = slot;
                    count++;
                }
            }
        }
        s_nmatch = count;
    }
    __syncthreads();
    const unsigned int nmatch = s_nmatch;
    if (nmatch == 0) return;

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_packed == 0) {
        for (unsigned int m = 0; m < nmatch; m++) {
            for (unsigned int i = threadIdx.x; i < N_TILE && n_base + i < N; i += BLOCK_SIZE) {
                C[((unsigned long long)s_m_tok[m] * top_k + s_m_slot[m]) * N + n_base + i] =
                    __float2bfloat16(0.0f);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    float* s_lut = (float*)smem_raw;
    __nv_bfloat16* s_A = (__nv_bfloat16*)(smem_raw + 64);
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_GRP[threadIdx.x];

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    for (unsigned int base = 0; base < nmatch; base += BATCH_M) {
        const unsigned int batch = nmatch - base < BATCH_M ? nmatch - base : BATCH_M;
        for (unsigned int m = 0; m < batch; m++) {
            const unsigned int tok = s_m_tok[base + m];
            const unsigned int slot = s_m_slot[base + m];
            const __nv_bfloat16* input = input_stride > 0
                ? A + ((unsigned long long)tok * top_k + slot) * input_stride
                : A + (unsigned long long)tok * K;
            __nv_bfloat16* dst = s_A + (unsigned long long)m * K;
            for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) dst[i] = input[i];
        }
        __syncthreads();

        #pragma unroll 1
        for (unsigned int t = 0; t < N_PER_WARP; t++) {
            const unsigned int n = n_base + warp * N_PER_WARP + t;
            if (n < N) {
                float acc[BATCH_M];
                #pragma unroll
                for (int m = 0; m < BATCH_M; m++) acc[m] = 0.0f;

                for (unsigned int k8 = lane; k8 < K8; k8 += WARP_SIZE) {
                    const unsigned int base_k = k8 * 8;
                    unsigned int packed4 = *(const unsigned int*)(
                        B_packed + (unsigned long long)n * half_K + k8 * 4);
                    unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + base_k / GROUP_SIZE];
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                    float scale = scl_fp8_grp(scale_byte) * scale2;
#else
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    float scale = (float)fp8 * scale2;
#endif
                    float w_lo[4], w_hi[4];
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
                        w_lo[b] = s_lut[byte_val & 0xF] * scale;
                        w_hi[b] = s_lut[byte_val >> 4] * scale;
                    }
                    for (unsigned int m = 0; m < batch; m++) {
                        uint4 a_data = ((const uint4*)(s_A + (unsigned long long)m * K))[k8];
                        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                        #pragma unroll
                        for (int b = 0; b < 4; b++) {
                            __nv_bfloat16 a_lo, a_hi;
                            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                            float al = __bfloat162float(a_lo);
                            float ah = __bfloat162float(a_hi);
                            if (relu2_input) {
                                al = fmaxf(al, 0.0f); al *= al;
                                ah = fmaxf(ah, 0.0f); ah *= ah;
                                // Match moe_expert_relu2_down_wide exactly:
                                // one paired RHS before accumulation.
                                acc[m] += al * w_lo[b] + ah * w_hi[b];
                            } else {
                                acc[m] += al * w_lo[b];
                                acc[m] += ah * w_hi[b];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int m = 0; m < BATCH_M; m++)
                        acc[m] += __shfl_down_sync(0xFFFFFFFF, acc[m], off);
                }
                if (lane == 0) {
                    for (unsigned int m = 0; m < batch; m++) {
                        const unsigned int tok = s_m_tok[base + m];
                        const unsigned int slot = s_m_slot[base + m];
                        C[((unsigned long long)tok * top_k + slot) * N + n] =
                            __float2bfloat16(acc[m]);
                    }
                }
            }
        }
        __syncthreads();
    }
}
