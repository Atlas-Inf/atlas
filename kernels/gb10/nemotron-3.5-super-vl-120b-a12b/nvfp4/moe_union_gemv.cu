// SPDX-License-Identifier: AGPL-3.0-only
//
// Omni-local overlay: union-table-driven variant of
// common/moe_expert_gemv_wide_grouped (verify path).
//
// The common kernel gives each of (N/32 x num_experts) CTAs a serial
// thread-0 scan over num_tokens*top_k indices (~90% of CTAs exit empty).
// moe_union_match builds the table once; moe_expert_gemv_wide_grouped_u
// launches only unique experts and reads their match list directly.
// The per-row math is the common kernel's verbatim: same BATCH_M batching
// in scan order (tok, then slot), same k8 lane loop, same paired-RHS
// relu2 accumulation, same shuffle reduction and stores.
//
// u_buf layout (u32 words, MAXU=num_experts, MAX_MATCH=32):
//   [0]               u_count
//   [1 .. 1+MAXU)     u_experts (unique ids, ascending)
//   [1+MAXU .. )      u_nmatch per unique expert
//   then u_tok[MAXU][MAX_MATCH] and u_slot[MAXU][MAX_MATCH]

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define UG_BLOCK_SIZE 256
#define UG_N_TILE 32
#define UG_N_PER_WARP 4
#define UG_WARP_SIZE 32
#define UG_GROUP_SIZE 16
#define UG_BATCH_M 4
#define UG_MAX_MATCH 32
#define UG_MAXU 512

__device__ __constant__ float E2M1_LUT_UG[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_union_match(
    const unsigned int* __restrict__ expert_indices,
    unsigned int* __restrict__ u_buf,
    unsigned int num_tokens,
    unsigned int top_k
) {
    const unsigned int e = threadIdx.x;
    unsigned int* u_experts = u_buf + 1;
    unsigned int* u_nmatch = u_buf + 1 + UG_MAXU;
    unsigned int* u_tok = u_buf + 1 + 2 * UG_MAXU;
    unsigned int* u_slot = u_tok + UG_MAXU * UG_MAX_MATCH;
    unsigned int count = 0;
    for (unsigned int tok = 0; tok < num_tokens && count < UG_MAX_MATCH; tok++) {
        for (unsigned int slot = 0; slot < top_k && count < UG_MAX_MATCH; slot++) {
            if (expert_indices[(unsigned long long)tok * top_k + slot] == e) {
                u_tok[e * UG_MAX_MATCH + count] = tok;
                u_slot[e * UG_MAX_MATCH + count] = slot;
                count++;
            }
        }
    }
    u_nmatch[e] = count;
    // Prefix must read THIS call's counts — thread 0 summing global
    // u_nmatch before a barrier would see the previous verify's table.
    __shared__ unsigned int s_cnt[UG_MAXU];
    __shared__ unsigned int s_sum[UG_MAXU];
    s_cnt[e] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int s = 0;
        for (unsigned int i = 0; i < UG_MAXU; i++) {
            s_sum[i] = s;
            s += s_cnt[i] > 0 ? 1 : 0;
        }
        u_buf[0] = s;
    }
    __syncthreads();
    if (count > 0) u_experts[s_sum[e]] = e;
}

extern "C" __global__ void moe_expert_gemv_wide_grouped_u(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ u_buf,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride,
    unsigned int num_tokens,
    unsigned int relu2_input
) {
    const unsigned int u_idx = blockIdx.y;
    if (u_idx >= u_buf[0]) return;
    const unsigned int expert_id = u_buf[1 + u_idx];
    const unsigned int n_base = blockIdx.x * UG_N_TILE;
    if (n_base >= N) return;
    const unsigned int* u_nmatch = u_buf + 1 + UG_MAXU;
    const unsigned int* u_tok = u_buf + 1 + 2 * UG_MAXU;
    const unsigned int* u_slot = u_tok + UG_MAXU * UG_MAX_MATCH;
    const unsigned int nmatch = u_nmatch[expert_id];
    if (nmatch == 0) return;

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_packed == 0) {
        for (unsigned int m = 0; m < nmatch; m++) {
            for (unsigned int i = threadIdx.x; i < UG_N_TILE && n_base + i < N; i += UG_BLOCK_SIZE) {
                C[((unsigned long long)u_tok[expert_id * UG_MAX_MATCH + m] * top_k +
                   u_slot[expert_id * UG_MAX_MATCH + m]) * N + n_base + i] =
                    __float2bfloat16(0.0f);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    float* s_lut = (float*)smem_raw;
    __nv_bfloat16* s_A = (__nv_bfloat16*)(smem_raw + 64);
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_UG[threadIdx.x];

    const unsigned int warp = threadIdx.x / UG_WARP_SIZE;
    const unsigned int lane = threadIdx.x % UG_WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / UG_GROUP_SIZE;
    const unsigned int K8 = K / 8;

    for (unsigned int base = 0; base < nmatch; base += UG_BATCH_M) {
        const unsigned int batch = nmatch - base < UG_BATCH_M ? nmatch - base : UG_BATCH_M;
        for (unsigned int m = 0; m < batch; m++) {
            const unsigned int tok = u_tok[expert_id * UG_MAX_MATCH + base + m];
            const unsigned int slot = u_slot[expert_id * UG_MAX_MATCH + base + m];
            const __nv_bfloat16* input = input_stride > 0
                ? A + ((unsigned long long)tok * top_k + slot) * input_stride
                : A + (unsigned long long)tok * K;
            __nv_bfloat16* dst = s_A + (unsigned long long)m * K;
            for (unsigned int i = threadIdx.x; i < K; i += UG_BLOCK_SIZE) dst[i] = input[i];
        }
        __syncthreads();

        #pragma unroll 1
        for (unsigned int t = 0; t < UG_N_PER_WARP; t++) {
            const unsigned int n = n_base + warp * UG_N_PER_WARP + t;
            if (n < N) {
                float acc[UG_BATCH_M];
                #pragma unroll
                for (int m = 0; m < UG_BATCH_M; m++) acc[m] = 0.0f;

                for (unsigned int k8 = lane; k8 < K8; k8 += UG_WARP_SIZE) {
                    const unsigned int base_k = k8 * 8;
                    unsigned int packed4 = *(const unsigned int*)(
                        B_packed + (unsigned long long)n * half_K + k8 * 4);
                    unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + base_k / UG_GROUP_SIZE];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    float scale = (float)fp8 * scale2;
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
                                acc[m] += al * w_lo[b] + ah * w_hi[b];
                            } else {
                                acc[m] += al * w_lo[b];
                                acc[m] += ah * w_hi[b];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int off = UG_WARP_SIZE / 2; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int m = 0; m < UG_BATCH_M; m++)
                        acc[m] += __shfl_down_sync(0xFFFFFFFF, acc[m], off);
                }
                if (lane == 0) {
                    for (unsigned int m = 0; m < batch; m++) {
                        const unsigned int tok = u_tok[expert_id * UG_MAX_MATCH + base + m];
                        const unsigned int slot = u_slot[expert_id * UG_MAX_MATCH + base + m];
                        C[((unsigned long long)tok * top_k + slot) * N + n] =
                            __float2bfloat16(acc[m]);
                    }
                }
            }
        }
        __syncthreads();
    }
}
