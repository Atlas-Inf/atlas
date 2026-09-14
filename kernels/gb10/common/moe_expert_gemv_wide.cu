// SPDX-License-Identifier: AGPL-3.0-only
//
// Coarse-N pointer-table W4A16 expert GEMV. Same FMA/reduction as
// moe_expert_gemv. A[K] staged once in smem so the 3 GB weight stream
// cannot evict it from L2 on every k-iteration.
//
// Grid: (ceil(N/32), top_k, num_tokens)  Block: (256, 1, 1)
// 8 warps × 4 sequential outputs = 32 N / CTA. ~1392 CTAs at Lightning
// verify (N=1856, top_k=6, tokens=4) — fills 48 SMs, unlike the 28-CTA fuse.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8_wide(unsigned char b) {
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

__device__ __constant__ float E2M1_LUT_WIDE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_expert_gemv_wide(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;
    const unsigned int tok = blockIdx.z;
    const unsigned int expert_id = expert_indices[(unsigned long long)tok * top_k + expert_slot];
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    const unsigned int n_base = blockIdx.x * N_TILE;
    if (n_base >= N) return;

    if (B_packed == 0) {
        for (unsigned int i = threadIdx.x; i < N_TILE && n_base + i < N; i += BLOCK_SIZE) {
            C[((unsigned long long)tok * top_k + expert_slot) * N + n_base + i] =
                __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* input = A
        + (unsigned long long)tok * K
        + (input_stride > 0 ? (unsigned long long)expert_slot * input_stride : 0);

    extern __shared__ char smem_raw[];
    float* s_lut = (float*)smem_raw;
    __nv_bfloat16* s_A = (__nv_bfloat16*)(smem_raw + 64);
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_WIDE[threadIdx.x];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) s_A[i] = input[i];
    __syncthreads();

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    #pragma unroll 1
    for (unsigned int t = 0; t < N_PER_WARP; t++) {
        const unsigned int n = n_base + warp * N_PER_WARP + t;
        if (n >= N) return;

        float acc = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += WARP_SIZE) {
            const unsigned int base_k = k8 * 8;
            uint4 a_data = ((const uint4*)s_A)[k8];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            unsigned int packed4 = *(const unsigned int*)(
                B_packed + (unsigned long long)n * half_K + k8 * 4);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + base_k / GROUP_SIZE];
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8_wide(scale_byte) * scale2;
#else
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
            float scale = (float)fp8 * scale2;
#endif
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
                float w_lo = s_lut[byte_val & 0xF] * scale;
                float w_hi = s_lut[byte_val >> 4] * scale;
                __nv_bfloat16 a_lo, a_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                acc += __bfloat162float(a_lo) * w_lo;
                acc += __bfloat162float(a_hi) * w_hi;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
        if (lane == 0) {
            C[((unsigned long long)tok * top_k + expert_slot) * N + n] = __float2bfloat16(acc);
        }
    }
}
