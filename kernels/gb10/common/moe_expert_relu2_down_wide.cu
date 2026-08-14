// SPDX-License-Identifier: AGPL-3.0-only
//
// Coarse-N relu²+down. Same FMA as moe_expert_relu2_down_shared acc1 path.
// relu²(up) staged once per CTA; 32 N / CTA so the 14 KB act fill is not
// repeated on every 8-wide N tile.
//
// Grid: (ceil(N/32), top_k+1, num_tokens)  Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3_w(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3_w(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

#define BLOCK_SIZE 256
#define N_TILE 64
#define N_PER_WARP 8
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_R2W[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_expert_relu2_down_wide(
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N,
    unsigned int K_routed,
    unsigned int K_shared,
    unsigned int N_shared,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int tok = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);
    const unsigned int n_base = blockIdx.x * N_TILE;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    unsigned int K;
    unsigned int N_out;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_packed = sh_down_packed;
        B_scale = sh_down_scale;
        s2 = sh_down_s2;
        u_ptr = sh_up_in + (unsigned long long)tok * K_shared;
        K = K_shared;
        N_out = N_shared;
    } else {
        const unsigned int expert_id = expert_indices[(unsigned long long)tok * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        u_ptr = up_out + ((unsigned long long)tok * top_k + expert_slot) * K_routed;
        K = K_routed;
        N_out = N;
        if (B_packed == 0) {
            for (unsigned int i = threadIdx.x; i < N_TILE && n_base + i < N_out; i += BLOCK_SIZE) {
                C[((unsigned long long)tok * top_k + expert_slot) * N + n_base + i] =
                    __float2bfloat16(0.0f);
            }
            return;
        }
    }
    if (n_base >= N_out) return;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_R2W[threadIdx.x];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float u = __bfloat162float(u_ptr[i]);
        float r = fmaxf(u, 0.0f);
        s_act[i] = r * r;
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;
    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)tok * N_shared)
        : (C + ((unsigned long long)tok * top_k + expert_slot) * N);

    #pragma unroll 1
    for (unsigned int t = 0; t < N_PER_WARP; t++) {
        const unsigned int n = n_base + warp * N_PER_WARP + t;
        if (n >= N_out) return;
        float acc = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += WARP_SIZE) {
            const unsigned int base_k = k8 * 8;
            unsigned int packed4 = *(const unsigned int*)(
                B_packed + (unsigned long long)n * half_K + k8 * 4);
            unsigned char sb = B_scale[(unsigned long long)n * num_groups + base_k / GROUP_SIZE];
            float sc = atlas_dec_e4m3_w(sb) * s2;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                float al = s_act[base_k + b * 2];
                float ah = s_act[base_k + b * 2 + 1];
                unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
                acc += al * (s_lut[bv & 0xF] * sc) + ah * (s_lut[bv >> 4] * sc);
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
        if (lane == 0) out[n] = __float2bfloat16(acc);
    }
}
