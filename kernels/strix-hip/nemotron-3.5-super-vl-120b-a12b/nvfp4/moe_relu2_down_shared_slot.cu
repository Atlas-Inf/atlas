// SPDX-License-Identifier: AGPL-3.0-only

// Shared-expert-only slice of moe_expert_relu2_down_shared, for the K2
// batched LatentMoE verify (decode_verify_exact). One grid.z row per batch
// row; each row runs the exact shared-slot math of the fused kernel
// (relu^2 into fp32 smem, pair-multiply f32 accumulation, warp reduce).
// Kept as a separate entry point so the original kernel's PTX is untouched.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_R2S[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3_s(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3_s(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// Grid: (ceil(N_shared/8), 1, m)   Block: (128, 1, 1)
extern "C" __global__ void moe_relu2_down_shared_slot(
    const __nv_bfloat16* __restrict__ sh_up_in,     // [m, K_shared] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,         // [m, N_shared] BF16
    unsigned int N_shared,                           // hidden_size
    unsigned int K_shared                            // shared_inter
) {
    const unsigned int tok = blockIdx.z;

    const unsigned char* B_packed = sh_down_packed;
    const unsigned char* B_scale = sh_down_scale;
    float s2 = sh_down_s2;
    const __nv_bfloat16* u_ptr = sh_up_in + (unsigned long long)tok * K_shared;
    const unsigned int K = K_shared;
    const unsigned int N_out = N_shared;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_R2S[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;
    // B2: 16-byte packed-B loads + per-lane relu^2 from global (no smem —
    // the k32 stride made every lane hit the same smem bank). Per-lane k32
    // re-stride identical to the fused/wide/union twins: k8 = {4*lane+128i+q},
    // MAC order k-ascending al*w + ah*w — twins stay bitwise-equal.
    const uint4* Bp1 = (const uint4*)(B_packed + (unsigned long long)n1 * half_K);
    const uint4* Bp2 = (const uint4*)(B_packed + (unsigned long long)n2 * half_K);
    const unsigned int K32 = K / 32;
    for (unsigned int k32 = lane; k32 < K32; k32 += threads_per_out) {
        const unsigned int base_k = k32 * 32;
        uint4 v1 = Bp1[k32];
        unsigned char sb1g0 = B_scale[(unsigned long long)n1 * num_groups + base_k / GROUP_SIZE];
        unsigned char sb1g1 = B_scale[(unsigned long long)n1 * num_groups + (base_k + 16) / GROUP_SIZE];
        uint4 v2 = make_uint4(0, 0, 0, 0);
        unsigned char sb2g0 = 0, sb2g1 = 0;
        if (have_n2) {
            v2 = Bp2[k32];
            sb2g0 = B_scale[(unsigned long long)n2 * num_groups + base_k / GROUP_SIZE];
            sb2g1 = B_scale[(unsigned long long)n2 * num_groups + (base_k + 16) / GROUP_SIZE];
        }
        const unsigned int p1q[4] = {v1.x, v1.y, v1.z, v1.w};
        const unsigned int p2q[4] = {v2.x, v2.y, v2.z, v2.w};
        const unsigned char s1q[4] = {sb1g0, sb1g0, sb1g1, sb1g1};
        const unsigned char s2q[4] = {sb2g0, sb2g0, sb2g1, sb2g1};
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            const unsigned int base = base_k + q * 8;
            const unsigned int packed4_1 = p1q[q];
            const unsigned int packed4_2 = p2q[q];
            const float sc1 = atlas_dec_e4m3_s(s1q[q]) * s2;
            const float sc2 = have_n2 ? atlas_dec_e4m3_s(s2q[q]) * s2 : 0.0f;
            const uint4 a4 = ((const uint4*)u_ptr)[base / 8];
            const unsigned int a_raw[4] = {a4.x, a4.y, a4.z, a4.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 u_lo, u_hi;
                *(unsigned short*)&u_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&u_hi = (unsigned short)(a_raw[b] >> 16);
                float rl = fmaxf(__bfloat162float(u_lo), 0.0f);
                float rh = fmaxf(__bfloat162float(u_hi), 0.0f);
                float al = rl * rl, ah = rh * rh;
                unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
                float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
                unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
                float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
                acc1 += al * w1l + ah * w1h;
                acc2 += al * w2l + ah * w2h;
            }
        }
    }

    // Tail for K % 32 != 0 (model dims are multiples of 32; kept for generality).
    for (unsigned int k8 = K32 * 4 + lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base = k8 * 8;
        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int packed4_2 = have_n2 ? *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + base / GROUP_SIZE];
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + base / GROUP_SIZE] : 0;
        const float sc1 = atlas_dec_e4m3_s(sb1) * s2;
        const float sc2 = have_n2 ? atlas_dec_e4m3_s(sb2) * s2 : 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 u_lo = u_ptr[base + b * 2];
            __nv_bfloat16 u_hi = u_ptr[base + b * 2 + 1];
            float rl = fmaxf(__bfloat162float(u_lo), 0.0f);
            float rh = fmaxf(__bfloat162float(u_hi), 0.0f);
            float al = rl * rl, ah = rh * rh;
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    __nv_bfloat16* out = sh_down_out + (unsigned long long)tok * N_shared;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
