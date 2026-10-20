// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert relu²+down GEMV — for 2-projection experts (Nemotron-H).
//
// Nemotron-H MoE experts have only up_proj + down_proj (no gate_proj).
// Activation: relu(x)^2 (ReLU-squared) instead of SiLU(gate)*up.
//
// Same structure as moe_expert_silu_down_shared.cu but with relu² activation.
// blockIdx.y < top_k: routed expert
// blockIdx.y == top_k: shared expert (direct weight pointers, different K)
//
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_R2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; NVIDIA path is the verbatim cast.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// Fused relu²+down 2x GEMV with shared expert.
//
// Reads up_out, applies relu², loads result into shared memory,
// then performs down-proj GEMV using the activated values as input.
//
// Routed experts: up_out at [slot*K_routed, K_routed], weights from pointer table
// Shared expert:  up_out at sh_up_in [K_shared], weights from direct pointers
//
// K_routed and K_shared may differ (e.g., 1856 vs 3712 for Nemotron-H).
extern "C" __global__ void moe_expert_relu2_down_shared(
    const __nv_bfloat16* __restrict__ up_out,       // [top_k, K_routed] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                   // [top_k, N] routed output
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_up_in,     // [K_shared] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,         // [N_shared] BF16
    unsigned int N,             // output dim (hidden_size for routed)
    unsigned int K_routed,      // input dim for routed experts (moe_intermediate_size)
    unsigned int K_shared,      // input dim for shared expert (shared_expert_intermediate_size)
    unsigned int N_shared,      // output dim for shared expert (hidden_size)
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int tok = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

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
        // EP: NULL pointer means remote expert
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                C[((unsigned long long)tok * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

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
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_R2[threadIdx.x];
    __syncthreads();
    // B2: no smem staging — per-lane relu²(u) straight from global, matching
    // moe_relu2_down_shared_slot element-for-element.

    // Phase 2: GEMV reading precomputed activation from shared memory
    float acc1 = 0.0f, acc2 = 0.0f;

    // B2: 16-byte packed-B loads (4 packed4 = 32 weights per lane step) with
    // the same k32 re-stride as moe_relu2_down_shared_slot — every verify/slot
    // twin shares this per-row order, keeping them bitwise-equal to each other.
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
            const float sc1 = atlas_dec_e4m3(s1q[q]) * s2;
            const float sc2 = have_n2 ? atlas_dec_e4m3(s2q[q]) * s2 : 0.0f;
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



    // Output: shared writes to sh_down_out, routed writes to C[slot*N]

    // Tail for K % 32 != 0 (model dims are multiples of 32; kept for generality).
    for (unsigned int k8 = K32 * 4 + lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base = k8 * 8;
        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int packed4_2 = have_n2 ? *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + base / GROUP_SIZE];
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + base / GROUP_SIZE] : 0;
        const float sc1 = atlas_dec_e4m3(sb1) * s2;
        const float sc2 = have_n2 ? atlas_dec_e4m3(sb2) * s2 : 0.0f;
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

    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)tok * N_shared)
        : (C + ((unsigned long long)tok * top_k + expert_slot) * N);

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
