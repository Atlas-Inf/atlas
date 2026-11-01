// SPDX-License-Identifier: AGPL-3.0-only

// Prefetched variant of common/moe_expert_gemv.cu (moe_expert_gemv_pf only; original kernel dropped) — Atlas Batched MoE Expert GEMV — All top-K experts in one kernel launch.
//
// Replaces 10 individual w4a16_gemv + 10 scaled_add with:
//   1 batched GEMV (gate) + 1 batched GEMV (up) + 1 SiLU + 1 batched GEMV (down) + 1 weighted sum
//
// Pointer indirection: kernel reads expert_id from device-side indices array,
// then looks up weight pointers from device-side tables (set up at model load).
// No D2H sync needed — everything stays on device.
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight layout: [N, K/2] packed, [N, K/GROUP_SIZE] scales (K-dim packing).
//
// 4 outputs per block, 32 threads (1 warp) per output. Warp shuffle reduction.
// Grid: (ceil(N / 4), top_k, 1)
// Block: (128, 1, 1)
// blockIdx.y = expert slot (0..top_k-1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format which mismatches the
// standard E4M3 scales written by the encoder -> corrupts every block scale.
// HIP/gfx1151 shares the same software path (no cvt.rn.satfinite.e4m3x2.f32 PTX).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_EXP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Batched MoE Expert W4A16 GEMV (vectorized)
//
// For top-K experts simultaneously:
//   C[slot, n] = sum_k A[slot_or_shared, k] * dequant(B[expert_id, n, k])
//
// input_stride: 0 = shared input (gate/up: all experts read same A[1,K])
//               K = strided input (down: slot e reads from A + e * K)
extern "C" __global__ void moe_expert_gemv_pf(
    const __nv_bfloat16* __restrict__ A,               // [1, K] or [top_k, K]
    const unsigned long long* __restrict__ packed_ptrs, // [num_experts] B_packed device ptrs
    const unsigned long long* __restrict__ scale_ptrs,  // [num_experts] B_scale device ptrs
    const float* __restrict__ scale2_vals,              // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                      // [top_k, N] contiguous output
    const unsigned int* __restrict__ expert_indices,    // [top_k] from GPU top-K
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride                           // 0 = shared, K = per-expert
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;
    const unsigned int tok = blockIdx.z;

    const unsigned int expert_id = expert_indices[(unsigned long long)tok * top_k + expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * N_PER_BLOCK;
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK && n_base + i < N; i += BLOCK_SIZE) {
            C[((unsigned long long)tok * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* input = A
        + (unsigned long long)tok * K
        + (input_stride > 0 ? (unsigned long long)expert_slot * input_stride : 0);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_EXP[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // Software prefetch: issue the next iteration's 4B weight word, scale byte
    // and activation vec while the current one computes. Identical loads,
    // identical accumulation order — byte-identical to moe_expert_gemv.
    unsigned int k8 = lane;
    uint4 a_data;
    unsigned int packed4 = 0;
    unsigned char scale_byte = 0;
    if (k8 < K8) {
        a_data = ((const uint4*)input)[k8];
        packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        scale_byte = B_scale[(unsigned long long)n * num_groups + (k8 * 8) / GROUP_SIZE];
    }
    for (; k8 < K8; k8 += threads_per_out) {
        const uint4 a_cur = a_data;
        const unsigned int p_cur = packed4;
        const unsigned char s_cur = scale_byte;
        const unsigned int k8n = k8 + threads_per_out;
        if (k8n < K8) {
            a_data = ((const uint4*)input)[k8n];
            packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8n * 4);
            scale_byte = B_scale[(unsigned long long)n * num_groups + (k8n * 8) / GROUP_SIZE];
        }

        const unsigned int a_raw[4] = {a_cur.x, a_cur.y, a_cur.z, a_cur.w};
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = s_cur;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(s_cur) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (p_cur >> (b * 8)) & 0xFF;
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[((unsigned long long)tok * top_k + expert_slot) * N + n] = __float2bfloat16(acc);
    }
}
