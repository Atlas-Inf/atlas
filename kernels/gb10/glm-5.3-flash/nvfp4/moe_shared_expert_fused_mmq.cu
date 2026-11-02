// SPDX-License-Identifier: AGPL-3.0-only
//
// Single-token MoE decode for routed weights in Atlas's equal-size MMQ
// block_nvfp4 layout.  The shared expert remains in checkpoint row-major
// NVFP4, avoiding another layout conversion for a small, always-active path.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define MMQ_DEC_BLOCK 128
#define MMQ_DEC_N_PER_BLOCK 4
#define MMQ_DEC_GROUP 16

__device__ __constant__ float E2M1_LUT_MMQ_DEC[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float mmq_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f;
    *reinterpret_cast<unsigned char *>(&f) = b;
    return static_cast<float>(f);
}

// Load one K16 group from a routed block_nvfp4 row.  The eight bytes encode
// [k0..k7] in low nibbles and [k8..k15] in high nibbles.
__device__ __forceinline__ const unsigned char * mmq_dec_group_ptr(
        const unsigned char * row, unsigned int k16) {
    return row + (k16 >> 2) * 32 + (k16 & 3) * 8;
}

__device__ __forceinline__ unsigned char mmq_dec_group_scale(
        const unsigned char * row, unsigned int k16, unsigned int k) {
    const unsigned int blocks = k / 64;
    return row[blocks * 32 + (k16 >> 2) * 4 + (k16 & 3)];
}

extern "C" __global__ void moe_expert_gate_up_shared_mmq(
    const __nv_bfloat16 * __restrict__ A,
    const unsigned long long * __restrict__ gate_mmq_ptrs,
    const unsigned long long * __restrict__ gate_unused_scale_ptrs,
    const float * __restrict__ gate_scale2,
    __nv_bfloat16 * __restrict__ gate_out,
    const unsigned long long * __restrict__ up_mmq_ptrs,
    const unsigned long long * __restrict__ up_unused_scale_ptrs,
    const float * __restrict__ up_scale2,
    __nv_bfloat16 * __restrict__ up_out,
    const unsigned int * __restrict__ expert_indices,
    const unsigned char * __restrict__ sh_gate_packed,
    const unsigned char * __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16 * __restrict__ sh_gate_out,
    const unsigned char * __restrict__ sh_up_packed,
    const unsigned char * __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16 * __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k) {
    (void) gate_unused_scale_ptrs;
    (void) up_unused_scale_ptrs;
    const unsigned int slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool shared = slot == top_k;
    const unsigned int expert = shared ? 0 : expert_indices[slot];
    const unsigned char * packed;
    const unsigned char * scale = nullptr;
    float scale2;
    __nv_bfloat16 * out;
    if (shared) {
        packed = proj == 0 ? sh_gate_packed : sh_up_packed;
        scale = proj == 0 ? sh_gate_scale : sh_up_scale;
        scale2 = proj == 0 ? sh_gate_s2 : sh_up_s2;
        out = proj == 0 ? sh_gate_out : sh_up_out;
    } else {
        packed = reinterpret_cast<const unsigned char *>(
            proj == 0 ? gate_mmq_ptrs[expert] : up_mmq_ptrs[expert]);
        scale2 = proj == 0 ? gate_scale2[expert] : up_scale2[expert];
        out = proj == 0 ? gate_out : up_out;
    }
    if (packed == nullptr) {
        const unsigned int base = blockIdx.x * (MMQ_DEC_N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < MMQ_DEC_N_PER_BLOCK * 2; i += blockDim.x) {
            if (base + i < N) out[(shared ? 0 : (unsigned long long) slot * N) + base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned int threads_per_out = MMQ_DEC_BLOCK / MMQ_DEC_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (MMQ_DEC_N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;

    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT_MMQ_DEC[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;
    const unsigned int groups = K / MMQ_DEC_GROUP;
    const unsigned int row_bytes = K * 9 / 16;
    const unsigned char * row1 = packed + (unsigned long long) n1 * row_bytes;
    const unsigned char * row2 = packed + (unsigned long long) n2 * row_bytes;
    for (unsigned int k16 = lane; k16 < groups; k16 += threads_per_out) {
        const unsigned int base = k16 * 16;
        if (shared) {
            const unsigned char * p1 = packed + (unsigned long long) n1 * (K / 2) + k16 * 8;
            const unsigned char * p2 = packed + (unsigned long long) n2 * (K / 2) + k16 * 8;
            const float sc1 = mmq_dec_e4m3(scale[(unsigned long long) n1 * groups + k16]) * scale2;
            const float sc2 = n2 < N
                ? mmq_dec_e4m3(scale[(unsigned long long) n2 * groups + k16]) * scale2 : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned char w1 = p1[j];
                const unsigned char w2 = n2 < N ? p2[j] : 0;
                const float a0 = __bfloat162float(A[base + 2 * j]);
                const float a1 = __bfloat162float(A[base + 2 * j + 1]);
                acc1 += a0 * lut[w1 & 15] * sc1 + a1 * lut[w1 >> 4] * sc1;
                acc2 += a0 * lut[w2 & 15] * sc2 + a1 * lut[w2 >> 4] * sc2;
            }
        } else {
            const unsigned char * p1 = mmq_dec_group_ptr(row1, k16);
            const unsigned char * p2 = mmq_dec_group_ptr(row2, k16);
            const float sc1 = mmq_dec_e4m3(mmq_dec_group_scale(row1, k16, K)) * scale2;
            const float sc2 = n2 < N
                ? mmq_dec_e4m3(mmq_dec_group_scale(row2, k16, K)) * scale2 : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned char w1 = p1[j];
                const unsigned char w2 = n2 < N ? p2[j] : 0;
                const float a0 = __bfloat162float(A[base + j]);
                const float a1 = __bfloat162float(A[base + 8 + j]);
                acc1 += a0 * lut[w1 & 15] * sc1 + a1 * lut[w1 >> 4] * sc1;
                acc2 += a0 * lut[w2 & 15] * sc2 + a1 * lut[w2 >> 4] * sc2;
            }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc1 += __shfl_down_sync(0xffffffff, acc1, d);
    if (lane == 0) out[(shared ? 0 : (unsigned long long) slot * N) + n1] = __float2bfloat16(acc1);
    if (n2 < N) {
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, d);
        if (lane == 0) out[(shared ? 0 : (unsigned long long) slot * N) + n2] = __float2bfloat16(acc2);
    }
}

extern "C" __global__ void moe_expert_silu_down_shared_mmq(
    const __nv_bfloat16 * __restrict__ gate_out,
    const __nv_bfloat16 * __restrict__ up_out,
    const unsigned long long * __restrict__ down_mmq_ptrs,
    const unsigned long long * __restrict__ unused_scale_ptrs,
    const float * __restrict__ down_scale2,
    __nv_bfloat16 * __restrict__ out,
    const unsigned int * __restrict__ expert_indices,
    const __nv_bfloat16 * __restrict__ sh_gate,
    const __nv_bfloat16 * __restrict__ sh_up,
    const unsigned char * __restrict__ sh_down_packed,
    const unsigned char * __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16 * __restrict__ sh_out,
    unsigned int N, unsigned int K, unsigned int top_k) {
    (void) unused_scale_ptrs;
    const unsigned int slot = blockIdx.y;
    const bool shared = slot == top_k;
    const unsigned int expert = shared ? 0 : expert_indices[slot];
    const unsigned char * packed = shared ? sh_down_packed
        : reinterpret_cast<const unsigned char *>(down_mmq_ptrs[expert]);
    if (packed == nullptr) {
        const unsigned int base = blockIdx.x * (MMQ_DEC_N_PER_BLOCK * 2);
        __nv_bfloat16 * dst = shared ? sh_out : out + (unsigned long long) slot * N;
        for (unsigned int i = threadIdx.x; i < MMQ_DEC_N_PER_BLOCK * 2; i += blockDim.x) {
            if (base + i < N) dst[base + i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const unsigned char * scale = shared ? sh_down_scale : nullptr;
    const float scale2 = shared ? sh_down_s2 : down_scale2[expert];
    const __nv_bfloat16 * g = shared ? sh_gate : gate_out + (unsigned long long) slot * K;
    const __nv_bfloat16 * u = shared ? sh_up : up_out + (unsigned long long) slot * K;

    extern __shared__ float act[];
    for (unsigned int i = threadIdx.x; i < K; i += blockDim.x) {
        float gf = fminf(fmaxf(__bfloat162float(g[i]), -10.0f), 10.0f);
        float uf = fminf(fmaxf(__bfloat162float(u[i]), -10.0f), 10.0f);
        act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT_MMQ_DEC[threadIdx.x];
    __syncthreads();

    const unsigned int threads_per_out = MMQ_DEC_BLOCK / MMQ_DEC_N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (MMQ_DEC_N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const unsigned int groups = K / 16;
    const unsigned int row_bytes = K * 9 / 16;
    const unsigned char * row1 = packed + (unsigned long long) n1 * row_bytes;
    const unsigned char * row2 = packed + (unsigned long long) n2 * row_bytes;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k16 = lane; k16 < groups; k16 += threads_per_out) {
        const unsigned int base = k16 * 16;
        if (shared) {
            const unsigned char * p1 = packed + (unsigned long long) n1 * (K / 2) + k16 * 8;
            const unsigned char * p2 = packed + (unsigned long long) n2 * (K / 2) + k16 * 8;
            const float sc1 = mmq_dec_e4m3(scale[(unsigned long long) n1 * groups + k16]) * scale2;
            const float sc2 = n2 < N ? mmq_dec_e4m3(scale[(unsigned long long) n2 * groups + k16]) * scale2 : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned char w1 = p1[j], w2 = n2 < N ? p2[j] : 0;
                acc1 += act[base + 2*j] * lut[w1 & 15] * sc1 + act[base + 2*j+1] * lut[w1 >> 4] * sc1;
                acc2 += act[base + 2*j] * lut[w2 & 15] * sc2 + act[base + 2*j+1] * lut[w2 >> 4] * sc2;
            }
        } else {
            const unsigned char * p1 = mmq_dec_group_ptr(row1, k16);
            const unsigned char * p2 = mmq_dec_group_ptr(row2, k16);
            const float sc1 = mmq_dec_e4m3(mmq_dec_group_scale(row1, k16, K)) * scale2;
            const float sc2 = n2 < N ? mmq_dec_e4m3(mmq_dec_group_scale(row2, k16, K)) * scale2 : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned char w1 = p1[j], w2 = n2 < N ? p2[j] : 0;
                acc1 += act[base+j] * lut[w1 & 15] * sc1 + act[base+8+j] * lut[w1 >> 4] * sc1;
                acc2 += act[base+j] * lut[w2 & 15] * sc2 + act[base+8+j] * lut[w2 >> 4] * sc2;
            }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc1 += __shfl_down_sync(0xffffffff, acc1, d);
    __nv_bfloat16 * dst = shared ? sh_out : out + (unsigned long long) slot * N;
    if (lane == 0) dst[n1] = __float2bfloat16(acc1);
    if (n2 < N) {
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, d);
        if (lane == 0) dst[n2] = __float2bfloat16(acc2);
    }
}
