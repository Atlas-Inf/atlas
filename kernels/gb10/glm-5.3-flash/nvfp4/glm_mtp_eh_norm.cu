// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5 MTP input preparation for one draft row.
//
// The reference path copies embed[token] to scratch, then launches the
// vanilla RMSNorm kernel once for the embedding and once for target_hidden.
// These two rows are independent.  One two-CTA launch can read the embedding
// table directly and write the same concatenated [enorm(embed), hnorm(hidden)]
// buffer.  Each CTA deliberately preserves rms_norm_vanilla's thread mapping,
// reduction tree, FP32 operation order, and BF16 conversion.

#include <cuda_bf16.h>

__device__ __forceinline__ void glm_mtp_unpack_bf16x2(
    unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int glm_mtp_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float glm_mtp_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Grid: (2, 1, 1), block: (min(hidden_size, 1024), 1, 1).
// block 0 normalizes embed_table[token], block 1 normalizes target_hidden.
extern "C" __global__ void glm_mtp_eh_norm(
    const __nv_bfloat16* __restrict__ embed_table,
    unsigned int token,
    const __nv_bfloat16* __restrict__ target_hidden,
    const __nv_bfloat16* __restrict__ enorm,
    const __nv_bfloat16* __restrict__ hnorm,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* x = row == 0
        ? embed_table + (size_t)token * hidden_size
        : target_hidden;
    const __nv_bfloat16* weight = row == 0 ? enorm : hnorm;
    __nv_bfloat16* out = output + (size_t)row * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        glm_mtp_unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = glm_mtp_warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = glm_mtp_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        glm_mtp_unpack_bf16x2(x32[i], xv0, xv1);
        glm_mtp_unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = glm_mtp_pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        const float val = __bfloat162float(x[hidden_size - 1]);
        const float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}
