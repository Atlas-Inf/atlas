// SPDX-License-Identifier: AGPL-3.0-only

// GLM-only register twin of rms_norm_vanilla. It reuses the common file's
// helpers; the legacy exports of this include are contained in this new
// module, and every serving caller of rms_norm_vanilla keeps its module.
#include "../../common/rms_norm_vanilla.cu"

// rms_norm_vanilla with each thread's input and weight pairs loaded once, up
// front, into registers (ATLAS_GLM_DECODE_FUSE): the original reads the row
// for the sum of squares, then the row again and the weight after both
// barriers, each a dependent global load. Every statement, the reduction
// order and the tail handling are the original's, so the output is identical.
// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1); at most
// four pairs a thread (hidden_size <= 8 * blockDim.x, else it traps).
#define RMSV_REGS_PAIRS 4
extern "C" __global__ void rms_norm_vanilla_regs(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    atlas_pdl_enter();
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int half_size = hidden_size / 2;
    if (half_size > RMSV_REGS_PAIRS * blockDim.x) __trap();

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;
    const unsigned int* x32 = (const unsigned int*)x;
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int xv[RMSV_REGS_PAIRS], wv[RMSV_REGS_PAIRS];
    #pragma unroll
    for (unsigned int k = 0; k < RMSV_REGS_PAIRS; k++) {
        const unsigned int i = tid + k * blockDim.x;
        if (i < half_size) {
            xv[k] = x32[i];
            wv[k] = w32[i];
        }
    }

    float sum_sq = 0.0f;
    #pragma unroll
    for (unsigned int k = 0; k < RMSV_REGS_PAIRS; k++) {
        if (tid + k * blockDim.x < half_size) {
            float v0, v1;
            unpack_bf16x2(xv[k], v0, v1);
            sum_sq += v0 * v0 + v1 * v1;
        }
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    unsigned int* out32 = (unsigned int*)out;
    #pragma unroll
    for (unsigned int k = 0; k < RMSV_REGS_PAIRS; k++) {
        const unsigned int i = tid + k * blockDim.x;
        if (i < half_size) {
            float xv0, xv1, wv0, wv1;
            unpack_bf16x2(xv[k], xv0, xv1);
            unpack_bf16x2(wv[k], wv0, wv1);
            out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
        }
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}
