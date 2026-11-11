// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the deepseek-v4-flash `mla_absorbed.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../deepseek-v4-flash/nvfp4/mla_absorbed.cu"

// Exact-row twins used by GLM-5's C2/C3/C4 decode and K=5 MTP verify paths.
// Activations share the same per-head weight matrix, so each loaded weight
// feeds all row accumulators instead of being fetched by independent launches.
//
// input:  [ROWS, num_heads, input_head_stride], with input_row_stride between rows
// output: [ROWS, num_heads, output_head_stride], with output_row_stride between rows
template <unsigned int ROWS>
__device__ __forceinline__ void mla_batched_gemv_batch_impl(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int N_out,
    unsigned int K,
    unsigned int input_head_stride,
    unsigned int output_head_stride,
    unsigned int input_row_stride,
    unsigned int output_row_stride
) {
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = tid / threads_per_out;
    const unsigned int lane = tid % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = n2 < N_out;

    const __nv_bfloat16* B = weight + (unsigned long long)head * N_out * K;
    const unsigned int K4 = K / 4;
    float acc1[ROWS] = {};
    float acc2[ROWS] = {};

    for (unsigned int k4 = lane; k4 < K4; k4 += threads_per_out) {
        const unsigned int base_k = k4 * 4;
        const float w10 = __bfloat162float(B[n1 * K + base_k]);
        const float w11 = __bfloat162float(B[n1 * K + base_k + 1]);
        const float w12 = __bfloat162float(B[n1 * K + base_k + 2]);
        const float w13 = __bfloat162float(B[n1 * K + base_k + 3]);
        float w20 = 0.0f, w21 = 0.0f, w22 = 0.0f, w23 = 0.0f;
        if (have_n2) {
            w20 = __bfloat162float(B[n2 * K + base_k]);
            w21 = __bfloat162float(B[n2 * K + base_k + 1]);
            w22 = __bfloat162float(B[n2 * K + base_k + 2]);
            w23 = __bfloat162float(B[n2 * K + base_k + 3]);
        }

        #pragma unroll
        for (unsigned int row = 0; row < ROWS; ++row) {
            const __nv_bfloat16* A = input
                + (unsigned long long)row * input_row_stride
                + (unsigned long long)head * input_head_stride;
            const unsigned long long av = ((const unsigned long long*)A)[k4];
            const unsigned int lo = (unsigned int)av;
            const unsigned int hi = (unsigned int)(av >> 32);
            __nv_bfloat16 tmp;
            *(unsigned short*)&tmp = (unsigned short)(lo & 0xFFFF);
            const float a0 = __bfloat162float(tmp);
            *(unsigned short*)&tmp = (unsigned short)(lo >> 16);
            const float a1 = __bfloat162float(tmp);
            *(unsigned short*)&tmp = (unsigned short)(hi & 0xFFFF);
            const float a2 = __bfloat162float(tmp);
            *(unsigned short*)&tmp = (unsigned short)(hi >> 16);
            const float a3 = __bfloat162float(tmp);
            acc1[row] += a0 * w10 + a1 * w11 + a2 * w12 + a3 * w13;
            if (have_n2) {
                acc2[row] += a0 * w20 + a1 * w21 + a2 * w22 + a3 * w23;
            }
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (unsigned int row = 0; row < ROWS; ++row) {
            acc1[row] += __shfl_down_sync(0xFFFFFFFF, acc1[row], offset);
            if (have_n2) acc2[row] += __shfl_down_sync(0xFFFFFFFF, acc2[row], offset);
        }
    }

    __shared__ float s_partial[ROWS][N_PER_BLOCK * 2][2];
    const unsigned int warp_in_out = (tid % threads_per_out) / WARP_SIZE;
    const unsigned int lane_in_warp = tid % WARP_SIZE;
    if (lane_in_warp == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < ROWS; ++row) {
            s_partial[row][local_out * 2][warp_in_out] = acc1[row];
            if (have_n2) s_partial[row][local_out * 2 + 1][warp_in_out] = acc2[row];
        }
    }
    __syncthreads();

    if (lane_in_warp == 0 && warp_in_out == 0) {
        #pragma unroll
        for (unsigned int row = 0; row < ROWS; ++row) {
            float sum1 = 0.0f, sum2 = 0.0f;
            #pragma unroll
            for (unsigned int w = 0; w < threads_per_out / WARP_SIZE; ++w) {
                sum1 += s_partial[row][local_out * 2][w];
                if (have_n2) sum2 += s_partial[row][local_out * 2 + 1][w];
            }
            __nv_bfloat16* C = output
                + (unsigned long long)row * output_row_stride
                + (unsigned long long)head * output_head_stride;
            C[n1] = __float2bfloat16(sum1);
            if (have_n2) C[n2] = __float2bfloat16(sum2);
        }
    }
}

#define MLA_BATCH_ARGS \
    const __nv_bfloat16* __restrict__ input, \
    const __nv_bfloat16* __restrict__ weight, \
    __nv_bfloat16* __restrict__ output, \
    unsigned int N_out, unsigned int K, \
    unsigned int input_head_stride, unsigned int output_head_stride, \
    unsigned int input_row_stride, unsigned int output_row_stride

#define MLA_BATCH_CALL(ROWS) \
    mla_batched_gemv_batch_impl<ROWS>(input, weight, output, N_out, K, \
        input_head_stride, output_head_stride, input_row_stride, output_row_stride)

extern "C" __global__ void mla_batched_gemv_batch2(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(2);
}

extern "C" __global__ void mla_batched_gemv_batch3(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(3);
}

extern "C" __global__ void mla_batched_gemv_batch4(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(4);
}

extern "C" __global__ void mla_batched_gemv_batch5(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(5);
}

extern "C" __global__ void mla_batched_gemv_batch6(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(6);
}

extern "C" __global__ void mla_batched_gemv_batch7(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(7);
}

extern "C" __global__ void mla_batched_gemv_batch8(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(8);
}

// Staged two-session K5 projection. No serving dispatcher selects this export.
extern "C" __global__ void mla_batched_gemv_batch10(MLA_BATCH_ARGS) {
    MLA_BATCH_CALL(10);
}

#undef MLA_BATCH_ARGS
#undef MLA_BATCH_CALL
