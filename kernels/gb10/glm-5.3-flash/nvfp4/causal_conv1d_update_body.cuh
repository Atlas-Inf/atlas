// SPDX-License-Identifier: AGPL-3.0-only

// Private shared scalar/indexed arithmetic. Preserve operation order and runtime
// dimension arithmetic; see scripts/dev/glm_kda_legacy_reference.cuh for the
// independent pre-extraction numerical oracle.
#pragma once
#include <cuda_bf16.h>

static __device__ __forceinline__ void atlas_causal_conv1d_update_body(
    float* __restrict__ conv_state,             // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ input,    // N tokens, input[t * input_stride + ch]
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // N tokens, output[t * output_stride + ch]
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,          // number of tokens
    unsigned int input_stride,     // BF16 elements between consecutive tokens in input
    unsigned int output_stride     // BF16 elements between consecutive tokens in output
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + ch * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    // Load weights into registers (d_conv = 4 for Qwen3-Next)
    float w_reg[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }

    // Load current sliding window state into registers
    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }

    // Process all tokens sequentially — state stays in registers
    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];

        // Shift state left by 1, insert new value
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;

        // Depthwise convolution
        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];

        // SiLU activation
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
    }

    // Write final state back to global memory
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        state[k] = s[k];
    }
}

