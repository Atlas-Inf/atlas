// SPDX-License-Identifier: AGPL-3.0-only

// PLE tower for qwen4_exp (Qwen3.8-Flash-Next).
//
// The tower gates a shared value by how well each residual stream matches the
// token's n-gram key, then adds local lexical context through a dilated
// depthwise convolution.
//
// The n-gram gather itself is NOT here: that table is 51.2 B parameters and is
// read by row from disk (atlas_core::ngram_table), so the gathered embedding
// arrives as an ordinary activation.
//
// Reference: Qwen4ExpTextPLELayer in modeling_qwen4_exp.py, and the CPU oracle
// in atlas_core::qwen4exp_reference (checked against it at 5.1e-7).

#include <cuda_bf16.h>

__device__ __forceinline__ float q4e_ple_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float q4e_ple_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
    return v;
}

// Per-stream gate, then scale the shared value by it.
//
//   dot_s   = <key_normed[s], query_normed[s]> / sqrt(hidden)
//   gate_s  = sign(dot_s) * sqrt(max(|dot_s|, 1e-6))     <- SIGNED sqrt, floored
//   out[s,h] = sigmoid(gate_s) * value[h]
//
// The signed square root is the step that looks like a typo and is not: it
// compresses the gate's dynamic range while keeping its sign, and dropping the
// sign leaves a model that still produces fluent text.
//
// Grid: (num_tokens, hc_count, 1)  Block: (min(hidden, 1024), 1, 1)
extern "C" __global__ void q4e_ple_gate(
    const __nv_bfloat16* __restrict__ key_normed,    // [num_tokens, hc_count * hidden]
    const __nv_bfloat16* __restrict__ query_normed,  // [num_tokens, hc_count * hidden]
    const __nv_bfloat16* __restrict__ value,         // [num_tokens, hidden]
    __nv_bfloat16* __restrict__ out,                  // [num_tokens, hc_count * hidden]
    unsigned int hidden,
    unsigned int hc_count
) {
    unsigned int token = blockIdx.x;
    unsigned int stream = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned long long wide = (unsigned long long)hc_count * hidden;
    unsigned long long base = (unsigned long long)token * wide + (unsigned long long)stream * hidden;

    const __nv_bfloat16* k = key_normed + base;
    const __nv_bfloat16* q = query_normed + base;
    const __nv_bfloat16* v = value + (unsigned long long)token * hidden;
    __nv_bfloat16* o = out + base;

    float partial = 0.0f;
    for (unsigned int h = tid; h < hidden; h += blockDim.x) {
        partial += __bfloat162float(k[h]) * __bfloat162float(q[h]);
    }

    partial = q4e_ple_warp_sum(partial);
    __shared__ float warp_sums[32];
    unsigned int lane = tid & 31;
    unsigned int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        val = q4e_ple_warp_sum(val);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float dot = warp_sums[0] / sqrtf((float)hidden);
    float magnitude = fabsf(dot);
    if (magnitude < 1e-6f) magnitude = 1e-6f;
    float gate = sqrtf(magnitude) * (dot < 0.0f ? -1.0f : 1.0f);
    float scale = q4e_ple_sigmoid(gate);

    for (unsigned int h = tid; h < hidden; h += blockDim.x) {
        o[h] = __float2bfloat16(scale * __bfloat162float(v[h]));
    }
}

// Dilated depthwise causal conv with SiLU, added onto the gated value.
//
//   out[t,c] = gated[t,c] + silu( sum_tap w[c,tap] * normed[t - (K-1-tap)*D, c] )
//
// Dilation D is `ngram_size`, so the state this needs is (K-1)*D wide -- 9 on
// the published model, not K-1 = 3. Sizing it as an undilated conv silently
// truncates the receptive field.
//
// Grid: (num_tokens, 1, 1)  Block: (min(wide, 1024), 1, 1)
extern "C" __global__ void q4e_ple_conv_add(
    const __nv_bfloat16* __restrict__ normed,  // [num_tokens, wide] conv input
    const __nv_bfloat16* __restrict__ weight,  // [wide, kernel] depthwise
    __nv_bfloat16* __restrict__ gated,          // [num_tokens, wide] in/out
    unsigned int wide,
    unsigned int kernel_size,
    unsigned int dilation
) {
    unsigned int token = blockIdx.x;
    __nv_bfloat16* g = gated + (unsigned long long)token * wide;

    for (unsigned int c = threadIdx.x; c < wide; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int tap = 0; tap < kernel_size; ++tap) {
            unsigned int back = (kernel_size - 1 - tap) * dilation;
            if (token >= back) {
                unsigned long long src = (unsigned long long)(token - back) * wide + c;
                acc += __bfloat162float(weight[(unsigned long long)c * kernel_size + tap]) *
                       __bfloat162float(normed[src]);
            }
        }
        float activated = acc * q4e_ple_sigmoid(acc); // SiLU
        g[c] = __float2bfloat16(__bfloat162float(g[c]) + activated);
    }
}
