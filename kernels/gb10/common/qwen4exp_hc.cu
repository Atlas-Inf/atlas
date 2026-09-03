// SPDX-License-Identifier: AGPL-3.0-only

// Hyper-connection mixing for qwen4_exp (Qwen3.8-Flash-Next).
//
// The residual stream is `hc_count` streams of `hidden_size` concatenated. A
// block collapses it to `hidden_size`, computes, and the result is scattered
// back scaled per stream. These are the three elementwise/reduction steps in
// that collapse; the two projections are ordinary GEMVs.
//
// Every one of them divides by hc_count BEFORE its activation, which is easy
// to reorder and produces a model that still reads fluently.
//
// Reference: Qwen4ExpTextGatedResidual in modeling_qwen4_exp.py, and the CPU
// oracle in atlas_core::qwen4exp_reference (checked against it at 1.6e-7).
//
// NAMED `qwen4exp_hc`, NOT `hyper_connection`. That module already exists --
// kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu, carrying hc_expand
// / hc_pre / hc_post / hc_head -- and it is DeepSeek-V4's Sinkhorn-normalised
// mHC, a DIFFERENT formulation from this low-rank sigmoid gate. Sharing the
// module name let the model shadow win and the lookup fail with "named symbol
// not found"; sharing it the other way would have been worse, since one
// model's weights would have gone through the other's mixing.

#include <cuda_bf16.h>

__device__ __forceinline__ float q4e_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Trunk entry: tile one hidden state across all hc_count residual streams.
//
//   streams[t, s, d] = hidden[t, d]     for every s
//
// The reference does exactly this after the embedding lookup -- the streams
// start identical and only diverge once the first block's injection lands.
// Starting them at zero instead (the obvious alternative) makes the first
// hyper-connection collapse read a zero mean and the model never recovers.
//
// DeepSeek-V4's hc_expand does the same job but lives in that model's shadow,
// not in common/, so it does not resolve for this target. See the module-name
// note above: a shadow beats common/, which is why these are q4e_-prefixed.
//
// Grid: (num_tokens, hc_count, 1)  Block: (min(hidden, 1024), 1, 1)
extern "C" __global__ void q4e_hc_expand(
    const __nv_bfloat16* __restrict__ hidden,  // [num_tokens, hidden]
    __nv_bfloat16* __restrict__ streams,        // [num_tokens, hc_count * hidden]
    unsigned int hidden_size,
    unsigned int hc_count
) {
    unsigned int token = blockIdx.x;
    unsigned int stream = blockIdx.y;
    const __nv_bfloat16* src = hidden + (unsigned long long)token * hidden_size;
    __nv_bfloat16* dst = streams
        + (unsigned long long)token * hc_count * hidden_size
        + (unsigned long long)stream * hidden_size;

    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// silu(x / hc_count), in place, on the low-rank projection output.
//
// Grid: (num_tokens, 1, 1)  Block: (min(lowrank, 1024), 1, 1)
extern "C" __global__ void q4e_hc_lowrank_act(
    __nv_bfloat16* __restrict__ values,   // [num_tokens, lowrank]
    unsigned int lowrank,
    unsigned int hc_count
) {
    unsigned long long base = (unsigned long long)blockIdx.x * lowrank;
    float inv = 1.0f / (float)hc_count;
    for (unsigned int i = threadIdx.x; i < lowrank; i += blockDim.x) {
        float v = __bfloat162float(values[base + i]) * inv;
        values[base + i] = __float2bfloat16(v * q4e_sigmoid(v));
    }
}

// Collapse the streams: out[h] = mean_s( sigmoid(gate[s,h]) * normed[s,h] ).
//
// A MEAN, not a sum. Reading it as a sum is an hc_count-fold error that looks
// like a temperature change rather than a bug.
//
// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
extern "C" __global__ void q4e_hc_stream_mix(
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, hc_count * hidden] pre-sigmoid
    const __nv_bfloat16* __restrict__ normed,  // [num_tokens, hc_count * hidden]
    __nv_bfloat16* __restrict__ out,            // [num_tokens, hidden]
    unsigned int hidden,
    unsigned int hc_count
) {
    unsigned long long wide = (unsigned long long)hc_count * hidden;
    const __nv_bfloat16* g = gate + (unsigned long long)blockIdx.x * wide;
    const __nv_bfloat16* n = normed + (unsigned long long)blockIdx.x * wide;
    __nv_bfloat16* o = out + (unsigned long long)blockIdx.x * hidden;

    float inv = 1.0f / (float)hc_count;
    for (unsigned int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < hc_count; ++s) {
            unsigned long long i = (unsigned long long)s * hidden + h;
            acc += q4e_sigmoid(__bfloat162float(g[i])) * __bfloat162float(n[i]);
        }
        o[h] = __float2bfloat16(acc * inv);
    }
}

// Per-stream injection gains: 2 * sigmoid(x / hc_count).
//
// Centred on 1, not on 0.5 — the factor of two is what makes "no injection"
// the identity rather than a halving.
//
// Grid: (num_tokens, 1, 1)  Block: (hc_count, 1, 1)
extern "C" __global__ void q4e_hc_injection(
    const __nv_bfloat16* __restrict__ raw,  // [num_tokens, hc_count]
    __nv_bfloat16* __restrict__ out,         // [num_tokens, hc_count]
    unsigned int hc_count
) {
    unsigned long long base = (unsigned long long)blockIdx.x * hc_count;
    for (unsigned int s = threadIdx.x; s < hc_count; s += blockDim.x) {
        float v = __bfloat162float(raw[base + s]) / (float)hc_count;
        out[base + s] = __float2bfloat16(2.0f * q4e_sigmoid(v));
    }
}

// Scatter a block's hidden-wide output back across the streams, scaled.
//
// out[s,h] += mixed[h] * injection[s]  — accumulating onto the UN-normalised
// hyper input, which is what the residual actually carries.
//
// Grid: (num_tokens, hc_count, 1)  Block: (min(hidden, 1024), 1, 1)
extern "C" __global__ void q4e_hc_scatter_add(
    const __nv_bfloat16* __restrict__ mixed,      // [num_tokens, hidden]
    const __nv_bfloat16* __restrict__ injection,  // [num_tokens, hc_count]
    __nv_bfloat16* __restrict__ residual,          // [num_tokens, hc_count * hidden]
    unsigned int hidden,
    unsigned int hc_count
) {
    unsigned int token = blockIdx.x;
    unsigned int stream = blockIdx.y;
    const __nv_bfloat16* m = mixed + (unsigned long long)token * hidden;
    float gain = __bfloat162float(injection[(unsigned long long)token * hc_count + stream]);
    __nv_bfloat16* r =
        residual + (unsigned long long)token * hc_count * hidden + (unsigned long long)stream * hidden;

    for (unsigned int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float v = __bfloat162float(r[h]) + __bfloat162float(m[h]) * gain;
        r[h] = __float2bfloat16(v);
    }
}
