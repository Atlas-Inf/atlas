// SPDX-License-Identifier: AGPL-3.0-only

// BF16 residual addition: residual[i] += src[i]
//
// Used in the transformer loop after attention/SSM and FFN blocks.
// Operates in-place on the residual stream.

#include "atlas_pdl.cuh"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_residual_add(
    __nv_bfloat16* __restrict__ residual,  // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    unsigned int n
) {
    atlas_pdl_enter();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = __bfloat162float(residual[i]);
        float s = __bfloat162float(src[i]);
        residual[i] = __float2bfloat16(r + s);
    }
}

// BF16 → FP32 element-wise conversion.
// Used to promote embedding output (BF16) to FP32 residual stream.
extern "C" __global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,  // [n] BF16
    float* __restrict__ dst,                 // [n] FP32
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Separate-input SiLU multiplication: output = SiLU(gate) * up.
//
// Unlike fused_silu_mul in dense_gemm_bf16.cu which expects interleaved
// gate_up, this takes separate gate and up buffers. Used by MoE shared
// expert which has separate gate_proj and up_proj.
extern "C" __global__ void silu_mul_separate(
    const __nv_bfloat16* __restrict__ gate,   // [n]
    const __nv_bfloat16* __restrict__ up,     // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        // SiLU(x) = x * sigmoid(x)
        float silu_g = g / (1.0f + expf(-g));
        output[i] = __float2bfloat16(silu_g * u);
    }
}

// Scaled accumulate: output[i] += scale * src[i] (BF16 with f32 scale).
//
// Used for MoE routing: accumulate weighted expert outputs.
extern "C" __global__ void bf16_scaled_add(
    __nv_bfloat16* __restrict__ output,    // [n] — modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n]
    float scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + scale * s);
    }
}

// Sigmoid-gated blend: output = sigmoid_gate * src + output (BF16).
//
// Used for MoE shared expert: blend shared expert output with routed output.
// sigmoid_gate is a single f32 value applied to all elements.
extern "C" __global__ void bf16_sigmoid_blend(
    __nv_bfloat16* __restrict__ output,    // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,  // [n] — shared expert output
    float sigmoid_gate,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_gate * s);
    }
}

// Element-wise sigmoid gate: output[i] = input[i] * sigmoid(gate[i]).
//
// Used for gated attention: attn_output = attn_output * sigmoid(q_gate).
// gate and input are separate BF16 vectors of the same size.
extern "C" __global__ void sigmoid_gate_mul(
    const __nv_bfloat16* __restrict__ input,   // [n]
    const __nv_bfloat16* __restrict__ gate,    // [n]
    __nv_bfloat16* __restrict__ output,        // [n]
    unsigned int n
) {
    atlas_pdl_enter();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[i]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// Batched sigmoid gate multiply across multiple tokens.
// Replaces per-token kernel launches with a single launch for all tokens.
//
// input is contiguous [num_tokens, dim]. gate is strided [num_tokens, gate_stride]
// (gate data starts at gate pointer, with gate_stride elements between tokens).
// output[t * dim + d] = input[t * dim + d] * sigmoid(gate[t * gate_stride + d])
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_batched(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, dim] contiguous
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, gate_stride] strided
    __nv_bfloat16* __restrict__ output,        // [num_tokens, dim] contiguous
    unsigned int dim,
    unsigned int gate_stride,
    unsigned int total_elements               // num_tokens * dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int t = i / dim;
        unsigned int d = i % dim;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * gate_stride + d]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

// BF16 concatenation: out[0..N] = a[0..N], out[N..2N] = b[0..N].
//
// Used by MTP head to concatenate normed embedding + normed hidden state
// before the FC projection. Both inputs are [N] BF16, output is [2N] BF16.
extern "C" __global__ void bf16_concat(
    const __nv_bfloat16* __restrict__ a,      // [N]
    const __nv_bfloat16* __restrict__ b,      // [N]
    __nv_bfloat16* __restrict__ out,          // [2N]
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i];
        out[N + i] = b[i];
    }
}

// Row-batched BF16 concatenation: for each row r < rows,
// out[r, 0..N] = a[r, 0..N] and out[r, N..2N] = b[r, 0..N].
// One launch for the MTP drafter prefill's [rows, 2N] fc input, instead of
// one bf16_concat launch per row.
// Grid: (ceil(N/256), rows, 1)  Block: (256, 1, 1)
extern "C" __global__ void bf16_concat_rows(
    const __nv_bfloat16* __restrict__ a,      // [rows, N]
    const __nv_bfloat16* __restrict__ b,      // [rows, N]
    __nv_bfloat16* __restrict__ out,          // [rows, 2N]
    unsigned int N
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t r = blockIdx.y;
    if (i < N) {
        out[r * 2 * N + i] = a[r * N + i];
        out[r * 2 * N + N + i] = b[r * N + i];
    }
}

// Device-pointer sigmoid blend: reads gate scalar from device BF16 pointer.
//
// output[i] += sigmoid(bf16_to_f32(*gate_ptr)) * src[i]
//
// Eliminates the D2H + CPU sigmoid for shared expert gate.
// Each block reads the single BF16 scalar and computes sigmoid in shared mem.
// Per-head sigmoid gate multiply with broadcast over head_dim.
// Step 3.7 attention gate: g_proj produces one scalar per head,
// which is sigmoid-gated and broadcast-multiplied across head_dim.
//
// input/output: [num_tokens, nq * hd] contiguous BF16
// gate: [num_tokens, nq] contiguous BF16 (one value per head)
//
// output[t, h, d] = input[t, h, d] * sigmoid(gate[t, h])
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_head_broadcast(
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, nq * hd]
    const __nv_bfloat16* __restrict__ gate,    // [num_tokens, nq]
    __nv_bfloat16* __restrict__ output,        // [num_tokens, nq * hd]
    unsigned int nq,
    unsigned int hd,
    unsigned int total_elements               // num_tokens * nq * hd
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int dim = nq * hd;
        unsigned int t = i / dim;
        unsigned int within_token = i % dim;
        unsigned int head_idx = within_token / hd;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * nq + head_idx]);
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = __float2bfloat16(x * sigmoid_g);
    }
}

extern "C" __global__ void softplus_gate_mul_head_broadcast(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    unsigned int nq,
    unsigned int hd,
    unsigned int total_elements
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        unsigned int dim = nq * hd;
        unsigned int t = i / dim;
        unsigned int head_idx = (i % dim) / hd;
        float x = __bfloat162float(input[i]);
        float g = __bfloat162float(gate[t * nq + head_idx]);
        float softplus_g = fmaxf(g, 0.0f) + log1pf(expf(-fabsf(g)));
        output[i] = __float2bfloat16(x * softplus_g);
    }
}

extern "C" __global__ void bf16_sigmoid_blend_device(
    __nv_bfloat16* __restrict__ output,             // [n] — routed output, modified in-place
    const __nv_bfloat16* __restrict__ src,          // [n] — shared expert output
    const __nv_bfloat16* __restrict__ gate_ptr,     // device pointer to single BF16 scalar
    unsigned int n
) {
    __shared__ float sigmoid_val;
    if (threadIdx.x == 0) {
        float g = __bfloat162float(*gate_ptr);
        sigmoid_val = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float o = __bfloat162float(output[i]);
        float s = __bfloat162float(src[i]);
        output[i] = __float2bfloat16(o + sigmoid_val * s);
    }
}

// Round-trip `rows` rows of `len` contiguous elements in place through a
// narrower storage format, as a compressed score-pack cache would hold them
// (ATLAS_SCORE_PACK_ROUNDTRIP): `fmt` 0 = E4M3 with a per-row FP32 scale
// (amax / 448), 1 = BF16, 2 = FP16. `dtype`: 0 = FP32 data, 1 = BF16 data.
// One warp a row. Grid: (ceil(rows / 8), 1, 1)  Block: (256, 1, 1)
#include <cuda_fp8.h>
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256) roundtrip_rows(
    void* __restrict__ data, unsigned int rows, unsigned int len, unsigned int dtype, unsigned int fmt
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float* f = (float*)data + row * len;
    __nv_bfloat16* b = (__nv_bfloat16*)data + row * len;
    auto load = [&](unsigned int i) { return dtype == 0 ? f[i] : __bfloat162float(b[i]); };
    auto store = [&](unsigned int i, float v) {
        if (dtype == 0) f[i] = v; else b[i] = __float2bfloat16(v);
    };
    float amax = 0.0f;
    for (unsigned int i = lane; i < len; i += 32) amax = fmaxf(amax, fabsf(load(i)));
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    for (unsigned int i = lane; i < len; i += 32) {
        const float x = load(i);
        float y;
        if (fmt == 0) y = float(__nv_fp8_e4m3(x / scale)) * scale;
        else if (fmt == 1) y = __bfloat162float(__float2bfloat16(x));
        else y = __half2float(__float2half(x));
        store(i, y);
    }
}

// The score pack store's FP8 rows (score_store.rs): `rows` rows of `len`
// contiguous elements (`dtype` 0 = FP32, 1 = BF16) as E4M3 codes [rows, len]
// and a per-row FP32 scale (amax / 448), and back. rows_from_fp8 after
// rows_to_fp8 gives exactly roundtrip_rows' fmt 0. One warp a row.
// Grid: (ceil(rows / 8), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) rows_to_fp8(
    const void* __restrict__ src, __nv_fp8_e4m3* __restrict__ codes, float* __restrict__ scales,
    unsigned int rows, unsigned int len, unsigned int dtype
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const float* f = (const float*)src + row * len;
    const __nv_bfloat16* b = (const __nv_bfloat16*)src + row * len;
    auto load = [&](unsigned int i) { return dtype == 0 ? f[i] : __bfloat162float(b[i]); };
    float amax = 0.0f;
    for (unsigned int i = lane; i < len; i += 32) amax = fmaxf(amax, fabsf(load(i)));
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    if (lane == 0) scales[row] = scale;
    for (unsigned int i = lane; i < len; i += 32) codes[row * len + i] = __nv_fp8_e4m3(load(i) / scale);
}

extern "C" __global__ void __launch_bounds__(256) rows_from_fp8(
    const __nv_fp8_e4m3* __restrict__ codes, const float* __restrict__ scales, void* __restrict__ dst,
    unsigned int rows, unsigned int len, unsigned int dtype
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    const float scale = scales[row];
    float* f = (float*)dst + row * len;
    __nv_bfloat16* b = (__nv_bfloat16*)dst + row * len;
    for (unsigned int i = lane; i < len; i += 32) {
        const float y = float(codes[row * len + i]) * scale;
        if (dtype == 0) f[i] = y; else b[i] = __float2bfloat16(y);
    }
}
