// SPDX-License-Identifier: AGPL-3.0-only
//
// Plain BF16 matrix-vector multiply for unquantized weights:
//
//   y[n] = sum_k(W[n, k] * x[k])
//
// The Qwen3.5-VL vision tower stores its `attn.qkv`, `attn.proj`,
// `mlp.linear_fc1/2` weights as `BF16 [out, in]` with no
// quantization, so the MLX-int8 fused-dequant kernels don't apply
// here. Caller adds the bias separately (the vision tower's linear
// layers have an explicit `.bias` tensor).
//
// One threadgroup per output row; threads stride over K with a
// simdgroup → cross-simdgroup reduction matching `mlx_int8_gemv`.
// FP32 accumulation throughout.
//
// Layout:
//   w : bfloat [N, K]
//   x : bfloat [K]
//   y : bfloat [N]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void dense_gemv_bf16(
    constant uint &N        [[buffer(0)]],
    constant uint &K        [[buffer(1)]],
    device const bfloat *w  [[buffer(2)]],
    device const bfloat *x  [[buffer(3)]],
    device bfloat       *y  [[buffer(4)]],
    uint   row     [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }
    threadgroup float partial[MAX_SIMDGROUPS];

    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        float wv = float(w[row * K + k]);
        float xv = float(x[k]);
        acc += wv * xv;
    }

    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v);
        }
    }
}
