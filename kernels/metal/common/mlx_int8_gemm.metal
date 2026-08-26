// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path matrix multiply with on-the-fly MLX 8-bit dequant:
//
//   Y[m, n] = sum over k of (X[m, k] * W_dequant[n, k])
//
// One thread per (m, n) output element. Loops over K with FP32
// accumulation, dequantizing each W byte inline. This is the
// straightforward correctness reference; performance work
// (simdgroup_matrix tiling, K-block dequant + reuse) lives in a
// follow-on PR — the call shape stays stable.
//
// Layout:
//   x      : bfloat [M, K]
//   packed : uint32 [N, K / 4]
//   scales : bfloat [N, K / group_size]
//   biases : bfloat [N, K / group_size]
//   y      : bfloat [M, N]

#include <metal_stdlib>
using namespace metal;

kernel void mlx_int8_gemm(
    constant uint &M          [[buffer(0)]],
    constant uint &N          [[buffer(1)]],
    constant uint &K          [[buffer(2)]],
    constant uint &group_size [[buffer(3)]],
    device const bfloat *x      [[buffer(4)]],
    device const uint   *packed [[buffer(5)]],
    device const bfloat *scales [[buffer(6)]],
    device const bfloat *biases [[buffer(7)]],
    device bfloat       *y      [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.y;
    uint n = gid.x;
    if (m >= M || n >= N) {
        return;
    }

    uint groups_per_row = K / group_size;
    float acc = 0.0;
    for (uint k = 0; k < K; ++k) {
        uint word = packed[n * (K / 4u) + (k >> 2)];
        uint byte = (word >> ((k & 3u) * 8u)) & 0xFFu;
        uint g    = k / group_size;
        float s   = float(scales[n * groups_per_row + g]);
        float b   = float(biases[n * groups_per_row + g]);
        float w   = float(byte) * s + b;
        acc += float(x[m * K + k]) * w;
    }
    y[m * N + n] = bfloat(acc);
}
