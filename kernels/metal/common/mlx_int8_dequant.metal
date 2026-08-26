// SPDX-License-Identifier: AGPL-3.0-only
//
// MLX 8-bit (`mlx-community/...-MLX-8bit`) → BF16 dequantization.
//
// MLX packs 4 unsigned 8-bit weights into each `uint32`. Per group of
// `group_size` columns (default 64), one bf16 scale and one bf16 bias
// determine the affine recovery:
//
//   w[r, c] = byte(packed[r, c/4], c%4) * scales[r, c/group_size]
//                                       + biases[r, c/group_size]
//
// Layout:
//   packed : uint32  [out_features, in_features / 4]
//   scales : bfloat  [out_features, in_features / group_size]
//   biases : bfloat  [out_features, in_features / group_size]
//   out    : bfloat  [out_features, in_features]

#include <metal_stdlib>
using namespace metal;

kernel void mlx_int8_dequant(
    constant uint &out_features [[buffer(0)]],
    constant uint &in_features  [[buffer(1)]],
    constant uint &group_size   [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device bfloat       *out    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.y;
    uint c = gid.x;
    if (r >= out_features || c >= in_features) {
        return;
    }

    uint  word          = packed[r * (in_features / 4u) + (c >> 2)];
    uint  byte          = (word >> ((c & 3u) * 8u)) & 0xFFu;
    uint  groups_per_row = in_features / group_size;
    uint  group_idx     = c / group_size;

    float s = float(scales[r * groups_per_row + group_idx]);
    float b = float(biases[r * groups_per_row + group_idx]);

    out[r * in_features + c] = bfloat(float(byte) * s + b);
}
