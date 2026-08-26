// SPDX-License-Identifier: AGPL-3.0-only
//
// 3-D non-overlapping convolution for vision-transformer patch
// embedding. Stride equals kernel size in every spatial dimension,
// so this is equivalent to "tile the image into kT × kH × kW
// patches, then matmul each patch by the kernel" — but expressed
// as a direct conv so the weight layout matches Qwen3.5-VL's
// `vision_tower.patch_embed.proj.weight` exactly:
//
//   weight: bfloat [out_channels, kT, kH, kW, in_channels]
//   bias:   bfloat [out_channels]
//   input:  bfloat [in_channels, T, H, W]
//   output: bfloat [out_channels, T_out, H_out, W_out]
//
// where `T_out = T / kT`, `H_out = H / kH`, `W_out = W / kW`.
//
// One thread per output cell. Grid is `(out_channels,
// T_out * H_out * W_out, 1)`. The flat second axis lets a single
// 2-D grid cover the spatial volume without a 4-D dispatch.

#include <metal_stdlib>
using namespace metal;

kernel void conv3d_patch_embed(
    constant uint &out_channels [[buffer(0)]],
    constant uint &in_channels  [[buffer(1)]],
    constant uint &kt           [[buffer(2)]],
    constant uint &kh           [[buffer(3)]],
    constant uint &kw           [[buffer(4)]],
    constant uint &t_out        [[buffer(5)]],
    constant uint &h_out        [[buffer(6)]],
    constant uint &w_out        [[buffer(7)]],
    device const bfloat *input  [[buffer(8)]],
    device const bfloat *weight [[buffer(9)]],
    device const bfloat *bias   [[buffer(10)]],
    device bfloat       *output [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint c_out = gid.x;
    uint flat  = gid.y;
    if (c_out >= out_channels || flat >= t_out * h_out * w_out) {
        return;
    }
    uint w_o = flat % w_out;
    uint h_o = (flat / w_out) % h_out;
    uint t_o = flat / (h_out * w_out);

    // Spatial dimensions of the input volume (kT × kH × kW patches
    // pack with stride == kernel, so input dims = output dims × kernel).
    uint t_in = t_out * kt;
    uint h_in = h_out * kh;
    uint w_in = w_out * kw;

    // Bias is broadcast across every spatial output cell of channel c_out.
    float acc = float(bias[c_out]);
    for (uint dt = 0; dt < kt; ++dt) {
        uint t_idx = t_o * kt + dt;
        for (uint dh = 0; dh < kh; ++dh) {
            uint h_idx = h_o * kh + dh;
            for (uint dw = 0; dw < kw; ++dw) {
                uint w_idx = w_o * kw + dw;
                for (uint ic = 0; ic < in_channels; ++ic) {
                    // weight idx: ((((c_out*kt + dt)*kh + dh)*kw + dw)*in_channels + ic
                    uint w_off = (((c_out * kt + dt) * kh + dh) * kw + dw)
                                  * in_channels + ic;
                    // input idx: ((ic * T + t_idx) * H + h_idx) * W + w_idx
                    uint i_off = ((ic * t_in + t_idx) * h_in + h_idx) * w_in + w_idx;
                    acc += float(weight[w_off]) * float(input[i_off]);
                }
            }
        }
    }
    uint out_idx = ((c_out * t_out + t_o) * h_out + h_o) * w_out + w_o;
    output[out_idx] = bfloat(acc);
}
