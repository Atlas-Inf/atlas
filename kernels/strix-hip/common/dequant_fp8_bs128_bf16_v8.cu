// SPDX-License-Identifier: AGPL-3.0-only
// FP8 E4M3 x FP32 128x128 block scale -> BF16, 8 elements per thread.
//
// Same arithmetic as `dequant_fp8_blockscaled_bf16`: E4M3_LUT_DEQ[b] * scale in
// FP32, then __float2bfloat16 RNE. Its LUT is copied verbatim below under a
// module-local name (so this module does not also export that kernel), and the
// output is bit-identical.
//
// That kernel maps ONE element to each thread with 64x4 blocks: fine at load
// time, but on the gfx1151 prefill route (hip_fp8_blas.rs) it runs once per FP8
// projection per chunk and measured 3.8 ms per launch, 7.3% of a 16k 27B prefill
// (strix s69). Eight consecutive k per thread (one 8-byte load, one 16-byte
// store; 8 divides 128, so all eight share a scale) did the same work at ~130 GB/s
// in the s61 bench (~1.1 ms for the 16384x5120 GDN qkvz weight).
//
// Requires K % 8 == 0 and 128x128 FP32 block scales (the route checks both).
#include <cuda_bf16.h>

__device__ __constant__ float E4M3_LUT_DEQ_V8[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

extern "C" __global__ void dequant_fp8_bs128_bf16_v8(
    const unsigned char* __restrict__ fp8_in,   // [N, K] FP8 E4M3 row-major
    const float* __restrict__ scale_inv,        // [ceil(N/128), sk] FP32
    __nv_bfloat16* __restrict__ bf16_out,       // [N, K] BF16 row-major
    unsigned int N,
    unsigned int K,
    unsigned int sk
) {
    const unsigned long long i8 = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 8ull;
    if (i8 >= (unsigned long long)N * K) return;
    const unsigned int n = (unsigned int)(i8 / K), k = (unsigned int)(i8 % K);
    const float scale = scale_inv[(n / 128u) * sk + k / 128u];
    const uint2 raw = *reinterpret_cast<const uint2*>(fp8_in + i8);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&raw);
    __nv_bfloat16 o[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(E4M3_LUT_DEQ_V8[b[j]] * scale);
    *reinterpret_cast<uint4*>(bf16_out + i8) = *reinterpret_cast<const uint4*>(o);
}
