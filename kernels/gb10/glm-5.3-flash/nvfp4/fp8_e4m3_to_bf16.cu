// SPDX-License-Identifier: AGPL-3.0-only
// Exact E4M3 -> BF16 widening (every E4M3 value is representable in BF16), so
// an FP8 weight cache can feed a BF16 tensor-core GEMM with unchanged math.
// Eight values per thread. `count8` = element count / 8.
// Grid: (ceil(count8 / 256), 1, 1)  Block: (256, 1, 1).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void fp8_e4m3_to_bf16(
    const uint2* __restrict__ src,
    uint4* __restrict__ dst,
    unsigned long long count8
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count8) return;
    const uint2 v = src[i];
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    unsigned int out[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        __nv_fp8_e4m3 lo, hi;
        lo.__x = b[2 * j];
        hi.__x = b[2 * j + 1];
        const __nv_bfloat16 l = __float2bfloat16(static_cast<float>(lo));
        const __nv_bfloat16 h = __float2bfloat16(static_cast<float>(hi));
        out[j] = (unsigned int)__bfloat16_as_ushort(l) | ((unsigned int)__bfloat16_as_ushort(h) << 16);
    }
    dst[i] = make_uint4(out[0], out[1], out[2], out[3]);
}
