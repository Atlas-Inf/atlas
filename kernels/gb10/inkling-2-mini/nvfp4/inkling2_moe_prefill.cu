// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 routed experts for prefill rows: the pieces around the grouped GEMMs
// (moe_sort_by_expert, inkling2_moe_gemm_{nvfp4,mxfp8}, moe_unpermute_reduce_indexed).
// Reference: transformers PR #51050 `Inkling2SoftcappedSwiGLU` (Apache-2.0).

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// act[r, i] = BF16(tanh(g / cg) * cg * sigmoid(g) * tanh(u / cu) * cu) with g = gu[r, 2i],
// u = gu[r, 2i + 1] (the checkpoint's interleaved w13 rows, so the grouped GEMM output is
// interleaved too), both already BF16 as the reference rounds them.
// grid (ceil(I / 256), rows), block 256
extern "C" __global__ void inkling2_softcap_swiglu(const __nv_bfloat16* __restrict__ gu,
                                                   __nv_bfloat16* __restrict__ act, const unsigned int I,
                                                   const float cg, const float cu) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long r = blockIdx.y;
    if (i >= I) return;
    const __nv_bfloat162 p = ((const __nv_bfloat162*)(gu + r * 2 * I))[i];
    const float g = __low2float(p), u = __high2float(p);
    const float v = tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * tanhf(u / cu) * cu;
    act[r * I + i] = __float2bfloat16(v);
}

// Load time: an overlay's gate_up rows in transformers' order (gate rows 0..half-1, then
// up rows) into the checkpoint's interleaved order (row 2i gate i, row 2i + 1 up i), per
// expert. src/dst [E, 2 * half, row_bytes]. grid (E * 2 * half), block 128.
extern "C" __global__ void inkling2_interleave_rows(const unsigned char* __restrict__ src,
                                                    unsigned char* __restrict__ dst, const unsigned int half,
                                                    const unsigned int row_bytes) {
    const unsigned long long d = blockIdx.x;
    const unsigned long long n = 2ull * half;
    const unsigned long long e = d / n, r = d % n;
    const unsigned long long s = e * n + (r % 2 == 0 ? r / 2 : half + r / 2);
    for (unsigned int b = threadIdx.x; b < row_bytes; b += blockDim.x) dst[d * row_bytes + b] = src[s * row_bytes + b];
}
