// SPDX-License-Identifier: AGPL-3.0-only
// exllamav3's int8-activation "sq" GEMV for EXL3 mul1 weights (vendored in exl3_vendor/, see NOTICE.md): one
// regular launch, block 256, dynamic shared memory and grid chosen by the host (M6b); `locks` is a zeroed
// workspace whose counters reset themselves. Entry points: bits K in {4,5,6} x rows M in {1,2}, fp32 output.
#include <cuda_bf16.h>
#include "exl3_vendor/exl3_gemv_int8_kernel.cuh"

#define EXL3_SQ_ENTRY(K, M)                                                                                     \
    extern "C" __global__ __launch_bounds__(256) void exl3_int8_sq_k##K##_m##M(                                 \
        const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C, const int size_m,    \
        const int size_k, const int size_n, int* __restrict__ locks, const half* __restrict__ suh,             \
        half* __restrict__ A_had, const half* __restrict__ svh)                                                 \
    { exl3_gemv_int8_sq_kernel<K, M, true, false>(A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); }

EXL3_SQ_ENTRY(4, 1) EXL3_SQ_ENTRY(4, 2) EXL3_SQ_ENTRY(5, 1) EXL3_SQ_ENTRY(5, 2) EXL3_SQ_ENTRY(6, 1) EXL3_SQ_ENTRY(6, 2)

extern "C" __global__ void exl3_f32_to_bf16(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = __float2bfloat16_rn(in[i]);
}
