// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Multiply 16 routed token rows by 32 output channels per block. The
// checkpoint stores each expert separately in [N,K/2] order. E2M1 times
// E4M3 is exactly representable in BF16. Apply the per-expert FP32 scale
// after accumulating the dot product, the convention used by Marlin.
static __device__ __forceinline__ unsigned ling_dequant_pair(unsigned char packed, float scale) {
    unsigned pair;
    asm("{ .reg .b8 b; cvt.u8.u32 b, %1; cvt.rn.f16x2.e2m1x2 %0, b; }"
        : "=r"(pair) : "r"(static_cast<unsigned>(packed)));
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&pair));
    const __nv_bfloat162 result = __floats2bfloat162_rn(v.x * scale, v.y * scale);
    return *reinterpret_cast<const unsigned*>(&result);
}

extern "C" __global__ void ling_a16_grouped(
    const __nv_bfloat16* input, const unsigned long long* weights,
    const unsigned long long* scales, const float* globals,
    const int* sorted_tokens, const int* offsets, __nv_bfloat16* output,
    unsigned n, unsigned k) {
    const unsigned expert = blockIdx.z;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    if (begin + static_cast<int>(blockIdx.y * 16) >= end) return;
    const unsigned lane = threadIdx.x & 31;
    const unsigned group = lane / 4, thread = lane % 4;
    const unsigned column = blockIdx.x * 32 + (threadIdx.x / 32) * 8;
    const unsigned feature = column + group < n ? column + group : 0;
    const auto* wp = reinterpret_cast<const unsigned char*>(weights[expert]) + size_t(feature) * (k / 2);
    const auto* sp = reinterpret_cast<const __nv_fp8_e4m3*>(scales[expert]) + size_t(feature) * (k / 16);
    const float global = globals[expert];
    for (int first = begin + blockIdx.y * 16; first < end; first += gridDim.y * 16) {
        const int r0 = first + group, r1 = r0 + 8;
        const int i0 = r0 < end ? (sorted_tokens ? sorted_tokens[r0] : r0) : 0;
        const int i1 = r1 < end ? (sorted_tokens ? sorted_tokens[r1] : r1) : 0;
        const auto* a0 = reinterpret_cast<const unsigned*>(input + size_t(i0) * k);
        const auto* a1 = reinterpret_cast<const unsigned*>(input + size_t(i1) * k);
        float d0=0, d1=0, d2=0, d3=0;
        for (unsigned base = 0; base < k; base += 16) {
            const float scale = float(sp[base / 16]);
            const unsigned off = base / 2 + thread;
            const unsigned b0 = ling_dequant_pair(wp[off], scale);
            const unsigned b1 = ling_dequant_pair(wp[off+4], scale);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0[off]), "r"(a1[off]), "r"(a0[off+4]), "r"(a1[off+4]), "r"(b0), "r"(b1));
        }
        const unsigned j = column + thread * 2;
        if (r0 < end && j < n) {
            output[size_t(r0)*n+j] = __float2bfloat16(d0 * global);
            if (j+1 < n) output[size_t(r0)*n+j+1] = __float2bfloat16(d1 * global);
        }
        if (r1 < end && j < n) {
            output[size_t(r1)*n+j] = __float2bfloat16(d2 * global);
            if (j+1 < n) output[size_t(r1)*n+j+1] = __float2bfloat16(d3 * global);
        }
    }
}
