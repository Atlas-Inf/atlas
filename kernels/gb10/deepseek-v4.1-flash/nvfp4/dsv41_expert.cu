// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the experts of one token in three launches.
//
// A token runs seven experts: the six the router chose, FP4, and the shared
// one, FP8. Each is two projections in (gate and up), an activation, and one
// projection out. Launched one projection at a time that is 21 projections
// and 14 small kernels per part, and at one token the launches cost more
// than the arithmetic. Here every projection in is a row of one grid, the
// activations are one block, and every projection out is a row of another.
//
// A block computes what the same block of `dsv41_{fp4,fp8}_gemv_tc8[_f32]`
// computes, from staged weights (`dsv41_tc.cuh`), and the activation is
// `dsv41_swiglu` then `dsv41_fp8_quant_dequant` (`dsv41_common.cuh`), so the
// bytes are those of the launches these replace.
//
// A slot is an expert: the chosen in ascending expert order (`chosen`, as
// `dsv41_moe_pick` leaves them with `by_expert`), then the shared expert.
// The routed experts' weights are one part of the layer's slab: expert e at
// `slab + e * stride`, as w1's rows of the part, w3's, and w2's columns of
// the part, each `data` bytes followed by `scales` bytes of E8M0 scales.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "dsv41_common.cuh"
#include "dsv41_tc.cuh"

// Projection `which` (0 w1, 1 w3, 2 w2) of the expert in routed slot `slot`.
__device__ __forceinline__ const unsigned char* dsv41_expert_weights(
    const unsigned char* slab, const unsigned int* chosen, unsigned int slot,
    unsigned long long stride, unsigned int data, unsigned int scales, unsigned int which) {
    return slab + (unsigned long long)chosen[slot] * stride +
           (unsigned long long)which * (data + scales);
}

// gate_up[slot][0] = BF16(w1 . x), gate_up[slot][1] = BF16(w3 . x): the
// part's `rows` rows of each. x [K] BF16, FP8-rounded. shared_w / shared_s
// are the shared expert's w1 then w3 (two pointers each).
// Grid (ceil(rows / 16), 2 * (routed + 1)), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_expert_in(
    const __nv_bfloat16* __restrict__ x,
    const unsigned char* __restrict__ slab,
    const unsigned int* __restrict__ chosen,
    const unsigned char* __restrict__ shared_w1,
    const unsigned char* __restrict__ shared_s1,
    const unsigned char* __restrict__ shared_w3,
    const unsigned char* __restrict__ shared_s3,
    __nv_bfloat16* __restrict__ gate_up,
    unsigned int routed,
    unsigned int rows,
    unsigned int K,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales) {
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned int slot = blockIdx.y / 2, which = blockIdx.y & 1u;
    __nv_bfloat16* out = gate_up + (size_t)blockIdx.y * rows;
    if (slot < routed) {
        const unsigned char* w =
            dsv41_expert_weights(slab, chosen, slot, stride, data, scales, which);
        dsv41_gemv_staged_impl<1, true, false>(sh, staged, x, w, w + data, out, 1, rows, K, rows,
                                               blockIdx.x, K);
    } else {
        dsv41_gemv_staged_impl<1, false, false>(
            sh, staged, x, which ? shared_w3 : shared_w1, which ? shared_s3 : shared_s1, out, 1,
            rows, K, rows, blockIdx.x, K);
    }
}

// inner[slot] = fp8(BF16(weight * silu(min(gate, limit)) * clamp(up, limit))),
// the routed slots under their routing `weights`, the shared one under 1.
// One thread per 32 values; rows is a multiple of 32.
extern "C" __global__ void dsv41_expert_act(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ inner,
    unsigned int routed,
    unsigned int slots,
    unsigned int rows,
    float limit) {
    const unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_slot = rows / 32;
    if (b >= slots * per_slot) return;
    const unsigned int slot = b / per_slot, at = (b % per_slot) * 32;
    const float weight = slot < routed ? weights[slot] : 1.0f;
    const __nv_bfloat16* gate = gate_up + (size_t)(2 * slot) * rows + at;
    const __nv_bfloat16* up = gate + rows;
    float v[32];
    for (int i = 0; i < 32; ++i)
        v[i] = dsv41_swiglu_one(__bfloat162float(gate[i]), __bfloat162float(up[i]), weight, limit);
    dsv41_fp8_block(v, inner + (size_t)slot * rows + at);
}

// out[slot] = this part's FP32 partial of w2 . inner[slot]: [slots, H].
// Grid (ceil(H / 16), routed + 1), block 256.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_expert_out(
    const __nv_bfloat16* __restrict__ inner,
    const unsigned char* __restrict__ slab,
    const unsigned int* __restrict__ chosen,
    const unsigned char* __restrict__ shared_w2,
    const unsigned char* __restrict__ shared_s2,
    float* __restrict__ out,
    unsigned int routed,
    unsigned int H,
    unsigned int rows,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales) {
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned int slot = blockIdx.y;
    const __nv_bfloat16* a = inner + (size_t)slot * rows;
    float* c = out + (size_t)slot * H;
    if (slot < routed) {
        const unsigned char* w = dsv41_expert_weights(slab, chosen, slot, stride, data, scales, 2);
        dsv41_gemv_staged_impl<1, true, true>(sh, staged, a, w, w + data, c, 1, H, rows, H, blockIdx.x,
                                              rows);
    } else {
        dsv41_gemv_staged_impl<1, false, true>(sh, staged, a, shared_w2, shared_s2, c, 1, H, rows, H,
                                               blockIdx.x, rows);
    }
}
