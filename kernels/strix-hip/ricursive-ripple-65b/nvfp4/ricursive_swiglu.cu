// SPDX-License-Identifier: AGPL-3.0-only

// SiLU activation + multiply with Ricursive Ripple's SwiGLU clamp, installed
// on the TARGET's dense FFN layers only (`DenseFfnLayer::set_act_mul`). It is
// not a `moe_silu_mul` shadow: the DFlash2 drafter's SwiGLU is unclamped in
// the reference and keeps `common/moe_silu_mul`.
//
//   output[i] = silu(min(gate[i], L)) * clamp(up[i], -L, L),  L = 20
//
// Reference `RicursiveMLP.forward`:
//     gate = gate.clamp(max=ffn_clamp_threshold)
//     up   = up.clamp(min=-ffn_clamp_threshold, max=ffn_clamp_threshold)
//     x    = SiluAndMul(cat(gate, up))
// `ffn_clamp_threshold` = 20.0 in the checkpoint; the config parser refuses
// any other value (`RICURSIVE_FFN_CLAMP`), since this kernel has no argument.
// Same signature as `moe_silu_mul`.
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_clamp20(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    atlas_pdl_enter();
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float LIMIT = 20.0f;
    float g = fminf(__bfloat162float(gate[idx]), LIMIT);
    float u = fminf(fmaxf(__bfloat162float(up[idx]), -LIMIT), LIMIT);
    float silu = g / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(silu * u);
}
