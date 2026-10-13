// SPDX-License-Identifier: AGPL-3.0-only

// Plain SiLU·mul for heads whose own checkpoint declares no SwiGLU limit.
//
// output[i] = silu(gate[i]) * up[i]
//
// `moe_silu_mul` is shadowed per target: GLM-5.3 inherits the DeepSeek-V4
// kernel set, whose shadow clamps gate <= 10 and |up| <= 10 (both
// checkpoints declare swiglu_limit = 10). A draft head is a separate
// checkpoint (the DFlash2 drafter is Qwen3-shaped, no limit); resolving the
// target's shadow for it truncated its MLP (gate reached 27, up 38). This
// module is never shadowed, so a head without a limit gets the bare product.
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void silu_mul_plain(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    const float g = __bfloat162float(gate[idx]);
    const float u = __bfloat162float(up[idx]);
    output[idx] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
}
