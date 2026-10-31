// SPDX-License-Identifier: AGPL-3.0-only
// GLM-5.3-Flash shadow of deepseek-v4-flash/nvfp4/moe_silu_mul.cu: the same
// DeepSeek-clamped SiLU*mul, entered through atlas_pdl_enter() so it may be
// launched with programmatic dependent launch, plus the GLM-only fused
// SiLU*mul + NVFP4 quantization used by the prequantized MoE prefill.

// Atlas MoE element-wise SiLU activation + multiply, WITH DeepSeek-V4's
// configured SwiGLU clamp. Shadows `common/moe_silu_mul.cu` for this model
// only.
//
//   output[i] = silu(clamp(gate[i])) * clamp(up[i])
//
// WHY THIS FILE EXISTS. The clamp used to live in `common/`, added there by the
// DeepSeek-V4 port (#186) as six lines on a kernel that was already shared. But
// `moe_silu_mul` is not a DeepSeek kernel: it is the SiLU activation for every
// dense model's decode and K-verify FFN (`DenseFfnLayer::act_mul`), for every
// MoE model's grouped prefill (`MoeLayer::moe_act_mul`), and for the MTP and
// DFlash draft heads. So one checkpoint's config value was applied to about
// twenty checkpoints, none of which declare a SwiGLU limit and none of whose
// reference implementations clamp at all (`Qwen3_5MLP.forward` and
// `Qwen3_5MoeExperts.forward` are a bare `act_fn(gate) * up`).
//
// `swiglu_limit` is genuinely model-specific — DeepSeek-V4 sets 10.0, GPT-OSS
// sets 7.0, Qwen sets nothing — so the value belongs with the model. Keeping it
// in a shadow is the smallest correct home for it until the limit is threaded
// from config into a kernel argument, which is what the checkpoint actually
// asks for and what Step-3.7's per-LAYER `swiglu_limits` array will require.
//
// The reference is `inference/model.py` in `deepseek-ai/DeepSeek-V4-Flash`:
//
//     if self.swiglu_limit > 0:
//         up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
//         gate = torch.clamp(gate, max=self.swiglu_limit)
//     x = F.silu(gate) * up
//
// Note the asymmetry — gate is bounded ABOVE only, up is bounded on both sides.
// The math below is byte-for-byte what `common/` computed before the move, so
// DeepSeek-V4's numerics are unchanged by relocating it.
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>

__device__ __forceinline__ void moe_silu_mul_body(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output, unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // swiglu_limit = 10.0, from the checkpoint's config.json. Hardcoded because
    // no Rust parser reads the field yet; `ModelConfig` has no home for it.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    atlas_pdl_enter();
    moe_silu_mul_body(gate, up, output, total_elements);
}

// moe_silu_mul that also asks the routed FFN's router weight (`rows` rows of
// `row_bytes`) and its correction bias (f32 [rows]) into L2 during its PDL
// wait (atlas_pdl_touch.cuh; every CTA takes a share). The shared expert's
// SiLU of a decode step is resident through the attention all-reduce, while
// the memory bus idles, and the router (glm_moe_route_one) runs two launches
// later, behind the shared down whose CTAs leave it no room to touch its own
// weight. Discarded loads of immutable weights: the same bytes.
extern "C" __global__ void moe_silu_mul_touch(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output, unsigned int total_elements,
    const unsigned char* __restrict__ router_w, const unsigned char* __restrict__ router_bias,
    unsigned int rows, unsigned int row_bytes
) {
    atlas_pdl_enter_touch({router_w, row_bytes, row_bytes}, {router_bias, 4u, 4ull}, rows,
                          blockIdx.x, gridDim.x);
    moe_silu_mul_body(gate, up, output, total_elements);
}


#include "silu_nvfp4_quant.cuh"

// Fused DeepSeek-clamped SiLU·mul + standard token-major NVFP4 quantization.
//
// This is the native-FP4 analogue of common/moe_silu_mul.cu's fused FP8
// helper.  It emits exactly the packed `[M,K/2]` E2M1 + `[M,K/16]` E4M3
// layout consumed by moe_w4a4_grouped_gemm_prequant_t_k64.  The product is
// explicitly rounded through BF16 before the group maximum and FP4 encode,
// matching the old moe_silu_mul -> quantize_bf16_to_nvfp4 pair.
//
// `out_bf16` is nullable.  Base-model serving passes null and avoids the full
// BF16 intermediate; MoE down-projection LoRA may request it because its fold
// consumes the post-SiLU activations.

__device__ __forceinline__ void silu_mul_quant_nvfp4_row(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int row,
    unsigned int K,
    float g
) {

    const __nv_bfloat16* grow = gate + (unsigned long long)row * K;
    const __nv_bfloat16* urow = up + (unsigned long long)row * K;
    unsigned char* prow = packed_out + (unsigned long long)row * (K / 2);
    unsigned char* srow = scale_out + (unsigned long long)row * (K / 16);
    __nv_bfloat16* brow = out_bf16 ? out_bf16 + (unsigned long long)row * K : nullptr;
    const unsigned int groups = K / 16;

    for (unsigned int group = threadIdx.x; group < groups; group += blockDim.x) {
        float vals[16];
        float group_max = 0.0f;
        const unsigned int base = group * 16;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float r = silu_nvfp4_act(__bfloat162float(grow[base + i]),
                                           __bfloat162float(urow[base + i]));
            if (brow) brow[base + i] = __float2bfloat16(r);
            vals[i] = r;
            group_max = fmaxf(group_max, fabsf(r));
        }

        float inv;
        srow[group] = silu_nvfp4_group_scale(group_max, &inv, g);
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const unsigned int q0 = silu_nvfp4_quantize_e2m1(vals[i] * inv);
            const unsigned int q1 = silu_nvfp4_quantize_e2m1(vals[i + 1] * inv);
            prow[group * 8 + i / 2] = (unsigned char)((q1 << 4) | q0);
        }
    }
}

extern "C" __global__ void silu_mul_quant_nvfp4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int M,
    unsigned int K
) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x;
    if (row >= M) return;
    silu_mul_quant_nvfp4_row(gate, up, packed_out, scale_out, out_bf16, row, K, 1.0f);
}

// silu_mul_quant_nvfp4 over rows sorted by expert (`expert_offsets`, E + 1
// starts), each quantized with its expert's global scale `a2_gscale[e]`
// (1 / w2 input_scale, ATLAS_GLM_FP4L_INPUT_SCALES).
extern "C" __global__ void silu_mul_quant_nvfp4_by_expert(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ packed_out,
    unsigned char* __restrict__ scale_out,
    __nv_bfloat16* __restrict__ out_bf16,
    unsigned int M,
    unsigned int K,
    const int* __restrict__ expert_offsets,
    const float* __restrict__ a2_gscale,
    unsigned int num_experts
) {
    atlas_pdl_enter();
    const unsigned int row = blockIdx.x;
    if (row >= M) return;
    // The expert e with expert_offsets[e] <= row < expert_offsets[e + 1].
    unsigned int lo = 0, hi = num_experts;
    while (hi - lo > 1) {
        const unsigned int mid = (lo + hi) / 2;
        if ((unsigned int)__ldcg(expert_offsets + mid) <= row) lo = mid; else hi = mid;
    }
    silu_mul_quant_nvfp4_row(gate, up, packed_out, scale_out, out_bf16, row, K, __ldcg(a2_gscale + lo));
}
