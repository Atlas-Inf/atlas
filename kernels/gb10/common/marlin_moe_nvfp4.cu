// SPDX-License-Identifier: Apache-2.0
// Vendored Marlin MoE NVFP4 (vLLM 0.26 marlin_moe_wna16). Torch-free.
// Instantiation: BF16 x NVFP4, group=16, M-tile 8, N-tile 64, K-tile 128.

#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#include "marlin_nvfp4/moe_marlin_template.h"

extern "C" __global__ void atlas_marlin_moe_nvfp4_m8(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    const float* __restrict__ topk_weights_ptr, int top_k, int mul_topk_weights,
    int num_groups, int prob_m, int prob_n, int prob_k, int* locks, int has_bias,
    int use_atomic_add, int use_fp32_reduce) {
  marlin_moe_wna16::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                           vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4,
                           8, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, sorted_token_ids_ptr, expert_ids_ptr,
      num_tokens_past_padded_ptr, topk_weights_ptr, top_k, mul_topk_weights != 0,
      num_groups, prob_m, prob_n, prob_k, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0);
}

extern "C" __global__ void atlas_marlin_moe_nvfp4_m8_k64n128(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    const float* __restrict__ topk_weights_ptr, int top_k, int mul_topk_weights,
    int num_groups, int prob_m, int prob_n, int prob_k, int* locks, int has_bias,
    int use_atomic_add, int use_fp32_reduce) {
  marlin_moe_wna16::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                           vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 8,
                           4, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, sorted_token_ids_ptr, expert_ids_ptr,
      num_tokens_past_padded_ptr, topk_weights_ptr, top_k, mul_topk_weights != 0,
      num_groups, prob_m, prob_n, prob_k, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0);
}
