// SPDX-License-Identifier: Apache-2.0
// Vendored Marlin NVFP4 GEMM (vLLM 0.26 libtorch_stable/quantization/marlin).
// Torch-free. One instantiation: BF16 x NVFP4 -> BF16, group=16, M<=8, N%64==0.
// Launch from host: grid=sms, block=128, dynamic smem >= 64KiB.

#define MARLIN_NAMESPACE_NAME marlin
#include "marlin_nvfp4/kernel.h"
#include "marlin_nvfp4/marlin_template.h"

// threads=128, tm=1, tn=4 (64), tk=8 (128), m8=true, stages=4, group_blocks=1
extern "C" __global__ void atlas_marlin_nvfp4_m8(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, int num_groups, int prob_m, int prob_n,
    int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}
