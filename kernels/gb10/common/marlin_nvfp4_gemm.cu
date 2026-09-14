// SPDX-License-Identifier: AGPL-3.0-only
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

// DOWN: K=1856 % 64 == 0, N=2688 % 128 == 0. threads=128, tn=8, tk=4.
extern "C" __global__ void atlas_marlin_nvfp4_m8_k64n128(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, int num_groups, int prob_m, int prob_n,
    int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

// ── Prefill instantiations (cfg3-style): M-tile 32 (tm=2) at 128 threads. ──
// The 256-thread cfg4 B-fragment path in this vendored template is BROKEN
// (cols 0-15/32-47 of every tile wrong vs m8 on identical data, harness A/B).
// UP: tm=2, tn=4, tk=8.  DOWN: tm=2, tn=8, tk=4.
extern "C" __global__ void atlas_marlin_nvfp4_cfg4(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, int num_groups, int prob_m, int prob_n,
    int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 2, 4, 8, false, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

extern "C" __global__ void atlas_marlin_nvfp4_cfg4_k64n128(
    const int4* __restrict__ A, const int4* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ C_tmp, const int4* __restrict__ b_bias_ptr,
    const float* __restrict__ a_scales_ptr, const int4* __restrict__ scales_ptr,
    const float* __restrict__ global_scale_ptr, const int4* __restrict__ zp_ptr,
    const int* __restrict__ g_idx, int num_groups, int prob_m, int prob_n,
    int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 2, 8, 4, false, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

// Graph-safe: B/scales/gs indexed by device expert_ids[slot]. e<0 is a no-op.
extern "C" __global__ void atlas_marlin_nvfp4_m8_slot(
    const int4* __restrict__ A_base, const int4* __restrict__ B_base,
    int4* __restrict__ C_base, int4* __restrict__ C_tmp,
    const int4* __restrict__ b_bias_ptr, const float* __restrict__ a_scales_ptr,
    const int4* __restrict__ scales_base, const float* __restrict__ gs_base,
    const int4* __restrict__ zp_ptr, const int* __restrict__ g_idx,
    const int32_t* __restrict__ expert_ids, int slot, int num_groups, int prob_m,
    int prob_n, int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  int e = expert_ids[slot];
  if (e < 0) return;
  const int4* A = A_base + (size_t)slot * (size_t)prob_m * (size_t)prob_k / 8;
  int4* C = C_base + (size_t)slot * (size_t)prob_m * (size_t)prob_n / 8;
  const int4* B = B_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 32;
  const int4* scales = scales_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 256;
  const float* gs = gs_base + e;
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales, gs, zp_ptr, g_idx,
      num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

extern "C" __global__ void atlas_marlin_nvfp4_m8_k64n128_slot(
    const int4* __restrict__ A_base, const int4* __restrict__ B_base,
    int4* __restrict__ C_base, int4* __restrict__ C_tmp,
    const int4* __restrict__ b_bias_ptr, const float* __restrict__ a_scales_ptr,
    const int4* __restrict__ scales_base, const float* __restrict__ gs_base,
    const int4* __restrict__ zp_ptr, const int* __restrict__ g_idx,
    const int32_t* __restrict__ expert_ids, int slot, int num_groups, int prob_m,
    int prob_n, int prob_k, int lda, int* locks, int has_bias, int use_atomic_add,
    int use_fp32_reduce, int max_shared_mem) {
  int e = expert_ids[slot];
  if (e < 0) return;
  const int4* A = A_base + (size_t)slot * (size_t)prob_m * (size_t)prob_k / 8;
  int4* C = C_base + (size_t)slot * (size_t)prob_m * (size_t)prob_n / 8;
  const int4* B = B_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 32;
  const int4* scales = scales_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 256;
  const float* gs = gs_base + e;
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales, gs, zp_ptr, g_idx,
      num_groups, prob_m, prob_n, prob_k, lda, locks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

#ifndef MAX_SLOTS
#define MAX_SLOTS 128
#endif

// Parallel slots: grid.x = CTAs per expert (Marlin persistent), grid.y = slot.
// Empty slots return. Per-slot locks + C_tmp. No grid barrier.
extern "C" __global__ void atlas_marlin_nvfp4_m8_allslots(
    const int4* __restrict__ A_base, const int4* __restrict__ B_base,
    int4* __restrict__ C_base, int4* __restrict__ C_tmp,
    const int4* __restrict__ b_bias_ptr, const float* __restrict__ a_scales_ptr,
    const int4* __restrict__ scales_base, const float* __restrict__ gs_base,
    const int4* __restrict__ zp_ptr, const int* __restrict__ g_idx,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ n_slots,
    int* __restrict__ bars, int num_groups,
    int prob_m, int prob_n, int prob_k, int lda, int* locks, int has_bias,
    int use_atomic_add, int use_fp32_reduce, int max_shared_mem) {
  (void)bars;
  const int slot = (int)blockIdx.y;
  const int nlive = *n_slots;
  if (slot >= nlive || slot >= MAX_SLOTS) return;
  const int e = expert_ids[slot];
  if (e < 0) return;
  const int lock_stride = 256;
  const size_t ctmp_stride = (size_t)4 * (size_t)prob_n;
  const int4* A = A_base + (size_t)slot * (size_t)prob_m * (size_t)prob_k / 8;
  int4* C = C_base + (size_t)slot * (size_t)prob_m * (size_t)prob_n / 8;
  const int4* B = B_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 32;
  const int4* scales = scales_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 256;
  const float* gs = gs_base + e;
  int* slocks = locks + slot * lock_stride;
  int4* ctmp = C_tmp + slot * ctmp_stride;
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>(
      A, B, C, ctmp, b_bias_ptr, a_scales_ptr, scales, gs, zp_ptr, g_idx,
      num_groups, prob_m, prob_n, prob_k, lda, slocks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}

extern "C" __global__ void atlas_marlin_nvfp4_m8_k64n128_allslots(
    const int4* __restrict__ A_base, const int4* __restrict__ B_base,
    int4* __restrict__ C_base, int4* __restrict__ C_tmp,
    const int4* __restrict__ b_bias_ptr, const float* __restrict__ a_scales_ptr,
    const int4* __restrict__ scales_base, const float* __restrict__ gs_base,
    const int4* __restrict__ zp_ptr, const int* __restrict__ g_idx,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ n_slots,
    int* __restrict__ bars, int num_groups,
    int prob_m, int prob_n, int prob_k, int lda, int* locks, int has_bias,
    int use_atomic_add, int use_fp32_reduce, int max_shared_mem) {
  (void)bars;
  const int slot = (int)blockIdx.y;
  const int nlive = *n_slots;
  if (slot >= nlive || slot >= MAX_SLOTS) return;
  const int e = expert_ids[slot];
  if (e < 0) return;
  const int lock_stride = 256;
  const size_t ctmp_stride = (size_t)4 * (size_t)prob_n;
  const int4* A = A_base + (size_t)slot * (size_t)prob_m * (size_t)prob_k / 8;
  int4* C = C_base + (size_t)slot * (size_t)prob_m * (size_t)prob_n / 8;
  const int4* B = B_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 32;
  const int4* scales = scales_base + (size_t)e * (size_t)prob_n * (size_t)prob_k / 256;
  const float* gs = gs_base + e;
  int* slocks = locks + slot * lock_stride;
  int4* ctmp = C_tmp + slot * ctmp_stride;
  marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
                 vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>(
      A, B, C, ctmp, b_bias_ptr, a_scales_ptr, scales, gs, zp_ptr, g_idx,
      num_groups, prob_m, prob_n, prob_k, lda, slocks, has_bias != 0,
      use_atomic_add != 0, use_fp32_reduce != 0, max_shared_mem);
}
