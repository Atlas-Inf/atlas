// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime_api.h>

#include "atlas_stale_cuda_error.h"

#include <cublasLt.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K>
int atlas_cutlass_bf16_gemm_act_weight_t_impl(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
      cutlass::gemm::GemmShape<W_M, W_N, W_K>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>>;

  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(act), k},
      {static_cast<Element const*>(weight), k},
      {static_cast<Element const*>(out), n},
      {static_cast<Element*>(out), n},
      {1.0f, 0.0f});

  size_t needed = Gemm::get_workspace_size(args);
  if (needed > workspace_size) {
    return -2;
  }
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  return atlas_launch_status(static_cast<int>(gemm(stream)));
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 128, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_128x256(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 256, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_256x128(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<256, 128, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_64x128(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<64, 128, 32, 32, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_128x64(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 64, 32, 64, 32, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_64x64(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<64, 64, 32, 32, 32, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

// Head-batched `out[t*c_stride+h*n+j] = sum_k act[t*a_stride+h*k+i]*w[h*n*k+j*k+i]`
// (the MLA absorb / V-up shape): one strided batch per head, token rows keep
// their interleaved stride, so no gather/scatter.
template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K, int STAGES>
int atlas_cutlass_bf16_grouped_impl(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int g,
    int n,
    int k,
    int a_stride,
    int c_stride,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::GemmBatched<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
      cutlass::gemm::GemmShape<W_M, W_N, W_K>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      STAGES>;
  Element const* a = static_cast<Element const*>(act);
  Element const* w = static_cast<Element const*>(weight);
  Element* d = static_cast<Element*>(out);
  typename Gemm::Arguments args(
      {m, n, k},
      {a, a_stride},
      static_cast<int64_t>(k),
      {w, k},
      static_cast<int64_t>(n) * k,
      {d, c_stride},
      static_cast<int64_t>(n),
      {d, c_stride},
      static_cast<int64_t>(n),
      {1.0f, 0.0f},
      g);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  return atlas_launch_status(static_cast<int>(gemm(stream)));
}

extern "C" int atlas_cutlass_bf16_grouped_gemm_act_weight_t(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int g,
    int n,
    int k,
    int a_stride,
    int c_stride,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  return atlas_cutlass_bf16_grouped_impl<128, 128, 32, 64, 64, 32, 3>(
      act, weight, out, m, g, n, k, a_stride, c_stride, stream);
}

// Row-major `out[m, ldc] = act[m, lda] @ weight[n, k]^T` with an L2-aware CTA
// raster: without it, K >= 4096 re-streams every weight strip from DRAM per
// M-row of tiles (the whole [N, K] panel outgrows L2).
template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K, int STAGES, int SWIZZLE>
int atlas_cutlass_bf16_tuned_impl(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    int lda,
    int ldc,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
      cutlass::gemm::GemmShape<W_M, W_N, W_K>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SWIZZLE>,
      STAGES>;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(act), lda},
      {static_cast<Element const*>(weight), k},
      {static_cast<Element const*>(out), ldc},
      {static_cast<Element*>(out), ldc},
      {1.0f, 0.0f});
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  return atlas_launch_status(static_cast<int>(gemm(stream)));
}

extern "C" int atlas_cutlass_bf16_gemm_tuned(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    int lda,
    int ldc,
    int config,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
  switch (config) {
    case 0:
      return atlas_cutlass_bf16_tuned_impl<128, 128, 32, 64, 64, 32, 4, 1>(
          act, weight, out, m, n, k, lda, ldc, stream);
    case 4:
      return atlas_cutlass_bf16_tuned_impl<128, 256, 32, 64, 64, 32, 3, 4>(
          act, weight, out, m, n, k, lda, ldc, stream);
    case 5:
      return atlas_cutlass_bf16_tuned_impl<128, 256, 32, 64, 64, 32, 3, 8>(
          act, weight, out, m, n, k, lda, ldc, stream);
    case 9:
      return atlas_cutlass_bf16_tuned_impl<128, 128, 64, 64, 64, 64, 3, 8>(
          act, weight, out, m, n, k, lda, ldc, stream);
    default:
      return -5;
  }
}

extern "C" int atlas_cublaslt_bf16_gemm_act_weight_t_algo(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream,
    int algo_index,
    int* returned_count) {
  static cublasLtHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasStatus_t st = cublasLtCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  }

  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
    if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
    if (desc) cublasLtMatmulDescDestroy(desc);
  };

  cublasStatus_t st =
      cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  st = cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }

  st = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, k, n, k);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, k, m, k);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, n, m, n);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulPreferenceCreate(&pref);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulPreferenceSetAttribute(
      pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }

  constexpr int kMaxAlgos = 16;
  cublasLtMatmulHeuristicResult_t results[kMaxAlgos];
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(
      handle,
      desc,
      layout_a,
      layout_b,
      layout_d,
      layout_d,
      pref,
      kMaxAlgos,
      results,
      &returned);
  if (returned_count) {
    *returned_count = returned;
  }
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  if (algo_index < 0 || algo_index >= returned) {
    cleanup();
    return -3;
  }
  if (results[algo_index].state != CUBLAS_STATUS_SUCCESS ||
      results[algo_index].workspaceSize > workspace_size) {
    cleanup();
    return -4;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  st = cublasLtMatmul(
      handle,
      desc,
      &alpha,
      weight,
      layout_a,
      act,
      layout_b,
      &beta,
      out,
      layout_d,
      out,
      layout_d,
      &results[algo_index].algo,
      workspace,
      workspace_size,
      stream);
  cleanup();
  return static_cast<int>(st);
}

// Stale sticky errors the wrappers drained: the count so far and the most
// recent code (see atlas_stale_cuda_error.h).
extern "C" unsigned long long atlas_cuda_stale_error_stats(int* last_code) {
  if (last_code != nullptr) {
    *last_code = atlas_stale_cuda_error_last.load(std::memory_order_relaxed);
  }
  return atlas_stale_cuda_error_count.load(std::memory_order_relaxed);
}
