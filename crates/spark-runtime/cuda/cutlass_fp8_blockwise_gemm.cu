// SPDX-License-Identifier: AGPL-3.0-only

// W8A8 block-FP8 GEMM (NVIDIA's vLLM / DeepSeek recipe) on GB10's SM120
// tensor cores through CUTLASS's blockwise-scaled mainloop:
//
//   out[m, n] = bf16( sum_g ( sum_(k in g) act[m, k] * w[n, k] ) * act_sf[m, g] * w_sf[n / 128, g] )
//
// over K-groups g of 128 (the in-tree fp8_gemm_t_blockscaled's arithmetic
// contract). Operands, all row-major: `act` [M, K] E4M3 and `act_sf`
// [M, K/128] FP32 (per_token_group_quant_fp8), `weight` [N, K] E4M3 and
// `weight_sf` [N/128, K/128] FP32 (a ModelOpt FP8_PB_WO grid), `out` [M, N]
// BF16. N and K multiples of 128. cuBLASLt 13.0 has no 128-block FP8 kernel
// for SM121 (AlgoGetHeuristic returns NOT_SUPPORTED).
//
// After CUTLASS examples/87_blackwell_geforce_gemm_blockwise (87b, BSD-3).

#include <cuda_runtime_api.h>

#include "atlas_stale_cuda_error.h"

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace atlas_fp8_blockwise {

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;
using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 8;
using ElementAccumulator = float;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
// 1 x 128 activation groups, 128 x 128 weight blocks, both K-contiguous.
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128, UMMA::Major::K,
                                                               UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
    cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
    AlignmentB, ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue,
                                                        void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace atlas_fp8_blockwise

#endif

extern "C" int atlas_cutlass_fp8_blockwise_gemm_act_weight_t(
    const void* act,
    const void* act_sf,
    const void* weight,
    const void* weight_sf,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  atlas_drain_stale_cuda_error();
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace atlas_fp8_blockwise;
  if (m <= 0 || n <= 0 || k <= 0 || n % 128 != 0 || k % 128 != 0) {
    return -1;
  }
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<ElementA const*>(act), stride_a, static_cast<ElementB const*>(weight),
       stride_b, static_cast<ElementAccumulator const*>(act_sf), layout_sfa,
       static_cast<ElementAccumulator const*>(weight_sf), layout_sfb},
      {{}, static_cast<ElementC const*>(out), stride_c, static_cast<ElementC*>(out), stride_c}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_size) {
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
  return atlas_launch_status(static_cast<int>(gemm.run(stream)));
#else
  (void)act, (void)act_sf, (void)weight, (void)weight_sf, (void)out, (void)m, (void)n, (void)k,
      (void)workspace, (void)workspace_size, (void)stream;
  return -3;
#endif
}
