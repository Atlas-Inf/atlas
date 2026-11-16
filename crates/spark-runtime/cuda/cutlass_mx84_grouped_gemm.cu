// SPDX-License-Identifier: AGPL-3.0-only
// One Sm120 block-scaled grouped GEMM of MXFP8 activations against MXFP4
// weights: D_g [m_g, n] BF16 = A_g [m_g, k] . B_g [n, k]^T for every group g
// of one launch (`GemmUniversalMode::kGrouped`), FP32 accumulation.
//
// This is the shape of a routed MoE whose checkpoint ships its experts as
// MXFP4 (E2M1, two values a byte, low nibble first, one UE8M0 scale per 32 K
// values of a row) and whose activations are rounded to MXFP8 (E4M3, the same
// kind of scale): DeepSeek-V4.1-Flash. A group is one expert's projection and
// the rows routed to it. Nothing is requantized: the tensor cores multiply
// E4M3 by E2M1 under both operands' block scales (`kind::mxf8f6f4`), which
// cuBLASLt does not offer for a mixed pair.
//
// Layouts, per group:
//   A    E4M3 [m_g, k] row-major (16-byte aligned).
//   SFA  UE8M0, CUTLASS's block-scaled atom for [m_g, k / 32]: 128 x 4 tiles
//        of 512 bytes, the layout cuBLASLt calls tiled
//        (`kernels/gb10/common/mxfp8_act_quant.cuh`).
//   B    E2M1 packed [n, k / 2] bytes row-major, as the checkpoint has it:
//        CUTLASS's column-major K x N. 64-byte aligned, k % 128 == 0.
//   SFB  the same atom for [n, k / 32].
//   D    BF16 [m_g, n] row-major, rows contiguous.
//
// Types and host assembly follow `cutlass_nvfp4_grouped_gemm.cu`.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "atlas_stale_cuda_error.h"
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// The grouped schedule; a build may name another to compare.
#ifndef ATLAS_MX84_SCHEDULE
#define ATLAS_MX84_SCHEDULE cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
#endif

namespace mx84 {

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue8m0_t;
using ElementAccumulator = float;
using ElementCompute = float;

// A pointer to a layout selects the grouped kernel.
using GmemLayoutA = cutlass::layout::RowMajor;
using GmemLayoutB = cutlass::layout::ColumnMajor;
using GmemLayoutC = cutlass::layout::RowMajor;
using GmemLayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 128;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutC*,
    AlignmentC,
    ElementD,
    GmemLayoutD*,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    GmemLayoutA*,
    AlignmentA,
    ElementB,
    GmemLayoutB*,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    ATLAS_MX84_SCHEDULE>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using BlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ProblemShape = cute::Shape<int, int, int>;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

}  // namespace mx84

#endif  // arch guard

// Every group of one projection in one launch. The six arrays are the host's,
// `groups` long: device addresses of each group's A, SFA, B, SFB and D, and
// its row count (at least 1). `workspace` is device scratch for the per-group
// argument arrays and the scheduler, read by the kernel in stream order, so a
// later call on the same stream may reuse it.
//
// Returns 0, or: -1 unsupported arch, -2 workspace too small, -3 bad
// argument, -50 can_implement rejected, a CUTLASS status from initialize, or
// ATLAS_CUTLASS_LAUNCH_FAILED plus one from the launch.
extern "C" int atlas_cutlass_mx84_grouped(
    const unsigned long long* a_ptrs,
    const unsigned long long* sfa_ptrs,
    const unsigned long long* b_ptrs,
    const unsigned long long* sfb_ptrs,
    const unsigned long long* d_ptrs,
    const int* rows,
    int groups,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    void* stream_handle) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace mx84;
  if (groups == 0) {
    return 0;
  }
  if (groups < 0 || n <= 0 || k <= 0 || k % 128 != 0 || n % 8 != 0 || workspace == nullptr) {
    return -3;
  }
  atlas_drain_stale_cuda_error();
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  const int G = groups;
  std::vector<ProblemShape> shapes(G);
  std::vector<const ElementA::DataType*> hA(G);
  std::vector<const ElementB::DataType*> hB(G);
  std::vector<const ElementSF*> hSFA(G), hSFB(G);
  std::vector<const ElementC*> hC(G);
  std::vector<ElementD*> hD(G);
  std::vector<StrideA> sA(G);
  std::vector<StrideB> sB(G);
  std::vector<StrideC> sC(G);
  std::vector<StrideD> sD(G);
  std::vector<LayoutSFA> lSFA(G);
  std::vector<LayoutSFB> lSFB(G);
  for (int g = 0; g < G; ++g) {
    const int m = rows[g];
    if (m <= 0) {
      return -3;
    }
    shapes[g] = cute::make_shape(m, n, k);
    hA[g] = reinterpret_cast<const ElementA::DataType*>(a_ptrs[g]);
    hB[g] = reinterpret_cast<const ElementB::DataType*>(b_ptrs[g]);
    hSFA[g] = reinterpret_cast<const ElementSF*>(sfa_ptrs[g]);
    hSFB[g] = reinterpret_cast<const ElementSF*>(sfb_ptrs[g]);
    hC[g] = reinterpret_cast<const ElementC*>(d_ptrs[g]);
    hD[g] = reinterpret_cast<ElementD*>(d_ptrs[g]);
    sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    sC[g] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    lSFA[g] = BlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    lSFB[g] = BlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }

  unsigned char* ws = static_cast<unsigned char*>(workspace);
  size_t cursor = 0;
  bool fits = true;
  auto put = [&](const void* src, size_t bytes) -> void* {
    void* dst = ws + cursor;
    if (cursor + bytes > workspace_size) {
      fits = false;
      return dst;
    }
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    cursor = align_up(cursor + bytes, 256);
    return dst;
  };
  auto* dShapes = (ProblemShape*)put(shapes.data(), G * sizeof(ProblemShape));
  auto* dA = (const ElementA::DataType**)put(hA.data(), G * sizeof(void*));
  auto* dB = (const ElementB::DataType**)put(hB.data(), G * sizeof(void*));
  auto* dSFA = (const ElementSF**)put(hSFA.data(), G * sizeof(void*));
  auto* dSFB = (const ElementSF**)put(hSFB.data(), G * sizeof(void*));
  auto* dC = (const ElementC**)put(hC.data(), G * sizeof(void*));
  auto* dD = (ElementD**)put(hD.data(), G * sizeof(void*));
  auto* dsA = (StrideA*)put(sA.data(), G * sizeof(StrideA));
  auto* dsB = (StrideB*)put(sB.data(), G * sizeof(StrideB));
  auto* dsC = (StrideC*)put(sC.data(), G * sizeof(StrideC));
  auto* dsD = (StrideD*)put(sD.data(), G * sizeof(StrideD));
  auto* dlSFA = (LayoutSFA*)put(lSFA.data(), G * sizeof(LayoutSFA));
  auto* dlSFB = (LayoutSFB*)put(lSFB.data(), G * sizeof(LayoutSFB));
  if (!fits) {
    return -2;
  }

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename Gemm::GemmKernel::CollectiveMainloop::Arguments mainloop_args{
      dA, dsA, dB, dsB, dSFA, dlSFA, dSFB, dlSFB};
  typename Gemm::GemmKernel::CollectiveEpilogue::Arguments epi_args{};
  epi_args.thread.alpha = 1.0f;
  epi_args.thread.beta = 0.0f;
  epi_args.ptr_C = dC;
  epi_args.dC = dsC;
  epi_args.ptr_D = dD;
  epi_args.dD = dsD;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, dShapes, shapes.data()},
      mainloop_args,
      epi_args,
      hw};

  Gemm gemm;
  const size_t need = Gemm::get_workspace_size(args);
  if (cursor + need > workspace_size) {
    return -2;
  }
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return -50;
  }
  cutlass::Status st = gemm.initialize(args, ws + cursor, stream);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st);
  }
  return atlas_launch_status(static_cast<int>(gemm.run(stream)));
#else
  (void)a_ptrs; (void)sfa_ptrs; (void)b_ptrs; (void)sfb_ptrs; (void)d_ptrs; (void)rows;
  (void)groups; (void)n; (void)k; (void)workspace; (void)workspace_size; (void)stream_handle;
  return -1;
#endif
}
