// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// SparkGLM 0.2.0's FlashKDA library (`libatlas_glm53_flash_kda.so`), built
// in-tree: MoonshotAI/FlashKDA @1ce47ea3 with SparkGLM's state-slot, TMA
// swizzle and FP32 recurrence patches (flashkda/, verbatim; provenance.json)
// plus SparkGLM's C bridge (bridge.cu, verbatim). Compiled with the flags of
// SparkGLM's flash_kda/rebuild.sh (spark-runtime build_glm_kda.rs).
//
// The upstream kernels share names with Atlas's own KDA operator
// (cuda/glm_kda/vendor/), so this translation unit keeps them in a namespace.
// Every header they include is included first, outside it: their include
// guards then make the nested includes empty. The bridge's extern "C" entry
// points keep their unmangled names.
// utils.cuh's includes in its order (CuTe's headers are order-sensitive),
// then the remaining ones of fwd_kernel2.cuh and bridge.cu.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

#include <cstdint>
#include <type_traits>

namespace atlas_sparkglm_kda {
#include "flashkda/fwd_launch.cu"
#include "bridge.cu"
}  // namespace atlas_sparkglm_kda
