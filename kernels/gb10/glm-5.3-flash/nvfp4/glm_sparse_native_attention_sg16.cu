// SPDX-License-Identifier: AGPL-3.0-only
// Atlas-owned TP4 entry of NVIDIA's BSD-3-Clause single-group prefill kernel:
// 16 local heads, the kernel NVIDIA's own prefill dispatch picks for them.
// NVIDIA source provenance and notices: third_party/nvidia_sparse_mla_sm120/.
// No external host dispatcher, shared library, or foreign runtime ABI.
//
// The pinned header stays byte-identical, and its single-group kernel is
// itself the `__global__` template, so its entry attributes are spelled away
// around the include: the template compiles as a device function, and only
// the instantiation below (an entry with Atlas's arguments) reaches the module.
#include <cuda_runtime.h>

#pragma push_macro("__global__")
#pragma push_macro("__launch_bounds__")
#pragma push_macro("__grid_constant__")
#undef __global__
#undef __launch_bounds__
#undef __grid_constant__
#define __global__ __device__ __forceinline__
#define __launch_bounds__(...)
#define __grid_constant__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177
#include "../../../../third_party/nvidia_sparse_mla_sm120/prefill_kernel.cuh"
#pragma nv_diagnostic pop
#pragma pop_macro("__grid_constant__")
#pragma pop_macro("__launch_bounds__")
#pragma pop_macro("__global__")

static_assert(BLOCK_THREADS == 384);
static_assert(SmemLayout<ModelType::GLM_NSA, ComputeMode::FP8>::TOTAL == 84368);

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
glm_sparse_native_attention_sg16(const bf16* __restrict__ q,
                                 const uint8_t* __restrict__ kv,
                                 const int32_t* __restrict__ selected,
                                 bf16* __restrict__ output,
                                 float* __restrict__ lse,
                                 const int* __restrict__ lengths,
                                 unsigned rows) {
    const PrefillColdParams cold{0.0625f, static_cast<int>(rows), 64u * 656u,
                                0, 0, nullptr, lengths, nullptr};
    sparse_mla_prefill_kernel<ModelType::GLM_NSA, ComputeMode::FP8, 16, 2048, 64>(
        q, kv, selected, nullptr, output, lse, cold);
}
