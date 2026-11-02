// SPDX-License-Identifier: AGPL-3.0-only
// Atlas-owned GLM specialization of NVIDIA's BSD-3-Clause device kernel.
// NVIDIA source provenance and notices: third_party/nvidia_sparse_mla_sm120/.
// No external host dispatcher, shared library, or foreign runtime ABI.
#include "../../../../third_party/nvidia_sparse_mla_sm120/prefill_kernel.cuh"

static_assert(BLOCK_THREADS == 384);
static_assert(SmemLayoutMG<ModelType::GLM_NSA, ComputeMode::FP8>::TOTAL == 91920);

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
glm_sparse_native_attention(const bf16* __restrict__ q,
                            const uint8_t* __restrict__ kv,
                            const int32_t* __restrict__ selected,
                            bf16* __restrict__ output,
                            float* __restrict__ lse,
                            const int* __restrict__ lengths,
                            unsigned rows) {
    const PrefillColdParams cold{0.0625f, static_cast<int>(rows), 64u * 656u,
                                0, 0, nullptr, lengths, nullptr};
    prefill_mg_impl<ModelType::GLM_NSA, ComputeMode::FP8, 32, 2048, 64,
                    false, 64, 2>(q, kv, selected, nullptr, nullptr,
                                  output, lse, nullptr, cold);
}
