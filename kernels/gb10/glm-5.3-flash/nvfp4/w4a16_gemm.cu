// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the deepseek-v4-flash `w4a16_gemm.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../deepseek-v4-flash/nvfp4/w4a16_gemm.cu"

// GLM-only small-M shared projection; shares the primitives above.
#include "glm_shared_m16.cuh"
