#pragma once
#include <hip/hip_bf16.h>
typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#ifndef ATLAS_CVTA_COMPAT
#define ATLAS_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
#ifndef ATLAS_TRAP_COMPAT
#define ATLAS_TRAP_COMPAT
// CUDA's __trap() has no global HIP spelling: Linux ROCm only has
// libhipcxx::__trap (amd/amd_utils.h) and the Windows HIP SDK has none.
// gated_delta_rule_wy17.cu (#58) traps on state_is_table and is shadowed into
// strix-hip targets, so map it onto the same builtin libhipcxx uses.
__device__ __forceinline__ void __trap() { __builtin_trap(); }
#endif
