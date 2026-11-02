#pragma once
#include <hip/hip_bf16.h>
typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#ifndef ATLAS_CVTA_COMPAT
#define ATLAS_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
#ifndef __trap
// CUDA's device __trap() (gated_delta_rule_wy17's state_is_table guard) has no
// global HIP spelling on Linux ROCm either (only libhipcxx::__trap). Windows
// already gets this macro from the force-included atlas_hip_win_shims.h, so the
// guard makes this a no-op there; Linux HIP builds pick it up here.
#define __trap() __builtin_trap()
#endif
