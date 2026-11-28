#pragma once
#include <hip/hip_bf16.h>
typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
// __floats2bfloat162_rn has no HIP spelling on ROCm. Packing two independent
// __float2bfloat16 (RN) conversions is the same arithmetic — CUDA's builtin
// is exactly two cvt.rn.bf16.f32 lanes — so this keeps gb10 semantics.
__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b) {
    __nv_bfloat162 r;
    r.x = __float2bfloat16(a);
    r.y = __float2bfloat16(b);
    return r;
}
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
