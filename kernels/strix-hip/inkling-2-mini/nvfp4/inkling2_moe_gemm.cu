// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 grouped expert GEMMs, strix-hip: the gb10 file's W4A4 / W8A8 entries run on the
// block-scaled tensor cores (mma.sync kind::mxf4nvf4 / kind::mxf8f6f4), which gfx1151 does not
// have. Stubs keep the kernel table resolvable; under cfg!(atlas_scale) prefill rows take the
// grouped expert GEMVs (inkling2/calls.rs) over BF16 rows.

extern "C" __global__ void inkling2_moe_tiles() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w4a4() { __builtin_trap(); }
extern "C" __global__ void inkling2_moe_gemm_w8a8() { __builtin_trap(); }
