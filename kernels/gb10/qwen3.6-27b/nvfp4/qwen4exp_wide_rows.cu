// SPDX-License-Identifier: AGPL-3.0-only
//
// The qwen4_exp exact wide-row GEMVs (`qwen4exp_bf16_rows16/32`, row t
// byte-identical to `dense_gemv_bf16` on row t) for the Qwen3.6/3.8-27B
// targets: the batched multi-sequence head of `model/exact_verify.rs`'s batch
// lever (ATLAS_EXACT_VERIFY=1 + ATLAS_SKINNY_MMA=1) takes 16 or 32 rows a
// pass over the BF16 LM head instead of 8. Same source as the Flash-Next
// target, built here with this directory's flags (--fmad=false, as there,
// and as `common/dense_gemv_bf16.cu` is built here, which is what the
// bit-identity rests on). Gate: `examples/bf16_wide_rows_microtest.rs`.

#include "../../qwen3.8-flash-next/nvfp4/qwen4exp_wide_rows.cu"
