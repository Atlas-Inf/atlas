// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from turboderp-org/exllamav3 @ 6b84a21 (MIT, Copyright (c) 2025 Turboderp); see NOTICE.md.
// Modifications: none beyond includes (added #pragma once, <cuda_fp16.h>) and one warning fix — dropped the
// ignored __device__ annotation from the explicitly defaulted ctor (nvcc #20012-D, fatal under
// --Werror all-warnings); took only the half4 struct
// (util.cuh lines 8-18, verbatim) that had_hf_r_128_inner needs — the adjacent bfloat164/half8 types
// and helpers are not vendored.
#pragma once

#include <cuda_fp16.h>

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;
