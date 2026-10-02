// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from turboderp-org/exllamav3 @ 6b84a21 (MIT, Copyright (c) 2025 Turboderp); see NOTICE.md.
// Modifications: none beyond includes (dropped unused <cuda/atomic>, added <cuda_fp16.h> and <cstdint>).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;
