// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from turboderp-org/exllamav3 @ 6b84a21 (MIT, Copyright (c) 2025 Turboderp); see NOTICE.md.
// Modifications: none beyond includes (added #pragma once, <cuda_fp16.h>, <cstdint>); also carries the
// adjacent half2_uint32 union (util.cuh lines 74-81, verbatim) that codebook.cuh uses.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};
