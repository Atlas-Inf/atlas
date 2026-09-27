// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from turboderp-org/exllamav3 @ 6b84a21 (MIT, Copyright (c) 2025 Turboderp); see NOTICE.md.
// Modifications: collects the few util.h / ptx.cuh helpers exl3_gemv_int8_kernel.cuh uses; each is byte-for-byte upstream.
#pragma once

// util.h

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// ptx.cuh

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}
