// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Barriers for bodies that run on a whole CTA in their own launch and on a
// group of a wider CTA in a fused one (glm_mla_fuse.cu).

// The whole CTA's barrier.
struct GlmCtaBarrier {
    __device__ __forceinline__ void operator()() const { __syncthreads(); }
};

// Named barrier `id` (1..15) over `count` threads (a multiple of 32).
struct GlmNamedBarrier {
    unsigned int id, count;
    __device__ __forceinline__ void operator()() const {
        asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
    }
};
