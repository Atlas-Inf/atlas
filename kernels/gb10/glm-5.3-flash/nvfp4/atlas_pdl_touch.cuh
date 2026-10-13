// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Weight touch before the PDL wait (ATLAS_GLM_DECODE_GEMV_BATCH=1). A kernel
// launched with PDL usually starts while its predecessors (small per-layer
// kernels) still run, and sits in `griddepcontrol.wait` with the memory bus
// idle. Its weights do not depend on the predecessor, so the first `ctas` CTAs
// (the ones certain to be resident) issue one discarded byte load per 32-byte
// sector of the first `rows` rows of each region, pulling them into L2 during
// the wait; the kernel body then reads the same bytes from L2. Loads only: no
// value reaches the kernel's arithmetic, so every output bit is unchanged.
// `ld` is the byte stride between rows.
//
// Invariant: every region is an immutable weight (a `const unsigned char*
// __restrict__` weight or scale parameter of the kernel), never an activation,
// KV or state buffer a predecessor may still be writing: these loads run
// before the wait. The runtime's PDL source-contract test checks it.
//
// GLM-local (not in common/atlas_pdl.cuh) so the include closure of every
// other gb10 target stays unchanged.
//
// Prior art (docs/glm-prior-art.md): TensorFold's L2 weight touch
// (github.com/jayleaton/glm53-tensorfold-spark patches 0040, and 0440 where
// each CTA prefetches its weights before griddepcontrol.wait; Apache-2.0) and
// knapcio's GLM_L2_PREFETCH; compare mmastrac's arx L2 prefetch during
// all-reduce waits. Ours are discarded byte loads from the kernel's own CTAs,
// not a side kernel or cp.async.bulk.prefetch. No code copied.
#include "../../common/atlas_pdl.cuh"

struct AtlasTouch {
    const unsigned char* p;
    unsigned int row_bytes;
    unsigned long long ld;
};

__device__ __forceinline__ void atlas_l2_touch(const unsigned char* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("{\n\t.reg .u32 t;\n\tld.volatile.global.u8 t, [%0];\n\t}" :: "l"(p) : "memory");
#endif
}

// `atlas_pdl_enter()` with the touch of two regions (weights, scales) between
// its halves. `cta` is this CTA's index in the grid. `rows == 0` touches
// nothing: the kernel is then its plain PDL-entered twin.
__device__ __forceinline__ void atlas_pdl_enter_touch(
    const AtlasTouch a, const AtlasTouch b, unsigned int rows, unsigned int cta, unsigned int ctas
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (cta < ctas) {
        const AtlasTouch r[2] = {a, b};
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            const unsigned int per_row = (r[q].row_bytes + 31u) / 32u;
            const unsigned int total = rows * per_row;
            for (unsigned int i = cta * blockDim.x + threadIdx.x; i < total; i += ctas * blockDim.x)
                atlas_l2_touch(r[q].p + (unsigned long long)(i / per_row) * r[q].ld + (i % per_row) * 32u);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// `atlas_pdl_enter_touch` for a launch of two projections (grid z = plane):
// plane z touches its own pair (`a0`, `b0` or `a1`, `b1`).
__device__ __forceinline__ void atlas_pdl_enter_touch_pair(
    const AtlasTouch a0, const AtlasTouch b0, const AtlasTouch a1, const AtlasTouch b1,
    unsigned int rows, unsigned int cta, unsigned int ctas
) {
    const bool one = blockIdx.z != 0u;
    atlas_pdl_enter_touch(one ? a1 : a0, one ? b1 : b0, rows, cta, ctas);
}
