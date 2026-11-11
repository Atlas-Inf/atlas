// SPDX-License-Identifier: AGPL-3.0-only
//
// The rank-split drafter's join (`rank_split_exec.rs` `RankSplit::join`):
// `[m, n]` BF16 rows at `dst` from every rank's share of the n output rows,
// rank r's `[m, rows_r]` at `src + r * stride` bytes, its columns starting at
// cut(r) = n * r / ranks / 16 * 16 (`rank_split::share`; the last rank takes
// the rest). One launch in place of one pitched copy-engine copy a rank,
// which cost 4-12 us each between the drafter's collectives (dec32 nsys,
// 2026-10-18). A pure copy: every value is the one the copies wrote.
// PDL-entered (GLM-5.3-Flash's PDL list): it reads the gather it follows
// only after the wait.

#include "atlas_pdl.cuh"
#include <stdint.h>

extern "C" __global__ void rank_split_join_bf16(
    unsigned short* dst, const unsigned char* src, uint64_t stride, uint32_t m, uint32_t n,
    uint32_t ranks) {
    atlas_pdl_enter();
    const uint64_t total = (uint64_t)m * n;
    for (uint64_t e = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; e < total;
         e += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t i = (uint32_t)(e / n);
        const uint32_t j = (uint32_t)(e % n);
        uint32_t r = 0, lo = 0, hi = n;
        for (; r < ranks; ++r) {
            lo = (uint32_t)((uint64_t)n * r / ranks / 16 * 16);
            hi = r + 1 == ranks ? n : (uint32_t)((uint64_t)n * (r + 1) / ranks / 16 * 16);
            if (j < hi) {
                break;
            }
        }
        const unsigned short* share = reinterpret_cast<const unsigned short*>(src + r * stride);
        dst[e] = share[(uint64_t)i * (hi - lo) + (j - lo)];
    }
}
