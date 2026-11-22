// SPDX-License-Identifier: AGPL-3.0-only
// A visit's input projections (q, k, v, relative bias, output gate, sinks and the
// dynamic convs' taps) for up to MAX_M decode or verify rows in one launch. Each
// segment is one projection: its own input rows, BF16 weight and output; the
// blocks over all segments' outputs run together, and every output row is the
// chain dense_gemv_bf16_batchm computes for it (for one row, dense_gemv_bf16's).
// One launch instead of six to eight keeps more of the weight reads in flight:
// the projections are small (24 to 3,072 outputs over K = 2,304).

#include "../../common/atlas_pdl.cuh"
#include "../../common/dense_gemv_bf16_batchm.cu"

// The host's table entry (spark-model inkling2 `ProjSeg`): 32 bytes.
struct ProjSeg {
    const __nv_bfloat16* a;  // [M, K] input rows
    const __nv_bfloat16* w;  // [n, K]
    __nv_bfloat16* c;        // [M, n] output rows
    unsigned int n;          // outputs
    unsigned int block0;     // first block of this segment
};

// Grid: (sum over segments of ceil(n / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void inkling2_proj_gemv(
    const ProjSeg* __restrict__ segs, unsigned int nseg, unsigned int M, unsigned int K
) {
    atlas_pdl_enter();
    unsigned int s = 0;
    while (s + 1 < nseg && blockIdx.x >= segs[s + 1].block0) s++;
    const ProjSeg g = segs[s];
    dense_gemv_bf16_batchm_body(g.a, g.w, g.c, M, g.n, K, g.n, blockIdx.x - g.block0);
}
