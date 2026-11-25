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
// w4a16_gemv.cu names template parameters MAX_M; the BF16 body above is already expanded.
#undef MAX_M
#undef VEC_SIZE
#include "../../common/w4a16_gemv.cu"

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

// The same for one decode row when some projections are NVFP4 (the Spark overlay's q, k, v,
// relative bias and gate; sinks and the dynamic convs' taps stay BF16). The host's table
// entry (spark-model inkling2 `MixSeg`): 48 bytes, BF16 when `scale` is null.
struct MixSeg {
    const __nv_bfloat16* a;      // [1, K] input row
    const unsigned char* w;      // [n, K] BF16, or [n, K/2] packed E2M1 codes
    const unsigned char* scale;  // [n, K/16] E4M3 block scales; null: a BF16 weight
    __nv_bfloat16* c;            // [1, n] output row
    unsigned int n;              // outputs
    unsigned int block0;         // first block of this segment
    float scale2;                // the NVFP4 weight's global scale
    unsigned int pad;
};

// Every output is the chain its own kernel computes: dense_gemv_bf16's for a BF16 segment,
// w4a16_gemv's for an NVFP4 one (its body below, the block index taken within the segment).
// Grid: (sum over segments of ceil(n / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void inkling2_proj_gemv_mixed(
    const MixSeg* __restrict__ segs, unsigned int nseg, unsigned int K
) {
    atlas_pdl_enter();
    unsigned int s = 0;
    while (s + 1 < nseg && blockIdx.x >= segs[s + 1].block0) s++;
    const MixSeg g = segs[s];
    const unsigned int blk = blockIdx.x - g.block0;
    if (g.scale == nullptr) {
        dense_gemv_bf16_batchm_body(g.a, (const __nv_bfloat16*)g.w, g.c, 1u, g.n, K, g.n, blk);
        return;
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blk * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
#if defined(__HIP_PLATFORM_AMD__)
    __shared__ unsigned int s_lut2[256];
    stage_e2m1_lut2_block(s_lut2);
#else
    const unsigned int* s_lut2 = nullptr;
#endif
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    if (n < g.n) {
        acc = w4a16_gemv_partial(g.a, g.w, g.scale, g.scale2, n, half_K, num_groups, K16, lane, s_lut, s_lut2);
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (warp_lane == 0) smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    __syncthreads();
    if (lane == 0 && n < g.n) {
        g.c[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
    }
}

