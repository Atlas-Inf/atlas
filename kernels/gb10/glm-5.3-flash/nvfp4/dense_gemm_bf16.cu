// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the common `dense_gemm_bf16.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../common/dense_gemm_bf16.cu"

// KDA prefill beta | f_a | g_a: three weights over one activation in one grid.
// blockIdx.x walks the planes' N tiles (plane 0 has N0 columns, planes 1-2
// N12), so the CTAs of one A row tile launch back to back and share it in L2
// instead of three launches each streaming A from DRAM. Every plane runs the
// unchanged dense_gemm_bf16_pipelined body: bit-identical to its own launch.
// Grid: (ceil(N0/DM_N_TILE) + 2*ceil(N12/DM_N_TILE), ceil(M/DM_M_TILE), 1)
// Block: (256, 1, 1).
extern "C" __global__ void dense_gemm_bf16_pipelined_triple_n(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B0,
    const __nv_bfloat16* __restrict__ B1,
    const __nv_bfloat16* __restrict__ B2,
    __nv_bfloat16* __restrict__ C0,
    __nv_bfloat16* __restrict__ C1,
    __nv_bfloat16* __restrict__ C2,
    unsigned int M,
    unsigned int N0,
    unsigned int N12,
    unsigned int K
) {
    const unsigned int t0 = (N0 + DM_N_TILE - 1) / DM_N_TILE;
    const unsigned int t12 = (N12 + DM_N_TILE - 1) / DM_N_TILE;
    const unsigned int x = blockIdx.x;
    const unsigned int plane = x < t0 ? 0u : (x < t0 + t12 ? 1u : 2u);
    const unsigned int tile = plane == 0u ? x : x - t0 - (plane - 1u) * t12;
    dense_gemm_bf16_pipelined_tile(A, plane == 0u ? B0 : (plane == 1u ? B1 : B2),
        plane == 0u ? C0 : (plane == 1u ? C1 : C2), M, plane == 0u ? N0 : N12, K,
        blockIdx.y * DM_M_TILE, tile * DM_N_TILE);
}

// Exact-M=5 router GEMM for GLM speculative verification.
//
// The generic order-preserving router tile has 16 row lanes, so eleven lanes
// execute the complete K loop on zero-padded A rows at M=5. This specialization
// keeps one thread per real (row, column) output, raises the grid from five to
// eighteen CTAs for GLM's N=288 router, and retains the identical scalar
// k=0..K-1 FP32 accumulation chain. This translation unit is compiled with
// --fmad=false, so no FMA reassociation is introduced.
//
// Grid: (ceil(N/16), 1, 1)  Block: (16, 5, 1). Dispatch is shape-guarded.
#define R5_M 5
#define R5_BN 16
#define R5_BK 64

extern "C" __global__ void dense_gemm_bf16_router_m5(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    if (M != R5_M || blockDim.x != R5_BN || blockDim.y != R5_M) return;

    __shared__ float sA[R5_M][R5_BK + 1];
    __shared__ float sB[R5_BN][R5_BK + 1];

    const unsigned int tid = threadIdx.y * R5_BN + threadIdx.x; // 0..79
    const unsigned int row = threadIdx.y;
    const unsigned int local_col = threadIdx.x;
    const unsigned int col = blockIdx.x * R5_BN + local_col;
    float acc = 0.0f;

    for (unsigned int kb = 0; kb < K; kb += R5_BK) {
        // A is exactly 5x64 BF16 values per full tile: 80 vector loads.
        const unsigned int a_chunk = tid;
        const unsigned int ar = a_chunk / (R5_BK / 4);
        const unsigned int ak = (a_chunk % (R5_BK / 4)) * 4;
        if (kb + ak + 3 < K) {
            ushort4 v = *(const ushort4*)(A + (size_t)ar * K + kb + ak);
            sA[ar][ak + 0] = __bfloat162float(__ushort_as_bfloat16(v.x));
            sA[ar][ak + 1] = __bfloat162float(__ushort_as_bfloat16(v.y));
            sA[ar][ak + 2] = __bfloat162float(__ushort_as_bfloat16(v.z));
            sA[ar][ak + 3] = __bfloat162float(__ushort_as_bfloat16(v.w));
        } else {
            #pragma unroll
            for (unsigned int j = 0; j < 4; j++)
                sA[ar][ak + j] = kb + ak + j < K
                    ? __bfloat162float(A[(size_t)ar * K + kb + ak + j]) : 0.0f;
        }

        // B is 16x64 BF16 values. Flattened vector chunks keep each load
        // contiguous within one checkpoint row while all 80 threads assist.
        for (unsigned int b_chunk = tid; b_chunk < R5_BN * (R5_BK / 4); b_chunk += R5_M * R5_BN) {
            const unsigned int bn = b_chunk / (R5_BK / 4);
            const unsigned int bk = (b_chunk % (R5_BK / 4)) * 4;
            const unsigned int global_n = blockIdx.x * R5_BN + bn;
            if (global_n < N && kb + bk + 3 < K) {
                ushort4 v = *(const ushort4*)(B + (size_t)global_n * K + kb + bk);
                sB[bn][bk + 0] = __bfloat162float(__ushort_as_bfloat16(v.x));
                sB[bn][bk + 1] = __bfloat162float(__ushort_as_bfloat16(v.y));
                sB[bn][bk + 2] = __bfloat162float(__ushort_as_bfloat16(v.z));
                sB[bn][bk + 3] = __bfloat162float(__ushort_as_bfloat16(v.w));
            } else {
                #pragma unroll
                for (unsigned int j = 0; j < 4; j++)
                    sB[bn][bk + j] = global_n < N && kb + bk + j < K
                        ? __bfloat162float(B[(size_t)global_n * K + kb + bk + j]) : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (unsigned int kk = 0; kk < R5_BK; kk++)
            acc += sA[row][kk] * sB[local_col][kk];
        __syncthreads();
    }

    if (col < N) C[(size_t)row * N + col] = __float2bfloat16(acc);
}

#undef R5_BK
#undef R5_BN
#undef R5_M

#include "router_prefill_bn32.cuh"

// ─────────────────────────────────────────────────────────────────────────
// Order-preserving router GEMM for short decode/verify batches (M <= 32).
//
// dense_gemm_bf16's 16x16 tiles and dense_gemm_bf16_router's 16x64 tiles give
// a [<=16, 288] router product only 18 or 5 blocks. Here one warp owns one
// output column n and lane m owns C[m, n]: every lane accumulates in strict
// k = 0..K-1 order with the same separate FMUL/FADD (--fmad=false) as the
// scalar kernel, so each output is BIT-IDENTICAL to dense_gemm_bf16, while
// all M*N accumulators run concurrently. The warp first stages its weight row
// in shared memory with coalesced loads, so the sequential chain reads the
// weight as a shared-memory broadcast instead of one dependent global load
// per 8 elements; each lane streams its own (cache-resident) activation row.
//
// Grid: (ceil(N/4), 1, 1)   Block: (128, 1, 1)
#define ROUTER_ROWS_WARPS 4
#define ROUTER_ROWS_MAX_K 4096
extern "C" __global__ void __launch_bounds__(128) dense_gemm_bf16_router_rows(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) __nv_bfloat16 s_b[ROUTER_ROWS_WARPS][ROUTER_ROWS_MAX_K];
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int n = blockIdx.x * ROUTER_ROWS_WARPS + warp;
    if (n >= N) return;
    const __nv_bfloat16* a = A + (unsigned long long)lane * K;
    const __nv_bfloat16* b = B + (unsigned long long)n * K;
    const bool vec = (K % 8) == 0 && ((unsigned long long)A % 16) == 0
        && ((unsigned long long)b % 16) == 0;
    const bool staged = vec && K <= ROUTER_ROWS_MAX_K;
    if (staged) {
        uint4* dst = (uint4*)s_b[warp];
        for (unsigned int c = lane; c < K / 8; c += 32) dst[c] = ((const uint4*)b)[c];
        __syncwarp();
    }
    if (lane >= M) return;
    float acc = 0.0f;
    unsigned int k = 0;
    if (staged) {
        const uint4* a4 = (const uint4*)a;
        const uint4* b4 = (const uint4*)s_b[warp];
        #pragma unroll 8
        for (unsigned int c = 0; c < K / 8; ++c) {
            const uint4 av = a4[c];
            const uint4 bv = b4[c];
            const __nv_bfloat16* ae = (const __nv_bfloat16*)&av;
            const __nv_bfloat16* be = (const __nv_bfloat16*)&bv;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc += __bfloat162float(ae[i]) * __bfloat162float(be[i]);
            }
        }
        k = K;
    } else if (vec) {
        for (; k < K; k += 8) {
            const uint4 av = *(const uint4*)(a + k);
            const uint4 bv = *(const uint4*)(b + k);
            const __nv_bfloat16* ae = (const __nv_bfloat16*)&av;
            const __nv_bfloat16* be = (const __nv_bfloat16*)&bv;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc += __bfloat162float(ae[i]) * __bfloat162float(be[i]);
            }
        }
    }
    for (; k < K; ++k) {
        acc += __bfloat162float(a[k]) * __bfloat162float(b[k]);
    }
    C[(unsigned long long)lane * N + n] = __float2bfloat16(acc);
}
