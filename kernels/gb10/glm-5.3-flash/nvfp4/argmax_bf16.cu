// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash: the common `argmax_bf16.cu` kernels verbatim, plus the GLM-only
// entry points below. Including (not copying) keeps this shadow in step
// with its namesake.
#include "../../common/argmax_bf16.cu"

// Same tie rule as common/argmax_bf16.cu: maximize value, then take the
// LOWER vocabulary index, so both TP2 vocabulary halves and every serial
// call agree.
__device__ __forceinline__ bool glm_argmax_other_better(
    float other,
    unsigned int other_idx,
    float mine,
    unsigned int mine_idx
) {
    return other > mine || (other == mine && other_idx < mine_idx);
}

// Local shard argmax for distributed vocabulary projection. The reduction is
// byte-for-byte the same as `argmax_bf16`; thread zero additionally preserves
// the winning BF16 value as FP32 beside its shard-local index. Two ranks can
// exchange this 8-byte pair instead of all BF16 logits. Merged like
// `argmax_pair_merge_ban`, a tied valid maximum keeps rank zero's pair (the
// lower vocabulary IDs, as a single-GPU argmax would pick), and two untouched
// -1e30f sentinels select index zero, preserving the full-row all-invalid
// fallback. No BF16 value equals that FP32 sentinel, so the host can
// distinguish those cases.
extern "C" __global__ void argmax_bf16_value(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out_value_index,
    unsigned int n
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = blockDim.x;
    float local_max = -1e30f;
    unsigned int local_idx = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = __bfloat162float(logits[i]);
        if (glm_argmax_other_better(v, i, local_max, local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && glm_argmax_other_better(
                s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_value_index[0] = __float_as_uint(s_val[0]);
        out_value_index[1] = s_idx[0];
    }
}

// Per-row TP2 shard argmax that also reports the best index outside a small
// banned set (min_tokens end tokens), in one pass. Row r reads
// logits[r * row_stride .. + n]; `ban0..3` are shard-local indices
// (0xFFFFFFFF = unused). out[r] = {best bits, best index, unbanned bits,
// unbanned index}, same comparator as argmax_bf16_value.
//
// ALLOW: only indices whose bit is set in `allow` (this row's full-vocabulary
// grammar bitmask; shard-local index i is vocabulary bit allow_bit0 + i, so a
// shard may start inside a 32-bit word) compete, the masked argmax of a strict
// structured-output row (vLLM #14702 applies the same per-row bitmask before
// its sampler). A row with no allowed index keeps both -1e30f sentinels.
template <bool ALLOW>
__device__ __forceinline__ void glm_value_ban_row(
    const __nv_bfloat16* __restrict__ row,
    unsigned int* __restrict__ o,
    unsigned int n,
    const unsigned int* __restrict__ allow,
    unsigned int allow_bit0,
    unsigned int ban0,
    unsigned int ban1,
    unsigned int ban2,
    unsigned int ban3,
    float (*s_val)[1024],
    unsigned int (*s_idx)[1024]
) {
    const unsigned int tid = threadIdx.x;
    float best = -1e30f, free_best = -1e30f;
    unsigned int best_idx = 0, free_idx = 0;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        if (ALLOW) {
            const unsigned int b = allow_bit0 + i;
            if (!((allow[b >> 5] >> (b & 31)) & 1u)) {
                continue;
            }
        }
        const float v = __bfloat162float(row[i]);
        if (glm_argmax_other_better(v, i, best, best_idx)) {
            best = v;
            best_idx = i;
        }
        const bool banned = i == ban0 || i == ban1 || i == ban2 || i == ban3;
        if (!banned && glm_argmax_other_better(v, i, free_best, free_idx)) {
            free_best = v;
            free_idx = i;
        }
    }
    s_val[0][tid] = best;
    s_idx[0][tid] = best_idx;
    s_val[1][tid] = free_best;
    s_idx[1][tid] = free_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int k = 0; k < 2; ++k) {
                if (glm_argmax_other_better(
                        s_val[k][tid + s], s_idx[k][tid + s], s_val[k][tid], s_idx[k][tid])) {
                    s_val[k][tid] = s_val[k][tid + s];
                    s_idx[k][tid] = s_idx[k][tid + s];
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        o[0] = __float_as_uint(s_val[0][0]);
        o[1] = s_idx[0][0];
        o[2] = __float_as_uint(s_val[1][0]);
        o[3] = s_idx[1][0];
    }
}

// Grid: (rows, 1, 1)  Block: (1024, 1, 1)
extern "C" __global__ void argmax_bf16_value_ban(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride,
    unsigned int ban0,
    unsigned int ban1,
    unsigned int ban2,
    unsigned int ban3
) {
    __shared__ float s_val[2][1024];
    __shared__ unsigned int s_idx[2][1024];
    glm_value_ban_row<false>(
        logits + (size_t)blockIdx.x * row_stride, out + 4 * blockIdx.x, n, nullptr, 0,
        ban0, ban1, ban2, ban3, s_val, s_idx);
}

// argmax_bf16_value_ban under a per-row bitmask: row r is masked by the
// full-vocabulary mask at allow[r * allow_stride ..] (32-bit words); this
// rank's shard starts at vocabulary bit allow_bit0 (any offset).
// Grid: (rows, 1, 1)  Block: (1024, 1, 1)
extern "C" __global__ void argmax_bf16_value_ban_allow(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride,
    const unsigned int* __restrict__ allow,
    unsigned int allow_stride,
    unsigned int allow_bit0,
    unsigned int ban0,
    unsigned int ban1,
    unsigned int ban2,
    unsigned int ban3
) {
    __shared__ float s_val[2][1024];
    __shared__ unsigned int s_idx[2][1024];
    glm_value_ban_row<true>(
        logits + (size_t)blockIdx.x * row_stride, out + 4 * blockIdx.x, n,
        allow + (size_t)blockIdx.x * allow_stride, allow_bit0, ban0, ban1, ban2, ban3,
        s_val, s_idx);
}

// Merge per-row TP2 shard quads from argmax_bf16_value_ban into global token
// IDs, on device so a graph-captured verify needs no host round trip. Ties
// follow the engine's first-index-wins argmax: rank 0 holds the lower IDs, so
// an exact tie (and two untouched sentinels) keeps rank 0's pair. Row r merges
// the unbanned pairs when bit r of (mask_hi:mask_lo) is set, else the best pairs.
//
// Grid: (1, 1, 1)  Block: (32, 1, 1)
extern "C" __global__ void argmax_pair_merge_ban(
    const unsigned int* __restrict__ local,
    const unsigned int* __restrict__ peer,
    unsigned int* __restrict__ out,
    unsigned int rows,
    unsigned int shard,
    unsigned int rank,
    unsigned int mask_lo,
    unsigned int mask_hi
) {
    for (unsigned int r = threadIdx.x; r < rows; r += blockDim.x) {
        const unsigned int word = r < 32 ? mask_lo : mask_hi;
        const unsigned int o = 4 * r + (((word >> (r & 31)) & 1u) ? 2 : 0);
        const unsigned int* p0 = rank == 0 ? local : peer;
        const unsigned int* p1 = rank == 0 ? peer : local;
        const float v0 = __uint_as_float(p0[o]);
        const float v1 = __uint_as_float(p1[o]);
        out[r] = v1 > v0 ? p1[o + 1] + shard : p0[o + 1];
    }
}
