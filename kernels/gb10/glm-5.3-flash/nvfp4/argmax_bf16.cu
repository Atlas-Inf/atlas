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

// ── Sampled verify on the vocabulary split (`glm_verify_sample`) ──
//
// A sampled verify row needs the target's distribution, not its argmax, but
// each rank holds only its part of the row. Each rank reports its best
// GLM_SAMPLE_C logits (value descending, then the lower index: the sampler's
// order), the logits of up to four special tokens the host pipeline may mask,
// and the mass of everything else at the row's temperature; the head samples
// from those (`spark_runtime::sampler::sample_with_absent`).
#define GLM_SAMPLE_C 64
#define GLM_SAMPLE_REC_WORDS (8 + 2 * GLM_SAMPLE_C)

// BF16 bits to an unsigned key ordered like the values (-0 is +0).
__device__ __forceinline__ unsigned int glm_bf16_key(unsigned short b) {
    if (b == 0x8000u) {
        b = 0;
    }
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ float glm_key_value(unsigned int k) {
    const unsigned int b = (k & 0x8000u) ? (k ^ 0x8000u) : (~k & 0xFFFFu);
    return __uint_as_float(b << 16);
}

// The highest key `t` with at least `need` keys >= t among keys whose high
// byte is `high` (pass 2, `high_only`) or among all keys (pass 1, on the high
// byte): thread 0 scans the 256-bin histogram from the top. Returns the bin and
// the count strictly above it.
__device__ __forceinline__ void glm_hist_cut(
    const unsigned int* hist, unsigned int above, unsigned int need,
    unsigned int* bin, unsigned int* count_above
) {
    unsigned int acc = above;
    for (int h = 255; h >= 0; --h) {
        if (acc + hist[h] >= need) {
            *bin = (unsigned int)h;
            *count_above = acc;
            return;
        }
        acc += hist[h];
    }
    *bin = 0;
    *count_above = acc;
}

// Grid: (rows, 1, 1)  Block: (1024, 1, 1). Row r reads this rank's `n` logits
// at logits[r * row_stride ..] (vocabulary ids start..start+n) and writes
// out[r * GLM_SAMPLE_REC_WORDS ..]:
//   [0] candidates c = min(C, n)   [1] best value (f32)
//   [2] Σ exp((v - best) * inv_t) over this rank's other non-special tokens
//   [4..8) the special tokens' logits (f32), 0xFFFFFFFF where not this rank's
//   [8 + 2j, 9 + 2j) candidate j: value (f32), vocabulary id; sorted.
// inv_t == 0 skips the row (c = 0). Every step is deterministic.
extern "C" __global__ void argmax_bf16_candidates(
    const unsigned short* __restrict__ logits,
    unsigned int* __restrict__ out,
    unsigned int n,
    unsigned int row_stride,
    unsigned int start,
    float inv_t,
    unsigned int s0,
    unsigned int s1,
    unsigned int s2,
    unsigned int s3
) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_cut[4];
    __shared__ unsigned int s_warp[2][32];
    __shared__ unsigned int s_off[2][33];
    __shared__ unsigned long long s_cand[GLM_SAMPLE_C];
    __shared__ float s_red[1024];

    const unsigned short* row = logits + (size_t)blockIdx.x * row_stride;
    unsigned int* o = out + (size_t)blockIdx.x * GLM_SAMPLE_REC_WORDS;
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned int specials[4] = {s0, s1, s2, s3};
    if (inv_t == 0.0f) {
        if (tid == 0) {
            o[0] = 0;
        }
        return;
    }
    const unsigned int c = n < GLM_SAMPLE_C ? n : GLM_SAMPLE_C;

    // Pass 1: high-byte histogram and the best key.
    if (tid < 256) {
        hist[tid] = 0;
    }
    if (tid == 0) {
        s_cut[3] = 0;
    }
    __syncthreads();
    unsigned int best = 0;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        const unsigned int k = glm_bf16_key(row[i]);
        atomicAdd(&hist[k >> 8], 1u);
        best = max(best, k);
    }
    atomicMax(&s_cut[3], best);
    __syncthreads();
    if (tid == 0) {
        glm_hist_cut(hist, 0, c, &s_cut[0], &s_cut[1]);
    }
    __syncthreads();
    const unsigned int high = s_cut[0], above_high = s_cut[1];
    const float vmax = glm_key_value(s_cut[3]);

    // Pass 2: low-byte histogram inside the cut's high byte: the C-th key.
    if (tid < 256) {
        hist[tid] = 0;
    }
    __syncthreads();
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        const unsigned int k = glm_bf16_key(row[i]);
        if ((k >> 8) == high) {
            atomicAdd(&hist[k & 255], 1u);
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int low, gt;
        glm_hist_cut(hist, above_high, c, &low, &gt);
        s_cut[0] = (high << 8) | low;
        s_cut[1] = gt;
    }
    __syncthreads();
    const unsigned int cut = s_cut[0], count_gt = s_cut[1], need_eq = c - count_gt;

    // Pass 3, in index order a tile at a time: keys above the cut, then the
    // lowest-index ties at it, are the candidates; the rest is the tail.
    float tail = 0.0f;
    unsigned int base_gt = 0, base_eq = 0;
    const unsigned int lt_mask = (1u << lane) - 1u;
    for (unsigned int t0 = 0; t0 < n; t0 += blockDim.x) {
        const unsigned int i = t0 + tid;
        const bool valid = i < n;
        const unsigned short b = valid ? row[i] : 0;
        const unsigned int k = valid ? glm_bf16_key(b) : 0;
        const bool gt = valid && k > cut, eq = valid && k == cut;
        const unsigned int bg = __ballot_sync(0xFFFFFFFFu, gt);
        const unsigned int be = __ballot_sync(0xFFFFFFFFu, eq);
        if (lane == 0) {
            s_warp[0][warp] = __popc(bg);
            s_warp[1][warp] = __popc(be);
        }
        __syncthreads();
        if (tid < 2) {
            unsigned int acc = 0;
            for (unsigned int w = 0; w < blockDim.x / 32; ++w) {
                s_off[tid][w] = acc;
                acc += s_warp[tid][w];
            }
            s_off[tid][32] = acc;
        }
        __syncthreads();
        const unsigned int pos_gt = base_gt + s_off[0][warp] + __popc(bg & lt_mask);
        const unsigned int pos_eq = base_eq + s_off[1][warp] + __popc(be & lt_mask);
        const bool take_eq = eq && pos_eq < need_eq;
        if (gt || take_eq) {
            const unsigned int slot = gt ? pos_gt : count_gt + pos_eq;
            s_cand[slot] = ((unsigned long long)(0xFFFFu - k) << 32) | i;
        } else if (valid) {
            const unsigned int id = start + i;
            if (id != specials[0] && id != specials[1] && id != specials[2] && id != specials[3]) {
                tail += __expf((__uint_as_float((unsigned int)b << 16) - vmax) * inv_t);
            }
        }
        base_gt += s_off[0][32];
        base_eq += s_off[1][32];
        __syncthreads();
    }

    // Deterministic tree sum of the tail.
    s_red[tid] = tail;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_red[tid] += s_red[tid + s];
        }
        __syncthreads();
    }

    // Bitonic sort of the candidates on (value descending, index ascending).
    if (tid < GLM_SAMPLE_C && tid >= c) {
        s_cand[tid] = ~0ull;
    }
    __syncthreads();
    for (unsigned int size = 2; size <= GLM_SAMPLE_C; size <<= 1) {
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < GLM_SAMPLE_C) {
                const unsigned int j = tid ^ stride;
                if (j > tid) {
                    const bool up = (tid & size) == 0;
                    const unsigned long long a = s_cand[tid], d = s_cand[j];
                    if ((a > d) == up) {
                        s_cand[tid] = d;
                        s_cand[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid < c) {
        const unsigned long long e = s_cand[tid];
        const unsigned int k = 0xFFFFu - (unsigned int)(e >> 32);
        o[8 + 2 * tid] = __float_as_uint(glm_key_value(k));
        o[9 + 2 * tid] = start + (unsigned int)(e & 0xFFFFFFFFu);
    }
    if (tid < 4) {
        const unsigned int id = specials[tid];
        o[4 + tid] = (id >= start && id - start < n)
            ? __float_as_uint(__uint_as_float((unsigned int)row[id - start] << 16))
            : 0xFFFFFFFFu;
    }
    if (tid == 0) {
        o[0] = c;
        o[1] = __float_as_uint(vmax);
        o[2] = __float_as_uint(s_red[0]);
        o[3] = 0;
    }
}
