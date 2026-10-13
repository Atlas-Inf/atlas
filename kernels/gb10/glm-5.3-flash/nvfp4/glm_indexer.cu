// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 four-token semantic-index primitives. The correctness baseline
// stores pooled keys as BF16; the later FP8 path adds a Hadamard bookend and a
// per-vector scale without changing the pool addressing established here.

#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void glm_index_layernorm_bf16(
    __nv_bfloat16* __restrict__ values,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    unsigned int rows,
    unsigned int dim,
    float eps) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float reduce[];
    __nv_bfloat16* x = values + (unsigned long long)row * dim;
    float sum = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        sum += __bfloat162float(x[d]);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = reduce[0] / (float)dim;
    // Every thread must hold the mean before thread 0 reuses reduce[0] for
    // its squared sum: without this barrier a warp scheduled late reads the
    // squared sum as the row sum and normalizes its lanes with a wrong mean.
    __syncthreads();
    float sq = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float centered = __bfloat162float(x[d]) - mean;
        sq += centered * centered;
    }
    reduce[threadIdx.x] = sq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduce[0] / (float)dim + eps);
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float y = (__bfloat162float(x[d]) - mean) * inv_std
                      * __bfloat162float(weight[d]) + __bfloat162float(bias[d]);
        x[d] = __float2bfloat16(y);
    }
}

// Raw tails are indexed by physical block, or through `tail_map` (block ->
// lent tail slot) when the cache slot-maps them; an unmapped block is skipped.
__device__ __forceinline__ bool glm_index_tail_index(
    const unsigned int* __restrict__ tail_map,
    unsigned int physical_block,
    unsigned int* tail_index) {
    *tail_index = tail_map ? tail_map[physical_block] : physical_block;
    return *tail_index != 0xFFFFFFFFu;
}

// Persist the raw key and gate for every token in the physical block that owns
// it. This is deliberately a separate launch from finalization: CUDA provides
// stream ordering between launches, while different CTAs in one launch cannot
// safely publish and consume the four members of a pool.
extern "C" __global__ void glm_index_tail_write_bf16(
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ gates,
    __nv_bfloat16* __restrict__ tail,
    const unsigned int* __restrict__ tail_map,
    const long long* __restrict__ slots,
    unsigned int num_tokens,
    unsigned int block_size,
    unsigned int pool_size,
    unsigned int head_dim,
    unsigned long long tail_block_stride_bytes) {
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned int physical_block = (unsigned int)(slot / block_size);
    const unsigned int raw_offset = (unsigned int)(slot % block_size);
    unsigned int tail_index;
    if (!glm_index_tail_index(tail_map, physical_block, &tail_index)) return;
    __nv_bfloat16* block_tail = (__nv_bfloat16*)((char*)tail
        + (unsigned long long)tail_index * tail_block_stride_bytes);
    __nv_bfloat16* key_dst = block_tail + (unsigned long long)raw_offset * head_dim;
    __nv_bfloat16* gate_dst = block_tail
        + (unsigned long long)block_size * head_dim
        + (unsigned long long)raw_offset * head_dim;
    const __nv_bfloat16* key_src = keys + (unsigned long long)token * head_dim;
    const __nv_bfloat16* gate_src = gates + (unsigned long long)token * head_dim;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_dst[d] = key_src[d];
        gate_dst[d] = gate_src[d];
    }
}

// Finalize every pool-ending token after all raw keys/gates in this chunk have
// been published by the preceding stream-ordered launch. Addressing by raw
// physical-block offset makes arbitrary scheduler chunk boundaries exact.
extern "C" __global__ void glm_index_kpool_finalize_bf16(
    const __nv_bfloat16* __restrict__ tail,
    const unsigned int* __restrict__ tail_map,
    const __nv_bfloat16* __restrict__ ape,
    __nv_bfloat16* __restrict__ cache,
    const long long* __restrict__ slots,
    unsigned int num_tokens,
    unsigned int block_size,
    unsigned int pool_size,
    unsigned int head_dim,
    unsigned long long tail_block_stride_bytes,
    unsigned long long values_block_stride_bytes) {
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens || pool_size != 4) return;
    const long long slot = slots[token];
    if (slot < 0) return;
    const unsigned int physical_block = (unsigned int)(slot / block_size);
    const unsigned int raw_offset = (unsigned int)(slot % block_size);
    if (raw_offset % pool_size != pool_size - 1) return;
    unsigned int tail_index;
    if (!glm_index_tail_index(tail_map, physical_block, &tail_index)) return;
    const __nv_bfloat16* block_tail = (const __nv_bfloat16*)((const char*)tail
        + (unsigned long long)tail_index * tail_block_stride_bytes);
    const unsigned int first = raw_offset - (pool_size - 1);
    const __nv_bfloat16* tail_keys = block_tail
        + (unsigned long long)first * head_dim;
    const __nv_bfloat16* tail_gates = block_tail
        + (unsigned long long)block_size * head_dim
        + (unsigned long long)first * head_dim;
    __nv_bfloat16* dst = (__nv_bfloat16*)((char*)cache
        + (unsigned long long)physical_block * values_block_stride_bytes)
        + (unsigned long long)(raw_offset / pool_size) * head_dim;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float score[4];
        float max_score = -INFINITY;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            score[i] = __bfloat162float(tail_gates[(unsigned long long)i * head_dim + d])
                     + __bfloat162float(ape[(unsigned long long)i * head_dim + d]);
            max_score = fmaxf(max_score, score[i]);
        }
        float denom = 0.0f;
        float weighted = 0.0f;
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            const float p = expf(score[i] - max_score);
            denom += p;
            weighted += p * __bfloat162float(
                tail_keys[(unsigned long long)i * head_dim + d]);
        }
        dst[d] = __float2bfloat16(weighted / denom);
    }
}

extern "C" __global__ void glm_index_fill_causal(
    int* __restrict__ output,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int width) {
    const unsigned int row = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || col >= width) return;
    const unsigned int query_pos = seq_len_start + row;
    output[(unsigned long long)row * width + col] = col <= query_pos ? (int)col : -1;
}

// One decode row selecting every cached token: `kv_len[0]` (the device
// sequence length including the new token) stays graph-capture safe.
extern "C" __global__ void glm_index_fill_causal_dev(
    int* __restrict__ output,
    const int* __restrict__ kv_len,
    unsigned int width) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) output[col] = (int)col < kv_len[0] ? (int)col : -1;
}

// Eight warps score eight pooled keys per CTA. Each warp holds one pool and
// accumulates its 32 index-head dot products without materializing headwise
// logits. The cache remains paged with the main KV block table.
__device__ __forceinline__ void glm_index_logits_bf16_impl(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    const unsigned int row = blockIdx.y;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int pool_id = blockIdx.x * 8 + warp;
    if (row >= rows || pool_id >= logits_stride) return;
    const unsigned int pool_count = seq_len / pool_size;
    if (pool_id >= pool_count) {
        if (lane == 0) logits[(unsigned long long)row * logits_stride + pool_id] = -INFINITY;
        return;
    }

    const unsigned int raw_pos = pool_id * pool_size;
    const unsigned int logical_block = raw_pos / cache_block_size;
    const unsigned int raw_offset = raw_pos % cache_block_size;
    const unsigned int physical_block = block_table[logical_block];
    const unsigned int pool_offset = raw_offset / pool_size;
    const __nv_bfloat16* key = (const __nv_bfloat16*)((const char*)index_cache
        + (unsigned long long)physical_block * index_block_stride_bytes)
        + (unsigned long long)pool_offset * head_dim;
    const __nv_bfloat16* q = query
        + (unsigned long long)row * index_heads * head_dim;
    const __nv_bfloat16* w = weights + (unsigned long long)row * index_heads;
    float score = 0.0f;
    for (unsigned int head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            dot += __bfloat162float(q[(unsigned long long)head * head_dim + d])
                 * __bfloat162float(key[d]);
        }
        for (unsigned int offset = 16; offset; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) score += __bfloat162float(w[head]) * fmaxf(dot, 0.0f);
    }
    if (lane == 0) {
        logits[(unsigned long long)row * logits_stride + pool_id] =
            score * rsqrtf((float)(head_dim * index_heads));
    }
}

extern "C" __global__ void glm_index_logits_bf16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    glm_index_logits_bf16_impl(query, weights, index_cache, logits, block_table,
        rows, seq_len_start + blockIdx.y + 1, logits_stride, index_heads, head_dim,
        pool_size, cache_block_size, index_block_stride_bytes);
}

// Single-row graph scorer: launch the fixed capacity (ceil(stride/8),1,1)
// with 256 threads and no dynamic shared memory. Device length includes the
// token just appended. Short, empty, over-capacity and future pools produce
// -infinity before any query/key/page-table read.
extern "C" __global__ void glm_index_logits_bf16_dynamic(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    const unsigned int* __restrict__ seq_len,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes,
    unsigned int dense_threshold) {
    if (rows != 1 || blockIdx.y != 0 || blockDim.x != 256) return;
    const unsigned int length = seq_len[0];
    const bool valid = pool_size > 0 && cache_block_size > 0
        && length > dense_threshold
        && (unsigned long long)length <= (unsigned long long)logits_stride * pool_size;
    if (!valid) {
        const unsigned int pool = blockIdx.x * 8 + (threadIdx.x >> 5);
        if ((threadIdx.x & 31) == 0 && pool < logits_stride) logits[pool] = -INFINITY;
        return;
    }
    glm_index_logits_bf16_impl(query, weights, index_cache, logits, block_table,
        1, length, logits_stride, index_heads, head_dim, pool_size,
        cache_block_size, index_block_stride_bytes);
}

// Eight warps score an 8-row x 8-pool tile. A pooled key is loaded once into
// shared memory and reused by all query rows, while each warp loads its query
// values once and accumulates all eight pool scores. This preserves the
// per-score reduction order of glm_index_logits_bf16 while removing the
// redundant query and key traffic that dominates long-context prefill.
extern "C" __global__ void glm_index_logits_bf16_row8(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ index_cache,
    float* __restrict__ logits,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int index_heads,
    unsigned int head_dim,
    unsigned int pool_size,
    unsigned int cache_block_size,
    unsigned long long index_block_stride_bytes) {
    extern __shared__ __nv_bfloat16 pooled_keys[];
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int pool_base = blockIdx.x * 8;
    const unsigned int row_base = blockIdx.y * 8;

    // All warps cooperatively stage eight pooled keys before out-of-range row
    // warps exit, so partial row tiles cannot strand a block-wide barrier.
    const unsigned int max_pool_count = (seq_len_start + rows) / pool_size;
    for (unsigned int item = threadIdx.x; item < 8 * head_dim; item += blockDim.x) {
        const unsigned int pool_offset_in_tile = item / head_dim;
        const unsigned int d = item % head_dim;
        const unsigned int pool_id = pool_base + pool_offset_in_tile;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (pool_id < logits_stride && pool_id < max_pool_count) {
            const unsigned int raw_pos = pool_id * pool_size;
            const unsigned int logical_block = raw_pos / cache_block_size;
            const unsigned int raw_offset = raw_pos % cache_block_size;
            const unsigned int physical_block = block_table[logical_block];
            const unsigned int cache_pool_offset = raw_offset / pool_size;
            const __nv_bfloat16* key =
                (const __nv_bfloat16*)((const char*)index_cache
                    + (unsigned long long)physical_block * index_block_stride_bytes)
                + (unsigned long long)cache_pool_offset * head_dim;
            value = key[d];
        }
        pooled_keys[item] = value;
    }
    __syncthreads();

    const unsigned int row = row_base + warp;
    if (row >= rows) return;
    const __nv_bfloat16* q = query
        + (unsigned long long)row * index_heads * head_dim;
    const __nv_bfloat16* w = weights + (unsigned long long)row * index_heads;
    float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int head = 0; head < index_heads; ++head) {
        float dots[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int d = lane; d < head_dim; d += 32) {
            const float qv = __bfloat162float(q[(unsigned long long)head * head_dim + d]);
#pragma unroll
            for (unsigned int pool = 0; pool < 8; ++pool) {
                dots[pool] += qv * __bfloat162float(pooled_keys[pool * head_dim + d]);
            }
        }
        for (unsigned int offset = 16; offset; offset >>= 1) {
#pragma unroll
            for (unsigned int pool = 0; pool < 8; ++pool) {
                dots[pool] += __shfl_down_sync(0xffffffff, dots[pool], offset);
            }
        }
        if (lane == 0) {
            const float weight = __bfloat162float(w[head]);
#pragma unroll
            for (unsigned int pool = 0; pool < 8; ++pool) {
                scores[pool] += weight * fmaxf(dots[pool], 0.0f);
            }
        }
    }
    if (lane == 0) {
        const unsigned int pool_count = (seq_len_start + row + 1) / pool_size;
        const float scale = rsqrtf((float)(head_dim * index_heads));
#pragma unroll
        for (unsigned int pool = 0; pool < 8; ++pool) {
            const unsigned int pool_id = pool_base + pool;
            if (pool_id < logits_stride) {
                logits[(unsigned long long)row * logits_stride + pool_id] =
                    pool_id < pool_count ? scores[pool] * scale : -INFINITY;
            }
        }
    }
}

__device__ __forceinline__ unsigned int glm_ordered_float(float value) {
    if (isnan(value)) value = -INFINITY;
    const unsigned int bits = __float_as_uint(value);
    const unsigned int flip = ((int)bits < 0) ? 0xffffffffu : 0x80000000u;
    return bits ^ flip;
}

// Exact per-row top-K for signed FP32 logits using a radix threshold, found
// with four 8-bit histogram passes over the row (a bit-at-a-time search read
// it 32 times). The selected pools — every pool above the Kth score, then the
// lowest-indexed pools equal to it — are expanded to their four raw token IDs
// in ascending order, so the selection and the attention's summation order
// are deterministic. Block: any multiple of 32 up to 1024 threads.
__device__ __forceinline__ unsigned int glm_topk_block_exclusive_scan(
    unsigned int flag, unsigned int* warp_sums, unsigned int& total) {
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int warps = (blockDim.x + 31) >> 5;
    const unsigned int ballot = __ballot_sync(0xffffffffu, flag);
    const unsigned int in_warp = __popc(ballot & ((1u << lane) - 1u));
    __syncthreads();   // warp_sums free from any previous scan
    if (lane == 0) warp_sums[warp] = __popc(ballot);
    __syncthreads();
    unsigned int before = 0;
    total = 0;
    for (unsigned int w = 0; w < warps; ++w) {
        const unsigned int c = warp_sums[w];
        if (w < warp) before += c;
        total += c;
    }
    return before + in_warp;
}

__device__ __forceinline__ void glm_index_topk_expand_impl(
    const float* __restrict__ logits,
    int* __restrict__ output,
    unsigned int rows,
    unsigned int seq_len,
    unsigned int logits_stride,
    unsigned int topk_tokens,
    unsigned int pool_size,
    unsigned int output_width) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int warp_sums[32];
    __shared__ unsigned int s_prefix, s_rank;
    const unsigned int pool_count = seq_len / pool_size;
    const unsigned int pool_budget = topk_tokens / pool_size;
    const unsigned int select_pools = pool_budget < pool_count ? pool_budget : pool_count;
    int* out = output + (unsigned long long)row * output_width;
    for (unsigned int i = threadIdx.x; i < output_width; i += blockDim.x) out[i] = -1;
    const float* row_logits = logits + (unsigned long long)row * logits_stride;

    // Threshold: the select_pools-th largest ordered score, and how many
    // pools equal to it are taken (`rank`).
    unsigned int prefix = 0, mask = 0, rank = select_pools;
    if (select_pools == pool_count) {
        prefix = 0;      // everything: any score is >= 0 in ordered space
        rank = 0;        // ... and no tie filling is needed
    } else {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (unsigned int p = threadIdx.x; p < pool_count; p += blockDim.x) {
                const unsigned int bits = glm_ordered_float(row_logits[p]);
                if ((bits & mask) == prefix) atomicAdd(&hist[(bits >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned int need = rank, bin = 255;
                for (;; --bin) {
                    const unsigned int c = hist[bin];
                    if (c >= need || bin == 0) break;
                    need -= c;
                }
                s_prefix = prefix | (bin << shift);
                s_rank = need;
            }
            __syncthreads();
            prefix = s_prefix;
            rank = s_rank;
            mask |= 255u << shift;
        }
    }

    // Ordered selection: scores above the threshold, then the first `rank`
    // pools equal to it, written in ascending pool order.
    unsigned int ties_seen = 0, written = 0;
    for (unsigned int base = 0; base < pool_count; base += blockDim.x) {
        const unsigned int p = base + threadIdx.x;
        const bool live = p < pool_count;
        const unsigned int bits = live ? glm_ordered_float(row_logits[p]) : 0u;
        const bool gt = live && (select_pools == pool_count || bits > prefix);
        const bool eq = live && select_pools != pool_count && bits == prefix;
        unsigned int eq_total;
        const unsigned int tie_rank = ties_seen + glm_topk_block_exclusive_scan(eq, warp_sums, eq_total);
        const bool take = gt || (eq && tie_rank < rank);
        unsigned int take_total;
        const unsigned int dst = written + glm_topk_block_exclusive_scan(take, warp_sums, take_total);
        if (take && dst < select_pools) {
            for (unsigned int i = 0; i < pool_size; ++i)
                out[dst * pool_size + i] = (int)(p * pool_size + i);
        }
        ties_seen += eq_total;
        written += take_total;
    }
    const unsigned int tail_start = pool_count * pool_size;
    const unsigned int tail_count = seq_len - tail_start;
    for (unsigned int i = threadIdx.x; i < tail_count; i += blockDim.x) {
        out[topk_tokens + i] = (int)(tail_start + i);
    }
}

extern "C" __global__ void glm_index_topk_expand(
    const float* __restrict__ logits,
    int* __restrict__ output,
    unsigned int rows,
    unsigned int seq_len_start,
    unsigned int logits_stride,
    unsigned int topk_tokens,
    unsigned int pool_size,
    unsigned int output_width) {
    glm_index_topk_expand_impl(logits, output, rows, seq_len_start + blockIdx.x + 1,
        logits_stride, topk_tokens, pool_size, output_width);
}

// Single-row graph selector: grid=(1,1,1), block=256, shared=16 bytes.
// Always reset both outputs, including transitions long->short and slot reuse.
// dense_seq_len belongs to this row until both attention launches finish.
extern "C" __global__ void glm_index_topk_expand_dynamic(
    const float* __restrict__ logits,
    int* __restrict__ output,
    unsigned int rows,
    const unsigned int* __restrict__ seq_len,
    unsigned int logits_stride,
    unsigned int topk_tokens,
    unsigned int pool_size,
    unsigned int output_width,
    unsigned int* __restrict__ dense_seq_len) {
    if (rows != 1 || blockIdx.x != 0 || blockDim.x != 256) return;
    const unsigned int length = seq_len[0];
    const bool valid = length > 0 && pool_size > 0 && topk_tokens > 0
        && topk_tokens % pool_size == 0
        && (unsigned long long)output_width >= (unsigned long long)topk_tokens + pool_size - 1
        && (unsigned long long)length <= (unsigned long long)logits_stride * pool_size;
    if (threadIdx.x == 0)
        dense_seq_len[0] = valid && length <= topk_tokens ? length : 0;
    if (!valid || length <= topk_tokens) {
        for (unsigned int i = threadIdx.x; i < output_width; i += blockDim.x) output[i] = -1;
        return;
    }
    glm_index_topk_expand_impl(logits, output, 1, length, logits_stride,
        topk_tokens, pool_size, output_width);
}

// Tiled sparse absorbed MLA. One CTA owns one (query, head); its eight warps
// score eight selected tokens concurrently, then all 256 lanes update two of
// the 512 output dimensions. This preserves the online-softmax recurrence of
// the scalar oracle while reducing synchronization by roughly eightfold.
__device__ __forceinline__ void glm_sparse_mla_prefill_bf16_impl(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (row >= rows || head >= num_heads || head_dim != 512) return;
    extern __shared__ float glm_sparse_head1_shared[];
    float* scores = glm_sparse_head1_shared;
    float* betas = glm_sparse_head1_shared + 8;
    float& alpha = glm_sparse_head1_shared[16];
    float& running_max = glm_sparse_head1_shared[17];
    float& running_denom = glm_sparse_head1_shared[18];
    const __nv_bfloat16* q = query
        + ((unsigned long long)row * num_heads + head) * head_dim;
    const int* indices = token_indices + (unsigned long long)row * index_width;
    const unsigned long long cache_block_stride =
        (unsigned long long)cache_block_size * head_dim;
    float out0 = 0.0f, out1 = 0.0f;
    if (tid == 0) {
        running_max = -INFINITY;
        running_denom = 0.0f;
    }
    __syncthreads();

    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    for (unsigned int tile = 0; tile < index_width; tile += 8) {
        const unsigned int selected = tile + warp;
        const int token = selected < index_width ? indices[selected] : -1;
        float partial = 0.0f;
        if (token >= 0) {
            const unsigned int logical_block = (unsigned int)token / cache_block_size;
            const unsigned int block_offset = (unsigned int)token % cache_block_size;
            const unsigned int physical_block = block_table[logical_block];
            const __nv_bfloat16* k = k_cache
                + (unsigned long long)physical_block * cache_block_stride
                + (unsigned long long)block_offset * head_dim;
            for (unsigned int d = lane; d < head_dim; d += 32)
                partial += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        }
        for (unsigned int offset = 16; offset; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) scores[warp] = token >= 0 ? partial * inv_sqrt_d : -INFINITY;
        __syncthreads();

        if (tid == 0) {
            float tile_max = -INFINITY;
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) tile_max = fmaxf(tile_max, scores[i]);
            const float next_max = fmaxf(running_max, tile_max);
            alpha = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                betas[i] = scores[i] == -INFINITY ? 0.0f : expf(scores[i] - next_max);
                tile_sum += betas[i];
            }
            running_denom = running_denom * alpha + tile_sum;
            running_max = next_max;
        }
        __syncthreads();

        float next0 = out0 * alpha;
        float next1 = out1 * alpha;
#pragma unroll
        for (unsigned int i = 0; i < 8; ++i) {
            const unsigned int item = tile + i;
            const int value_token = item < index_width ? indices[item] : -1;
            if (value_token < 0 || betas[i] == 0.0f) continue;
            const unsigned int logical_block = (unsigned int)value_token / cache_block_size;
            const unsigned int block_offset = (unsigned int)value_token % cache_block_size;
            const unsigned int physical_block = block_table[logical_block];
            const __nv_bfloat16* v = v_cache
                + (unsigned long long)physical_block * cache_block_stride
                + (unsigned long long)block_offset * head_dim;
            next0 += betas[i] * __bfloat162float(v[tid]);
            next1 += betas[i] * __bfloat162float(v[tid + 256]);
        }
        out0 = next0;
        out1 = next1;
        __syncthreads();
    }
    __nv_bfloat16* dst = output
        + ((unsigned long long)row * num_heads + head) * head_dim;
    const float inv_denom = running_denom > 0.0f ? 1.0f / running_denom : 0.0f;
    dst[tid] = __float2bfloat16(out0 * inv_denom);
    dst[tid + 256] = __float2bfloat16(out1 * inv_denom);
}

extern "C" __global__ void glm_sparse_mla_prefill_bf16(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d) {
    glm_sparse_mla_prefill_bf16_impl(query, k_cache, v_cache, token_indices, output,
        block_table, rows, num_heads, head_dim, index_width, cache_block_size, inv_sqrt_d);
}

// Single-row graph sparse attention: grid=(num_heads,1,1), block=256,
// shared=19*sizeof(float). This uniform guard is before every CTA barrier.
// For invalid over-capacity lengths the preceding dynamic selector supplies
// only -1 IDs, so this body performs no paged reads and writes zeros.
extern "C" __global__ void glm_sparse_mla_prefill_bf16_dynamic(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d,
    const unsigned int* __restrict__ seq_len,
    unsigned int dense_threshold) {
    if (rows != 1 || blockIdx.y != 0 || blockDim.x != 256
        || cache_block_size == 0 || seq_len[0] <= dense_threshold) return;
    glm_sparse_mla_prefill_bf16_impl(query, k_cache, v_cache, token_indices, output,
        block_table, 1, num_heads, head_dim, index_width, cache_block_size, inv_sqrt_d);
}

// Eight-head sparse absorbed MLA. GLM's compressed K/V row is shared by all
// query heads, so assigning a CTA to eight heads lets every warp load a
// selected K row once for eight dot products and lets every lane load each V
// element once for eight accumulators. The one-head kernel above remains the
// low-register-pressure fallback selected with ATLAS_GLM_SPARSE_HEAD_GROUP=1.
extern "C" __global__ void glm_sparse_mla_prefill_bf16_head8(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d) {
    constexpr unsigned int HEADS = 8;
    constexpr unsigned int ITEMS = 8;
    const unsigned int head_base = blockIdx.x * HEADS;
    const unsigned int row = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (row >= rows || head_base >= num_heads || head_dim != 512) return;

    extern __shared__ float shared_f[];
    float* scores = shared_f;
    float* betas = scores + HEADS * ITEMS;
    float* alphas = betas + HEADS * ITEMS;
    float* running_max = alphas + HEADS;
    float* running_denom = running_max + HEADS;
    const int* indices = token_indices + (unsigned long long)row * index_width;
    const unsigned long long cache_block_stride =
        (unsigned long long)cache_block_size * head_dim;
    float out0[HEADS];
    float out1[HEADS];
#pragma unroll
    for (unsigned int h = 0; h < HEADS; ++h) {
        out0[h] = 0.0f;
        out1[h] = 0.0f;
    }
    if (tid < HEADS) {
        running_max[tid] = -INFINITY;
        running_denom[tid] = 0.0f;
    }
    __syncthreads();

    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    for (unsigned int tile = 0; tile < index_width; tile += ITEMS) {
        const unsigned int selected = tile + warp;
        const int token = selected < index_width ? indices[selected] : -1;
        float partial[HEADS];
#pragma unroll
        for (unsigned int h = 0; h < HEADS; ++h) partial[h] = 0.0f;
        if (token >= 0) {
            const unsigned int logical_block = (unsigned int)token / cache_block_size;
            const unsigned int block_offset = (unsigned int)token % cache_block_size;
            const unsigned int physical_block = block_table[logical_block];
            const __nv_bfloat16* k = k_cache
                + (unsigned long long)physical_block * cache_block_stride
                + (unsigned long long)block_offset * head_dim;
            for (unsigned int d = lane; d < head_dim; d += 32) {
                const float kval = __bfloat162float(k[d]);
#pragma unroll
                for (unsigned int h = 0; h < HEADS; ++h) {
                    const unsigned int head = head_base + h;
                    if (head < num_heads) {
                        const __nv_bfloat16* q = query
                            + ((unsigned long long)row * num_heads + head) * head_dim;
                        partial[h] += __bfloat162float(q[d]) * kval;
                    }
                }
            }
        }
#pragma unroll
        for (unsigned int h = 0; h < HEADS; ++h) {
            for (unsigned int offset = 16; offset; offset >>= 1)
                partial[h] += __shfl_down_sync(0xffffffff, partial[h], offset);
            if (lane == 0) {
                scores[h * ITEMS + warp] =
                    token >= 0 && head_base + h < num_heads
                        ? partial[h] * inv_sqrt_d
                        : -INFINITY;
            }
        }
        __syncthreads();

        if (tid < HEADS) {
            const unsigned int h = tid;
            float tile_max = -INFINITY;
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i)
                tile_max = fmaxf(tile_max, scores[h * ITEMS + i]);
            const float next_max = fmaxf(running_max[h], tile_max);
            alphas[h] = running_max[h] == -INFINITY
                ? 0.0f
                : expf(running_max[h] - next_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i) {
                const float score = scores[h * ITEMS + i];
                const float beta = score == -INFINITY ? 0.0f : expf(score - next_max);
                betas[h * ITEMS + i] = beta;
                tile_sum += beta;
            }
            running_denom[h] = running_denom[h] * alphas[h] + tile_sum;
            running_max[h] = next_max;
        }
        __syncthreads();

        float values0[ITEMS];
        float values1[ITEMS];
#pragma unroll
        for (unsigned int i = 0; i < ITEMS; ++i) {
            const unsigned int item = tile + i;
            const int value_token = item < index_width ? indices[item] : -1;
            if (value_token < 0) {
                values0[i] = 0.0f;
                values1[i] = 0.0f;
                continue;
            }
            const unsigned int logical_block = (unsigned int)value_token / cache_block_size;
            const unsigned int block_offset = (unsigned int)value_token % cache_block_size;
            const unsigned int physical_block = block_table[logical_block];
            const __nv_bfloat16* v = v_cache
                + (unsigned long long)physical_block * cache_block_stride
                + (unsigned long long)block_offset * head_dim;
            values0[i] = __bfloat162float(v[tid]);
            values1[i] = __bfloat162float(v[tid + 256]);
        }
#pragma unroll
        for (unsigned int h = 0; h < HEADS; ++h) {
            float next0 = out0[h] * alphas[h];
            float next1 = out1[h] * alphas[h];
#pragma unroll
            for (unsigned int i = 0; i < ITEMS; ++i) {
                const float beta = betas[h * ITEMS + i];
                next0 += beta * values0[i];
                next1 += beta * values1[i];
            }
            out0[h] = next0;
            out1[h] = next1;
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int h = 0; h < HEADS; ++h) {
        const unsigned int head = head_base + h;
        if (head >= num_heads) continue;
        __nv_bfloat16* dst = output
            + ((unsigned long long)row * num_heads + head) * head_dim;
        const float inv_denom = running_denom[h] > 0.0f
            ? 1.0f / running_denom[h]
            : 0.0f;
        dst[tid] = __float2bfloat16(out0[h] * inv_denom);
        dst[tid + 256] = __float2bfloat16(out1[h] * inv_denom);
    }
}
