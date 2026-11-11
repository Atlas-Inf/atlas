// SPDX-License-Identifier: AGPL-3.0-only

// Experimental BF16 tensor-core semantic scorer. This is deliberately separate
// from glm_indexer.cu: FP32 tensor-core dot products have a different reduction
// order from the scalar scorer and must pass top-K and model-quality checks
// before runtime dispatch is enabled.
//
// All entry points have the glm_index_logits_bf16_row8 argument ABI. Launch:
//   row8:        grid=(ceil(logits_stride/16), ceil(rows/8)), block=(256,1,1)
//   row8_pool32: grid=(ceil(logits_stride/32), ceil(rows/8)), block=(256,1,1)
//   mma_v2:      grid=(any x >= 1, ceil(rows/4)), block=(128,1,1)
// No scratch allocation or special shared-memory opt-in is needed. row8 and
// row8_pool32 use 12,800 / 25,600 bytes of static shared memory; mma_v2 uses
// 32,768 bytes of dynamic shared memory and 16-byte aligned cache blocks.
// Requirements: SM80+, 32-byte aligned query, heads=32, head_dim=128,
// pool_size=4, cache_block_size a positive multiple of 4. Cache page/table and
// output sizing follow the scalar scorer. Unsupported geometry (for mma_v2 also
// too little dynamic shared memory) does no work; the caller must validate
// these requirements before selecting this kernel.

#include <cuda_bf16.h>
#include <math.h>
#include <mma.h>

namespace {

template <unsigned int KeyTiles>
__device__ __forceinline__ void glm_index_logits_wmma_impl(
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
    namespace wmma = nvcuda::wmma;
    constexpr unsigned int key_count = KeyTiles * 16;
    // A 16-element skew reduces shared-memory bank conflicts while retaining
    // WMMA's 32-byte matrix-base and 16-byte leading-dimension alignment.
    constexpr unsigned int key_stride = 128 + 16;
    __shared__ __align__(32) __nv_bfloat16 pooled_keys[key_count * key_stride];
    __shared__ __align__(32) float head_dots[8 * KeyTiles * 16 * 16];

    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        index_heads != 32 || head_dim != 128 || pool_size != 4 ||
        cache_block_size == 0 || cache_block_size % 4 != 0) return;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int pool_base = blockIdx.x * key_count;
    const unsigned int row_base = blockIdx.y * 8;
    if (row_base >= rows || pool_base >= logits_stride) return;

    // Use the latest row in this CTA, so wholly future tiles never access
    // pages beyond its causal extent. Partial row tiles still reach the one
    // block-wide barrier before their inactive warps return.
    const unsigned int tile_rows = rows - row_base < 8 ? rows - row_base : 8;
    const unsigned int max_pool_count = (seq_len_start + row_base + tile_rows) / 4;
    if (pool_base >= max_pool_count) {
        for (unsigned int item = threadIdx.x; item < tile_rows * key_count;
             item += blockDim.x) {
            const unsigned int pool_id = pool_base + item % key_count;
            if (pool_id < logits_stride) {
                logits[(unsigned long long)(row_base + item / key_count) *
                           logits_stride + pool_id] = -INFINITY;
            }
        }
        return;
    }
    for (unsigned int item = threadIdx.x; item < key_count * 128;
         item += blockDim.x) {
        const unsigned int pool = item / 128;
        const unsigned int d = item % 128;
        const unsigned int pool_id = pool_base + pool;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (pool_id < logits_stride && pool_id < max_pool_count) {
            const unsigned int raw_pos = pool_id * 4;
            const unsigned int physical_block = block_table[raw_pos / cache_block_size];
            const unsigned int cache_pool = (raw_pos % cache_block_size) / 4;
            const __nv_bfloat16* key =
                (const __nv_bfloat16*)((const char*)index_cache +
                    (unsigned long long)physical_block * index_block_stride_bytes) +
                (unsigned long long)cache_pool * 128;
            value = key[d];
        }
        pooled_keys[pool * key_stride + d] = value;
    }
    __syncthreads();

    const unsigned int row = row_base + warp;
    if (row >= rows) return;
    const unsigned int pool_count = (seq_len_start + row + 1) / 4;
    const __nv_bfloat16* q = query + (unsigned long long)row * 32 * 128;
    // Each lane holds one head weight; warp broadcasts preserve head order.
    const float head_weight = __bfloat162float(weights[(unsigned long long)row * 32 + lane]);
    float* warp_dots = head_dots + warp * KeyTiles * 256;
    const unsigned int pool = lane % key_count;
    const unsigned int dot_column = pool % 16;
    const unsigned int dot_tile_offset = (pool / 16) * 256;
    float score = 0.0f;

#pragma unroll
    for (unsigned int head_base = 0; head_base < 32; head_base += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[KeyTiles];
#pragma unroll
        for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
            wmma::fill_fragment(accum[tile], 0.0f);
        }
#pragma unroll
        for (unsigned int d = 0; d < 128; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16,
                           __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, q + head_base * 128 + d, 128);
#pragma unroll
            for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(b, pooled_keys + tile * 16 * key_stride + d,
                                       key_stride);
                wmma::mma_sync(accum[tile], a, b, accum[tile]);
            }
        }
#pragma unroll
        for (unsigned int tile = 0; tile < KeyTiles; ++tile) {
            wmma::store_matrix_sync(warp_dots + tile * 256, accum[tile], 16,
                                    wmma::mem_row_major);
        }
        __syncwarp();
        // Materialize only this warp's 16-head tile in shared memory. ReLU is
        // applied independently to each head before its weight is multiplied.
#pragma unroll
        for (unsigned int head = 0; head < 16; ++head) {
            const float weight = __shfl_sync(0xffffffff, head_weight, head_base + head);
            const float dot = warp_dots[dot_tile_offset + head * 16 + dot_column];
            score += weight * fmaxf(dot, 0.0f);
        }
        // A subsequent store must not overwrite dots still read by a peer.
        __syncwarp();
    }
    if (lane < key_count) {
        const unsigned int pool_id = pool_base + pool;
        if (pool_id < logits_stride) {
            // sqrt(128 * 32) = 64 exactly.
            logits[(unsigned long long)row * logits_stride + pool_id] =
                pool_id < pool_count ? score * (1.0f / 64.0f) : -INFINITY;
        }
    }
}

} // namespace

extern "C" __global__ void glm_index_logits_bf16_wmma_row8(
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
    glm_index_logits_wmma_impl<1>(query, weights, index_cache, logits, block_table,
        rows, seq_len_start, logits_stride, index_heads, head_dim, pool_size,
        cache_block_size, index_block_stride_bytes);
}

extern "C" __global__ void glm_index_logits_bf16_wmma_row8_pool32(
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
    glm_index_logits_wmma_impl<2>(query, weights, index_cache, logits, block_table,
        rows, seq_len_start, logits_stride, index_heads, head_dim, pool_size,
        cache_block_size, index_block_stride_bytes);
}

// glm_index_logits_bf16_mma_v2: row8_pool32 restructured for throughput, with
// bit-identical logits. Each warp owns one query row and keeps its 32x128 BF16
// query resident in registers (m16n8k16 A fragments) across the CTA's whole
// pool range. The CTA walks that range in 32-pool chunks whose keys are staged
// by 16-byte cp.async (one block-table lookup per pool, double buffered,
// XOR-swizzled) and shared by the CTA's four rows. blockIdx.x selects a
// contiguous run of chunks, so gridDim.x trades query reloads against
// parallelism. Four rows per CTA (three CTAs per SM) measured ~7% faster
// than eight rows (two CTAs per SM, register-capped) on GB10.
//
// Exactness against row8_pool32: each (row, head, pool) dot is the same chain
// of m16n8k16 BF16 MMAs (WMMA 16x16x16's lowering) over d = 0, 16, ..., 112,
// accumulated in FP32 from zero, with the same A (query heads) and B (pooled
// keys) operand roles and k positions. This module is compiled with
// --fmad=false, so `weight * relu(dot)` is a separately rounded product; only
// the FP32 head sum (heads 0..31, from 0) has an order. Products are formed in
// the MMA fragment layout and transposed through a per-warp shared tile so each
// lane sums its pool's heads in that order. The 1/64 scale and -INF causal
// masking are unchanged.
namespace {

// Mirrored by GLM_INDEX_LOGITS_V2_ROWS and V2_CHUNK_POOLS in
// crates/spark-model/src/layers/ops/glm_indexer.rs, whose tests check these.
constexpr unsigned int kIndexV2Warps = 4;
constexpr unsigned int kIndexV2Pools = 32;

__device__ __forceinline__ void index_v2_cp_async(
    unsigned int dst, const void* src, unsigned int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void index_v2_ldsm_x4(unsigned int (&r)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void index_v2_mma(
    float (&c)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Dynamic shared memory: two [Pools][128] BF16 key stages, then one
// [32 pools][32 heads] FP32 product tile per warp.
template <unsigned int Warps, unsigned int Pools>
__host__ __device__ constexpr unsigned int index_v2_smem_bytes() {
    return 2 * Pools * 256 + Warps * 32 * 32 * 4;
}

template <unsigned int Warps, unsigned int Pools>
__device__ __forceinline__ void glm_index_logits_mma_v2_impl(
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
    constexpr unsigned int threads = Warps * 32;
    constexpr unsigned int stage_bytes = Pools * 256;
    constexpr unsigned int segs = Pools * 16 / threads;  // 16-byte key segments
    constexpr unsigned int n_tiles = Pools / 8;
    static_assert(Pools % 32 == 0 && segs > 0 && Pools * 16 % threads == 0 &&
                  16 % segs == 0, "unsupported key staging geometry");
    extern __shared__ __align__(128) unsigned char index_v2_smem[];
    unsigned int dynamic_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem));

    if (blockDim.x != threads || blockDim.y != 1 || blockDim.z != 1 ||
        dynamic_smem < index_v2_smem_bytes<Warps, Pools>() ||
        index_heads != 32 || head_dim != 128 || pool_size != 4 ||
        cache_block_size == 0 || cache_block_size % 4 != 0) return;

    const unsigned int row_base = blockIdx.y * Warps;
    if (row_base >= rows) return;
    const unsigned int tile_rows = rows - row_base < Warps ? rows - row_base : Warps;
    // Keys past the tile's latest causal pool are never read; those logits are
    // -INF for every row in the tile.
    const unsigned int max_pool_count = (seq_len_start + row_base + tile_rows) / 4;
    const unsigned int valid_pools = min(logits_stride, max_pool_count);
    const unsigned int chunks = (logits_stride + Pools - 1) / Pools;
    const unsigned int per_cta = (chunks + gridDim.x - 1) / gridDim.x;
    const unsigned int chunk_begin = blockIdx.x * per_cta;
    if (chunk_begin >= chunks) return;
    const unsigned int chunk_end = min(chunk_begin + per_cta, chunks);
    const unsigned int compute_end =
        max(chunk_begin, min(chunk_end, (valid_pools + Pools - 1) / Pools));
    const unsigned int inf_end = min(chunk_end * Pools, logits_stride);
    for (unsigned int r = 0; r < tile_rows; ++r) {
        float* out = logits + (unsigned long long)(row_base + r) * logits_stride;
        for (unsigned int p = compute_end * Pools + threadIdx.x; p < inf_end; p += threads) {
            out[p] = -INFINITY;
        }
    }
    if (chunk_begin == compute_end) return;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int smem = (unsigned int)__cvta_generic_to_shared(index_v2_smem);
    const unsigned int stage_pool = threadIdx.x * segs / 16;
    const unsigned int stage_seg = threadIdx.x * segs % 16;
    auto stage_keys = [&](unsigned int chunk, unsigned int stage) {
        const unsigned int pool_id = chunk * Pools + stage_pool;
        const char* src = (const char*)index_cache;
        unsigned int bytes = 0;  // zero-fill pools that are never scored
        if (pool_id < valid_pools) {
            const unsigned int raw_pos = pool_id * 4;
            src += (unsigned long long)block_table[raw_pos / cache_block_size] *
                       index_block_stride_bytes +
                   (unsigned long long)((raw_pos % cache_block_size) / 4) * 256 +
                   stage_seg * 16;
            bytes = 16;
        }
        const unsigned int dst = smem + stage * stage_bytes + stage_pool * 256;
#pragma unroll
        for (unsigned int s = 0; s < segs; ++s) {
            index_v2_cp_async(dst + (((stage_seg + s) ^ (stage_pool & 7)) << 4),
                              src + s * 16, bytes);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    stage_keys(chunk_begin, 0);

    const unsigned int row = row_base + warp;
    const bool live = row < rows;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    unsigned int qa[2][8][4];
    float w[4];  // weights of heads g, g + 8, g + 16, g + 24
    unsigned int pool_count = 0;
    if (live) {
        const unsigned int* q =
            (const unsigned int*)(query + (unsigned long long)row * 32 * 128);
#pragma unroll
        for (unsigned int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (unsigned int ks = 0; ks < 8; ++ks) {
                const unsigned int at = (mt * 16 + g) * 64 + ks * 8 + t;
                qa[mt][ks][0] = __ldg(q + at);
                qa[mt][ks][1] = __ldg(q + at + 8 * 64);
                qa[mt][ks][2] = __ldg(q + at + 4);
                qa[mt][ks][3] = __ldg(q + at + 8 * 64 + 4);
            }
        }
#pragma unroll
        for (unsigned int i = 0; i < 4; ++i) {
            w[i] = __bfloat162float(weights[(unsigned long long)row * 32 + g + 8 * i]);
        }
        pool_count = (seq_len_start + row + 1) / 4;
    }
    float4* products = reinterpret_cast<float4*>(index_v2_smem + 2 * stage_bytes) +
                       warp * 32 * 8;
    const unsigned int ld_row = (lane & 7) + ((lane >> 4) << 3);
    const unsigned int ld_seg = (lane >> 3) & 1;

    for (unsigned int chunk = chunk_begin; chunk < compute_end; ++chunk) {
        const unsigned int stage = (chunk - chunk_begin) & 1;
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        // Publishes this chunk's keys; every warp has also finished reading the
        // other stage, which is refilled next.
        __syncthreads();
        if (chunk + 1 < compute_end) stage_keys(chunk + 1, stage ^ 1);
        if (!live) continue;

        float acc[2][n_tiles][4];
#pragma unroll
        for (unsigned int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (unsigned int nt = 0; nt < n_tiles; ++nt) {
                acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
            }
        }
        const unsigned int keys = smem + stage * stage_bytes;
#pragma unroll
        for (unsigned int ks = 0; ks < 8; ++ks) {
            unsigned int b[n_tiles / 2][4];
#pragma unroll
            for (unsigned int np = 0; np < n_tiles / 2; ++np) {
                const unsigned int pool = np * 16 + ld_row;
                index_v2_ldsm_x4(b[np], keys + pool * 256 +
                                            (((ks * 2 + ld_seg) ^ (pool & 7)) << 4));
            }
#pragma unroll
            for (unsigned int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (unsigned int nt = 0; nt < n_tiles; ++nt) {
                    index_v2_mma(acc[mt][nt], qa[mt][ks], b[nt / 2][(nt & 1) * 2],
                                 b[nt / 2][(nt & 1) * 2 + 1]);
                }
            }
        }
#pragma unroll
        for (unsigned int half = 0; half < Pools / 32; ++half) {
            // Accumulator (mt, nt, j) is head mt*16 + g + 8*(j/2), pool
            // nt*8 + 2t + j%2. Pool p's heads land at float4 (p, g ^ p%8) as
            // (g, g + 8, g + 16, g + 24): conflict-free stores and loads.
#pragma unroll
            for (unsigned int ntl = 0; ntl < 4; ++ntl) {
#pragma unroll
                for (unsigned int j = 0; j < 2; ++j) {
                    const unsigned int nt = half * 4 + ntl;
                    const unsigned int p = ntl * 8 + 2 * t + j;
                    products[p * 8 + (g ^ (p & 7))] = make_float4(
                        w[0] * fmaxf(acc[0][nt][j], 0.0f),
                        w[1] * fmaxf(acc[0][nt][j + 2], 0.0f),
                        w[2] * fmaxf(acc[1][nt][j], 0.0f),
                        w[3] * fmaxf(acc[1][nt][j + 2], 0.0f));
                }
            }
            __syncwarp();
            float4 heads[8];
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                heads[i] = products[lane * 8 + (i ^ (lane & 7))];
            }
            float score = 0.0f;
#pragma unroll
            for (unsigned int head = 0; head < 32; ++head) {
                const float4 h = heads[head & 7];
                const unsigned int part = (head >> 4) * 2 + ((head >> 3) & 1);
                score += part == 0 ? h.x : part == 1 ? h.y : part == 2 ? h.z : h.w;
            }
            const unsigned int pool_id = chunk * Pools + half * 32 + lane;
            if (pool_id < logits_stride) {
                logits[(unsigned long long)row * logits_stride + pool_id] =
                    pool_id < pool_count ? score * (1.0f / 64.0f) : -INFINITY;
            }
            __syncwarp();
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(kIndexV2Warps * 32, 3)
glm_index_logits_bf16_mma_v2(
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
    glm_index_logits_mma_v2_impl<kIndexV2Warps, kIndexV2Pools>(query, weights,
        index_cache, logits, block_table, rows, seq_len_start, logits_stride,
        index_heads, head_dim, pool_size, cache_block_size, index_block_stride_bytes);
}
