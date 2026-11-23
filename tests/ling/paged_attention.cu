// SPDX-License-Identifier: AGPL-3.0-only
// CPU parity for Ling's dedicated 192-wide BF16 paged decoder.
//
// Compile on a GB10 with the same architecture flag as the target, for
// example:
//   nvcc -std=c++17 -O2 -arch=sm_121 tests/ling/paged_attention.cu \
//     -o /tmp/ling-paged-attention && /tmp/ling-paged-attention
//
// The three rows exercise causal lengths 1, 17 and 37.  The latter two cross
// block boundaries and every row uses a shuffled physical block table.  Seven
// complete cache layers are allocated and the kernel reads layer 6, matching
// Ling's last compact MLA pool (transformer layers 5, 11, ..., 41).

#include "../../kernels/gb10/ling-3.2-air/nvfp4/paged_decode_attn_ling.cu"
#undef HDIM
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using Bf16 = __nv_bfloat16;

#define CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        return 1; \
    } \
} while (0)

static float draw(uint64_t& state, float scale) {
    state += 0x9E37'79B9'7F4A'7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBull;
    z ^= z >> 31;
    const float unit = static_cast<float>(z >> 40) / static_cast<float>(1u << 24);
    return (unit * 2.0f - 1.0f) * scale;
}

static int cache_offset(
    int physical_block,
    int block_offset,
    int head,
    int dim,
    int block_size,
    int num_heads,
    int head_dim
) {
    return (((physical_block * block_size + block_offset) * num_heads + head) * head_dim) + dim;
}

int main() {
    constexpr int CACHE_LAYERS = 7;
    constexpr int CACHE_LAYER = 6;
    constexpr int BLOCKS = 3;
    constexpr int BLOCK_SIZE = 16;
    constexpr int HEADS = 32;
    constexpr int HDIM = 192;
    constexpr int ROWS = 3;
    constexpr int MAX_BLOCKS = 3;
    const int lengths[ROWS] = {1, 17, 37};
    const int tables[ROWS][MAX_BLOCKS] = {
        {2, 0, 1},
        {1, 2, 0},
        {0, 2, 1},
    };

    const size_t cache_elems = static_cast<size_t>(BLOCKS) * BLOCK_SIZE * HEADS * HDIM;
    std::vector<Bf16*> k_pool(CACHE_LAYERS), v_pool(CACHE_LAYERS);
    for (int layer = 0; layer < CACHE_LAYERS; ++layer) {
        CHECK(cudaMallocManaged(&k_pool[layer], cache_elems * sizeof(Bf16)));
        CHECK(cudaMallocManaged(&v_pool[layer], cache_elems * sizeof(Bf16)));
        std::fill_n(k_pool[layer], cache_elems, Bf16(0.0f));
        std::fill_n(v_pool[layer], cache_elems, Bf16(0.0f));
    }
    Bf16* k_cache = k_pool[CACHE_LAYER];
    Bf16* v_cache = v_pool[CACHE_LAYER];

    uint64_t state = 0x1f2e'3d4c'5b6a'7988ull;
    for (int physical = 0; physical < BLOCKS; ++physical) {
        for (int offset = 0; offset < BLOCK_SIZE; ++offset) {
            for (int head = 0; head < HEADS; ++head) {
                for (int dim = 0; dim < HDIM; ++dim) {
                    const int i = cache_offset(
                        physical, offset, head, dim, BLOCK_SIZE, HEADS, HDIM);
                    k_cache[i] = Bf16(draw(state, 0.35f));
                    v_cache[i] = Bf16(draw(state, 0.45f));
                }
            }
        }
    }

    const size_t q_elems = static_cast<size_t>(ROWS) * HEADS * HDIM;
    Bf16* q = nullptr;
    Bf16* output = nullptr;
    CHECK(cudaMallocManaged(&q, q_elems * sizeof(Bf16)));
    CHECK(cudaMallocManaged(&output, q_elems * sizeof(Bf16)));
    for (size_t i = 0; i < q_elems; ++i) q[i] = Bf16(draw(state, 0.40f));
    std::fill_n(output, q_elems, Bf16(0.0f));

    int* block_tables = nullptr;
    int* seq_lens = nullptr;
    CHECK(cudaMallocManaged(&block_tables, ROWS * MAX_BLOCKS * sizeof(int)));
    CHECK(cudaMallocManaged(&seq_lens, ROWS * sizeof(int)));
    for (int row = 0; row < ROWS; ++row) {
        seq_lens[row] = lengths[row];
        for (int block = 0; block < MAX_BLOCKS; ++block)
            block_tables[row * MAX_BLOCKS + block] = tables[row][block];
    }

    paged_decode_attn_ling<<<dim3(HEADS, ROWS, 1), dim3(256, 1, 1)>>>(
        q,
        k_cache,
        v_cache,
        output,
        block_tables,
        seq_lens,
        MAX_BLOCKS,
        HEADS,
        HEADS,
        HDIM,
        BLOCK_SIZE,
        1.0f / std::sqrt(static_cast<float>(HDIM)),
        HEADS * HDIM,
        0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float worst = 0.0f;
    size_t worst_index = 0;
    for (int row = 0; row < ROWS; ++row) {
        for (int head = 0; head < HEADS; ++head) {
            std::vector<float> scores(lengths[row]);
            float max_score = -INFINITY;
            for (int pos = 0; pos < lengths[row]; ++pos) {
                const int physical = tables[row][pos / BLOCK_SIZE];
                const int offset = pos % BLOCK_SIZE;
                float dot = 0.0f;
                for (int dim = 0; dim < HDIM; ++dim) {
                    const size_t qi = (static_cast<size_t>(row) * HEADS + head) * HDIM + dim;
                    const int ki = cache_offset(
                        physical, offset, head, dim, BLOCK_SIZE, HEADS, HDIM);
                    dot += static_cast<float>(q[qi]) * static_cast<float>(k_cache[ki]);
                }
                scores[pos] = dot / std::sqrt(static_cast<float>(HDIM));
                max_score = std::max(max_score, scores[pos]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            for (int dim = 0; dim < HDIM; ++dim) {
                float expected = 0.0f;
                for (int pos = 0; pos < lengths[row]; ++pos) {
                    const int physical = tables[row][pos / BLOCK_SIZE];
                    const int offset = pos % BLOCK_SIZE;
                    const int vi = cache_offset(
                        physical, offset, head, dim, BLOCK_SIZE, HEADS, HDIM);
                    expected += (scores[pos] / denominator) * static_cast<float>(v_cache[vi]);
                }
                const size_t oi = (static_cast<size_t>(row) * HEADS + head) * HDIM + dim;
                const float error = std::abs(static_cast<float>(output[oi]) - expected);
                if (error > worst) {
                    worst = error;
                    worst_index = oi;
                }
            }
        }
    }

    for (int layer = 0; layer < CACHE_LAYERS; ++layer) {
        CHECK(cudaFree(k_pool[layer]));
        CHECK(cudaFree(v_pool[layer]));
    }
    CHECK(cudaFree(q));
    CHECK(cudaFree(output));
    CHECK(cudaFree(block_tables));
    CHECK(cudaFree(seq_lens));

    if (worst > 0.02f) {
        std::fprintf(stderr, "192-wide paged attention mismatch: worst=%g at %zu\n", worst, worst_index);
        return 1;
    }
    std::printf("Ling paged attention: 192-wide, 7th pool, shuffled 1/17/37-token rows; max error=%g\n", worst);
    return 0;
}
