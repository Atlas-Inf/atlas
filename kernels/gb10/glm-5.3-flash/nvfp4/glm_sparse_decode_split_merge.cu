// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <math.h>
#include <cstddef>
#include <cstdint>

__device__ __forceinline__ void glm_split_merge_store(__nv_bfloat16* out, size_t i, float v) {
    out[i] = __float2bfloat16_rn(v);
}
__device__ __forceinline__ void glm_split_merge_store(float* out, size_t i, float v) {
    out[i] = v;
}

// Merge partitioned attention outputs: weighted combine of partial outputs by LSE.
// part_o   : float[splits, rows, heads, dim] normalized partial attention outputs
// part_lse : float[splits, rows, heads] natural-logsumexp (-INFINITY == empty partition)
// out      : OutT[rows, heads, dim] (BF16, or FP32 for a further exact merge)
// out_lse  : float[rows, heads]
// EXTRA    : one more partition (index `local_splits`) held at `extra`:
//            float[rows, heads, dim] then float[rows, heads] LSEs. Compile-time,
//            so that without it the body is the kernel it was factored from,
//            access for access.
// The entry point declares the shared state (GLM_SPLIT_MERGE_SHARED) and hands
// it in, so its symbols are named after the entry point as they always were.
#define GLM_SPLIT_MERGE_SHARED \
    __shared__ float s_w[16]; \
    __shared__ unsigned s_nempty; \
    __shared__ float s_bad   /* nonfinite LSE flag */
template <typename OutT, bool EXTRA>
__device__ __forceinline__ void glm_split_merge_body(float* s_w, unsigned& s_nempty, float& s_bad,
                                                     const float* __restrict__ part_o,
                                                     const float* __restrict__ part_lse,
                                                     OutT* __restrict__ out_bf16,
                                                     float* __restrict__ out_lse,
                                                     unsigned rows, unsigned heads,
                                                     unsigned dim, unsigned local_splits,
                                                     const float* __restrict__ extra) {
    unsigned rh = blockIdx.x;                 // flattened row*heads + head
    unsigned row = rh / heads, head = rh % heads;
    size_t base_o = (size_t)row * heads * dim + (size_t)head * dim;
    size_t stride_o = (size_t)rows * heads * dim;
    const unsigned splits = local_splits + (EXTRA ? 1u : 0u);
    // Partition `s`'s LSE and element `d` of its output for this (row, head).
    auto lse_of = [&](unsigned s) {
        if constexpr (EXTRA) {
            if (s >= local_splits) return extra[stride_o + rh];
        }
        return part_lse[(size_t)s * rows * heads + rh];
    };
    auto o_of = [&](unsigned s, unsigned d) {
        if constexpr (EXTRA) {
            if (s >= local_splits) return extra[base_o + d];
        }
        return part_o[(size_t)s * stride_o + base_o + d];
    };

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        unsigned n = 0;
        float bad = 0.0f;
        float sum = 0.0f;
        for (unsigned s = 0; s < splits; ++s) {
            float l = lse_of(s);
            if (isnan(l) || l == INFINITY) bad = 1.0f;   // propagate nonfinite
            if (l != -INFINITY) { if (l > mx) mx = l; ++n; }
        }
        s_nempty = n;
        s_bad = bad;
        if (bad != 0.0f || n == 0) {
            s_w[0] = bad != 0.0f ? 1.0f : 0.0f;   // weight for nonfinite path
        } else {
            for (unsigned s = 0; s < splits; ++s) {
                float l = lse_of(s);
                if (l == -INFINITY) { s_w[s] = 0.0f; continue; }
                float e = expf(l - mx);
                s_w[s] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (unsigned s = 0; s < splits; ++s) s_w[s] *= inv;
        }
        out_lse[(size_t)row * heads + head] =
            (bad != 0.0f) ? CUDART_NAN_F
                          : (n == 0 ? -INFINITY : mx + logf(sum));
    }
    __syncthreads();

    bool nonfinite = (s_bad != 0.0f);
    bool empty = (s_nempty == 0);
    bool single = (s_nempty == 1);

    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc;
        if (nonfinite) {
            acc = CUDART_NAN_F;
        } else if (empty) {
            acc = 0.0f;
        } else if (single) {
            // weight is 1.0; copy active partition's normalized FP32 output
            unsigned s = 0;
            while (s < splits && lse_of(s) == -INFINITY) ++s;
            acc = o_of(s, d);
        } else {
            acc = 0.0f;
            for (unsigned s = 0; s < splits; ++s) {
                if (s_w[s] == 0.0f && lse_of(s) == -INFINITY)
                    continue;
                acc += s_w[s] * o_of(s, d);
            }
        }
        glm_split_merge_store(out_bf16, (size_t)row * heads * dim + (size_t)head * dim + d, acc);
    }
}

// glm_kv_shard.cu includes this file for the body above and instantiates the
// FP32 and extra-partition merges; this module keeps the one it always had.
#ifndef GLM_KV_SHARD_MODULE
extern "C" __global__ void glm_sparse_decode_split_merge(const float* __restrict__ part_o,
                                             const float* __restrict__ part_lse,
                                             __nv_bfloat16* __restrict__ out_bf16,
                                             float* __restrict__ out_lse,
                                             unsigned rows, unsigned heads,
                                             unsigned dim, unsigned splits) {
    // Init-only zero rows: uniform return before any pointer access.
    if (rows == 0) return;
    GLM_SPLIT_MERGE_SHARED;
    glm_split_merge_body<__nv_bfloat16, false>(s_w, s_nempty, s_bad, part_o, part_lse, out_bf16,
                                               out_lse, rows, heads, dim, splits, nullptr);
}
#endif  // GLM_KV_SHARD_MODULE
