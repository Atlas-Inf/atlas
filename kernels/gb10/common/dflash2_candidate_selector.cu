// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 on-device bilinear candidate selector.
//
// Eliminates the 4MB D2H sync and 1.74M CPU float loop by:
// 1. Collaborative parallel top-4 reduction per row directly in GPU shared memory.
// 2. On-device context vector & bilinear scoring across rank=64 codebooks:
//      Score(c) = Unary(c) + sum_{r=0}^{rank-1} (pred[prev, r] * H_proj[row, r]) * succ[c, r]
// 3. Greedy chain advancement entirely within registers and shared memory.
// 4. Writing all gamma draft tokens directly into device memory.
//
// Grid: (1, 1, 1)   Block: (1024, 1, 1)

__device__ __forceinline__ void insert_top4(
    float* __restrict__ vals,
    unsigned int* __restrict__ idxs,
    float v,
    unsigned int idx
) {
    if (v < vals[3] || (v == vals[3] && idx <= idxs[3])) return;
    int pos = 2;
    while (pos >= 0 && (v > vals[pos] || (v == vals[pos] && idx > idxs[pos]))) {
        vals[pos + 1] = vals[pos];
        idxs[pos + 1] = idxs[pos];
        pos--;
    }
    vals[pos + 1] = v;
    idxs[pos + 1] = idx;
}

__device__ __forceinline__ void merge_top4(
    const float* __restrict__ a_v,
    const unsigned int* __restrict__ a_i,
    const float* __restrict__ b_v,
    const unsigned int* __restrict__ b_i,
    float* __restrict__ out_v,
    unsigned int* __restrict__ out_i
) {
    int i = 0, j = 0;
    for (int k = 0; k < 4; k++) {
        bool take_a = false;
        if (i < 4 && j < 4) {
            take_a = (a_v[i] > b_v[j]) || (a_v[i] == b_v[j] && a_i[i] > b_i[j]);
        } else if (i < 4) {
            take_a = true;
        }
        if (take_a) {
            out_v[k] = a_v[i];
            out_i[k] = a_i[i];
            i++;
        } else {
            out_v[k] = b_v[j];
            out_i[k] = b_i[j];
            j++;
        }
    }
}

extern "C" __global__ void dflash2_candidate_selector(
    const __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ projected_hidden,
    const __nv_bfloat16* __restrict__ pred_codebook,
    const __nv_bfloat16* __restrict__ succ_codebook,
    unsigned int* __restrict__ out_tokens,
    unsigned int last_token,
    unsigned int gamma,
    unsigned int vocab_size,
    unsigned int rank
) {
    __shared__ float s_warp_v[32][4];
    __shared__ unsigned int s_warp_i[32][4];
    __shared__ float s_final_v[4];
    __shared__ unsigned int s_final_i[4];
    __shared__ float s_context[64];
    __shared__ unsigned int s_prev_token;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp_id = tid / 32;

    if (tid == 0) {
        s_prev_token = last_token;
    }
    __syncthreads();

    // Iterate through all gamma rows.
    // Row 0: anchor token (unary argmax).
    // Rows 1..gamma-1: mask draft tokens conditioned on predecessor chain.
    for (unsigned int row = 0; row < gamma; row++) {
        const __nv_bfloat16* row_logits = logits + (unsigned long long)row * (unsigned long long)vocab_size;

        // Phase 1: Local thread top-4
        float local_v[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
        unsigned int local_i[4] = {0, 0, 0, 0};

        for (unsigned int i = tid; i < vocab_size; i += blockDim.x) {
            float v = __bfloat162float(row_logits[i]);
            insert_top4(local_v, local_i, v, i);
        }

        // Phase 2: Warp-level top-4 reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_v[4];
            unsigned int other_i[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                other_v[k] = __shfl_down_sync(0xffffffff, local_v[k], offset);
                other_i[k] = __shfl_down_sync(0xffffffff, local_i[k], offset);
            }
            float merged_v[4];
            unsigned int merged_i[4];
            merge_top4(local_v, local_i, other_v, other_i, merged_v, merged_i);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                local_v[k] = merged_v[k];
                local_i[k] = merged_i[k];
            }
        }

        // Leader of each warp writes to shared memory
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s_warp_v[warp_id][k] = local_v[k];
                s_warp_i[warp_id][k] = local_i[k];
            }
        }
        __syncthreads();

        // Phase 3: Warp 0 reduces across all 32 warps
        if (warp_id == 0) {
            float warp0_v[4];
            unsigned int warp0_i[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                warp0_v[k] = s_warp_v[lane][k];
                warp0_i[k] = s_warp_i[lane][k];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_v[4];
                unsigned int other_i[4];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    other_v[k] = __shfl_down_sync(0xffffffff, warp0_v[k], offset);
                    other_i[k] = __shfl_down_sync(0xffffffff, warp0_i[k], offset);
                }
                float merged_v[4];
                unsigned int merged_i[4];
                merge_top4(warp0_v, warp0_i, other_v, other_i, merged_v, merged_i);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    warp0_v[k] = merged_v[k];
                    warp0_i[k] = merged_i[k];
                }
            }

            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s_final_v[k] = warp0_v[k];
                    s_final_i[k] = warp0_i[k];
                }
            }
        }
        __syncthreads();

        if (row == 0) {
            // Anchor row: top-1 unary argmax
            if (tid == 0) {
                out_tokens[0] = s_final_i[0];
            }
            __syncthreads();
            continue;
        }

        // Mask row > 0: evaluate context vector and bilinear candidate scoring
        if (tid < rank && tid < 64) {
            unsigned int prev = s_prev_token;
            if (prev >= vocab_size) prev = 0;
            float pred_val = __bfloat162float(pred_codebook[(unsigned long long)prev * (unsigned long long)rank + tid]);
            float h_val = __bfloat162float(projected_hidden[(unsigned long long)row * (unsigned long long)rank + tid]);
            s_context[tid] = pred_val * h_val;
        }
        __syncthreads();

        // Score all 4 candidates in parallel
        #pragma unroll
        for (int c_idx = 0; c_idx < 4; c_idx++) {
            unsigned int cand_id = s_final_i[c_idx];
            if (cand_id >= vocab_size) cand_id = 0;

            float dot_part = 0.0f;
            if (tid < rank && tid < 64) {
                float succ_val = __bfloat162float(succ_codebook[(unsigned long long)cand_id * (unsigned long long)rank + tid]);
                dot_part = s_context[tid] * succ_val;
            }

            // Warp reduction for dot product across first 64 threads (2 warps)
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot_part += __shfl_down_sync(0xffffffff, dot_part, offset);
            }

            if (lane == 0 && tid < 64) {
                s_warp_v[warp_id][c_idx] = dot_part;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best_score = -1e30f;
            unsigned int best_cand = s_final_i[0];

            for (int c_idx = 0; c_idx < 4; c_idx++) {
                float total_dot = s_warp_v[0][c_idx] + s_warp_v[1][c_idx];
                float total_score = s_final_v[c_idx] + total_dot;
                if (total_score > best_score) {
                    best_score = total_score;
                    best_cand = s_final_i[c_idx];
                }
            }

            out_tokens[row] = best_cand;
            s_prev_token = best_cand;
        }
        __syncthreads();
    }
}
