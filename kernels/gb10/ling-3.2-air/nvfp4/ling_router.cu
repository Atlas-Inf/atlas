// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <math.h>

// One block per token, 256 threads (one warp per group). The checkpoint
// routes on bias-corrected probabilities but mixes with the original scores.
extern "C" __global__ void ling_grouped_top8(const float* logits, const float* bias,
                                              int* ids, float* weights) {
    __shared__ float corrected[512], probabilities[512], group_scores[8];
    __shared__ int selected_groups[8];
    const int tid = threadIdx.x, lane = tid % 32, group = tid / 32;
    const int first = group * 64 + lane;
    for (int i = first; i < first + 64; i += 32) {
        const float p = 1.f / (1.f + expf(-logits[blockIdx.x * 512 + i]));
        probabilities[i] = p;
        corrected[i] = p + bias[i];
    }
    float best = fmaxf(corrected[first], corrected[first + 32]);
    float second = fminf(corrected[first], corrected[first + 32]);
    for (int stride = 16; stride; stride /= 2) {
        const float a = __shfl_down_sync(0xffffffff, best, stride);
        const float b = __shfl_down_sync(0xffffffff, second, stride);
        second = fmaxf(fminf(best, a), fmaxf(second, b));
        best = fmaxf(best, a);
    }
    if (!lane) group_scores[group] = best + second;
    __syncthreads();
    if (tid < 8) {
        int rank = 0;
        for (int j = 0; j < 8; ++j)
            rank += group_scores[j] > group_scores[tid] ||
                    (group_scores[j] == group_scores[tid] && j < tid);
        selected_groups[tid] = rank < 4;
    }
    __syncthreads();
    // Each warp supplies its best remaining expert. Warp zero merges the
    // eight candidates. Selected experts are masked for the next iteration.
    __shared__ int winners[8], used;
    const int row = blockIdx.x * 8;
    float sum = 0;
    for (int k = 0; k < 8; ++k) {
        int idx = corrected[first] >= corrected[first+32] ? first : first+32;
        float value = selected_groups[group] ? corrected[idx] : -INFINITY;
        for (int stride = 16; stride; stride /= 2) {
            float other = __shfl_down_sync(0xffffffff, value, stride);
            int oi = __shfl_down_sync(0xffffffff, idx, stride);
            if (other > value || (value == other && oi < idx)) { idx = oi; value = other; }
        }
        if (!lane) winners[group] = idx;
        __syncthreads();
        if (!tid) {
            int result = -1;
            for (int g = 0; g < 8; ++g) {
                int i = winners[g];
                if (selected_groups[g] && (result < 0 || corrected[i] > corrected[result] ||
                    (corrected[i] == corrected[result] && i < result))) result = i;
            }
            used = result;
            ids[row+k] = result;
            sum += probabilities[result];
        }
        __syncthreads();
        if (tid == 0) corrected[used] = -INFINITY;
        __syncthreads();
    }
    if (!tid)
        for (int k = 0; k < 8; ++k)
            weights[row+k] = (probabilities[ids[row+k]] / (sum + 1e-20f)) * 2.5f;
}
