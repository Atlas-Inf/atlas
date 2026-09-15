// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// query_rows: [B, gamma, hidden], projected: [B, hidden].
// Adds projected[b] only to anchor/query row 0 for each sequence.
extern "C" __global__ void dflash_batch_anchor_add(
    __nv_bfloat16* __restrict__ query_rows,
    const __nv_bfloat16* __restrict__ projected,
    unsigned int batch,
    unsigned int gamma,
    unsigned int hidden
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = batch * hidden;
    if (index >= total) return;
    const unsigned int sequence = index / hidden;
    const unsigned int column = index - sequence * hidden;
    const unsigned long long query_index =
        (unsigned long long)sequence * gamma * hidden + column;
    const float base = __bfloat162float(query_rows[query_index]);
    const float delta = __bfloat162float(projected[index]);
    query_rows[query_index] = __float2bfloat16(base + delta);
}
