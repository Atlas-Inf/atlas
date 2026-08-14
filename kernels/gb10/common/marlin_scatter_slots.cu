// SPDX-License-Identifier: Apache-2.0
// Scatter slot C[slot,row,H] to expert_down_out[flat_i, H].

#include <cuda_bf16.h>
#include <cstdint>

#ifndef MAX_SLOTS
#define MAX_SLOTS 64
#endif
#ifndef M_TILE
#define M_TILE 8
#endif

extern "C" __global__ void atlas_marlin_scatter_slots(
    const __nv_bfloat16* __restrict__ slot_c, const int32_t* __restrict__ slot_map,
    __nv_bfloat16* __restrict__ out, int hidden_size) {
  int s = blockIdx.x;
  int row = blockIdx.y;
  if (s >= MAX_SLOTS || row >= M_TILE) return;
  int flat = slot_map[s * M_TILE + row];
  if (flat < 0) return;
  const __nv_bfloat16* src = slot_c + ((size_t)s * M_TILE + row) * hidden_size;
  __nv_bfloat16* dst = out + (size_t)flat * hidden_size;
  for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) dst[c] = src[c];
}
