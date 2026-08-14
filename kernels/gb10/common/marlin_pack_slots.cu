// SPDX-License-Identifier: Apache-2.0
// Compact unique experts into fixed slots and gather A[slot, 8, K].
// Overflow rows for the same expert open another slot (C=4).

#include <cuda_bf16.h>
#include <cstdint>

#ifndef MAX_SLOTS
#define MAX_SLOTS 64
#endif
#ifndef M_TILE
#define M_TILE 8
#endif

extern "C" __global__ void atlas_marlin_pack_slots(
    const int32_t* __restrict__ topk_ids, const __nv_bfloat16* __restrict__ hidden,
    int32_t* __restrict__ slot_eids, int32_t* __restrict__ slot_map,
    __nv_bfloat16* __restrict__ slot_a, int32_t* __restrict__ n_slots,
    int tokens, int top_k, int num_experts, int hidden_size) {
  __shared__ int counts[128];
  __shared__ int slot_of[128];
  __shared__ int filled[MAX_SLOTS];
  int E = num_experts;
  if (E > 128) E = 128;
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    counts[e] = 0;
    slot_of[e] = -1;
  }
  for (int s = threadIdx.x; s < MAX_SLOTS; s += blockDim.x) {
    filled[s] = 0;
    slot_eids[s] = -1;
  }
  for (int s = threadIdx.x; s < MAX_SLOTS * M_TILE; s += blockDim.x) slot_map[s] = -1;
  __syncthreads();
  int n = tokens * top_k;
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; i++) {
      int e = topk_ids[i];
      if (e >= 0 && e < E) counts[e]++;
    }
    int ns = 0;
    for (int e = 0; e < E; e++) {
      if (counts[e] <= 0 || ns >= MAX_SLOTS) continue;
      slot_of[e] = ns;
      slot_eids[ns] = e;
      ns++;
    }
    for (int i = 0; i < n; i++) {
      int e = topk_ids[i];
      if (e < 0 || e >= E) continue;
      int s = slot_of[e];
      if (s < 0) continue;
      int row = filled[s];
      if (row >= M_TILE) {
        if (ns >= MAX_SLOTS) continue;
        slot_of[e] = ns;
        slot_eids[ns] = e;
        filled[ns] = 0;
        s = ns;
        ns++;
        row = 0;
      }
      filled[s] = row + 1;
      slot_map[s * M_TILE + row] = i;
    }
    *n_slots = ns;
  }
  __syncthreads();
  for (int s = 0; s < MAX_SLOTS; s++) {
    for (int row = 0; row < M_TILE; row++) {
      int flat = slot_map[s * M_TILE + row];
      if (flat < 0) continue;
      int tok = flat / top_k;
      const __nv_bfloat16* src = hidden + (size_t)tok * hidden_size;
      __nv_bfloat16* dst = slot_a + ((size_t)s * M_TILE + row) * hidden_size;
      for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) dst[c] = src[c];
    }
  }
}
