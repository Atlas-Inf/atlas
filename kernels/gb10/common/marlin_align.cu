// SPDX-License-Identifier: Apache-2.0
// Small-batch Marlin moe_align (n*top_k <= 256, E<=128, block=8).
// Device-only so it can sit inside a CUDA graph.

#include <cstdint>

#ifndef BLOCK
#define BLOCK 8
#endif

extern "C" __global__ void atlas_marlin_align_block8(
    const int32_t* __restrict__ topk_ids, // [tokens * top_k]
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_pad,
    int tokens, int top_k, int num_experts, int sorted_cap) {
  // single CTA
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int counts[128];
  int offsets[129];
  int E = num_experts;
  if (E > 128) return;
  for (int e = 0; e < E; e++) counts[e] = 0;
  int n = tokens * top_k;
  int sentinel = n; // pads must not alias a live slot (0..n-1)
  for (int i = 0; i < n; i++) {
    int e = topk_ids[i];
    if (e >= 0 && e < E) counts[e]++;
  }
  int cursor = 0;
  int n_blocks = 0;
  for (int e = 0; e < E; e++) {
    offsets[e] = cursor;
    int c = counts[e];
    int padded = (c + BLOCK - 1) / BLOCK * BLOCK;
    if (padded == 0) continue;
    for (int b = 0; b < padded / BLOCK; b++) {
      if (n_blocks < sorted_cap / BLOCK) expert_ids[n_blocks] = e;
      n_blocks++;
    }
    cursor += padded;
  }
  offsets[E] = cursor;
  if (cursor > sorted_cap) cursor = sorted_cap;
  for (int i = 0; i < cursor; i++) sorted_token_ids[i] = sentinel;
  int filled[128];
  for (int e = 0; e < E; e++) filled[e] = 0;
  for (int i = 0; i < n; i++) {
    int e = topk_ids[i];
    if (e < 0 || e >= E) continue;
    int dst = offsets[e] + filled[e];
    if (dst < cursor) sorted_token_ids[dst] = i;
    filled[e]++;
  }
  *num_tokens_post_pad = cursor;
}
