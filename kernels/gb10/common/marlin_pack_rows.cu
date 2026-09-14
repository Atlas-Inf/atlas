// SPDX-License-Identifier: AGPL-3.0-only
// Gather sorted token rows into a contiguous Marlin A staging buffer.
// One block per sorted row: dst[row] = src[sorted_token_ids[row]].
// Plain row copy — the m8 Marlin template reads BF16 A row-major.

#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void atlas_marlin_pack_rows(
    const int32_t* __restrict__ sorted_token_ids, // [te] token index per sorted pos
    const __nv_bfloat16* __restrict__ src,        // [tokens, hidden_size]
    __nv_bfloat16* __restrict__ dst,              // [te, hidden_size]
    int te, int hidden_size) {
  int row = blockIdx.x;
  if (row >= te) return;
  int tok = sorted_token_ids[row];
  if (tok < 0) return; // sentinel — leave zeros
  const __nv_bfloat16* s = src + (size_t)tok * hidden_size;
  __nv_bfloat16* d = dst + (size_t)row * hidden_size;
  for (int c = threadIdx.x; c < hidden_size; c += blockDim.x) d[c] = s[c];
}
