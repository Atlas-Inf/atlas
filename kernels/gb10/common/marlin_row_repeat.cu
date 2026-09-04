// SPDX-License-Identifier: AGPL-3.0-only
// Repeat each row `reps` times: out[t*reps+s, :] = in[t, :].
#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void atlas_row_repeat_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    int n, int k, int reps) {
  int row = blockIdx.x;
  if (row >= n) return;
  const __nv_bfloat16* src = in + (size_t)row * k;
  for (int s = 0; s < reps; s++) {
    __nv_bfloat16* dst = out + ((size_t)row * reps + s) * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x) dst[c] = src[c];
  }
}
