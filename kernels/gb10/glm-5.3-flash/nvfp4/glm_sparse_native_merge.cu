// SPDX-License-Identifier: AGPL-3.0-only
// Origin: Atlas 630d8856:scripts/dev/glm_sparse_native/merge-ds.cu.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void glm_sparse_native_merge(const __nv_bfloat16* __restrict__ main_out,
    const __nv_bfloat16* __restrict__ tail_out, const float* __restrict__ main_lse,
    const float* __restrict__ tail_lse, const int* __restrict__ tail_counts,
    __nv_bfloat16* __restrict__ output, float* __restrict__ output_lse,
    unsigned rows, unsigned heads){
  unsigned row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
  __shared__ float sw0, sw1;
  int tc = tail_counts[row];
  if(tid == 0){
    float l0 = main_lse[(size_t)row*heads + head];
    if(tc == 0){ sw0 = 1.0f; sw1 = 0.0f; output_lse[(size_t)row*heads + head] = l0; }
    else{
      float l1 = tail_lse[(size_t)row*heads + head];
      float m = fmaxf(l0, l1);
      float a = exp2f(l0 - m), b = exp2f(l1 - m);
      sw0 = a/(a+b); sw1 = b/(a+b);
      output_lse[(size_t)row*heads + head] = m + log2f(a+b);
    }
  }
  __syncthreads();
  size_t base = ((size_t)row*heads + head)*512;
  __nv_bfloat16* o = output + base;
  const __nv_bfloat16* mo = main_out + base;
  const __nv_bfloat16* to = tail_out + base;
  #pragma unroll
  for(int k = 0; k < 2; ++k){
    unsigned i = tid + k*256;
    if(tc == 0){ o[i] = mo[i]; }
    else{
      float mv = __bfloat162float(mo[i]);
      float tv = __bfloat162float(to[i]);
      o[i] = __float2bfloat16_rn(fmaf(mv, sw0, tv*sw1));
    }
  }
}
