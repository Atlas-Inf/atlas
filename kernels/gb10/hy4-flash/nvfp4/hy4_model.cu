// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "hy4_gemv.cuh"

extern "C" __global__ void hy4_model_embed(
    const __nv_bfloat16* table,__nv_bfloat16* out,unsigned token,unsigned h) {
    unsigned d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<4*h)out[d]=table[(size_t)token*h+d%h];
}
extern "C" __global__ void hy4_model_norm(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,__nv_bfloat16* out,unsigned h,float eps) {
    __shared__ float values[256];float ss=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {float v=(float)x[d];ss+=v*v;}
    values[threadIdx.x]=ss;__syncthreads();
    for(int n=128;n;n>>=1) {
        if(threadIdx.x<n)values[threadIdx.x]+=values[threadIdx.x+n];__syncthreads();
    }
    float inv=rsqrtf(values[0]/h+eps);
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x)
        out[d]=__float2bfloat16_rn((float)x[d]*inv*(float)weight[d]);
}
extern "C" __global__ void hy4_model_head(
    const __nv_bfloat16* x,const void* weight,float* logits,unsigned vocab,unsigned h,unsigned weight_f32) {
    unsigned n=blockIdx.x*8+threadIdx.x/32;if(n>=vocab)return;
    float sum=hy4_dense_dot(x,weight,n,h,weight_f32);
    if(threadIdx.x%32==0)logits[n]=sum;
}
extern "C" __global__ void hy4_model_argmax(const float* logits,unsigned* out,unsigned vocab) {
    if(threadIdx.x || blockIdx.x)return;
    unsigned best=0;for(unsigned i=1;i<vocab;i++)if(logits[i]>logits[best])best=i;
    *out=best;
}
