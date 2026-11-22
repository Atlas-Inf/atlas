// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "hy4_gemv.cuh"

extern "C" __global__ void hy4_head_bf16(const void* input,__nv_bfloat16* output,
    unsigned count,unsigned f32) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) output[i]=f32?__float2bfloat16_rn(((const float*)input)[i]):
        ((const __nv_bfloat16*)input)[i];
}

extern "C" __global__ void hy4_model_embed(
    const __nv_bfloat16* table,__nv_bfloat16* out,unsigned token,unsigned h) {
    unsigned d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<4*h)out[d]=table[(size_t)token*h+d%h];
}
extern "C" __global__ void hy4_model_batch_embed(
    const __nv_bfloat16* table,const unsigned* tokens,__nv_bfloat16* out,unsigned h) {
    unsigned d=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;
    if(d<4*h)out[(size_t)row*4*h+d]=table[(size_t)tokens[row]*h+d%h];
}
extern "C" __global__ void hy4_model_norm(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,__nv_bfloat16* out,unsigned h,float eps) {
    x+=(size_t)blockIdx.x*h;out+=(size_t)blockIdx.x*h;
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
    __shared__ float values[256];
    __shared__ unsigned indices[256];
    unsigned lane=threadIdx.x,best=0;float value=logits[0];
    for(unsigned i=lane;i<vocab;i+=blockDim.x)if(logits[i]>value) {value=logits[i];best=i;}
    values[lane]=value;indices[lane]=best;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride/=2) {
        if(lane<stride) {
            float other=values[lane+stride];unsigned index=indices[lane+stride];
            if(other>values[lane] || (other==values[lane] && index<indices[lane])) {
                values[lane]=other;indices[lane]=index;
            }
        }
        __syncthreads();
    }
    if(!lane)*out=indices[0];
}
extern "C" __global__ void hy4_model_argmax_batch(
    const float* logits,unsigned* out,unsigned vocab,unsigned rows) {
    unsigned row=blockIdx.x;
    if(row>=rows)return;
    const float* x=logits+(size_t)row*vocab;
    __shared__ float values[256];
    __shared__ unsigned indices[256];
    unsigned lane=threadIdx.x,best=0;float value=x[0];
    for(unsigned i=lane;i<vocab;i+=blockDim.x)if(x[i]>value) {value=x[i];best=i;}
    values[lane]=value;indices[lane]=best;__syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride/=2) {
        if(lane<stride) {
            float other=values[lane+stride];unsigned index=indices[lane+stride];
            if(other>values[lane] || (other==values[lane] && index<indices[lane])) {
                values[lane]=other;indices[lane]=index;
            }
        }
        __syncthreads();
    }
    if(!lane)out[row]=indices[0];
}
