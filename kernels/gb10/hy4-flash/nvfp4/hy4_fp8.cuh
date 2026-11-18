// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cuda_fp8.h>
#include "hy4_gemv.cuh"

__device__ __forceinline__ float hy4_fp8_dot(
    const __nv_bfloat16* x,const unsigned char* w,unsigned k,float scale) {
    unsigned lane=threadIdx.x%32;float sum=0.f;
    unsigned limit=k&~15u;
    for(unsigned col=lane*16;col<limit;col+=512) {
        uint4 weights=*(const uint4*)(w+col);
        uint4 lo=*(const uint4*)(x+col),hi=*(const uint4*)(x+col+8);
        unsigned ws[4]={weights.x,weights.y,weights.z,weights.w};
        unsigned xs[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        #pragma unroll
        for(unsigned j=0;j<8;j++) {
            __nv_fp8x2_e4m3 pair;pair.__x=(ws[j/2]>>((j%2)*16))&65535u;
            float2 v=__half22float2((__half2)pair);
            __nv_bfloat162 bx=*(const __nv_bfloat162*)&xs[j];
            float2 a=__bfloat1622float2(bx);
            sum+=a.x*(v.x*scale);sum+=a.y*(v.y*scale);
        }
    }
    for(unsigned col=limit+lane;col<k;col+=32) {
        __nv_fp8_e4m3 value;value.__x=w[col];
        sum+=(float)x[col]*((float)value*scale);
    }
    return hy4_warp_sum(sum);
}

extern "C" __global__ void hy4_fp8_gemv(
    const __nv_bfloat16* x,const unsigned char* w,const float* scales,
    __nv_bfloat16* y,unsigned n,unsigned k,unsigned input_stride,unsigned output_stride) {
    unsigned row=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y;
    if(row>=n)return;
    size_t index=(size_t)head*n+row;
    const __nv_bfloat16* input=x+(size_t)head*input_stride;
    const unsigned char* weight=w+index*k;
    float sum;
    if(((uintptr_t)input%16)==0 && ((uintptr_t)weight%16)==0) {
        sum=hy4_fp8_dot(input,weight,k,scales[index]);
    } else {
        sum=0.f;
        for(unsigned col=threadIdx.x%32;col<k;col+=32) {
            __nv_fp8_e4m3 value;value.__x=weight[col];
            sum+=(float)input[col]*((float)value*scales[index]);
        }
        sum=hy4_warp_sum(sum);
    }
    if(threadIdx.x%32==0)y[(size_t)head*output_stride+row]=__float2bfloat16_rn(sum);
}

extern "C" __global__ void hy4_join_qkv(const __nv_bfloat16* shares,
    __nv_bfloat16* q,__nv_bfloat16* kv,unsigned qrank) {
    unsigned qa=qrank/2,ka=288,stride=qa+ka;
    for(unsigned i=threadIdx.x;i<qrank;i+=blockDim.x) q[i]=shares[(i/qa)*stride+i%qa];
    for(unsigned i=threadIdx.x;i<576;i+=blockDim.x) kv[i]=shares[(i/ka)*stride+qa+i%ka];
}
