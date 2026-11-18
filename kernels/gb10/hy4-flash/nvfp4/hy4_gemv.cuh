// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cuda_bf16.h>

__device__ __forceinline__ float hy4_warp_sum(float value) {
    for (unsigned offset=16;offset;offset>>=1)
        value+=__shfl_down_sync(0xffffffff,value,offset);
    return value;
}

__device__ __forceinline__ float hy4_dense_dot(const __nv_bfloat16* x,
    const void* weight,unsigned row,unsigned k,unsigned weights_f32) {
    unsigned lane=threadIdx.x%32;
    float sum=0.f;
    for (unsigned col=lane*8;col<k;col+=256) {
        if (col+8<=k) {
            uint4 xv=*(const uint4*)(x+col);
            const unsigned xr[4]={xv.x,xv.y,xv.z,xv.w};
            if (weights_f32) {
                const float* w=(const float*)weight+(size_t)row*k+col;
                float4 lo=*(const float4*)w,hi=*(const float4*)(w+4);
                const float values[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
                #pragma unroll
                for (unsigned j=0;j<4;j++) {
                    float2 a=__bfloat1622float2(*((const __nv_bfloat162*)&xr[j]));
                    sum+=a.x*values[2*j];sum+=a.y*values[2*j+1];
                }
            } else {
                uint4 wv=*(const uint4*)((const __nv_bfloat16*)weight+(size_t)row*k+col);
                const unsigned wr[4]={wv.x,wv.y,wv.z,wv.w};
                #pragma unroll
                for (unsigned j=0;j<4;j++) {
                    float2 a=__bfloat1622float2(*((const __nv_bfloat162*)&xr[j]));
                    float2 b=__bfloat1622float2(*((const __nv_bfloat162*)&wr[j]));
                    sum+=a.x*b.x;sum+=a.y*b.y;
                }
            }
        } else {
            for (unsigned j=col;j<k;j++) {
                size_t at=(size_t)row*k+j;
                float w=weights_f32?((const float*)weight)[at]:(float)((const __nv_bfloat16*)weight)[at];
                sum+=(float)x[j]*w;
            }
        }
    }
    return hy4_warp_sum(sum);
}
