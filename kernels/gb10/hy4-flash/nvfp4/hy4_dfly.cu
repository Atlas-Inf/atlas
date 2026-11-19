// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void hy4_dfly_embed(const __nv_bfloat16* table,
    const unsigned* ids,__nv_bfloat16* out,unsigned h) {
    unsigned row=blockIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<h)out[(size_t)row*h+d]=table[(size_t)ids[row]*h+d];
}

extern "C" __global__ void hy4_dfly_norm(const __nv_bfloat16* x,
    const __nv_bfloat16* w,__nv_bfloat16* y,unsigned h,unsigned xs,unsigned ys,float eps) {
    x+=(size_t)blockIdx.x*xs;y+=(size_t)blockIdx.x*ys;
    __shared__ float sums[256];float sum=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=256) {float v=(float)x[d];sum+=v*v;}
    sums[threadIdx.x]=sum;__syncthreads();
    for(unsigned s=128;s;s>>=1) {
        if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();
    }
    float inv=rsqrtf(sums[0]/h+eps);
    for(unsigned d=threadIdx.x;d<h;d+=256)
        y[d]=__float2bfloat16_rn((float)x[d]*inv*(float)w[d]);
}

extern "C" __global__ void hy4_dfly_add(const __nv_bfloat16* a,
    const __nv_bfloat16* b,__nv_bfloat16* out,unsigned size) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size)out[i]=__float2bfloat16_rn((float)a[i]+(float)b[i]);
}

// Two BF16 operations, matching the unfused slot0 expression.
extern "C" __global__ void hy4_dfly_fuse(const __nv_bfloat16* anchor,
    const __nv_bfloat16* projected,const __nv_bfloat16* gate,__nv_bfloat16* out,unsigned h) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<h) {
        __nv_bfloat16 product=__float2bfloat16_rn((float)projected[i]*(float)gate[i]);
        out[i]=__float2bfloat16_rn((float)anchor[i]+(float)product);
    }
}

extern "C" __global__ void hy4_dfly_silu(const __nv_bfloat16* gate,
    const __nv_bfloat16* up,__nv_bfloat16* out,unsigned size) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) {
        float x=(float)gate[i];
        out[i]=__float2bfloat16_rn(x/(1.f+expf(-x))*(float)up[i]);
    }
}

extern "C" __global__ void hy4_dfly_gate(__nv_bfloat16* values,
    const __nv_bfloat16* gate,unsigned size) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) {
        __nv_bfloat16 sigmoid=__float2bfloat16_rn(1.f/(1.f+expf(-(float)gate[i])));
        values[i]=__float2bfloat16_rn((float)values[i]*(float)sigmoid);
    }
}

extern "C" __global__ void hy4_dfly_pack(const __nv_bfloat16* q,
    const __nv_bfloat16* kv,__nv_bfloat16* absorbed,__nv_bfloat16* cache,
    unsigned heads,unsigned anchor,float theta) {
    unsigned row=blockIdx.y,head=blockIdx.x,i=threadIdx.x;
    const __nv_bfloat16* qr=q+((size_t)row*heads+head)*256+192;
    __nv_bfloat16* dst=absorbed+((size_t)row*heads+head)*576+512;
    if(i<32) {
        float angle=(float)(anchor+row)*powf(theta,-(float)i/32.f),s,c;
        sincosf(angle,&s,&c);
        float a=(float)qr[i*2],b=(float)qr[i*2+1];
        dst[i*2]=__float2bfloat16_rn(a*c-b*s);
        dst[i*2+1]=__float2bfloat16_rn(a*s+b*c);
        if(!head) {
            const __nv_bfloat16* kr=kv+(size_t)row*576+512;
            __nv_bfloat16* kd=cache+(size_t)(anchor+row)*576+512;
            a=(float)kr[i*2];b=(float)kr[i*2+1];
            kd[i*2]=__float2bfloat16_rn(a*c-b*s);
            kd[i*2+1]=__float2bfloat16_rn(a*s+b*c);
        }
    }
    if(!head) for(unsigned d=i;d<512;d+=blockDim.x)
        cache[(size_t)(anchor+row)*576+d]=kv[(size_t)row*576+d];
}

extern "C" __global__ void hy4_dfly_attention(const __nv_bfloat16* q,
    const __nv_bfloat16* cache,const int* history,const __nv_bfloat16* sink,
    __nv_bfloat16* out,unsigned heads,unsigned anchor,unsigned topk) {
    unsigned row=blockIdx.y,head=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ float query[576],scores[2176],red[256],maximum,denominator;
    for(unsigned d=tid;d<576;d+=256)query[d]=(float)q[((size_t)row*heads+head)*576+d];
    __syncthreads();
    unsigned count=topk+row+1;
    for(unsigned j=warp;j<count;j+=8) {
        int p=j<topk?history[j]:(int)(anchor+j-topk);
        bool valid=p>=0 && (j>=topk || (unsigned)p<anchor);
        float dot=0.f;
        if(valid)for(unsigned d=lane;d<576;d+=32)
            dot+=query[d]*(float)cache[(size_t)p*576+d];
        for(unsigned s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
        if(!lane)scores[j]=valid?dot*(1.f/16.f):-INFINITY;
    }
    __syncthreads();
    float m=(float)sink[head];
    for(unsigned j=tid;j<count;j+=256)m=fmaxf(m,scores[j]);
    red[tid]=m;__syncthreads();
    for(unsigned s=128;s;s>>=1) {
        if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();
    }
    if(!tid)maximum=red[0];__syncthreads();
    float sum=0.f;
    for(unsigned j=tid;j<count;j+=256) {
        float v=expf(scores[j]-maximum);scores[j]=v;sum+=v;
    }
    red[tid]=sum;__syncthreads();
    for(unsigned s=128;s;s>>=1) {
        if(tid<s)red[tid]+=red[tid+s];__syncthreads();
    }
    if(!tid)denominator=red[0]+expf((float)sink[head]-maximum);__syncthreads();
    for(unsigned d=tid;d<512;d+=256) {
        float value=0.f;
        for(unsigned j=0;j<count;j++) {
            int p=j<topk?history[j]:(int)(anchor+j-topk);
            if(p>=0 && (j>=topk || (unsigned)p<anchor))
                value+=scores[j]*(float)cache[(size_t)p*576+d];
        }
        out[((size_t)row*heads+head)*512+d]=__float2bfloat16_rn(value/denominator);
    }
}
