// SPDX-License-Identifier: AGPL-3.0-only

#include "hy4_gemv.cuh"
#include "hy4_fp8.cuh"
#include "../../common/quant_rowwise_fp8.cu"
#include <math.h>

extern "C" __global__ void hy4_head_gemv(const __nv_bfloat16* x,
    const __nv_bfloat16* w, __nv_bfloat16* y, unsigned n, unsigned k,
    unsigned input_stride, unsigned output_stride) {
    unsigned row=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y;
    if (row>=n) return;
    float value=hy4_dense_dot(x+(size_t)head*input_stride,
        w+(size_t)head*n*k,row,k,0);
    if (threadIdx.x%32==0) y[(size_t)head*output_stride+row]=__float2bfloat16_rn(value);
}

extern "C" __global__ void hy4_rotary(const __nv_bfloat16* x, __nv_bfloat16* y,
    unsigned in_stride, unsigned out_stride, unsigned in_offset, unsigned out_offset,
    unsigned long long pt, unsigned long long ph, unsigned long long pw,
    float theta, unsigned sh, unsigned sw, unsigned interleaved, unsigned split) {
    unsigned i=threadIdx.x, head=blockIdx.x;
    if (i>=32) return;
    unsigned axis=interleaved?((i%3==1 && i<sh*3)?1:((i%3==2 && i<sw*3)?2:0)):
        (i<32-sh-sw?0:(i<32-sw?1:2));
    float p=(float)(axis==0?pt:(axis==1?ph:pw));
    float angle=p*powf(theta,-(float)i/32.f),s,c;
    sincosf(angle,&s,&c);
    const __nv_bfloat16* src=x+(size_t)head*in_stride+in_offset;
    float a=(float)src[i*2],b=(float)src[i*2+1];
    __nv_bfloat16* dst=y+(size_t)head*out_stride+out_offset;
    // A separate barrier permits the indexer's in-place pair-to-half shuffle.
    __syncthreads();
    dst[split?i:i*2]=__float2bfloat16_rn(a*c-b*s);
    dst[split?32+i:i*2+1]=__float2bfloat16_rn(a*s+b*c);
}

extern "C" __global__ void hy4_gate(__nv_bfloat16* out,
    const __nv_bfloat16* logits, unsigned width) {
    unsigned d=blockIdx.x*256+threadIdx.x;
    if (d<width) {
        float g=(float)logits[d];
        float s=g>=0.f?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
        out[d]=__float2bfloat16_rn((float)out[d]*s);
    }
}

// Each head/128-key partition produces an unnormalized FP32 softmax partial.
extern "C" __global__ void hy4_mla_partial(const __nv_bfloat16* q,
    const __nv_bfloat16* cache, const unsigned* table, const int* indices,
    float* partials, unsigned position, unsigned page_size,
    unsigned columns, unsigned window, unsigned count, unsigned splits) {
    unsigned h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ float scores[128],red[256],query[576],maximum,denominator;
    __shared__ int rows[128];
    for (unsigned d=tid;d<576;d+=256) query[d]=(float)q[(size_t)h*576+d];
    __syncthreads();
    unsigned begin=window && position+1>window?position+1-window:0;
    for (unsigned j=warp;j<128;j+=8) {
        unsigned at=part*128+j;
        int logical=at<count?(indices?indices[at]:(int)(begin+at)):-1;
        bool valid=logical>=0 && (unsigned)logical<=position && (unsigned)logical>=begin;
        size_t row=0;
        if (valid) {
            unsigned page=(unsigned)logical/page_size;
            row=(size_t)table[window?page%columns:page]*page_size+(unsigned)logical%page_size;
        }
        const __nv_bfloat16* kv=cache+row*576;
        float dot=0.f;
        if (valid) for (unsigned d=lane*8;d<576;d+=256) {
            uint4 v=*(const uint4*)(kv+d);
            const unsigned words[4]={v.x,v.y,v.z,v.w};
            #pragma unroll
            for (unsigned k=0;k<4;k++) {
                float2 b=__bfloat1622float2(*((const __nv_bfloat162*)&words[k]));
                dot+=query[d+k*2]*b.x;dot+=query[d+k*2+1]*b.y;
            }
        }
        dot=hy4_warp_sum(dot);
        if (lane==0) { scores[j]=valid?dot*(1.f/16.f):-INFINITY;rows[j]=valid?(int)row:-1; }
    }
    __syncthreads();
    red[tid]=tid<128?scores[tid]:-INFINITY;
    for (unsigned s=128;s;s>>=1) {
        __syncthreads();
        if (tid<s) red[tid]=fmaxf(red[tid],red[tid+s]);
    }
    if (tid==0) maximum=red[0];
    __syncthreads();
    float weight=tid<128 && rows[tid]>=0?expf(scores[tid]-maximum):0.f;
    if (tid<128) scores[tid]=weight;
    red[tid]=weight;
    for (unsigned s=128;s;s>>=1) {
        __syncthreads();
        if (tid<s) red[tid]+=red[tid+s];
    }
    if (tid==0) denominator=red[0];
    __syncthreads();
    float a=0.f,b=0.f;
    for (unsigned j=0;j<128;j++) if (rows[j]>=0) {
        const __nv_bfloat16* kv=cache+(size_t)rows[j]*576;
        a+=scores[j]*(float)kv[tid];b+=scores[j]*(float)kv[tid+256];
    }
    float* out=partials+((size_t)h*splits+part)*514;
    out[tid]=a;out[tid+256]=b;
    if (tid==0) {out[512]=maximum;out[513]=denominator;}
}

// The zero-value sink participates once, including when every key is masked.
extern "C" __global__ void hy4_mla_merge(const float* partials,const float* sink,
    __nv_bfloat16* out,unsigned splits) {
    unsigned h=blockIdx.x,d=threadIdx.x;
    const float* parts=partials+(size_t)h*splits*514;
    float maximum=sink[h];
    for (unsigned p=0;p<splits;p++) maximum=fmaxf(maximum,parts[p*514+512]);
    float denominator=expf(sink[h]-maximum),a=0.f,b=0.f;
    for (unsigned p=0;p<splits;p++) {
        float scale=expf(parts[p*514+512]-maximum);
        denominator+=scale*parts[p*514+513];
        a+=scale*parts[p*514+d];b+=scale*parts[p*514+d+256];
    }
    out[(size_t)h*512+d]=__float2bfloat16_rn(a/denominator);
    out[(size_t)h*512+d+256]=__float2bfloat16_rn(b/denominator);
}
