// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include "hy4_gemv.cuh"

// Row-major ModelOpt nibbles/scales are consumed without repacking.
struct Hy4Q4 { const uint8_t* w; const uint8_t* s; float scale; uint32_t pad; };
static_assert(sizeof(Hy4Q4)==24, "Hy4 descriptor ABI");
__device__ __constant__ float HY4_FP4[16]={
    0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

__device__ __forceinline__ float hy4_qdot(const __nv_bfloat16* x,Hy4Q4 q,
    unsigned row,unsigned k,const float* lut) {
    float sum=0.f;
    if (q.w) for (unsigned col=(threadIdx.x%32)*32;col<k;col+=1024) {
        unsigned count=min(32u,k-col);
        uint4 packed;
        if (count==32 && k%32==0) packed=*(const uint4*)(q.w+(size_t)row*(k/2)+col/2);
        else {
            const uint2* p=(const uint2*)(q.w+(size_t)row*(k/2)+col/2);
            const uint2 lo=p[0],hi=count==32?p[1]:make_uint2(0,0);
            packed=make_uint4(lo.x,lo.y,hi.x,hi.y);
        }
        const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
        for (unsigned group=0;group<count/16;group++) {
            __nv_fp8_e4m3 sf;sf.__x=q.s[(size_t)row*(k/16)+col/16+group];
            float scale=(float)sf*q.scale;
            #pragma unroll
            for (unsigned v=0;v<2;v++) {
                uint4 a=*(const uint4*)(x+col+group*16+v*8);
                const unsigned ar[4]={a.x,a.y,a.z,a.w};
                unsigned w=words[group*2+v];
                #pragma unroll
                for (unsigned j=0;j<4;j++) {
                    float2 values=__bfloat1622float2(*((const __nv_bfloat162*)&ar[j]));
                    sum+=(lut[(w>>(j*8))&15]*scale)*values.x;
                    sum+=(lut[(w>>(j*8+4))&15]*scale)*values.y;
                }
            }
        }
    }
    return hy4_warp_sum(sum);
}
__device__ float hy4_sigmoid(float v) {
    if (v>=0.f) return 1.f/(1.f+expf(-v));
    float e=expf(v); return e/(1.f+e);
}
__device__ float hy4_round(float x) {return (float)__float2bfloat16_rn(x);}

// FP32 dot products. Router emits FP32 logits; latent transforms emit BF16.
extern "C" __global__ void hy4_linear(const __nv_bfloat16* x, const void* w,
    void* y, unsigned N, unsigned K, unsigned weights_f32, unsigned output_f32) {
    unsigned n=blockIdx.x*8+threadIdx.x/32, t=blockIdx.y;
    if (n>=N) return;
    float sum=hy4_dense_dot(x+(size_t)t*K,w,n,K,weights_f32);
    if (threadIdx.x%32==0) {
        if (output_f32) ((float*)y)[(size_t)t*N+n]=sum;
        else ((__nv_bfloat16*)y)[(size_t)t*N+n]=__float2bfloat16_rn(sum);
    }
}

// Bias is selection-only. Ties deterministically choose the lower ID.
extern "C" __global__ void hy4_topk(const float* logits, const float* bias,
    unsigned* ids, float* weights, unsigned E, unsigned topk, unsigned renorm, float scaling) {
    unsigned t=blockIdx.x,e=threadIdx.x;
    __shared__ float candidate[512],selected[512],sum;
    __shared__ unsigned index[512],chosen;
    float score=e<E?hy4_sigmoid(logits[(size_t)t*E+e]):0.f;
    float corrected=e<E?score+bias[e]:-INFINITY;
    bool used=false;
    if (e==0) sum=0.f;
    for (unsigned j=0;j<topk;++j) {
        candidate[e]=used?-INFINITY:corrected;
        index[e]=e<E && !used?e:E;
        __syncthreads();
        for (unsigned stride=256;stride;stride>>=1) {
            if (e<stride) {
                float a=candidate[e],b=candidate[e+stride];
                if (b>a || (b==a && index[e+stride]<index[e])) {
                    candidate[e]=b;index[e]=index[e+stride];
                }
            }
            __syncthreads();
        }
        if (e==0) {chosen=index[0];ids[(size_t)t*topk+j]=chosen;}
        __syncthreads();
        if (e==chosen) {selected[j]=score;used=true;}
        __syncthreads();
        if (e==0) sum+=selected[j];
    }
    __syncthreads();
    if (e<topk)
        weights[(size_t)t*topk+e]=selected[e]/(renorm?sum+1e-20f:1.f)*scaling;
}

extern "C" __global__ void hy4_qgate_up(const __nv_bfloat16* x, const Hy4Q4* table,
    const unsigned* ids, __nv_bfloat16* activated, unsigned K, unsigned I,
    unsigned slots, unsigned group, float limit) {
    __shared__ float lut[16];
    if (threadIdx.x<16)lut[threadIdx.x]=HY4_FP4[threadIdx.x];
    __syncthreads();
    unsigned n=blockIdx.x*8+threadIdx.x/32, t=blockIdx.y, slot=blockIdx.z;
    if (n>=I) return;
    unsigned e=ids?ids[(size_t)t*slots+slot]:group;
    Hy4Q4 g=table[e*3], u=table[e*3+1];
    float a=hy4_qdot(x+(size_t)t*K,g,n,K,lut);
    float b=hy4_qdot(x+(size_t)t*K,u,n,K,lut);
    if (threadIdx.x%32!=0)return;
    a=hy4_round(a); b=hy4_round(b);
    if (limit>0.f) {a=fminf(a,limit); b=fminf(limit,fmaxf(-limit,b));}
    activated[((size_t)t*slots+slot)*I+n]=__float2bfloat16_rn(a*hy4_sigmoid(a)*b);
}

// Remote descriptors are null. Full expert outputs are rounded before weighting.
extern "C" __global__ void hy4_qdown(const __nv_bfloat16* activated, const Hy4Q4* table,
    const unsigned* ids, const float* weights, __nv_bfloat16* out, unsigned N,
    unsigned I, unsigned slots, unsigned group) {
    __shared__ float lut[16],partial[512];
    if (threadIdx.x<16)lut[threadIdx.x]=HY4_FP4[threadIdx.x];
    __syncthreads();
    unsigned n=blockIdx.x,t=blockIdx.y;
    if (n>=N)return;
    for (unsigned slot=threadIdx.x/32;slot<slots;slot+=8) {
        unsigned e=ids?ids[(size_t)t*slots+slot]:group;
        float v=hy4_qdot(activated+((size_t)t*slots+slot)*I,table[e*3+2],n,I,lut);
        if (threadIdx.x%32==0)partial[slot]=hy4_round(v)*(weights?weights[(size_t)t*slots+slot]:1.f);
    }
    __syncthreads();
    if (threadIdx.x==0) {
        float sum=0.f;
        for (unsigned slot=0;slot<slots;slot++)sum+=partial[slot];
        out[(size_t)t*N+n]=__float2bfloat16_rn(sum);
    }
}

// Combine before sandwich RMSNorm. Dense/shared SwiGLU has no clamp.
extern "C" __global__ void hy4_norm_add(const __nv_bfloat16* routed,
    const __nv_bfloat16* shared, const __nv_bfloat16* norm, __nv_bfloat16* out,
    unsigned H, float eps) {
    unsigned t=blockIdx.x, tid=threadIdx.x;
    __shared__ float red[256], r;
    float ss=0.f;
    for (unsigned d=tid;d<H;d+=256) {
        float v=(float)shared[(size_t)t*H+d];
        if (routed) v=hy4_round(v+(float)routed[(size_t)t*H+d]);
        ss+=v*v;
    }
    red[tid]=ss; __syncthreads();
    for (unsigned s=128;s;s>>=1) {
        if (tid<s) red[tid]+=red[tid+s];
        __syncthreads();
    }
    if (tid==0) r=rsqrtf(red[0]/H+eps);
    __syncthreads();
    for (unsigned d=tid;d<H;d+=256) {
        float v=(float)shared[(size_t)t*H+d];
        if (routed) v=hy4_round(v+(float)routed[(size_t)t*H+d]);
        out[(size_t)t*H+d]=__float2bfloat16_rn(norm?v*r*(float)norm[d]:v);
    }
}

#include "hy4_batch.cuh"


// A3 — tensor-core expert variant (opt-in via ATLAS_HY4_EXPERT_MMA=1).
// Loaded via the hy4_ffn module; launcher resolves it by name.
#include "hy4_batch_mma.cuh"
