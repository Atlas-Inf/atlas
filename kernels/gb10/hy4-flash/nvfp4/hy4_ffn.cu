// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

// Correctness-first W4A16 path: checkpoint nibbles/scales are never repacked.
struct Hy4Q4 { const uint8_t* w; const uint8_t* s; float scale; uint32_t pad; };
static_assert(sizeof(Hy4Q4)==24, "Hy4 descriptor ABI");
__device__ __constant__ float HY4_FP4[16]={
    0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

__device__ float hy4_weight(Hy4Q4 q, unsigned n, unsigned k, unsigned K) {
    uint8_t packed=q.w[(size_t)n*(K/2)+k/2];
    __nv_fp8_e4m3 scale;
    scale.__x=q.s[(size_t)n*(K/16)+k/16];
    return HY4_FP4[(packed>>((k&1)*4))&15]*((float)scale*q.scale);
}
__device__ float hy4_sigmoid(float v) {
    if (v>=0.f) return 1.f/(1.f+expf(-v));
    float e=expf(v); return e/(1.f+e);
}
__device__ float hy4_round(float x) {return (float)__float2bfloat16_rn(x);}

// FP32 dot products. Router emits FP32 logits; latent transforms emit BF16.
extern "C" __global__ void hy4_linear(const __nv_bfloat16* x, const void* w,
    void* y, unsigned N, unsigned K, unsigned weights_f32, unsigned output_f32) {
    unsigned n=blockIdx.x*128+threadIdx.x, t=blockIdx.y;
    if (n>=N) return;
    float sum=0.f;
    for (unsigned k=0;k<K;++k) {
        size_t at=(size_t)n*K+k;
        float weight=weights_f32?((const float*)w)[at]:(float)((const __nv_bfloat16*)w)[at];
        sum+=(float)x[(size_t)t*K+k]*weight;
    }
    if (output_f32) ((float*)y)[(size_t)t*N+n]=sum;
    else ((__nv_bfloat16*)y)[(size_t)t*N+n]=__float2bfloat16_rn(sum);
}

// Bias is selection-only. Ties deterministically choose the lower ID.
extern "C" __global__ void hy4_topk(const float* logits, const float* bias,
    unsigned* ids, float* weights, unsigned E, unsigned topk, unsigned renorm, float scaling) {
    unsigned t=blockIdx.x;
    float scores[512]; bool used[512];
    for (unsigned e=0;e<E;++e) {
        scores[e]=hy4_sigmoid(logits[(size_t)t*E+e]); used[e]=false;
    }
    float sum=0.f;
    for (unsigned j=0;j<topk;++j) {
        unsigned best=E; float mx=-INFINITY;
        for (unsigned e=0;e<E;++e) {
            float s=scores[e]+bias[e];
            if (!used[e] && (best==E || s>mx)) {best=e;mx=s;}
        }
        used[best]=true; ids[(size_t)t*topk+j]=best;
        weights[(size_t)t*topk+j]=scores[best]; sum+=scores[best];
    }
    for (unsigned j=0;j<topk;++j)
        weights[(size_t)t*topk+j]=weights[(size_t)t*topk+j]/(renorm?sum+1e-20f:1.f)*scaling;
}

extern "C" __global__ void hy4_qgate_up(const __nv_bfloat16* x, const Hy4Q4* table,
    const unsigned* ids, __nv_bfloat16* activated, unsigned K, unsigned I,
    unsigned slots, unsigned group, float limit) {
    unsigned n=blockIdx.x*128+threadIdx.x, t=blockIdx.y, slot=blockIdx.z;
    if (n>=I) return;
    unsigned e=ids?ids[(size_t)t*slots+slot]:group;
    Hy4Q4 g=table[e*3], u=table[e*3+1];
    float a=0.f,b=0.f;
    if (g.w) {
        for (unsigned k=0;k<K;++k) {
            float v=(float)x[(size_t)t*K+k];
            a+=hy4_weight(g,n,k,K)*v; b+=hy4_weight(u,n,k,K)*v;
        }
    }
    a=hy4_round(a); b=hy4_round(b);
    if (limit>0.f) {a=fminf(a,limit); b=fminf(limit,fmaxf(-limit,b));}
    activated[((size_t)t*slots+slot)*I+n]=__float2bfloat16_rn(a*hy4_sigmoid(a)*b);
}

// Remote descriptors are null. Full expert outputs are rounded before weighting.
extern "C" __global__ void hy4_qdown(const __nv_bfloat16* activated, const Hy4Q4* table,
    const unsigned* ids, const float* weights, __nv_bfloat16* out, unsigned N,
    unsigned I, unsigned slots, unsigned group) {
    unsigned n=blockIdx.x*128+threadIdx.x, t=blockIdx.y;
    if (n>=N) return;
    float sum=0.f;
    for (unsigned slot=0;slot<slots;++slot) {
        unsigned e=ids?ids[(size_t)t*slots+slot]:group;
        Hy4Q4 d=table[e*3+2]; float v=0.f;
        if (d.w) for (unsigned k=0;k<I;++k)
            v+=hy4_weight(d,n,k,I)*(float)activated[((size_t)t*slots+slot)*I+k];
        sum+=hy4_round(v)*(weights?weights[(size_t)t*slots+slot]:1.f);
    }
    out[(size_t)t*N+n]=__float2bfloat16_rn(sum);
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
