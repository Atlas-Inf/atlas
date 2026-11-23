// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// BF16 highway; FP32 gate projection and accumulation, no comb matrix.
__device__ float hy4_sum(float* v) {
    for (unsigned s=128;s;s>>=1) {
        if (threadIdx.x<s) v[threadIdx.x]+=v[threadIdx.x+s];
        __syncthreads();
    }
    return v[0];
}

template<bool HEAD>
__device__ void hy4_collapse(const __nv_bfloat16* streams, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* y, float* post,
    unsigned h, float norm_eps, float gate_eps, float magnitude) {
    const unsigned t=blockIdx.x, tid=threadIdx.x, dim=4*h;
    const __nv_bfloat16* x=streams+(size_t)t*dim;
    __shared__ float red[256], r, gates[8];
    float ss=0.f;
    for (unsigned k=tid;k<dim;k+=256) {float v=(float)x[k]; ss+=v*v;}
    red[tid]=ss;
    __syncthreads();
    float total=hy4_sum(red);
    if (tid==0) r=rsqrtf(total/dim+norm_eps);
    __syncthreads();
    for (unsigned m=0;m<(HEAD?4u:8u);++m) {
        float dot=0.f;
        for (unsigned k=tid;k<dim;k+=256) dot+=fn[(size_t)m*dim+k]*(float)x[k];
        red[tid]=dot;
        __syncthreads();
        total=hy4_sum(red);
        if (tid==0) {
            bool is_post=!HEAD && m>=4;
            float z=total*r*scale[is_post?1:0]+base[m];
            gates[m]=(is_post?magnitude:1.f)/(1.f+expf(-z))+gate_eps;
            if (is_post) post[(size_t)t*4+m-4]=gates[m];
        }
        __syncthreads();
    }
    for (unsigned d=tid;d<h;d+=256) {
        float sum=0.f;
        for (unsigned i=0;i<4;++i) sum+=gates[i]*(float)x[i*h+d];
        y[(size_t)t*h+d]=__float2bfloat16_rn(sum);
    }
}

template<bool HEAD>
__device__ void hy4_collapse_norm_at(const __nv_bfloat16* x, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* normed,
    const __nv_bfloat16* weight, float* post,
    unsigned h, float norm_eps, float gate_eps, float magnitude, float rms_eps) {
    const unsigned t=blockIdx.x, tid=threadIdx.x, dim=4*h;
    __shared__ float red[256], r, gates[8];
    float ss=0.f;
    for (unsigned k=tid;k<dim;k+=256) {float v=(float)x[k]; ss+=v*v;}
    red[tid]=ss;
    __syncthreads();
    float total=hy4_sum(red);
    if (tid==0) r=rsqrtf(total/dim+norm_eps);
    __syncthreads();
    for (unsigned m=0;m<(HEAD?4u:8u);++m) {
        float dot=0.f;
        for (unsigned k=tid;k<dim;k+=256) dot+=fn[(size_t)m*dim+k]*(float)x[k];
        red[tid]=dot;
        __syncthreads();
        total=hy4_sum(red);
        if (tid==0) {
            bool is_post=!HEAD && m>=4;
            float z=total*r*scale[is_post?1:0]+base[m];
            gates[m]=(is_post?magnitude:1.f)/(1.f+expf(-z))+gate_eps;
            if (is_post) post[(size_t)t*4+m-4]=gates[m];
        }
        __syncthreads();
    }
    __nv_bfloat16 vals[32];
    const unsigned dcount=(h+255)/256;
    float nss=0.f;
    for (unsigned d=tid;d<h;d+=256) {
        float sum=0.f;
        for (unsigned i=0;i<4;++i) sum+=gates[i]*(float)x[i*h+d];
        __nv_bfloat16 v=__float2bfloat16_rn(sum);
        vals[(d-tid)/256]=v;
        float f=(float)v;
        nss+=f*f;
    }
    red[tid]=nss;
    __syncthreads();
    for(int n=128;n;n>>=1) {
        if(tid<n) red[tid]+=red[tid+n];
        __syncthreads();
    }
    float inv=rsqrtf(red[0]/h+rms_eps);
    for (unsigned j=0;j<dcount;++j) {
        unsigned d=tid+j*256;
        if(d<h) normed[(size_t)t*h+d]=__float2bfloat16_rn((float)vals[j]*inv*(float)weight[d]);
    }
}

template<bool HEAD>
__device__ void hy4_collapse_norm(const __nv_bfloat16* streams, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* normed,
    const __nv_bfloat16* weight, float* post,
    unsigned h, float norm_eps, float gate_eps, float magnitude, float rms_eps) {
    hy4_collapse_norm_at<HEAD>(streams+(size_t)blockIdx.x*4*h,fn,scale,base,normed,weight,post,h,norm_eps,gate_eps,magnitude,rms_eps);
}

extern "C" __global__ void hy4_hc_pre(const __nv_bfloat16* x, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* y, float* post,
    unsigned h, float norm_eps, float gate_eps, float magnitude) {
    hy4_collapse<false>(x,fn,scale,base,y,post,h,norm_eps,gate_eps,magnitude);
}

extern "C" __global__ void hy4_hc_pre_norm(const __nv_bfloat16* x, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* normed,
    const __nv_bfloat16* weight, float* post,
    unsigned h, float norm_eps, float gate_eps, float magnitude, float rms_eps) {
    hy4_collapse_norm<false>(x,fn,scale,base,normed,weight,post,h,norm_eps,gate_eps,magnitude,rms_eps);
}

extern "C" __global__ void hy4_hc_head_norm(const __nv_bfloat16* x, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* normed,
    const __nv_bfloat16* weight,
    unsigned h, float norm_eps, float gate_eps, float rms_eps) {
    hy4_collapse_norm<true>(x,fn,scale,base,normed,weight,nullptr,h,norm_eps,gate_eps,1.f,rms_eps);
}

extern "C" __global__ void hy4_hc_head(const __nv_bfloat16* x, const float* fn,
    const float* scale, const float* base, __nv_bfloat16* y,
    unsigned h, float norm_eps, float gate_eps) {
    hy4_collapse<true>(x,fn,scale,base,y,nullptr,h,norm_eps,gate_eps,1.f);
}

extern "C" __global__ void hy4_hc_post(const __nv_bfloat16* block,
    const __nv_bfloat16* residual, const float* post, __nv_bfloat16* out, unsigned h) {
    size_t t=blockIdx.x;
    for (unsigned k=threadIdx.x;k<4*h;k+=256)
        out[t*4*h+k]=__float2bfloat16_rn((float)residual[t*4*h+k]
            +post[t*4+k/h]*(float)block[t*h+k%h]);
}

extern "C" __global__ void hy4_hc_post_pre_norm(const __nv_bfloat16* block,
    __nv_bfloat16* highway, float* post, const float* fn, const float* scale,
    const float* base, __nv_bfloat16* normed, const __nv_bfloat16* weight,
    unsigned h, float norm_eps, float gate_eps, float magnitude, float rms_eps) {
    const unsigned t=blockIdx.x, tid=threadIdx.x, dim=4*h;
    extern __shared__ __nv_bfloat16 sh[];
    const size_t tdim=(size_t)t*dim;
    const __nv_bfloat16* y=block+(size_t)t*h;
    __nv_bfloat16* hw=highway+tdim;
    for (unsigned k=tid;k<dim;k+=256) {
        const unsigned stream=k/h;
        const unsigned d=k%h;
        const float gate=post[(size_t)t*4+stream];
        const __nv_bfloat16 v=__float2bfloat16_rn((float)hw[k]+gate*(float)y[d]);
        hw[k]=v;
        sh[k]=v;
    }
    __syncthreads();
    hy4_collapse_norm_at<false>(sh,fn,scale,base,normed,weight,post,h,norm_eps,gate_eps,magnitude,rms_eps);
}
