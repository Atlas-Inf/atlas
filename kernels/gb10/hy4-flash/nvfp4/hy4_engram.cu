// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

static __device__ float eg_sum(float x) {
    __shared__ float values[256];
    values[threadIdx.x]=x;__syncthreads();
    for (int n=blockDim.x/2;n;n>>=1) {
        if (threadIdx.x<n) values[threadIdx.x]+=values[threadIdx.x+n];
        __syncthreads();
    }
    return values[0];
}
static __device__ float eg_round(float x) {return __bfloat162float(__float2bfloat16_rn(x));}

extern "C" __global__ void hy4_engram_hash(
    int token,int previous,int older,const int* map,const uint64_t* multipliers,
    const int* sizes,const int* offsets,int* hashes,unsigned vocab,unsigned heads,
    unsigned heads_per_order,unsigned pad) {
    unsigned h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=heads)return;
    if(token<0 || (unsigned)token>=vocab || (unsigned)token==pad) {hashes[h]=-1;return;}
    int tokens[3]={token,previous,older};uint64_t rolling=0;bool blocked=false;
    unsigned order=h/heads_per_order+2;
    for(unsigned s=0;s<order;s++) {
        int id=tokens[s];blocked|=id<0 || (unsigned)id>=vocab;
        int source=blocked?map[pad]:map[id];
        rolling^=(uint64_t)source*multipliers[s];
    }
    int64_t rem=(int64_t)rolling%(int64_t)sizes[h];
    if(rem<0)rem+=sizes[h];
    hashes[h]=(int)rem+offsets[h];
}

extern "C" __global__ void hy4_engram_lookup(
    const int* hashes,const __nv_fp8_e4m3* table,const __nv_bfloat16* scales,
    const __nv_bfloat16* gamma,__nv_bfloat16* output,
    unsigned first_head,unsigned base_row,unsigned local_rows,unsigned dim) {
    unsigned head=blockIdx.x;int row=hashes[first_head+head];
    bool valid=row>=0 && (unsigned)row>=base_row && (unsigned)row-base_row<local_rows;
    unsigned local=valid?(unsigned)row-base_row:0;
    float scale=valid?__bfloat162float(scales[local]):0.f;
    float v=(threadIdx.x<dim && valid)?(float)table[(size_t)local*dim+threadIdx.x]*scale:0.f;
    float inv=rsqrtf(eg_sum(v*v)/(float)dim+1.1920928955078125e-7f);
    if(threadIdx.x<dim)output[head*dim+threadIdx.x]=__float2bfloat16_rn(
        v*inv*__bfloat162float(gamma[head*dim+threadIdx.x]));
}

// Batched variants: one launch per stage for the whole chunk (grid y = row),
// identical math to the per-row launches they replace.
extern "C" __global__ void hy4_engram_hash_batch(
    const int* window,const int* map,const uint64_t* multipliers,
    const int* sizes,const int* offsets,int* hashes,unsigned vocab,unsigned heads,
    unsigned heads_per_order,unsigned pad,unsigned rows) {
    unsigned row=blockIdx.x;if(row>=rows)return;
    int token=window[row*3],previous=window[row*3+1],older=window[row*3+2];
    unsigned h=threadIdx.x;if(h>=heads)return;
    if(token<0 || (unsigned)token>=vocab || (unsigned)token==pad) {hashes[(size_t)row*heads+h]=-1;return;}
    int tokens[3]={token,previous,older};uint64_t rolling=0;bool blocked=false;
    unsigned order=h/heads_per_order+2;
    for(unsigned s=0;s<order;s++) {
        int id=tokens[s];blocked|=id<0 || (unsigned)id>=vocab;
        int source=blocked?map[pad]:map[id];
        rolling^=(uint64_t)source*multipliers[s];
    }
    int64_t rem=(int64_t)rolling%(int64_t)sizes[h];
    if(rem<0)rem+=sizes[h];
    hashes[(size_t)row*heads+h]=(int)rem+offsets[h];
}

extern "C" __global__ void hy4_engram_lookup_batch(
    const int* hashes,const __nv_fp8_e4m3* table,const __nv_bfloat16* scales,
    const __nv_bfloat16* gamma,__nv_bfloat16* output,
    unsigned first_head,unsigned base_row,unsigned local_rows,unsigned dim,
    unsigned hash_stride,unsigned local,unsigned rows) {
    unsigned head=blockIdx.x,row=blockIdx.y;
    if(row>=rows)return;
    int h=hashes[(size_t)row*hash_stride+first_head+head];
    bool valid=h>=0 && (unsigned)h>=base_row && (unsigned)h-base_row<local_rows;
    unsigned l=valid?(unsigned)h-base_row:0;
    float scale=valid?__bfloat162float(scales[l]):0.f;
    float v=(threadIdx.x<dim && valid)?(float)table[(size_t)l*dim+threadIdx.x]*scale:0.f;
    float inv=rsqrtf(eg_sum(v*v)/(float)dim+1.1920928955078125e-7f);
    if(threadIdx.x<dim)output[(size_t)row*local+head*dim+threadIdx.x]=__float2bfloat16_rn(
        v*inv*__bfloat162float(gamma[head*dim+threadIdx.x]));
}

extern "C" __global__ void hy4_engram_norm_batch(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,__nv_bfloat16* output,
    unsigned h,unsigned stride,unsigned rows) {
    unsigned c=blockIdx.x,row=blockIdx.y;
    if(row>=rows)return;
    size_t base=(size_t)row*stride+(size_t)c*h;
    float ss=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float v=__bfloat162float(x[base+d]);ss+=v*v;
    }
    float inv=rsqrtf(eg_sum(ss)/(float)h+1.1920928955078125e-7f);
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x)
        output[base+d]=__float2bfloat16_rn(__bfloat162float(x[base+d])*inv*
            __bfloat162float(weight[c*h+d]));
}

extern "C" __global__ void hy4_engram_inject_batch(
    __nv_bfloat16* hidden,const __nv_bfloat16* keys,const __nv_bfloat16* queries,
    const __nv_bfloat16* value,const __nv_bfloat16* gamma,
    const __nv_bfloat16* scales,const int* hashes,unsigned h,
    unsigned stride,unsigned hash_stride,unsigned rows) {
    unsigned c=blockIdx.x,row=blockIdx.y;
    if(row>=rows)return;
    if(hashes[(size_t)row*hash_stride]<0)return;
    size_t base=(size_t)row*stride+(size_t)c*h;
    const __nv_bfloat16* val=value+(size_t)row*h;
    float dot=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x)
        dot+=eg_round(__bfloat162float(keys[base+d])*__bfloat162float(queries[base+d]));
    float z=eg_round(eg_round(eg_sum(dot))*rsqrtf((float)h));
    float amplitude=eg_round(sqrtf(fmaxf(fabsf(z),eg_round(1e-6f))));
    float signed_z=z==0.f?0.f:copysignf(amplitude,z);
    float gate=eg_round(1.f/(1.f+expf(-signed_z)));
    float ss=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float v=__bfloat162float(val[d]);ss+=v*v;
    }
    float inv=rsqrtf(eg_sum(ss)/(float)h+1.1920928955078125e-7f);
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float signal=__bfloat162float(gamma[d])*__bfloat162float(val[d])*inv;
        float update=eg_round((gate+__bfloat162float(scales[c*h+d]))*signal);
        hidden[base+d]=__float2bfloat16_rn(__bfloat162float(hidden[base+d])+update);
    }
}

extern "C" __global__ void hy4_engram_project(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,const __nv_bfloat16* bias,
    __nv_bfloat16* output,unsigned rows,unsigned full_width,unsigned start,
    unsigned local_width,unsigned add_bias) {
    unsigned n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=rows)return;
    float sum=0.f;
    for(unsigned k=0;k<local_width;k++)
        sum+=__bfloat162float(x[k])*__bfloat162float(weight[(size_t)n*full_width+start+k]);
    if(add_bias)sum+=__bfloat162float(bias[n]);
    output[n]=__float2bfloat16_rn(sum);
}

extern "C" __global__ void hy4_engram_batch_project(
    const __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* bias,
    __nv_bfloat16* out,unsigned h,unsigned full,unsigned start,unsigned k,
    unsigned add_bias,unsigned rows,unsigned stride) {
    unsigned n=blockIdx.x*blockDim.x+threadIdx.x,first=blockIdx.y*8;
    if(n>=h)return;
    float sums[8]={};
    for(unsigned c=0;c<k;c++) {
        float weight=(float)w[(size_t)n*full+start+c];
        #pragma unroll
        for(unsigned r=0;r<8;r++)if(first+r<rows)
            sums[r]+=(float)x[(size_t)(first+r)*k+c]*weight;
    }
    #pragma unroll
    for(unsigned r=0;r<8;r++)if(first+r<rows) {
        float sum=sums[r];
        if(add_bias)sum+=(float)bias[n];
        out[(size_t)(first+r)*stride+n]=__float2bfloat16_rn(sum);
    }
}

extern "C" __global__ void hy4_engram_norm(
    const __nv_bfloat16* x,const __nv_bfloat16* weight,__nv_bfloat16* output,unsigned h) {
    unsigned c=blockIdx.x;float ss=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float v=__bfloat162float(x[c*h+d]);ss+=v*v;
    }
    float inv=rsqrtf(eg_sum(ss)/(float)h+1.1920928955078125e-7f);
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x)
        output[c*h+d]=__float2bfloat16_rn(__bfloat162float(x[c*h+d])*inv*
            __bfloat162float(weight[c*h+d]));
}

extern "C" __global__ void hy4_engram_inject(
    __nv_bfloat16* hidden,const __nv_bfloat16* keys,const __nv_bfloat16* queries,
    const __nv_bfloat16* value,const __nv_bfloat16* gamma,
    const __nv_bfloat16* scales,const int* hashes,unsigned h) {
    if(hashes[0]<0)return;
    unsigned c=blockIdx.x;float dot=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x)
        dot+=eg_round(__bfloat162float(keys[c*h+d])*__bfloat162float(queries[c*h+d]));
    float z=eg_round(eg_round(eg_sum(dot))*rsqrtf((float)h));
    float amplitude=eg_round(sqrtf(fmaxf(fabsf(z),eg_round(1e-6f))));
    float signed_z=z==0.f?0.f:copysignf(amplitude,z);
    float gate=eg_round(1.f/(1.f+expf(-signed_z)));
    float ss=0.f;
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float v=__bfloat162float(value[d]);ss+=v*v;
    }
    float inv=rsqrtf(eg_sum(ss)/(float)h+1.1920928955078125e-7f);
    for(unsigned d=threadIdx.x;d<h;d+=blockDim.x) {
        float signal=__bfloat162float(gamma[d])*__bfloat162float(value[d])*inv;
        float update=eg_round((gate+__bfloat162float(scales[c*h+d]))*signal);
        hidden[c*h+d]=__float2bfloat16_rn(__bfloat162float(hidden[c*h+d])+update);
    }
}
