// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void hy4_tail_write(const __nv_bfloat16* key,
    __nv_bfloat16* tail, unsigned member) {
    tail[member*128+threadIdx.x]=key[threadIdx.x];
}

// Values and gates are FP32 projections of four normalized BF16 token keys.
extern "C" __global__ void hy4_pool(const __nv_bfloat16* tail,
    const float* wv, const float* wg, const float* ape, const __nv_bfloat16* norm,
    __nv_bfloat16* pooled, float eps) {
    unsigned d=threadIdx.x;
    __shared__ float values[128],squares[128],inv;
    float v[4]={},g[4]={},mx=-INFINITY;
    for (unsigned j=0;j<4;++j) {
        for (unsigned k=0;k<128;++k) {
            float x=(float)tail[j*128+k];
            v[j]+=wv[d*128+k]*x;g[j]+=wg[d*128+k]*x;
        }
        g[j]+=ape[j*128+d];mx=fmaxf(mx,g[j]);
    }
    float sum=0.f,result=0.f;
    for (unsigned j=0;j<4;++j) {float p=expf(g[j]-mx);sum+=p;result+=p*v[j];}
    result/=sum;values[d]=result;squares[d]=result*result;
    __syncthreads();
    for (unsigned s=64;s;s>>=1) {
        if (d<s) squares[d]+=squares[d+s];
        __syncthreads();
    }
    if (d==0) inv=rsqrtf(squares[0]/128.f+eps);
    __syncthreads();
    pooled[d]=__float2bfloat16_rn(values[d]*inv*(float)norm[d]);
}

extern "C" __global__ void hy4_fwht(const __nv_bfloat16* input, uint8_t* output,
    float* scales) {
    unsigned h=blockIdx.x,d=threadIdx.x;
    __shared__ float x[128],maximum[128],scale;
    x[d]=(float)input[h*128+d];__syncthreads();
    for (unsigned s=1;s<128;s*=2) {
        float a=x[d],b=x[d^s];__syncthreads();
        x[d]=(d&s)?b-a:a+b;__syncthreads();
    }
    float value=x[d]*rsqrtf(128.f);
    maximum[d]=fabsf(value);__syncthreads();
    for (unsigned s=64;s;s>>=1) {
        if (d<s) maximum[d]=fmaxf(maximum[d],maximum[d+s]);
        __syncthreads();
    }
    if (d==0) {scale=exp2f(ceilf(log2f(fmaxf(maximum[0],1e-4f)/448.f)));scales[h]=scale;}
    __syncthreads();
    __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,value/scale)));
    output[h*128+d]=q.__x;
}

// Pool page ABI: 64 x 128 E4M3 bytes followed by 64 FP32 scales.
extern "C" __global__ void hy4_index_scores(const uint8_t* query,
    const float* q_scale, const __nv_bfloat16* weights, const uint8_t* pool,
    const unsigned* table, float* scores, unsigned groups, unsigned heads) {
    unsigned group=blockIdx.x*128+threadIdx.x;
    if (group>=groups) return;
    unsigned physical=table[group/64],row=group%64;
    const uint8_t* key=pool+(size_t)physical*8448+row*128;
    float key_scale=*(const float*)(pool+(size_t)physical*8448+8192+row*4);
    float sum=0.f;
    for (unsigned h=0;h<heads;++h) {
        float dot=0.f;
        for (unsigned d=0;d<128;++d) {
            __nv_fp8_e4m3 q,k;q.__x=query[h*128+d];k.__x=key[d];
            dot+=(float)q*(float)k;
        }
        sum+=(float)weights[h]*q_scale[h]*key_scale*fmaxf(0.f,dot);
    }
    scores[group]=sum*rsqrtf((float)heads*128.f);
}

// The trained budget is 511 historical groups plus the current four-token tail.
extern "C" __global__ void hy4_index_select(const float* scores, int* output,
    unsigned groups, unsigned position) {
    float best[511];int ids[511];
    for (unsigned j=0;j<511;++j) {best[j]=-INFINITY;ids[j]=-1;}
    for (unsigned g=0;g<groups;++g) {
        float s=scores[g];unsigned at=511;
        for (unsigned j=0;j<511;++j)
            if (ids[j]<0 || s>best[j]) {at=j;break;}
        if (at<511) {
            for (unsigned j=510;j>at;--j) {best[j]=best[j-1];ids[j]=ids[j-1];}
            best[at]=s;ids[at]=(int)g;
        }
    }
    unsigned n=min(groups,511u);
    for (unsigned i=0;i<n;++i) for (unsigned j=i+1;j<n;++j)
        if (ids[j]<ids[i]) {int tmp=ids[i];ids[i]=ids[j];ids[j]=tmp;}
    unsigned count=0;
    for (unsigned i=0;i<n;++i) for (unsigned j=0;j<4;++j) output[count++]=ids[i]*4+j;
    for (unsigned j=position/4*4;j<=position;++j) output[count++]=(int)j;
    while (count<2048) output[count++]=-1;
}
