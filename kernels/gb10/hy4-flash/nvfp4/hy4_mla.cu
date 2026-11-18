// SPDX-License-Identifier: AGPL-3.0-only

// Reuse the generic Q-absorption and V-extraction GEMVs without changing the
// DeepSeek target. Its cache/sink attention ABIs do not match latent576 here.
#include "../../deepseek-v4-flash/nvfp4/mla_absorbed.cu"

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

extern "C" __global__ void hy4_cache_latent(const __nv_bfloat16* latent,
    __nv_bfloat16* row) {
    for (unsigned d=threadIdx.x;d<512;d+=256) row[d]=latent[d];
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

// One accepted token at a time; request-local page tables. Online softmax is
// initialized by the zero-value sink, which participates even with no real keys.
extern "C" __global__ void hy4_paged_mla(const __nv_bfloat16* q,
    const __nv_bfloat16* cache, const unsigned* table, const int* indices,
    const float* sink, __nv_bfloat16* out, unsigned position, unsigned page_size,
    unsigned columns, unsigned window, unsigned count) {
    unsigned h=blockIdx.x,tid=threadIdx.x;
    __shared__ float red[256],maximum,denominator,alpha,beta;
    if (tid==0) {maximum=sink[h];denominator=1.f;}
    float a=0.f,b=0.f;
    __syncthreads();
    unsigned begin=window && position+1>window?position+1-window:0;
    for (unsigned j=0;j<count;++j) {
        int logical=indices?indices[j]:(int)(begin+j);
        if (logical<0 || (unsigned)logical>position || (unsigned)logical<begin) continue;
        unsigned page=(unsigned)logical/page_size;
        unsigned physical=table[window?page%columns:page];
        const __nv_bfloat16* kv=cache+((size_t)physical*page_size+logical%page_size)*576;
        float dot=0.f;
        for (unsigned d=tid;d<576;d+=256) dot+=(float)q[(size_t)h*576+d]*(float)kv[d];
        red[tid]=dot; __syncthreads();
        for (unsigned s=128;s;s>>=1) {
            if (tid<s) red[tid]+=red[tid+s];
            __syncthreads();
        }
        if (tid==0) {
            float score=red[0]*(1.f/16.f),next=fmaxf(maximum,score);
            alpha=expf(maximum-next);beta=expf(score-next);
            denominator=denominator*alpha+beta;maximum=next;
        }
        __syncthreads();
        a=a*alpha+beta*(float)kv[tid];
        b=b*alpha+beta*(float)kv[tid+256];
        __syncthreads();
    }
    out[(size_t)h*512+tid]=__float2bfloat16_rn(a/denominator);
    out[(size_t)h*512+tid+256]=__float2bfloat16_rn(b/denominator);
}
