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
    unsigned group=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    if (group>=groups) return;
    unsigned physical=table[group/64],row=group%64;
    const uint8_t* key=pool+(size_t)physical*8448+row*128;
    float key_scale=*(const float*)(pool+(size_t)physical*8448+8192+row*4);
    __shared__ float sums[8];
    float sum=0.f;
    for (unsigned h=warp;h<heads;h+=8) {
        float dot=0.f;
        unsigned qv=*(const unsigned*)(query+h*128+lane*4);
        unsigned kv=*(const unsigned*)(key+lane*4);
        #pragma unroll
        for (unsigned d=0;d<4;++d) {
            __nv_fp8_e4m3 q,k;q.__x=(qv>>(8*d))&255;k.__x=(kv>>(8*d))&255;
            dot+=(float)q*(float)k;
        }
        for (unsigned offset=16;offset;offset>>=1) dot+=__shfl_down_sync(0xffffffff,dot,offset);
        if (lane==0) sum+=(float)weights[h]*q_scale[h]*key_scale*fmaxf(0.f,dot);
    }
    if (lane==0) sums[warp]=sum;
    __syncthreads();
    if (tid==0) {
        float total=0.f;
        for (unsigned i=0;i<8;i++) total+=sums[i];
        scores[group]=total*rsqrtf((float)heads*128.f);
    }
}

// Batched key norm for one index tile: block r RMSNorms key row r into the
// rolling tail slot ((len_base+r)%4) and into the per-tile snapshot
// tile_tail[(r+3)*H] (three-row headroom preloaded from the tail) consumed by
// hy4_pool_fwht_batch. Math identical to the per-row hy4_norm_add(NULL,
// key+r*H, w, tail+slot*H, H, eps) launches it replaces.
extern "C" __global__ void hy4_tail_norm_batch(const __nv_bfloat16* shared,
    const __nv_bfloat16* norm, __nv_bfloat16* tail, __nv_bfloat16* tile_tail,
    unsigned len_base, unsigned H, float eps) {
    unsigned t=blockIdx.x, tid=threadIdx.x;
    __shared__ float red[256], r;
    float ss=0.f;
    for (unsigned d=tid;d<H;d+=256) {
        float v=(float)shared[(size_t)t*H+d];
        ss+=v*v;
    }
    red[tid]=ss; __syncthreads();
    for (unsigned s=128;s;s>>=1) {
        if (tid<s) red[tid]+=red[tid+s];
        __syncthreads();
    }
    if (tid==0) r=rsqrtf(red[0]/H+eps);
    __syncthreads();
    unsigned slot=(len_base+t)%4;
    for (unsigned d=tid;d<H;d+=256) {
        float v=(float)shared[(size_t)t*H+d];
        __nv_bfloat16 o=__float2bfloat16_rn(v*r*(float)norm[d]);
        tail[(size_t)slot*H+d]=o;
        tile_tail[(size_t)(t+3)*H+d]=o;
    }
}

// Batched pool + rotary + FWHT for one index tile: block g handles the group
// ending at tile row r = r0 + 4*g, where (len_base+r)%4==3. Fuses the three
// per-group launches (hy4_pool, hy4_rotary on the pooled vector with layout
// [128,128,64,64] and split=1, hy4_fwht into the pool page). tile_tail holds
// the normalized keys of rows r-3..r at tile_tail[(r+j)*128]. Position of the
// group is [base0+r-3; 3] because positions are [p; 3] per row.
extern "C" __global__ void hy4_pool_fwht_batch(const __nv_bfloat16* tile_tail,
    const float* wv, const float* wg, const float* ape, const __nv_bfloat16* norm,
    uint8_t* pool, unsigned len_base, unsigned r0, unsigned rows,
    unsigned long long base0, float theta, float eps) {
    unsigned d=threadIdx.x;
    unsigned r=r0+4*blockIdx.x;
    if (r>=rows) return;
    __shared__ float values[128],squares[128],x[128],maximum[128],inv,scale;
    __shared__ __nv_bfloat16 pooled[128];
    float v[4]={},g[4]={},mx=-INFINITY;
    for (unsigned j=0;j<4;++j) {
        for (unsigned k=0;k<128;++k) {
            float t=(float)tile_tail[(size_t)(r+j)*128+k];
            v[j]+=wv[d*128+k]*t;g[j]+=wg[d*128+k]*t;
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
    __syncthreads();
    // RoPE on the rope half [64,128), in place, identical to hy4_rotary with
    // head 0, strides/offsets [128,128,64,64], split=1 and pt=ph=pw=p.
    float a=0.f,b=0.f,s=0.f,c=0.f;
    if (d<32) {
        float p=(float)((long long)base0+(long long)r-3);
        float angle=p*powf(theta,-(float)d/32.f);
        sincosf(angle,&s,&c);
        a=(float)pooled[64+d*2];b=(float)pooled[64+d*2+1];
    }
    __syncthreads();
    if (d<32) {
        pooled[64+d]=__float2bfloat16_rn(a*c-b*s);
        pooled[96+d]=__float2bfloat16_rn(a*s+b*c);
    }
    __syncthreads();
    // FWHT + FP8 quantize into the pool page, identical to hy4_fwht head 0.
    x[d]=(float)pooled[d];__syncthreads();
    for (unsigned st=1;st<128;st*=2) {
        float ta=x[d],tb=x[d^st];__syncthreads();
        x[d]=(d&st)?tb-ta:ta+tb;__syncthreads();
    }
    float value=x[d]*rsqrtf(128.f);
    maximum[d]=fabsf(value);__syncthreads();
    for (unsigned st=64;st;st>>=1) {
        if (d<st) maximum[d]=fmaxf(maximum[d],maximum[d+st]);
        __syncthreads();
    }
    unsigned group=(len_base+r)/4,page=group/64,rowg=group%64;
    if (d==0) {
        scale=exp2f(ceilf(log2f(fmaxf(maximum[0],1e-4f)/448.f)));
        *(float*)(pool+(size_t)page*8448+8192+rowg*4)=scale;
    }
    __syncthreads();
    __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,value/scale)));
    pool[(size_t)page*8448+rowg*128+d]=q.__x;
}

// The trained budget is 511 historical groups plus the current four-token tail.
extern "C" __global__ void hy4_index_select(const float* scores, int* output,
    unsigned groups, unsigned position) {
    unsigned tid=threadIdx.x;
    __shared__ unsigned prefix[256],base;
    if (tid==0) base=0;
    __syncthreads();
    if (groups<=511) {
        for (unsigned g=tid;g<groups;g+=256)
            for (unsigned j=0;j<4;j++) output[g*4+j]=(int)(g*4+j);
    } else {
        for (unsigned batch=0;batch<groups;batch+=256) {
            unsigned g=batch+tid,rank=0;
            if (g<groups) {
                float value=scores[g];
                for (unsigned other=0;other<groups;other++)
                    rank+=scores[other]>value || (scores[other]==value && other<g);
            }
            unsigned keep=g<groups && rank<511;
            prefix[tid]=keep;
            __syncthreads();
            for (unsigned stride=1;stride<256;stride*=2) {
                unsigned add=tid>=stride?prefix[tid-stride]:0;
                __syncthreads();
                prefix[tid]+=add;
                __syncthreads();
            }
            if (keep) for (unsigned j=0;j<4;j++)
                output[(base+prefix[tid]-1)*4+j]=(int)(g*4+j);
            __syncthreads();
            if (tid==0) base+=prefix[255];
            __syncthreads();
        }
    }
    unsigned historical=min(groups,511u)*4,tail=position%4+1;
    if (tid<tail) output[historical+tid]=(int)(position/4*4+tid);
    for (unsigned i=historical+tail+tid;i<2048;i+=256) output[i]=-1;
}

// Batched FWHT: one (head, row) per block, identical math to hy4_fwht.
extern "C" __global__ void hy4_fwht_batch(const __nv_bfloat16* input, uint8_t* output,
    float* scales, unsigned heads, unsigned rows) {
    unsigned h=blockIdx.x,r=blockIdx.y,d=threadIdx.x;
    if (h>=heads||r>=rows) return;
    size_t base=(size_t)r*heads*128;
    __shared__ float x[128],maximum[128],scale;
    x[d]=(float)input[base+h*128+d];__syncthreads();
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
    if (d==0) {scale=exp2f(ceilf(log2f(fmaxf(maximum[0],1e-4f)/448.f)));scales[(size_t)r*heads+h]=scale;}
    __syncthreads();
    __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,value/scale)));
    output[base+h*128+d]=q.__x;
}

// Batched index scoring: block (g, row) computes one group score for one row.
// Identical math to hy4_index_scores; row r uses len = groups_base + r.
extern "C" __global__ void hy4_index_scores_batch(const uint8_t* query,
    const float* q_scale, const __nv_bfloat16* weights, const uint8_t* pool,
    const unsigned* table, float* scores, unsigned groups_base, unsigned heads,
    unsigned max_groups, unsigned rows) {
    unsigned g=blockIdx.x,r=blockIdx.y;
    if (r>=rows) return;
    unsigned groups=(groups_base+r)/4;
    if (g>=groups) return;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    size_t rbase=(size_t)r*heads;
    const uint8_t* q=query+rbase*128;
    const float* qs=q_scale+rbase;
    const __nv_bfloat16* hw=weights+rbase;
    unsigned physical=table[g/64],rowg=g%64;
    const uint8_t* key=pool+(size_t)physical*8448+rowg*128;
    float key_scale=*(const float*)(pool+(size_t)physical*8448+8192+rowg*4);
    __shared__ float sums[8];
    float sum=0.f;
    for (unsigned h=warp;h<heads;h+=8) {
        float dot=0.f;
        unsigned qv=*(const unsigned*)(q+h*128+lane*4);
        unsigned kv=*(const unsigned*)(key+lane*4);
        #pragma unroll
        for (unsigned d=0;d<4;++d) {
            __nv_fp8_e4m3 qq,k;qq.__x=(qv>>(8*d))&255;k.__x=(kv>>(8*d))&255;
            dot+=(float)qq*(float)k;
        }
        for (unsigned offset=16;offset;offset>>=1) dot+=__shfl_down_sync(0xffffffff,dot,offset);
        if (lane==0) sum+=(float)hw[h]*qs[h]*key_scale*fmaxf(0.f,dot);
    }
    if (lane==0) sums[warp]=sum;
    __syncthreads();
    if (tid==0) {
        float total=0.f;
        for (unsigned i=0;i<8;i++) total+=sums[i];
        scores[(size_t)r*max_groups+g]=total*rsqrtf((float)heads*128.f);
    }
}

// Batched top-K selection: one 256-thread block per row. Output is identical to
// hy4_index_select (kept set = top 511 groups by score with ties broken toward
// the lower group id; kept groups are written in ascending group-id order, each
// emitting four consecutive token ids), but computed in O(groups log groups):
// threshold bisection plus one ordered compaction sweep, instead of the
// O(groups^2) rank scan of the serial kernel.
extern "C" __global__ void hy4_index_select_batch(const float* scores, int* output,
    unsigned groups_base, unsigned position_base, unsigned max_groups, unsigned rows) {
    unsigned row=blockIdx.x;
    if (row>=rows) return;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    unsigned len=groups_base+row;
    unsigned groups=len/4;
    unsigned position=position_base+row;
    const float* sc=scores+(size_t)row*max_groups;
    int* out=output+(size_t)row*2048;
    __shared__ unsigned prefix[256];
    __shared__ float smin,smax,sT;
    __shared__ unsigned s_total,s_wbase,s_ewbase;
    __shared__ float red[8];
    if (groups<=511) {
        for (unsigned g=tid;g<groups;g+=256)
            for (unsigned j=0;j<4;++j) out[g*4+j]=(int)(g*4+j);
    } else {
        float lmin=INFINITY,lmax=-INFINITY;
        for (unsigned g=tid;g<groups;g+=256) {float v=sc[g];lmin=fminf(lmin,v);lmax=fmaxf(lmax,v);}
        for (unsigned off=16;off;off>>=1) {
            lmin=fminf(lmin,__shfl_down_sync(0xffffffff,lmin,off));
            lmax=fmaxf(lmax,__shfl_down_sync(0xffffffff,lmax,off));
        }
        if (lane==0) red[warp]=lmin;
        __syncthreads();
        if (tid<8) {
            float v=red[tid];
            for (unsigned off=4;off;off>>=1) v=fminf(v,__shfl_down_sync(0xff,v,off));
            if (tid==0) smin=v;
        }
        __syncthreads();
        if (lane==0) red[warp]=lmax;
        __syncthreads();
        if (tid<8) {
            float v=red[tid];
            for (unsigned off=4;off;off>>=1) v=fmaxf(v,__shfl_down_sync(0xff,v,off));
            if (tid==0) smax=v;
        }
        __syncthreads();
        float lo=smin-1.f,hi=smax;
        for (unsigned it=0;it<64;++it) {
            float mid=0.5f*(lo+hi);
            unsigned c=0;
            for (unsigned g=tid;g<groups;g+=256) c+=(sc[g]>mid)?1u:0u;
            for (unsigned off=16;off;off>>=1) c+=__shfl_down_sync(0xffffffff,c,off);
            if (lane==0) prefix[warp]=c;
            __syncthreads();
            if (tid<8) {
                unsigned v=prefix[tid];
                for (unsigned off=4;off;off>>=1) v+=__shfl_down_sync(0xff,v,off);
                if (tid==0) s_total=v;
            }
            __syncthreads();
            if (s_total>511u) lo=mid; else hi=mid;
        }
        if (tid==0) sT=hi;
        __syncthreads();
        float T=sT;
        unsigned cnt=0;
        for (unsigned g=tid;g<groups;g+=256) cnt+=(sc[g]>T)?1u:0u;
        prefix[tid]=cnt;
        __syncthreads();
        for (unsigned stride=1;stride<256;stride*=2) {
            unsigned add=tid>=stride?prefix[tid-stride]:0;
            __syncthreads();
            prefix[tid]+=add;
            __syncthreads();
        }
        unsigned total_gt=prefix[255];
        __syncthreads();
        unsigned need=511u-total_gt;
        if (tid==0) {s_wbase=0;s_ewbase=0;}
        __syncthreads();
        for (unsigned k=0;k<groups;k+=256) {
            unsigned g=k+tid;
            bool in=g<groups;
            bool gt=in&&sc[g]>T;
            bool eq=in&&sc[g]==T;
            unsigned eqrank=0,eqtotal=0,keep=0;
            // scan eq bits first to know the tie order within this slab
            prefix[tid]=eq?1u:0u;
            __syncthreads();
            for (unsigned stride=1;stride<256;stride*=2) {
                unsigned add=tid>=stride?prefix[tid-stride]:0;
                __syncthreads();
                prefix[tid]+=add;
                __syncthreads();
            }
            eqrank=prefix[tid]-(eq?1u:0u);
            eqtotal=prefix[255];
            __syncthreads();
            keep=(gt||(eq&&s_ewbase+eqrank<need))?1u:0u;
            prefix[tid]=keep;
            __syncthreads();
            for (unsigned stride=1;stride<256;stride*=2) {
                unsigned add=tid>=stride?prefix[tid-stride]:0;
                __syncthreads();
                prefix[tid]+=add;
                __syncthreads();
            }
            unsigned rank=prefix[tid]-keep;
            unsigned ktotal=prefix[255];
            if (keep) {
                unsigned pos=s_wbase+rank;
                for (unsigned j=0;j<4;++j) out[pos*4+j]=(int)(g*4+j);
            }
            __syncthreads();
            if (tid==0) {s_wbase+=ktotal;s_ewbase+=eqtotal;}
            __syncthreads();
        }
    }
    unsigned historical=min(groups,511u)*4,tail=position%4+1;
    if (tid<tail) out[historical+tid]=(int)(position/4*4+tid);
    for (unsigned i=historical+tail+tid;i<2048;i+=256) out[i]=-1;
}
