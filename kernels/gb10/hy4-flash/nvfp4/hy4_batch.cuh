// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

extern "C" __global__ void hy4_batch_linear(
    const __nv_bfloat16* x,const void* weights,const float* scales,void* out,
    unsigned n,unsigned k,unsigned rows,unsigned heads,unsigned xh,unsigned yh,
    unsigned xr,unsigned yr,unsigned wr,unsigned kind,unsigned float_out) {
    unsigned o=blockIdx.x*8+threadIdx.x/32,first=blockIdx.y*8,head=blockIdx.z;
    if(o>=n || head>=heads)return;
    unsigned lane=threadIdx.x%32,width=kind==1?16:8;
    size_t wi=((size_t)head*n+o)*wr;
    float sums[8]={};
    for(unsigned col=lane*width;col<k;col+=32*width) {
        unsigned packed[8]={};
        const char* base=(const char*)weights+wi*(kind==1?1:kind==2?4:2);
        bool vector=col+width<=k && ((uintptr_t)(base+col*(kind==1?1:kind==2?4:2))%16)==0;
        if(vector) {
            const uint4 a=*(const uint4*)(base+col*(kind==1?1:kind==2?4:2));
            packed[0]=a.x;packed[1]=a.y;packed[2]=a.z;packed[3]=a.w;
            if(kind==2) {
                const uint4 b=*(const uint4*)(base+col*4+16);
                packed[4]=b.x;packed[5]=b.y;packed[6]=b.z;packed[7]=b.w;
            }
        }
        for(unsigned j=0;j<width && col+j<k;j++) {
            float w;
            if(kind==1) {
                __nv_fp8_e4m3 v;
                v.__x=vector?(packed[j/4]>>(8*(j%4)))&255:((const unsigned char*)weights)[wi+col+j];
                w=(float)v*scales[(size_t)head*n+o];
            } else if(kind==2) {
                w=vector?__uint_as_float(packed[j]):((const float*)weights)[wi+col+j];
            } else {
                __nv_bfloat16 v;
                if(vector) {unsigned short b=(packed[j/2]>>(16*(j%2)))&65535;v=*(const __nv_bfloat16*)&b;}
                else v=((const __nv_bfloat16*)weights)[wi+col+j];
                w=(float)v;
            }
            #pragma unroll
            for(unsigned r=0;r<8;r++)
                if(first+r<rows)sums[r]+=(float)x[(size_t)(first+r)*xr+(size_t)head*xh+col+j]*w;
        }
    }
    #pragma unroll
    for(unsigned r=0;r<8;r++) {
        float v=hy4_warp_sum(sums[r]);
        if(lane==0 && first+r<rows) {
            size_t at=(size_t)(first+r)*yr+(size_t)head*yh+o;
            if(float_out)((float*)out)[at]=v;
            else ((__nv_bfloat16*)out)[at]=__float2bfloat16_rn(v);
        }
    }
}

extern "C" __global__ void hy4_group_init(unsigned* groups,unsigned rows,unsigned experts) {
    if(threadIdx.x==0)groups[(size_t)experts*(rows+1)]=0;
}
extern "C" __global__ void hy4_group_rows(const unsigned* ids,unsigned* groups,
    unsigned rows,unsigned slots) {
    __shared__ unsigned prefix[256];
    unsigned t=threadIdx.x,e=blockIdx.x,slot=slots;
    if(t<rows)for(unsigned s=0;s<slots;s++)
        if(!ids || ids[(size_t)t*slots+s]==e){slot=s;break;}
    prefix[t]=slot<slots;__syncthreads();
    for(unsigned d=1;d<256;d*=2) {
        unsigned v=t>=d?prefix[t-d]:0;__syncthreads();
        prefix[t]+=v;__syncthreads();
    }
    unsigned* list=groups+(size_t)e*(rows+1);
    if(t==0) {
        list[0]=prefix[255];
        if(prefix[255]) {
            unsigned* active=groups+(size_t)gridDim.x*(rows+1);
            active[1+atomicAdd(active,1u)]=e;
        }
    }
    if(slot<slots)list[prefix[t]] = t*slots+slot;
}

__device__ void hy4_qdots(const __nv_bfloat16* x,Hy4Q4 q,unsigned o,unsigned k,
    const unsigned* entries,unsigned count,unsigned slots,bool activated,float* sums,const float* lut) {
    unsigned lane=threadIdx.x%32;
    #pragma unroll
    for(unsigned r=0;r<8;r++)sums[r]=0.f;
    if(q.w)for(unsigned col=lane*32;col<k;col+=1024) {
        unsigned count_k=min(32u,k-col);
        const unsigned char* p=q.w+(size_t)o*(k/2)+col/2;
        uint4 packed;
        if(count_k==32 && ((uintptr_t)p%16)==0)packed=*(const uint4*)p;
        else {
            unsigned words[4]={};
            for(unsigned b=0;b<count_k/2;b++)words[b/4]|=(unsigned)p[b]<<(8*(b%4));
            packed=make_uint4(words[0],words[1],words[2],words[3]);
        }
        unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
        for(unsigned g=0;g<count_k/16;g++) {
            __nv_fp8_e4m3 sf;sf.__x=q.s[(size_t)o*(k/16)+col/16+g];
            float scale=(float)sf*q.scale;
            #pragma unroll
            for(unsigned j=0;j<16;j++) {
                unsigned at=g*16+j;
                float w=lut[(words[at/8]>>(4*(at%8)))&15]*scale;
                #pragma unroll
                for(unsigned r=0;r<8;r++)if(r<count) {
                    unsigned entry=entries[r],row=activated?entry:entry/slots;
                    sums[r]+=w*(float)x[(size_t)row*k+col+at];
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<8;r++)sums[r]=hy4_warp_sum(sums[r]);
}

extern "C" __global__ void hy4_batch_expert(const __nv_bfloat16* x,
    const Hy4Q4* table,const unsigned* groups,__nv_bfloat16* out,
    unsigned n,unsigned k,unsigned rows,unsigned slots,unsigned group,
    unsigned down,float limit,unsigned experts) {
    const unsigned* active=groups+(size_t)experts*(rows+1);
    if(blockIdx.z>=active[0])return;
    unsigned e=active[1+blockIdx.z],o=blockIdx.x*8+threadIdx.x/32;
    __shared__ float lut[16];
    if(threadIdx.x<16)lut[threadIdx.x]=HY4_FP4[threadIdx.x];
    __syncthreads();
    const unsigned* list=groups+(size_t)e*(rows+1);
    if(o>=n)return;
    for(unsigned first=0;first<list[0];first+=8) {
    unsigned count=min(8u,list[0]-first);
    const unsigned* entries=list+1+first;
    float a[8],b[8];
    hy4_qdots(x,table[(e+group)*3+(down?2:0)],o,k,entries,count,slots,down,a,lut);
    if(!down)hy4_qdots(x,table[(e+group)*3+1],o,k,entries,count,slots,false,b,lut);
    if(threadIdx.x%32==0)for(unsigned r=0;r<count;r++) {
        float v=hy4_round(a[r]);
        if(!down) {
            float u=hy4_round(b[r]);
            if(limit>0.f){v=fminf(v,limit);u=fminf(limit,fmaxf(-limit,u));}
            v=v*hy4_sigmoid(v)*u;
        }
        out[(size_t)entries[r]*n+o]=__float2bfloat16_rn(v);
    }
    }
}

extern "C" __global__ void hy4_batch_combine(const __nv_bfloat16* partials,
    const float* weights,__nv_bfloat16* out,unsigned n,unsigned slots) {
    unsigned row=blockIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o<n) {
        float v=0.f;
        for(unsigned s=0;s<slots;s++)
            v+=(float)partials[((size_t)row*slots+s)*n+o]*(weights?weights[(size_t)row*slots+s]:1.f);
        out[(size_t)row*n+o]=__float2bfloat16_rn(v);
    }
}
