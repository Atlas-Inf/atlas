// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

template<unsigned KIND,bool ALIGNED>
__device__ __forceinline__ void hy4_batch_linear_impl(
    const __nv_bfloat16* x,const void* weights,const float* scales,void* out,
    unsigned n,unsigned k,unsigned rows,unsigned heads,unsigned xh,unsigned yh,
    unsigned xr,unsigned yr,unsigned wr,unsigned kind,unsigned float_out) {
    unsigned o=blockIdx.x*8+threadIdx.x/32,first=blockIdx.y*8,head=blockIdx.z;
    if(o>=n || head>=heads)return;
    constexpr unsigned width=KIND==1?16:8;
    unsigned lane=threadIdx.x%32;
    size_t wi=((size_t)head*n+o)*wr;
    float sums[8]={};
    for(unsigned col=lane*width;col<k;col+=32*width) {
        unsigned packed[8]={};
        const char* base=(const char*)weights+wi*(KIND==1?1:KIND==2?4:2);
        bool vector=ALIGNED || (col+width<=k && ((uintptr_t)(base+col*(KIND==1?1:KIND==2?4:2))%16)==0);
        if(vector) {
            const uint4 a=*(const uint4*)(base+col*(KIND==1?1:KIND==2?4:2));
            packed[0]=a.x;packed[1]=a.y;packed[2]=a.z;packed[3]=a.w;
            if(KIND==2) {
                const uint4 b=*(const uint4*)(base+col*4+16);
                packed[4]=b.x;packed[5]=b.y;packed[6]=b.z;packed[7]=b.w;
            }
        }
        float values[width];
        #pragma unroll
        for(unsigned j=0;j<width;j++) {
            float w;
            if(KIND==1) {
                __nv_fp8_e4m3 v;
                v.__x=vector?(packed[j/4]>>(8*(j%4)))&255:col+j<k?((const unsigned char*)weights)[wi+col+j]:0;
                w=(float)v*scales[(size_t)head*n+o];
            } else if(KIND==2) {
                w=vector?__uint_as_float(packed[j]):col+j<k?((const float*)weights)[wi+col+j]:0.f;
            } else {
                __nv_bfloat16 v;
                if(vector) {unsigned short b=(packed[j/2]>>(16*(j%2)))&65535;v=*(const __nv_bfloat16*)&b;}
                else v=col+j<k?((const __nv_bfloat16*)weights)[wi+col+j]:__float2bfloat16_rn(0.f);
                w=(float)v;
            }
            values[j]=w;
        }
        #pragma unroll
        for(unsigned r=0;r<8;r++)if(first+r<rows) {
            const __nv_bfloat16* input=x+(size_t)(first+r)*xr+(size_t)head*xh+col;
            bool xv=ALIGNED || (col+width<=k && (uintptr_t)input%16==0);
            unsigned data[8]={};
            if(xv) {
                uint4 a=*(const uint4*)input;
                data[0]=a.x;data[1]=a.y;data[2]=a.z;data[3]=a.w;
                if(width==16) {
                    uint4 b=*(const uint4*)(input+8);
                    data[4]=b.x;data[5]=b.y;data[6]=b.z;data[7]=b.w;
                }
            }
            #pragma unroll
            for(unsigned j=0;j<width;j++)if(ALIGNED || col+j<k) {
                unsigned short bits=(data[j/2]>>(16*(j%2)))&65535;
                float value=xv?(float)*(const __nv_bfloat16*)&bits:(float)input[j];
                sums[r]+=value*values[j];
            }
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

extern "C" __global__ void hy4_batch_linear(
    const __nv_bfloat16* x,const void* weights,const float* scales,void* out,
    unsigned n,unsigned k,unsigned rows,unsigned heads,unsigned xh,unsigned yh,
    unsigned xr,unsigned yr,unsigned wr,unsigned kind,unsigned float_out) {
    if(k%16==0 && wr%16==0 && xr%8==0 && xh%8==0 && (uintptr_t)x%16==0 && (uintptr_t)weights%16==0) {
        if(kind==1)hy4_batch_linear_impl<1,true>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
        else if(kind==2)hy4_batch_linear_impl<2,true>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
        else hy4_batch_linear_impl<0,true>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
    } else {
        if(kind==1)hy4_batch_linear_impl<1,false>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
        else if(kind==2)hy4_batch_linear_impl<2,false>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
        else hy4_batch_linear_impl<0,false>(x,weights,scales,out,n,k,rows,heads,xh,yh,xr,yr,wr,kind,float_out);
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

__device__ __forceinline__ void hy4_qdots(const __nv_bfloat16* x,Hy4Q4 q,unsigned o,unsigned k,
    const unsigned* entries,unsigned count,unsigned slots,bool activated,float* sums,const float* lut) {
    unsigned lane=threadIdx.x%32;
    #pragma unroll
    for(unsigned r=0;r<8;r++)sums[r]=0.f;
    if(count==1) {
        unsigned row=activated?entries[0]:entries[0]/slots;
        sums[0]=hy4_qdot(x+(size_t)row*k,q,o,k,lut);
        return;
    }
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
        #pragma unroll
        for(unsigned g=0;g<2;g++)if(g<count_k/16) {
            __nv_fp8_e4m3 sf;sf.__x=q.s[(size_t)o*(k/16)+col/16+g];
            float scale=(float)sf*q.scale;
            float values[16];
            #pragma unroll
            for(unsigned j=0;j<16;j++) {
                unsigned at=g*16+j;
                values[j]=lut[(words[at/8]>>(4*(at%8)))&15]*scale;
            }
            #pragma unroll
            for(unsigned r=0;r<8;r++)if(r<count) {
                unsigned entry=entries[r],row=activated?entry:entry/slots;
                const __nv_bfloat16* input=x+(size_t)row*k+col+g*16;
                uint4 lo=*(const uint4*)input,hi=*(const uint4*)(input+8);
                unsigned data[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
                #pragma unroll
                for(unsigned j=0;j<16;j++) {
                    unsigned short bits=(data[j/2]>>(16*(j%2)))&65535;
                    sums[r]+=values[j]*(float)*(const __nv_bfloat16*)&bits;
                }
            }
        }
    }
    #pragma unroll
    for(unsigned r=0;r<8;r++)sums[r]=hy4_warp_sum(sums[r]);
}

extern "C" __global__ __launch_bounds__(256,4) void hy4_batch_expert(const __nv_bfloat16* x,
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
    if(!table[(e+group)*3+(down?2:0)].w) {
        for(unsigned r=0;r<list[0];r++)if(threadIdx.x%32==0)
            out[(size_t)list[1+r]*n+o]=__float2bfloat16_rn(0.f);
        return;
    }
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
