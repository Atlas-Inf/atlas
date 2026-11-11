// SPDX-License-Identifier: AGPL-3.0-only
// BF16-input B-tile SSOT; include after existing scalar primitives.
// Both register variants are exported; no serving selection is enabled.
#pragma once

#define BTREG_ARGS \
 const __nv_bfloat16* __restrict__ A, \
 const unsigned long long* __restrict__ gp,const unsigned long long* __restrict__ gs,const float* __restrict__ gf,__nv_bfloat16* __restrict__ go, \
 const unsigned long long* __restrict__ up,const unsigned long long* __restrict__ us,const float* __restrict__ uf,__nv_bfloat16* __restrict__ uo, \
 const unsigned int* __restrict__ ids, \
 const unsigned char* __restrict__ sgp,const unsigned char* __restrict__ sgs,float sgf,__nv_bfloat16* __restrict__ sgo, \
 const unsigned char* __restrict__ sup,const unsigned char* __restrict__ sus,float suf,__nv_bfloat16* __restrict__ suo, \
 unsigned int N,unsigned int K,unsigned int top_k
#define BTREG_PASS A,gp,gs,gf,go,up,us,uf,uo,ids,sgp,sgs,sgf,sgo,sup,sus,suf,suo,N,K,top_k

template<unsigned ROWS,bool VEC>
__device__ __forceinline__ void btile_decode_register_impl(BTREG_ARGS){
    // Host requires exact N2048/K4096; retain dynamic arithmetic dimensions.
    if(blockDim.x!=32||blockDim.y!=1||blockDim.z!=1||N%128||K%64||!top_k)return;
    const unsigned y=blockIdx.y,proj=blockIdx.z;
    if(y>=ROWS*(top_k+1)||proj>=2)return;
    const bool shared=y>=ROWS*top_k;
    const unsigned token=shared?y-ROWS*top_k:y/top_k;
    const unsigned n=blockIdx.x*32+threadIdx.x;
    if(n>=N)return; // N multiple128 makes the complete CTA inactive.
    const __nv_bfloat16* A_token=A+(unsigned long long)token*K;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;__nv_bfloat16* C;unsigned long long c_offset;
    if(shared){
        B_packed=proj?sup:sgp;B_scale=proj?sus:sgs;s2=proj?suf:sgf;
        C=proj?suo:sgo;c_offset=(unsigned long long)token*N;
    }else{
        const unsigned expert=ids[y];
        B_packed=(const unsigned char*)(proj?up[expert]:gp[expert]);
        B_scale=(const unsigned char*)(proj?us[expert]:gs[expert]);
        s2=proj?uf[expert]:gf[expert];C=proj?uo:go;c_offset=(unsigned long long)y*N;
    }
    if(!B_packed){C[c_offset+n]=__float2bfloat16(0.0f);return;}
    __shared__ float btreg_lut[16];
    if(threadIdx.x<16)btreg_lut[threadIdx.x]=E2M1_LUT_T[threadIdx.x];
    __syncthreads();
    const unsigned num_groups=K/16;
    // Entire shared loop is separate: no tiled-address predicate inside it.
    if(shared){
        float acc=0.0f;
        for(unsigned sg=0;sg<num_groups;++sg){
            unsigned char sb=B_scale[(unsigned long long)sg*N+n];
            float sc=atlas_dec_e4m3(sb)*s2;
            const unsigned kh_base=sg*8;
            #pragma unroll
            for(unsigned kh_off=0;kh_off<8;++kh_off){
                unsigned k_half=kh_base+kh_off;
                unsigned char byte=B_packed[(unsigned long long)k_half*N+n];
                float a_lo=__bfloat162float(A_token[k_half*2]);
                float a_hi=__bfloat162float(A_token[k_half*2+1]);
                float w_lo=btreg_lut[byte&0xFu]*sc;
                float w_hi=btreg_lut[(byte>>4)&0xFu]*sc;
                acc+=a_lo*w_lo+a_hi*w_hi;
            }
        }
        C[c_offset+n]=__float2bfloat16(acc);
        return;
    }
    float acc=0.0f;
    // Macro retains the exact original group scale and ordered eight updates.
    #define BTREG_ACCUM(SG,LO,HI) do { \
        unsigned char sb=B_scale[(unsigned long long)(SG)*N+n]; \
        float sc=atlas_dec_e4m3(sb)*s2; \
        const unsigned kh_base=(SG)*8; \
        _Pragma("unroll") \
        for(unsigned kh_off=0;kh_off<8;++kh_off){ \
            unsigned k_half=kh_base+kh_off; \
            unsigned char byte=(unsigned char)(((kh_off<4)?(LO):(HI))>>(8*(kh_off%4))); \
            float a_lo=__bfloat162float(A_token[k_half*2]); \
            float a_hi=__bfloat162float(A_token[k_half*2+1]); \
            float w_lo=btreg_lut[byte&0xFu]*sc; \
            float w_hi=btreg_lut[(byte>>4)&0xFu]*sc; \
            acc+=a_lo*w_lo+a_hi*w_hi; \
        } \
    } while(0)
    const unsigned long long n_base=(unsigned long long)(n/128)*(K/64)*4096+(n%128)*32;
    if constexpr(!VEC){
        for(unsigned sg=0;sg<num_groups;++sg){
            const unsigned long long offset=n_base+(unsigned long long)(sg/4)*4096+(sg%4)*8;
            const uint2 packed=*(const uint2*)(B_packed+offset);
            BTREG_ACCUM(sg,packed.x,packed.y);
        }
    }else{
        for(unsigned sg_base=0;sg_base<num_groups;sg_base+=4){
            const unsigned long long offset=n_base+(unsigned long long)(sg_base/4)*4096;
            const uint4 low=*(const uint4*)(B_packed+offset);
            const uint4 high=*(const uint4*)(B_packed+offset+16);
            // Literal group order keeps eight registers directly addressed.
            BTREG_ACCUM(sg_base,low.x,low.y);
            BTREG_ACCUM(sg_base+1,low.z,low.w);
            BTREG_ACCUM(sg_base+2,high.x,high.y);
            BTREG_ACCUM(sg_base+3,high.z,high.w);
        }
    }
    #undef BTREG_ACCUM
    C[c_offset+n]=__float2bfloat16(acc);
}

#define BTREG_EXPORT(ROWS) \
extern "C" __global__ void glm_btile_decode_word##ROWS(BTREG_ARGS){btile_decode_register_impl<ROWS,false>(BTREG_PASS);} \
extern "C" __global__ void glm_btile_decode_vec##ROWS(BTREG_ARGS){btile_decode_register_impl<ROWS,true>(BTREG_PASS);}
BTREG_EXPORT(1)
BTREG_EXPORT(2)
BTREG_EXPORT(3)
#undef BTREG_EXPORT
#undef BTREG_PASS
#undef BTREG_ARGS
