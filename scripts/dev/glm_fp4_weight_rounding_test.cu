// SPDX-License-Identifier: AGPL-3.0-only
// Standalone GPU regression: CPU nearest-value oracle, bounded BF16
// bit-pattern coverage, and explicit midpoint/underflow witnesses.
// Compile with --fmad=false. Pass "legacy" to demonstrate the old failure.
#include "quantize_bf16_to_nvfp4.cu"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
float decode8(unsigned i) {
    unsigned e=i>>3,m=i&7;
    return e?std::ldexp(float(8+m),int(e)-10):std::ldexp(float(m),-9);
}
template<size_t N> unsigned nearest(float x,const std::array<float,N>& levels) {
    auto it=std::lower_bound(levels.begin(),levels.end(),x);
    if(it==levels.end())return N-1;
    unsigned hi=it-levels.begin();if(!hi)return 0;
    unsigned lo=hi-1;float dl=x-levels[lo],dh=levels[hi]-x;
    return dl<dh?lo:dh<dl?hi:(lo&1?hi:lo);
}
int main(int argc,char**argv) {
    bool legacy=argc>1 && std::string(argv[1])=="legacy";
    const unsigned n=4096,k=128;std::vector<__nv_bfloat16> a(n*k);
    // Repeat the complete BF16 finite range, then constrain very large
    // values to keep group reductions finite under all tested scale2s.
    for(unsigned i=0;i<a.size();i++){
        unsigned short bits=(i%65280); // 0..0xfeff excludes infinities/NaNs except +inf gap below
        if((bits&0x7f80)==0x7f80)bits=0;
        std::memcpy(&a[i],&bits,2);
        float v=__bfloat162float(a[i]);
        if(std::fabs(v)>1000000.f)a[i]=__float2bfloat16(std::copysign(1000000.f,v));
    }
    const float witnesses[5][4]={{6,.75f,1.75f,3.5f},{6.375f,1,2,3},
        {.0908203125f,.05f,.01f,0},{.0087890625f,.004f,.001f,0},{0,0,0,0}};
    for(unsigned g=0;g<5;g++)for(unsigned j=0;j<16;j++)
        a[g*16+j]=__float2bfloat16(witnesses[g][j%4]*(j<8?1.f:-1.f));
    __nv_bfloat16* da;unsigned char *dw,*ds;CK(cudaMalloc(&da,a.size()*2));
    CK(cudaMalloc(&dw,a.size()/2));CK(cudaMalloc(&ds,a.size()/16));
    CK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));
    std::array<float,127> l8;for(unsigned i=0;i<127;i++)l8[i]=decode8(i);
    const std::array<float,8> l4={0,.5f,1,1.5f,2,3,4,6};
    uint64_t wb=0,sb=0,checked=0;std::vector<unsigned char>w(a.size()/2),s(a.size()/16);
    for(float scale2:{1.f,0.0003f,100.f}) {
        if(legacy)quantize_bf16_to_nvfp4<<<n,256>>>(da,dw,ds,scale2,n,k);
        else quantize_bf16_to_nvfp4_weight_rne<<<n,256>>>(da,dw,ds,scale2,n,k);
        CK(cudaMemcpy(w.data(),dw,w.size(),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.data(),ds,s.size(),cudaMemcpyDeviceToHost));
        float inv2=1.f/scale2;
        for(size_t g=0;g<a.size()/16;g++){
            float gm=0;for(unsigned j=0;j<16;j++)gm=std::max(gm,std::fabs(__bfloat162float(a[g*16+j])));
            unsigned scale=nearest(gm>0?gm*inv2/6.f:0.f,l8);
            sb+=s[g]!=scale;float eff=l8[scale]*scale2,inv=eff>0?1.f/eff:0.f;
            for(unsigned j=0;j<16;j++){
                float x=__bfloat162float(a[g*16+j])*inv;
                unsigned expected=nearest(std::fabs(x),l4)|(std::signbit(x)?8:0);
                unsigned got=(w[g*8+j/2]>>(4*(j&1)))&15;
                if(got!=expected && wb<4)fprintf(stderr,"g=%zu j=%u scale2=%g x=%g got=%u expected=%u\n",g,j,scale2,x,got,expected);
                wb+=got!=expected;++checked;
            }
        }
    }
    printf("{\"legacy\":%s,\"weight_values\":%llu,\"weight_differences\":%llu,\"scale_differences\":%llu}\n",
        legacy?"true":"false",(unsigned long long)checked,(unsigned long long)wb,(unsigned long long)sb);
    return wb||sb?1:0;
}
