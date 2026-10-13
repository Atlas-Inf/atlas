// SPDX-License-Identifier: AGPL-3.0-only
// GLM-specific M5 BF16 router; production compiles with --fmad=false.
#ifdef __CUDACC__
#define ROUTER_HD __host__ __device__
#else
#define ROUTER_HD
#endif
namespace glm_router_bn4 {
constexpr unsigned rows=5, columns=288, width=4096, bn=4, bk=64, threads=32;
ROUTER_HD constexpr bool owns_output(unsigned lane) { return lane < rows*bn; }
ROUTER_HD constexpr bool geometry(unsigned m,unsigned n,unsigned k,unsigned x,unsigned y,unsigned z) {
    return m==rows && n==columns && k==width && x==threads && y==1 && z==1;
}
ROUTER_HD constexpr unsigned row(unsigned lane) { return lane / bn; }
ROUTER_HD constexpr unsigned column(unsigned lane) { return lane % bn; }
ROUTER_HD constexpr unsigned load_row(unsigned chunk) { return chunk / (bk/4); }
ROUTER_HD constexpr unsigned load_k(unsigned chunk) { return (chunk % (bk/4))*4; }
}
#undef ROUTER_HD
#ifdef __CUDACC__
#include <cuda_bf16.h>
extern "C" __global__ void glm_router_m5_bn4(
    const __nv_bfloat16* __restrict__ a,const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,unsigned m,unsigned n,unsigned k) {
    using namespace glm_router_bn4;
    if(!geometry(m,n,k,blockDim.x,blockDim.y,blockDim.z)
        || gridDim.x!=columns/bn || gridDim.y!=1 || gridDim.z!=1) return;
    __shared__ float sa[rows][bk+1],sb[bn][bk+1];
    const unsigned lane=threadIdx.x;
    float acc=0.0f;
    for(unsigned kb=0;kb<k;kb+=bk) {
        for(unsigned chunk=lane;chunk<rows*bk/4;chunk+=threads) {
            const unsigned ar=load_row(chunk),ak=load_k(chunk);
            const ushort4 v=*(const ushort4*)(a+size_t(ar)*k+kb+ak);
            sa[ar][ak]=__bfloat162float(__ushort_as_bfloat16(v.x));
            sa[ar][ak+1]=__bfloat162float(__ushort_as_bfloat16(v.y));
            sa[ar][ak+2]=__bfloat162float(__ushort_as_bfloat16(v.z));
            sa[ar][ak+3]=__bfloat162float(__ushort_as_bfloat16(v.w));
        }
        for(unsigned chunk=lane;chunk<bn*bk/4;chunk+=threads) {
            const unsigned br=load_row(chunk),bk0=load_k(chunk);
            const ushort4 v=*(const ushort4*)(b+size_t(blockIdx.x*bn+br)*k+kb+bk0);
            sb[br][bk0]=__bfloat162float(__ushort_as_bfloat16(v.x));
            sb[br][bk0+1]=__bfloat162float(__ushort_as_bfloat16(v.y));
            sb[br][bk0+2]=__bfloat162float(__ushort_as_bfloat16(v.z));
            sb[br][bk0+3]=__bfloat162float(__ushort_as_bfloat16(v.w));
        }
        // All32 lanes participate, including the twelve load-only lanes.
        __syncwarp(0xffffffffu);
        if(owns_output(lane)) {
            #pragma unroll 8
            for(unsigned kk=0;kk<bk;++kk) acc+=sa[row(lane)][kk]*sb[column(lane)][kk];
        }
        __syncwarp(0xffffffffu);
    }
    if(owns_output(lane)) out[row(lane)*n+blockIdx.x*bn+column(lane)]=__float2bfloat16(acc);
}
#endif
