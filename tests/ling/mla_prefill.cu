// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -std=c++17 -O2 --fmad=false -gencode arch=compute_121a,code=sm_121a tests/ling/mla_prefill.cu -o /tmp/ling-mla
#include "../../kernels/gb10/ling-3.2-air/nvfp4/paged_decode_attn_ling.cu"
#undef paged_decode_attn
#undef HDIM
#undef BC
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_mla_prefill.cu"
#undef HDIM
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr,"%s\n", cudaGetErrorString(result));
        std::exit(2);
    }
}
template<typename T> static T* allocate(size_t count) {
    T* result;
    check(cudaMallocManaged(&result,count*sizeof(T)));
    return result;
}
int main(int argc, char** argv) {
    const int past = argc>1 ? std::atoi(argv[1]) : 5;
    const int rows = argc>2 ? std::atoi(argv[2]) : 35;
    if (past < 0 || past > 32768 || rows < 1 || rows > 128) return 2;
    constexpr int d=192, heads=32, block_size=16, width=heads*d;
    const int blocks=(past+rows+block_size-1)/block_size;
    const size_t cache_size=size_t(blocks)*block_size*width;
    auto q=allocate<__nv_bfloat16>(rows*width);
    auto k=allocate<__nv_bfloat16>(cache_size);
    auto v=allocate<__nv_bfloat16>(cache_size);
    auto scalar=allocate<__nv_bfloat16>(rows*width);
    auto tiled=allocate<__nv_bfloat16>(rows*width);
    auto table=allocate<int>(blocks);
    auto lens=allocate<int>(rows);
    std::mt19937 rng(318);
    std::uniform_real_distribution<float> draw(-0.5f,0.5f);
    for (int i=0;i<blocks;++i) table[i]=blocks-1-i;
    for (int i=0;i<rows;++i) lens[i]=past+i+1;
    for (int i=0;i<rows*width;++i) q[i]=__float2bfloat16(draw(rng));
    for (size_t i=0;i<cache_size;++i) {
        k[i]=__float2bfloat16(draw(rng));
        v[i]=__float2bfloat16(draw(rng));
    }
    float ms[2];
    for (int kernel=0;kernel<2;++kernel) {
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin));
        check(cudaEventCreate(&end));
        for (int trial=-1;trial<4;++trial) {
            if (trial==0) check(cudaEventRecord(begin));
            if (kernel==0) {
                paged_decode_attn_ling<<<dim3(heads,rows),256>>>(q,k,v,scalar,table,lens,0,
                    heads,heads,d,block_size,1.f/std::sqrt(float(d)),width,0);
            } else {
                ling_mla_prefill<<<dim3(heads,(rows+31)/32),128>>>(q,k,v,tiled,table,rows,
                    past+rows,past,heads,heads,d,block_size,0,1,1.f/std::sqrt(float(d)));
            }
            if (trial==-1) check(cudaDeviceSynchronize());
        }
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        check(cudaEventElapsedTime(&ms[kernel],begin,end));
    }
    float worst=0;
    for (int i=0;i<rows*width;++i) {
        const float error=std::abs(float(scalar[i])-float(tiled[i]));
        if (!std::isfinite(error)) return 1;
        worst=std::max(worst,error);
    }
    std::printf("%d prompt rows, %d earlier tokens, worst %.8g, per-query %.3f ms, tiled %.3f ms\n",
                rows,past,worst,ms[0]/4,ms[1]/4);
    return worst > 0.002f;
}
