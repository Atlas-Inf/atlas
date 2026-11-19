// SPDX-License-Identifier: AGPL-3.0-only

// Equivalence harness for the batched IndexPool kernels:
// hy4_fwht vs hy4_fwht_batch, hy4_index_scores vs hy4_index_scores_batch,
// hy4_index_select vs hy4_index_select_batch. All comparisons are bitwise.

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_indexpool.cu"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> struct Buffer {
    T* p;size_t n;
    Buffer(size_t count):n(count){CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static std::mt19937 rng(42);

template<class T> void expect_eq(const std::vector<T>& a, const std::vector<T>& b, const char* name) {
    if (a.size()!=b.size()) {fprintf(stderr,"%s size %zu != %zu\n",name,a.size(),b.size());exit(1);}
    for (size_t i=0;i<a.size();i++)
        if (memcmp(&a[i],&b[i],sizeof(T))!=0) {
            fprintf(stderr,"%s at %zu: %g != %g\n",name,i,(float)a[i],(float)b[i]);exit(1);
        }
}

static void test_fwht() {
    unsigned heads=8, rows=8, n=rows*heads*128;
    std::vector<__nv_bfloat16> in(n);
    std::uniform_real_distribution<float> d(-2.f,2.f);
    for (auto& x : in) x=__float2bfloat16(d(rng));
    Buffer<__nv_bfloat16> bi(n), bo1(n), bo2(n);
    Buffer<uint8_t> bo1q(n), bo2q(n);
    Buffer<float> bs1(rows*heads), bs2(rows*heads);
    bi.put(in);
    for (unsigned r=0;r<rows;r++)
        hy4_fwht<<<heads,128>>>(bi.p+(size_t)r*heads*128, bo1q.p+(size_t)r*heads*128, bs1.p+r*heads);
    hy4_fwht_batch<<<dim3(heads,rows),128>>>(bi.p, bo2q.p, bs2.p, heads, rows);
    CUDA(cudaDeviceSynchronize());
    expect_eq(bo1q.get(), bo2q.get(), "fwht_batch out");
    expect_eq(bs1.get(), bs2.get(), "fwht_batch scales");
    printf("fwht_batch OK (heads=%u rows=%u)\n", heads, rows);
}

static void test_scores() {
    unsigned heads=8, rows=8, len_base=37, max_groups=65536;
    unsigned pages=(len_base+rows)/4/64+2;
    std::vector<uint8_t> pool((size_t)pages*8448);
    std::vector<unsigned> table(pages);
    for (unsigned i=0;i<pages;i++) table[i]=i;
    std::vector<uint8_t> quant((size_t)rows*heads*128);
    std::vector<float> qs((size_t)rows*heads);
    std::vector<__nv_bfloat16> hw((size_t)rows*heads);
    std::uniform_int_distribution<int> ib(0,255);
    std::uniform_real_distribution<float> fd(-1.f,1.f);
    for (auto& x : pool) x=(uint8_t)ib(rng);
    for (auto& x : quant) x=(uint8_t)ib(rng);
    for (auto& x : qs) x=fd(rng);
    for (auto& x : hw) x=__float2bfloat16(fd(rng));
    Buffer<uint8_t> bp((size_t)pages*8448), bt(pages), bq((size_t)rows*heads*128), bqs((size_t)rows*heads);
    Buffer<__nv_bfloat16> bhw((size_t)rows*heads);
    Buffer<float> bs1((size_t)rows*max_groups), bs2((size_t)rows*max_groups);
    bp.put(pool); bt.put(table); bq.put(quant); bqs.put(qs); bhw.put(hw);
    for (unsigned r=0;r<rows;r++) {
        unsigned groups=(len_base+r)/4;
        if (groups>0)
            hy4_index_scores<<<groups,256>>>(bq.p+(size_t)r*heads*128, bqs.p+r*heads,
                bhw.p+r*heads, bp.p, bt.p, bs1.p+(size_t)r*max_groups, groups, heads);
    }
    unsigned gmax=(len_base+rows)/4+1;
    hy4_index_scores_batch<<<dim3(gmax,rows),256>>>(bq.p, bqs.p, bhw.p, bp.p, bt.p,
        bs2.p, len_base, heads, max_groups, rows);
    CUDA(cudaDeviceSynchronize());
    auto s1=bs1.get(), s2=bs2.get();
    for (unsigned r=0;r<rows;r++) {
        unsigned groups=(len_base+r)/4;
        expect_eq(std::vector<float>(s1.begin()+(size_t)r*max_groups, s1.begin()+(size_t)r*max_groups+groups),
                  std::vector<float>(s2.begin()+(size_t)r*max_groups, s2.begin()+(size_t)r*max_groups+groups),
                  "scores_batch row");
    }
    printf("scores_batch OK (len_base=%u rows=%u)\n", len_base, rows);
}

static void test_select() {
    unsigned rows=8, max_groups=65536;
    for (unsigned len_base : {0u, 1u, 3u, 2043u, 2044u, 2047u, 4000u, 20000u, 80000u}) {
        std::vector<float> sc((size_t)rows*max_groups);
        std::uniform_real_distribution<float> fd(-1.f,1.f);
        for (auto& x : sc) x=floorf(fd(rng)*8.f)/8.f;  // quantized: forces ties
        Buffer<float> bsc((size_t)rows*max_groups);
        Buffer<int> bo1((size_t)rows*2048), bo2((size_t)rows*2048);
        bsc.put(sc);
        for (unsigned r=0;r<rows;r++) {
            unsigned len=len_base+r, groups=len/4;
            hy4_index_select<<<1,256>>>(bsc.p+(size_t)r*max_groups, bo1.p+(size_t)r*2048, groups, len);
        }
        hy4_index_select_batch<<<rows,256>>>(bsc.p, bo2.p, len_base, len_base, max_groups, rows);
        CUDA(cudaDeviceSynchronize());
        expect_eq(bo1.get(), bo2.get(), "select_batch");
        printf("select_batch OK (len_base=%u)\n", len_base);
    }
}

int main() {
    test_fwht();
    test_scores();
    test_select();
    printf("ALL OK\n");
    return 0;
}
