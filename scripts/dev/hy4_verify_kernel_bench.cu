// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_ffn.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"
#include <vector>
#include <cstdio>
#include <cstdlib>

#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> struct Buffer {
    T* p;size_t n;
    Buffer(size_t count):n(count){CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
struct Quant {
    Buffer<unsigned char> w,s;
    Quant(unsigned n,unsigned k):w(n*k/2),s(n*k/16) {
        std::vector<unsigned char> a(w.n),b(s.n);
        for(unsigned j=0;j<a.size();j++)a[j]=(j*73+19)%256;
        for(unsigned j=0;j<b.size();j++)b[j]=0x28+j%4;
        w.put(a);s.put(b);
    }
    Hy4Q4 descriptor(float scale){return {w.p,s.p,scale,0};}
};

template<class F> float elapsed(F run) {
    cudaEvent_t a,b;CUDA(cudaEventCreate(&a));CUDA(cudaEventCreate(&b));
    for(int i=0;i<3;i++)run();
    CUDA(cudaEventRecord(a));
    for(int i=0;i<20;i++)run();
    CUDA(cudaEventRecord(b));CUDA(cudaEventSynchronize(b));
    float ms;CUDA(cudaEventElapsedTime(&ms,a,b));
    CUDA(cudaEventDestroy(a));CUDA(cudaEventDestroy(b));
    return ms/20.f;
}
int main() {
    constexpr unsigned n=8192,k=3072;
    Buffer<__nv_bfloat16> x(6*k),w(n*k),out(6*n);
    Buffer<unsigned char> fp8(n*k);
    Buffer<float> scales(n);
    std::vector<__nv_bfloat16> input(x.n,__float2bfloat16_rn(.03125f));
    std::vector<__nv_bfloat16> weights(w.n,__float2bfloat16_rn(.125f));
    x.put(input);w.put(weights);
    quant_rowwise_fp8<<<n,256>>>(w.p,fp8.p,scales.p,n,k);
    for(unsigned rows:{1u,2u,4u,6u}) {
        float serial=elapsed([&] {
            for(unsigned r=0;r<rows;r++)
                hy4_fp8_gemv<<<n/8,256>>>(x.p+r*k,fp8.p,scales.p,out.p+r*n,n,k,0,0);
        });
        float batch=elapsed([&] {
            hy4_batch_linear<<<dim3(n/8,1),256>>>(x.p,fp8.p,scales.p,out.p,n,k,rows,1,0,0,k,n,k,1,0);
        });
        printf("HY4 projection rows=%u n=%u k=%u single_loop_ms=%.6f batch_ms=%.6f\n",rows,n,k,serial,batch);
    }
    constexpr unsigned h=1536,inter=1536,experts=512,slots=16;
    Quant gate(inter,h),up(inter,h),down(h,inter);
    std::vector<Hy4Q4> descriptors(experts*3,Hy4Q4{});
    for(unsigned e=0;e<experts/2;e++) {
        descriptors[3*e]=gate.descriptor(.5f);
        descriptors[3*e+1]=up.descriptor(.5f);
        descriptors[3*e+2]=down.descriptor(.5f);
    }
    Buffer<Hy4Q4> table(descriptors.size());table.put(descriptors);
    Buffer<unsigned> ids(6*slots),groups(experts*8);
    Buffer<__nv_bfloat16> latent(6*h),activated(6*slots*inter),partials(6*slots*h);
    for(unsigned rows:{2u,6u})for(unsigned overlap:{0u,1u}) {
        std::vector<unsigned> selected(6*slots);
        for(unsigned r=0;r<6;r++)for(unsigned s=0;s<slots;s++)
            selected[r*slots+s]=(overlap?s:r*slots+s);
        ids.put(selected);
        hy4_group_init<<<1,1>>>(groups.p,rows,experts);
        hy4_group_rows<<<experts,256>>>(ids.p,groups.p,rows,slots);
        float gate_ms=elapsed([&]{hy4_batch_expert<<<dim3(inter/8,1,rows*slots),256>>>(latent.p,table.p,groups.p,activated.p,inter,h,rows,slots,0,0,7.f,experts);});
        float down_ms=elapsed([&]{hy4_batch_expert<<<dim3(h/8,1,rows*slots),256>>>(activated.p,table.p,groups.p,partials.p,h,inter,rows,slots,0,1,7.f,experts);});
        printf("HY4 experts rows=%u overlap=%u gate_up_ms=%.6f down_ms=%.6f\n",rows,overlap,gate_ms,down_ms);
    }
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
}
