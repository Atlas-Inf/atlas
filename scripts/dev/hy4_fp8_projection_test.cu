// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA(call) do { auto e=(call);if(e!=cudaSuccess) {fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);} } while(0)
template<class T>T* buffer(size_t n) {T* p;CUDA(cudaMalloc(&p,n*sizeof(T)));return p;}
void check(unsigned k) {
    unsigned n=17,heads=3,in_stride=k+16,out_stride=n+7,rows=n*heads;
    std::vector<__nv_bfloat16> x(heads*in_stride),w(rows*k),y(heads*out_stride);
    for(unsigned i=0;i<x.size();i++) x[i]=__float2bfloat16_rn((int(i%23)-11)*0.03125f);
    for(unsigned r=1;r<rows;r++)for(unsigned c=0;c<k;c++)
        w[r*k+c]=__float2bfloat16_rn((int((r*17+c)%37)-18)*0.0173f);
    auto dx=buffer<__nv_bfloat16>(x.size()),dw=buffer<__nv_bfloat16>(w.size()),dy=buffer<__nv_bfloat16>(y.size());
    auto qw=buffer<unsigned char>(w.size());auto ds=buffer<float>(rows);
    CUDA(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
    quant_rowwise_fp8<<<rows,256>>>(dw,qw,ds,rows,k);
    hy4_fp8_gemv<<<dim3((n+7)/8,heads),256>>>(dx,qw,ds,dy,n,k,in_stride,out_stride);
    CUDA(cudaGetLastError());CUDA(cudaMemcpy(y.data(),dy,y.size()*2,cudaMemcpyDeviceToHost));
    std::vector<float> scales(rows);std::vector<unsigned char> packed(w.size());
    CUDA(cudaMemcpy(scales.data(),ds,rows*4,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(packed.data(),qw,w.size(),cudaMemcpyDeviceToHost));
    for(unsigned r=0;r<rows;r++) {
        float mx=0;for(unsigned c=0;c<k;c++)mx=std::max(mx,fabsf(float(w[r*k+c])));
        float scale=std::max(mx/448.f,1e-12f);
        if(fabsf(scale-scales[r])>scale*1e-6f) {puts("row scale mismatch");exit(1);}
        float sum=0;
        for(unsigned c=0;c<k;c++) {
            __nv_fp8_e4m3 v(float(w[r*k+c])/scale);
            if(v.__x!=packed[r*k+c]) {puts("E4M3 quant mismatch");exit(1);}
            sum+=float(x[(r/n)*in_stride+c])*(float(v)*scale);
        }
        float expected=float(__float2bfloat16_rn(sum)),actual=float(y[(r/n)*out_stride+r%n]);
        if(!std::isfinite(actual)||fabsf(expected-actual)>std::max(0.0001f,fabsf(expected)*0.008f)) {
            fprintf(stderr,"k=%u row=%u: %g != %g\n",k,r,actual,expected);exit(1);
        }
    }
    CUDA(cudaFree(dx));CUDA(cudaFree(dw));CUDA(cudaFree(dy));CUDA(cudaFree(qw));CUDA(cudaFree(ds));
}
int main() {
    for(unsigned k:{17u,33u,192u,512u,1024u,3072u})check(k);
    unsigned qrank=1024,total=qrank+576;std::vector<__nv_bfloat16> shares(total),q(qrank),kv(576);
    for(unsigned i=0;i<total;i++)shares[i]=__float2bfloat16_rn(float(i));
    auto ds=buffer<__nv_bfloat16>(total),dq=buffer<__nv_bfloat16>(qrank),dk=buffer<__nv_bfloat16>(576);
    CUDA(cudaMemcpy(ds,shares.data(),total*2,cudaMemcpyHostToDevice));
    hy4_join_qkv<<<1,256>>>(ds,dq,dk,qrank);
    CUDA(cudaMemcpy(q.data(),dq,qrank*2,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(kv.data(),dk,576*2,cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<qrank;i++)if(float(q[i])!=float(shares[(i/512)*800+i%512]))return 1;
    for(unsigned i=0;i<576;i++)if(float(kv[i])!=float(shares[(i/288)*800+512+i%288]))return 1;
    CUDA(cudaFree(ds));CUDA(cudaFree(dq));CUDA(cudaFree(dk));
    puts("HY4 FP8: E4M3 row quantization/scales/zero rows, vector and unaligned GEMV, heads/tails, exact TP2 Q/KV join passed");
}
