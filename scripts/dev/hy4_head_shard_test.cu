// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_model.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1); } } while(0)
int main() {
    const unsigned vocab=34,h=16,half=vocab/2;
    std::vector<float> w(vocab*h),logits(vocab),expected(vocab);
    std::vector<__nv_bfloat16> x(h);
    for(unsigned i=0;i<w.size();i++) w[i]=(int(i%31)-15)*0.00391f;
    for(unsigned i=0;i<h;i++) x[i]=__float2bfloat16_rn((int(i)-8)*0.03125f);
    float *dw,*dy;__nv_bfloat16 *dx,*local;
    CUDA(cudaMalloc(&dw,w.size()*4));CUDA(cudaMalloc(&dy,vocab*4));
    CUDA(cudaMalloc(&dx,h*2));CUDA(cudaMalloc(&local,half*h*2));
    CUDA(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx,x.data(),h*2,cudaMemcpyHostToDevice));
    for(unsigned rank=0;rank<2;rank++) {
        hy4_head_bf16<<<(half*h+255)/256,256>>>(dw+rank*half*h,local,half*h,1);
        std::vector<__nv_bfloat16> converted(half*h);
        CUDA(cudaMemcpy(converted.data(),local,half*h*2,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<half*h;i++) {
            auto b=__float2bfloat16_rn(w[rank*half*h+i]);
            if(memcmp(&b,&converted[i],2)) {puts("head cast mismatch");return 1;}
        }
        hy4_model_head<<<(half+7)/8,256>>>(dx,local,dy+rank*half,half,h,0);
    }
    CUDA(cudaGetLastError());CUDA(cudaMemcpy(logits.data(),dy,vocab*4,cudaMemcpyDeviceToHost));
    for(unsigned r=0;r<vocab;r++) {
        float sum=0;for(unsigned k=0;k<h;k++) sum+=float(x[k])*float(__float2bfloat16_rn(w[r*h+k]));
        if(fabsf(sum-logits[r])>1e-6f) {puts("sharded head mismatch");return 1;}
    }
    CUDA(cudaFree(dw));CUDA(cudaFree(dy));CUDA(cudaFree(dx));CUDA(cudaFree(local));
    puts("HY4 BF16 head: FP32 conversion, uneven block tails, both vocab shards and full FP32 logits passed");
}
