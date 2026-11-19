// SPDX-License-Identifier: AGPL-3.0-only

#include <cublas_v2.h>
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_dfly.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_model.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"

using bf=__nv_bfloat16;
extern "C" int dfly_linear(const bf* x,const bf* w,void* y,
    unsigned m,unsigned n,unsigned k,unsigned fp32,unsigned long long stream) {
    static cublasHandle_t handle=nullptr;
    if(!handle) {auto err=cublasCreate(&handle);if(err)return (int)err;}
    auto err=cublasSetStream(handle,(cudaStream_t)stream);if(err)return (int)err;
    float alpha=1.f,beta=0.f;
    return (int)cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,
        &alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,y,
        fp32?CUDA_R_32F:CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
extern "C" int dfly_norm(const bf* x,const bf* w,bf* y,unsigned m,unsigned h,
    unsigned xs,unsigned ys,float eps,unsigned long long stream) {
    hy4_dfly_norm<<<m,256,0,(cudaStream_t)stream>>>(x,w,y,h,xs,ys,eps);
    return (int)cudaGetLastError();
}
extern "C" int dfly_embed(const bf* table,const unsigned* ids,bf* out,unsigned rows,
    unsigned h,unsigned long long stream) {
    hy4_dfly_embed<<<dim3((h+255)/256,rows),256,0,(cudaStream_t)stream>>>(table,ids,out,h);
    return (int)cudaGetLastError();
}
extern "C" int dfly_pick(const float* logits,unsigned* out,unsigned vocab,unsigned long long stream) {
    hy4_model_argmax<<<1,256,0,(cudaStream_t)stream>>>(logits,out,vocab);
    return (int)cudaGetLastError();
}
extern "C" int dfly_add(const bf* a,const bf* b,bf* y,unsigned size,unsigned long long stream) {
    hy4_dfly_add<<<(size+255)/256,256,0,(cudaStream_t)stream>>>(a,b,y,size);
    return (int)cudaGetLastError();
}
extern "C" int dfly_fuse(const bf* a,const bf* p,const bf* g,bf* y,unsigned h,
    unsigned long long stream) {
    hy4_dfly_fuse<<<(h+255)/256,256,0,(cudaStream_t)stream>>>(a,p,g,y,h);
    return (int)cudaGetLastError();
}
extern "C" int dfly_silu(const bf* g,const bf* u,bf* y,unsigned size,unsigned long long stream) {
    hy4_dfly_silu<<<(size+255)/256,256,0,(cudaStream_t)stream>>>(g,u,y,size);
    return (int)cudaGetLastError();
}
extern "C" int dfly_gate(bf* x,const bf* g,unsigned size,unsigned long long stream) {
    hy4_dfly_gate<<<(size+255)/256,256,0,(cudaStream_t)stream>>>(x,g,size);
    return (int)cudaGetLastError();
}
extern "C" int dfly_heads(const bf* x,const bf* w,bf* y,unsigned rows,unsigned heads,
    unsigned n,unsigned k,unsigned input_stride,unsigned output_stride,unsigned long long stream) {
    for(unsigned r=0;r<rows;r++)
        hy4_head_gemv<<<dim3((n+7)/8,heads),256,0,(cudaStream_t)stream>>>(
            x+(size_t)r*heads*input_stride,w,y+(size_t)r*heads*output_stride,
            n,k,input_stride,output_stride);
    return (int)cudaGetLastError();
}
extern "C" int dfly_pack(const bf* q,const bf* kv,bf* absorbed,bf* cache,
    unsigned rows,unsigned heads,unsigned anchor,float theta,unsigned long long stream) {
    hy4_dfly_pack<<<dim3(heads,rows),256,0,(cudaStream_t)stream>>>(q,kv,absorbed,cache,heads,anchor,theta);
    return (int)cudaGetLastError();
}
extern "C" int dfly_attention(const bf* q,const bf* cache,const int* history,const bf* sink,
    bf* out,unsigned rows,unsigned heads,unsigned anchor,unsigned topk,unsigned long long stream) {
    if(rows<1 || rows>6 || topk>2048)return -1;
    hy4_dfly_attention<<<dim3(heads,rows),256,0,(cudaStream_t)stream>>>(
        q,cache,history,sink,out,heads,anchor,topk);
    return (int)cudaGetLastError();
}
extern "C" int dfly_head(const bf* x,const bf* w,float* out,unsigned vocab,unsigned h,
    unsigned long long stream) {
    hy4_model_head<<<(vocab+7)/8,256,0,(cudaStream_t)stream>>>(x,w,out,vocab,h,0);
    return (int)cudaGetLastError();
}
