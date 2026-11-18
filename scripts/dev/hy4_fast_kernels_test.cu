// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_ffn.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_model.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); exit(1); } } while(0)

template<class T> struct Buffer {
    T* p;
    Buffer(size_t n) { CUDA(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T>& v) { CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) {
        std::vector<T> v(n);
        CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
        return v;
    }
};
float bf(float x) { return (float)__float2bfloat16_rn(x); }
void close(float actual,float expected,float tolerance,const char* name) {
    if(!std::isfinite(actual) || fabsf(actual-expected)>tolerance) {
        fprintf(stderr,"%s: actual %.9g expected %.9g tolerance %.9g\n",name,actual,expected,tolerance);
        exit(1);
    }
}
void linear(unsigned n,unsigned k,unsigned rows,bool f32,bool float_out) {
    std::vector<__nv_bfloat16> x(rows*k),w(n*k);
    std::vector<float> wf(n*k);
    for(unsigned j=0;j<x.size();j++) x[j]=__float2bfloat16_rn((int(j%23)-11)*0.03125f);
    for(unsigned j=0;j<w.size();j++) {
        wf[j]=(int(j%19)-9)*0.015625f; w[j]=__float2bfloat16_rn(wf[j]);
    }
    Buffer<__nv_bfloat16> dx(x.size()),dw(w.size()),dy(n*rows);
    Buffer<float> df(wf.size()),out(n*rows);
    dx.put(x);dw.put(w);df.put(wf);
    CUDA(cudaMemset(out.p,0xff,n*rows*sizeof(float)));
    CUDA(cudaMemset(dy.p,0xff,n*rows*sizeof(__nv_bfloat16)));
    hy4_linear<<<dim3((n+7)/8,rows),256>>>(dx.p,f32?(void*)df.p:(void*)dw.p,
        float_out?(void*)out.p:(void*)dy.p,n,k,f32,float_out);
    CUDA(cudaGetLastError());
    auto a=out.get(n*rows); auto b=dy.get(n*rows);
    for(unsigned t=0;t<rows;t++) for(unsigned r=0;r<n;r++) {
        float expected=0;
        for(unsigned j=0;j<k;j++) expected+=(float)x[t*k+j]*(f32?wf[r*k+j]:(float)w[r*k+j]);
        close(float_out?a[t*n+r]:(float)b[t*n+r],float_out?expected:bf(expected),0.0001f,"linear");
    }
    if(rows==1) {
        hy4_model_head<<<(n+7)/8,256>>>(dx.p,f32?(void*)df.p:(void*)dw.p,out.p,n,k,f32);
        CUDA(cudaGetLastError()); a=out.get(n);
        for(unsigned r=0;r<n;r++) {
            float expected=0;
            for(unsigned j=0;j<k;j++) expected+=(float)x[j]*(f32?wf[r*k+j]:(float)w[r*k+j]);
            close(a[r],expected,0.0001f,"head");
        }
    }
}
void routing(unsigned e,unsigned topk,bool renorm,bool ties) {
    std::vector<float> logits(e),bias(e);
    for(unsigned j=0;j<e;j++) {
        logits[j]=ties?0.f:(int(j%31)-15)*0.25f;
        bias[j]=ties?0.f:(int(j%7)-3)*0.03125f;
    }
    Buffer<float> dl(e),db(e),dw(topk);
    Buffer<unsigned> ids(topk);dl.put(logits);db.put(bias);
    hy4_topk<<<1,512>>>(dl.p,db.p,ids.p,dw.p,e,topk,renorm,2.5f);
    CUDA(cudaGetLastError());auto got=ids.get(topk);auto weights=dw.get(topk);
    std::vector<unsigned> expected(e);
    std::vector<float> scores(e);
    for(unsigned j=0;j<e;j++) {
        expected[j]=j;
        scores[j]=logits[j]>=0?1.f/(1.f+expf(-logits[j])):expf(logits[j])/(1.f+expf(logits[j]));
    }
    std::sort(expected.begin(),expected.end(),[&](unsigned a,unsigned b) {
        float x=scores[a]+bias[a],y=scores[b]+bias[b];
        return x>y || (x==y && a<b);
    });
    float sum=0;
    for(unsigned j=0;j<topk;j++)sum+=scores[expected[j]];
    for(unsigned j=0;j<topk;j++) {
        if(got[j]!=expected[j]) {fprintf(stderr,"topk id mismatch at %u\n",j);exit(1);}
        close(weights[j],scores[expected[j]]/(renorm?sum+1e-20f:1.f)*2.5f,1e-6f,"topk weight");
    }
}
const float fp4[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
struct Quant {
    unsigned n,k;
    std::vector<uint8_t> w,s;
    Buffer<uint8_t> dw,ds;
    Quant(unsigned n_,unsigned k_):n(n_),k(k_),w(n*k/2),s(n*k/16),dw(w.size()),ds(s.size()) {
        for(unsigned j=0;j<w.size();j++)w[j]=(j*73+19)%256;
        for(unsigned j=0;j<s.size();j++)s[j]=0x28+j%4;
        dw.put(w);ds.put(s);
    }
    Hy4Q4 descriptor() { return {dw.p,ds.p,0.5f,0}; }
    float weight(unsigned row,unsigned col) {
        __nv_fp8_e4m3 scale;scale.__x=s[row*k/16+col/16];
        return fp4[(w[row*k/2+col/2]>>((col%2)*4))&15]*((float)scale*0.5f);
    }
};
void mlp(unsigned k,unsigned inter,unsigned slots,bool remote,float limit) {
    const unsigned n=k,rows=2;
    Quant gate(inter,k),up(inter,k),down(n,inter);
    std::vector<Hy4Q4> table;
    for(unsigned e=0;e<slots;e++) {
        table.push_back(gate.descriptor());table.push_back(up.descriptor());table.push_back(down.descriptor());
        if(remote && e%2) for(unsigned j=0;j<3;j++)table[e*3+j]={nullptr,nullptr,0,0};
    }
    std::vector<unsigned> ids(rows*slots);
    std::vector<float> weights(rows*slots);
    for(unsigned j=0;j<ids.size();j++) {ids[j]=slots-1-j%slots;weights[j]=1.f/(j%slots+1);}
    std::vector<__nv_bfloat16> x(rows*k);
    for(unsigned j=0;j<x.size();j++)x[j]=__float2bfloat16_rn((int(j%13)-6)*0.125f);
    Buffer<Hy4Q4> dt(table.size());dt.put(table);
    Buffer<unsigned> di(ids.size());di.put(ids);
    Buffer<float> dw(weights.size());dw.put(weights);
    Buffer<__nv_bfloat16> dx(x.size()),da(rows*slots*inter),out(rows*n);
    dx.put(x);
    bool routed=slots>1;
    hy4_qgate_up<<<dim3((inter+7)/8,rows,slots),256>>>(dx.p,dt.p,routed?di.p:nullptr,da.p,k,inter,slots,0,limit);
    CUDA(cudaGetLastError());auto activated=da.get(rows*slots*inter);
    std::vector<float> reference(activated.size());
    for(unsigned t=0;t<rows;t++)for(unsigned slot=0;slot<slots;slot++)for(unsigned r=0;r<inter;r++) {
        unsigned e=ids[t*slots+slot];float a=0,b=0;
        if(!remote || e%2==0)for(unsigned j=0;j<k;j++) {
            a+=gate.weight(r,j)*(float)x[t*k+j];b+=up.weight(r,j)*(float)x[t*k+j];
        }
        a=bf(a);b=bf(b);
        if(limit>0) {a=fminf(a,limit);b=fminf(limit,fmaxf(-limit,b));}
        float sig=a>=0?1.f/(1.f+expf(-a)):expf(a)/(1.f+expf(a));
        unsigned at=(t*slots+slot)*inter+r;
        reference[at]=bf(a*sig*b);
        close((float)activated[at],reference[at],0.0001f,"gate/up boundary");
    }
    hy4_qdown<<<dim3(n,rows),256>>>(da.p,dt.p,routed?di.p:nullptr,routed?dw.p:nullptr,out.p,n,inter,slots,0);
    CUDA(cudaGetLastError());auto got=out.get(rows*n);
    for(unsigned t=0;t<rows;t++)for(unsigned r=0;r<n;r++) {
        float sum=0;
        for(unsigned slot=0;slot<slots;slot++) {
            unsigned e=ids[t*slots+slot];float v=0;
            if(!remote || e%2==0)for(unsigned j=0;j<inter;j++)v+=down.weight(r,j)*reference[(t*slots+slot)*inter+j];
            sum+=bf(v)*(routed?weights[t*slots+slot]:1.f);
        }
        close((float)got[t*n+r],bf(sum),0.0001f,"down expert boundary/combine");
    }
}
int main() {
    for(bool f32:{false,true})for(bool out:{false,true}) {
        linear(19,16,1,f32,out);linear(129,1536,2,f32,out);linear(35,3072,1,f32,out);
    }
    routing(512,16,true,false);routing(512,16,true,true);routing(17,5,false,false);
    mlp(16,16,1,false,0);mlp(48,32,16,true,0.25f);mlp(1536,48,16,true,0.5f);
    CUDA(cudaDeviceSynchronize());
    puts("HY4 CUDA kernels: linear/head BF16+FP32, tails, multi-row, top-k ties/bias, clamps, remote experts and expert combine passed");
}
