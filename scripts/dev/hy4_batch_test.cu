// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_ffn.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_engram.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_model.cu"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> struct Buffer {
    T* p;size_t n;
    Buffer(size_t count):n(count){CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<class T>void equal(const std::vector<T>& a,const std::vector<T>& b,const char* name,bool exact=true){
    if(a.size()!=b.size())exit(1);
    for(size_t i=0;i<a.size();i++){
        float x=float(a[i]),y=float(b[i]);
        if(!std::isfinite(x)||!std::isfinite(y)||(exact?memcmp(&a[i],&b[i],sizeof(T))!=0:fabsf(x-y)>fmaxf(.0001f,fabsf(y)*.008f))){
            fprintf(stderr,"%s at %zu: %g != %g\n",name,i,x,y);exit(1);
        }
    }
}
template<class T>void linear(unsigned rows,unsigned k,unsigned kind,unsigned heads){
    unsigned n=19,xh=k+16,yh=n+7,xr=heads*xh,yr=heads*yh;
    std::vector<__nv_bfloat16>x(rows*xr),w(heads*n*k);
    std::vector<float>wf(w.size());
    for(unsigned i=0;i<x.size();i++)x[i]=__float2bfloat16_rn((int(i%29)-14)*.03125f);
    for(unsigned i=0;i<w.size();i++){wf[i]=(int(i%31)-15)*.0121f;w[i]=__float2bfloat16_rn(wf[i]);}
    Buffer<__nv_bfloat16>dx(x.size()),dw(w.size());dx.put(x);dw.put(w);
    Buffer<float>df(w.size()),scales(heads*n);df.put(wf);
    Buffer<unsigned char>fp8(w.size());
    quant_rowwise_fp8<<<heads*n,256>>>(dw.p,fp8.p,scales.p,heads*n,k);
    Buffer<T>out(rows*yr),reference(rows*yr);
    const void*weight=kind==1?(void*)fp8.p:kind==2?(void*)df.p:(void*)dw.p;
    unsigned of32=sizeof(T)==4;
    hy4_batch_linear<<<dim3((n+7)/8,(rows+7)/8,heads),256>>>(dx.p,weight,scales.p,out.p,n,k,rows,heads,xh,yh,xr,yr,k,kind,of32);
    for(unsigned r=0;r<rows;r++)for(unsigned h=0;h<heads;h++){
        auto input=dx.p+r*xr+h*xh;
        auto output=reference.p+r*yr+h*yh;
        if(kind==1)hy4_fp8_gemv<<<(n+7)/8,256>>>(input,fp8.p+h*n*k,scales.p+h*n,(__nv_bfloat16*)output,n,k,0,0);
        else hy4_linear<<<(n+7)/8,256>>>(input,kind==2?(void*)(df.p+h*n*k):(void*)(dw.p+h*n*k),output,n,k,kind==2,of32);
    }
    CUDA(cudaGetLastError());equal(out.get(),reference.get(),"batch projection",kind!=1||k%16==0);
}
struct Quant{
    Buffer<unsigned char>w,s;
    Quant(unsigned n,unsigned k):w(n*k/2),s(n*k/16){
        std::vector<unsigned char>a(w.n),b(s.n);
        for(unsigned j=0;j<a.size();j++)a[j]=(j*73+19)%256;
        for(unsigned j=0;j<b.size();j++)b[j]=0x28+j%4;
        w.put(a);s.put(b);
    }
    Hy4Q4 descriptor(float scale){return {w.p,s.p,scale,0};}
};
void experts(unsigned rows,unsigned slots,bool remote,unsigned group){
    unsigned h=48,inter=32,e=slots==1?1:4;
    Quant gate(inter,h),up(inter,h),down(h,inter);
    std::vector<Hy4Q4>desc((e+group)*3);
    for(unsigned j=group;j<e+group;j++){
        desc[j*3]=gate.descriptor(.25f*(j+1));desc[j*3+1]=up.descriptor(.5f);desc[j*3+2]=down.descriptor(.125f);
        if(remote&&j==1)for(unsigned z=0;z<3;z++)desc[j*3+z]={nullptr,nullptr,0,0};
    }
    std::vector<unsigned>ids(rows*slots);
    std::vector<float>weights(rows*slots);
    std::vector<__nv_bfloat16>x(rows*h);
    for(unsigned t=0;t<rows;t++)for(unsigned s=0;s<slots;s++){ids[t*slots+s]=(t+s)%e;weights[t*slots+s]=1.f/(s+1);}
    for(unsigned j=0;j<x.size();j++)x[j]=__float2bfloat16_rn((int(j%13)-6)*.125f);
    Buffer<Hy4Q4>table(desc.size());table.put(desc);
    Buffer<unsigned>di(ids.size()),groups(e*(rows+2)+1);di.put(ids);
    Buffer<float>dw(weights.size());dw.put(weights);
    Buffer<__nv_bfloat16>dx(x.size()),a(rows*slots*inter),refa(rows*slots*inter),partial(rows*slots*h),out(rows*h),ref(rows*h);
    dx.put(x);
    auto ip=slots==1?nullptr:di.p;auto wp=slots==1?nullptr:dw.p;
    hy4_group_init<<<1,1>>>(groups.p,rows,e);
    hy4_group_rows<<<e,256>>>(ip,groups.p,rows,slots);
    auto list=groups.get();
    for(unsigned ex=0;ex<e;ex++){
        std::vector<unsigned>expected;
        for(unsigned t=0;t<rows;t++)for(unsigned s=0;s<slots;s++)if(!ip||ids[t*slots+s]==ex)expected.push_back(t*slots+s);
        if(list[ex*(rows+1)]!=expected.size()){puts("group count mismatch");exit(1);}
        for(unsigned j=0;j<expected.size();j++)if(list[ex*(rows+1)+1+j]!=expected[j]){puts("group order mismatch");exit(1);}
    }
    float limit=slots==1?0.f:.25f;
    hy4_batch_expert<<<dim3((inter+7)/8,1,e),256>>>(dx.p,table.p,groups.p,a.p,inter,h,rows,slots,group,0,limit,e);
    hy4_batch_expert<<<dim3((h+7)/8,1,e),256>>>(a.p,table.p,groups.p,partial.p,h,inter,rows,slots,group,1,limit,e);
    hy4_batch_combine<<<dim3((h+255)/256,rows),256>>>(partial.p,wp,out.p,h,slots);
    hy4_qgate_up<<<dim3((inter+7)/8,rows,slots),256>>>(dx.p,table.p,ip,refa.p,h,inter,slots,group,limit);
    hy4_qdown<<<dim3(h,rows),256>>>(refa.p,table.p,ip,wp,ref.p,h,inter,slots,group);
    CUDA(cudaGetLastError());equal(a.get(),refa.get(),"grouped gate/up");equal(out.get(),ref.get(),"grouped down/combine");
}
void joins(unsigned rows){
    unsigned qrank=32,stride=qrank/2+288;
    std::vector<__nv_bfloat16>s(rows*stride*2),q(rows*qrank),kv(rows*576);
    for(unsigned j=0;j<s.size();j++)s[j]=__float2bfloat16_rn(float(j%101)*.125f);
    for(unsigned r=0;r<rows;r++){
        for(unsigned j=0;j<qrank;j++)q[r*qrank+j]=s[(j/(qrank/2))*rows*stride+r*stride+j%(qrank/2)];
        for(unsigned j=0;j<576;j++)kv[r*576+j]=s[(j/288)*rows*stride+r*stride+qrank/2+j%288];
    }
    Buffer<__nv_bfloat16>ds(s.size()),dq(q.size()),dk(kv.size());ds.put(s);
    hy4_batch_join_qkv<<<rows,256>>>(ds.p,dq.p,dk.p,qrank,rows);
    equal(dq.get(),q,"batch Q join");equal(dk.get(),kv,"batch KV join");
}
void engram(unsigned rows,bool bias){
    unsigned h=19,k=32,full=64,start=16,stride=h*4;
    std::vector<__nv_bfloat16>x(rows*k),w(h*full),b(h);
    for(unsigned j=0;j<x.size();j++)x[j]=__float2bfloat16_rn((int(j%19)-9)*.0625f);
    for(unsigned j=0;j<w.size();j++)w[j]=__float2bfloat16_rn((int(j%23)-11)*.0213f);
    for(unsigned j=0;j<b.size();j++)b[j]=__float2bfloat16_rn((int(j%7)-3)*.016f);
    Buffer<__nv_bfloat16>dx(x.size()),dw(w.size()),db(b.size()),out(rows*stride),ref(rows*stride);
    dx.put(x);dw.put(w);db.put(b);
    hy4_engram_batch_project<<<dim3((h+127)/128,(rows+7)/8),128>>>(dx.p,dw.p,db.p,out.p,h,full,start,k,bias,rows,stride);
    for(unsigned r=0;r<rows;r++)hy4_engram_project<<<(h+127)/128,128>>>(dx.p+r*k,dw.p,db.p,ref.p+r*stride,h,full,start,k,bias);
    equal(out.get(),ref.get(),"batch Engram projection");
}
void model_rows(unsigned rows){
    unsigned h=19;
    std::vector<__nv_bfloat16>table(7*h),weight(h);
    std::vector<unsigned>tokens(rows);
    for(unsigned j=0;j<table.size();j++)table[j]=__float2bfloat16_rn((int(j%23)-11)*.03125f);
    for(unsigned j=0;j<h;j++)weight[j]=__float2bfloat16_rn(.5f+j*.125f);
    for(unsigned j=0;j<rows;j++)tokens[j]=(j*3)%7;
    Buffer<__nv_bfloat16>dt(table.size()),dw(h),out(rows*4*h),ref(rows*4*h),norm(rows*h),refnorm(rows*h);
    Buffer<unsigned>di(rows);dt.put(table);dw.put(weight);di.put(tokens);
    hy4_model_batch_embed<<<dim3(1,rows),256>>>(dt.p,di.p,out.p,h);
    for(unsigned r=0;r<rows;r++)hy4_model_embed<<<1,256>>>(dt.p,ref.p+r*4*h,tokens[r],h);
    equal(out.get(),ref.get(),"batch embedding");
    hy4_model_norm<<<rows,256>>>(out.p,dw.p,norm.p,h,1e-5f);
    for(unsigned r=0;r<rows;r++)hy4_model_norm<<<1,256>>>(out.p+r*h,dw.p,refnorm.p+r*h,h,1e-5f);
    equal(norm.get(),refnorm.get(),"batch norm");
}
int main(){
    for(unsigned rows:{1u,2u,4u,6u,9u,128u}){
        linear<__nv_bfloat16>(rows,192,0,3);
        linear<float>(rows,1024,0,1);
        linear<float>(rows,16,2,1);
        linear<__nv_bfloat16>(rows,512,1,3);
        joins(rows);engram(rows,true);engram(rows,false);model_rows(rows);
    }
    linear<__nv_bfloat16>(2,17,1,3);
    experts(6,2,true,0);experts(9,4,false,0);experts(128,1,false,4);
    Buffer<unsigned>empty(9);hy4_group_init<<<1,1>>>(empty.p,0,4);hy4_group_rows<<<4,256>>>(nullptr,empty.p,0,1);
    for(unsigned n:empty.get())if(n!=0)return 1;
    CUDA(cudaDeviceSynchronize());
    puts("HY4 batch: N=1/2/4/6/9/128 BF16/FP8/FP32 projections, grouped NVFP4 rows/clamps/remote experts/combine, empty groups, strided Engram and TP2 Q/KV join passed");
}
