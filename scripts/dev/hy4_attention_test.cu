// SPDX-License-Identifier: AGPL-3.0-only

#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_indexpool.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); exit(1); } } while(0)
template<class T> struct Buffer {
    T* p;
    Buffer(size_t n) { CUDA(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T>& v) {
        CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t n) {
        std::vector<T> v(n);
        CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
    }
};
void close(float a,float b,float tol,const char* name) {
    if (!std::isfinite(a) || fabsf(a-b)>tol) {
        fprintf(stderr,"%s actual %.9g expected %.9g tolerance %.9g\n",name,a,b,tol);
        exit(1);
    }
}
void attention(unsigned position,unsigned window,bool indexed,bool invalid,bool zero) {
    const unsigned heads=3,page_size=16,columns=(position+16)/16;
    const unsigned count=indexed?2048:std::min(position+1,window?window:position+1);
    const unsigned splits=(count+127)/128;
    std::vector<unsigned> table(columns);
    for(unsigned i=0;i<columns;i++) table[i]=columns-1-i;
    std::vector<__nv_bfloat16> q(heads*576),cache(columns*page_size*576);
    for(unsigned i=0;i<q.size();i++) q[i]=__float2bfloat16_rn(zero?0.f:(int(i%29)-14)*0.03125f);
    for(unsigned i=0;i<cache.size();i++) cache[i]=__float2bfloat16_rn((int(i%37)-18)*0.015625f);
    std::vector<int> indices(2048,-1);
    for(unsigned i=0;i<std::min(position+1,2048u);i++) indices[i]=invalid?-1:int(i);
    std::vector<float> sink={-7.f,0.f,7.f};
    Buffer<__nv_bfloat16> dq(q.size()),dc(cache.size()),out(heads*512);
    Buffer<unsigned> dt(table.size());Buffer<int> di(indices.size());
    Buffer<float> ds(heads),parts(heads*splits*514);
    dq.put(q);dc.put(cache);dt.put(table);di.put(indices);ds.put(sink);
    hy4_mla_partial<<<dim3(heads,splits),256>>>(dq.p,dc.p,dt.p,indexed?di.p:nullptr,
        parts.p,position,page_size,columns,window,count,splits);
    hy4_mla_merge<<<heads,256>>>(parts.p,ds.p,out.p,splits);
    CUDA(cudaGetLastError());auto actual=out.get(heads*512);
    unsigned begin=window && position+1>window?position+1-window:0;
    for(unsigned h=0;h<heads;h++) {
        std::vector<double> scores;std::vector<unsigned> rows;
        double maximum=sink[h];
        for(unsigned j=0;j<count;j++) {
            int logical=indexed?indices[j]:int(begin+j);
            if(logical<0 || unsigned(logical)>position || unsigned(logical)<begin) continue;
            unsigned row=table[(unsigned(logical)/page_size)%columns]*page_size+unsigned(logical)%page_size;
            double dot=0;
            for(unsigned d=0;d<576;d++) dot+=float(q[h*576+d])*float(cache[row*576+d]);
            scores.push_back(dot/16.);rows.push_back(row);maximum=std::max(maximum,dot/16.);
        }
        double den=exp(double(sink[h])-maximum);
        for(double s:scores) den+=exp(s-maximum);
        for(unsigned d=0;d<512;d++) {
            double sum=0;
            for(unsigned j=0;j<rows.size();j++) sum+=exp(scores[j]-maximum)*float(cache[rows[j]*576+d]);
            close(float(actual[h*512+d]),float(__float2bfloat16_rn(float(sum/den))),0.002f,"latent attention");
        }
    }
}
void selection(unsigned groups,bool ties) {
    std::vector<float> scores(std::max(groups,1u));
    for(unsigned i=0;i<groups;i++) scores[i]=ties?1.f:float(int((i*311)%997)-498);
    Buffer<float> ds(scores.size());Buffer<int> out(2048);ds.put(scores);
    unsigned position=groups*4+2;
    hy4_index_select<<<1,256>>>(ds.p,out.p,groups,position);
    CUDA(cudaGetLastError());auto got=out.get(2048);
    std::vector<unsigned> ids(groups);std::iota(ids.begin(),ids.end(),0);
    std::stable_sort(ids.begin(),ids.end(),[&](unsigned a,unsigned b){return scores[a]>scores[b];});
    ids.resize(std::min(groups,511u));std::sort(ids.begin(),ids.end());
    std::vector<int> expected;
    for(unsigned i:ids) for(unsigned j=0;j<4;j++) expected.push_back(i*4+j);
    for(unsigned i=position/4*4;i<=position;i++) expected.push_back(i);
    expected.resize(2048,-1);
    if(got!=expected) {fprintf(stderr,"index selection mismatch groups=%u\n",groups);exit(1);}
}
void index_scores() {
    const unsigned groups=79,heads=16,pages=2;
    std::vector<uint8_t> q(heads*128),pool(pages*8448,0);
    std::vector<float> qs(heads);
    std::vector<__nv_bfloat16> weights(heads);
    std::vector<unsigned> table={1,0};
    for(unsigned h=0;h<heads;h++) {
        qs[h]=0.5f;weights[h]=__float2bfloat16_rn((int(h%7)-3)*0.03125f);
        for(unsigned d=0;d<128;d++) q[h*128+d]=__nv_fp8_e4m3((int((h*17+d)%31)-15)*0.125f).__x;
    }
    for(unsigned g=0;g<groups;g++) {
        unsigned physical=table[g/64],row=g%64;
        for(unsigned d=0;d<128;d++) pool[physical*8448+row*128+d]=__nv_fp8_e4m3((int((g*13+d)%23)-11)*0.125f).__x;
        float scale=0.25f;memcpy(pool.data()+physical*8448+8192+row*4,&scale,4);
    }
    Buffer<uint8_t> dq(q.size()),dp(pool.size());Buffer<float> dqs(heads),out(groups);
    Buffer<__nv_bfloat16> dw(heads);Buffer<unsigned> dt(pages);
    dq.put(q);dp.put(pool);dqs.put(qs);dw.put(weights);dt.put(table);
    hy4_index_scores<<<groups,256>>>(dq.p,dqs.p,dw.p,dp.p,dt.p,out.p,groups,heads);
    CUDA(cudaGetLastError());auto actual=out.get(groups);
    for(unsigned g=0;g<groups;g++) {
        float sum=0;
        for(unsigned h=0;h<heads;h++) {
            float dot=0;
            for(unsigned d=0;d<128;d++) {
                __nv_fp8_e4m3 a,b;a.__x=q[h*128+d];b.__x=pool[table[g/64]*8448+(g%64)*128+d];
                dot+=float(a)*float(b);
            }
            sum+=float(weights[h])*qs[h]*0.25f*fmaxf(0.f,dot);
        }
        close(actual[g],sum/sqrtf(float(heads*128)),1e-5f,"index scores");
    }
}
void projection(unsigned n,unsigned k) {
    const unsigned heads=3,is=k+64,os=n+64;
    std::vector<__nv_bfloat16> x(heads*is),w(heads*n*k);
    for(unsigned i=0;i<x.size();i++) x[i]=__float2bfloat16_rn((int(i%23)-11)*0.03125f);
    for(unsigned i=0;i<w.size();i++) w[i]=__float2bfloat16_rn((int(i%19)-9)*0.015625f);
    Buffer<__nv_bfloat16> dx(x.size()),dw(w.size()),out(heads*os);dx.put(x);dw.put(w);
    hy4_head_gemv<<<dim3((n+7)/8,heads),256>>>(dx.p,dw.p,out.p,n,k,is,os);
    CUDA(cudaGetLastError());auto actual=out.get(heads*os);
    for(unsigned h=0;h<heads;h++) for(unsigned r=0;r<n;r++) {
        float sum=0;for(unsigned d=0;d<k;d++) sum+=float(x[h*is+d])*float(w[(h*n+r)*k+d]);
        close(float(actual[h*os+r]),float(__float2bfloat16_rn(sum)),0.f,"head GEMV");
    }
}
int main() {
    projection(512,192);projection(256,512);
    for(unsigned n:{0u,1u,79u,511u,512u,777u,1024u}) {selection(n,false);selection(n,true);}
    index_scores();
    for(unsigned p:{0u,16u,130u,516u,2048u}) {
        attention(p,512,false,false,false);attention(p,0,true,false,false);
    }
    attention(17,0,true,true,false);attention(130,512,false,false,true);
    puts("HY4 attention CUDA: head GEMV, split-KV softmax/sink/pages/windows/masks, FP8 scores and stable top-511 passed");
}
