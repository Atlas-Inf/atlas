// SPDX-License-Identifier: AGPL-3.0-only
// Build from the Atlas root: nvcc -std=c++17 -arch=sm_121 tests/ling/fp8_linear.cu -o /tmp/ling-fp8
#include "../../kernels/gb10/common/per_token_group_quant_fp8.cu"
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_fp8_linear.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static void check(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(2); }
}

template<class T> static T* buffer(size_t n) {
    T* ptr;
    check(cudaMallocManaged(&ptr, n * sizeof(T)));
    return ptr;
}

template<class Scale> static void run(int m, int n, int k, bool bf16_scale) {
    const int groups = k/128;
    auto x = buffer<__nv_bfloat16>(m*k);
    auto a = buffer<__nv_fp8_e4m3>(m*k);
    auto ax = buffer<float>(m*groups);
    auto w = buffer<__nv_fp8_e4m3>(n*k);
    auto ws = buffer<Scale>(((n+127)/128)*groups);
    auto y = buffer<__nv_bfloat16>(m*n);
    for (int t=0; t<m; ++t) for (int j=0; j<k; ++j) {
        const int q = j%128 ? (j*13 + t)%43-21 : 448;
        x[t*k+j] = __float2bfloat16(std::ldexp(float(q), (t+j/128)%5-4));
    }
    for (int i=0; i<n; ++i) for (int j=0; j<k; ++j)
        w[i*k+j] = __nv_fp8_e4m3(float((i*3+j*13)%23-11));
    for (int i=0; i<((n+127)/128)*groups; ++i) ws[i] = Scale(std::ldexp(1.f, i%7-6));
    per_token_group_quant_fp8<<<dim3(m,groups),128>>>(x,reinterpret_cast<unsigned char*>(a),ax,m,k);
    ling_fp8_linear<<<dim3((n+7)/8,m),256>>>(a,ax,w,ws,y,n,k,bf16_scale);
    check(cudaDeviceSynchronize());
    for (int t=0; t<m; ++t) for (int i=0; i<n; ++i) {
        float reference = 0.f;
        for (int g=0; g<groups; ++g) {
            const float scale = std::ldexp(1.f,(t+g)%5-4);
            float block=0;
            for (int o=0; o<128; ++o) {
                const int j=g*128+o;
                const auto q = __nv_fp8_e4m3(float(x[t*k+j]) / scale);
                block += float(q) * float(w[i*k+j]);
            }
            reference += block * (scale * float(ws[(i/128)*groups+g]));
        }
        if (float(y[t*n+i]) != float(__float2bfloat16(reference))) {
            std::fprintf(stderr,"row %d channel %d: %f vs %f\n",t,i,float(y[t*n+i]),reference);
            std::exit(1);
        }
    }
    std::printf("%d x %d x %d (%s scales): %d BF16 outputs matched\n",
                m,n,k,bf16_scale ? "BF16" : "F32",m*n);
    check(cudaFree(y)); check(cudaFree(ws)); check(cudaFree(w));
    check(cudaFree(ax)); check(cudaFree(a)); check(cudaFree(x));
}
int main() {
    run<float>(1,768,2560,false);
    run<__nv_bfloat16>(17,576,2560,true);
    run<float>(4,2560,768,false);
}
