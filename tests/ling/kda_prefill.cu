// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -std=c++17 -O2 --fmad=false -arch=sm_121 tests/ling/kda_prefill.cu -o /tmp/ling-kda
#include "../../kernels/gb10/glm-5.3-flash/nvfp4/kda.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}
template<class T> static T* alloc(size_t n) {
    T* data;
    check(cudaMallocManaged(&data, n * sizeof(T)));
    return data;
}
int main() {
    constexpr int rows = 257, heads = 32, dim = 128;
    constexpr size_t plane = rows * heads * dim, state_size = heads * dim * dim;
    auto qkv = alloc<__nv_bfloat16>(3 * plane), gate = alloc<__nv_bfloat16>(plane);
    auto beta = alloc<__nv_bfloat16>(rows * heads);
    auto a_log = alloc<float>(heads), bias = alloc<float>(heads * dim);
    auto q_norm = alloc<float>(plane), k_norm = alloc<float>(plane), decay = alloc<float>(plane);
    auto beta_out = alloc<float>(rows * heads);
    auto base_state = alloc<float>(state_size), fast_state = alloc<float>(state_size);
    auto base_out = alloc<__nv_bfloat16>(plane), fast_out = alloc<__nv_bfloat16>(plane);
    std::mt19937 rng(197);
    std::uniform_real_distribution<float> draw(-1.f, 1.f);
    for (size_t i=0; i<3*plane; ++i) qkv[i] = __float2bfloat16(draw(rng));
    for (size_t i=0; i<plane; ++i) gate[i] = __float2bfloat16(4.f * draw(rng));
    for (int i=0; i<rows*heads; ++i) beta[i] = __float2bfloat16(draw(rng));
    for (int i=0; i<heads; ++i) a_log[i] = draw(rng);
    for (int i=0; i<heads*dim; ++i) bias[i] = draw(rng);
    for (size_t i=0; i<state_size; ++i) base_state[i] = fast_state[i] = draw(rng);
    kda_recurrent_bf16<<<heads, dim>>>(qkv,gate,beta,a_log,bias,base_state,base_out,rows,heads,dim,-5.f);
    kda_preprocess_regresident<<<dim3(heads,rows), dim>>>(qkv,gate,beta,a_log,bias,q_norm,k_norm,
        decay,beta_out,rows,heads,dim,-5.f);
    kda_recurrent_bf16_regresident<<<dim3(heads,dim/4),128>>>(qkv,q_norm,k_norm,decay,
        beta_out,fast_state,fast_out,rows,heads,dim);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    float output_error=0, state_error=0;
    for (size_t i=0; i<plane; ++i) {
        float error = std::abs(float(base_out[i])-float(fast_out[i]));
        if (!std::isfinite(error)) return 1;
        output_error=std::max(error,output_error);
    }
    for (size_t i=0; i<state_size; ++i) {
        float error = std::abs(base_state[i]-fast_state[i]);
        if (!std::isfinite(error)) return 1;
        state_error=std::max(error,state_error);
    }
    std::printf("%zu BF16 outputs max absolute error %.9g, %zu FP32 state values max absolute error %.9g\n",
        plane, output_error, state_size, state_error);
    return output_error > 0.00049f || state_error > 0.00001f;
}
