// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -std=c++17 -arch=sm_121 -O2 -shared -Xcompiler -fPIC tests/ling/real_fp8.cu -o /tmp/ling-real.so
#include "../../kernels/gb10/common/per_token_group_quant_fp8.cu"
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_fp8_linear.cu"
#include "../../kernels/gb10/common/fp8_gemm_t_blockscaled.cu"
extern "C" int project(const __nv_bfloat16* x, __nv_fp8_e4m3* a, float* ax,
                       const __nv_fp8_e4m3* w, const void* scales, __nv_bfloat16* y,
                       int m, int n, int k, int bf16) {
    per_token_group_quant_fp8<<<dim3(m,k/128),128>>>(x,reinterpret_cast<unsigned char*>(a),ax,m,k);
    if (m >= 16) {
        if (bf16) return int(cudaErrorInvalidValue);
        fp8_gemm_t_blockscaled<<<dim3((n+127)/128,(m+63)/64),128>>>(
            reinterpret_cast<const unsigned char*>(a), ax, reinterpret_cast<const unsigned char*>(w),
            static_cast<const float*>(scales), y, m, n, k);
    } else {
        ling_fp8_linear<<<dim3((n+7)/8,m),256>>>(a,ax,w,scales,y,n,k,bf16);
    }
    return int(cudaDeviceSynchronize());
}
