// devlab-only microbench: sustained 16x16x16 WMMA issue rate, bf16 vs f16 on
// gfx1151. One warp per CTA spins the same WMMA on register operands; reported
// as wmma/µs/warp. Build: hipcc --offload-arch=gfx1151 -O3 -x hip wmma_rate.cu
#include <cstdio>
#include <hip/hip_runtime.h>
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16f __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__global__ void rate_bf16(v8f* out, int iters) {
    v16bf a = v16bf{1}, b = v16bf{1};
    v8f acc = v8f{0,0,0,0,0,0,0,0};
    for (int i = 0; i < iters; i++)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    out[threadIdx.x] = acc;
}
__global__ void rate_f16(v8f* out, int iters) {
    v16f a = v16f{1}, b = v16f{1};
    v8f acc = v8f{0,0,0,0,0,0,0,0};
    for (int i = 0; i < iters; i++)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    out[threadIdx.x] = acc;
}

template <typename K>
double run(K k, int warps_per_cta, int ctas, int iters) {
    v8f* out; hipMalloc(&out, warps_per_cta * ctas * 32 * sizeof(v8f));
    k<<<ctas, warps_per_cta * 32>>>(out, iters); // warm
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    k<<<ctas, warps_per_cta * 32>>>(out, iters);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    hipFree(out);
    return (double)iters * warps_per_cta * ctas / (ms * 1e3); // wmma/us
}

int main() {
    const int it = 20000;
    printf("wmma/us: warps/cta,ctas\n");
    for (int w = 1; w <= 8; w *= 2) {
        double bf = run(rate_bf16, w, 40, it);
        double f = run(rate_f16, w, 40, it);
        printf("  w%d x40: bf16 %.1f  f16 %.1f  ratio %.3f\n", w, bf, f, f / bf);
    }
    return 0;
}
