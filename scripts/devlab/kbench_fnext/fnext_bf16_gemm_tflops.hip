// KLOOP #25 (G3): Flash-Next keeps linear_attn / self_attn / shared_expert / hyper-connection weights BF16 in the NVFP4
// pack, and those run as Tensile Cijk_Alik_Bljk BF16 GEMMs: 18-21% of prefill GPU time (runner profiles 42825832209,
// 42835428664). Is that at the gfx1151 BF16 ceiling (then quantizing them is the only lever), or underfed?
// rocblas_gemm_ex, Y[N,M] = W[K,N]^T X[K,M] (the Alik layout), BF16 in/out, f32 compute, at M = 512/2048/4096 tokens.
#include <rocblas/rocblas.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define RB(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { printf("rocblas %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    struct S { const char* name; int N, K; } shapes[] = {
        {"gdn in_proj qkvz", 16384, 2560}, {"gdn out_proj", 2560, 6144}, {"qsa q (gated)", 12288, 2560},
        {"qsa kv", 1024, 2560}, {"qsa o", 2560, 6144}, {"shared gate+up", 1280, 2560}, {"shared down", 2560, 640},
        {"square 4096 (ceiling ref)", 4096, 4096}};
    rocblas_handle h; RB(rocblas_create_handle(&h));
    for (int M : {512, 2048, 4096}) {
        for (auto& s : shapes) {
            const int N = s.N, K = s.K;
            std::vector<hip_bfloat16> hw((size_t)K * N), hx((size_t)K * M);
            srand(3); for (auto& v : hw) v = hip_bfloat16(((rand() % 2001) - 1000) / 8000.f); for (auto& v : hx) v = hip_bfloat16(((rand() % 2001) - 1000) / 1000.f);
            hip_bfloat16 *W, *X, *Y; CHECK(hipMalloc(&W, hw.size() * 2)); CHECK(hipMalloc(&X, hx.size() * 2)); CHECK(hipMalloc(&Y, (size_t)N * M * 2));
            CHECK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            const float a = 1.f, b = 0.f;
            auto run = [&] { RB(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &a,
                W, rocblas_datatype_bf16_r, K, X, rocblas_datatype_bf16_r, K, &b, Y, rocblas_datatype_bf16_r, N,
                Y, rocblas_datatype_bf16_r, N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0)); };
            for (int i = 0; i < 3; i++) run();
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int it = 20; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) run(); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= it;
            printf("  M=%-5d %-26s N=%-6d K=%-5d %8.1f us  %6.1f TFLOP/s\n", M, s.name, N, K, 1000 * ms, 2.0 * M * N * K / (ms / 1000) / 1e12);
            hipFree(W); hipFree(X); hipFree(Y); hipEventDestroy(e0); hipEventDestroy(e1);
        }
    }
    rocblas_destroy_handle(h);
    return 0;
}
