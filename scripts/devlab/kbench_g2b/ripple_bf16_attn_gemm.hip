// KLOOP #51 (G2): Ripple-65B keeps attention BF16 in the NVFP4 pack, and its prefill projections run
// dense_gemm_bf16_pipelined: 32.5% of Ripple prefill GPU time (runner rocprof 43104415771). Is the shipped
// 128x128 WMMA kernel at the gfx1151 BF16 ceiling, or is rocBLAS (Tensile) materially faster at these shapes?
// Shapes: q+gate (g_proj interleaved at load) N 20480 K 5120, k / v N 1024 K 5120, o N 5120 K 10240; M = 512/2048/4096.
// Same math C[M,N] = A[M,K] * B[N,K]^T, BF16 in/out, f32 accumulate. Reports time, TFLOP/s, and max |shipped - rocBLAS|
// (a routing change would be numerics-changing -> ST-995 gate, so the diff size matters).
#include "../common/dense_gemm_bf16.cu"
#include <rocblas/rocblas.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define RB(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { printf("rocblas %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

static float bf2f(unsigned short v) { unsigned int u = (unsigned int)v << 16; float f; memcpy(&f, &u, 4); return f; }
static unsigned short f2bf(float f) { unsigned int u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

int main() {
    struct S { const char* name; int N, K; } shapes[] = {
        {"q+gate", 20480, 5120}, {"k|v", 1024, 5120}, {"o", 5120, 10240}};
    rocblas_handle h; RB(rocblas_create_handle(&h));
    for (int M : {512, 2048, 4096}) {
        for (auto& s : shapes) {
            const int N = s.N, K = s.K;
            std::vector<unsigned short> hw((size_t)N * K), hx((size_t)M * K), y0((size_t)M * N), y1((size_t)M * N);
            srand(3);
            for (auto& v : hw) v = f2bf(((rand() % 2001) - 1000) / 8000.f);
            for (auto& v : hx) v = f2bf(((rand() % 2001) - 1000) / 1000.f);
            void *W, *X, *Y0, *Y1;
            CHECK(hipMalloc(&W, hw.size() * 2)); CHECK(hipMalloc(&X, hx.size() * 2));
            CHECK(hipMalloc(&Y0, y0.size() * 2)); CHECK(hipMalloc(&Y1, y1.size() * 2));
            CHECK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            auto shipped = [&] {
                dense_gemm_bf16_pipelined<<<dim3((N + DP_N_TILE - 1) / DP_N_TILE, (M + DP_M_TILE - 1) / DP_M_TILE), DP_THREADS>>>(
                    (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y0, M, N, K);
            };
            const float a = 1.f, b = 0.f;
            // Column-major view: Y^T[N,M] = W^T(K x N col-major = [N,K] row-major) * X(K x M col-major = [M,K] row-major).
            auto roc = [&] { RB(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &a,
                W, rocblas_datatype_bf16_r, K, X, rocblas_datatype_bf16_r, K, &b, Y1, rocblas_datatype_bf16_r, N,
                Y1, rocblas_datatype_bf16_r, N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0)); };
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            auto time = [&](auto f) {
                for (int i = 0; i < 3; i++) f();
                CHECK(hipDeviceSynchronize());
                const int it = 20; CHECK(hipEventRecord(e0)); for (int i = 0; i < it; i++) f();
                CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
                float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return ms / it;
            };
            const float ts = time(shipped), tr = time(roc);
            CHECK(hipMemcpy(y0.data(), Y0, y0.size() * 2, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(y1.data(), Y1, y1.size() * 2, hipMemcpyDeviceToHost));
            double maxd = 0, maxv = 0; size_t ndiff = 0;
            for (size_t i = 0; i < y0.size(); i++) {
                const double d = fabs((double)bf2f(y0[i]) - bf2f(y1[i]));
                ndiff += y0[i] != y1[i]; maxd = fmax(maxd, d); maxv = fmax(maxv, fabs(bf2f(y1[i])));
            }
            const double fl = 2.0 * M * N * K;
            printf("  M=%-5d %-7s N=%-6d K=%-6d shipped %8.1f us %5.1f TF | rocblas %8.1f us %5.1f TF (roc/shipped %.3f) | diff %zu/%zu max %.3g (|y|max %.3g)\n",
                   M, s.name, N, K, 1000 * ts, fl / (ts / 1000) / 1e12, 1000 * tr, fl / (tr / 1000) / 1e12, tr / ts, ndiff, y0.size(), maxd, maxv);
            hipFree(W); hipFree(X); hipFree(Y0); hipFree(Y1); hipEventDestroy(e0); hipEventDestroy(e1);
        }
    }
    rocblas_destroy_handle(h);
    return 0;
}
