// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #135 (G3 Flash-Next C1 decode): hc_pre_finish_x4 is 13.2% of single-user decode GPU time (~109 us/call,
// rocprof 43677480400). Block (t, 0) also computes the injection vector: 4 warps x a 10240-wide BF16 row with one
// 2-byte load per FMA (the latency-bound pattern hc_pre_down had; #373 fixed it 2.7x by prefetching 16 lane-strided
// weights). Twin: in-order register prefetch of UI weights in the injection loop and UP rank steps in the up loop.
// Same per-accumulator order everywhere, so outputs must be BIT-IDENTICAL (y_out and inj_out).
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int UP, int UI>
__global__ void finish_u(const float* __restrict__ normed, const float* __restrict__ low, const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ inject_w, __nv_bfloat16* __restrict__ y_out, float* __restrict__ inj_out,
    const unsigned int hidden_size, const unsigned int hc, const unsigned int rank) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned int H = hidden_size, hc_dim = hc * H;
    const float* nx = normed + (size_t)t * hc_dim;
    const float inv_hc = 1.0f / (float)hc;
    extern __shared__ float smem_lo[];
    __shared__ float part[4][32];
    for (unsigned int r = tid; r < rank; r += blockDim.x) smem_lo[r] = low[(size_t)t * rank + r];
    __syncthreads();
    const unsigned int d = blockIdx.y * 32u + lane;
    if (d < H) {
        const unsigned int i = warp * H + d;
        const __nv_bfloat16* ub = up_w + i;
        float acc = 0.0f; unsigned int r = 0;
        for (; r + UP <= rank; r += UP) {
            float l[UP]; __nv_bfloat16 u[UP];
            #pragma unroll
            for (unsigned int k = 0; k < UP; ++k) { l[k] = smem_lo[r + k]; u[k] = ub[(size_t)(r + k) * hc_dim]; }
            #pragma unroll
            for (unsigned int k = 0; k < UP; ++k) acc += (float)u[k] * l[k];
        }
        for (; r < rank; ++r) acc += (float)ub[(size_t)r * hc_dim] * smem_lo[r];
        part[warp][lane] = qhc_sigmoid(acc) * nx[i];
    }
    __syncthreads();
    if (warp == 0 && d < H) {
        float mixed = 0.0f; mixed += part[0][lane]; mixed += part[1][lane]; mixed += part[2][lane]; mixed += part[3][lane];
        y_out[(size_t)t * H + d] = __float2bfloat16(mixed * inv_hc);
    }
    if (inject_w != nullptr && blockIdx.y == 0) {
        const __nv_bfloat16* row = inject_w + (size_t)warp * hc_dim;
        float acc = 0.0f; unsigned int j = lane;
        for (; j + 32 * (UI - 1) < hc_dim; j += 32 * UI) {
            __nv_bfloat16 w[UI];
            #pragma unroll
            for (int u = 0; u < UI; u++) w[u] = row[j + 32 * u];
            #pragma unroll
            for (int u = 0; u < UI; u++) acc += (float)w[u] * nx[j + 32 * u];
        }
        for (; j < hc_dim; j += 32) acc += (float)row[j] * nx[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) inj_out[(size_t)t * hc + warp] = 2.0f * qhc_sigmoid(acc * inv_hc);
    }
}

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HD = H * HC; const int copies = 16;
    std::vector<__nv_bfloat16> hu((size_t)RANK * HD), hi((size_t)HC * HD); srand(9);
    for (auto& v : hu) v = __float2bfloat16(((rand() % 2001) - 1000) / 40000.f); for (auto& v : hi) v = __float2bfloat16(((rand() % 2001) - 1000) / 40000.f);
    std::vector<__nv_bfloat16*> U(copies), I(copies);
    for (int c = 0; c < copies; c++) { CHECK(hipMalloc(&U[c], hu.size() * 2)); CHECK(hipMemcpy(U[c], hu.data(), hu.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&I[c], hi.size() * 2)); CHECK(hipMemcpy(I[c], hi.data(), hi.size() * 2, hipMemcpyHostToDevice)); }
    for (unsigned T : {1u, 2u, 4u}) {
        std::vector<float> hn((size_t)T * HD), hl((size_t)T * RANK); for (auto& v : hn) v = ((rand() % 2001) - 1000) / 1000.f; for (auto& v : hl) v = ((rand() % 2001) - 1000) / 1000.f;
        float *N, *L, *J0, *J1; __nv_bfloat16 *Y0, *Y1;
        CHECK(hipMalloc(&N, hn.size() * 4)); CHECK(hipMemcpy(N, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&L, hl.size() * 4)); CHECK(hipMemcpy(L, hl.data(), hl.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&Y0, (size_t)T * H * 2)); CHECK(hipMalloc(&Y1, (size_t)T * H * 2)); CHECK(hipMalloc(&J0, T * HC * 4)); CHECK(hipMalloc(&J1, T * HC * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        const dim3 g(T, (H + 31) / 32), b(128); const size_t sm = RANK * 4;
        auto tm = [&](auto f) { for (int i = 0; i < 4; i++) f(i % copies); CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) f(i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / 64; };
        auto ship = [&](int c) { hipLaunchKernelGGL(hc_pre_finish_x4, g, b, sm, 0, N, L, U[c], I[c], Y0, J0, H, HC, RANK); };
        const float ts = tm(ship); ship(0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> y0((size_t)T * H), y1((size_t)T * H); std::vector<float> j0(T * HC), j1(T * HC);
        CHECK(hipMemcpy(y0.data(), Y0, y0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(j0.data(), J0, j0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- T=%u ship hc_pre_finish_x4: %7.1f us\n", T, ts);
        struct V { const char* nm; void (*f)(const float*, const float*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, unsigned, unsigned, unsigned); } vs[] = {
            {"UP8  UI16", finish_u<8, 16>}, {"UP16 UI16", finish_u<16, 16>}, {"UP32 UI16", finish_u<32, 16>}, {"UP16 UI32", finish_u<16, 32>}, {"UP8  UI1 (=ship)", finish_u<8, 1>}};
        for (auto& v : vs) {
            auto f = [&](int c) { hipLaunchKernelGGL(v.f, g, b, sm, 0, N, L, U[c], I[c], Y1, J1, H, HC, RANK); };
            const float t = tm(f); CHECK(hipMemset(Y1, 0xFF, y1.size() * 2)); CHECK(hipMemset(J1, 0xFF, j1.size() * 4)); f(0); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(y1.data(), Y1, y1.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(j1.data(), J1, j1.size() * 4, hipMemcpyDeviceToHost));
            const bool same = !memcmp(y0.data(), y1.data(), y0.size() * 2) && !memcmp(j0.data(), j1.data(), j0.size() * 4);
            printf("   %-17s %7.1f us  x%.2f  %s\n", v.nm, t, ts / t, same ? "bit-identical" : "DIFFERS");
        }
        hipFree(N); hipFree(L); hipFree(Y0); hipFree(Y1); hipFree(J0); hipFree(J1);
    }
    return 0;
}
