// SPDX-License-Identifier: AGPL-3.0-only
// kbench (G3/G4 Flash-Next C8 decode): hc_pre_down is 14.3% of a strix C8 decode step (runner 42915863015,
// 324 us/call) because the shipped single-pass kernel launches one block per TOKEN and every block re-reads the
// whole 6.55 MB [rank=320, hc*H=10240] BF16 down_w. Candidates must be BIT-IDENTICAL to it:
//
//   plain   the shipped kernel and geometry (grid [T, clamp(48/T,1,10)], 1024 threads, hc_dim*4 B dynamic LDS)
//   tiled   the shipped hc_pre_down_tiled (prefill path, T > 8) forced onto decode widths
//   rt      one warp per row r over ALL T tokens: lane walks i = lane, lane+32, ... once, each weight feeds T
//           accumulators, normed read from global (T x 40 KB, cache-resident). Same per-(t, r) walk order and the
//           same shfl reduce as plain, so bit-identical; down_w is read once per call instead of T times.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int TMAX>
__global__ void hc_pre_down_rt(const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w,
                               float* __restrict__ low_out, unsigned int hidden_size, unsigned int hc,
                               unsigned int rank, unsigned int num_tokens) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rank) return;
    const unsigned int hc_dim = hc * hidden_size;
    const float inv_hc = 1.0f / (float)hc;
    const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
    float acc[TMAX];
    #pragma unroll
    for (int t = 0; t < TMAX; ++t) acc[t] = 0.0f;
    for (unsigned int i = lane; i < hc_dim; i += 32) {
        const float w = (float)row[i];
        #pragma unroll
        for (int t = 0; t < TMAX; ++t)
            if (t < (int)num_tokens) acc[t] += w * normed[(size_t)t * hc_dim + i];
    }
    #pragma unroll
    for (int t = 0; t < TMAX; ++t) {
        if (t >= (int)num_tokens) continue;
        float a = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFFULL, a, off);
        if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(a * inv_hc);
    }
}

int main() {
    const unsigned H = 2560, HC = 4, R = 320, D = H * HC;
    const int ROT = 8;  // rotate weight copies so each call reads DRAM, not a warm cache
    std::vector<uint16_t> hw((size_t)R * D);
    std::vector<float> hn((size_t)8 * D);
    srand(7);
    for (auto& v : hw) v = (uint16_t)(0x3c00u + (rand() % 0x0800u) - 0x0400u) ^ ((rand() & 1) << 15);
    for (auto& v : hn) v = (float)rand() / RAND_MAX - 0.5f;
    __nv_bfloat16* dw[ROT]; float *dn, *o0, *o1;
    for (int k = 0; k < ROT; ++k) { CHECK(hipMalloc(&dw[k], hw.size() * 2)); CHECK(hipMemcpy(dw[k], hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&o0, 8 * R * 4)); CHECK(hipMalloc(&o1, 8 * R * 4));
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    const double wbytes = (double)R * D * 2;
    auto run = [&](const char* name, unsigned T, float* out, auto launch) {
        for (int k = 0; k < ROT; ++k) launch(dw[k], out);
        CHECK(hipDeviceSynchronize());
        const int N = 200;
        CHECK(hipEventRecord(a));
        for (int it = 0; it < N; ++it) launch(dw[it % ROT], out);
        CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b));
        float ms; CHECK(hipEventElapsedTime(&ms, a, b));
        const double us = ms * 1000.0 / N;
        printf("  T=%u %-10s %8.1f us  %6.1f GB/s(weight once)", T, name, us, wbytes / (us * 1e3));
        return us;
    };
    for (unsigned T : {1u, 2u, 4u, 8u}) {
        const unsigned ds = std::min(10u, std::max(1u, 48u / T));
        double up = run("plain", T, o0, [&](__nv_bfloat16* w, float* o) {
            hipLaunchKernelGGL(hc_pre_down, dim3(T, ds), dim3(1024), D * 4, 0, dn, w, o, H, HC, R, T); });
        printf("\n");
        std::vector<float> ref(T * R), got(T * R);
        CHECK(hipMemcpy(ref.data(), o0, T * R * 4, hipMemcpyDeviceToHost));
        auto cmp = [&](double us) {
            CHECK(hipMemcpy(got.data(), o1, T * R * 4, hipMemcpyDeviceToHost));
            printf("  %5.2fx  %s\n", up / us, memcmp(ref.data(), got.data(), T * R * 4) ? "DIFFERS" : "bit-identical");
            CHECK(hipMemset(o1, 0, 8 * R * 4));
        };
        cmp(run("tiled", T, o1, [&](__nv_bfloat16* w, float* o) {
            hipLaunchKernelGGL(hc_pre_down_tiled, dim3((T + 7) / 8, 10), dim3(1024), 0, 0, dn, w, o, H, HC, R, T); }));
        for (unsigned wpb : {2u, 4u, 8u}) {
            char nm[32]; snprintf(nm, sizeof nm, "rt_w%u", wpb);
            cmp(run(nm, T, o1, [&](__nv_bfloat16* w, float* o) {
                hipLaunchKernelGGL(hc_pre_down_rt<8>, dim3((R + wpb - 1) / wpb), dim3(32 * wpb), 0, 0, dn, w, o, H, HC, R, T); }));
        }
    }
    return 0;
}
