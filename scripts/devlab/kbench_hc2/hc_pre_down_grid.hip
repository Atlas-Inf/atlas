// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #129 (G3 Flash-Next C1 decode): hc_pre_down + hc_pre_finish_x4 are 27% of single-user decode GPU time
// (rocprof 43677480400; hc_pre_down ~114 us/call). At T=1 the call site launches dsplit = min(48/T, 10) = 10 blocks
// of 1024 threads, so 10 of 40 CUs stream the 6.5 MB down_w (rank 320 x hc*H 10240 BF16), ~57 GB/s. Same kernel,
// more and smaller blocks (rows still one warp each, same per-lane FMA order), so outputs must be BIT-IDENTICAL.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
// Unrolled twin: each lane prefetches U lane-strided BF16 (i, i+32, .., i+32(U-1)) into registers, then runs the SAME
// in-order FMAs (acc += row[i] * nx[i], ascending i), so the per-lane sum and the shfl tree are unchanged (bit-exact).
template <int U>
__global__ void hc_pre_down_u(const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w, float* __restrict__ low_out,
    const unsigned int hidden_size, const unsigned int hc, const unsigned int rank, const unsigned int num_tokens) {
    const unsigned int t = blockIdx.x;
    if (t >= num_tokens) return;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned int hc_dim = hc * hidden_size;
    const float inv_hc = 1.0f / (float)hc;
    extern __shared__ float smem_nx[];
    for (unsigned int i = threadIdx.x; i < hc_dim; i += blockDim.x) smem_nx[i] = normed[(size_t)t * hc_dim + i];
    __syncthreads();
    const unsigned int rows_per_split = (rank + gridDim.y - 1) / gridDim.y;
    const unsigned int r0 = blockIdx.y * rows_per_split, r1 = min(r0 + rows_per_split, rank);
    for (unsigned int r = r0 + warp; r < r1; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        unsigned int i = lane;
        for (; i + 32 * (U - 1) < hc_dim; i += 32 * U) {
            __nv_bfloat16 w[U];
            #pragma unroll
            for (int u = 0; u < U; u++) w[u] = row[i + 32 * u];
            #pragma unroll
            for (int u = 0; u < U; u++) acc += (float)w[u] * smem_nx[i + 32 * u];
        }
        for (; i < hc_dim; i += 32) acc += (float)row[i] * smem_nx[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(acc * inv_hc);
    }
}

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HD = H * HC;
    const int copies = 16;   // rotate 16 x 6.5 MB weights past L2
    std::vector<__nv_bfloat16> hw((size_t)RANK * HD); srand(5); for (auto& v : hw) v = __float2bfloat16(((rand() % 2001) - 1000) / 40000.f);
    std::vector<__nv_bfloat16*> W(copies); for (auto& p : W) { CHECK(hipMalloc(&p, hw.size() * 2)); CHECK(hipMemcpy(p, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    for (unsigned T : {1u, 2u, 4u}) {
        std::vector<float> hn((size_t)T * HD); for (auto& v : hn) v = ((rand() % 2001) - 1000) / 1000.f;
        float *N, *L0, *L1; CHECK(hipMalloc(&N, hn.size() * 4)); CHECK(hipMemcpy(N, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&L0, (size_t)T * RANK * 4)); CHECK(hipMalloc(&L1, (size_t)T * RANK * 4));
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
        auto run = [&](unsigned ds, unsigned bs, float* out, int c) { hipLaunchKernelGGL(hc_pre_down, dim3(T, ds), dim3(bs), HD * 4, 0, N, W[c], out, H, HC, RANK, T); };
        auto tm = [&](unsigned ds, unsigned bs) { for (int i = 0; i < 4; i++) run(ds, bs, L1, i % copies); CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) run(ds, bs, L1, i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); return 1000.f * ms / 64; };
        const unsigned ship = (48 / T) < 10 ? (48 / T) : 10;
        run(ship, 1024, L0, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> r0((size_t)T * RANK), r1((size_t)T * RANK); CHECK(hipMemcpy(r0.data(), L0, r0.size() * 4, hipMemcpyDeviceToHost));
        const float tship = tm(ship, 1024);
        printf("-- T=%u ship dsplit %u x 1024: %7.1f us (%.0f GB/s)\n", T, ship, tship, hw.size() * 2 / 1e3 / tship);
        struct V { unsigned ds, bs; } vs[] = {{20, 512}, {40, 256}};
        for (auto v : vs) {
            const float t = tm(v.ds, v.bs); CHECK(hipMemset(L1, 0xFF, r1.size() * 4)); run(v.ds, v.bs, L1, 0); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(r1.data(), L1, r1.size() * 4, hipMemcpyDeviceToHost));
            printf("   dsplit %3u x %4u: %7.1f us (%.0f GB/s) x%.2f  %s\n", v.ds, v.bs, t, hw.size() * 2 / 1e3 / t, tship / t, memcmp(r0.data(), r1.data(), r0.size() * 4) ? "DIFFERS" : "bit-identical");
        }
        struct U { const char* nm; unsigned ds, bs; void (*f)(const float*, const __nv_bfloat16*, float*, unsigned, unsigned, unsigned, unsigned); } us[] = {
            {"u4 ship-grid", ship, 1024, hc_pre_down_u<4>}, {"u8 ship-grid", ship, 1024, hc_pre_down_u<8>}, {"u16 ship-grid", ship, 1024, hc_pre_down_u<16>},
            {"u8 20x512", 20, 512, hc_pre_down_u<8>}, {"u16 20x512", 20, 512, hc_pre_down_u<16>}, {"u8 40x256", 40, 256, hc_pre_down_u<8>}};
        for (auto& u : us) {
            auto ru = [&](float* out, int c) { hipLaunchKernelGGL(u.f, dim3(T, u.ds), dim3(u.bs), HD * 4, 0, N, W[c], out, H, HC, RANK, T); };
            for (int i = 0; i < 4; i++) ru(L1, i % copies);
            CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) ru(L1, i % copies); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); const float t = 1000.f * ms / 64;
            CHECK(hipMemset(L1, 0xFF, r1.size() * 4)); ru(L1, 0); CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(r1.data(), L1, r1.size() * 4, hipMemcpyDeviceToHost));
            printf("   %-14s: %7.1f us (%.0f GB/s) x%.2f  %s\n", u.nm, t, hw.size() * 2 / 1e3 / t, tship / t, memcmp(r0.data(), r1.data(), r0.size() * 4) ? "DIFFERS" : "bit-identical");
        }
        hipFree(N); hipFree(L0); hipFree(L1);
    }
    return 0;
}
