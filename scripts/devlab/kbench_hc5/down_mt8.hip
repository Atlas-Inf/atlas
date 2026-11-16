// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #169 (Flash-Next C4/C8): hc_pre_down_tiled is 11.4% of C8 decode (205 us/call at T=8) and hc_pre_down 3.9% at
// C4 (69 us x T=1..4). The gb10 tree has a token-fused hc_pre_down_mt (each 6.55 MB down_w element read once for all T
// tokens, 16 lane-strided loads in flight, nx from L2), but strix-hip never got it and gb10 caps it at T <= 4. Twin:
// that kernel with HC_MT_MAX as a template (4 / 8). Each (t, r) is still one warp walking i = lane, lane+32, ... in
// order into one FP32 accumulator, then the same shuffle tree: BIT-IDENTICAL to hc_pre_down / hc_pre_down_tiled.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template<unsigned MT, unsigned UN>
__global__ void down_mt(const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w, float* __restrict__ low_out,
    unsigned int hidden_size, unsigned int hc, unsigned int rank, unsigned int num_tokens) {
    const unsigned int lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rank) return;
    const unsigned int hc_dim = hc * hidden_size; const float inv_hc = 1.0f / (float)hc;
    const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
    float acc[MT];
    #pragma unroll
    for (unsigned int t = 0; t < MT; ++t) acc[t] = 0.0f;
    unsigned int i = lane;
    for (; i + 32u * (UN - 1u) < hc_dim; i += 32u * UN) {
        float w[UN];
        #pragma unroll
        for (unsigned int k = 0; k < UN; ++k) w[k] = (float)row[i + 32u * k];
        #pragma unroll
        for (unsigned int k = 0; k < UN; ++k) {
            #pragma unroll
            for (unsigned int t = 0; t < MT; ++t) if (t < num_tokens) acc[t] += w[k] * normed[(size_t)t * hc_dim + i + 32u * k];
        }
    }
    for (; i < hc_dim; i += 32u) { const float wv = (float)row[i];
        #pragma unroll
        for (unsigned int t = 0; t < MT; ++t) if (t < num_tokens) acc[t] += wv * normed[(size_t)t * hc_dim + i]; }
    #pragma unroll
    for (unsigned int t = 0; t < MT; ++t) if (t < num_tokens) { float a = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xFFFFFFFFu, a, off);
        if (lane == 0) low_out[(size_t)t * rank + r] = qhc_silu(a * inv_hc); }
}

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HCD = H * HC; const int NW = 8;
    std::vector<__nv_bfloat16> hw((size_t)RANK * HCD); srand(3); for (auto& x : hw) x = __float2bfloat16(((rand() % 2001) - 1000) / 8000.f);
    __nv_bfloat16* dw[NW]; for (int i = 0; i < NW; i++) { CHECK(hipMalloc(&dw[i], hw.size() * 2)); CHECK(hipMemcpy(dw[i], hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); }
    for (unsigned T : {1u, 2u, 4u, 5u, 8u}) {
        std::vector<float> hn((size_t)T * HCD); for (auto& x : hn) x = ((rand() % 2001) - 1000) / 1000.f;
        float* dn; CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        float* dout[5]; for (auto& p : dout) CHECK(hipMalloc(&p, (size_t)T * RANK * 4));
        const unsigned dsplit_pre = std::min(10u, std::max(1u, 48u / T)), dsplit_t = std::min(16u, RANK / 32u) * 2u;
        // v0: ship (hc_pre_down for T <= 4, hc_pre_down_tiled for T >= 5), v1: mt8 block 128, v2: mt8 block 256, v3: mt4/8 UN 8, v4: mt8 UN 32
        auto launch = [&](int v, int i) {
            if (v == 0) { if (T <= 4) hipLaunchKernelGGL(hc_pre_down, dim3(T, dsplit_pre), dim3(1024), HCD * 4, 0, dn, dw[i], dout[0], H, HC, RANK, T);
                          else hipLaunchKernelGGL(hc_pre_down_tiled, dim3(1, dsplit_t), dim3(1024), 0, 0, dn, dw[i], dout[0], H, HC, RANK, T); }
            else if (v == 1) hipLaunchKernelGGL((down_mt<8, 16>), dim3(RANK / 4), dim3(128), 0, 0, dn, dw[i], dout[1], H, HC, RANK, T);
            else if (v == 2) hipLaunchKernelGGL((down_mt<8, 16>), dim3(RANK / 8), dim3(256), 0, 0, dn, dw[i], dout[2], H, HC, RANK, T);
            else if (v == 3) { if (T <= 4) hipLaunchKernelGGL((down_mt<4, 16>), dim3(RANK / 4), dim3(128), 0, 0, dn, dw[i], dout[3], H, HC, RANK, T);
                               else hipLaunchKernelGGL((down_mt<8, 8>), dim3(RANK / 4), dim3(128), 0, 0, dn, dw[i], dout[3], H, HC, RANK, T); }
            else hipLaunchKernelGGL((down_mt<8, 32>), dim3(RANK / 4), dim3(128), 0, 0, dn, dw[i], dout[4], H, HC, RANK, T); };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[5];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 5; v++) { for (int i = 0; i < 3; i++) launch(v, i % NW); CHECK(hipEventRecord(e0)); for (int i = 0; i < 40; i++) launch(v, i % NW); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 40; }
        for (int v = 0; v < 5; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<float> o0((size_t)T * RANK), o1((size_t)T * RANK); CHECK(hipMemcpy(o0.data(), dout[0], o0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- hc down T=%u: ship %s %6.1f us", T, T <= 4 ? "pre_down" : "tiled", t[0]);
        const char* nm[5] = {"", "mt8 b128", "mt8 b256", T <= 4 ? "mt4 b128" : "mt8 un8", "mt8 un32"};
        for (int v = 1; v < 5; v++) { CHECK(hipMemcpy(o1.data(), dout[v], o1.size() * 4, hipMemcpyDeviceToHost));
            printf(" | %s %6.1f x%.2f %s", nm[v], t[v], t[0] / t[v], memcmp(o0.data(), o1.data(), o0.size() * 4) ? "DIFF" : "same"); }
        printf("\n");
        CHECK(hipFree(dn)); for (auto p : dout) CHECK(hipFree(p));
    }
    return 0;
}
