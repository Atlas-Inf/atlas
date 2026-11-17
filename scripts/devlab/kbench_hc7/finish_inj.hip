// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #187 (Flash-Next C8): hc_pre_finish_x4 is 3.8% of C8 decode (68 us x 49470, grid T x 80 of 128). The injection
// contraction (4 warps x a 10240-wide BF16 dot, one warp per stream) runs in block (t, 0) AFTER that block's main work:
// a serialized ~38-42 us tail (kbench_hc3) per token. Twins, bit-exact by construction (each (t, stream) is still one
// warp walking j = lane, lane+32, ... in order, then the same shfl tree; the main part is untouched):
//  (a) INJ in dedicated blocks (grid y = H/32 + 1, the last y does only injection) so it overlaps the main part;
//  (b) (a) with 32 lane-strided weights in flight instead of 16.
#include "hyper_connection.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template<unsigned INJ_U>
__global__ void finish_x4_sep(const float* __restrict__ normed, const float* __restrict__ low, const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ inject_w, __nv_bfloat16* __restrict__ y_out, float* __restrict__ inj_out,
    unsigned int hidden_size, unsigned int hc, unsigned int rank) {
    const unsigned int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, H = hidden_size, hc_dim = hc * H;
    const float* nx = normed + (size_t)t * hc_dim; const float inv_hc = 1.0f / (float)hc;
    if (blockIdx.y == (H + 31u) / 32u) {                       // dedicated injection block
        if (inject_w == nullptr) return;
        const __nv_bfloat16* row = inject_w + (size_t)warp * hc_dim;
        float acc = 0.0f; unsigned int j = lane;
        for (; j + 32 * (INJ_U - 1) < hc_dim; j += 32 * INJ_U) {
            __nv_bfloat16 w[INJ_U];
            #pragma unroll
            for (unsigned int u = 0; u < INJ_U; u++) w[u] = row[j + 32 * u];
            #pragma unroll
            for (unsigned int u = 0; u < INJ_U; u++) acc += (float)w[u] * nx[j + 32 * u];
        }
        for (; j < hc_dim; j += 32) acc += (float)row[j] * nx[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) inj_out[(size_t)t * hc + warp] = 2.0f * qhc_sigmoid(acc * inv_hc);
        return;
    }
    extern __shared__ float smem_lo[];
    __shared__ float part[4][32];
    for (unsigned int r = tid; r < rank; r += blockDim.x) smem_lo[r] = low[(size_t)t * rank + r];
    __syncthreads();
    const unsigned int d = blockIdx.y * 32u + lane;
    if (d < H) {
        const unsigned int i = warp * H + d; const __nv_bfloat16* ub = up_w + i; float acc = 0.0f; unsigned int r = 0;
        for (; r + 8 <= rank; r += 8) { float l[8]; __nv_bfloat16 u[8];
            #pragma unroll
            for (unsigned int k = 0; k < 8; ++k) { l[k] = smem_lo[r + k]; u[k] = ub[(size_t)(r + k) * hc_dim]; }
            #pragma unroll
            for (unsigned int k = 0; k < 8; ++k) acc += (float)u[k] * l[k]; }
        for (; r < rank; ++r) acc += (float)ub[(size_t)r * hc_dim] * smem_lo[r];
        part[warp][lane] = qhc_sigmoid(acc) * nx[i];
    }
    __syncthreads();
    if (warp == 0 && d < H) { float mixed = 0.0f; mixed += part[0][lane]; mixed += part[1][lane]; mixed += part[2][lane]; mixed += part[3][lane];
        y_out[(size_t)t * H + d] = __float2bfloat16(mixed * inv_hc); }
}

int main() {
    const unsigned H = 2560, HC = 4, RANK = 320, HCD = H * HC; const int NW = 8;
    std::vector<__nv_bfloat16> hu((size_t)RANK * HCD), hi((size_t)HC * HCD); srand(5);
    for (auto& x : hu) x = __float2bfloat16(((rand() % 2001) - 1000) / 8000.f); for (auto& x : hi) x = __float2bfloat16(((rand() % 2001) - 1000) / 8000.f);
    __nv_bfloat16 *du[NW], *di[NW];
    for (int i = 0; i < NW; i++) { CHECK(hipMalloc(&du[i], hu.size() * 2)); CHECK(hipMemcpy(du[i], hu.data(), hu.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&di[i], hi.size() * 2)); CHECK(hipMemcpy(di[i], hi.data(), hi.size() * 2, hipMemcpyHostToDevice)); }
    for (unsigned T : {1u, 2u, 8u}) {
        std::vector<float> hn((size_t)T * HCD), hl((size_t)T * RANK); for (auto& x : hn) x = ((rand() % 2001) - 1000) / 1000.f; for (auto& x : hl) x = ((rand() % 2001) - 1000) / 4000.f;
        float *dn, *dl; CHECK(hipMalloc(&dn, hn.size() * 4)); CHECK(hipMemcpy(dn, hn.data(), hn.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&dl, hl.size() * 4)); CHECK(hipMemcpy(dl, hl.data(), hl.size() * 4, hipMemcpyHostToDevice));
        __nv_bfloat16* dy[3]; float* dj[3]; for (int v = 0; v < 3; v++) { CHECK(hipMalloc(&dy[v], (size_t)T * H * 2)); CHECK(hipMalloc(&dj[v], (size_t)T * HC * 4)); }
        auto launch = [&](int v, int i) {
            if (v == 0) hipLaunchKernelGGL(hc_pre_finish_x4, dim3(T, H / 32), dim3(128), RANK * 4, 0, dn, dl, du[i], di[i], dy[0], dj[0], H, HC, RANK);
            else if (v == 1) hipLaunchKernelGGL(finish_x4_sep<16>, dim3(T, H / 32 + 1), dim3(128), RANK * 4, 0, dn, dl, du[i], di[i], dy[1], dj[1], H, HC, RANK);
            else hipLaunchKernelGGL(finish_x4_sep<32>, dim3(T, H / 32 + 1), dim3(128), RANK * 4, 0, dn, dl, du[i], di[i], dy[2], dj[2], H, HC, RANK); };
        hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float t[3];
        for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 3; v++) { for (int i = 0; i < 3; i++) launch(v, i % NW); CHECK(hipEventRecord(e0)); for (int i = 0; i < 40; i++) launch(v, i % NW); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 40; }
        for (int v = 0; v < 3; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
        std::vector<__nv_bfloat16> y0((size_t)T * H), y1((size_t)T * H); std::vector<float> j0((size_t)T * HC), j1((size_t)T * HC);
        CHECK(hipMemcpy(y0.data(), dy[0], y0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(j0.data(), dj[0], j0.size() * 4, hipMemcpyDeviceToHost));
        printf("-- hc_pre_finish_x4 T=%u: ship %6.1f us", T, t[0]);
        const char* nm[3] = {"", "inj-blocks U16", "inj-blocks U32"};
        for (int v = 1; v < 3; v++) { CHECK(hipMemcpy(y1.data(), dy[v], y1.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(j1.data(), dj[v], j1.size() * 4, hipMemcpyDeviceToHost));
            const bool same = !memcmp(y0.data(), y1.data(), y0.size() * 2) && !memcmp(j0.data(), j1.data(), j0.size() * 4);
            printf(" | %s %6.1f x%.2f %s", nm[v], t[v], t[0] / t[v], same ? "same" : "DIFF"); }
        printf("\n");
        CHECK(hipFree(dn)); CHECK(hipFree(dl)); for (int v = 0; v < 3; v++) { CHECK(hipFree(dy[v])); CHECK(hipFree(dj[v])); }
    }
    return 0;
}
