// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #145 (G3 Flash-Next C1 decode): moe_expert_silu_down_shared_t is ~12% of single-user decode (43709734119).
// gate_up's transposed-LDS twin was 1.35x bit-identical (43764631447); same transform here: stage the packed tile
// transposed (s_pkT[n][k_half]) so each lane reads 8 k_half per uint2 and decodes each scale once. s_act already in LDS.
#include "moe_shared_expert_fused_t.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void silu_down_tlds(const __nv_bfloat16* __restrict__ gate_out, const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ pp, const unsigned long long* __restrict__ sp, const float* __restrict__ s2v,
    __nv_bfloat16* __restrict__ C, const unsigned int* __restrict__ expert_indices, const __nv_bfloat16* __restrict__ sh_g,
    const __nv_bfloat16* __restrict__ sh_u, const unsigned char* __restrict__ shp, const unsigned char* __restrict__ shs, float shs2,
    __nv_bfloat16* __restrict__ sh_out, unsigned int N, unsigned int K, unsigned int top_k) {
    const unsigned int expert_slot = blockIdx.y; const bool is_shared = (expert_slot == top_k);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2; const __nv_bfloat16* g_ptr; const __nv_bfloat16* u_ptr;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_shared) { B_packed = shp; B_scale = shs; s2 = shs2; g_ptr = sh_g; u_ptr = sh_u; }
    else { const unsigned int e = expert_indices[expert_slot]; B_packed = (const unsigned char*)pp[e]; B_scale = (const unsigned char*)sp[e]; s2 = s2v[e];
        g_ptr = gate_out + (unsigned long long)expert_slot * K; u_ptr = up_out + (unsigned long long)expert_slot * K; }
    const unsigned int lane = threadIdx.x, ns0 = blockIdx.x * blockDim.x; const bool valid = (n < N);
    extern __shared__ float s_act[];
    for (unsigned int i = lane; i < K; i += blockDim.x) { float gf = __bfloat162float(g_ptr[i]), uf = __bfloat162float(u_ptr[i]); s_act[i] = (gf / (1.0f + __expf(-gf))) * uf; }
    constexpr unsigned int KP = TILE_KH + 4;
    __shared__ float s_lut[16]; if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ __align__(16) unsigned char s_pkT[2][32 * KP];
    __shared__ unsigned char s_scT[2][32 * (TILE_KH / 8)];
    __syncthreads();
    const unsigned int r = lane / 2u, off = (lane % 2u) * 16u;
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        #pragma unroll
        for (unsigned int w = 0; w < TILE_KH / 16u; w++) { unsigned int rr = r + w * 16u;
            const uint4 v = *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + off];
            const unsigned char* vb = (const unsigned char*)&v;
            #pragma unroll
            for (int j = 0; j < 16; j++) s_pkT[buf][(off + j) * KP + rr] = vb[j]; }
        const unsigned int ng = TILE_KH / 8u;
        if (lane < ng * 2u) { unsigned int sg = lane / 2u, off2 = (lane % 2u) * 16u;
            const uint4 v = *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + off2];
            const unsigned char* vb = (const unsigned char*)&v;
            #pragma unroll
            for (int j = 0; j < 16; j++) s_scT[buf][(off2 + j) * ng + sg] = vb[j]; }
    };
    float acc = 0.0f; const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (valid) {
            #pragma unroll 2
            for (unsigned int g8 = 0; g8 < TILE_KH / 8u; g8++) {
                const float sc = mx_block_scale<false>(s_scT[buf][lane * (TILE_KH / 8u) + g8], s2);
                const uint2 w8 = *(const uint2*)&s_pkT[buf][lane * KP + g8 * 8u];
                const unsigned int wd[2] = {w8.x, w8.y};
                #pragma unroll
                for (unsigned int j = 0; j < 8; j++) {
                    const unsigned int k_half = tg * TILE_KH + g8 * 8u + j;
                    const unsigned char byte = (unsigned char)(wd[j >> 2] >> ((j & 3) * 8));
                    acc += s_act[k_half * 2] * (s_lut[byte & 0xFu] * sc) + s_act[k_half * 2 + 1] * (s_lut[(byte >> 4) & 0xFu] * sc);
                }
            }
        }
    }
    __syncthreads();
    if (!valid) return;
    if (is_shared) sh_out[n] = __float2bfloat16(acc); else C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
}

int main() {
    const unsigned N = 2560, K = 640, TOPK = 10, E = 16;
    const size_t pb = (size_t)(K / 2) * N, sb = (size_t)(K / 16) * N;
    srand(5); std::vector<unsigned char> hp(pb), hs(sb); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    const int SETS = 8; std::vector<unsigned long long> hpp[SETS], hsp[SETS]; std::vector<unsigned char*> shp(SETS), shs(SETS);
    for (int st = 0; st < SETS; st++) for (unsigned e = 0; e < E + 1; e++) {
        unsigned char *P, *S; CHECK(hipMalloc(&P, pb)); CHECK(hipMemcpy(P, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&S, sb)); CHECK(hipMemcpy(S, hs.data(), sb, hipMemcpyHostToDevice));
        if (e == E) { shp[st] = P; shs[st] = S; } else { hpp[st].push_back((unsigned long long)P); hsp[st].push_back((unsigned long long)S); } }
    std::vector<unsigned long long*> dpp(SETS), dsp(SETS);
    for (int st = 0; st < SETS; st++) { CHECK(hipMalloc(&dpp[st], E * 8)); CHECK(hipMalloc(&dsp[st], E * 8));
        CHECK(hipMemcpy(dpp[st], hpp[st].data(), E * 8, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dsp[st], hsp[st].data(), E * 8, hipMemcpyHostToDevice)); }
    std::vector<float> hs2(E, 0.01f); float* ds2; CHECK(hipMalloc(&ds2, E * 4)); CHECK(hipMemcpy(ds2, hs2.data(), E * 4, hipMemcpyHostToDevice));
    std::vector<unsigned> hidx(TOPK); for (unsigned i = 0; i < TOPK; i++) hidx[i] = (i * 3) % E; unsigned* didx; CHECK(hipMalloc(&didx, TOPK * 4)); CHECK(hipMemcpy(didx, hidx.data(), TOPK * 4, hipMemcpyHostToDevice));
    std::vector<__nv_bfloat16> hg((TOPK + 1) * K), hu((TOPK + 1) * K);
    for (auto& a : hg) a = __float2bfloat16(((rand() % 2001) - 1000) / 500.f); for (auto& a : hu) a = __float2bfloat16(((rand() % 2001) - 1000) / 500.f);
    __nv_bfloat16 *dg, *du; CHECK(hipMalloc(&dg, hg.size() * 2)); CHECK(hipMalloc(&du, hu.size() * 2));
    CHECK(hipMemcpy(dg, hg.data(), hg.size() * 2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(du, hu.data(), hu.size() * 2, hipMemcpyHostToDevice));
    const __nv_bfloat16 *dsg = dg + TOPK * K, *dsu = du + TOPK * K;
    __nv_bfloat16 *co[2], *so[2]; for (int v = 0; v < 2; v++) { CHECK(hipMalloc(&co[v], TOPK * N * 2)); CHECK(hipMalloc(&so[v], N * 2)); }
    const dim3 g((N + 31) / 32, TOPK + 1, 1), b(32); const unsigned smem = K * 4;
    auto launch = [&](int v, int st) {
        if (v == 0) hipLaunchKernelGGL(moe_expert_silu_down_shared_t, g, b, smem, 0, dg, du, dpp[st], dsp[st], ds2, co[0], didx, dsg, dsu, shp[st], shs[st], 0.01f, so[0], N, K, TOPK);
        else hipLaunchKernelGGL(silu_down_tlds, g, b, smem, 0, dg, du, dpp[st], dsp[st], ds2, co[1], didx, dsg, dsu, shp[st], shs[st], 0.01f, so[1], N, K, TOPK); };
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const double bytes = (double)(TOPK + 1) * (pb + sb);
    float t[2];
    for (int rep = 0; rep < 2; rep++) for (int v = 0; v < 2; v++) { for (int i = 0; i < 4; i++) launch(v, i % SETS); CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) launch(v, i % SETS); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 64; }
    launch(0, 0); launch(1, 0); CHECK(hipDeviceSynchronize());
    std::vector<__nv_bfloat16> a0(TOPK * N), a1(TOPK * N); bool same = true;
    CHECK(hipMemcpy(a0.data(), co[0], a0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), co[1], a1.size() * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), a0.size() * 2);
    CHECK(hipMemcpy(a0.data(), so[0], N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), so[1], N * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), N * 2);
    printf("-- silu_down_shared_t N=%u K=%u top_k=%u (+shared): ship %7.1f us (%.0f GB/s) | transposed-LDS %7.1f us (%.0f GB/s) x%.2f %s\n",
           N, K, TOPK, t[0], bytes / 1e3 / t[0], t[1], bytes / 1e3 / t[1], t[0] / t[1], same ? "bit-identical" : "DIFFERS");
    return 0;
}
