// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #143 (G3 Flash-Next C1 decode): moe_expert_gate_up_shared_t (routed top-10 + shared expert, NVFP4 transposed)
// is 22% of single-user decode GPU time with silu_down 12% more (rocprof 43709734119), ~242 us/call. Each 32-thread
// block LDS-stages its B tile, but every lane re-reads A[k] (BF16) from global on every k. Twin: also stage the
// TILE_KH*2 activation values into LDS per tile; the per-output FP32 sum order is unchanged (bit-exact).
#include "moe_shared_expert_fused_t.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void gate_up_alds(const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gp, const unsigned long long* __restrict__ gs,
    const float* __restrict__ gs2, __nv_bfloat16* __restrict__ gate_out, const unsigned long long* __restrict__ upp, const unsigned long long* __restrict__ ups,
    const float* __restrict__ us2, __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ shgp, const unsigned char* __restrict__ shgs, float shgs2, __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ shup, const unsigned char* __restrict__ shus, float shus2, __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k) {
    const unsigned int expert_slot = blockIdx.y, proj = blockIdx.z; const bool is_shared = (expert_slot == top_k);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2; __nv_bfloat16* C;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_shared) { if (proj == 0) { B_packed = shgp; B_scale = shgs; s2 = shgs2; C = sh_gate_out; } else { B_packed = shup; B_scale = shus; s2 = shus2; C = sh_up_out; } }
    else { const unsigned int e = expert_indices[expert_slot];
        if (proj == 0) { B_packed = (const unsigned char*)gp[e]; B_scale = (const unsigned char*)gs[e]; s2 = gs2[e]; C = gate_out; }
        else { B_packed = (const unsigned char*)upp[e]; B_scale = (const unsigned char*)ups[e]; s2 = us2[e]; C = up_out; } }
    const unsigned int lane = threadIdx.x, ns0 = blockIdx.x * blockDim.x; const bool valid = (n < N);
    __shared__ float s_lut[16]; if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ unsigned char s_pk[2][TILE_KH * TILE_STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * TILE_STRIDE];
    __shared__ float s_a[2][TILE_KH * 2];            // the tile's activations, as float
    __syncthreads();
    const unsigned int r = lane / 2u, off = (lane % 2u) * 16u;
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        #pragma unroll
        for (unsigned int w = 0; w < TILE_KH / 16u; w++) { unsigned int rr = r + w * 16u;
            *(uint4*)&s_pk[buf][rr * TILE_STRIDE + off] = *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + off]; }
        const unsigned int ng = TILE_KH / 8u;
        if (lane < ng * 2u) { unsigned int sg = lane / 2u, off2 = (lane % 2u) * 16u;
            *(uint4*)&s_sc[buf][sg * TILE_STRIDE + off2] = *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + off2]; }
        #pragma unroll
        for (unsigned int q = 0; q < (TILE_KH * 2u) / 32u; q++) s_a[buf][lane + 32u * q] = __bfloat162float(A[r0 * 2u + lane + 32u * q]);
    };
    float acc = 0.0f; const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (valid) {
            #pragma unroll 8
            for (unsigned int rr = 0; rr < TILE_KH; rr++) {
                float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * TILE_STRIDE + lane], s2);
                unsigned char byte = s_pk[buf][rr * TILE_STRIDE + lane];
                float a_lo = s_a[buf][rr * 2], a_hi = s_a[buf][rr * 2 + 1];
                acc += a_lo * (s_lut[byte & 0xFu] * sc) + a_hi * (s_lut[(byte >> 4) & 0xFu] * sc);
            }
        }
    }
    __syncthreads();
    if (!valid) return;
    C[(is_shared ? 0ull : (unsigned long long)expert_slot * N) + n] = __float2bfloat16(acc);
}

// KLOOP #144 twin: transposed LDS tile s_pkT[n][k_half] (+ s_scT[n][group]) so each thread reads 4 consecutive k_half
// bytes per LDS load and decodes each 8-k_half scale once. The per-output FMA sequence over k_half is unchanged.
__global__ void gate_up_tlds(const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gp, const unsigned long long* __restrict__ gs,
    const float* __restrict__ gs2, __nv_bfloat16* __restrict__ gate_out, const unsigned long long* __restrict__ upp, const unsigned long long* __restrict__ ups,
    const float* __restrict__ us2, __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ shgp, const unsigned char* __restrict__ shgs, float shgs2, __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ shup, const unsigned char* __restrict__ shus, float shus2, __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k) {
    const unsigned int expert_slot = blockIdx.y, proj = blockIdx.z; const bool is_shared = (expert_slot == top_k);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2; __nv_bfloat16* C;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_shared) { if (proj == 0) { B_packed = shgp; B_scale = shgs; s2 = shgs2; C = sh_gate_out; } else { B_packed = shup; B_scale = shus; s2 = shus2; C = sh_up_out; } }
    else { const unsigned int e = expert_indices[expert_slot];
        if (proj == 0) { B_packed = (const unsigned char*)gp[e]; B_scale = (const unsigned char*)gs[e]; s2 = gs2[e]; C = gate_out; }
        else { B_packed = (const unsigned char*)upp[e]; B_scale = (const unsigned char*)ups[e]; s2 = us2[e]; C = up_out; } }
    const unsigned int lane = threadIdx.x, ns0 = blockIdx.x * blockDim.x; const bool valid = (n < N);
    constexpr unsigned int KP = TILE_KH + 4;          // row pitch (bytes) per n
    __shared__ float s_lut[16]; if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ __align__(16) unsigned char s_pkT[2][32 * KP];
    __shared__ unsigned char s_scT[2][32 * (TILE_KH / 8)];
    __shared__ float s_a[2][TILE_KH * 2];
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
        #pragma unroll
        for (unsigned int q = 0; q < (TILE_KH * 2u) / 32u; q++) s_a[buf][lane + 32u * q] = __bfloat162float(A[r0 * 2u + lane + 32u * q]);
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
                    const unsigned int rr = g8 * 8u + j;
                    const unsigned char byte = (unsigned char)(wd[j >> 2] >> ((j & 3) * 8));
                    const float a_lo = s_a[buf][rr * 2], a_hi = s_a[buf][rr * 2 + 1];
                    acc += a_lo * (s_lut[byte & 0xFu] * sc) + a_hi * (s_lut[(byte >> 4) & 0xFu] * sc);
                }
            }
        }
    }
    __syncthreads();
    if (!valid) return;
    C[(is_shared ? 0ull : (unsigned long long)expert_slot * N) + n] = __float2bfloat16(acc);
}

int main() {
    const unsigned N = 640, K = 2560, TOPK = 10, E = 16;
    const size_t pb = (size_t)(K / 2) * N, sb = (size_t)(K / 16) * N;
    srand(4); std::vector<unsigned char> hp(pb), hs(sb); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    // E experts x {gate, up} + shared {gate, up}; 8 rotation sets so the weights come from DRAM.
    const int SETS = 8; std::vector<unsigned long long> hgp[SETS], hgsx[SETS], hup[SETS], husx[SETS]; std::vector<unsigned char*> shg(SETS), shgsc(SETS), shu(SETS), shusc(SETS);
    for (int st = 0; st < SETS; st++) for (unsigned e = 0; e < E + 1; e++) for (int pj = 0; pj < 2; pj++) {
        unsigned char *P, *S; CHECK(hipMalloc(&P, pb)); CHECK(hipMemcpy(P, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&S, sb)); CHECK(hipMemcpy(S, hs.data(), sb, hipMemcpyHostToDevice));
        if (e == E) { if (pj == 0) { shg[st] = P; shgsc[st] = S; } else { shu[st] = P; shusc[st] = S; } }
        else if (pj == 0) { hgp[st].push_back((unsigned long long)P); hgsx[st].push_back((unsigned long long)S); } else { hup[st].push_back((unsigned long long)P); husx[st].push_back((unsigned long long)S); } }
    std::vector<unsigned long long*> dgp(SETS), dgs(SETS), dup(SETS), dus(SETS);
    for (int st = 0; st < SETS; st++) { for (auto* pp : {&dgp[st], &dgs[st], &dup[st], &dus[st]}) CHECK(hipMalloc(pp, E * 8));
        CHECK(hipMemcpy(dgp[st], hgp[st].data(), E * 8, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dgs[st], hgsx[st].data(), E * 8, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dup[st], hup[st].data(), E * 8, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dus[st], husx[st].data(), E * 8, hipMemcpyHostToDevice)); }
    std::vector<float> hs2(E, 0.01f); float* ds2; CHECK(hipMalloc(&ds2, E * 4)); CHECK(hipMemcpy(ds2, hs2.data(), E * 4, hipMemcpyHostToDevice));
    std::vector<unsigned> hidx(TOPK); for (unsigned i = 0; i < TOPK; i++) hidx[i] = (i * 3) % E; unsigned* didx; CHECK(hipMalloc(&didx, TOPK * 4)); CHECK(hipMemcpy(didx, hidx.data(), TOPK * 4, hipMemcpyHostToDevice));
    std::vector<__nv_bfloat16> ha(K); for (auto& a : ha) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f); __nv_bfloat16* dA; CHECK(hipMalloc(&dA, K * 2)); CHECK(hipMemcpy(dA, ha.data(), K * 2, hipMemcpyHostToDevice));
    __nv_bfloat16 *go[3], *uo[3], *sgo[3], *suo[3];
    for (int v = 0; v < 3; v++) { CHECK(hipMalloc(&go[v], TOPK * N * 2)); CHECK(hipMalloc(&uo[v], TOPK * N * 2)); CHECK(hipMalloc(&sgo[v], N * 2)); CHECK(hipMalloc(&suo[v], N * 2)); }
    const dim3 g((N + 31) / 32, TOPK + 1, 2), b(32);
    auto launch = [&](int v, int st) {
        if (v == 0) hipLaunchKernelGGL(moe_expert_gate_up_shared_t, g, b, 0, 0, dA, dgp[st], dgs[st], ds2, go[0], dup[st], dus[st], ds2, uo[0], didx, shg[st], shgsc[st], 0.01f, sgo[0], shu[st], shusc[st], 0.01f, suo[0], N, K, TOPK);
        else if (v == 2) hipLaunchKernelGGL(gate_up_tlds, g, b, 0, 0, dA, dgp[st], dgs[st], ds2, go[2], dup[st], dus[st], ds2, uo[2], didx, shg[st], shgsc[st], 0.01f, sgo[2], shu[st], shusc[st], 0.01f, suo[2], N, K, TOPK);
        else hipLaunchKernelGGL(gate_up_alds, g, b, 0, 0, dA, dgp[st], dgs[st], ds2, go[1], dup[st], dus[st], ds2, uo[1], didx, shg[st], shgsc[st], 0.01f, sgo[1], shu[st], shusc[st], 0.01f, suo[1], N, K, TOPK); };
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const double bytes = (double)(TOPK + 1) * 2 * (pb + sb);
    float t[3];
    for (int v = 0; v < 3; v++) { for (int i = 0; i < 4; i++) launch(v, i % SETS); CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) launch(v, i % SETS); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 64; }
    launch(0, 0); launch(1, 0); launch(2, 0); CHECK(hipDeviceSynchronize());
    std::vector<__nv_bfloat16> a0(TOPK * N), a1(TOPK * N), c0(N), c1(N); bool same = true;
    for (auto pr : {std::make_pair(go[0], go[1]), std::make_pair(uo[0], uo[1])}) { CHECK(hipMemcpy(a0.data(), pr.first, a0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, a1.size() * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), a0.size() * 2); }
    for (auto pr : {std::make_pair(sgo[0], sgo[1]), std::make_pair(suo[0], suo[1])}) { CHECK(hipMemcpy(c0.data(), pr.first, N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(c1.data(), pr.second, N * 2, hipMemcpyDeviceToHost)); same &= !memcmp(c0.data(), c1.data(), N * 2); }
    bool same2 = true;
    for (auto pr : {std::make_pair(go[0], go[2]), std::make_pair(uo[0], uo[2])}) { CHECK(hipMemcpy(a0.data(), pr.first, a0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, a1.size() * 2, hipMemcpyDeviceToHost)); same2 &= !memcmp(a0.data(), a1.data(), a0.size() * 2); }
    for (auto pr : {std::make_pair(sgo[0], sgo[2]), std::make_pair(suo[0], suo[2])}) { CHECK(hipMemcpy(c0.data(), pr.first, N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(c1.data(), pr.second, N * 2, hipMemcpyDeviceToHost)); same2 &= !memcmp(c0.data(), c1.data(), N * 2); }
    printf("-- transposed-LDS twin: %7.1f us (%.0f GB/s) x%.2f %s\n", t[2], bytes / 1e3 / t[2], t[0] / t[2], same2 ? "bit-identical" : "DIFFERS");
    printf("-- gate_up_shared_t N=%u K=%u top_k=%u (+shared): ship %7.1f us (%.0f GB/s) | A-in-LDS %7.1f us (%.0f GB/s) x%.2f %s\n",
           N, K, TOPK, t[0], bytes / 1e3 / t[0], t[1], bytes / 1e3 / t[1], t[0] / t[1], same ? "bit-identical" : "DIFFERS");
    return 0;
}
