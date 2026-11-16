// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #147 (G3 Flash-Next C4/C8 decode): moe_expert_gate_up_shared_t_rows_v3_r8_b256 is 30.8% of C8 decode GPU time
// (43764651143, ~1.25 ms/call). The single-row twins went 1.35x/1.66x bit-identical once the tile was staged transposed
// (8 k_half per LDS load, one scale decode per 8). Same here for ROWS=8, BS=256: the w_lo/w_hi for each k are computed
// identically and fed to the same 8-row accumulation order, so the result is bit-exact by construction. KP = row pitch.
#include "moe_shared_expert_fused_t.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template<unsigned KP>
__global__ void __launch_bounds__(256) gu_rows_tlds(const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gpk,
    const unsigned long long* __restrict__ gsc, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ upk, const unsigned long long* __restrict__ usc, const float* __restrict__ us2,
    __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ u_experts, const unsigned int* __restrict__ u_rowmask,
    const unsigned char* __restrict__ u_rowslot, const unsigned int* __restrict__ u_count,
    const unsigned char* __restrict__ sgp, const unsigned char* __restrict__ sgs, float sg2, __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sup, const unsigned char* __restrict__ sus, float su2, __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int h) {
    constexpr unsigned int ROWS = 8, BS = 256;
    const unsigned int u = blockIdx.y, proj = blockIdx.z;
    if (u > *u_count) return;
    const bool is_shared = (u == *u_count);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2;
    if (is_shared) { if (proj == 0) { B_packed = sgp; B_scale = sgs; s2 = sg2; } else { B_packed = sup; B_scale = sus; s2 = su2; } }
    else { const unsigned int eid = u_experts[u];
        if (proj == 0) { B_packed = (const unsigned char*)gpk[eid]; B_scale = (const unsigned char*)gsc[eid]; s2 = gs2[eid]; }
        else { B_packed = (const unsigned char*)upk[eid]; B_scale = (const unsigned char*)usc[eid]; s2 = us2[eid]; } }
    const unsigned int lane = threadIdx.x, ns0 = blockIdx.x * BS, col = ns0 + lane; const bool vc = col < N;
    __shared__ float s_lut[16]; if (lane < 16) s_lut[lane] = E2M1_LUT_T[lane];
    __shared__ __align__(16) unsigned char s_pkT[2][BS * KP];
    __shared__ unsigned char s_scT[2][BS * (TILE_KH / 8)];
    __shared__ __nv_bfloat16 s_a[2][ROWS * TILE_KH * 2];
    __syncthreads();
    const unsigned int mask = is_shared ? ((1u << ROWS) - 1u) : u_rowmask[u];
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        #pragma unroll
        for (unsigned int j = lane; j < TILE_KH * (BS / 16u); j += BS) {
            const unsigned int rr = j / (BS / 16u), cb = (j % (BS / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            const uint4 v = *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + cb];
            const unsigned char* vb = (const unsigned char*)&v;
            #pragma unroll
            for (int i = 0; i < 16; i++) s_pkT[buf][(cb + i) * KP + rr] = vb[i]; }
        const unsigned int ng = TILE_KH / 8u;
        #pragma unroll
        for (unsigned int j = lane; j < ng * (BS / 16u); j += BS) {
            const unsigned int sg = j / (BS / 16u), cb = (j % (BS / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            const uint4 v = *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + cb];
            const unsigned char* vb = (const unsigned char*)&v;
            #pragma unroll
            for (int i = 0; i < 16; i++) s_scT[buf][(cb + i) * ng + sg] = vb[i]; }
        #pragma unroll
        for (unsigned int j = lane; j < ROWS * (TILE_KH * 2 / 8u); j += BS) {
            const unsigned int row = j / (TILE_KH * 2 / 8u), chunk = j % (TILE_KH * 2 / 8u);
            *(uint4*)&s_a[buf][row * TILE_KH * 2 + chunk * 8] = *(const uint4*)&A[(unsigned long long)row * h + tg * TILE_KH * 2 + chunk * 8]; }
    };
    float acc[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) acc[r] = 0.0f;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (vc) {
            #pragma unroll 2
            for (unsigned int g8 = 0; g8 < TILE_KH / 8u; g8++) {
                const float sc = mx_block_scale<false>(s_scT[buf][lane * (TILE_KH / 8u) + g8], s2);
                const uint2 w8 = *(const uint2*)&s_pkT[buf][lane * KP + g8 * 8u];
                const unsigned int wd[2] = {w8.x, w8.y};
                #pragma unroll
                for (unsigned int hq = 0; hq < 2; hq++) {
                    const unsigned int rr4 = g8 * 2u + hq;
                    __nv_bfloat16 av[ROWS][8];
                    #pragma unroll
                    for (unsigned int r = 0; r < ROWS; r++) *(uint4*)&av[r][0] = *(const uint4*)&s_a[buf][r * TILE_KH * 2 + rr4 * 8];
                    #pragma unroll
                    for (unsigned int j = 0; j < 4; j++) {
                        const unsigned char byte = (unsigned char)(wd[hq] >> (j * 8));
                        const float w_lo = s_lut[byte & 0xFu] * sc, w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
                        #pragma unroll
                        for (unsigned int r = 0; r < ROWS; r++)
                            acc[r] += __bfloat162float(av[r][2 * j]) * w_lo + __bfloat162float(av[r][2 * j + 1]) * w_hi;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (!vc) return;
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        if (!(mask & (1u << r))) continue;
        if (is_shared) { if (proj == 0) sh_gate_out[r * N + col] = __float2bfloat16(acc[r]); else sh_up_out[r * N + col] = __float2bfloat16(acc[r]); }
        else { const unsigned int slot = u_rowslot[u * MOE_MAX_K + r]; __nv_bfloat16* C = (proj == 0) ? gate_out : up_out;
            C[(unsigned long long)(r * top_k + slot) * N + col] = __float2bfloat16(acc[r]); }
    }
}

int main() {
    const unsigned N = 640, K = 2560, TOPK = 10, E = 48, U = 48, R = 8;
    const size_t pb = (size_t)(K / 2) * N, sb = (size_t)(K / 16) * N;
    srand(6); std::vector<unsigned char> hp(pb), hs(sb); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    const int SETS = 4; std::vector<unsigned long long> tab[SETS][4];
    for (int st = 0; st < SETS; st++) for (unsigned e = 0; e < E; e++) for (int q = 0; q < 2; q++) {
        unsigned char *P, *S; CHECK(hipMalloc(&P, pb)); CHECK(hipMemcpy(P, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&S, sb)); CHECK(hipMemcpy(S, hs.data(), sb, hipMemcpyHostToDevice));
        tab[st][2 * q].push_back((unsigned long long)P); tab[st][2 * q + 1].push_back((unsigned long long)S); }
    unsigned long long* dtab[SETS][4];
    for (int st = 0; st < SETS; st++) for (int q = 0; q < 4; q++) { CHECK(hipMalloc(&dtab[st][q], E * 8)); CHECK(hipMemcpy(dtab[st][q], tab[st][q].data(), E * 8, hipMemcpyHostToDevice)); }
    unsigned char *sgp, *sgs, *sup, *sus; for (auto* pp : {&sgp, &sup}) { CHECK(hipMalloc(pp, pb)); CHECK(hipMemcpy(*pp, hp.data(), pb, hipMemcpyHostToDevice)); }
    for (auto* pp : {&sgs, &sus}) { CHECK(hipMalloc(pp, sb)); CHECK(hipMemcpy(*pp, hs.data(), sb, hipMemcpyHostToDevice)); }
    std::vector<float> hs2(E, 0.01f); float* ds2; CHECK(hipMalloc(&ds2, E * 4)); CHECK(hipMemcpy(ds2, hs2.data(), E * 4, hipMemcpyHostToDevice));
    // Collision-free union: each (row, slot) owned by exactly one unique expert.
    std::vector<unsigned> hue(U), hmask(U, 0); std::vector<unsigned char> hslot(U * MOE_MAX_K, 0xFF); unsigned cnt[8] = {0};
    for (unsigned u = 0; u < U; u++) { hue[u] = u % E; for (unsigned r = 0; r < R; r++) if ((u * 7 + r * 3) % 5 < 2 && cnt[r] < TOPK) { hmask[u] |= 1u << r; hslot[u * MOE_MAX_K + r] = cnt[r]++; } }
    unsigned *due, *dmask, *dcnt; unsigned char* dslot; unsigned hU = U;
    CHECK(hipMalloc(&due, U * 4)); CHECK(hipMalloc(&dmask, U * 4)); CHECK(hipMalloc(&dslot, U * MOE_MAX_K)); CHECK(hipMalloc(&dcnt, 4));
    CHECK(hipMemcpy(due, hue.data(), U * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dmask, hmask.data(), U * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dslot, hslot.data(), U * MOE_MAX_K, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dcnt, &hU, 4, hipMemcpyHostToDevice));
    std::vector<__nv_bfloat16> ha(R * K); for (auto& a : ha) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f); __nv_bfloat16* dA; CHECK(hipMalloc(&dA, ha.size() * 2)); CHECK(hipMemcpy(dA, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    const int NV = 3; __nv_bfloat16 *go[NV], *uo[NV], *sgo[NV], *suo[NV];
    for (int v = 0; v < NV; v++) { CHECK(hipMalloc(&go[v], R * TOPK * N * 2)); CHECK(hipMalloc(&uo[v], R * TOPK * N * 2)); CHECK(hipMalloc(&sgo[v], R * N * 2)); CHECK(hipMalloc(&suo[v], R * N * 2));
        CHECK(hipMemset(go[v], 0, R * TOPK * N * 2)); CHECK(hipMemset(uo[v], 0, R * TOPK * N * 2)); }
    const dim3 g((N + 255) / 256, U + 1, 2), b(256);
    auto launch = [&](int v, int st) { unsigned long long** t = dtab[st];
        if (v == 0) hipLaunchKernelGGL(moe_expert_gate_up_shared_t_rows_v3_r8_b256, g, b, 0, 0, dA, t[0], t[1], ds2, go[0], t[2], t[3], ds2, uo[0], due, dmask, dslot, dcnt, sgp, sgs, 0.01f, sgo[0], sup, sus, 0.01f, suo[0], N, K, TOPK, K, R);
        else if (v == 1) hipLaunchKernelGGL(gu_rows_tlds<TILE_KH + 4>, g, b, 0, 0, dA, t[0], t[1], ds2, go[1], t[2], t[3], ds2, uo[1], due, dmask, dslot, dcnt, sgp, sgs, 0.01f, sgo[1], sup, sus, 0.01f, suo[1], N, K, TOPK, K);
        else hipLaunchKernelGGL(gu_rows_tlds<TILE_KH + 8>, g, b, 0, 0, dA, t[0], t[1], ds2, go[2], t[2], t[3], ds2, uo[2], due, dmask, dslot, dcnt, sgp, sgs, 0.01f, sgo[2], sup, sus, 0.01f, suo[2], N, K, TOPK, K); };
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const double bytes = (double)(U + 1) * 2 * (pb + sb);
    float t[NV];
    for (int rep = 0; rep < 2; rep++) for (int v = 0; v < NV; v++) { for (int i = 0; i < 4; i++) launch(v, i % SETS); CHECK(hipEventRecord(e0)); for (int i = 0; i < 32; i++) launch(v, i % SETS); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 32; }
    for (int v = 0; v < NV; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
    auto eq = [&](int v) { bool same = true; std::vector<__nv_bfloat16> a0(R * TOPK * N), a1(R * TOPK * N);
        for (auto pr : {std::make_pair(go[0], go[v]), std::make_pair(uo[0], uo[v])}) { CHECK(hipMemcpy(a0.data(), pr.first, a0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, a1.size() * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), a0.size() * 2); }
        for (auto pr : {std::make_pair(sgo[0], sgo[v]), std::make_pair(suo[0], suo[v])}) { CHECK(hipMemcpy(a0.data(), pr.first, R * N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, R * N * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), R * N * 2); }
        return same; };
    printf("-- gate_up rows_v3_r8_b256 N=%u K=%u U=%u (+shared): ship %7.1f us (%.0f GB/s)\n", N, K, U, t[0], bytes / 1e3 / t[0]);
    printf("-- rows transposed KP=68: %7.1f us x%.2f %s\n", t[1], t[0] / t[1], eq(1) ? "bit-identical" : "DIFFERS");
    printf("-- rows transposed KP=72: %7.1f us x%.2f %s\n", t[2], t[0] / t[2], eq(2) ? "bit-identical" : "DIFFERS");
    return 0;
}
