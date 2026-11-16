// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #156 (Flash-Next C8): in situ the rows_v3 kernels run ~120 GB/s with ~80 unique experts (grid y 81) and every
// compute-side twin (SGPR-A 1.32x, WMMA 1.36x in kbench) LOST in the model: they are memory-latency bound there. Ship
// LDS is ~43 KB a 256-thread block (2 x TILE_KH 64 x 272 B tiles + scales + A), so ONE block (8 waves) fits per CU.
// Twin: the ship gate_up rows body with the k tile TKH as a parameter (32 / 16) -> 2-4 blocks per CU. The per-output
// FP32 sum order over k is unchanged, so it is bit-exact. Realistic shape: U = 80 unique experts + shared, cold weights.
#include "moe_shared_expert_fused_t.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template<unsigned TKH, int MINB>
__global__ void __launch_bounds__(256, MINB) gu_rows_tkh(const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gpk,
    const unsigned long long* __restrict__ gsc, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ upk, const unsigned long long* __restrict__ usc, const float* __restrict__ us2,
    __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ u_experts, const unsigned int* __restrict__ u_rowmask,
    const unsigned char* __restrict__ u_rowslot, const unsigned int* __restrict__ u_count,
    const unsigned char* __restrict__ sgp, const unsigned char* __restrict__ sgs, float sg2, __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sup, const unsigned char* __restrict__ sus, float su2, __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int h) {
    constexpr unsigned int ROWS = 8, BS = 256, STRIDE = BS + 16;
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
    __shared__ unsigned char s_pk[2][TKH * STRIDE];
    __shared__ unsigned char s_sc[2][(TKH / 8) * STRIDE];
    __shared__ __align__(16) __nv_bfloat16 s_a[2][ROWS * TKH * 2];
    __syncthreads();
    const unsigned int mask = is_shared ? ((1u << ROWS) - 1u) : u_rowmask[u];
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TKH;
        #pragma unroll
        for (unsigned int j = lane; j < TKH * (BS / 16u); j += BS) {
            const unsigned int rr = j / (BS / 16u), cb = (j % (BS / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            *(uint4*)&s_pk[buf][rr * STRIDE + cb] = *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + cb]; }
        const unsigned int ng = TKH / 8u;
        for (unsigned int j = lane; j < ng * (BS / 16u); j += BS) {
            const unsigned int sg = j / (BS / 16u), cb = (j % (BS / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            *(uint4*)&s_sc[buf][sg * STRIDE + cb] = *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + cb]; }
        for (unsigned int j = lane; j < ROWS * (TKH * 2 / 8u); j += BS) {
            const unsigned int row = j / (TKH * 2 / 8u), chunk = j % (TKH * 2 / 8u);
            *(uint4*)&s_a[buf][row * TKH * 2 + chunk * 8] = *(const uint4*)&A[(unsigned long long)row * h + tg * TKH * 2 + chunk * 8]; }
    };
    float acc[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) acc[r] = 0.0f;
    const unsigned int ntg = (K / 2u) / TKH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        if (vc) {
            #pragma unroll
            for (unsigned int rr4 = 0; rr4 < TKH / 4u; rr4++) {
                __nv_bfloat16 av[ROWS][8];
                #pragma unroll
                for (unsigned int r = 0; r < ROWS; r++) *(uint4*)&av[r][0] = *(const uint4*)&s_a[buf][r * TKH * 2 + rr4 * 8];
                #pragma unroll
                for (unsigned int j = 0; j < 4; j++) {
                    const unsigned int rr = rr4 * 4 + j;
                    const float sc = mx_block_scale<false>(s_sc[buf][(rr >> 3) * STRIDE + lane], s2);
                    const unsigned char byte = s_pk[buf][rr * STRIDE + lane];
                    const float w_lo = s_lut[byte & 0xFu] * sc, w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
                    #pragma unroll
                    for (unsigned int r = 0; r < ROWS; r++)
                        acc[r] += __bfloat162float(av[r][2 * j]) * w_lo + __bfloat162float(av[r][2 * j + 1]) * w_hi;
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
    const unsigned N = 640, K = 2560, TOPK = 10, E = 80, U = 80, R = 8;
    const size_t pb = (size_t)(K / 2) * N, sb = (size_t)(K / 16) * N;
    srand(6); std::vector<unsigned char> hp(pb), hs(sb); for (auto& b : hp) b = rand() & 0xFF; for (auto& b : hs) b = 0x30 + (rand() & 0x0F);
    const int SETS = 3; std::vector<unsigned long long> tab[SETS][4];
    for (int st = 0; st < SETS; st++) for (unsigned e = 0; e < E; e++) for (int q = 0; q < 2; q++) {
        unsigned char *P, *S; CHECK(hipMalloc(&P, pb)); CHECK(hipMemcpy(P, hp.data(), pb, hipMemcpyHostToDevice)); CHECK(hipMalloc(&S, sb)); CHECK(hipMemcpy(S, hs.data(), sb, hipMemcpyHostToDevice));
        tab[st][2 * q].push_back((unsigned long long)P); tab[st][2 * q + 1].push_back((unsigned long long)S); }
    unsigned long long* dtab[SETS][4];
    for (int st = 0; st < SETS; st++) for (int q = 0; q < 4; q++) { CHECK(hipMalloc(&dtab[st][q], E * 8)); CHECK(hipMemcpy(dtab[st][q], tab[st][q].data(), E * 8, hipMemcpyHostToDevice)); }
    unsigned char *sgp, *sgs, *sup, *sus; for (auto* pp : {&sgp, &sup}) { CHECK(hipMalloc(pp, pb)); CHECK(hipMemcpy(*pp, hp.data(), pb, hipMemcpyHostToDevice)); }
    for (auto* pp : {&sgs, &sus}) { CHECK(hipMalloc(pp, sb)); CHECK(hipMemcpy(*pp, hs.data(), sb, hipMemcpyHostToDevice)); }
    std::vector<float> hs2(E, 0.01f); float* ds2; CHECK(hipMalloc(&ds2, E * 4)); CHECK(hipMemcpy(ds2, hs2.data(), E * 4, hipMemcpyHostToDevice));
    // 80 unique experts, each used by exactly one of the 8 rows (rows x top-10 = 80 slots, no overlap): worst case.
    std::vector<unsigned> hue(U), hmask(U, 0); std::vector<unsigned char> hslot(U * MOE_MAX_K, 0xFF);
    for (unsigned u = 0; u < U; u++) { hue[u] = u; const unsigned r = u / TOPK; hmask[u] = 1u << r; hslot[u * MOE_MAX_K + r] = u % TOPK; }
    unsigned *due, *dmask, *dcnt; unsigned char* dslot; unsigned hU = U;
    CHECK(hipMalloc(&due, U * 4)); CHECK(hipMalloc(&dmask, U * 4)); CHECK(hipMalloc(&dslot, U * MOE_MAX_K)); CHECK(hipMalloc(&dcnt, 4));
    CHECK(hipMemcpy(due, hue.data(), U * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dmask, hmask.data(), U * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dslot, hslot.data(), U * MOE_MAX_K, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dcnt, &hU, 4, hipMemcpyHostToDevice));
    std::vector<__nv_bfloat16> ha(R * K); for (auto& a : ha) a = __float2bfloat16(((rand() % 2001) - 1000) / 1000.f);
    const int NA = 16; __nv_bfloat16* dA[NA]; for (int i = 0; i < NA; i++) { CHECK(hipMalloc(&dA[i], ha.size() * 2)); CHECK(hipMemcpy(dA[i], ha.data(), ha.size() * 2, hipMemcpyHostToDevice)); }
    const int NV = 5; __nv_bfloat16 *go[NV], *uo[NV], *sgo[NV], *suo[NV];
    for (int v = 0; v < NV; v++) { CHECK(hipMalloc(&go[v], R * TOPK * N * 2)); CHECK(hipMalloc(&uo[v], R * TOPK * N * 2)); CHECK(hipMalloc(&sgo[v], R * N * 2)); CHECK(hipMalloc(&suo[v], R * N * 2));
        CHECK(hipMemset(go[v], 0, R * TOPK * N * 2)); CHECK(hipMemset(uo[v], 0, R * TOPK * N * 2)); }
    const dim3 g((N + 255) / 256, U + 1, 2), b(256);
    auto launch = [&](int v, int st, int ai) { unsigned long long** t = dtab[st];
#define ARGS_(V) dA[ai], t[0], t[1], ds2, go[V], t[2], t[3], ds2, uo[V], due, dmask, dslot, dcnt, sgp, sgs, 0.01f, sgo[V], sup, sus, 0.01f, suo[V], N, K, TOPK, K
        if (v == 0) hipLaunchKernelGGL(moe_expert_gate_up_shared_t_rows_v3_r8_b256, g, b, 0, 0, ARGS_(0), R);
        else if (v == 1) hipLaunchKernelGGL((gu_rows_tkh<64, 1>), g, b, 0, 0, ARGS_(1));
        else if (v == 2) hipLaunchKernelGGL((gu_rows_tkh<32, 2>), g, b, 0, 0, ARGS_(2));
        else if (v == 3) hipLaunchKernelGGL((gu_rows_tkh<16, 2>), g, b, 0, 0, ARGS_(3));
        else hipLaunchKernelGGL((gu_rows_tkh<32, 1>), g, b, 0, 0, ARGS_(4)); };
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const double bytes = (double)(U + 1) * 2 * (pb + sb);
    float t[NV];
    for (int rep = 0; rep < 2; rep++) for (int v = 0; v < NV; v++) { for (int i = 0; i < 3; i++) launch(v, i % SETS, i % NA); CHECK(hipEventRecord(e0)); for (int i = 0; i < 30; i++) launch(v, i % SETS, i % NA); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 30; }
    for (int v = 0; v < NV; v++) launch(v, 0, 0); CHECK(hipDeviceSynchronize());
    auto eq = [&](int v) { bool same = true; std::vector<__nv_bfloat16> a0(R * TOPK * N), a1(R * TOPK * N);
        for (auto pr : {std::make_pair(go[0], go[v]), std::make_pair(uo[0], uo[v])}) { CHECK(hipMemcpy(a0.data(), pr.first, a0.size() * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, a1.size() * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), a0.size() * 2); }
        for (auto pr : {std::make_pair(sgo[0], sgo[v]), std::make_pair(suo[0], suo[v])}) { CHECK(hipMemcpy(a0.data(), pr.first, R * N * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, R * N * 2, hipMemcpyDeviceToHost)); same &= !memcmp(a0.data(), a1.data(), R * N * 2); }
        return same; };
    printf("-- rows_v3_r8 gate_up N=%u K=%u U=%u (+shared, 16 rotating A, cold weights): ship %7.1f us (%.0f GB/s)\n", N, K, U, t[0], bytes / 1e3 / t[0]);
    const char* nm[NV] = {"", "TKH64 minb1 (=ship)", "TKH32 minb2", "TKH16 minb2", "TKH32 minb1"};
    for (int v = 1; v < NV; v++) printf("-- rows %-20s %7.1f us (%.0f GB/s) x%.2f %s\n", nm[v], t[v], bytes / 1e3 / t[v], t[0] / t[v], eq(v) ? "bit-identical" : "DIFFERS");
    return 0;
}
