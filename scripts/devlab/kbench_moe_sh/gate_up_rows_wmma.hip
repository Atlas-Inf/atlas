// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #154 (Flash-Next C4/C8, user priority): moe_expert_gate_up_shared_t_rows_v3_r8_b256 is 31-33% of C8 decode and
// runs 8 rows x scalar FP32 FMA per decoded weight. SGPR-A won 1.32x in kbench but lost 9% in situ (43807908595), so
// the LDS/ALU knobs are exhausted. Twin: WMMA (gfx11 wmma_f32_16x16x16_bf16_w32). 8 waves per 256-thread block, each
// wave owns 16 output columns. Each lane decodes its column's 16-k slice to bf16 (lut x E4M3 is exact in bf16; s2 is
// applied once at the end) and one WMMA per 16 k takes 8 real + 8 zero rows. NUMERICS-CHANGING (summation order, s2
// placement) -> reports max |diff| / max |ref| and bf16 mismatches vs ship; ST-995 gates any ship.
#include "moe_shared_expert_fused_t.cu"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

template<int WAVES>
__global__ void __launch_bounds__(WAVES * 32) gu_rows_wmma(const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gpk,
    const unsigned long long* __restrict__ gsc, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ upk, const unsigned long long* __restrict__ usc, const float* __restrict__ us2,
    __nv_bfloat16* __restrict__ up_out, const unsigned int* __restrict__ u_experts, const unsigned int* __restrict__ u_rowmask,
    const unsigned char* __restrict__ u_rowslot, const unsigned int* __restrict__ u_count,
    const unsigned char* __restrict__ sgp, const unsigned char* __restrict__ sgs, float sg2, __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sup, const unsigned char* __restrict__ sus, float su2, __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k, unsigned int h) {
    constexpr unsigned int ROWS = 8, BS = WAVES * 32, COLW = WAVES * 16, STRIDE = COLW + 16;
    const unsigned int u = blockIdx.y, proj = blockIdx.z;
    if (u > *u_count) return;
    const bool is_shared = (u == *u_count);
    const unsigned char* B_packed; const unsigned char* B_scale; float s2;
    if (is_shared) { if (proj == 0) { B_packed = sgp; B_scale = sgs; s2 = sg2; } else { B_packed = sup; B_scale = sus; s2 = su2; } }
    else { const unsigned int eid = u_experts[u];
        if (proj == 0) { B_packed = (const unsigned char*)gpk[eid]; B_scale = (const unsigned char*)gsc[eid]; s2 = gs2[eid]; }
        else { B_packed = (const unsigned char*)upk[eid]; B_scale = (const unsigned char*)usc[eid]; s2 = us2[eid]; } }
    const unsigned int tid = threadIdx.x, wave = tid / 32u, l = tid % 32u, ns0 = blockIdx.x * COLW;
    const unsigned int ccol = wave * 16u + (l & 15u);          // this lane's column within the block
    __shared__ float s_lut[16]; if (tid < 16) s_lut[tid] = E2M1_LUT_T[tid];
    __shared__ unsigned char s_pk[2][TILE_KH * STRIDE];
    __shared__ unsigned char s_sc[2][(TILE_KH / 8) * STRIDE];
    __shared__ __align__(16) __nv_bfloat16 s_a[2][ROWS * TILE_KH * 2];
    __syncthreads();
    const unsigned int mask = is_shared ? ((1u << ROWS) - 1u) : u_rowmask[u];
    auto stage = [&](unsigned int tg, unsigned int buf) {
        const unsigned int r0 = tg * TILE_KH;
        for (unsigned int j = tid; j < TILE_KH * (COLW / 16u); j += BS) {
            const unsigned int rr = j / (COLW / 16u), cb = (j % (COLW / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            *(uint4*)&s_pk[buf][rr * STRIDE + cb] = *(const uint4*)&B_packed[(unsigned long long)(r0 + rr) * N + ns0 + cb]; }
        const unsigned int ng = TILE_KH / 8u;
        for (unsigned int j = tid; j < ng * (COLW / 16u); j += BS) {
            const unsigned int sg = j / (COLW / 16u), cb = (j % (COLW / 16u)) * 16u;
            if (ns0 + cb >= N) continue;
            *(uint4*)&s_sc[buf][sg * STRIDE + cb] = *(const uint4*)&B_scale[(unsigned long long)(tg * ng + sg) * N + ns0 + cb]; }
        for (unsigned int j = tid; j < ROWS * (TILE_KH * 2 / 8u); j += BS) {
            const unsigned int row = j / (TILE_KH * 2 / 8u), chunk = j % (TILE_KH * 2 / 8u);
            *(uint4*)&s_a[buf][row * TILE_KH * 2 + chunk * 8] = *(const uint4*)&A[(unsigned long long)row * h + tg * TILE_KH * 2 + chunk * 8]; }
    };
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int arow = l & 15u; const bool areal = arow < ROWS;
    const unsigned int ntg = (K / 2u) / TILE_KH;
    stage(0, 0);
    for (unsigned int tg = 0; tg < ntg; tg++) {
        __syncthreads();
        if (tg + 1u < ntg) stage(tg + 1u, (tg + 1u) & 1u);
        const unsigned int buf = tg & 1u;
        #pragma unroll
        for (unsigned int s = 0; s < TILE_KH / 8u; s++) {         // one 16-k WMMA step = 8 k_half = one scale group
            v16bf a, b;
            if (areal) {
                const uint4 a0 = *(const uint4*)&s_a[buf][arow * TILE_KH * 2 + s * 16];
                const uint4 a1 = *(const uint4*)&s_a[buf][arow * TILE_KH * 2 + s * 16 + 8];
                const __bf16* p0 = (const __bf16*)&a0; const __bf16* p1 = (const __bf16*)&a1;
                #pragma unroll
                for (int i = 0; i < 8; i++) { a[i] = p0[i]; a[8 + i] = p1[i]; }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) a[i] = (__bf16)0.0f;
            }
            const float sc = mx_block_scale<false>(s_sc[buf][s * STRIDE + ccol], 1.0f);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned char byte = s_pk[buf][(s * 8u + j) * STRIDE + ccol];
                b[2 * j] = (__bf16)(s_lut[byte & 0xFu] * sc);
                b[2 * j + 1] = (__bf16)(s_lut[(byte >> 4) & 0xFu] * sc);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
        }
    }
    __syncthreads();
    const unsigned int col = ns0 + ccol;
    if (col >= N) return;
    #pragma unroll
    for (int e = 0; e < 4; e++) {                                 // rows 2e + (l >> 4): 0..7; e 4..7 are the zero pad
        const unsigned int r = 2u * e + (l >> 4);
        if (!(mask & (1u << r))) continue;
        const __nv_bfloat16 v = __float2bfloat16(acc[e] * s2);
        if (is_shared) { if (proj == 0) sh_gate_out[r * N + col] = v; else sh_up_out[r * N + col] = v; }
        else { const unsigned int slot = u_rowslot[u * MOE_MAX_K + r]; __nv_bfloat16* C = (proj == 0) ? gate_out : up_out;
            C[(unsigned long long)(r * top_k + slot) * N + col] = v; }
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
#define ARGS_(V) dA, t[0], t[1], ds2, go[V], t[2], t[3], ds2, uo[V], due, dmask, dslot, dcnt, sgp, sgs, 0.01f, sgo[V], sup, sus, 0.01f, suo[V], N, K, TOPK, K
        if (v == 0) hipLaunchKernelGGL(moe_expert_gate_up_shared_t_rows_v3_r8_b256, g, b, 0, 0, ARGS_(0), R);
        else if (v == 1) hipLaunchKernelGGL((gu_rows_wmma<8>), dim3((N + 127) / 128, U + 1, 2), dim3(256), 0, 0, ARGS_(1));
        else hipLaunchKernelGGL((gu_rows_wmma<4>), dim3((N + 63) / 64, U + 1, 2), dim3(128), 0, 0, ARGS_(2)); };
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const double bytes = (double)(U + 1) * 2 * (pb + sb);
    float t[NV];
    for (int rep = 0; rep < 2; rep++) for (int v = 0; v < NV; v++) { for (int i = 0; i < 4; i++) launch(v, i % SETS); CHECK(hipEventRecord(e0)); for (int i = 0; i < 32; i++) launch(v, i % SETS); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); t[v] = 1000.f * ms / 32; }
    for (int v = 0; v < NV; v++) launch(v, 0); CHECK(hipDeviceSynchronize());
    auto cmp = [&](int v, double& rel, size_t& mism, size_t& tot) { std::vector<__nv_bfloat16> a0(R * TOPK * N), a1(R * TOPK * N); double md = 0, mr = 0; mism = 0; tot = 0;
        for (auto pr : {std::make_pair(go[0], go[v]), std::make_pair(uo[0], uo[v]), std::make_pair(sgo[0], sgo[v]), std::make_pair(suo[0], suo[v])}) {
            const size_t n = (pr.first == sgo[0] || pr.first == suo[0]) ? R * N : R * TOPK * N;
            CHECK(hipMemcpy(a0.data(), pr.first, n * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(a1.data(), pr.second, n * 2, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < n; i++) { const double x = __bfloat162float(a0[i]), y = __bfloat162float(a1[i]); md = fmax(md, fabs(x - y)); mr = fmax(mr, fabs(x));
                mism += memcmp(&a0[i], &a1[i], 2) != 0; tot += x != 0.0; } }
        rel = md / (mr > 0 ? mr : 1); };
    printf("-- gate_up rows_v3_r8_b256 N=%u K=%u U=%u (+shared): ship %7.1f us (%.0f GB/s)\n", N, K, U, t[0], bytes / 1e3 / t[0]);
    for (int v = 1; v < NV; v++) { double rel; size_t mism, tot; cmp(v, rel, mism, tot);
        printf("-- rows WMMA %d waves:   %7.1f us x%.2f | max|diff|/max|ref| %.2e, bf16 mismatches %zu / %zu nonzero\n", v == 1 ? 8 : 4, t[v], t[0] / t[v], rel, mism, tot); }
    return 0;
}
