// SPDX-License-Identifier: AGPL-3.0-only
// KLOOP #155 (Flash-Next C8): moe_expert_union_t is 101 us/call x27648 (2.7% of C8 GPU time, grid 1x32): lane 0 runs
// the K*top_k dedup serially and re-reads u_experts from GLOBAL memory in the inner compare loop. Twin: ids in LDS,
// first-occurrence flags in parallel, a prefix over the flags gives each unique id its index in first-occurrence
// order (the serial loop's order), then a parallel scatter. Outputs must be byte-identical (u_experts / rowmask /
// rowslot for u < U, u_count).
#include "moe_shared_expert_fused_t.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define UNION_MAX 256u

__global__ void union_par(const unsigned int* __restrict__ expert_indices, unsigned int* __restrict__ u_experts, unsigned int* __restrict__ u_rowmask,
    unsigned char* __restrict__ u_rowslot, unsigned int* __restrict__ u_count, unsigned int K, unsigned int top_k) {
    const unsigned int n = K * top_k, lane = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned int s_id[UNION_MAX], s_fi[UNION_MAX], s_pref[UNION_MAX], s_mask[UNION_MAX];
    __shared__ unsigned int s_U;
    for (unsigned int t = lane; t < n; t += nt) s_id[t] = expert_indices[t];
    __syncthreads();
    for (unsigned int t = lane; t < n; t += nt) {
        const unsigned int id = s_id[t]; unsigned int f = t;
        for (unsigned int q = 0; q < t; q++) if (s_id[q] == id) { f = q; break; }
        s_fi[t] = f;
    }
    __syncthreads();
    if (lane == 0) { unsigned int U = 0; for (unsigned int t = 0; t < n; t++) { s_pref[t] = U; U += (s_fi[t] == t); } s_U = U; *u_count = U; }
    __syncthreads();
    for (unsigned int t = lane; t < n; t += nt) if (s_fi[t] == t) {
        const unsigned int u = s_pref[t]; u_experts[u] = s_id[t]; s_mask[u] = 0;
        for (unsigned int rr = 0; rr < MOE_MAX_K; rr++) u_rowslot[u * MOE_MAX_K + rr] = 0xFF; }
    __syncthreads();
    for (unsigned int t = lane; t < n; t += nt) {
        const unsigned int u = s_pref[s_fi[t]], r = t / top_k;
        atomicOr(&s_mask[u], 1u << r); u_rowslot[u * MOE_MAX_K + r] = (unsigned char)(t % top_k); }
    __syncthreads();
    for (unsigned int u = lane; u < s_U; u += nt) u_rowmask[u] = s_mask[u];
}

int main() {
    const unsigned TOPK = 10, NE = 256; bool all_same = true; float tser = 0, tpar = 0;
    unsigned *dids, *due[2], *dmask[2], *dcnt[2]; unsigned char* dslot[2];
    CHECK(hipMalloc(&dids, 8 * TOPK * 4));
    for (int v = 0; v < 2; v++) { CHECK(hipMalloc(&due[v], 8 * TOPK * 4)); CHECK(hipMalloc(&dmask[v], 8 * TOPK * 4)); CHECK(hipMalloc(&dslot[v], 8 * TOPK * MOE_MAX_K)); CHECK(hipMalloc(&dcnt[v], 4)); }
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    srand(9);
    for (unsigned K = 2; K <= 8; K++) for (int trial = 0; trial < 20; trial++) {
        std::vector<unsigned> ids(K * TOPK);
        for (unsigned r = 0; r < K; r++) { std::vector<unsigned> row; while (row.size() < TOPK) { unsigned id = (trial % 3 == 0) ? rand() % 24 : rand() % NE; bool dup = false; for (auto x : row) dup |= x == id; if (!dup) row.push_back(id); }
            for (unsigned i = 0; i < TOPK; i++) ids[r * TOPK + i] = row[i]; }
        CHECK(hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
        for (int v = 0; v < 2; v++) { CHECK(hipMemset(due[v], 0xAB, 8 * TOPK * 4)); CHECK(hipMemset(dmask[v], 0xAB, 8 * TOPK * 4)); CHECK(hipMemset(dslot[v], 0xAB, 8 * TOPK * MOE_MAX_K)); }
        hipLaunchKernelGGL(moe_expert_union_t, dim3(1), dim3(32), 0, 0, dids, due[0], dmask[0], dslot[0], dcnt[0], K, TOPK);
        hipLaunchKernelGGL(union_par, dim3(1), dim3(32), 0, 0, dids, due[1], dmask[1], dslot[1], dcnt[1], K, TOPK);
        CHECK(hipDeviceSynchronize());
        unsigned c[2]; std::vector<unsigned> a[2], m[2]; std::vector<unsigned char> s[2];
        for (int v = 0; v < 2; v++) { CHECK(hipMemcpy(&c[v], dcnt[v], 4, hipMemcpyDeviceToHost)); a[v].resize(8 * TOPK); m[v].resize(8 * TOPK); s[v].resize(8 * TOPK * MOE_MAX_K);
            CHECK(hipMemcpy(a[v].data(), due[v], a[v].size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(m[v].data(), dmask[v], m[v].size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(s[v].data(), dslot[v], s[v].size(), hipMemcpyDeviceToHost)); }
        bool same = c[0] == c[1] && !memcmp(a[0].data(), a[1].data(), c[0] * 4) && !memcmp(m[0].data(), m[1].data(), c[0] * 4) && !memcmp(s[0].data(), s[1].data(), c[0] * MOE_MAX_K);
        if (!same && all_same) printf("-- MISMATCH K=%u trial %d (U %u vs %u)\n", K, trial, c[0], c[1]);
        all_same &= same;
        if (K == 8 && trial == 1) for (int v = 0; v < 2; v++) { CHECK(hipEventRecord(e0)); for (int i = 0; i < 64; i++) {
                if (v == 0) hipLaunchKernelGGL(moe_expert_union_t, dim3(1), dim3(32), 0, 0, dids, due[0], dmask[0], dslot[0], dcnt[0], K, TOPK);
                else hipLaunchKernelGGL(union_par, dim3(1), dim3(32), 0, 0, dids, due[1], dmask[1], dslot[1], dcnt[1], K, TOPK); }
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); (v == 0 ? tser : tpar) = 1000.f * ms / 64; }
    }
    printf("-- moe_expert_union_t K=8 top_k=10: ship %6.1f us | parallel %6.1f us x%.1f | 140 routings %s\n", tser, tpar, tser / tpar, all_same ? "byte-identical" : "DIFFER");
    return 0;
}
