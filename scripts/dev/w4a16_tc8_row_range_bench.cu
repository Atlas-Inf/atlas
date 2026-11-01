// SPDX-License-Identifier: AGPL-3.0-only
// Standalone check of the rank-split propose (ATLAS_GLM_DRAFT_TP): does a
// w4a16_gemv_tc{8,16,32} launch over a ROW RANGE of an NVFP4 twin write the
// bits the whole-weight launch writes for those rows? Each rank computes one
// share of the output rows of the DFlash2 drafter's gate/up [12288, 4096],
// down [4096, 12288] and vocabulary head [154856, 4096], and with
// ATLAS_GLM_DRAFT_TP_CTX of its context `fc` [4096, 20480] and fused K/V
// [10240, 4096] twins, cut at the largest CTA boundary at or under N / 2
// (`rank_split::half`), so the two launches run exactly the CTAs of the
// unsplit one.
//
// For every shape and M = 1..32 on the tier `w4a16_gemv_tiers::tc_kernel`
// picks (tc8 up to 8 rows, tc16 up to 16, tc32 up to 32; incl. an all-zero
// and an all -0.0 activation row) it compares every output bit of the whole
// launch with the two share launches, then times the whole launch against
// one share at M = 8 under PDL.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/w4a16_tc8_row_range_bench.cu -o tc8_row_range_bench
//   ./tc8_row_range_bench [iters=40] [reps=9]
// Prints "bitwise: 0 of N outputs differ" (pass) and exits 0; any differing
// output exits 2. Device memory: ~0.5 GB (the head twin is 357 MB).
#include "w4a16_gemv.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned MAXM = 32, TIMEM = 8, CTA = 16;

struct Shape { const char* name; unsigned N, K; };

typedef void (*Gemv)(const bf*, const unsigned char*, const unsigned char*, float, bf*,
                     unsigned, unsigned, unsigned);
// The narrowest tier covering M rows, as `w4a16_gemv_tiers::tc_kernel` picks.
static Gemv tier(unsigned M) { return M <= 8 ? w4a16_gemv_tc8 : M <= 16 ? w4a16_gemv_tc16 : w4a16_gemv_tc32; }

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 40;
    const int reps = argc > 2 ? atoi(argv[2]) : 9;
    const Shape shapes[] = {{"gate/up", 12288, 4096}, {"down", 4096, 12288}, {"head", 154856, 4096},
                            {"ctx fc", 4096, 20480}, {"ctx kv", 10240, 4096}};
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto tobf = [](float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; };

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    size_t diff = 0, checked = 0;
    for (const Shape& s : shapes) {
        const unsigned N = s.N, K = s.K;
        const size_t packed = (size_t)N * K / 2, groups = (size_t)N * K / GROUP_SIZE;
        std::vector<unsigned char> hW(packed), hS(groups);
        for (auto& x : hW) x = (unsigned char)rng();
        // E4M3 scales across the normal range (never NaN/inf, both signs).
        for (auto& x : hS) x = (unsigned char)(((rng() % 13 + 1) << 3) | (rng() & 7) | ((rng() & 1) << 7));
        std::vector<unsigned short> hA((size_t)MAXM * K);
        for (auto& x : hA) x = tobf(nd(rng) * 1.5f);
        for (unsigned k = 0; k < K; k++) { hA[3 * (size_t)K + k] = 0; hA[5 * (size_t)K + k] = 0x8000; }

        unsigned char *dW, *dS;
        bf *dA, *dFull, *dPart[2];
        const unsigned cut = N / 2 / CTA * CTA;
        const unsigned first[2] = {0, cut}, rows[2] = {cut, N - cut};
        CK(cudaMalloc(&dW, packed)); CK(cudaMalloc(&dS, groups));
        CK(cudaMalloc(&dA, hA.size() * 2));
        CK(cudaMalloc(&dFull, (size_t)MAXM * N * 2));
        for (int r = 0; r < 2; r++) CK(cudaMalloc(&dPart[r], (size_t)MAXM * rows[r] * 2));
        CK(cudaMemcpy(dW, hW.data(), packed, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dS, hS.data(), groups, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        const float scale2 = 0.0173f;

        // Rows [f, f + n) of the twin into `c` ([M, n], stride n), as
        // `rank_split::row_range` hands them to `ops::w4a16_gemv_batchm`.
        auto launch = [&](unsigned f, unsigned n, unsigned M, bf* c) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3((n + CTA - 1) / CTA, 1, 1);
            cfg.blockDim = dim3(W4A16_TC_WARPS * WARP_SIZE, 1, 1);
            cudaLaunchAttribute at;
            at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at.val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = &at; cfg.numAttrs = 1;
            CK(cudaLaunchKernelEx(&cfg, tier(M), (const bf*)dA,
                                  (const unsigned char*)(dW + (size_t)f * K / 2),
                                  (const unsigned char*)(dS + (size_t)f * K / GROUP_SIZE),
                                  scale2, c, M, n, K));
        };

        std::vector<unsigned short> full((size_t)MAXM * N), part[2];
        for (int r = 0; r < 2; r++) part[r].resize((size_t)MAXM * rows[r]);
        size_t shape_diff = 0, shape_checked = 0;
        for (unsigned M = 1; M <= MAXM; M++) {
            CK(cudaMemset(dFull, 0xFF, (size_t)MAXM * N * 2));
            launch(0, N, M, dFull);
            for (int r = 0; r < 2; r++) {
                CK(cudaMemset(dPart[r], 0xEE, (size_t)MAXM * rows[r] * 2));
                launch(first[r], rows[r], M, dPart[r]);
            }
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(full.data(), dFull, full.size() * 2, cudaMemcpyDeviceToHost));
            for (int r = 0; r < 2; r++)
                CK(cudaMemcpy(part[r].data(), dPart[r], part[r].size() * 2, cudaMemcpyDeviceToHost));
            for (unsigned m = 0; m < M; m++)
                for (int r = 0; r < 2; r++)
                    for (unsigned n = 0; n < rows[r]; n++, shape_checked++)
                        shape_diff += full[(size_t)m * N + first[r] + n] != part[r][(size_t)m * rows[r] + n];
        }
        diff += shape_diff; checked += shape_checked;

        // Whole launch against rank 0's share, M = 8, PDL-chained.
        float us[2] = {0, 0};
        for (int which = 0; which < 2; which++) {
            std::vector<float> t;
            for (int rep = 0; rep < reps; rep++) {
                for (int w = 0; w < 3; w++) launch(0, which ? rows[0] : N, TIMEM, which ? dPart[0] : dFull);
                CK(cudaEventRecord(e0));
                for (int it = 0; it < iters; it++) launch(0, which ? rows[0] : N, TIMEM, which ? dPart[0] : dFull);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                t.push_back(ms * 1000.f / iters);
            }
            std::sort(t.begin(), t.end());
            us[which] = t[reps / 2];
        }
        const double mb = (packed + groups) / 1e6;
        printf("%-8s N=%6u K=%5u cut=%6u: %zu of %zu outputs differ; whole %7.1f us (%5.1f GB/s), "
               "rank-0 share %7.1f us, saves %6.1f us\n",
               s.name, N, K, cut, shape_diff, shape_checked, us[0], mb / us[0] * 1e3, us[1], us[0] - us[1]);
        CK(cudaFree(dW)); CK(cudaFree(dS)); CK(cudaFree(dA)); CK(cudaFree(dFull));
        for (auto& p : dPart) CK(cudaFree(p));
    }
    printf("bitwise: %zu of %zu outputs differ\n", diff, checked);
    return diff == 0 ? 0 : 2;
}
