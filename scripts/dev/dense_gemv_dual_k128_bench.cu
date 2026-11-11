// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the GLM KDA gate pair (f_b/g_b: N = 4096 per rank,
// K = 128) GEMV: dense_gemv_bf16_batchm_dual (generic) versus the K = 128
// tier dense_gemv_bf16_batchm_dual_k128. Checks every output bit for
// M = 1..8 (incl. an all-zero and an all -0.0 activation row), then times
// both under PDL launches, cycling `copies` weight pairs so the weights come
// from DRAM as in decode (2 MiB per pair; 24 pairs = 48 MiB > 24 MiB L2).
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/dense_gemv_dual_k128_bench.cu -o dual_k128_bench
//   ./dual_k128_bench [copies=24] [iters=400] [reps=15]
// Device memory: ~50 MB at the defaults.
#include "dense_gemv_bf16_batchm.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef void (*Kern)(const bf*, const bf*, const bf*, const bf*, bf*, bf*, unsigned, unsigned, unsigned);

int main(int argc, char** argv) {
    const unsigned N = 4096, K = 128;
    const int copies = argc > 1 ? atoi(argv[1]) : 24;
    const int iters = argc > 2 ? atoi(argv[2]) : 400;
    const int reps = argc > 3 ? atoi(argv[3]) : 15;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto tobf = [](float f) { bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; };
    std::vector<unsigned short> hA(2 * MAX_M * K), hB((size_t)copies * 2 * N * K);
    for (auto& x : hA) x = tobf(nd(rng) * 2.f);
    for (auto& x : hB) x = tobf(nd(rng) * 0.05f);
    for (unsigned k = 0; k < K; k++) { hA[3 * K + k] = 0; hA[(MAX_M + 5) * K + k] = 0x8000; }
    bf *dA, *dB, *dC[2];
    const size_t c_elems = 2 * MAX_M * N;
    CK(cudaMalloc(&dA, hA.size() * 2));
    CK(cudaMalloc(&dB, hB.size() * 2));
    for (auto& c : dC) CK(cudaMalloc(&c, c_elems * 2));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));

    const Kern kern[2] = {dense_gemv_bf16_batchm_dual, dense_gemv_bf16_batchm_dual_k128};
    const unsigned outs_per_cta[2] = {N_PER_BLOCK, BLOCK_SIZE / DUAL_K128_LANES};
    const char* name[2] = {"generic", "k128"};
    auto launch = [&](int k, int copy, unsigned M, bf* c) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(N / outs_per_cta[k], 1, 2);
        cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
        cudaLaunchAttribute at;
        at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at.val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = &at; cfg.numAttrs = 1;
        const bf* B = dB + (size_t)copy * 2 * N * K;
        CK(cudaLaunchKernelEx(&cfg, kern[k], (const bf*)dA, (const bf*)(dA + MAX_M * K), B, B + N * K,
                              c, c + MAX_M * N, M, N, K));
    };

    std::vector<unsigned short> o[2] = {std::vector<unsigned short>(c_elems), std::vector<unsigned short>(c_elems)};
    size_t diff = 0, checked = 0;
    for (unsigned M = 1; M <= MAX_M; M++)
        for (int copy = 0; copy < std::min(copies, 4); copy++) {
            for (int k = 0; k < 2; k++) {
                CK(cudaMemset(dC[k], k ? 0xEE : 0xFF, c_elems * 2));
                launch(k, copy, M, dC[k]);
            }
            CK(cudaDeviceSynchronize());
            for (int k = 0; k < 2; k++) CK(cudaMemcpy(o[k].data(), dC[k], c_elems * 2, cudaMemcpyDeviceToHost));
            for (int p = 0; p < 2; p++)
                for (size_t i = 0; i < (size_t)M * N; i++, checked++)
                    diff += o[0][p * MAX_M * N + i] != o[1][p * MAX_M * N + i];
        }
    printf("bitwise: %zu of %zu outputs differ\n", diff, checked);

    const unsigned Ms[] = {2, 4, 5, 8};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("us/call, median of %d reps x %d PDL launches over %d weight pairs\n", reps, iters, copies);
    for (unsigned M : Ms) {
        std::vector<float> us[2];
        for (int r = 0; r < reps; r++)
            for (int k = 0; k < 2; k++) {
                for (int w = 0; w < 10; w++) launch(k, w % copies, M, dC[0]);
                CK(cudaEventRecord(e0));
                for (int it = 0; it < iters; it++) launch(k, it % copies, M, dC[0]);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                us[k].push_back(ms * 1000.f / iters);
            }
        for (auto& v : us) std::sort(v.begin(), v.end());
        const float g = us[0][reps / 2], t = us[1][reps / 2];
        printf("M=%u  %s %6.2f  %s %6.2f  speedup %.2fx\n", M, name[0], g, name[1], t, g / t);
    }
    return diff == 0 ? 0 : 2;
}
