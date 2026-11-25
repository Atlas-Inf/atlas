// SPDX-License-Identifier: AGPL-3.0-only
// Compile with nvcc -std=c++17 -arch=sm_121 tests/ling/router.cu -o /tmp/ling-router
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_router.cu"

#define CHECK(call) do { auto err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 1; } } while (0)

int main() {
    constexpr int rows = 35;
    float *logits, *bias, *out;
    int *ids;
    CHECK(cudaMallocManaged(&logits, rows * 512 * sizeof(float)));
    CHECK(cudaMallocManaged(&bias, 512 * sizeof(float)));
    CHECK(cudaMallocManaged(&ids, rows * 8 * sizeof(int)));
    CHECK(cudaMallocManaged(&out, rows * 8 * sizeof(float)));
    std::mt19937 random(123);
    std::uniform_real_distribution<float> draw(-1,1);
    for (int i = 0; i < 512; ++i) bias[i] = draw(random) * 0.2f;
    for (int i = 0; i < rows * 512; ++i) logits[i] = draw(random) * 4;
    // A group with the single best expert can still be eliminated.
    for (int i = 0; i < 512; ++i) logits[i] = -6;
    logits[0] = 6;
    for (int g = 1; g <= 4; ++g) logits[g*64] = logits[g*64 + 1] = 4;
    ling_grouped_top8<<<rows, 256>>>(logits,bias,ids,out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < rows; ++r) {
        std::array<float,512> prob, corrected;
        std::array<float,8> group_scores;
        std::array<int,8> groups;
        for (int i = 0; i < 512; ++i) {
            prob[i] = 1.f / (1.f + std::exp(-logits[r*512 + i]));
            corrected[i] = prob[i] + bias[i];
        }
        for (int g = 0; g < 8; ++g) {
            groups[g] = g;
            auto first = corrected.begin()+64*g;
            std::vector<float> a(first,first+64);
            std::sort(a.rbegin(),a.rend());
            group_scores[g] = a[0]+a[1];
        }
        std::stable_sort(groups.begin(),groups.end(),[&](int a,int b){return group_scores[a]>group_scores[b];});
        std::vector<int> eligible;
        for (int g = 0; g < 4; ++g)
            for (int i = 0; i < 64; ++i) eligible.push_back(groups[g]*64+i);
        std::sort(eligible.begin(),eligible.end(),[&](int a,int b){
            return corrected[a] == corrected[b] ? a < b : corrected[a] > corrected[b];
        });
        float sum = 0;
        for (int k=0; k<8; ++k) sum += prob[eligible[k]];
        for (int k=0; k<8; ++k) {
            if (ids[8*r+k] != eligible[k] || std::abs(out[8*r+k]-2.5f*prob[eligible[k]]/sum)>1.e-6f) {
                std::fprintf(stderr,"row %d choice %d differs\n",r,k);
                return 1;
            }
        }
    }
    std::printf("%d rows: all grouped expert IDs and unbiased weights match CPU oracle\n",rows);
}
