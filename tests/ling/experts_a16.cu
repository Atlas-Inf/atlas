// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -std=c++17 -O2 -arch=sm_121 tests/ling/experts_a16.cu -o /tmp/ling-a16
#include "../../kernels/gb10/common/moe_expert_gemv.cu"
#undef BLOCK_SIZE
#include "../../kernels/gb10/common/moe_expert_gemv_wide.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

template<typename T> T* allocate(size_t count) {
    T* result;
    check(cudaMallocManaged(&result, count * sizeof(T)));
    return result;
}

int main() {
    constexpr unsigned tokens = 3, routes = 8, experts = 9, cols = 768, rows = 35;
    auto packed = allocate<unsigned long long>(experts);
    auto scales = allocate<unsigned long long>(experts);
    auto global = allocate<float>(experts);
    auto ids = allocate<unsigned>(tokens * routes);
    auto input = allocate<__nv_bfloat16>(tokens * routes * cols);
    auto output = allocate<__nv_bfloat16>(tokens * routes * rows);
    for (unsigned e = 0; e < experts; ++e) {
        auto weight = allocate<unsigned char>(rows * cols / 2);
        auto scale = allocate<unsigned char>(rows * cols / 16);
        std::fill_n(weight, rows * cols / 2, 0x22); // E2M1 +1
        std::fill_n(scale, rows * cols / 16, 0x38); // E4M3 +1
        packed[e] = reinterpret_cast<unsigned long long>(weight);
        scales[e] = reinterpret_cast<unsigned long long>(scale);
        global[e] = (e % 3 + 1) * 0.5f;
    }
    for (unsigned r = 0; r < tokens * routes; ++r) {
        ids[r] = (r * 5 + 1) % experts;
        std::fill_n(input + r * cols, cols, __float2bfloat16(float(r + 1)));
    }
    for (auto stride : {0u, cols}) {
        for (bool wide : {false, true}) {
            std::fill_n(output, tokens * routes * rows, __float2bfloat16(-1.f));
            if (wide) {
                moe_expert_gemv_wide<<<dim3((rows + 31) / 32, routes, tokens), 256, 64 + cols * sizeof(__nv_bfloat16)>>>(
                    input, packed, scales, global, output, ids, rows, cols, routes, stride);
            } else {
                moe_expert_gemv<<<dim3((rows + 3) / 4, routes, tokens), 128>>>(
                    input, packed, scales, global, output, ids, rows, cols, routes, stride);
            }
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            for (unsigned r = 0; r < tokens * routes; ++r) {
                const auto source = stride ? r : r / routes;
                const float expected = __bfloat162float(__float2bfloat16((source + 1) * cols * global[ids[r]]));
                for (unsigned n = 0; n < rows; ++n) {
                    if (__bfloat162float(output[r * rows + n]) != expected) {
                        std::fprintf(stderr, "wide=%d stride=%u route=%u column=%u expected=%g got=%g\n",
                                     wide, stride, r, n, expected, __bfloat162float(output[r * rows + n]));
                        return 1;
                    }
                }
            }
        }
    }
    std::printf("A16 narrow/wide, shared and per-route inputs: %u BF16 outputs matched\n",
                4 * tokens * routes * rows);
}
