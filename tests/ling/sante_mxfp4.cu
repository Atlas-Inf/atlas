// SPDX-License-Identifier: AGPL-3.0-only
// Build with nvcc -gencode arch=compute_121a,code=sm_121a tests/ling/sante_mxfp4.cu -o /tmp/sante_mxfp4.
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_ffn.cu"
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

template <class T> static T* array(size_t count) {
    T* p;
    check(cudaMallocManaged(&p, count * sizeof(T)));
    return p;
}

static void compare(int projection, int k, int n) {
    constexpr int slots = 8;
    const float codebook[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    auto* table = array<const unsigned char*>(slots * 6);
    auto* ids = array<int>(slots);
    auto* x = array<__nv_bfloat16>(slots * k);
    auto* y = array<__nv_bfloat16>(slots * n);
    for (int i = 0; i < slots * k; ++i)
        x[i] = __float2bfloat16(static_cast<float>((i * 7 + i / k) % 17 - 8) / 8);
    for (int e = 0; e < slots; ++e) {
        ids[e] = (3 * e + 1) % slots;
        unsigned char* w = array<unsigned char>(n * k / 2);
        unsigned char* scale = array<unsigned char>(n * k / 32);
        table[(e * 3 + projection) * 2] = w;
        table[(e * 3 + projection) * 2 + 1] = scale;
        for (int i = 0; i < n * k / 2; ++i)
            w[i] = static_cast<unsigned char>(i * 53 + i / 11 + e * 17);
        for (int i = 0; i < n * k / 32; ++i)
            scale[i] = static_cast<unsigned char>(122 + (i + e) % 7);
    }
    sante_expert_gemv<<<dim3((n + 3) / 4, slots), 128>>>(x, y, table, ids, n, k, projection, 1);
    check(cudaDeviceSynchronize());
    for (int slot = 0; slot < slots; ++slot) {
        const unsigned char* w = table[(ids[slot] * 3 + projection) * 2];
        const unsigned char* scales = table[(ids[slot] * 3 + projection) * 2 + 1];
        for (int row = 0; row < n; ++row) {
            double sum = 0;
            for (int col = 0; col < k; ++col) {
                int i = row * k + col;
                int q = (w[i / 2] >> (4 * (i % 2))) & 15;
                double v = std::ldexp((q & 8 ? -1 : 1) * codebook[q & 7], scales[i / 32] - 127);
                sum += v * __bfloat162float(x[(projection == 2 ? slot * k : 0) + col]);
            }
            float expected = __bfloat162float(__float2bfloat16(static_cast<float>(sum)));
            if (__bfloat162float(y[slot * n + row]) != expected) {
                std::fprintf(stderr, "mismatch at projection %d, slot %d, row %d\n", projection, slot, row);
                std::exit(1);
            }
        }
    }
    for (int e = 0; e < slots; ++e) {
        check(cudaFree(const_cast<unsigned char*>(table[(e * 3 + projection) * 2])));
        check(cudaFree(const_cast<unsigned char*>(table[(e * 3 + projection) * 2 + 1])));
    }
    check(cudaFree(x));
    check(cudaFree(y));
    check(cudaFree(ids));
    check(cudaFree(table));
}

int main() {
    compare(0, 2560, 9);
    compare(2, 768, 13);
    std::puts("MXFP4 gate and down match the CPU BF16 outputs for all eight routed slots.");
}
