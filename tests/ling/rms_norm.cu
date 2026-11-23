// SPDX-License-Identifier: AGPL-3.0-only
// nvcc -arch=sm_121 tests/ling/rms_norm.cu -o /tmp/ling-norm && /tmp/ling-norm
#include "../../kernels/gb10/ling-3.2-air/nvfp4/ling_norm.cu"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Bf16 = __nv_bfloat16;
static void check(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr,"%s\n", cudaGetErrorString(error)); std::exit(2); }
}
static void run(int width, int op) {
    constexpr int rows = 9;
    Bf16* storage;
    const int count = width * rows;
    check(cudaMallocManaged(&storage, (4 * count + width) * sizeof(Bf16)));
    Bf16* branch = storage + count;
    Bf16* out = branch + count;
    Bf16* residual = out + count;
    Bf16* weight = residual + count;
    for (int i = 0; i < width; ++i) weight[i] = Bf16(0.41f + (i % 17) * 0.055f);
    std::vector<Bf16> expected(count), added(count);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int j = 0; j < width; ++j) {
            const int i = r * width + j;
            storage[i] = Bf16((i % 93 - 46) * 0.019f);
            branch[i] = Bf16((i % 43 - 21) * 0.003f);
            added[i] = op == 2 ? Bf16(float(storage[i]) + float(branch[i])) : storage[i];
            ss += double(float(added[i])) * float(added[i]);
        }
        const float inv = 1.f / std::sqrt(float(ss / width) + 1.e-6f);
        for (int j = 0; j < width; ++j) {
            const int i = r * width + j;
            expected[i] = Bf16(float(Bf16(float(added[i]) * inv)) * float(weight[j]));
        }
    }
    const int threads = width < 1024 ? width : 1024;
    if (op == 0) ling_rms_norm<<<rows,threads>>>(storage, weight, out, width, 1.e-6f);
    if (op == 1) ling_rms_norm_residual<<<rows,threads>>>(storage, weight, out, residual, width, 1.e-6f);
    if (op == 2) ling_residual_add_rms_norm<<<rows,threads>>>(storage, branch, weight, out, residual, width, 1.e-6f);
    check(cudaDeviceSynchronize());
    for (int i = 0; i < count; ++i) {
        if (float(out[i]) != float(expected[i]) || (op && float(residual[i]) != float(added[i]))) {
            std::fprintf(stderr,"width=%d op=%d index=%d: %g expected %g\n",width,op,i,float(out[i]),float(expected[i]));
            std::exit(1);
        }
    }
    std::printf("width=%d op=%d: %d outputs matched\n", width, op, count);
    check(cudaFree(storage));
}
int main() {
    for (int width : {512, 2560}) for (int op = 0; op < 3; ++op) run(width, op);
}
