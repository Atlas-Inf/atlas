// SPDX-License-Identifier: AGPL-3.0-only
// `dense_gemm_bf16_pipelined_g8` (grouped tile raster) against
// `dense_gemm_bf16_pipelined` (kernels/gb10/common/dense_gemm_bf16.cu) at the
// qwen4_exp PLE projection shapes: every output byte equal, then GPU time.
// Build/run (repo root, GB10): scripts/dev/qwen4exp_ple_gemm_bench.sh
#include "qwen4exp_ptx_harness.h"

#include <random>

typedef unsigned short bf;

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    init_driver();
    PtxModule m;
    m.load(dir + "/dense_gemm_bf16.ptx");
    CUfunction f0 = m.fn("dense_gemm_bf16_pipelined"), f1 = m.fn("dense_gemm_bf16_pipelined_g8");
    std::mt19937 rng(5);
    std::normal_distribution<float> d(0.f, 1.f);
    struct { const char* n; unsigned M, N, K; } shapes[] = {
        {"PLE key_proj", 8192, 10240, 2560}, {"PLE value_proj", 8192, 2560, 2560},
        {"PLE key_proj tail", 7936, 10240, 2560}, {"ragged", 1000, 4100, 2560}, {"router", 16128, 512, 2560}};
    int fail = 0;
    for (auto& s : shapes) {
        Buf<bf> A, B, C0, C1;
        A.alloc((size_t)s.M * s.K); B.alloc((size_t)s.N * s.K);
        C0.alloc((size_t)s.M * s.N); C1.alloc((size_t)s.M * s.N);
        std::vector<bf> h(A.n);
        for (auto& x : h) x = f2bf(d(rng));
        A.put(h);
        h.resize(B.n);
        for (auto& x : h) x = f2bf(0.05f * d(rng));
        B.put(h);
        auto run = [&](CUfunction f, bf* C) {
            Args a;
            a.add(A.p).add(B.p).add(C).add(s.M).add(s.N).add(s.K);
            launch(f, dim3((s.N + 127) / 128, (s.M + 127) / 128), dim3(256), 0, a);
        };
        C0.fill(0x55); C1.fill(0x33);
        run(f0, C0.p); run(f1, C1.p);
        CK(cudaDeviceSynchronize());
        const size_t diff = diff_bytes(C0.get(), C1.get());
        fail += diff != 0;
        const float t0 = time_ms([&] { run(f0, C0.p); }, 5, 5), t1 = time_ms([&] { run(f1, C1.p); }, 5, 5);
        const double tf = 2.0 * s.M * s.N * s.K / 1e12;
        printf("  %s %-18s %5ux%5ux%4u  %d differing bytes   default %7.2f ms (%5.1f TF/s)  g8 %7.2f ms (%5.1f TF/s)\n",
               diff ? "BAD" : "ok ", s.n, s.M, s.N, s.K, (int)diff, t0, tf / (t0 * 1e-3), t1, tf / (t1 * 1e-3));
        A.free_(); B.free_(); C0.free_(); C1.free_();
    }
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
