// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of `qwen4exp_fp8_gemm_w2` (qwen4exp_fp8_gemm.cu)
// against the qwen4_exp attention prefill projections' default kernels
// (w4a16_gemm.cu): `fp8_fp8_gemm_t_m128` (FP8 activations: q+gate, k, v)
// and `fp8_gemm_t_m128` (BF16 activations, converted in its K loop: o_proj;
// the twin reads them converted once by `moe_q38_a_to_e4m3`). Every output
// byte must match.
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/w4a16.ptx $K/w4a16_gemm.cu
//   nvcc $F -o $D/qwen4exp_fp8_gemm.ptx $K/qwen4exp_fp8_gemm.cu
//   nvcc $F -o $D/moe_prefill_q38.ptx $K/moe_prefill_q38.cu
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_fp8_gemm_bench.cu -lcuda
//   $D/bench $D [QF_BK=128] [QF_STAGES=2]
#include "qwen4exp_ptx_harness.h"
#include <random>

static unsigned char e4m3_any(std::mt19937& r) {
    unsigned char b = (unsigned char)(r() & 0xFF);
    return (b & 0x7F) == 0x7F ? (unsigned char)(b & 0xF0) : b;   // no NaN
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    init_driver();
    PtxModule m_def, m_new, m_q38;
    m_def.load(dir + "/w4a16.ptx");
    m_new.load(dir + "/qwen4exp_fp8_gemm.ptx");
    m_q38.load(dir + "/moe_prefill_q38.ptx");
    CUfunction k_ff = m_def.fn("fp8_fp8_gemm_t_m128"), k_bf = m_def.fn("fp8_gemm_t_m128");
    // QF_BK / QF_STAGES the PTX was built with (argv[2], argv[3]).
    const unsigned BK = argc > 2 ? atoi(argv[2]) : 128, ST = argc > 3 ? atoi(argv[3]) : 2;
    const unsigned smem = ST * 256 * (BK + 16);
    CUfunction k_w2 = m_new.fn("qwen4exp_fp8_gemm_w2", smem), k_a8 = m_q38.fn("moe_q38_a_to_e4m3");
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool ok = true;
    struct S { const char* name; unsigned M, N, K; bool bf16_a; };
    for (S s : {S{"q+gate", 16016, 6144, 2560, false}, S{"k/v", 16016, 256, 2560, false},
                S{"o_proj", 16016, 2560, 3072, true}, S{"q+gate tail", 1001, 6144, 2560, false},
                S{"o_proj tail", 333, 2560, 3072, true}}) {
        std::vector<unsigned char> b((size_t)s.N * s.K), a8((size_t)s.M * s.K);
        for (auto& x : b) x = e4m3_any(rng);
        for (auto& x : a8) x = e4m3_any(rng);
        std::vector<unsigned short> abf((size_t)s.M * s.K);
        for (auto& x : abf) x = f2bf(nd(rng));
        Buf<unsigned char> db, da8, da8c;
        Buf<unsigned short> dabf, o1, o2;
        db.alloc(b.size()); db.put(b);
        da8.alloc(a8.size()); da8.put(a8);
        da8c.alloc(a8.size());
        dabf.alloc(abf.size()); dabf.put(abf);
        o1.alloc((size_t)s.M * s.N); o2.alloc((size_t)s.M * s.N);
        o1.fill(0x11); o2.fill(0x22);
        auto run_def = [&] {
            Args x;
            if (s.bf16_a) x.add(dabf.p); else x.add(da8.p);
            x.add(db.p).add(o1.p).add(s.M).add(s.N).add(s.K);
            launch(s.bf16_a ? k_bf : k_ff, dim3((s.N + 127) / 128, (s.M + 127) / 128), dim3(128), 0, x);
        };
        auto run_new = [&] {
            if (s.bf16_a) {
                Args c; c.add(dabf.p).add(da8c.p).add(s.M * s.K);
                launch(k_a8, dim3((s.M * s.K / 4 + 255) / 256), dim3(256), 0, c);
            }
            Args x;
            x.add(s.bf16_a ? da8c.p : da8.p).add(db.p).add(o2.p).add(s.M).add(s.N).add(s.K);
            launch(k_w2, dim3((s.N + 127) / 128, (s.M + 127) / 128), dim3(256), smem, x);
        };
        run_def(); run_new();
        CK(cudaDeviceSynchronize());
        const size_t d = diff_bytes(o1.get(), o2.get());
        ok = ok && d == 0;
        const double tf = 2.0 * s.M * s.N * s.K / 1e9;
        const float t1 = time_ms(run_def), t2 = time_ms(run_new);
        printf("%-12s M=%u N=%u K=%u: %zu differing bytes; default %.3f ms (%.0f TFLOP/s) -> w2 %.3f ms (%.0f TFLOP/s) %.2fx\n",
               s.name, s.M, s.N, s.K, d, t1, tf / t1, t2, tf / t2, t1 / t2);
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
