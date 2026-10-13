// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of `qwen4exp_ba_gates_prefill_rows`
// (qwen4exp_gdn_prefill.cu) against `dense_gemm_ba_gates_prefill`
// (common/ssm_preprocess.cu), the GDN prefill BA GEMM + gates, at the
// qwen4_exp TP2 rank shape (48 BA outputs, 24 v-heads, hidden 2560) and a
// smaller one (32 outputs). Every gate/beta byte must match.
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4; C=kernels/gb10/common
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/ssm_preprocess.ptx $C/ssm_preprocess.cu
//   nvcc $F -o $D/qwen4exp_gdn_prefill.ptx $K/qwen4exp_gdn_prefill.cu
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_ba_gates_bench.cu -lcuda
//   $D/bench $D [QBA_TOK=4]
#include "qwen4exp_ptx_harness.h"
#include <random>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    init_driver();
    PtxModule m_def, m_new;
    m_def.load(dir + "/ssm_preprocess.ptx");
    m_new.load(dir + "/qwen4exp_gdn_prefill.ptx");
    CUfunction k_def = m_def.fn("dense_gemm_ba_gates_prefill");
    CUfunction k_new = m_new.fn("qwen4exp_ba_gates_prefill_rows");
    const unsigned K = 2560, TOK = argc > 2 ? atoi(argv[2]) : 4;   // QBA_TOK the PTX was built with
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool ok = true;
    for (unsigned nv : {24u, 16u}) {
        const unsigned N = 2 * nv, vpg = 2, gs = 2 * nv;
        for (unsigned M : {1u, 3u, 30u, 1001u, 16016u}) {
            std::vector<unsigned short> a((size_t)M * K), b((size_t)N * K);
            for (auto& x : a) x = f2bf(nd(rng));
            for (auto& x : b) x = f2bf(0.05f * nd(rng));
            std::vector<float> alog(nv), dtb(nv);
            for (auto& x : alog) x = 0.5f * nd(rng);
            for (auto& x : dtb) x = 0.5f * nd(rng);
            Buf<unsigned short> da, db;
            Buf<float> dal, ddt, o1, o2;
            da.alloc(a.size()); da.put(a);
            db.alloc(b.size()); db.put(b);
            dal.alloc(nv); dal.put(alog);
            ddt.alloc(nv); ddt.put(dtb);
            o1.alloc((size_t)M * gs); o2.alloc((size_t)M * gs);
            o1.fill(0x11); o2.fill(0x22);
            auto args = [&](Buf<float>& o) {
                Args x;
                x.add(da.p).add(db.p).add(dal.p).add(ddt.p).add(o.p).add(M).add(N).add(K).add(K)
                 .add(gs).add(nv).add(vpg);
                return x;
            };
            auto run_def = [&] { Args x = args(o1); launch(k_def, dim3((N + 3) / 4, M), dim3(256), 0, x); };
            auto run_new = [&] { Args x = args(o2); launch(k_new, dim3((M + TOK - 1) / TOK), dim3(256), 0, x); };
            run_def(); run_new();
            CK(cudaDeviceSynchronize());
            const size_t d = diff_bytes(o1.get(), o2.get());
            ok = ok && d == 0;
            printf("nv=%u M=%u: %zu differing bytes", nv, M, d);
            if (M >= 1001) {
                const float t1 = time_ms(run_def), t2 = time_ms(run_new);
                printf("  default %.3f ms -> rows %.3f ms (%.2fx)", t1, t2, t1 / t2);
            }
            printf("\n");
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
