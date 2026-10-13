// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of `qsa_score_rows_exact_v4` against
// `qsa_score_rows_exact` (qsa_indexer.cu): the QSA prefill block scorer, 4
// indexer heads x hd 128, one 2048-row slab at a given position.
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4
//   nvcc --ptx -arch=sm_121f -O3 --fmad=false -o $D/qsa_indexer.ptx $K/qsa_indexer.cu
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_qsa_score_bench.cu -lcuda
//   $D/bench $D [first_pos=14000] [rows=2048] [rows_per_thread=2]
#include "qwen4exp_ptx_harness.h"
#include <random>

static const unsigned NH = 4, HD = 128, RATIO = 4, BM = 8, BN = 32;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [first_pos] [rows]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned F = argc > 2 ? atoi(argv[2]) : 14000;
    const unsigned R = argc > 3 ? atoi(argv[3]) : 2048;
    init_driver();
    PtxModule m;
    m.load(dir + "/qsa_indexer.ptx");
    const unsigned smem_ref = (BM * NH * HD + BN * (HD + 1)) * 4;
    const unsigned RPT = argc > 4 ? atoi(argv[4]) : 2;   // QSA_V4_RPT the PTX was built with
    const unsigned smem_v4 = (BM * RPT * NH * HD + BN * (HD + 4)) * 4;
    CUfunction k_ref = m.fn("qsa_score_rows_exact", smem_ref);
    CUfunction k_v4 = m.fn("qsa_score_rows_exact_v4", smem_v4);

    const unsigned total = F + R;
    const unsigned stride = (total + RATIO - 1) / RATIO;
    const unsigned nb_max = (F + R) / RATIO;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t)R * NH * HD);
    for (auto& x : q) x = nd(rng);
    std::vector<unsigned short> keys((size_t)stride * HD);
    for (auto& x : keys) x = f2bf(nd(rng));
    Buf<float> d_q, d_s1, d_s2;
    Buf<unsigned short> d_k;
    d_q.alloc(q.size()); d_q.put(q);
    d_k.alloc(keys.size()); d_k.put(keys);
    d_s1.alloc((size_t)R * stride); d_s2.alloc((size_t)R * stride);
    auto run = [&](CUfunction k, unsigned smem, float* out) {
        const unsigned bm = k == k_v4 ? BM * RPT : BM;
        Args a;
        a.add(d_q.p).add(d_k.p).add(out).add(F).add(stride).add(RATIO).add(NH).add(HD).add(R).add(nb_max);
        launch(k, dim3((R + bm - 1) / bm, (nb_max + BN - 1) / BN), dim3(BM * BN), smem, a);
    };
    d_s1.fill(0x3C); d_s2.fill(0x3C);
    run(k_ref, smem_ref, d_s1.p);
    run(k_v4, smem_v4, d_s2.p);
    CK(cudaDeviceSynchronize());
    size_t d = diff_bytes(d_s1.get(), d_s2.get());
    printf("bitwise scores (exact vs exact_v4) rows=%u first_pos=%u blocks=%u: %zu differing bytes\n",
           R, F, nb_max, d);
    float t1 = time_ms([&] { run(k_ref, smem_ref, d_s1.p); });
    float t2 = time_ms([&] { run(k_v4, smem_v4, d_s2.p); });
    printf("exact %.3f ms -> exact_v4 %.3f ms (%.2fx)\n", t1, t2, t1 / t2);
    printf("%s\n", d == 0 ? "PASS" : "FAIL");
    return d == 0 ? 0 : 1;
}
