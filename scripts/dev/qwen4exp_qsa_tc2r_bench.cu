// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of `qsa_prefill_attn_tc2r` (TP2 QSA prefill
// attention, two rows per CTA, one kv head) against:
//
//   tc2  -- TP1's default kernel on the SAME data laid out with two kv heads:
//           every output byte of tc2r must equal tc2's for those heads, for
//           kv head 0 and (by feeding tc2r the other half) kv head 1, and for
//           an odd row count (the last CTA's second half is empty).
//   _g   -- the exact scalar kernel TP2 runs today (reported, not bitwise:
//           tc2 is a different summation tree; max |diff| is printed).
//   tc   -- the existing opt-in tensor-core kernel (BR=32, one kv head).
//
// Build (from the repo root, on a GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4
//   for f in qsa_attn_tc2 qsa_attn_tc2r qsa_attn_tc qsa_indexer; do
//     nvcc --ptx -arch=sm_121f -O3 --fmad=false -o $D/$f.ptx $K/$f.cu; done
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_qsa_tc2r_bench.cu -lcuda
//   $D/bench $D [first_pos=14000] [rows=2048]
#include "qwen4exp_ptx_harness.h"
#include <random>

static const unsigned HD = 256, BS = 16, RATIO = 4, TOPK = 512;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [first_pos] [rows]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned F = argc > 2 ? atoi(argv[2]) : 14000;
    const unsigned R = argc > 3 ? atoi(argv[3]) : 2048;
    init_driver();
    PtxModule m_tc2, m_tc2r, m_tc, m_idx;
    m_tc2.load(dir + "/qsa_attn_tc2.ptx");
    m_tc2r.load(dir + "/qsa_attn_tc2r.ptx");
    m_tc.load(dir + "/qsa_attn_tc.ptx");
    m_idx.load(dir + "/qsa_indexer.ptx");
    const unsigned g_smem = (8 * 4 * HD + 2 * 8 * 4) * 4;
    CUfunction k_tc2 = m_tc2.fn("qsa_prefill_attn_tc2");
    CUfunction k_tc2r = m_tc2r.fn("qsa_prefill_attn_tc2r");
    // Optional variants built from the same sources with -DQSA_TC2_LEAN
    // (qsa_attn_tc2r_lean.ptx, qsa_attn_tc2_lean.ptx): checked and timed too.
    auto exists = [&](const char* f) { std::ifstream t(dir + "/" + f); return (bool)t; };
    PtxModule m_tc2r_lean, m_tc2_lean;
    CUfunction k_tc2r_lean = nullptr, k_tc2_lean = nullptr;
    if (exists("qsa_attn_tc2r_lean.ptx")) {
        m_tc2r_lean.load(dir + "/qsa_attn_tc2r_lean.ptx");
        k_tc2r_lean = m_tc2r_lean.fn("qsa_prefill_attn_tc2r");
    }
    if (exists("qsa_attn_tc2_lean.ptx")) {
        m_tc2_lean.load(dir + "/qsa_attn_tc2_lean.ptx");
        k_tc2_lean = m_tc2_lean.fn("qsa_prefill_attn_tc2");
    }
    CUfunction k_tc = m_tc.fn("qsa_prefill_attn_tc");
    CUfunction k_g = m_idx.fn("qsa_prefill_attn_g", g_smem);

    const unsigned ctx = F + R;
    const unsigned pages = (ctx + BS - 1) / BS;
    const size_t slots = (size_t)pages * BS;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Two-kv-head pools [slot, 2, HD] and the per-head one-kv-head views.
    std::vector<unsigned short> k2(slots * 2 * HD), v2(slots * 2 * HD);
    for (auto& x : k2) x = f2bf(nd(rng));
    for (auto& x : v2) x = f2bf(nd(rng));
    std::vector<unsigned short> k1[2], v1[2];
    for (int h = 0; h < 2; ++h) {
        k1[h].resize(slots * HD);
        v1[h].resize(slots * HD);
        for (size_t s = 0; s < slots; ++s)
            for (unsigned d = 0; d < HD; ++d) {
                k1[h][s * HD + d] = k2[(s * 2 + h) * HD + d];
                v1[h][s * HD + d] = v2[(s * 2 + h) * HD + d];
            }
    }
    std::vector<unsigned short> q2((size_t)R * 24 * HD), q1[2];
    for (auto& x : q2) x = f2bf(0.5f * nd(rng));
    for (int h = 0; h < 2; ++h) {
        q1[h].resize((size_t)R * 12 * HD);
        for (unsigned r = 0; r < R; ++r)
            memcpy(&q1[h][(size_t)r * 12 * HD], &q2[((size_t)r * 24 + h * 12) * HD], 12 * HD * 2);
    }
    std::vector<int> table(pages);
    for (unsigned i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<int> lists((size_t)R * TOPK);
    for (unsigned r = 0; r < R; ++r) {
        const unsigned complete = (F + r + 1) / RATIO;
        std::vector<int> ids(complete);
        for (unsigned i = 0; i < complete; ++i) ids[i] = i;
        for (unsigned i = 0; i < TOPK; ++i) std::swap(ids[i], ids[i + rng() % (complete - i)]);
        memcpy(&lists[(size_t)r * TOPK], ids.data(), TOPK * 4);
    }

    Buf<unsigned short> dk2, dv2, dq2, do2, dk1[2], dv1[2], dq1[2], do1, dog;
    Buf<int> dtab, dlist;
    dk2.alloc(k2.size()); dk2.put(k2);
    dv2.alloc(v2.size()); dv2.put(v2);
    dq2.alloc(q2.size()); dq2.put(q2);
    do2.alloc(q2.size());
    for (int h = 0; h < 2; ++h) {
        dk1[h].alloc(k1[h].size()); dk1[h].put(k1[h]);
        dv1[h].alloc(v1[h].size()); dv1[h].put(v1[h]);
        dq1[h].alloc(q1[h].size()); dq1[h].put(q1[h]);
    }
    do1.alloc(q1[0].size());
    dog.alloc(q1[0].size());
    dtab.alloc(pages); dtab.put(table);
    dlist.alloc(lists.size()); dlist.put(lists);
    const float inv = 1.0f / sqrtf((float)HD);

    auto run_tc2k = [&](CUfunction k, unsigned rows) {
        Args a;
        a.add(dq2.p).add(dk2.p).add(dv2.p).add(do2.p).add(dtab.p).add(dlist.p)
         .add(F).add(TOPK).add(RATIO).add(BS).add(24u).add(2u).add(HD).add(inv);
        launch(k, dim3(1, rows), dim3(128), 0, a);
    };
    auto run_tc2 = [&](unsigned rows) { run_tc2k(k_tc2, rows); };
    auto run_tc2rk = [&](CUfunction k, int h, unsigned rows) {
        Args a;
        a.add(dq1[h].p).add(dk1[h].p).add(dv1[h].p).add(do1.p).add(dtab.p).add(dlist.p)
         .add(F).add(TOPK).add(RATIO).add(BS).add(12u).add(1u).add(HD).add(inv).add(rows);
        launch(k, dim3(1, (rows + 1) / 2), dim3(128), 0, a);
    };
    auto run_tc2r = [&](int h, unsigned rows) { run_tc2rk(k_tc2r, h, rows); };
    auto run_tc1 = [&](unsigned rows) {
        Args a;
        a.add(dq1[0].p).add(dk1[0].p).add(dv1[0].p).add(dog.p).add(dtab.p).add(dlist.p)
         .add(F).add(TOPK).add(RATIO).add(BS).add(12u).add(1u).add(HD).add(inv);
        launch(k_tc, dim3(1, rows), dim3(128), 0, a);
    };
    auto run_g = [&](unsigned rows) {
        Args a;
        a.add(dq1[0].p).add(dk1[0].p).add(dv1[0].p).add(dtab.p).add(dlist.p).add(dog.p)
         .add(F).add(TOPK).add(RATIO).add(BS).add(12u).add(1u).add(HD).add(inv);
        launch(k_g, dim3(rows, 12 / 4), dim3(256), g_smem, a);
    };

    bool ok = true;
    for (unsigned rows : {R, R - 1}) {
        do2.fill(0xEE);
        run_tc2(rows);
        CK(cudaDeviceSynchronize());
        auto ref = do2.get();
        if (k_tc2_lean) {
            do2.fill(0xEE);
            run_tc2k(k_tc2_lean, rows);
            CK(cudaDeviceSynchronize());
            size_t d = diff_bytes(do2.get(), ref);
            printf("bitwise tc2(lean) vs tc2  rows=%u: %zu differing bytes\n", rows, d);
            ok = ok && d == 0;
        }
        for (int v = 0; v < 2; ++v)
        for (int h = 0; h < 2; ++h) {
            CUfunction k = v ? k_tc2r_lean : k_tc2r;
            if (!k) continue;
            do1.fill(0xEE);
            run_tc2rk(k, h, rows);
            CK(cudaDeviceSynchronize());
            auto got = do1.get();
            size_t d = 0, untouched = 0;
            for (unsigned r = 0; r < R; ++r)
                for (unsigned e = 0; e < 12 * HD; ++e) {
                    unsigned short x = got[(size_t)r * 12 * HD + e];
                    if (r >= rows) { untouched += x != 0xEEEE; continue; }
                    d += x != ref[((size_t)r * 24 + h * 12) * HD + e];
                }
            printf("bitwise tc2r%s vs tc2  rows=%u kv-head=%d: %zu differing elements%s\n",
                   v ? "(lean)" : "", rows, h, d, untouched ? " (WROTE PAST rows)" : "");
            ok = ok && d == 0 && untouched == 0;
        }
    }
    {
        run_g(R);
        run_tc2r(0, R);
        CK(cudaDeviceSynchronize());
        auto g = dog.get(), t = do1.get();
        double mx = 0, dot = 0, na = 0, nb = 0;
        size_t same = 0;
        for (size_t i = 0; i < g.size(); ++i) {
            double a = bf2f(g[i]), b = bf2f(t[i]);
            mx = std::max(mx, fabs(a - b));
            dot += a * b; na += a * a; nb += b * b;
            same += g[i] == t[i];
        }
        printf("tc2r vs _g (not bitwise by design): max|d|=%.3g cos=%.9f same=%.2f%%\n",
               mx, dot / sqrt(na * nb), 100.0 * same / g.size());
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    const double gflop = 2.0 * 2.0 * R * 12.0 * (TOPK * RATIO) * HD / 1e9;  // QK + PV, per rank
    float t_g = time_ms([&] { run_g(R); });
    float t_tc1 = time_ms([&] { run_tc1(R); });
    float t_tc2r = time_ms([&] { run_tc2r(0, R); });
    float t_tc2 = time_ms([&] { run_tc2(R); });
    printf("rows=%u first_pos=%u (12 q / 1 kv per rank; tc2 = TP1 shape, 24 q / 2 kv)\n", R, F);
    printf("  _g    %8.3f ms  %6.2f TFLOP/s\n", t_g, gflop / t_g);
    printf("  tc    %8.3f ms  %6.2f TFLOP/s\n", t_tc1, gflop / t_tc1);
    printf("  tc2r  %8.3f ms  %6.2f TFLOP/s  (%.2fx vs _g)\n", t_tc2r, gflop / t_tc2r, t_g / t_tc2r);
    printf("  tc2   %8.3f ms  %6.2f TFLOP/s  (24 heads: 2x the work)\n", t_tc2, 2 * gflop / t_tc2);
    if (k_tc2r_lean) {
        float t = time_ms([&] { run_tc2rk(k_tc2r_lean, 0, R); });
        printf("  tc2r lean %8.3f ms  %6.2f TFLOP/s  (%.2fx vs _g)\n", t, gflop / t, t_g / t);
    }
    if (k_tc2_lean) {
        float t = time_ms([&] { run_tc2k(k_tc2_lean, R); });
        printf("  tc2 lean  %8.3f ms  %6.2f TFLOP/s  (%.2fx vs tc2)\n", t, 2 * gflop / t, t_tc2 / t);
    }
    return ok ? 0 : 1;
}
