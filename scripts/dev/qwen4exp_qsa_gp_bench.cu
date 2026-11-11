// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of `qsa_prefill_attn_gp` (qsa_attn_gp.cu) against
// `qsa_prefill_attn_g` (qsa_indexer.cu), the exact scalar QSA prefill
// attention TP2 runs today. Every output byte must match, at the TP2 rank
// shape (12 q / 1 kv) and the TP1 shape (24 q / 2 kv, so kv head 1 is read
// too), for several positions (tails 0..3) and row counts.
//
// Build (repo root, GB10):
//   D=$(mktemp -d); K=kernels/gb10/qwen3.8-flash-next/nvfp4
//   F="--ptx -arch=sm_121f -O3 --fmad=false"
//   nvcc $F -o $D/qsa_indexer.ptx $K/qsa_indexer.cu
//   nvcc $F -o $D/qsa_attn_gp.ptx $K/qsa_attn_gp.cu
//   (optional ring-depth sweep points: -DQSA_GP_DEPTH=<n> as
//    $D/qsa_attn_gp_d<n>.ptx, named on the command line as d<n>)
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_qsa_gp_bench.cu -lcuda
//   $D/bench $D [first_pos=14000] [rows=2048] [tag ...]
#include "qwen4exp_ptx_harness.h"
#include <random>

static const unsigned HD = 256, BS = 16, RATIO = 4, TOPK = 512, G = 4, WARPS = 8;

static unsigned g_smem() { return (WARPS * G * HD + 2 * WARPS * G) * 4; }
static unsigned gp_smem(unsigned depth) {
    const unsigned loop = WARPS * depth * 2 * 256 * 2 + (TOPK * RATIO + RATIO) * 4;
    return std::max(loop, g_smem());
}

struct Case {
    unsigned nq, nkv, first_pos, rows;
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [first_pos] [rows] [tag...]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const unsigned F = argc > 2 ? atoi(argv[2]) : 14000;
    const unsigned R = argc > 3 ? atoi(argv[3]) : 2048;
    init_driver();
    PtxModule m_idx;
    m_idx.load(dir + "/qsa_indexer.ptx");
    CUfunction k_g = m_idx.fn("qsa_prefill_attn_g", g_smem());
    struct Variant { std::string tag; unsigned depth; PtxModule mod; CUfunction fn; };
    std::vector<Variant> vs;
    std::vector<std::string> tags = {""};
    for (int i = 4; i < argc; ++i) tags.push_back(argv[i]);
    for (auto& tag : tags) {
        Variant v;
        v.tag = tag.empty() ? "gp" : tag;
        // Tag "d<n>..." sets the ring depth the launcher sizes for.
        v.depth = 4;
        if (tag.size() > 1 && tag[0] == 'd') v.depth = atoi(tag.c_str() + 1);
        const std::string f = dir + "/qsa_attn_gp" + (tag.empty() ? "" : "_" + tag) + ".ptx";
        if (!v.mod.try_load(f)) { fprintf(stderr, "skip %s\n", f.c_str()); continue; }
        v.fn = v.mod.fn("qsa_prefill_attn_gp", gp_smem(v.depth));
        vs.push_back(std::move(v));
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float inv = 1.0f / sqrtf((float)HD);
    bool ok = true;
    std::vector<Case> cases = {{12, 1, F, R}, {12, 1, F + 1, R - 1}, {12, 1, F + 2, 37},
                               {12, 1, 2047, 300}, {24, 2, F + 3, 513}, {24, 2, 30001, 64},
                               {12, 1, 2051, R}, {12, 1, 30000, R}};
    for (const Case& c : cases) {
        const unsigned ctx = c.first_pos + c.rows;
        const unsigned pages = (ctx + BS - 1) / BS;
        const size_t slots = (size_t)pages * BS;
        std::vector<unsigned short> k(slots * c.nkv * HD), v(slots * c.nkv * HD), q((size_t)c.rows * c.nq * HD);
        for (auto& x : k) x = f2bf(nd(rng));
        for (auto& x : v) x = f2bf(nd(rng));
        for (auto& x : q) x = f2bf(0.5f * nd(rng));
        std::vector<int> table(pages);
        for (unsigned i = 0; i < pages; ++i) table[i] = i;
        std::shuffle(table.begin(), table.end(), rng);
        std::vector<int> lists((size_t)c.rows * TOPK);
        for (unsigned r = 0; r < c.rows; ++r) {
            const unsigned complete = (c.first_pos + r + 1) / RATIO;
            std::vector<int> ids(std::max(complete, TOPK));
            for (unsigned i = 0; i < ids.size(); ++i) ids[i] = i % std::max(complete, 1u);
            for (unsigned i = 0; i < TOPK; ++i) std::swap(ids[i], ids[i + rng() % (ids.size() - i)]);
            memcpy(&lists[(size_t)r * TOPK], ids.data(), TOPK * 4);
        }
        Buf<unsigned short> dk, dv, dq, dog, dop;
        Buf<int> dtab, dlist;
        dk.alloc(k.size()); dk.put(k);
        dv.alloc(v.size()); dv.put(v);
        dq.alloc(q.size()); dq.put(q);
        dog.alloc(q.size()); dop.alloc(q.size());
        dtab.alloc(pages); dtab.put(table);
        dlist.alloc(lists.size()); dlist.put(lists);
        auto run = [&](CUfunction f, unsigned smem, Buf<unsigned short>& out) {
            Args a;
            a.add(dq.p).add(dk.p).add(dv.p).add(dtab.p).add(dlist.p).add(out.p)
             .add(c.first_pos).add(TOPK).add(RATIO).add(BS).add(c.nq).add(c.nkv).add(HD).add(inv);
            launch(f, dim3(c.rows, c.nq / G), dim3(256), smem, a);
        };
        dog.fill(0xEE);
        run(k_g, g_smem(), dog);
        CK(cudaDeviceSynchronize());
        const auto ref = dog.get();
        for (auto& vv : vs) {
            dop.fill(0xEE);
            run(vv.fn, gp_smem(vv.depth), dop);
            CK(cudaDeviceSynchronize());
            const size_t d = diff_bytes(dop.get(), ref);
            printf("bitwise %-8s vs _g  nq=%u nkv=%u first_pos=%u rows=%u: %zu differing bytes\n",
                   vv.tag.c_str(), c.nq, c.nkv, c.first_pos, c.rows, d);
            ok = ok && d == 0;
        }
        if (c.rows == R) {
            const double gflop = 2.0 * 2.0 * c.rows * c.nq * (TOPK * RATIO) * HD / 1e9;
            const float t_g = time_ms([&] { run(k_g, g_smem(), dog); });
            printf("timing rows=%u first_pos=%u (%u q / %u kv):\n  _g       %8.3f ms  %6.2f TFLOP/s\n",
                   c.rows, c.first_pos, c.nq, c.nkv, t_g, gflop / t_g);
            for (auto& vv : vs) {
                const float t = time_ms([&] { run(vv.fn, gp_smem(vv.depth), dop); });
                printf("  %-8s %8.3f ms  %6.2f TFLOP/s  (%.2fx vs _g)\n", vv.tag.c_str(), t, gflop / t, t_g / t);
            }
        }
        dk.free_(); dv.free_(); dq.free_(); dog.free_(); dop.free_(); dtab.free_(); dlist.free_();
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
