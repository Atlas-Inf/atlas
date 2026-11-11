// SPDX-License-Identifier: AGPL-3.0-only
// QSA decode past the inert bound at long context: bitwise checks and timings
// of the per-row serial path against the batched rows path
// (ATLAS_QWEN4EXP_QSA_DECODE_ROWS, qsa_decode_rows.cu), one TP2 rank's shapes.
//
//   check  — every row: qsa_qprep == qsa_qprep_rows; qsa_score ==
//            qsa_score_rows_exact (== _v4); qsa_select_topk_radix == the host
//            sort == qsa_select_topk_radix_rows; gather + paged_decode_attn ==
//            qsa_sparse_decode_attn. Byte for byte.
//   time   — one layer's QSA work for R rows: (a) serial rows with the host
//            top-k (the arm past 16384 blocks), (b) serial rows with the device
//            radix top-k, (c) the batched rows path; plus each kernel alone.
//
// Build + run: scripts/dev/qwen4exp_qsa_decode_bench.sh [check|time] [pos] [rows] [nq] [nkv]
#include "qwen4exp_ptx_harness.h"
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

static const unsigned NH = 4, IHD = 128, QKW = (NH + 1) * IHD, RATIO = 4, TOPK = 512;
static const unsigned HD = 256, BS = 16, ROT = 32, MAXSEL = 4096;
static const float THETA = 1e7f, EPS = 1e-6f;

struct Geo { unsigned complete, tail, visible, n_sel; };
static Geo geo(unsigned pos) {
    Geo g; g.visible = pos + 1; g.complete = g.visible / RATIO; g.tail = g.complete * RATIO;
    g.n_sel = TOPK * RATIO + (g.visible - g.tail); return g;
}

// Host reference of `select_blocks` + `expand_selection` (qsa_decode_select.rs).
static float rank_key(float s) { return std::isnan(s) ? -INFINITY : (s == 0.0f ? 0.0f : s); }
static std::vector<int> host_select(const std::vector<float>& sc, const Geo& g) {
    std::vector<unsigned> o(g.complete);
    std::iota(o.begin(), o.end(), 0u);
    std::stable_sort(o.begin(), o.end(), [&](unsigned a, unsigned b) {
        float ka = rank_key(sc[a]), kb = rank_key(sc[b]);
        return ka != kb ? ka > kb : a < b;
    });
    o.resize(TOPK);
    std::sort(o.begin(), o.end());
    std::vector<int> sel;
    for (unsigned b : o) for (unsigned r = 0; r < RATIO; ++r) sel.push_back(b * RATIO + r);
    for (unsigned t = g.tail; t < g.visible; ++t) sel.push_back(t);
    return sel;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [check|time] [pos] [rows] [nq] [nkv]\n", argv[0]); return 2; }
    std::string dir = argv[1], mode = argc > 2 ? argv[2] : "check";
    const unsigned P = argc > 3 ? atoi(argv[3]) : 77000;
    const unsigned R = argc > 4 ? atoi(argv[4]) : 4;
    const unsigned NQ = argc > 5 ? atoi(argv[5]) : 12, NKV = argc > 6 ? atoi(argv[6]) : 1;
    init_driver();
    PtxModule mi, mr, mp;
    mi.load(dir + "/qsa_indexer.ptx");
    mr.load(dir + "/qsa_decode_rows.ptx");
    mp.load(dir + "/paged_decode_attn.ptx");
    const unsigned smem_se = (8 * NH * IHD + 32 * (IHD + 1)) * 4, smem_v4 = (16 * NH * IHD + 32 * (IHD + 4)) * 4;
    CUfunction k_qprep = mi.fn("qsa_qprep"), k_qprep_rows = mi.fn("qsa_qprep_rows"), k_score = mi.fn("qsa_score");
    CUfunction k_se = mi.fn("qsa_score_rows_exact", smem_se), k_v4 = mi.fn("qsa_score_rows_exact_v4", smem_v4);
    CUfunction k_radix = mi.fn("qsa_select_topk_radix"), k_gather = mi.fn("qsa_gather");
    CUfunction k_radix_rows = mr.fn("qsa_select_topk_radix_rows"), k_sparse = mr.fn("qsa_sparse_decode_attn");
    CUfunction k_attn = mp.fn("paged_decode_attn");
    CUfunction k_dec = mr.fn("qsa_score_rows_dec");

    const unsigned last = P + R - 1, ntok = last + 1, nblk_keys = ntok / RATIO + 1;
    const unsigned stride = nblk_keys, pages = (ntok + BS - 1) / BS + 1;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto bf_vec = [&](size_t n, float sc) { std::vector<unsigned short> v(n); for (auto& x : v) x = f2bf(nd(rng) * sc); return v; };
    // Indexer: qk rows and pooled block keys. KV: a shuffled page table.
    Buf<unsigned short> d_qk, d_keys, d_k, d_v, d_q, d_ks, d_vs, d_o1, d_o2;
    Buf<float> d_qp1, d_qp2, d_s1, d_s2, d_s3;
    Buf<int> d_tab, d_ident, d_sel1, d_sel2, d_seqlen;
    d_qk.alloc((size_t)R * QKW); d_qk.put(bf_vec((size_t)R * QKW, 1.0f));
    d_keys.alloc((size_t)nblk_keys * IHD); d_keys.put(bf_vec((size_t)nblk_keys * IHD, 1.0f));
    std::vector<int> tab(pages); std::iota(tab.begin(), tab.end(), 0); std::shuffle(tab.begin(), tab.end(), rng);
    d_tab.alloc(pages); d_tab.put(tab);
    std::vector<int> ident(MAXSEL / BS + 1); std::iota(ident.begin(), ident.end(), 0);
    d_ident.alloc(ident.size()); d_ident.put(ident);
    const size_t kv_elems = (size_t)pages * BS * NKV * HD;
    d_k.alloc(kv_elems); d_k.put(bf_vec(kv_elems, 1.0f));
    d_v.alloc(kv_elems); d_v.put(bf_vec(kv_elems, 1.0f));
    const unsigned qstride = NQ * HD + 2 * NKV * HD + NQ * HD;   // a [Q|K|V|gate] row, as the verify buffer
    d_q.alloc((size_t)R * qstride); d_q.put(bf_vec((size_t)R * qstride, 1.0f));
    d_ks.alloc((size_t)MAXSEL * NKV * HD); d_vs.alloc((size_t)MAXSEL * NKV * HD);
    d_o1.alloc((size_t)R * NQ * HD); d_o2.alloc((size_t)R * NQ * HD);
    d_qp1.alloc((size_t)R * NH * IHD); d_qp2.alloc((size_t)R * NH * IHD);
    d_s1.alloc((size_t)R * stride); d_s2.alloc((size_t)R * stride); d_s3.alloc((size_t)R * stride);
    d_sel1.alloc((size_t)R * MAXSEL); d_sel2.alloc((size_t)R * MAXSEL); d_seqlen.alloc(1);
    const float inv_sqrt_d = 1.0f / sqrtf((float)HD);

    // ---- per-kernel launchers (launch shapes as ops/qsa.rs, ops/qsa_rows.rs) ----
    auto qprep = [&](unsigned r) { Args a; a.add(d_qk.p + (size_t)r * QKW).add(d_keys.p /*q_norm_w*/).add(d_qp1.p + (size_t)r * NH * IHD)
        .add(IHD).add(ROT).add(P + r).add(THETA).add(EPS); launch(k_qprep, dim3(NH), dim3(IHD), (IHD + 32) * 4, a); };
    auto qprep_rows = [&]() { Args a; a.add(d_qk.p).add(d_keys.p).add(d_qp2.p).add(P).add(QKW).add(NH).add(IHD).add(ROT).add(THETA).add(EPS);
        launch(k_qprep_rows, dim3(R, NH), dim3(IHD), (IHD + 32) * 4, a); };
    auto score = [&](unsigned r) { Args a; a.add(d_qp1.p + (size_t)r * NH * IHD).add(d_keys.p).add(d_s1.p + (size_t)r * stride).add(NH).add(IHD);
        launch(k_score, dim3(geo(P + r).complete), dim3(IHD), 32 * 4, a); };
    auto score_rows = [&](bool v4, float* out) { Args a; a.add(d_qp2.p).add(d_keys.p).add(out).add(P).add(stride).add(RATIO).add(NH).add(IHD).add(R).add(geo(last).complete);
        const unsigned bm = v4 ? 16 : 8; launch(v4 ? k_v4 : k_se, dim3((R + bm - 1) / bm, (geo(last).complete + 31) / 32), dim3(256), v4 ? smem_v4 : smem_se, a); };
    auto score_dec = [&](float* out) { Args a; a.add(d_qp2.p).add(d_keys.p).add(out).add(P).add(stride).add(RATIO).add(NH).add(IHD).add(R).add(geo(last).complete);
        launch(k_dec, dim3((geo(last).complete + 127) / 128), dim3(128), R * NH * IHD * 4, a); };
    auto radix = [&](unsigned r) { Geo g = geo(P + r); Args a; a.add(d_s1.p + (size_t)r * stride).add(d_sel1.p + (size_t)r * MAXSEL)
        .add((int)g.complete).add((int)TOPK).add((int)RATIO).add((int)g.tail).add((int)g.visible); launch(k_radix, dim3(1), dim3(1024), 0, a); };
    auto radix_rows = [&](float* sc) { Args a; a.add(sc).add(d_sel2.p).add(stride).add(MAXSEL).add(P).add((int)TOPK).add((int)RATIO);
        launch(k_radix_rows, dim3(R), dim3(1024), 0, a); };
    auto gather = [&](unsigned r) { Args a; a.add(d_k.p).add(d_v.p).add(d_tab.p).add(d_sel1.p + (size_t)r * MAXSEL).add(d_ks.p).add(d_vs.p)
        .add(BS).add(NKV).add(HD); launch(k_gather, dim3(geo(P + r).n_sel), dim3(256), 0, a); };
    auto attn = [&](unsigned r) { const unsigned n = geo(P + r).n_sel;
        CK(cudaMemcpyAsync(d_seqlen.p, &n, 4, cudaMemcpyHostToDevice, 0));
        Args a; a.add(d_q.p + (size_t)r * qstride).add(d_ks.p).add(d_vs.p).add(d_o1.p + (size_t)r * NQ * HD).add(d_ident.p).add(d_seqlen.p)
        .add((n + BS - 1) / BS).add(NQ).add(NKV).add(HD).add(BS).add(inv_sqrt_d).add(NQ * HD).add(0u); launch(k_attn, dim3(NQ, 1), dim3(256), 0, a); };
    auto sparse = [&]() { Args a; a.add(d_q.p).add(d_k.p).add(d_v.p).add(d_o2.p).add(d_tab.p).add(d_sel2.p).add(MAXSEL).add(P).add((int)RATIO)
        .add((int)TOPK).add(NQ).add(NKV).add(HD).add(BS).add(inv_sqrt_d).add(qstride); launch(k_sparse, dim3(NQ, R), dim3(256), 0, a); };
    auto host_topk = [&](unsigned r) {   // the host arm: D2H, sort, H2D (a stream drain)
        Geo g = geo(P + r); std::vector<float> sc(g.complete);
        CK(cudaMemcpy(sc.data(), d_s1.p + (size_t)r * stride, g.complete * 4, cudaMemcpyDeviceToHost));
        std::vector<int> sel = host_select(sc, g);
        CK(cudaMemcpyAsync(d_sel1.p + (size_t)r * MAXSEL, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice, 0));
        CK(cudaStreamSynchronize(0));   // pageable H2D source must outlive the copy
    };

    printf("pos %u rows %u: complete %u..%u blocks, n_sel %u, per rank nq %u nkv %u hd %u\n",
           P, R, geo(P).complete, geo(last).complete, geo(P).n_sel, NQ, NKV, HD);
    if (mode == "check") {
        size_t bad = 0;
        for (unsigned r = 0; r < R; ++r) { qprep(r); score(r); radix(r); }
        qprep_rows(); score_rows(false, d_s2.p); score_rows(true, d_s3.p); radix_rows(d_s3.p);
        CK(cudaDeviceSynchronize());
        Buf<float> d_s4; d_s4.alloc((size_t)R * stride); score_dec(d_s4.p); CK(cudaDeviceSynchronize());
        auto s4 = d_s4.get();
        size_t dq = diff_bytes(d_qp1.get(), d_qp2.get());
        printf("qprep vs qprep_rows: %zu differing bytes\n", dq); bad += dq;
        auto s1 = d_s1.get(), s2 = d_s2.get(), s3 = d_s3.get();
        auto sel1 = d_sel1.get(), sel2 = d_sel2.get();
        for (unsigned r = 0; r < R; ++r) {
            Geo g = geo(P + r);
            size_t e = 0, v = 0, sh = 0, sr = 0, dd = 0;
            for (unsigned b = 0; b < g.complete; ++b) {
                size_t i = (size_t)r * stride + b;
                e += memcmp(&s1[i], &s2[i], 4) != 0; v += memcmp(&s1[i], &s3[i], 4) != 0;
                dd += memcmp(&s1[i], &s4[i], 4) != 0;
            }
            std::vector<float> sc(s1.begin() + (size_t)r * stride, s1.begin() + (size_t)r * stride + g.complete);
            std::vector<int> hs = host_select(sc, g);
            for (unsigned i = 0; i < g.n_sel; ++i) {
                sh += hs[i] != sel1[(size_t)r * MAXSEL + i]; sr += sel1[(size_t)r * MAXSEL + i] != sel2[(size_t)r * MAXSEL + i];
            }
            printf("row %u: score!=exact %zu, score!=v4 %zu, score!=dec %zu, radix!=host %zu, radix!=radix_rows %zu ids\n", r, e, v, dd, sh, sr);
            bad += e + v + dd + sh + sr;
        }
        for (unsigned r = 0; r < R; ++r) { gather(r); attn(r); CK(cudaDeviceSynchronize()); }
        sparse();
        CK(cudaDeviceSynchronize());
        size_t da = diff_bytes(d_o1.get(), d_o2.get());
        printf("gather+paged_decode_attn vs qsa_sparse_decode_attn (%u rows): %zu differing bytes\n", R, da);
        bad += da;
        printf("%s\n", bad == 0 ? "PASS" : "FAIL");
        return bad == 0 ? 0 : 1;
    }
    // ---- timings: one layer's QSA work for R rows ----
    auto wall_ms = [&](auto body, int iters) {
        body(); CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) body();
        CK(cudaDeviceSynchronize());
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    };
    for (unsigned r = 0; r < R; ++r) { qprep(r); score(r); }
    double a = wall_ms([&] { for (unsigned r = 0; r < R; ++r) { qprep(r); score(r); host_topk(r); gather(r); attn(r); } }, 20);
    double b = wall_ms([&] { for (unsigned r = 0; r < R; ++r) { qprep(r); score(r); radix(r); gather(r); attn(r); } }, 50);
    double c = wall_ms([&] { qprep_rows(); score_dec(d_s3.p); radix_rows(d_s3.p); sparse(); }, 50);
    printf("layer, %u rows: (a) serial + host top-k %.3f ms | (b) serial + device radix %.3f ms | (c) batched rows %.3f ms\n", R, a, b, c);
    printf("  x12 layers: (a) %.2f ms  (b) %.2f ms  (c) %.2f ms per verify step\n", 12 * a, 12 * b, 12 * c);
    printf("score_rows_dec (R rows) %.1f us\n", 1e3 * time_ms([&] { score_dec(d_s3.p); }));
    printf("kernels (us): qsa_score %.1f | score_rows_exact %.1f | v4 %.1f | radix %.1f | radix_rows %.1f | gather %.1f | paged_decode_attn %.1f | sparse_attn(R) %.1f\n",
           1e3 * time_ms([&] { score(0); }), 1e3 * time_ms([&] { score_rows(false, d_s2.p); }), 1e3 * time_ms([&] { score_rows(true, d_s3.p); }),
           1e3 * time_ms([&] { radix(0); }), 1e3 * time_ms([&] { radix_rows(d_s3.p); }), 1e3 * time_ms([&] { gather(0); }),
           1e3 * time_ms([&] { attn(0); }), 1e3 * time_ms([&] { sparse(); }));
    std::vector<float> sc(geo(P).complete);
    CK(cudaMemcpy(sc.data(), d_s1.p, sc.size() * 4, cudaMemcpyDeviceToHost));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) (void)host_select(sc, geo(P));
    double hs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
    printf("host sort alone %.3f ms; host arm round trip %.3f ms\n", hs, wall_ms([&] { host_topk(0); }, 20));
    return 0;
}
