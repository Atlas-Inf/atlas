// SPDX-License-Identifier: AGPL-3.0-only
// What a tensor-core verify MoE (exactness contract (b): row-invariant, a new
// numerics baseline) would cost at C1..C8, measured on the kernels the tree
// already serves prefill with: moe_prefill_q38.cu's FP8 mma.sync chain
// (ATLAS_QWEN4EXP_PREFILL_MOE[_W2]: W = e4m3(lut * scale * s2), A = e4m3(bf16),
// act = e4m3 of the BF16 SiLU*up, one FP32 accumulator per output over k32
// MMAs in increasing k) on the transposed [K/2, N] expert copy, at decode row
// counts: a2e4m3 -> routed gate/up+SiLU -> routed down, and the shared
// expert's dense pair. Then ROW INVARIANCE: every row's routed and shared
// outputs computed alone (1 row) and inside waves of 2..64 rows with varied
// expert overlap must be byte-equal.
//
// Build/run: scripts/dev/qwen4exp_moe_c8_bench.sh builds moe_prefill_q38.ptx
// and this (tc-time | tc-check).
#include "qwen4exp_moe_c8_bench.h"

struct TPool {  // transposed layout: packed [K/2][N], scales [K/16][N]
    std::vector<Proj> g, u, d;
    Proj sg, su, sd;
    u64 *gp, *gs, *up, *us, *dp, *ds;
    float *g2, *u2, *d2;
    explicit TPool(unsigned n) {
        for (unsigned e = 0; e < n; e++) {
            g.push_back(make_proj(I, H)); u.push_back(make_proj(I, H)); d.push_back(make_proj(H, I));
        }
        sg = make_proj(I, H); su = make_proj(I, H); sd = make_proj(H, I);
        // make_proj sizes [n*k/2] and [n*k/16]: the same bytes, read as [K/2][N].
        std::vector<u64> a(NE / 2), b(NE / 2), c(NE / 2), dd(NE / 2), e(NE / 2), f(NE / 2);
        std::vector<float> x(NE / 2), y(NE / 2), z(NE / 2);
        for (unsigned i = 0; i < NE / 2; i++) {
            const unsigned k = i % n;
            a[i] = (u64)g[k].packed; b[i] = (u64)g[k].scale; c[i] = (u64)u[k].packed;
            dd[i] = (u64)u[k].scale; e[i] = (u64)d[k].packed; f[i] = (u64)d[k].scale;
            x[i] = g[k].s2; y[i] = u[k].s2; z[i] = d[k].s2;
        }
        gp = dput(a); gs = dput(b); up = dput(c); us = dput(dd); dp = dput(e); ds = dput(f);
        g2 = dput(x); u2 = dput(y); d2 = dput(z);
    }
};

// One launch's routing as the q38 chain takes it: local entries (even global
// id e -> local e / 2) sorted by (expert, row); offsets [257].
struct Sorted {
    std::vector<int> off, tok;
    std::vector<std::pair<unsigned, unsigned>> who;  // (row, local expert) per sorted row
};
static Sorted sort_routes(const std::vector<unsigned>& ids) {
    Sorted s;
    std::vector<std::pair<unsigned, unsigned>> v;  // (local expert, row)
    for (size_t q = 0; q < ids.size(); q++)
        if (ids[q] % 2 == 0) v.push_back({ids[q] / 2, (unsigned)(q / TOPK)});
    std::sort(v.begin(), v.end());
    s.off.assign(NE / 2 + 1, 0);
    for (auto& p : v) s.off[p.first + 1]++;
    for (unsigned e = 0; e < NE / 2; e++) s.off[e + 1] += s.off[e];
    for (auto& p : v) { s.tok.push_back((int)p.second); s.who.push_back({p.second, p.first}); }
    if (s.tok.empty()) s.tok.push_back(0);
    return s;
}

struct TcBufs { void *A, *A8, *act8, *C, *sact8, *sC, *off, *tok; };
static TcBufs tc_bufs() {
    TcBufs b;
    b.A = dzero((size_t)MAXR * H * 2); b.A8 = dzero((size_t)MAXR * H);
    b.act8 = dzero((size_t)MAXR * TOPK * I); b.C = dzero((size_t)MAXR * TOPK * H * 2);
    b.sact8 = dzero((size_t)MAXR * I); b.sC = dzero((size_t)MAXR * H * 2);
    b.off = dzero((NE / 2 + 1) * 4); b.tok = dzero((size_t)MAXR * TOPK * 4);
    return b;
}

struct Tc {
    CUfunction a8, gu, dn, sgu, sdn;
    explicit Tc(bool w2) {
        const char* M = "moe_prefill_q38";
        a8 = load(M, "moe_q38_a_to_e4m3");
        gu = load(M, w2 ? "moe_q38w_gate_up_silu" : "moe_q38_gate_up_silu");
        dn = load(M, w2 ? "moe_q38w_down" : "moe_q38_down");
        sgu = load(M, w2 ? "moe_q38w_dense_gate_up_silu" : "moe_q38_dense_gate_up_silu");
        sdn = load(M, w2 ? "moe_q38w_dense_down" : "moe_q38_dense_down");
    }
    void a_to_e4m3(TcBufs& b, unsigned rows) const {
        unsigned n = rows * H;
        launch(a8, dim3((n / 4 + 255) / 256), dim3(256), {&b.A, &b.A8, &n});
    }
    void routed(TPool& p, TcBufs& b) const {
        unsigned ne = NE / 2, n_i = I, k_h = H, n_h = H, k_i = I;
        launch(gu, dim3(I / 64, 1, NE / 2), dim3(256),
               {&b.A8, &p.gp, &p.gs, &p.g2, &p.up, &p.us, &p.u2, &b.act8, &b.off, &b.tok, &ne, &n_i, &k_h});
        launch(dn, dim3(H / 128, 1, NE / 2), dim3(256),
               {&b.act8, &p.dp, &p.ds, &p.d2, &b.C, &b.off, &ne, &n_h, &k_i});
    }
    void shared(TPool& p, TcBufs& b, unsigned rows) const {
        unsigned m = rows, n_i = I, k_h = H, n_h = H, k_i = I;
        launch(sgu, dim3(I / 64, (rows + 127) / 128), dim3(256),
               {&b.A8, &p.sg.packed, &p.sg.scale, &p.sg.s2, &p.su.packed, &p.su.scale, &p.su.s2,
                &b.sact8, &m, &n_i, &k_h});
        launch(sdn, dim3(H / 128, (rows + 127) / 128), dim3(256),
               {&b.sact8, &p.sd.packed, &p.sd.scale, &p.sd.s2, &b.sC, &m, &n_h, &k_i});
    }
};

static void upload(TcBufs& b, const Sorted& s) {
    CK(cudaMemcpy(b.off, s.off.data(), s.off.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.tok, s.tok.data(), s.tok.size() * 4, cudaMemcpyHostToDevice));
}

static void tc_time(bool w2) {
    TPool p(256);
    Tc tc(w2);
    TcBufs base = tc_bufs();
    CK(cudaMemcpy(base.A, rand_bf16((size_t)MAXR * H, 1.0f).data(), (size_t)MAXR * H * 2, cudaMemcpyHostToDevice));
    Timer tm;
    const double gu_e = 2.0 * (I * H / 2 + I * H / 16), sd_e = H * I / 2 + H * I / 16;
    struct Cfg { const char* name; unsigned rows; double uniq; };
    const Cfg cfgs[] = {{"C1 decode", 1, 0}, {"C1 verify", 4, 0}, {"C4 verify", 16, 40},
                        {"C8 u50", 32, 50}, {"C8 u70", 32, 70}, {"C8 u90", 32, 90}, {"C8 indep", 32, 0}};
    printf("\nq38 chain (%s) at decode rows: us; GB/s of unique local + shared bytes\n", w2 ? "W2" : "q38");
    printf("%-10s %6s %8s %16s %16s %8s %8s\n", "routing", "a2e4m3", "", "gate_up (GB/s)", "down (GB/s)",
           "shared", "total");
    for (const Cfg& c : cfgs) {
        const double reuse = c.uniq > 0 ? solve_reuse(c.rows, c.uniq) : 0.0;
        std::mt19937 rng(11);
        const int n = 40;
        std::vector<TcBufs> bs;
        double uniq = 0;
        for (int i = 0; i < n; i++) {
            auto r = c8_routes(c.rows, reuse, rng);
            uniq += local_unique(r);
            TcBufs b = base;
            b.off = dzero((NE / 2 + 1) * 4);
            b.tok = dzero((size_t)MAXR * TOPK * 4);
            upload(b, sort_routes(r));
            bs.push_back(b);
        }
        uniq /= n;
        const double t_a = tm.run(n, [&](int i) { tc.a_to_e4m3(bs[i], c.rows); });
        const double t_r = tm.run(n, [&](int i) { tc.routed(p, bs[i]); });
        const double t_s = tm.run(n, [&](int i) { tc.shared(p, bs[i], c.rows); });
        // gate/up and down alone (routed + shared each).
        const double t_gu = tm.run(n, [&](int i) {
            unsigned ne = NE / 2, n_i = I, k_h = H, m = c.rows;
            launch(tc.gu, dim3(I / 64, 1, NE / 2), dim3(256),
                   {&bs[i].A8, &p.gp, &p.gs, &p.g2, &p.up, &p.us, &p.u2, &bs[i].act8, &bs[i].off, &bs[i].tok,
                    &ne, &n_i, &k_h});
            launch(tc.sgu, dim3(I / 64, 1), dim3(256),
                   {&bs[i].A8, &p.sg.packed, &p.sg.scale, &p.sg.s2, &p.su.packed, &p.su.scale, &p.su.s2,
                    &bs[i].sact8, &m, &n_i, &k_h});
        });
        const double t_all = tm.run(n, [&](int i) {
            tc.a_to_e4m3(bs[i], c.rows); tc.routed(p, bs[i]); tc.shared(p, bs[i], c.rows);
        });
        const double t_dn = t_r + t_s - t_gu;
        printf("%-10s %6.1f %8s %8.1f (%5.0f) %8.1f (%5.0f) %8.1f %8.1f   rows %u, %.1f unique\n", c.name, t_a,
               "", t_gu, (uniq + 1) * gu_e / t_gu / 1e3, t_dn, (uniq + 1) * sd_e / t_dn / 1e3, t_s, t_all,
               c.rows, uniq);
        for (auto& b : bs) { CK(cudaFree(b.off)); CK(cudaFree(b.tok)); }
    }
}

// Row invariance: rows alone vs in waves, byte for byte (routed act8 and C
// per (row, expert); shared act8 and C per row).
static int tc_check(bool w2) {
    TPool p(24);
    Tc tc(w2);
    TcBufs wv = tc_bufs(), so = tc_bufs();
    std::mt19937 rng(9);
    int bad = 0, cases = 0;
    for (unsigned rows : {2u, 3u, 4u, 8u, 16u, 32u, 33u, 64u}) {
        for (double reuse : {0.0, 0.5, 0.9}) {
            auto a = rand_bf16((size_t)rows * H, 1.0f);
            auto ids = c8_routes(rows, reuse, rng);
            CK(cudaMemcpy(wv.A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
            Sorted sw = sort_routes(ids);
            upload(wv, sw);
            tc.a_to_e4m3(wv, rows); tc.routed(p, wv); tc.shared(p, wv, rows);
            auto w_act = dget(wv.act8, sw.who.size() * I), w_c = dget(wv.C, sw.who.size() * H * 2);
            auto w_sact = dget(wv.sact8, (size_t)rows * I), w_sc = dget(wv.sC, (size_t)rows * H * 2);
            for (unsigned r = 0; r < rows; r++) {
                std::vector<unsigned> one(ids.begin() + r * TOPK, ids.begin() + (r + 1) * TOPK);
                CK(cudaMemcpy(so.A, a.data() + (size_t)r * H, H * 2, cudaMemcpyHostToDevice));
                Sorted s1 = sort_routes(one);
                upload(so, s1);
                tc.a_to_e4m3(so, 1); tc.routed(p, so); tc.shared(p, so, 1);
                auto o_act = dget(so.act8, s1.who.size() * I), o_c = dget(so.C, s1.who.size() * H * 2);
                auto o_sact = dget(so.sact8, I), o_sc = dget(so.sC, H * 2);
                for (size_t j = 0; j < s1.who.size(); j++) {
                    size_t k = 0;
                    while (!(sw.who[k].first == r && sw.who[k].second == s1.who[j].second)) k++;
                    bad += memcmp(&o_act[j * I], &w_act[k * I], I) != 0;
                    bad += memcmp(&o_c[j * H * 2], &w_c[k * H * 2], H * 2) != 0;
                    cases += 2;
                }
                bad += memcmp(o_sact.data(), &w_sact[(size_t)r * I], I) != 0;
                bad += memcmp(o_sc.data(), &w_sc[(size_t)r * H * 2], H * 2) != 0;
                cases += 2;
            }
        }
    }
    printf("%s row invariance (%s): %d of %d per-row outputs differ between 1-row and 2..64-row "
           "launches (reuse 0 / 0.5 / 0.9)\n", bad ? "FAIL" : "PASS", w2 ? "W2" : "q38", bad, cases);
    return bad != 0;
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "tc-time";
    CU(cuInit(0));
    CK(cudaFree(0));
    if (mode == "tc-check") return tc_check(false) | tc_check(true);
    tc_time(false);
    tc_time(true);
    return 0;
}
