// SPDX-License-Identifier: AGPL-3.0-only
// The `time` and `check` drivers of scripts/dev/qwen4exp_moe_c8_bench.cu.
#pragma once
#include "qwen4exp_moe_c8_bench.h"

static int g_fail = 0;
static bool same(const std::string& what, const std::vector<unsigned char>& a,
                 const std::vector<unsigned char>& b) {
    size_t diff = 0, first = 0;
    for (size_t i = 0; i + 1 < a.size(); i += 2)
        if (a[i] != b[i] || a[i + 1] != b[i + 1]) { if (!diff) first = i / 2; diff++; }
    if (diff) {
        printf("  MISMATCH %-56s %zu of %zu values (first %zu)\n", what.c_str(), diff, a.size() / 2, first);
        g_fail++;
    }
    return diff == 0;
}

// Variants whose gate/up intermediates are not materialized say so by name
// ("fused"): only the down outputs are compared.
static int run_check(Pool& pool) {
    Variant ref = production();
    std::vector<Variant> vs;
    for (auto& v : variants())
        if (v.name.rfind("tc", 0) != 0) vs.push_back(v);  // contract (b): run_tc_check
    Bufs rb = alloc_bufs(), gb = alloc_bufs();
    std::vector<int> bad(vs.size(), 0);
    std::mt19937 rng(5);
    const size_t rk = (size_t)MAXR * TOPK;
    for (unsigned rows : {1u, 2u, 3u, 4u, 5u, 8u, 12u, 16u, 24u, 32u, 33u, 40u, 48u, 64u}) {
        for (double reuse : {0.0, 0.5, 0.9, 1.0}) {
            for (float scale : {0.02f, 1.0f, 30.0f}) {
                auto a = rand_bf16((size_t)rows * H, scale);
                auto ids = c8_routes(rows, reuse, rng);
                if (reuse == 1.0)  // every row row 0's picks, shuffled
                    for (unsigned r = 1; r < rows; r++) {
                        std::copy(ids.begin(), ids.begin() + TOPK, ids.begin() + r * TOPK);
                        std::shuffle(ids.begin() + r * TOPK, ids.begin() + (r + 1) * TOPK, rng);
                    }
                for (Bufs* b : {&rb, &gb}) {
                    CK(cudaMemcpy(b->A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(b->ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                }
                ref.plan(pool, rb, rows); ref.gate_up(pool, rb, rows); ref.silu_down(pool, rb, rows);
                const size_t n = (size_t)rows * TOPK;
                auto r_gate = dget(rb.gate, n * I * 2), r_up = dget(rb.up, n * I * 2);
                auto r_shg = dget(rb.shg, rows * I * 2), r_shu = dget(rb.shu, rows * I * 2);
                auto r_down = dget(rb.down, n * H * 2), r_shd = dget(rb.shd, rows * H * 2);
                for (size_t vi = 0; vi < vs.size(); vi++) {
                    Variant& v = vs[vi];
                    for (void* p : {gb.gate, gb.up}) CK(cudaMemset(p, 0x55, rk * I * 2));
                    for (void* p : {gb.shg, gb.shu}) CK(cudaMemset(p, 0x55, (size_t)MAXR * I * 2));
                    CK(cudaMemset(gb.down, 0x55, rk * H * 2));
                    CK(cudaMemset(gb.shd, 0x55, (size_t)MAXR * H * 2));
                    v.plan(pool, gb, rows); v.gate_up(pool, gb, rows); v.silu_down(pool, gb, rows);
                    char tag[128];
                    snprintf(tag, sizeof tag, "%s R=%u reuse=%.1f x%g", v.name.c_str(), rows, reuse, scale);
                    const std::string t(tag);
                    bool ok = true;
                    if (v.name.find("fused") == std::string::npos) {
                        ok &= same(t + " gate", r_gate, dget(gb.gate, n * I * 2));
                        ok &= same(t + " up", r_up, dget(gb.up, n * I * 2));
                        ok &= same(t + " sh gate", r_shg, dget(gb.shg, rows * I * 2));
                        ok &= same(t + " sh up", r_shu, dget(gb.shu, rows * I * 2));
                    }
                    ok &= same(t + " down", r_down, dget(gb.down, n * H * 2));
                    ok &= same(t + " sh down", r_shd, dget(gb.shd, rows * H * 2));
                    bad[vi] += !ok;
                }
            }
        }
    }
    for (size_t vi = 0; vi < vs.size(); vi++)
        printf("  %s  %-36s R = 1..64; reuse 0 / 0.5 / 0.9 / identical; 3 scales; every output byte "
               "vs the production rows pair\n", bad[vi] ? "BAD" : "ok ", vs[vi].name.c_str());
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

// us per launch over `iters` route sets (each its own plan), GB/s of the
// unique local expert bytes + the shared expert.
static void run_time(Pool& pool, const std::string& only) {
    std::vector<Variant> vs;
    vs.push_back(production());
    for (auto& v : variants())
        if (only.empty() || v.name.find(only) != std::string::npos) vs.push_back(v);
    Bufs base = alloc_bufs();
    Timer tm;
    const int iters = 40;
    const double gu_e = 2.0 * (I * H / 2 + I * H / 16), sd_e = H * I / 2 + H * I / 16;
    struct Cfg { const char* name; unsigned rows; double uniq; };
    const char* trace = getenv("ROUTES");
    std::vector<Cfg> cfgs = {{"C1 decode", 1, 0}, {"C1 verify", 4, 0}, {"C4 verify", 16, 40},
                             {"C8 u50", 32, 50}, {"C8 u70", 32, 70}, {"C8 u90", 32, 90},
                             {"C8 indep", 32, 0}};
    if (trace) cfgs = {{"trace", (unsigned)atoi(getenv("ROUTE_ROWS") ? getenv("ROUTE_ROWS") : "32"), 0}};
    const char* realbin = getenv("REALBIN");
    if (realbin)
        cfgs = {{"real 1", 1, 0}, {"real 4", 4, 0}, {"real 16", 16, 0}, {"real 24", 24, 0}, {"real 32", 32, 0},
                {"real 36", 36, 0}};
    printf("\n%-10s %-24s %8s %16s %16s %8s\n", "routing", "variant", "plan", "gate_up (GB/s)",
           "silu_down (GB/s)", "layer");
    const char* only_cfg = getenv("CFG");
    for (const Cfg& c : cfgs) {
        if (only_cfg && std::string(c.name) != only_cfg) continue;
        std::vector<std::vector<unsigned>> routes;
        if (realbin) {
            routes = bin_waves(realbin, c.rows);
        } else if (trace) {
            routes = trace_routes(trace, c.rows);
            if (routes.empty()) { printf("no %u-row launches in %s\n", c.rows, trace); return; }
        } else {
            const double reuse = c.uniq > 0 ? solve_reuse(c.rows, c.uniq) : 0.0;
            std::mt19937 rng(11);
            for (int i = 0; i < iters; i++) routes.push_back(c8_routes(c.rows, reuse, rng));
        }
        const int n = (int)routes.size();
        double uniq = 0;
        std::vector<Bufs> bs;
        for (auto& r : routes) {
            uniq += local_unique(r);
            Bufs b = base;
            b.ids = dput(r);
            b.order = dzero(MAXR * TOPK * 4 + 4096);
            b.ws = dzero(1 << 20);
            bs.push_back(b);
        }
        uniq /= n;
        const double gu_b = (uniq + 1) * gu_e, sd_b = (uniq + 1) * sd_e;
        printf("%-10s rows %u, %.1f unique local experts (+ shared), %.1f + %.1f MB\n", c.name, c.rows,
               uniq, gu_b / 1e6, sd_b / 1e6);
        // LAYER_PASSES=p: only plan + gate/up + down, the variants interleaved
        // p times (03 is shared: min and median of the passes).
        if (const char* lp = getenv("LAYER_PASSES")) {
            std::vector<std::vector<double>> t(vs.size());
            for (int pass = 0; pass < atoi(lp); pass++)
                for (size_t k = 0; k < vs.size(); k++) {
                    Variant& v = vs[k];
                    const auto layer = [&](int i) {
                        v.plan(pool, bs[i], c.rows); v.gate_up(pool, bs[i], c.rows); v.silu_down(pool, bs[i], c.rows);
                    };
                    tm.once(n, layer);  // the previous variant's wake (order bias at 1 row ~10%)
                    t[k].push_back(tm.once(n, layer));
                }
            for (size_t k = 0; k < vs.size(); k++) {
                std::sort(t[k].begin(), t[k].end());
                printf("%-10s %-24s layer min %7.1f med %7.1f us  (%5.0f GB/s of unique bytes at min)\n", "",
                       vs[k].name.c_str(), t[k][0], t[k][t[k].size() / 2], (gu_b + sd_b) / t[k][0] / 1e3);
            }
            for (auto& b : bs) { CK(cudaFree(b.ids)); CK(cudaFree(b.order)); CK(cudaFree(b.ws)); }
            continue;
        }
        for (Variant& v : vs) {
            for (int i = 0; i < n; i++) v.plan(pool, bs[i], c.rows);
            const double pl = tm.run(n, [&](int i) { v.plan(pool, bs[i], c.rows); });
            const double gu = tm.run(n, [&](int i) { v.gate_up(pool, bs[i], c.rows); });
            const double sd = tm.run(n, [&](int i) { v.silu_down(pool, bs[i], c.rows); });
            const double all = tm.run(n, [&](int i) {
                v.plan(pool, bs[i], c.rows); v.gate_up(pool, bs[i], c.rows); v.silu_down(pool, bs[i], c.rows);
            });
            printf("%-10s %-24s %8.1f %8.1f (%5.0f) %8.1f (%5.0f) %8.1f\n", "", v.name.c_str(), pl, gu,
                   gu_b / gu / 1e3, sd, sd_b / sd / 1e3, all);
        }
        for (auto& b : bs) { CK(cudaFree(b.ids)); CK(cudaFree(b.order)); CK(cudaFree(b.ws)); }
    }
}

// Contract (b) variants ("tc ..."): every row's down / shared-down / act
// bytes alone (1 row) vs inside waves of 2..64 rows, and the largest
// deviation from the production rows pair (relative to the row's max |out|).
static int run_tc_check(Pool& pool) {
    Variant ref = production();
    Bufs wb = alloc_bufs(), sb = alloc_bufs(), rb = alloc_bufs();
    std::mt19937 rng(17);
    for (auto& v : variants()) {
        if (v.name.rfind("tc", 0) != 0) continue;
        int bad = 0, cases = 0;
        double worst = 0;
        for (unsigned rows : {2u, 3u, 4u, 8u, 16u, 17u, 32u, 33u, 64u}) {
            for (double reuse : {0.0, 0.5, 0.9, 1.0}) {
                auto a = rand_bf16((size_t)rows * H, 1.0f);
                auto ids = c8_routes(rows, reuse, rng);
                if (reuse == 1.0)
                    for (unsigned r = 1; r < rows; r++)
                        std::copy(ids.begin(), ids.begin() + TOPK, ids.begin() + r * TOPK);
                for (Bufs* b : {&wb, &rb}) {
                    CK(cudaMemcpy(b->A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(b->ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                }
                v.plan(pool, wb, rows); v.gate_up(pool, wb, rows); v.silu_down(pool, wb, rows);
                ref.plan(pool, rb, rows); ref.gate_up(pool, rb, rows); ref.silu_down(pool, rb, rows);
                auto wd = dget(wb.down, (size_t)rows * TOPK * H * 2), ws = dget(wb.shd, (size_t)rows * H * 2);
                auto rd = dget(rb.down, (size_t)rows * TOPK * H * 2), rs = dget(rb.shd, (size_t)rows * H * 2);
                auto bf = [](const std::vector<unsigned char>& v, size_t i) {
                    unsigned u = (unsigned)(v[2 * i] | (v[2 * i + 1] << 8)) << 16;
                    float f; memcpy(&f, &u, 4); return f;
                };
                for (size_t row = 0; row < (size_t)rows * TOPK + rows; row++) {
                    const bool sh = row >= (size_t)rows * TOPK;
                    const auto& W = sh ? ws : wd; const auto& R = sh ? rs : rd;
                    const size_t base = (sh ? row - (size_t)rows * TOPK : row) * H;
                    double mx = 0, dv = 0;
                    for (unsigned n = 0; n < H; n++) {
                        mx = std::max(mx, (double)fabsf(bf(R, base + n)));
                        dv = std::max(dv, (double)fabsf(bf(W, base + n) - bf(R, base + n)));
                    }
                    if (mx > 0) worst = std::max(worst, dv / mx);
                }
                for (unsigned r = 0; r < rows; r++) {
                    std::vector<unsigned> one(ids.begin() + r * TOPK, ids.begin() + (r + 1) * TOPK);
                    CK(cudaMemcpy(sb.A, a.data() + (size_t)r * H, H * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(sb.ids, one.data(), TOPK * 4, cudaMemcpyHostToDevice));
                    v.plan(pool, sb, 1); v.gate_up(pool, sb, 1); v.silu_down(pool, sb, 1);
                    auto od = dget(sb.down, (size_t)TOPK * H * 2), os = dget(sb.shd, H * 2);
                    bad += memcmp(od.data(), &wd[(size_t)r * TOPK * H * 2], od.size()) != 0;
                    bad += memcmp(os.data(), &ws[(size_t)r * H * 2], os.size()) != 0;
                    cases += 2;
                }
            }
        }
        printf("  %s  %-24s row invariance: %d of %d row outputs (routed top-10 block, shared) differ "
               "between 1-row and 2..64-row launches; max |out - rows pair| / max|out| = %.3g\n",
               bad ? "BAD" : "ok ", v.name.c_str(), bad, cases, worst);
        g_fail += bad;
    }
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

// tc-ident: every TC variant's routed-down and shared-down bytes against
// "tc v1" (the kernels before the L2 look-ahead) at every row count 1..64,
// 4 overlap regimes, 3 activation scales.
static int run_tc_ident(Pool& pool) {
    std::vector<Variant> vs;
    for (auto& v : variants()) if (v.name.rfind("tc", 0) == 0) vs.push_back(v);
    const Variant* ref = nullptr;
    for (auto& v : vs) if (v.name.rfind("tc v1", 0) == 0) ref = &v;
    if (!ref) { printf("no tc v1\n"); return 1; }
    Bufs rb = alloc_bufs(), gb = alloc_bufs();
    std::mt19937 rng(23);
    std::vector<int> bad(vs.size(), 0);
    int cases = 0;
    for (unsigned rows = 1; rows <= MAXR; rows++)
        for (double reuse : {0.0, 0.5, 0.9, 1.0})
            for (float scale : {0.02f, 1.0f, 30.0f}) {
                auto a = rand_bf16((size_t)rows * H, scale);
                auto ids = c8_routes(rows, reuse, rng);
                if (reuse == 1.0)
                    for (unsigned r = 1; r < rows; r++) std::copy(ids.begin(), ids.begin() + TOPK, ids.begin() + r * TOPK);
                for (Bufs* b : {&rb, &gb}) {
                    CK(cudaMemcpy(b->A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(b->ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                }
                ref->plan(pool, rb, rows); ref->gate_up(pool, rb, rows); ref->silu_down(pool, rb, rows);
                auto rd = dget(rb.down, (size_t)rows * TOPK * H * 2), rs = dget(rb.shd, (size_t)rows * H * 2);
                cases++;
                for (size_t i = 0; i < vs.size(); i++) {
                    CK(cudaMemset(gb.down, 0x55, (size_t)MAXR * TOPK * H * 2));
                    CK(cudaMemset(gb.shd, 0x55, (size_t)MAXR * H * 2));
                    vs[i].plan(pool, gb, rows); vs[i].gate_up(pool, gb, rows); vs[i].silu_down(pool, gb, rows);
                    bad[i] += dget(gb.down, rd.size()) != rd || dget(gb.shd, rs.size()) != rs;
                }
            }
    for (size_t i = 0; i < vs.size(); i++)
        printf("  %s  %-28s %d of %d launches differ from tc v1 (rows 1..64, 4 routings, 3 scales%s)\n",
               bad[i] ? "BAD" : "ok ", vs[i].name.c_str(), bad[i], cases, nc() ? ", no clamp" : "");
    int b = 0;
    for (int x : bad) b += x;
    printf("%s\n", b ? "FAIL" : "PASS");
    return b != 0;
}
