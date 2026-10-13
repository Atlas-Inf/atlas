// SPDX-License-Identifier: AGPL-3.0-only
// The device paths and the driver of scripts/dev/qwen4exp_moe_fidelity.cu.
#pragma once

struct Tables { u64 *gp, *gs, *upk, *us, *dp, *ds; float *g2, *u2, *d2; };

// The used experts of a layer on the device: the served [N, K/2] layout and
// the prefill [K/2, N] copy (byte transposes), as 512-entry tables.
struct LayerDev {
    Tables o, t;
    std::vector<void*> bufs;
    static unsigned char* up(std::vector<void*>& b, const std::vector<unsigned char>& v) {
        unsigned char* p = dput(v); b.push_back(p); return p;
    }
    static std::vector<unsigned char> tr(const std::vector<unsigned char>& v, unsigned rows, unsigned cols) {
        std::vector<unsigned char> o(v.size());
        for (unsigned r = 0; r < rows; r++)
            for (unsigned c = 0; c < cols; c++) o[(size_t)c * rows + r] = v[(size_t)r * cols + c];
        return o;
    }
    LayerDev(const std::map<unsigned, Expert>& ex) {
        std::vector<u64> P[2][6];
        std::vector<float> S[3];
        for (auto& v : P) for (auto& w : v) w.assign(NE, 0);
        for (auto& v : S) v.assign(NE, 0.f);
        for (auto& [e, x] : ex) {
            const HostProj* pr[3] = {&x.g, &x.u, &x.d};
            for (int j = 0; j < 3; j++) {
                const HostProj& h = *pr[j];
                P[0][2 * j][e] = (u64)up(bufs, h.p);
                P[0][2 * j + 1][e] = (u64)up(bufs, h.s);
                P[1][2 * j][e] = (u64)up(bufs, tr(h.p, h.n, h.k / 2));
                P[1][2 * j + 1][e] = (u64)up(bufs, tr(h.s, h.n, h.k / 16));
                S[j][e] = h.s2;
            }
        }
        Tables* T[2] = {&o, &t};
        for (int l = 0; l < 2; l++) {
            T[l]->gp = dput(P[l][0]); T[l]->gs = dput(P[l][1]); T[l]->upk = dput(P[l][2]);
            T[l]->us = dput(P[l][3]); T[l]->dp = dput(P[l][4]); T[l]->ds = dput(P[l][5]);
            T[l]->g2 = dput(S[0]); T[l]->u2 = dput(S[1]); T[l]->d2 = dput(S[2]);
        }
    }
    ~LayerDev() { for (void* p : bufs) cudaFree(p); }
};

// Each path's [rows * 10, 2560] BF16 expert outputs, (row, slot) order.
static std::vector<unsigned short> run_path(char which, const LayerDev& L, const Rec& r) {
    const unsigned R = r.rows, S = R * TOPK;
    void* A = dput(r.x); void* ids = dput(r.ids);
    void *order = dzero(8192), *ws = dzero(1 << 20), *act = dzero((size_t)(S + R) * I * 4);
    void *go = dzero((size_t)S * I * 2), *uo = dzero((size_t)S * I * 2), *down = dzero((size_t)S * H * 2);
    void *shg = dzero((size_t)R * I * 2), *shu = dzero((size_t)R * I * 2), *shd = dzero((size_t)R * H * 2);
    void* null = nullptr;
    float z = 0.f;
    unsigned topk = TOPK, rows = R, n_i = I, k_h = H, n_h = H, k_i = I;
    const Tables& o = L.o;
    std::vector<unsigned short> y((size_t)S * H);
    if (which == 'b') {
        const char* M = "qwen4exp_moe_rows";
        unsigned slots = S;
        launch(load(M, "qwen4exp_moe_rows_plan"), dim3(1), dim3(256), {&ids, &order, &slots});
        launch(load(M, "qwen4exp_moe_rows_gate_up"), dim3(I / 8, S + R, 2), dim3(128),
               {&A, (void*)&o.gp, (void*)&o.gs, (void*)&o.g2, &go, (void*)&o.upk, (void*)&o.us, (void*)&o.u2, &uo,
                &ids, &order, &null, &null, &z, &shg, &null, &null, &z, &shu, &n_i, &k_h, &topk, &rows});
        launch(load(M, "qwen4exp_moe_rows_silu_down"), dim3(H / 64, (R + 7) / 8 + S), dim3(256),
               {&go, &uo, (void*)&o.dp, (void*)&o.ds, (void*)&o.d2, &down, &ids, &order, &shg, &shu, &null, &null,
                &z, &shd, &n_h, &k_i, &topk, &rows}, 64 * (I / 2 + I / 16) + 2 * I * 4);
        auto v = dget(down, y.size() * 2);
        memcpy(y.data(), v.data(), v.size());
    } else if (which == 'c') {
        const char* M = "qwen4exp_moe_c8_tc";
        const unsigned units = S + (R + 15) / 16;
        launch(load(M, "qwen4exp_moe_c8_tc_plan"), dim3(1), dim3(1024), {&ids, &ws, &topk, &rows});
        launch(load(M, "qwen4exp_moe_c8_tc_gate_up"), dim3(I / 8, units), dim3(256),
               {&A, (void*)&o.gp, (void*)&o.gs, (void*)&o.g2, (void*)&o.upk, (void*)&o.us, (void*)&o.u2, &null,
                &null, &z, &null, &null, &z, &ws, &null, &null, &null, &null, &act, &topk, &rows});
        launch(load(M, "qwen4exp_moe_c8_tc_down"), dim3(H / 64, units), dim3(256),
               {&act, (void*)&o.dp, (void*)&o.ds, (void*)&o.d2, &null, &null, &z, &ws, &down, &shd, &topk, &rows});
        auto v = dget(down, y.size() * 2);
        memcpy(y.data(), v.data(), v.size());
    } else {  // 'd': the prefill chain over all 512 experts, rows sorted by expert
        const Tables& t = L.t;
        std::vector<std::pair<unsigned, unsigned>> srt;  // (expert, slot q)
        for (unsigned q = 0; q < S; q++) srt.push_back({r.ids[q], q});
        std::sort(srt.begin(), srt.end());
        std::vector<int> off(NE + 1, 0), tok;
        for (auto& p : srt) { off[p.first + 1]++; tok.push_back((int)(p.second / TOPK)); }
        for (unsigned e = 0; e < NE; e++) off[e + 1] += off[e];
        void *doff = dput(off), *dtok = dput(tok), *a8 = dzero((size_t)R * H), *act8 = dzero((size_t)S * I);
        unsigned n = R * H, ne = NE;
        const char* M = "moe_prefill_q38";
        launch(load(M, "moe_q38_a_to_e4m3"), dim3((n / 4 + 255) / 256), dim3(256), {&A, &a8, &n});
        launch(load(M, "moe_q38w_gate_up_silu"), dim3(I / 64, 1, NE), dim3(256),
               {&a8, (void*)&t.gp, (void*)&t.gs, (void*)&t.g2, (void*)&t.upk, (void*)&t.us, (void*)&t.u2, &act8,
                &doff, &dtok, &ne, &n_i, &k_h});
        launch(load(M, "moe_q38w_down"), dim3(H / 128, 1, NE), dim3(256),
               {&act8, (void*)&t.dp, (void*)&t.ds, (void*)&t.d2, &down, &doff, &ne, &n_h, &k_i});
        auto v = dget(down, y.size() * 2);
        for (unsigned i = 0; i < S; i++) memcpy(&y[(size_t)srt[i].second * H], &v[(size_t)i * H * 2], H * 2);
        for (void* p : {doff, dtok, a8, act8}) cudaFree(p);
    }
    for (void* p : {A, ids, order, ws, act, go, uo, down, shg, shu, shd}) cudaFree(p);
    return y;
}

// Magnitudes the E4M3 conversions meet: count, sum of squares, max |v|, and
// nonzero values below E4M3's normal range (2^-6) or past its max (448).
struct Mag {
    double n = 0, ss = 0, mx = 0, sub = 0, sat = 0;
    void add(double v) {
        const double a = fabs(v);
        n++; ss += v * v; mx = std::max(mx, a);
        sub += a > 0 && a < 0.015625; sat += a > 448;
    }
    void merge(const Mag& o) { n += o.n; ss += o.ss; mx = std::max(mx, o.mx); sub += o.sub; sat += o.sat; }
    double rms() const { return sqrt(ss / std::max(1.0, n)); }
};

// The float64 reference of every (row, slot): with the decode kernels' routed
// clamp (refc) and without (refn). Counts the clamp's bites.
static void reference(const std::map<unsigned, Expert>& ex, const Rec& r, std::vector<double>& refn,
                      std::vector<double>& refc, long& bites, Mag& act) {
    const unsigned S = r.rows * TOPK;
    refn.assign((size_t)S * H, 0); refc.assign((size_t)S * H, 0);
    long b = 0;
    std::vector<Mag> am(S);
#pragma omp parallel for schedule(dynamic) reduction(+ : b)
    for (unsigned q = 0; q < S; q++) {
        const Expert& e = ex.at(r.ids[q]);
        std::vector<double> x(H), g(I), u(I), an(I), ac(I);
        for (unsigned k = 0; k < H; k++) x[k] = bf(r.x[(size_t)(q / TOPK) * H + k]);
        gemv64(e.g, x.data(), g.data());
        gemv64(e.u, x.data(), u.data());
        for (unsigned i = 0; i < I; i++) {
            an[i] = g[i] / (1 + exp(-g[i])) * u[i];
            const double gc = std::min(g[i], 10.0), uc = std::min(std::max(u[i], -10.0), 10.0);
            b += (gc != g[i]) + (uc != u[i]);
            ac[i] = gc / (1 + exp(-gc)) * uc;
            am[q].add(an[i]);
        }
        gemv64(e.d, an.data(), &refn[(size_t)q * H]);
        gemv64(e.d, ac.data(), &refc[(size_t)q * H]);
    }
    bites += b;
    for (auto& m : am) act.merge(m);
}


// ── EMU=1: where prefill's FP8 error comes from, emulated in float64 ──
// E4M3 round-to-nearest-even, saturating at 448 (cvt.rn.satfinite).
static double e4m3r(double v) {
    const double a = fabs(v);
    if (a == 0) return v;
    if (a >= 448) return copysign(448.0, v);
    int e;
    frexp(a, &e);
    const double q = ldexp(1.0, std::max(e - 1, -6) - 3);
    return copysign(std::min(nearbyint(a / q) * q, 448.0), v);
}
// Weights: 0 exact; 1 prefill's e4m3(lut * float(dec * s2)); 2 e4m3 after a
// per-output-row scale (row max -> 448), the scale back in the epilogue.
static void gemv_w(const HostProj& w, int mode, const double* x, double* y) {
    std::vector<double> row(w.k);
    for (unsigned n = 0; n < w.n; n++) {
        double mx = 0;
        for (unsigned k = 0; k < w.k; k++) {
            const unsigned char b = w.p[(size_t)n * (w.k / 2) + k / 2];
            const unsigned char sb = w.s[(size_t)n * (w.k / 16) + k / 16];
            const double l = LUT[k & 1 ? b >> 4 : b & 15];
            row[k] = mode == 1 ? e4m3r((float)(l * (float)(e4m3(sb) * (double)w.s2))) : l * e4m3(sb) * (double)w.s2;
            mx = std::max(mx, fabs(row[k]));
        }
        if (mode == 2 && mx > 0)
            for (auto& v : row) v = e4m3r(v / (mx / 448)) * (mx / 448);
        double acc = 0;
        for (unsigned k = 0; k < w.k; k++) acc += row[k] * x[k];
        y[n] = acc;
    }
}
// Activations: 0 exact; 1 e4m3 as prefill does (no scale); 2 e4m3 after a
// per-row scale (row max -> 448).
static void quant_a(std::vector<double>& v, int mode) {
    if (mode == 0) return;
    double mx = 0;
    for (double a : v) mx = std::max(mx, fabs(a));
    const double s = mode == 2 && mx > 0 ? mx / 448 : 1.0;
    for (auto& a : v) a = e4m3r(a / s) * s;
}
struct Emu { const char* name; int w, x, act; };
static const Emu EMUS[] = {
    {"W e4m3 (prefill's)", 1, 0, 0}, {"x e4m3, no scale", 0, 1, 0}, {"SiLU*up e4m3, no scale", 0, 0, 1},
    {"all three = prefill", 1, 1, 1}, {"x, act per-row scale", 1, 2, 2}, {"+ W per-out-row scale", 2, 2, 2},
};
static void emulate(const std::map<unsigned, Expert>& ex, const Rec& r, const std::vector<double>& refn,
                    std::vector<Err>& err) {
    const unsigned S = r.rows * TOPK, NV = sizeof(EMUS) / sizeof(EMUS[0]);
    std::vector<std::vector<double>> y(NV, std::vector<double>((size_t)S * H));
#pragma omp parallel for schedule(dynamic)
    for (unsigned q = 0; q < S * NV; q++) {
        const unsigned s = q / NV, v = q % NV;
        const Emu& m = EMUS[v];
        const Expert& e = ex.at(r.ids[s]);
        std::vector<double> x(H), g(I), u(I), a(I);
        for (unsigned k = 0; k < H; k++) x[k] = bf(r.x[(size_t)(s / TOPK) * H + k]);
        quant_a(x, m.x);
        gemv_w(e.g, m.w, x.data(), g.data());
        gemv_w(e.u, m.w, x.data(), u.data());
        for (unsigned i = 0; i < I; i++) a[i] = g[i] / (1 + exp(-g[i])) * u[i];
        quant_a(a, m.act);
        gemv_w(e.d, m.w, a.data(), &y[v][(size_t)s * H]);
    }
    for (unsigned v = 0; v < NV; v++)
        for (size_t i = 0; i < (size_t)S * H; i++) err[v].add(y[v][i], refn[i]);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: fidelity <ptx dir> <model dir>\n"); return 2; }
    g_dir = argv[1];
    CU(cuInit(0));
    CK(cudaFree(0));
    Ckpt ck(argv[2]);
    std::vector<Rec> recs;
    const char* dump = getenv("DUMP");
    std::vector<unsigned> layers;
    { std::istringstream ss(getenv("LAYERS") ? getenv("LAYERS") : "1 24 47"); unsigned l; while (ss >> l) layers.push_back(l); }
    const unsigned nrec = getenv("RECS") ? atoi(getenv("RECS")) : 4;
    if (dump) recs = read_dump(dump);
    printf("qwen4_exp routed-MoE fidelity vs float64 (%s inputs); rel L2 and max |err| over every (row, slot) output\n",
           dump ? "dumped decode" : "SYNTHETIC N(0, SIGMA)");
    for (unsigned L : layers) {
        std::vector<Rec> rs;
        for (auto& r : recs) if (r.layer == L && rs.size() < nrec) rs.push_back(r);
        if (!dump) {
            std::mt19937 rng(L);
            const float sigma = getenv("SIGMA") ? atof(getenv("SIGMA")) : 1.0f;
            Rec r{L, 32, c8_routes(32, 0.0, rng), std::vector<float>(32 * TOPK, 0.1f), rand_bf16((size_t)32 * H, sigma)};
            rs.push_back(r);
        }
        if (rs.empty()) { printf("layer %u: no records\n", L); continue; }
        std::set<unsigned> used;
        for (auto& r : rs) used.insert(r.ids.begin(), r.ids.end());
        std::map<unsigned, Expert> ex;
        const std::string base = "model.language_model.layers." + std::to_string(L) + ".mlp.experts.";
        for (unsigned e : used)
            ex[e] = {load_proj(ck, base + std::to_string(e) + ".gate_proj", I, H),
                     load_proj(ck, base + std::to_string(e) + ".up_proj", I, H),
                     load_proj(ck, base + std::to_string(e) + ".down_proj", H, I)};
        LayerDev dev(ex);
        // [path][ref n/c][slot outputs | routed sums]; floor = BF16(ref).
        Err e[5][2][2];
        long bites = 0, rows = 0;
        Mag xm, am;
        std::vector<Err> emu(sizeof(EMUS) / sizeof(EMUS[0]));
        for (auto& r : rs) {
            for (unsigned short v : r.x) xm.add(bf(v));
            std::vector<double> rn, rc;
            reference(ex, r, rn, rc, bites, am);
            if (getenv("EMU")) emulate(ex, r, rn, emu);
            auto yb = run_path('b', dev, r), yc = run_path('c', dev, r), yd = run_path('d', dev, r);
            rows += r.rows;
            for (unsigned row = 0; row < r.rows; row++)
                for (unsigned n = 0; n < H; n++) {
                    double z[5] = {0, 0, 0, 0, 0}, zr[2] = {0, 0};
                    for (unsigned s = 0; s < TOPK; s++) {
                        const size_t i = (size_t)(row * TOPK + s) * H + n;
                        const double w = r.w[row * TOPK + s];
                        const double v[5] = {bf_round(rn[i]), bf(yb[i]), bf(yc[i]), bf(yd[i]), bf_round(rc[i])};
                        for (int p = 0; p < 4; p++) {
                            e[p][0][0].add(v[p], p == 0 ? rn[i] : rn[i]);
                            e[p][1][0].add(p == 0 ? v[4] : v[p], rc[i]);
                            z[p] += w * (p == 0 ? v[0] : v[p]);
                        }
                        z[4] += w * v[4];
                        e[4][0][0].add(bf(yb[i]), bf(yd[i]));  // (b) against (d)
                        e[4][1][0].add(bf(yb[i]), bf(yc[i]));  // (b) against (c)
                        zr[0] += w * rn[i]; zr[1] += w * rc[i];
                    }
                    for (int p = 0; p < 4; p++) {
                        e[p][0][1].add(z[p], zr[0]);
                        e[p][1][1].add(p == 0 ? z[4] : z[p], zr[1]);
                    }
                }
        }
        const double slots = (double)rows * TOPK;
        if (getenv("SUMMARY")) {
            // L | x rms max sub% | act rms max sub% | clamp% | relL2 expert out vs model ref b c d
            // | relL2 routed sum b c d | vs clamp ref b c | max|err| expert out b c d
            printf("%3u | %.3f %7.2f %5.2f | %.4f %7.2f %5.2f | %.4f |", L, xm.rms(), xm.mx, 100 * xm.sub / xm.n,
                   am.rms(), am.mx, 100 * am.sub / am.n, 100.0 * bites / (2 * slots * I));
            for (int p = 1; p < 4; p++) printf(" %.2e", sqrt(e[p][0][0].num / e[p][0][0].den));
            printf(" |");
            for (int p = 1; p < 4; p++) printf(" %.2e", sqrt(e[p][0][1].num / e[p][0][1].den));
            printf(" |");
            for (int p = 1; p < 3; p++) printf(" %.2e", sqrt(e[p][1][0].num / e[p][1][0].den));
            printf(" |");
            for (int p = 1; p < 4; p++) printf(" %.2e", e[p][0][0].mx);
            printf(" | floor %.2e\n", sqrt(e[0][0][0].num / e[0][0][0].den));
            fflush(stdout);
            continue;
        }
        printf("\nlayer %u: %ld rows, %zu experts; routed clamp fires (g > 10 or |u| > 10) on %.4f%% of gate/up values\n",
               L, rows, used.size(), 100.0 * bites / (2 * slots * I));
        printf("  input x: rms %.4f max %.3f, %.2f%% nonzero below 2^-6, %.3f%% past 448;"
               " SiLU*up: rms %.5f max %.3f, %.2f%% nonzero below 2^-6\n",
               xm.rms(), xm.mx, 100 * xm.sub / xm.n, 100 * xm.sat / xm.n, am.rms(), am.mx, 100 * am.sub / am.n);
        printf("  %-22s | vs float64, no clamp (model)  | vs float64, decode clamp      |\n", "");
        printf("  %-22s | expert out    | routed sum    | expert out    | routed sum    |\n", "path");
        printf("  %-22s | relL2   maxabs| relL2   maxabs| relL2   maxabs| relL2   maxabs|\n", "");
        const char* names[4] = {"BF16 floor (ref->bf16)", "(b) decode rows (FP32)", "(c) TC (BF16 MMA)", "(d) prefill FP8 (q38)"};
        for (int p = 0; p < 4; p++) {
            printf("  %-22s |", names[p]);
            e[p][0][0].print(); e[p][0][1].print(); e[p][1][0].print(); e[p][1][1].print();
            printf("\n");
        }
        if (getenv("EMU")) {
            printf("  float64 emulations of prefill's FP8 steps, expert out vs float64 (no clamp):\n");
            for (size_t v = 0; v < emu.size(); v++) { printf("    %-24s", EMUS[v].name); emu[v].print(); printf("\n"); }
        }
        printf("  (b) vs (d) expert out:"); e[4][0][0].print();
        printf("   (b) vs (c) expert out:"); e[4][1][0].print();
        printf("\n");
    }
    return 0;
}
