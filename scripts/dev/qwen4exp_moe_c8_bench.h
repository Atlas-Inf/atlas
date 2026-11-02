// SPDX-License-Identifier: AGPL-3.0-only
// Fixtures for scripts/dev/qwen4exp_moe_c8_bench.cu: a 256-expert NVFP4 pool
// (one EP2 rank's experts), C1..C8 routing, the production rows pair and the
// variants under test, the parity check and the timer.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; \
    cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s_ ? s_ : "?"); exit(1); } } while (0)

typedef unsigned long long u64;
static const unsigned H = 2560, I = 640, TOPK = 10, NE = 512, MAXR = 64;
static std::string g_dir = ".";
static std::mt19937 g_rng(20261006);

static CUfunction load(const char* module, const char* fn) {
    static std::vector<std::pair<std::string, CUmodule>> mods;
    CUmodule m = nullptr;
    for (auto& p : mods) if (p.first == module) m = p.second;
    if (!m) {
        CU(cuModuleLoad(&m, (g_dir + "/" + module + ".ptx").c_str()));
        mods.push_back({module, m});
    }
    CUfunction f;
    CU(cuModuleGetFunction(&f, m, fn));
    return f;
}
static void launch(CUfunction f, dim3 g, dim3 b, std::vector<void*> args, unsigned smem = 0) {
    if (smem > 48 * 1024)
        CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, 0, args.data(), nullptr));
}
template <typename T> static T* dput(const std::vector<T>& h) {
    T* p;
    CK(cudaMalloc(&p, h.size() * sizeof(T) + 256));
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static void* dzero(size_t bytes) {
    void* p;
    CK(cudaMalloc(&p, bytes + 256));
    CK(cudaMemset(p, 0, bytes + 256));
    return p;
}
static std::vector<unsigned char> dget(const void* p, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    return h;
}
static unsigned short tobf(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}
static std::vector<unsigned short> rand_bf16(size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<unsigned short> v(n);
    for (auto& x : v) x = tobf(d(g_rng));
    for (size_t i = 0; i < n; i += 97) v[i] = (i / 97) % 2 ? 0x8000 : 0x0000;
    return v;
}

struct Proj { unsigned char* packed; unsigned char* scale; float s2; };
static Proj make_proj(unsigned n, unsigned k) {
    std::vector<unsigned char> c((size_t)n * k / 2), s((size_t)n * k / 16);
    for (auto& x : c) x = (unsigned char)g_rng();
    for (auto& x : s) { x = (unsigned char)(g_rng() & 0x7F); if (x == 0x7F) x = 0x7E; }
    std::uniform_real_distribution<float> d(0.5f, 2.0f);
    return {dput(c), dput(s), d(g_rng) / 448.f};
}

// One EP2 rank: global expert e is local iff e is even (pool index e / 2,
// cycling a smaller pool in `check`), remote experts are NULL in the tables.
struct Pool {
    std::vector<Proj> gate, up, down;
    Proj sg, su, sd;
    u64 *gp, *gs, *upk, *us, *dp, *ds;
    float *g2, *u2, *d2;
    explicit Pool(unsigned n) {
        for (unsigned e = 0; e < n; e++) {
            gate.push_back(make_proj(I, H)); up.push_back(make_proj(I, H)); down.push_back(make_proj(H, I));
        }
        sg = make_proj(I, H); su = make_proj(I, H); sd = make_proj(H, I);
        std::vector<u64> a(NE), b(NE), c(NE), d(NE), e2(NE), f(NE);
        std::vector<float> x(NE), y(NE), z(NE);
        for (unsigned e = 0; e < NE; e++) {
            const unsigned c0 = (e / 2) % n;
            const bool remote = e % 2 == 1;
            a[e] = remote ? 0 : (u64)gate[c0].packed; b[e] = remote ? 0 : (u64)gate[c0].scale;
            c[e] = remote ? 0 : (u64)up[c0].packed;   d[e] = remote ? 0 : (u64)up[c0].scale;
            e2[e] = remote ? 0 : (u64)down[c0].packed; f[e] = remote ? 0 : (u64)down[c0].scale;
            x[e] = gate[c0].s2; y[e] = up[c0].s2; z[e] = down[c0].s2;
        }
        gp = dput(a); gs = dput(b); upk = dput(c); us = dput(d); dp = dput(e2); ds = dput(f);
        g2 = dput(x); u2 = dput(y); d2 = dput(z);
    }
    double bytes() const { return gate.size() * 3.0 * (I * H / 2 + I * H / 16); }
    std::vector<unsigned> draw_distinct(unsigned k) {
        std::vector<unsigned> v(gate.size());
        for (unsigned i = 0; i < v.size(); i++) v[i] = i;
        std::shuffle(v.begin(), v.end(), g_rng);
        v.resize(k);
        return v;
    }
};

// ── routing ──
// `rows` rows (sequences of 4 verify rows; 1 row = C1). A pick reuses an
// earlier pick of the launch with probability `reuse` (the row's own
// sequence first), else draws uniformly from 512.
static std::vector<unsigned> c8_routes(unsigned rows, double reuse, std::mt19937& rng) {
    std::vector<unsigned> ids;
    std::uniform_int_distribution<unsigned> d(0, NE - 1);
    std::uniform_real_distribution<double> u(0, 1);
    for (unsigned r = 0; r < rows; r++) {
        const unsigned seq0 = r / 4 * 4;
        std::vector<unsigned> row;
        while (row.size() < TOPK) {
            unsigned e = d(rng);
            if (r > 0 && u(rng) < reuse) {
                const unsigned lo = (r > seq0 && u(rng) < 0.6) ? seq0 : 0;
                const unsigned src = lo + rng() % (r - lo);
                e = ids[src * TOPK + rng() % TOPK];
            }
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        ids.insert(ids.end(), row.begin(), row.end());
    }
    return ids;
}
static unsigned local_unique(const std::vector<unsigned>& ids) {
    std::vector<unsigned> l;
    for (unsigned e : ids) if (e % 2 == 0) l.push_back(e);
    std::sort(l.begin(), l.end());
    return (unsigned)(std::unique(l.begin(), l.end()) - l.begin());
}
// `reuse` giving `target` local unique experts on average (bisection).
static double solve_reuse(unsigned rows, double target) {
    double lo = 0, hi = 0.99;
    for (int it = 0; it < 20; it++) {
        const double mid = (lo + hi) / 2;
        std::mt19937 rng(77);
        double s = 0;
        for (int k = 0; k < 64; k++) s += local_unique(c8_routes(rows, mid, rng));
        (s / 64 > target ? lo : hi) = mid;
    }
    return (lo + hi) / 2;
}
// ROUTES=file (ATLAS_QWEN4EXP_MOE_ROUTE_DUMP lines): the `rows`-row launches.
// Global ids are remapped to this pool's convention (local = even) by
// ROUTE_LOCAL: "lo" (default; ids < 256 local, EP rank 0's half), "hi", or
// "even" (ids as they are).
static std::vector<std::vector<unsigned>> trace_routes(const char* path, unsigned rows) {
    const std::string mode = getenv("ROUTE_LOCAL") ? getenv("ROUTE_LOCAL") : "lo";
    std::vector<std::vector<unsigned>> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::vector<unsigned> v;
        unsigned x;
        while (ss >> x) {
            if (mode != "even") {
                const bool local = (x < NE / 2) == (mode == "lo");
                x = (x % (NE / 2)) * 2 + (local ? 0 : 1);
            }
            v.push_back(x);
        }
        if (v.size() == (size_t)rows * TOPK) out.push_back(v);
    }
    return out;
}

// REALBIN=<file> (ATLAS_QWEN4EXP_MOE_ROUTE_DUMP's .bin): per layer, its
// records' ids in order (remapped as trace_routes does). A wave of `rows`
// rows = the layer's first rows/4 records' rows (4-row records), so a
// config's launches are one per layer: the step's 48 MoE launches.
static std::vector<std::vector<unsigned>> bin_waves(const char* path, unsigned rows) {
    const std::string mode = getenv("ROUTE_LOCAL") ? getenv("ROUTE_LOCAL") : "lo";
    std::map<unsigned, std::vector<unsigned>> per_layer;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    unsigned hd[5];
    while (fread(hd, 4, 5, f) == 5) {
        std::vector<unsigned> ids(hd[2] * hd[3]);
        if (fread(ids.data(), 4, ids.size(), f) != ids.size()) break;
        fseek(f, (long)(hd[2] * hd[3] * 4 + (size_t)hd[2] * hd[4] * 2), SEEK_CUR);
        for (auto& x : ids)
            if (mode != "even") {
                const bool local = (x < NE / 2) == (mode == "lo");
                x = (x % (NE / 2)) * 2 + (local ? 0 : 1);
            }
        auto& v = per_layer[hd[1]];
        v.insert(v.end(), ids.begin(), ids.end());
    }
    fclose(f);
    std::vector<std::vector<unsigned>> out;
    for (auto& [l, v] : per_layer)
        if (v.size() >= (size_t)rows * TOPK) out.emplace_back(v.begin(), v.begin() + rows * TOPK);
    return out;
}

// ── one launch's buffers and the kernels under test ──
struct Bufs {
    void *A, *gate, *up, *shg, *shu, *down, *shd, *ids, *order, *ws, *act;
};
static Bufs alloc_bufs() {
    Bufs b;
    const size_t rk = (size_t)MAXR * TOPK;
    b.A = dzero((size_t)MAXR * H * 2);
    b.gate = dzero(rk * I * 2); b.up = dzero(rk * I * 2);
    b.shg = dzero((size_t)MAXR * I * 2); b.shu = dzero((size_t)MAXR * I * 2);
    b.down = dzero(rk * H * 2); b.shd = dzero((size_t)MAXR * H * 2);
    b.ids = dzero(rk * 4); b.order = dzero(rk * 4 + 4096); b.ws = dzero(1 << 20);
    b.act = dzero((rk + MAXR) * I * 4);
    return b;
}

// A variant: plan (may be the production one), gate/up, silu/down.
struct Variant {
    std::string name;
    std::function<void(Pool&, Bufs&, unsigned)> plan, gate_up, silu_down;
};

// NC=1: every kernel's no-clamp entry (ATLAS_QWEN4EXP_MOE_NO_CLAMP).
static bool nc() { return getenv("NC") && atoi(getenv("NC")); }

static Variant production() {
    const char* M = "qwen4exp_moe_rows";
    CUfunction pl = load(M, "qwen4exp_moe_rows_plan"), gu = load(M, "qwen4exp_moe_rows_gate_up"),
               sd = load(M, nc() ? "qwen4exp_moe_rows_silu_down_nc" : "qwen4exp_moe_rows_silu_down");
    Variant v;
    v.name = "production rows";
    v.plan = [=](Pool&, Bufs& b, unsigned rows) {
        unsigned slots = rows * TOPK;
        launch(pl, dim3(1), dim3(256), {&b.ids, &b.order, &slots});
    };
    v.gate_up = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = I, k = H, topk = TOPK, R = rows;
        launch(gu, dim3(I / 8, rows * TOPK + rows, 2), dim3(128),
               {&b.A, &p.gp, &p.gs, &p.g2, &b.gate, &p.upk, &p.us, &p.u2, &b.up, &b.ids, &b.order,
                &p.sg.packed, &p.sg.scale, &p.sg.s2, &b.shg, &p.su.packed, &p.su.scale, &p.su.s2,
                &b.shu, &n, &k, &topk, &R});
    };
    v.silu_down = [=](Pool& p, Bufs& b, unsigned rows) {
        unsigned n = H, k = I, topk = TOPK, R = rows;
        launch(sd, dim3(H / 64, (rows + 7) / 8 + rows * TOPK, 1), dim3(256),
               {&b.gate, &b.up, &p.dp, &p.ds, &p.d2, &b.down, &b.ids, &b.order, &b.shg, &b.shu,
                &p.sd.packed, &p.sd.scale, &p.sd.s2, &b.shd, &n, &k, &topk, &R},
               64 * (I / 2 + I / 16) + 2 * I * 4);
    };
    return v;
}

std::vector<Variant> variants();  // the candidates (qwen4exp_moe_c8_variants.h)

// REPS=n: each timing the fastest of n passes (03 is shared: another
// tenant's kernels only ever add time; default 1).
struct Timer {
    cudaEvent_t e0, e1;
    Timer() { CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); }
    template <typename F> double run(int iters, F&& body) {
        const int reps = getenv("REPS") ? std::max(1, atoi(getenv("REPS"))) : 1;
        std::vector<double> v;
        for (int r = 0; r < reps; r++) v.push_back(once(iters, body));
        return *std::min_element(v.begin(), v.end());
    }
    template <typename F> double once(int iters, F&& body) {
        float ms;
        body(0);  // warm (module load, attributes)
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) body(i);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms * 1000.0 / iters;
    }
};
