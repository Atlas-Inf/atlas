// SPDX-License-Identifier: AGPL-3.0-only
// Bit parity and cost of the wide NVFP4 row kernel `qwen4exp_w4_rows_wide`
// (ATLAS_QWEN4EXP_W4_ROWS_WIDE, kernels/gb10/qwen3.8-flash-next/nvfp4/
// qwen4exp_w4_wide.cu) against `w4a16_gemv` per row, and the 16-row
// `w4a16_gemv_batch16` chunking it replaces in the batched verify (MoE router
// 512x2560, attention K/V 256x2560 and o_proj 2560x3072 a rank at TP2).
//
// check: every M = 1..32 over random E2M1 codes (all 16 nibbles), E4M3 scales
// including 0 and subnormals, Gaussian activations with +-0, BF16 subnormals
// and +-1e30 planted; the three verify shapes, a ragged N and an unbalanced K.
// time:  GPU us per call, weights streamed from DRAM (a ring of copies past the
// 24 MiB L2), the production chunking (batch16 past 8 rows, batch8/4 below)
// against one launch.
//
// Build/run (repo root, GB10): scripts/dev/qwen4exp_w4_wide_bench.sh check|time|sweep
#include "qwen4exp_ptx_harness.h"

#include <random>

typedef unsigned short bf;
typedef unsigned char u8;

static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 777);
static int g_sms = 48;

static PtxModule& mod(const char* stem) {
    static std::vector<std::pair<std::string, PtxModule*>> mods;
    for (auto& m : mods) if (m.first == stem) return *m.second;
    PtxModule* m = new PtxModule;
    m->load(g_dir + "/" + stem + ".ptx");
    mods.push_back({stem, m});
    return *m;
}

static std::vector<bf> rbf(size_t n, bool planted) {
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<bf> v(n);
    for (auto& x : v) {
        const unsigned k = g_rng() % 211;
        const bf sign = (bf)(g_rng() & 0x8000);
        if (planted && k == 0) x = sign;
        else if (planted && k == 1) x = (bf)(1 + g_rng() % 0x7F) | sign;
        else if (planted && k == 2) x = f2bf(1e30f) | sign;
        else x = f2bf(d(g_rng));
    }
    return v;
}
struct Fp4 { Buf<u8> packed, scale; float s2; };
static Fp4 make_fp4(unsigned N, unsigned K) {
    Fp4 w;
    w.packed.alloc((size_t)N * K / 2); w.scale.alloc((size_t)N * K / 16);
    std::vector<u8> p(w.packed.n), s(w.scale.n);
    for (auto& x : p) x = (u8)g_rng();
    for (auto& x : s) {
        const unsigned k = g_rng() % 64;
        x = k == 0 ? 0 : k == 1 ? (u8)(g_rng() & 7) : (u8)(0x20 + g_rng() % 0x30);
    }
    w.packed.put(p); w.scale.put(s);
    w.s2 = 1.0f / 448.f;
    return w;
}

// A wide-kernel variant: entry, outputs a tile, threads, rows a CTA.
struct Wide { const char* fn; unsigned npb, threads, mt; };
static const Wide PROD = {"qwen4exp_w4_rows_wide", 32, 256, 8};
static const Wide SWEEP[] = {{"qw4w_n2_w16_m8", 32, 512, 8}, {"qw4w_n4_w16_m8", 64, 512, 8},{"qw4w_n4_w12_m8", 48, 384, 8}, {"qw4w_n4_w4_m8", 16, 128, 8},
                             {"qw4w_n4_w4_m4", 16, 128, 4}, {"qw4w_n2_w8_m8", 16, 256, 8}};

// Persistent CTAs a row group: the `per_sm` x SMs resident CTAs shared out
// over the row groups (rounded down, so all are resident), at most one a tile
// (0: one a tile). Production (`qwen4exp_w4_wide.rs`): 1.
static unsigned wide_grid_x(const Wide& v, unsigned M, unsigned N, unsigned per_sm) {
    const unsigned tiles = (N + v.npb - 1) / v.npb, groups = (M + v.mt - 1) / v.mt;
    if (per_sm == 0) return tiles;
    return std::max(1u, std::min(tiles, per_sm * g_sms / groups));
}
static void launch_wide(const Wide& v, const bf* A, const Fp4& w, bf* C, unsigned M, unsigned N,
                        unsigned K, unsigned per_sm = 1) {
    const unsigned smem = v.mt * ((K / 16 + 31) / 32) * 2048;
    CUfunction f = mod("qwen4exp_w4_wide").fn(v.fn, smem);
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C).add(M).add(N).add(K);
    launch(f, dim3(wide_grid_x(v, M, N, per_sm), (M + v.mt - 1) / v.mt), dim3(v.threads), smem, a);
}
static void launch_old(const char* fn, unsigned m, const bf* A, const Fp4& w, bf* C, unsigned N,
                       unsigned K) {
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C);
    if (strcmp(fn, "w4a16_gemv") != 0) a.add(m);
    a.add(N).add(K);
    launch(mod("w4a16_gemv").fn(fn), dim3((N + 3) / 4), dim3(256), 0, a);
}
// Production chunking before this kernel (`Qwen4ExpWideRows::w4a16_rows`):
// batch16 past 8 rows, else the exact-M scalar tier (batch4 at 4 or fewer).
static void launch_chunked(const bf* A, const Fp4& w, bf* C, unsigned M, unsigned N, unsigned K) {
    static const char* T[] = {"", "w4a16_gemv_batch4", "w4a16_gemv_batch4", "w4a16_gemv_batch4",
                              "w4a16_gemv_batch4", "w4a16_gemv_batch5", "w4a16_gemv_batch6",
                              "w4a16_gemv_batch7", "w4a16_gemv_batch8"};
    for (unsigned first = 0; first < M;) {
        const unsigned left = M - first, m = left > 8 ? std::min(left, 16u) : left;
        launch_old(left > 8 ? "w4a16_gemv_batch16" : T[m], m, A + (size_t)first * K, w,
                   C + (size_t)first * N, N, K);
        first += m;
    }
}

struct Shape { const char* n; unsigned N, K; };
static const unsigned MAXM = 40;  // C8 x K=4 plus the 36-row step

static void check() {
    Shape shapes[] = {{"router", 512, 2560}, {"K/V", 256, 2560}, {"o_proj", 2560, 3072},
                      {"ragged N", 4101, 2560}, {"unbalanced K", 300, 1040}};
    std::vector<Wide> vars = {PROD};
    if (getenv("QW4W_SWEEP_CHECK"))
        vars.insert(vars.end(), SWEEP, SWEEP + sizeof SWEEP / sizeof *SWEEP);
    for (auto& s : shapes) {
        Fp4 w = make_fp4(s.N, s.K);
        for (bool planted : {false, true}) {
            Buf<bf> A, R, G;
            A.alloc((size_t)MAXM * s.K); R.alloc((size_t)MAXM * s.N); G.alloc((size_t)MAXM * s.N);
            A.put(rbf(A.n, planted));
            for (unsigned r = 0; r < MAXM; r++)
                launch_old("w4a16_gemv", 1, A.p + (size_t)r * s.K, w, R.p + (size_t)r * s.N, s.N, s.K);
            const auto ref = R.get();
            for (auto& v : vars) {
                const int bad0 = g_fail;
                for (unsigned m = 1; m <= MAXM; m++) {
                    for (unsigned per_sm : {0u, 1u}) {
                        const unsigned r0 = MAXM - m;
                        G.fill(0x55);
                        launch_wide(v, A.p + (size_t)r0 * s.K, w, G.p, m, s.N, s.K, per_sm);
                        CK(cudaDeviceSynchronize());
                        const auto got = G.get();
                        size_t d = 0;
                        for (size_t i = 0; i < (size_t)m * s.N; i++) d += ref[(size_t)r0 * s.N + i] != got[i];
                        for (size_t i = (size_t)m * s.N; i < G.n; i++) d += got[i] != 0x5555;  // no stray writes
                        if (d) { printf("  MISMATCH %s %s %s M=%u grid=%u: %zu\n", v.fn, s.n, planted ? "planted" : "clean", m, per_sm, d); g_fail++; }
                    }
                }
                printf("  %s  %-22s M=1..40  %-12s %5ux%-5u %-8s vs w4a16_gemv per row\n",
                       g_fail != bad0 ? "BAD" : "ok ", v.fn, s.n, s.N, s.K, planted ? "planted" : "clean");
            }
            A.free_(); R.free_(); G.free_();
        }
        w.packed.free_(); w.scale.free_();
    }
}

static void time_all(bool sweep) {
    Shape shapes[] = {{"router 512x2560", 512, 2560}, {"K/V 256x2560", 256, 2560}, {"o_proj 2560x3072", 2560, 3072}};
    const unsigned Ms[] = {1, 2, 4, 6, 8, 9, 16, 24, 28, 32, 36};
    for (auto& s : shapes) {
        const double wb = (double)s.N * s.K / 2 + (double)s.N * s.K / 16;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Fp4> W;
        for (int c = 0; c < copies; c++) W.push_back(make_fp4(s.N, s.K));
        Buf<bf> A, C;
        A.alloc((size_t)MAXM * s.K); C.alloc((size_t)MAXM * s.N);
        A.put(rbf(A.n, false));
        printf("\n%s (%.2f MB): us per call", s.n, wb / 1e6);
        for (unsigned m : Ms) printf("  M=%-4u", m);
        printf("\n");
        auto row = [&](const char* name, auto body) {
            printf("  %-34s", name);
            for (unsigned m : Ms) {
                int it = 0;
                const float ms = time_ms([&] { body(m, it++ % copies); }, 20, 7);
                printf("  %6.1f", ms * 1e3);
            }
            printf("\n");
        };
        row("batch16 chunks (before)", [&](unsigned m, int c) { launch_chunked(A.p, W[c], C.p, m, s.N, s.K); });
        row("qwen4exp_w4_rows_wide", [&](unsigned m, int c) { launch_wide(PROD, A.p, W[c], C.p, m, s.N, s.K); });
        if (sweep) {
            for (auto& v : SWEEP)
                for (unsigned per_sm : {0u, 1u, 2u}) {
                    char name[64];
                    snprintf(name, sizeof name, "%s ctas/sm=%u", v.fn, per_sm);
                    row(name, [&](unsigned m, int c) { launch_wide(v, A.p, W[c], C.p, m, s.N, s.K, per_sm); });
                }
        }
        for (auto& w : W) { w.packed.free_(); w.scale.free_(); }
        A.free_(); C.free_();
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    init_driver();
    CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
    if (mode == "check") {
        check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    if (mode == "prof") {  // one launch of each, M = 32, for ncu
        Shape shapes[] = {{"router", 512, 2560}, {"o_proj", 2560, 3072}};
        for (auto& s : shapes) {
            Fp4 w = make_fp4(s.N, s.K);
            Buf<bf> A, C;
            A.alloc((size_t)32 * s.K); C.alloc((size_t)32 * s.N);
            A.put(rbf(A.n, false));
            launch_chunked(A.p, w, C.p, 32, s.N, s.K);
            const char* pv = getenv("QW4W_PROF");  // "<sweep index>,<ctas/sm>"
            if (pv) launch_wide(SWEEP[atoi(pv)], A.p, w, C.p, 32, s.N, s.K, atoi(strchr(pv, ',') + 1));
            else launch_wide(PROD, A.p, w, C.p, 32, s.N, s.K);
            CK(cudaDeviceSynchronize());
        }
        return 0;
    }
    time_all(mode == "sweep");
    return 0;
}
