// SPDX-License-Identifier: AGPL-3.0-only
// Bit parity and cost of the register-tiled 32-row BF16 tier
// `qwen4exp_bf16_rows32t` (ATLAS_QWEN4EXP_ROWS32_TILE,
// kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_rows32_tile.cu) against
// `dense_gemv_bf16` per row, the arithmetic serial decode runs, and the pair
// tier `qwen4exp_bf16_rows32` it replaces.
//
// check: every M = 1..32 (rows [32 - M, 32) of a 32-row block, so a row's
// bits may depend neither on M nor on its neighbours) over four input sets:
//   clean     Gaussian activations and weights, all in range: every pass fuses
//   wide      log-uniform magnitudes over the fused windows (activations
//             [2^-63, 2^65), weights [2^-63, 2)): products at both ends of
//             [2^-126, 2^66)
//   edge      wide, plus magnitudes just outside the window, +-0 and
//             2^100 planted (1 in 7500): steps fused and falling back mixed
//   planted   +-0, BF16 subnormals and +-1e30 everywhere: every step falls back
//   sparse    clean, plus ONE out-of-window activation (a subnormal or 2^70)
//             in one row and out-of-window weights on a few outputs: the
//             fallback per step and per thread next to fused steps
// time: GPU us per call, weights streamed from DRAM (a ring of copies past
// the 24 MiB L2), clean data (the fused path), at M = 8, 16, 24, 32.
//
// Build/run (repo root, GB10): scripts/dev/qwen4exp_rows32_tile_bench.sh check|time|sweep|prof,
// then `<out>/qwen4exp_rows32_tile_bench <out> real <weight.bin> <N> <K>`.
#include "qwen4exp_ptx_harness.h"

#include <cmath>
#include <random>

typedef unsigned short bf;

static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 4242);

static PtxModule& mod(const char* stem) {
    static std::vector<std::pair<std::string, PtxModule*>> mods;
    for (auto& m : mods) if (m.first == stem) return *m.second;
    PtxModule* m = new PtxModule;
    m->load(g_dir + "/" + stem + ".ptx");
    mods.push_back({stem, m});
    return *m;
}

enum Set { CLEAN, WIDE, EDGE, PLANTED, SPARSE };
static const char* SET_NAME[] = {"clean", "wide", "edge", "planted", "sparse"};

static bf sgn() { return (bf)(g_rng() & 0x8000); }
// The fused windows (qwen4exp_rows32_tile.cu): activations [2^-63, 2^65),
// weights [2^-63, 2^1); `weight` picks which.
static std::vector<bf> gen(size_t n, float scale, Set set, bool weight = false) {
    std::normal_distribution<float> d(0.f, scale);
    const float hi = weight ? 1.f : 65.f;
    std::uniform_real_distribution<float> e(-63.f, hi - 0.01f);
    std::vector<bf> v(n);
    for (auto& x : v) {
        const unsigned k = g_rng() % 211;
        if (set == WIDE || set == EDGE) {
            x = (bf)(f2bf(std::exp2(e(g_rng))) | sgn());
            // Keep the rounding of exp2 inside the window.
            const unsigned t = x & 0x7FFF, top = weight ? 0x4000 : 0x6000;
            if (t < 0x2000 || t >= top) x = (bf)(0x2000 | (x & 0x8000));
            const unsigned k2 = g_rng() % 30011;  // rare: ~1 an activation step
            if (set == EDGE && k2 == 0) x = sgn();
            else if (set == EDGE && k2 == 1) x = (bf)(0x1F80 | sgn() | (g_rng() & 0x7F));  // 2^-64..
            else if (set == EDGE && k2 == 2) x = (bf)((weight ? 0x4000 : 0x6000) | sgn() | (g_rng() & 0x7F));
            else if (set == EDGE && k2 == 3) x = f2bf(std::exp2(100.f)) | sgn();
        } else if (set == PLANTED && k == 0) x = sgn();
        else if (set == PLANTED && k == 1) x = (bf)(1 + g_rng() % 0x7F) | sgn();
        else if (set == PLANTED && k == 2) x = f2bf(1e30f) | sgn();
        else x = f2bf(d(g_rng));
    }
    return v;
}

// Kernel, module, outputs per CTA, threads per CTA, dynamic shared bytes.
struct Tier { const char* fn; const char* stem; unsigned npb, threads, smem; };
static const unsigned TILE_SMEM = 2 * 32 * 64 * 16;
static const Tier PAIR = {"qwen4exp_bf16_rows32", "qwen4exp_wide_rows", 16, 512, 0};
static const Tier TILE = {"qwen4exp_bf16_rows32t", "qwen4exp_rows32_tile", 16, 512, TILE_SMEM};
static const std::vector<Tier> TILE_SWEEP = {
    {"qt_m8_n4_r4_o2", "qwen4exp_rows32_tile", 8, 512, TILE_SMEM},
    {"qt_m16_n2_r2_o4", "qwen4exp_rows32_tile", 8, 512, TILE_SMEM},
    {"qt_m8_n4_r4_o1", "qwen4exp_rows32_tile", 4, 256, TILE_SMEM},
    {"qt_m4_n8_r8_o1", "qwen4exp_rows32_tile", 8, 512, TILE_SMEM}};

static void launch_tier(const Tier& t, const bf* A, const bf* W, bf* C, unsigned m, unsigned N,
                        unsigned K, unsigned stride) {
    Args a;
    a.add(A).add(W).add(C).add(m).add(N).add(K).add(stride);
    CUfunction f = mod(t.stem).fn(t.fn);
    if (t.smem > 48 * 1024)
        CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)t.smem));
    launch(f, dim3((N + t.npb - 1) / t.npb), dim3(t.threads), t.smem, a);
}

static void check() {
    CUfunction g1 = mod("dense_gemv_bf16").fn("dense_gemv_bf16");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2", 8192, 2560}, {"GDN out_proj TP2", 2560, 3072},
        {"LM head shard (ragged)", 4101, 2560}};
    std::vector<Tier> tiers = {TILE};
    if (getenv("QT_ALL")) tiers.insert(tiers.end(), TILE_SWEEP.begin(), TILE_SWEEP.end());
    for (auto& s : shapes) {
        for (Set set : {CLEAN, WIDE, EDGE, PLANTED, SPARSE}) {
            Buf<bf> W, A, R, G;
            W.alloc((size_t)s.N * s.K); A.alloc((size_t)32 * s.K);
            R.alloc((size_t)32 * s.N); G.alloc((size_t)32 * s.N);
            auto w = gen(W.n, 0.05f, set == SPARSE ? CLEAN : set, true);
            auto a = gen(A.n, 1.0f, set == SPARSE ? CLEAN : set);
            if (set == SPARSE) {
                a[(size_t)29 * s.K + 1234] = 0x0003;          // subnormal, row 29
                a[(size_t)31 * s.K + 77] = f2bf(std::exp2(70.f));  // 2^70, row 31
                for (unsigned n : {5u, 6u, 1000u, s.N - 1})   // a few outputs' weights
                    w[(size_t)n * s.K + (g_rng() % s.K)] = (n & 1) ? 0x8001 : f2bf(1e-30f);
            }
            W.put(w); A.put(a);
            for (unsigned r = 0; r < 32; r++) {
                Args g;
                g.add(A.p + (size_t)r * s.K).add(W.p).add(R.p + (size_t)r * s.N).add(s.N).add(s.K);
                launch(g1, dim3((s.N + 3) / 4), dim3(256), 0, g);
            }
            const auto ref = R.get();
            size_t inf = 0;
            for (bf x : ref) inf += (x & 0x7F80) == 0x7F80;
            for (auto& t : tiers) {
                const int bad0 = g_fail;
                for (unsigned m = 1; m <= 32; m++) {
                    const unsigned r0 = 32 - m;
                    G.fill(0x55);
                    launch_tier(t, A.p + (size_t)r0 * s.K, W.p, G.p, m, s.N, s.K, s.N);
                    CK(cudaDeviceSynchronize());
                    const auto got = G.get();
                    size_t d = 0;
                    for (size_t i = 0; i < (size_t)m * s.N; i++) d += ref[(size_t)r0 * s.N + i] != got[i];
                    if (d) {
                        printf("  MISMATCH %s M=%u %s %s: %zu of %zu\n", t.fn, m, s.n, SET_NAME[set], d,
                               (size_t)m * s.N);
                        g_fail++;
                    }
                }
                printf("  %s  %-28s M=1..32 %-24s %-8s vs dense_gemv_bf16 per row (%zu inf/NaN outputs)\n",
                       g_fail != bad0 ? "BAD" : "ok ", t.fn, s.n, SET_NAME[set], inf);
            }
            W.free_(); A.free_(); R.free_(); G.free_();
        }
    }
}

static void time_all(bool sweep) {
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2 8192x2560", 8192, 2560}, {"GDN out_proj TP2 2560x3072", 2560, 3072},
        {"LM head half 124045x2560", 124045, 2560}, {"LM head full 248077x2560", 248077, 2560}};
    std::vector<Tier> tiers = {PAIR, TILE};
    if (sweep) tiers.insert(tiers.end(), TILE_SWEEP.begin(), TILE_SWEEP.end());
    const unsigned MS[] = {8, 16, 24, 32};
    for (auto& s : shapes) {
        const double wb = (double)s.N * s.K * 2;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Buf<bf>> W(copies);
        for (auto& w : W) { w.alloc((size_t)s.N * s.K); w.put(gen(w.n, 0.05f, CLEAN)); }
        Buf<bf> A, C;
        A.alloc((size_t)32 * s.K); C.alloc((size_t)32 * s.N);
        A.put(gen(A.n, 1.0f, CLEAN));
        printf("\n%s (%.1f MB): us per call [GB/s of weight bytes] at M =", s.n, wb / 1e6);
        for (unsigned m : MS) printf(" %8u    ", m);
        printf("\n");
        const int iters = s.N > 100000 ? 4 : 20;
        for (auto& t : tiers) {
            printf("  %-28s", t.fn);
            int it = 0;
            for (unsigned m : MS) {
                const float ms = time_ms([&] {
                    const int c = it++ % copies;
                    launch_tier(t, A.p, W[c].p, C.p, m, s.N, s.K, s.N);
                }, iters, 5);
                printf(" %7.1f [%3.0f]", ms * 1e3, wb / (ms * 1e-3) / 1e9);
            }
            printf("\n");
        }
        for (auto& w : W) w.free_();
        A.free_(); C.free_();
    }
}

// real <file> <N> <K>: a checkpoint weight (raw BF16 [N, K], e.g. dumped
// from the safetensors) with Gaussian activations: parity at M = 1..32 and
// the pair tier against the tile at M = 24, 32 (the fallback on out-of-window
// weights included).
static void real(const char* path, unsigned N, unsigned K) {
    std::vector<bf> w((size_t)N * K);
    FILE* f = fopen(path, "rb");
    if (!f || fread(w.data(), 2, w.size(), f) != w.size()) { printf("read %s failed\n", path); exit(1); }
    fclose(f);
    const double wb = (double)N * K * 2;
    const int copies = std::max(2, (int)(160e6 / wb) + 1);
    std::vector<Buf<bf>> W(copies);
    for (auto& b : W) { b.alloc(w.size()); b.put(w); }
    Buf<bf> A, R, G;
    A.alloc((size_t)32 * K); R.alloc((size_t)32 * N); G.alloc((size_t)32 * N);
    A.put(gen(A.n, 1.0f, CLEAN));
    CUfunction g1 = mod("dense_gemv_bf16").fn("dense_gemv_bf16");
    for (unsigned r = 0; r < 32; r++) {
        Args g;
        g.add(A.p + (size_t)r * K).add(W[0].p).add(R.p + (size_t)r * N).add(N).add(K);
        launch(g1, dim3((N + 3) / 4), dim3(256), 0, g);
    }
    const auto ref = R.get();
    size_t bad = 0;
    for (unsigned m = 1; m <= 32; m++) {
        const unsigned r0 = 32 - m;
        G.fill(0x55);
        launch_tier(TILE, A.p + (size_t)r0 * K, W[0].p, G.p, m, N, K, N);
        CK(cudaDeviceSynchronize());
        const auto got = G.get();
        for (size_t i = 0; i < (size_t)m * N; i++) bad += ref[(size_t)r0 * N + i] != got[i];
    }
    printf("  %s  %s %ux%u: %s M=1..32 vs dense_gemv_bf16 per row (%zu differ)\n", bad ? "BAD" : "ok ",
           path, N, K, TILE.fn, bad);
    g_fail += bad != 0;
    for (const Tier* t : {&PAIR, &TILE}) {
        printf("  %-24s", t->fn);
        for (unsigned m : {24u, 32u}) {
            int it = 0;
            const float ms = time_ms([&] { launch_tier(*t, A.p, W[it++ % copies].p, G.p, m, N, K, N); },
                                     N > 100000 ? 4 : 20, 5);
            printf("  M=%u %7.1f us [%3.0f GB/s]", m, ms * 1e3, wb / (ms * 1e-3) / 1e9);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    init_driver();
    if (mode == "check") {
        check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    if (mode == "real" && argc > 5) {
        real(argv[3], (unsigned)atoi(argv[4]), (unsigned)atoi(argv[5]));
        return g_fail ? 1 : 0;
    }
    if (mode == "prof") {  // one LM-head-half launch a tier at M = 32, for ncu
        const unsigned N = 124045, K = 2560;
        Buf<bf> W, A, C;
        W.alloc((size_t)N * K); A.alloc((size_t)32 * K); C.alloc((size_t)32 * N);
        W.put(gen(W.n, 0.05f, CLEAN)); A.put(gen(A.n, 1.0f, CLEAN));
        std::vector<Tier> tiers = {PAIR, TILE};
        tiers.insert(tiers.end(), TILE_SWEEP.begin(), TILE_SWEEP.end());
        for (auto& t : tiers) launch_tier(t, A.p, W.p, C.p, 32, N, K, N);
        CK(cudaDeviceSynchronize());
        return 0;
    }
    time_all(mode == "sweep");
    return 0;
}
