// SPDX-License-Identifier: AGPL-3.0-only
// Bit parity and cost of the persistent NVFP4 row tier `qwen4exp_w4_rows{M}`
// (ATLAS_QWEN4EXP_W4_ROWS, kernels/gb10/qwen3.8-flash-next/nvfp4/
// qwen4exp_w4_rows.cu) against `w4a16_gemv` per row, the arithmetic the
// drafter's single-row head runs, and the `w4a16_gemv_batch{M}` tiers it
// replaces at the MTP draft head (50k rows a rank under DRAFT_TP, 100k whole).
//
// check: every M = 1..8 over random E2M1 codes (all 16 nibbles), E4M3 scales
// including 0 and subnormals, Gaussian activations with +-0, BF16 subnormals
// and +-1e30 planted; draft-head-half and ragged shapes.
// time:  GPU us per call, weights streamed from DRAM (a ring of copies past the
// 24 MiB L2), M = 1, 4, 8.
//
// Build/run (repo root, GB10): scripts/dev/qwen4exp_w4_rows_bench.sh check|time|sweep
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

static const char* NEW[] = {"", "qwen4exp_w4_rows1", "qwen4exp_w4_rows2", "qwen4exp_w4_rows3",
                            "qwen4exp_w4_rows4", "qwen4exp_w4_rows5", "qwen4exp_w4_rows6",
                            "qwen4exp_w4_rows7", "qwen4exp_w4_rows8"};
static const char* OLD[] = {"", "w4a16_gemv", "w4a16_gemv_batch2", "w4a16_gemv_batch3",
                            "w4a16_gemv_batch4", "w4a16_gemv_batch5", "w4a16_gemv_batch6",
                            "w4a16_gemv_batch7", "w4a16_gemv_batch8"};

// The launch exactly as `ops::Qwen4ExpW4Rows::launch` makes it: all shared
// memory dynamic (`w4_rows_smem`), the opt-in attribute set only past 48 KB
// (what `AtlasRegistry::launch_on_stream` does), grid = SM count. A size the
// launcher would not opt in for must fit the default limit, or this fails.
static unsigned smem_bytes(unsigned m, unsigned npb) { return m * 2 * 4096 + (2 * m * npb * 2 + 16) * 4; }
static void launch_new(const char* fn, unsigned m, unsigned threads, const bf* A, const Fp4& w,
                       bf* C, unsigned N, unsigned K, unsigned npb = 16) {
    CUfunction f = mod("qwen4exp_w4_rows").fn(fn);
    const unsigned smem = smem_bytes(m, npb);
    if (smem > 48 * 1024)
        CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C).add(N).add(K);
    launch(f, dim3(g_sms), dim3(threads), smem, a);
}
static void launch_old(unsigned m, const bf* A, const Fp4& w, bf* C, unsigned N, unsigned K) {
    Args a;
    a.add(A).add(w.packed.p).add(w.scale.p).add(w.s2).add(C);
    if (m == 1) {
        a.add(N).add(K);
    } else if (m <= 3) {
        a.add(N).add(K);  // batch2/3: fixed M
    } else {
        a.add(m).add(N).add(K);
    }
    launch(mod("w4a16_gemv").fn(OLD[m]), dim3((N + 3) / 4), dim3(256), 0, a);
}

static void check() {
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"draft head half", 50000, 2560}, {"ragged", 4101, 2560}, {"o_proj-like", 2560, 3072}};
    for (auto& s : shapes) {
        Fp4 w = make_fp4(s.N, s.K);
        for (bool planted : {false, true}) {
            Buf<bf> A, R, G;
            A.alloc((size_t)8 * s.K); R.alloc((size_t)8 * s.N); G.alloc((size_t)8 * s.N);
            A.put(rbf(A.n, planted));
            for (unsigned r = 0; r < 8; r++) launch_old(1, A.p + (size_t)r * s.K, w, R.p + (size_t)r * s.N, s.N, s.K);
            const auto ref = R.get();
            const int bad0 = g_fail;
            for (unsigned m = 1; m <= 8; m++) {
                const unsigned r0 = 8 - m;
                G.fill(0x55);
                launch_new(NEW[m], m, 256, A.p + (size_t)r0 * s.K, w, G.p, s.N, s.K);
                CK(cudaDeviceSynchronize());
                const auto got = G.get();
                size_t d = 0;
                for (size_t i = 0; i < (size_t)m * s.N; i++) d += ref[(size_t)r0 * s.N + i] != got[i];
                if (d) { printf("  MISMATCH %s %s %s: %zu of %zu\n", NEW[m], s.n, planted ? "planted" : "clean", d, (size_t)m * s.N); g_fail++; }
            }
            printf("  %s  qwen4exp_w4_rows1..8  %-16s %ux%u %-8s vs w4a16_gemv per row\n",
                   g_fail != bad0 ? "BAD" : "ok ", s.n, s.N, s.K, planted ? "planted" : "clean");
            A.free_(); R.free_(); G.free_();
        }
        w.packed.free_(); w.scale.free_();
    }
}

static void time_all(bool sweep) {
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"draft head half 50000x2560", 50000, 2560}, {"draft head 100000x2560", 100000, 2560}};
    for (auto& s : shapes) {
        const double wb = (double)s.N * s.K / 2 + (double)s.N * s.K / 16;
        const int copies = std::max(2, (int)(160e6 / wb) + 1);
        std::vector<Fp4> W;
        for (int c = 0; c < copies; c++) W.push_back(make_fp4(s.N, s.K));
        Buf<bf> A, C;
        A.alloc((size_t)8 * s.K); C.alloc((size_t)8 * s.N);
        A.put(rbf(A.n, false));
        printf("\n%s (%.1f MB): us per call [GB/s]       M=1              M=4              M=5              M=6              M=7              M=8\n", s.n, wb / 1e6);
        auto row = [&](const char* name, auto body) {
            printf("  %-24s", name);
            for (unsigned m : {1u, 4u, 5u, 6u, 7u, 8u}) {
                int it = 0;
                const float ms = time_ms([&] { body(m, it++ % copies); }, 10, 5);
                printf("  %7.1f [%3.0f]", ms * 1e3, wb / (ms * 1e-3) / 1e9);
            }
            printf("\n");
        };
        row("w4a16_gemv[_batchM]", [&](unsigned m, int c) { launch_old(m, A.p, W[c], C.p, s.N, s.K); });
        row("qwen4exp_w4_rowsM", [&](unsigned m, int c) { launch_new(NEW[m], m, 256, A.p, W[c], C.p, s.N, s.K); });
        if (sweep) {
            struct { const char* fn; unsigned threads, npb; } sw[] = {
                {"qw4_m8_n1_o8", 512, 8}, {"qw4_m8_n2_o8", 512, 16}, {"qw4_m8_n2_o4", 256, 8}};
            for (auto& x : sw) {
                printf("  %-24s", x.fn);
                int it = 0;
                const float ms = time_ms([&] { launch_new(x.fn, 8, x.threads, A.p, W[it++ % copies], C.p, s.N, s.K, x.npb); }, 10, 5);
                printf("                                    %7.1f [%3.0f]\n", ms * 1e3, wb / (ms * 1e-3) / 1e9);
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
    int dev = 0;
    CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
    if (mode == "check") {
        check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    time_all(mode == "sweep");
    return 0;
}
