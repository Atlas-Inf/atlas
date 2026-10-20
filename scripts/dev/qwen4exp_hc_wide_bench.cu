// SPDX-License-Identifier: AGPL-3.0-only
// Per-token bit parity and cost of the qwen4_exp mHC collapse over 9..32
// tokens with the down walk re-tiled (`hc_pre_down_vec_wide`, four chains a
// thread, ATLAS_QWEN4EXP_HC_WIDE) against `hc_pre_down_vec_rows`; both arms
// finish with `hc_pre_finish_vec_rows`. Real site shape (hidden 2560, hc 4,
// rank 320). -DHC_V_WIDE_* / -DHC_V_WIDE_STAGE_SPLIT sweep the wide arm.
//
// check: for T = 9..32, with and without the injection rows, every byte of
// `low`, `inj` and `y` of every token must equal the T = 1 `_vec` kernels
// run on that token alone (random highway at three scales with zeros and
// -0.0; weights with zeros and -0.0), with the scratch rows past T poisoned.
// time:  us per site and per kernel at T = 16..32, rows vs wide, the
// weights cycled over 8 copies (4x the 24 MiB L2) so each site starts cold.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 \
//        -I kernels/gb10/qwen3.8-flash-next/nvfp4 \
//        scripts/dev/qwen4exp_hc_wide_bench.cu -o qwen4exp_hc_wide_bench
//   ./qwen4exp_hc_wide_bench [check|time]
#include "hyper_connection.cu"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H, MAXT = 32, SCRATCH = 64, COPIES = 8;
static const float EPS = 1e-6f;
static int g_fail = 0;

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    void alloc(size_t count) { n = count; CK(cudaMalloc(&p, n * sizeof(T))); CK(cudaMemset(p, 0, n * sizeof(T))); }
    void put(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void fill(unsigned char b) { CK(cudaMemset(p, b, n * sizeof(T))); }
};

struct Site { Dev<bf> norm_w, down_w, up_w, inject_w; };
struct Out {
    Dev<float> normed, low, inj;
    Dev<bf> y;
    void alloc() { normed.alloc((size_t)SCRATCH * HCD); low.alloc((size_t)MAXT * RANK); inj.alloc(MAXT * HC); y.alloc((size_t)MAXT * H); }
    void poison() { normed.fill(0x7F); low.fill(0x7F); inj.fill(0x7F); y.fill(0x7F); }
};

static std::mt19937 g_rng(11);
static std::vector<bf> gen_bf(size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<bf> h(n);
    for (auto& v : h) {
        const float u = ud(g_rng);
        v = __float2bfloat16(u < 0.001f ? 0.f : u < 0.002f ? -0.f : nd(g_rng));
    }
    return h;
}
static void make_site(Site& s, unsigned copies) {
    s.norm_w.alloc((size_t)copies * HCD); s.norm_w.put(gen_bf(s.norm_w.n, 0.1f));
    s.down_w.alloc((size_t)copies * RANK * HCD); s.down_w.put(gen_bf(s.down_w.n, 0.02f));
    s.up_w.alloc((size_t)copies * RANK * HCD); s.up_w.put(gen_bf(s.up_w.n, 0.05f));
    s.inject_w.alloc((size_t)copies * HC * HCD); s.inject_w.put(gen_bf(s.inject_w.n, 0.02f));
}
struct W { const bf *norm, *down, *up, *inject; };
static W copy_of(const Site& s, unsigned c) {
    return {s.norm_w.p + (size_t)c * HCD, s.down_w.p + (size_t)c * RANK * HCD,
            s.up_w.p + (size_t)c * RANK * HCD, s.inject_w.p + (size_t)c * HC * HCD};
}

static unsigned down_grid(bool inject) {
    return ((RANK + (inject ? HC : 0)) * (32 / HC_V_DOWN_CPT) + 127) / 128;
}
static const unsigned FIN_GRID = (4 * H / HC_V_FIN_DPT + 127) / 128;

// The lane before: stage + `_vec`/`_vec8` down + finish on tokens [t0, t0 + T).
static void chunk_chain(const float* streams, const W& w, Out& o, unsigned t0, unsigned T, bool inject) {
    float* nx = o.normed.p + (size_t)t0 * HCD;
    hc_pre_stage_vec<<<dim3(T, HC_V_STAGE_SPLIT), 1024>>>(streams + (size_t)t0 * HCD, w.norm, nx, H, HC, EPS);
    (T <= 4 ? hc_pre_down_vec : hc_pre_down_vec8)<<<down_grid(inject), 128>>>(
        nx, w.down, inject ? w.inject : nullptr, o.low.p + (size_t)t0 * RANK, o.inj.p + (size_t)t0 * HC, H, HC,
        RANK, T);
    (T <= 4 ? hc_pre_finish_vec : hc_pre_finish_vec8)<<<FIN_GRID, 128, T * RANK * 4>>>(
        nx, o.low.p + (size_t)t0 * RANK, w.up, (bf*)o.y.p + (size_t)t0 * H, H, RANK, T);
}
// The row-grouped launch over all T tokens.
static void rows_chain(const float* streams, const W& w, Out& o, unsigned T, bool inject) {
    const unsigned groups = (T + HC_V_MAX - 1) / HC_V_MAX;
    hc_pre_stage_vec<<<dim3(T, HC_V_STAGE_SPLIT), 1024>>>(streams, w.norm, o.normed.p, H, HC, EPS);
    hc_pre_down_vec_rows<<<dim3(down_grid(inject), groups), 128>>>(
        o.normed.p, w.down, inject ? w.inject : nullptr, o.low.p, o.inj.p, H, HC, RANK, T);
    hc_pre_finish_vec_rows<<<dim3(FIN_GRID, groups), 128, HC_V_MAX * RANK * 4>>>(
        o.normed.p, o.low.p, w.up, (bf*)o.y.p, H, RANK, T);
}

static unsigned groups_of(unsigned T, unsigned g) { return (T + g - 1) / g; }
static const unsigned WIDE_DOWN_GRID_INJ = ((RANK + HC) * (32 / HC_V_WIDE_DOWN_CPT) + 127) / 128;
static const unsigned WIDE_DOWN_GRID_HEAD = (RANK * (32 / HC_V_WIDE_DOWN_CPT) + 127) / 128;
#ifndef HC_V_WIDE_STAGE_SPLIT
#define HC_V_WIDE_STAGE_SPLIT 2u  // the serving arm's (HC_V_WIDE_STAGE_SPLIT in the .rs)
#endif
// The re-tiled launch over all T tokens; `part` 1 = stage, 2 = down,
// 3 = finish, 0 = all three.
static void wide_chain(const float* streams, const W& w, Out& o, unsigned T, bool inject, int part = 0) {
    if (part == 0 || part == 1)
        hc_pre_stage_vec<<<dim3(T, HC_V_WIDE_STAGE_SPLIT), 1024>>>(streams, w.norm, o.normed.p, H, HC, EPS);
    if (part == 0 || part == 2)
        hc_pre_down_vec_wide<<<dim3(inject ? WIDE_DOWN_GRID_INJ : WIDE_DOWN_GRID_HEAD,
                                    groups_of(T, HC_V_WIDE_G)), 128>>>(
            o.normed.p, w.down, inject ? w.inject : nullptr, o.low.p, o.inj.p, H, HC, RANK, T);
    if (part == 0 || part == 3)
        hc_pre_finish_vec_rows<<<dim3(FIN_GRID, groups_of(T, HC_V_MAX)), 128, HC_V_MAX * RANK * 4>>>(
            o.normed.p, o.low.p, w.up, (bf*)o.y.p, H, RANK, T);
}

static void rows_part(const float* streams, const W& w, Out& o, unsigned T, bool inject, int part) {
    const unsigned groups = (T + HC_V_MAX - 1) / HC_V_MAX;
    if (part == 1)
        hc_pre_stage_vec<<<dim3(T, HC_V_STAGE_SPLIT), 1024>>>(streams, w.norm, o.normed.p, H, HC, EPS);
    if (part == 2)
        hc_pre_down_vec_rows<<<dim3(down_grid(inject), groups), 128>>>(
            o.normed.p, w.down, inject ? w.inject : nullptr, o.low.p, o.inj.p, H, HC, RANK, T);
    if (part == 3)
        hc_pre_finish_vec_rows<<<dim3(FIN_GRID, groups), 128, HC_V_MAX * RANK * 4>>>(
            o.normed.p, o.low.p, w.up, (bf*)o.y.p, H, RANK, T);
}

template <typename T>
static void same(const char* what, const Dev<T>& a, const Dev<T>& b, size_t n) {
    auto ha = a.get(), hb = b.get();
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    if (bad) { printf("  MISMATCH %-40s %zu of %zu\n", what, bad, n); g_fail++; }
}

static void check() {
    Site s;
    make_site(s, 1);
    const W w = copy_of(s, 0);
    Dev<float> streams;
    streams.alloc((size_t)MAXT * HCD);
    Out ref, got;
    ref.alloc(); got.alloc();
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (float scale : {1.f, 30.f, 1e-3f}) {
        std::vector<float> hs((size_t)MAXT * HCD);
        for (auto& v : hs) { const float u = ud(g_rng); v = u < 0.002f ? 0.f : u < 0.004f ? -0.f : scale * nd(g_rng); }
        streams.put(hs);
        for (bool inject : {true, false}) {
            ref.poison();
            for (unsigned t = 0; t < MAXT; t++) chunk_chain(streams.p, w, ref, t, 1, inject);
            for (unsigned T = HC_V_MAX + 1; T <= MAXT; T++) {
                got.poison();
                wide_chain(streams.p, w, got, T, inject);
                CK(cudaDeviceSynchronize());
                char what[96];
                snprintf(what, sizeof what, "T=%u x%g %s low", T, scale, inject ? "inject" : "head");
                same(what, ref.low, got.low, (size_t)T * RANK);
                if (inject) {
                    snprintf(what, sizeof what, "T=%u x%g inj", T, scale);
                    same(what, ref.inj, got.inj, (size_t)T * HC);
                }
                snprintf(what, sizeof what, "T=%u x%g %s y", T, scale, inject ? "inject" : "head");
                same(what, ref.y, got.y, (size_t)T * H);
            }
        }
    }
    printf("  %s  hc_pre_down_vec_wide + finish_vec_rows T = 9..32, with/without injection, 3 scales, vs T = 1 per token\n",
           g_fail ? "BAD" : "ok ");
    printf("%s\n", g_fail ? "FAIL" : "PASS");
}

static void time_sites() {
    Site s;
    make_site(s, COPIES);
    Dev<float> streams;
    streams.alloc((size_t)MAXT * HCD);
    Out o;
    o.alloc();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int iters = 64;
    auto timed = [&](auto&& body) {
        float ms = 0;
        for (int rep = 0; rep < 2; rep++) {  // first rep warms
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) body(copy_of(s, i % COPIES));
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
        }
        return ms * 1e3f / iters;
    };
    printf("us per launch, weights cold (inject on):\n  T   arm    site    stage    down  finish\n");
    for (unsigned T : {16u, 20u, 24u, 25u, 28u, 32u}) {
        for (int arm = 0; arm < 2; arm++) {
            float us[4];
            for (int part = 0; part < 4; part++) {
                us[part] = timed([&](const W& w) {
                    if (arm == 0) {
                        if (part == 0) rows_chain(streams.p, w, o, T, true);
                        else rows_part(streams.p, w, o, T, true, part);
                    } else {
                        wide_chain(streams.p, w, o, T, true, part);
                    }
                });
            }
            printf("  %2u  %s  %6.1f  %6.1f  %6.1f  %6.1f\n", T, arm ? "wide" : "rows", us[0], us[1], us[2], us[3]);
        }
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "check";
    CK(cudaFree(0));
    if (mode == "check") {
        check();
        return g_fail ? 1 : 0;
    }
    time_sites();
    return 0;
}
