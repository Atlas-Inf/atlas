// SPDX-License-Identifier: AGPL-3.0-only
// Per-token bit parity and cost of the qwen4_exp mHC collapse over 9..32
// tokens in one launch (`hc_pre_down_vec_rows` / `hc_pre_finish_vec_rows`,
// ATLAS_QWEN4EXP_BATCH_FAST + ATLAS_QWEN4EXP_HC_FAST) against the 8-token
// launches the lane ran before, on the real site shape (hidden 2560, hc 4,
// rank 320).
//
// check: for T = 1..32, with and without the injection rows, every byte of
// `low`, `inj` and `y` of every token must equal the T = 1 `_vec` kernels
// run on that token alone (random highway at three scales with zeros and
// -0.0; weights with zeros and -0.0).
// time:  us per site (stage + down + finish) at T = 8, 16, 24, 32 as serving
// launches it, 8-token chunks back to back vs one row-grouped launch, the
// weights cycled over 8 copies (4x the 24 MiB L2) so each site starts cold.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 \
//        -I kernels/gb10/qwen3.8-flash-next/nvfp4 \
//        scripts/dev/qwen4exp_hc_rows_bench.cu -o qwen4exp_hc_rows_bench
//   ./qwen4exp_hc_rows_bench [check|time]
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
static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H, MAXT = 32, COPIES = 8;
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
    void alloc() { normed.alloc((size_t)MAXT * HCD); low.alloc((size_t)MAXT * RANK); inj.alloc(MAXT * HC); y.alloc((size_t)MAXT * H); }
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
            for (unsigned T = 1; T <= MAXT; T++) {
                got.poison();
                rows_chain(streams.p, w, got, T, inject);
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
    printf("  %s  hc_pre_{down,finish}_vec_rows T = 1..32, with/without injection, 3 scales, vs T = 1 per token\n",
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
    printf("us per mHC site (stage + down + finish), weights cold:\n  T    8-token chunks   one grouped launch\n");
    for (unsigned T : {8u, 16u, 24u, 32u}) {
        float ms[2];
        for (int arm = 0; arm < 2; arm++) {
            for (int rep = 0; rep < 2; rep++) {  // first rep warms
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; i++) {
                    const W w = copy_of(s, i % COPIES);
                    if (arm == 0) {
                        for (unsigned t0 = 0; t0 < T; t0 += HC_V_MAX)
                            chunk_chain(streams.p, w, o, t0, std::min(HC_V_MAX, T - t0), true);
                    } else {
                        rows_chain(streams.p, w, o, T, true);
                    }
                }
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[arm], e0, e1));
            }
        }
        printf("  %2u   %8.1f          %8.1f   (%.2fx)\n", T, ms[0] * 1e3 / iters, ms[1] * 1e3 / iters,
               ms[0] / ms[1]);
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
