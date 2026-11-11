// SPDX-License-Identifier: AGPL-3.0-only
// Row invariance, accuracy and cost of the tensor-core mHC collapse
// (`hc_mma_down` + `hc_mma_finish`, ATLAS_QWEN4EXP_HC_MMA) against the FP32
// kernels serving runs today (`_vec` T <= 4, `_vec8` 5..8, `_vec_rows` 9..24,
// `_vec_wide` 25..32 under ATLAS_QWEN4EXP_HC_WIDE). Real site shape (hidden
// 2560, hc 4, rank 320).
//
// check: contract (b). For T = 1..32 at three batch offsets, with and without
//        the injection rows, every byte of `low`, `inj` and `y` of every row
//        must equal the T = 1 launch of the same kernels on that row alone
//        (random highway at three scales with zeros and -0.0; weights with
//        zeros and -0.0; scratch rows past T poisoned with NaN). Then the
//        error of both arms against an FP64 reference of the same `normed`,
//        and `normed` byte-equal across stage splits 8/4/2/1
//        (ATLAS_QWEN4EXP_HC_STAGE_FIT). HCM_PDL=1 runs it all with PDL
//        launches of down + finish.
// time:  us per site and per kernel at T = 1, 4, 8, 16, 24, 32, weights
//        cycled over 8 copies (4x the 24 MiB L2) so each site starts cold:
//        FP32 as served, FP32 + STAGE_FIT, mma + STAGE_FIT, and that with PDL.
// sweep: the mma down / finish alone (-DHCM_* geometry sweeps; HCM_WARM=1
//        keeps one weight copy, i.e. L2-resident).
// prof T arm: 32 sites of one arm (0 = FP32, 1 = mma) for ncu.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 \
//        -I kernels/gb10/qwen3.8-flash-next/nvfp4 \
//        scripts/dev/qwen4exp_hc_mma_bench.cu -o qwen4exp_hc_mma_bench
//   ./qwen4exp_hc_mma_bench [check|time|prof T arm]
#include "hyper_connection.cu"
#include "qwen4exp_hc_mma.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H, MAXT = 32, ROWS = 64, COPIES = 8;
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
    void alloc() { normed.alloc((size_t)ROWS * HCD); low.alloc((size_t)ROWS * RANK); inj.alloc(ROWS * HC); y.alloc((size_t)ROWS * H); }
    void poison() { normed.fill(0xFF); low.fill(0xFF); inj.fill(0xFF); y.fill(0xFF); }
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

// ── the two arms; `part` 1 = stage, 2 = down, 3 = finish, 0 = all ──
// Stage blocks a token: serving's fixed rule, or ATLAS_QWEN4EXP_HC_STAGE_FIT's
// one-wave rule (`g_fit`, the largest power of two <= 8 with T x split <= 48).
static bool g_fit = false;
static int g_split = getenv("HCM_SPLIT") ? atoi(getenv("HCM_SPLIT")) : 0;  // override, for sweeps
static unsigned stage_split(unsigned T) {
    if (g_split) return g_split;
    if (!g_fit) return T >= 25 ? 2u : HC_V_STAGE_SPLIT;
    unsigned s = HC_V_STAGE_SPLIT;
    while (s > 1 && T * s > 48) s /= 2;
    return s;
}
static void stage(const float* streams, const W& w, Out& o, unsigned T) {
    hc_pre_stage_vec<<<dim3(T, stage_split(T)), 1024>>>(streams, w.norm, o.normed.p, H, HC, EPS);
}
static void fp32_chain(const float* streams, const W& w, Out& o, unsigned T, bool inject, int part = 0) {
    const bf* inj = inject ? w.inject : nullptr;
    const unsigned rows = RANK + (inject ? HC : 0);
    const bool wide = T >= 25;
    const unsigned cpt = wide ? HC_V_WIDE_DOWN_CPT : HC_V_DOWN_CPT;
    const unsigned dgrid = (rows * (32 / cpt) + 127) / 128;
    const unsigned fgrid = (4 * H / HC_V_FIN_DPT + 127) / 128;
    const unsigned groups = (T + HC_V_MAX - 1) / HC_V_MAX;
    if (part == 0 || part == 1) stage(streams, w, o, T);
    if (part == 0 || part == 2) {
        if (wide)
            hc_pre_down_vec_wide<<<dim3(dgrid, (T + HC_V_WIDE_G - 1) / HC_V_WIDE_G), 128>>>(
                o.normed.p, w.down, inj, o.low.p, o.inj.p, H, HC, RANK, T);
        else if (T > HC_V_MAX)
            hc_pre_down_vec_rows<<<dim3(dgrid, groups), 128>>>(o.normed.p, w.down, inj, o.low.p, o.inj.p, H, HC, RANK, T);
        else
            (T <= 4 ? hc_pre_down_vec : hc_pre_down_vec8)<<<dgrid, 128>>>(
                o.normed.p, w.down, inj, o.low.p, o.inj.p, H, HC, RANK, T);
    }
    if (part == 0 || part == 3) {
        if (T > HC_V_MAX)
            hc_pre_finish_vec_rows<<<dim3(fgrid, groups), 128, HC_V_MAX * RANK * 4>>>(
                o.normed.p, o.low.p, w.up, o.y.p, H, RANK, T);
        else
            (T <= 4 ? hc_pre_finish_vec : hc_pre_finish_vec8)<<<fgrid, 128, T * RANK * 4>>>(
                o.normed.p, o.low.p, w.up, o.y.p, H, RANK, T);
    }
}
// `HCM_PDL=1`: launch with programmatic dependent launch, as ATLAS_QWEN4EXP_PDL
// does in serving (each kernel may start, and stream its weights, while its
// predecessor finishes).
static int g_pdl = getenv("HCM_PDL") ? atoi(getenv("HCM_PDL")) : 0;  // 1 both, 2 down, 3 finish
template <typename... KArgs, typename... Args>
static void launch(void (*k)(KArgs...), dim3 grid, unsigned block, unsigned smem, Args... args) {
    const bool pdl = g_pdl == 1 || (g_pdl == 2 && (void*)k == (void*)hc_mma_down) ||
                     (g_pdl == 3 && (void*)k == (void*)hc_mma_finish);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = dim3(block); cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, k, (KArgs)args...));
}
static unsigned mma_smem(unsigned T) {
    const unsigned nt = (T + 7) / 8, tok = nt * 8;
    const unsigned b = RANK / 16 * nt * 512;
    const unsigned p = 4 * tok * HCM_FN_DW * 4;
    const unsigned k = (HCM_FN_WK - 1) * 4 * (HCM_FN_DW / 16) * nt * 4 * 32 * 4;
    return std::max(b, std::max(p, k));
}
static void mma_chain(const float* streams, const W& w, Out& o, unsigned T, bool inject, int part = 0) {
    const unsigned tiles = (RANK + (inject ? HC : 0) + 15) / 16;
    if (part == 0 || part == 1) stage(streams, w, o, T);
    if (part == 0 || part == 2)
        launch(hc_mma_down, dim3((tiles + HCM_DN_WM - 1) / HCM_DN_WM, HCM_DN_CL), 32 * HCM_DN_WM * HCM_DN_WK, 0,
               (const float*)o.normed.p, w.down, inject ? w.inject : (const bf*)nullptr, o.low.p, o.inj.p, H, HC,
               RANK, T);
    if (part == 0 || part == 3)
        launch(hc_mma_finish, dim3(H / HCM_FN_DW), 128 * HCM_FN_WK, mma_smem(T), (const float*)o.normed.p,
               (const float*)o.low.p, w.up, o.y.p, H, RANK, T);
}
static void chain(int arm, const float* streams, const W& w, Out& o, unsigned T, bool inject, int part = 0) {
    if (arm == 0) fp32_chain(streams, w, o, T, inject, part);
    else mma_chain(streams, w, o, T, inject, part);
}

template <typename T>
static size_t diff(const std::vector<T>& a, size_t ao, const std::vector<T>& b, size_t bo, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += memcmp(&a[ao + i], &b[bo + i], sizeof(T)) != 0;
    return bad;
}

static double silu_d(double v) { return v / (1.0 + std::exp(-v)); }
static double sig_d(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// Error of one arm's low / inj / y over T rows against FP64 from its `normed`.
static void accuracy(int arm, const Site& s, const W& w, Out& o, const Dev<float>& streams, unsigned T) {
    o.poison();
    chain(arm, streams.p, w, o, T, true);
    CK(cudaDeviceSynchronize());
    const auto nx = o.normed.get(), low = o.low.get(), inj = o.inj.get();
    const auto y = o.y.get();
    const auto dw = s.down_w.get(), uw = s.up_w.get(), iw = s.inject_w.get();
    double e_low = 0, m_low = 0, e_inj = 0, m_y = 0, ulp_max = 0;
    size_t y_bad = 0;
    std::vector<double> lo(RANK);
    for (unsigned t = 0; t < T; t++) {
        const float* n = &nx[(size_t)t * HCD];
        for (unsigned r = 0; r < RANK + HC; r++) {
            const bf* wr = r < RANK ? &dw[(size_t)r * HCD] : &iw[(size_t)(r - RANK) * HCD];
            double acc = 0;
            for (unsigned k = 0; k < HCD; k++) acc += (double)__bfloat162float(wr[k]) * n[k];
            if (r < RANK) {
                lo[r] = silu_d(acc / HC);
                e_low = std::max(e_low, std::fabs(lo[r] - low[(size_t)t * RANK + r]));
                m_low = std::max(m_low, std::fabs(lo[r]));
            } else {
                e_inj = std::max(e_inj, std::fabs(2 * sig_d(acc / HC) - inj[(size_t)t * HC + r - RANK]));
            }
        }
        for (unsigned d = 0; d < H; d++) {
            double mix = 0;
            for (unsigned sm = 0; sm < HC; sm++) {
                double u = 0;
                for (unsigned r = 0; r < RANK; r++) u += (double)__bfloat162float(uw[(size_t)r * HCD + sm * H + d]) * lo[r];
                mix += sig_d(u) * n[sm * H + d];
            }
            mix /= HC;
            const float ref = __bfloat162float(__float2bfloat16((float)mix));
            const float got = __bfloat162float(y[(size_t)t * H + d]);
            m_y = std::max(m_y, std::fabs(mix));
            if (ref != got) {
                y_bad++;
                const double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(ref), 1e-30f)) - 7);
                ulp_max = std::max(ulp_max, std::fabs((double)got - mix) / ulp);
            }
        }
    }
    printf("  %-5s T=%u  low max|err| %.3e (of max %.3f)  inj max|err| %.3e  y != rn(fp64) %zu of %u (%.3f%%), worst %.2f ulp\n",
           arm ? "mma" : "fp32", T, e_low, m_low, e_inj, y_bad, T * H, 100.0 * y_bad / (T * H), ulp_max);
}

static void check() {
    Site s;
    make_site(s, 1);
    const W w = copy_of(s, 0);
    Dev<float> streams;
    streams.alloc((size_t)ROWS * HCD);
    Out ref, got;
    ref.alloc(); got.alloc();
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    size_t launches = 0;
    for (float scale : {1.f, 30.f, 1e-3f}) {
        std::vector<float> hs((size_t)ROWS * HCD);
        for (auto& v : hs) { const float u = ud(g_rng); v = u < 0.002f ? 0.f : u < 0.004f ? -0.f : scale * nd(g_rng); }
        streams.put(hs);
        for (bool inject : {true, false}) {
            // Every row alone: the T = 1 launch on its own (poisoned) scratch.
            std::vector<float> rl((size_t)ROWS * RANK), ri(ROWS * HC);
            std::vector<bf> ry((size_t)ROWS * H);
            for (unsigned t = 0; t < ROWS; t++) {
                ref.poison();
                mma_chain(streams.p + (size_t)t * HCD, w, ref, 1, inject);
                const auto l = ref.low.get(), i = ref.inj.get();
                const auto y = ref.y.get();
                std::copy(l.begin(), l.begin() + RANK, rl.begin() + (size_t)t * RANK);
                std::copy(i.begin(), i.begin() + HC, ri.begin() + (size_t)t * HC);
                std::copy(y.begin(), y.begin() + H, ry.begin() + (size_t)t * H);
            }
            for (unsigned T = 1; T <= MAXT; T++) {
                for (unsigned off : {0u, 13u, ROWS - T}) {
                    got.poison();
                    mma_chain(streams.p + (size_t)off * HCD, w, got, T, inject);
                    launches++;
                    const auto l = got.low.get(), i = got.inj.get();
                    const auto y = got.y.get();
                    size_t bad = diff(rl, (size_t)off * RANK, l, 0, (size_t)T * RANK) +
                                 diff(ry, (size_t)off * H, y, 0, (size_t)T * H) +
                                 (inject ? diff(ri, (size_t)off * HC, i, 0, (size_t)T * HC) : 0);
                    if (bad) {
                        printf("  MISMATCH T=%u off=%u x%g %s: %zu bytes-groups\n", T, off, scale,
                               inject ? "inject" : "head", bad);
                        g_fail++;
                    }
                }
            }
        }
    }
    // STAGE_FIT is geometry only: `normed` bytes at every split equal split 8's.
    for (unsigned T : {1u, 7u, 16u, 25u, 32u}) {
        std::vector<float> base;
        for (unsigned sp : {8u, 4u, 2u, 1u}) {
            got.poison();
            hc_pre_stage_vec<<<dim3(T, sp), 1024>>>(streams.p, w.norm, got.normed.p, H, HC, EPS);
            auto n = got.normed.get();
            n.resize((size_t)T * HCD);
            if (base.empty()) base = n;
            else if (diff(base, 0, n, 0, n.size())) { printf("  MISMATCH stage T=%u split %u\n", T, sp); g_fail++; }
        }
    }
    printf("  %s  row invariance: %zu batched launches (T = 1..32, 3 offsets, 3 scales, inject/head) vs every row alone\n",
           g_fail ? "BAD" : "ok ", launches);
    // Accuracy against FP64 of the same `normed`: both arms.
    std::vector<float> hs((size_t)ROWS * HCD);
    for (auto& v : hs) v = nd(g_rng);
    streams.put(hs);
    for (int arm = 0; arm < 2; arm++) accuracy(arm, s, w, got, streams, MAXT);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
}

// `sweep`: the mma arm's down and finish only, for -D geometry sweeps.
static void time_sites(bool sweep = false) {
    Site s;
    make_site(s, COPIES);
    Dev<float> streams;
    streams.alloc((size_t)ROWS * HCD);
    Out o;
    o.alloc();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int iters = 96;
    auto timed = [&](auto&& body) {
        float best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {  // first rep warms; best of the rest
            float ms = 0;
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) body(copy_of(s, i % COPIES));
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (rep) best = std::min(best, ms);
        }
        return best * 1e3f / iters;
    };
    if (sweep) {
        printf("dn WM%u WK%u CL%u U%u DN%u  fn DW%u WK%u U%u |", HCM_DN_WM, HCM_DN_WK, HCM_DN_CL, HCM_DN_U, HCM_DN_DN,
               HCM_FN_DW, HCM_FN_WK, HCM_FN_U);
        // `HCM_WARM=1`: one weight copy, so every launch after the first reads L2.
        const bool warm = getenv("HCM_WARM") != nullptr;
        for (unsigned T : {1u, 8u, 16u, 32u}) {
            const float d = timed([&](const W& w) { mma_chain(streams.p, warm ? copy_of(s, 0) : w, o, T, true, 2); });
            const float f = timed([&](const W& w) { mma_chain(streams.p, warm ? copy_of(s, 0) : w, o, T, true, 3); });
            printf("  T%-2u %5.1f %5.1f", T, d, f);
        }
        printf("\n");
        return;
    }
    // Arms: FP32 as served (fixed stage split), FP32 + STAGE_FIT, mma +
    // STAGE_FIT, and the last with PDL launches of down + finish.
    const char* names[4] = {"fp32       ", "fp32 fit   ", "mma fit    ", "mma fit pdl"};
    printf("us per launch, weights cold (inject on):\n  T   arm           site   stage    down  finish\n");
    for (unsigned T : {1u, 4u, 8u, 16u, 24u, 32u}) {
        for (int a = 0; a < 4; a++) {
            g_fit = a > 0;
            g_pdl = a == 3;
            float us[4];
            for (int part = 0; part < 4; part++)
                us[part] = timed([&](const W& w) { chain(a >= 2, streams.p, w, o, T, true, part); });
            printf("  %2u  %s  %6.1f  %6.1f  %6.1f  %6.1f\n", T, names[a], us[0], us[1], us[2], us[3]);
        }
    }
    g_fit = false;
    g_pdl = 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "check";
    CK(cudaFree(0));
    CK(cudaFuncSetAttribute(hc_mma_finish, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024));
    if (mode == "check") {
        check();
        return g_fail ? 1 : 0;
    }
    if (mode == "prof") {
        const unsigned T = argc > 2 ? atoi(argv[2]) : 32;
        const int arm = argc > 3 ? atoi(argv[3]) : 1;
        Site s;
        make_site(s, COPIES);
        Dev<float> streams;
        streams.alloc((size_t)ROWS * HCD);
        Out o;
        o.alloc();
        for (int i = 0; i < 32; i++) chain(arm, streams.p, copy_of(s, i % COPIES), o, T, true);
        CK(cudaDeviceSynchronize());
        return 0;
    }
    time_sites(mode == "sweep");
    return 0;
}
