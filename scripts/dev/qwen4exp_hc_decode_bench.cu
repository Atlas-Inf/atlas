// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check and A/B timing of the Qwen3.8-Flash-Next decode
// mHC collapse (ATLAS_QWEN4EXP_HC_FAST) against the kernels it replaces, on
// the real site shape (hidden 2560, hc 4, rank 320) at T = 1..8 rows (T > 4:
// multi-sequence batches, ATLAS_QWEN4EXP_BATCH_FAST):
//
//   stage   hc_pre_stage       (T) x 1024     vs hc_pre_stage_vec  (T, S) x 1024
//   down    hc_pre_down        (T, 48/T) x 1024, 40 KB shared
//                                             vs hc_pre_down_vec  (+ the hc injection rows;
//                                                hc_pre_down_vec8 at T = 5..8)
//   finish  hc_pre_finish_x4   (T, H/32) x 128 (+ injection in block 0)
//                                             vs hc_pre_finish_vec (hc_pre_finish_vec8 at T = 5..8)
//   post    hc_post            (T) x 256      vs hc_post_vec
//   mt      hc_pre_down_mt + hc_pre_finish_x4_mt (ATLAS_HC_MT, T <= 4), for reference
//
// Every output byte (normed, low, inj, y, post) must match between the
// default chain, the MT chain and the vectorized chain, with and without the
// injection rows (hc_pre vs hc_head), over random inputs at three scales that
// include zero and -0.0 entries; and every row of a T = 2..8 vectorized
// launch must equal the T = 1 vectorized kernels run on that row alone. It
// then times each kernel and each whole
// site (stage, down, finish, post) as serving launches them, cycling `copies`
// weight sets (13.1 MB a site, so 8 copies are 4x the 24 MiB L2 and every
// launch streams from DRAM, as serving's 96 distinct sites do).
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 \
//        -I kernels/gb10/qwen3.8-flash-next/nvfp4 \
//        scripts/dev/qwen4exp_hc_decode_bench.cu -o qwen4exp_hc_decode_bench
//   ./qwen4exp_hc_decode_bench [check|sweep] [copies=8] [groups=48] [reps=3] \
//        [down_block=128] [fin_block=128] [stage_split=8] [post_block=64]
//
// `check` prints one "bitwise" line per comparison, then PASS or FAIL (exit
// 1), then the median GPU time per launch in microseconds: stream events
// around every 8 launches, each batch queued behind a spin kernel so the GPU
// never waits for the host (so the numbers include the back-to-back launch
// gap). `sweep` bit-checks and times every templated down / finish shape
// (values per thread x unroll x block). Device memory: ~110 MB at 8 copies.
//
// Measured 2026-10-05 (ennspark03, atlas-release-builder:1.93.1, CUDA 13.0),
// us per site (stage + down + finish + post), old -> vec: T=1 124 -> 78,
// T=4 123 -> 85, T=5 137 -> 90, T=6 137 -> 94, T=8 132 -> 105; PASS.
#include "hyper_connection.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H, MAXT = 8, MT_MAXT = 4;
static const float EPS = 1e-6f;

static unsigned g_down_block = 128, g_fin_block = 128, g_stage_split = 8, g_post_block = 64;

static unsigned short tobf(float f) {
    bf b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}

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

struct Weights {
    Dev<unsigned short> norm_w, down_w, up_w, inject_w;
    int copies = 0;
    const bf* norm(int c) const { return (const bf*)norm_w.p + (size_t)(c % copies) * HCD; }
    const bf* down(int c) const { return (const bf*)down_w.p + (size_t)(c % copies) * RANK * HCD; }
    const bf* up(int c) const { return (const bf*)up_w.p + (size_t)(c % copies) * RANK * HCD; }
    const bf* inject(int c) const { return (const bf*)inject_w.p + (size_t)(c % copies) * HC * HCD; }
};

// One arm's buffers.
struct Arm {
    Dev<float> normed, low, inj, post_out;
    Dev<unsigned short> y;
    void alloc() {
        normed.alloc((size_t)MAXT * HCD); low.alloc((size_t)MAXT * RANK); inj.alloc(MAXT * HC);
        post_out.alloc((size_t)MAXT * HCD); y.alloc((size_t)MAXT * H);
    }
    void poison() { normed.fill(0x7F); low.fill(0x7F); inj.fill(0x7F); post_out.fill(0x7F); y.fill(0x7F); }
};

struct Inputs {
    Dev<float> streams, post_inj;
    Dev<unsigned short> block_out;
};

// ── templated shapes for the sweep ──
typedef void (*DownK)(const float*, const bf*, const bf*, float*, float*, unsigned, unsigned, unsigned, unsigned);
typedef void (*FinK)(const float*, const float*, const bf*, bf*, unsigned, unsigned, unsigned);

// One sweep shape covers T = 1..8: the T <= 4 template (hc_pre_*_vec) or the
// 5..8 one (hc_pre_*_vec8), as the dispatcher picks.
template <unsigned CPT, unsigned U, unsigned DN, unsigned DN8>
__global__ void bench_down(const float* normed, const bf* down_w, const bf* inject_w, float* low,
                           float* inj, unsigned hs, unsigned hc, unsigned rank, unsigned nt) {
    if (nt <= 4) qhc_down_vec_t<CPT, U, DN>(normed, down_w, inject_w, low, inj, hs, hc, rank, nt);
    else qhc_down_vec_t8<CPT, U, DN8>(normed, down_w, inject_w, low, inj, hs, hc, rank, nt);
}
template <unsigned DPT, unsigned U, unsigned U8>
__global__ void bench_fin(const float* normed, const float* low, const bf* up_w, bf* y,
                          unsigned hs, unsigned rank, unsigned nt) {
    if (nt <= 4) qhc_finish_vec_t<DPT, U>(normed, low, up_w, y, hs, rank, nt);
    else qhc_finish_vec_t8<DPT, U8>(normed, low, up_w, y, hs, rank, nt);
}

// Down: chains per thread, weight ring depth, normed ring depth at T <= 4
// and at T = 5..8.
struct DownCfg { const char* name; DownK k; unsigned cpt; };
struct FinCfg { const char* name; FinK k; unsigned dpt; };
static const DownCfg DOWN_CFGS[] = {
    {"c2 u32 n16/2", bench_down<2, 32, 16, 2>, 2}, {"c2 u32 n16/4", bench_down<2, 32, 16, 4>, 2},
    {"c2 u32 n16/8", bench_down<2, 32, 16, 8>, 2}, {"c2 u16 n8/4", bench_down<2, 16, 8, 4>, 2},
    {"c4 u16 n8/2", bench_down<4, 16, 8, 2>, 4},   {"c4 u16 n8/4", bench_down<4, 16, 8, 4>, 4},
};
// Finish: values per thread, ring depth at T <= 4 and at T = 5..8.
static const FinCfg FIN_CFGS[] = {
    {"dpt4 u32/16", bench_fin<4, 32, 16>, 4}, {"dpt4 u32/32", bench_fin<4, 32, 32>, 4},
    {"dpt4 u32/8", bench_fin<4, 32, 8>, 4},   {"dpt8 u16/16", bench_fin<8, 16, 16>, 8},
    {"dpt2 u32/32", bench_fin<2, 32, 32>, 2},
};

static void launch_down(DownK k, unsigned cpt, unsigned block, const Weights& w, Arm& a, unsigned T, int c,
                        bool inject) {
    const unsigned threads = (RANK + (inject ? HC : 0)) * (32 / cpt);
    k<<<(threads + block - 1) / block, block>>>(
        a.normed.p, w.down(c), inject ? w.inject(c) : nullptr, a.low.p, a.inj.p, H, HC, RANK, T);
}
static void launch_fin(FinK k, unsigned dpt, unsigned block, const Weights& w, Arm& a, unsigned T, int c) {
    const unsigned threads = 4 * H / dpt;
    k<<<(threads + block - 1) / block, block, T * RANK * 4>>>(a.normed.p, a.low.p, w.up(c), (bf*)a.y.p, H, RANK, T);
}

// ── the three chains ──
static void old_stage(const Inputs& in, const Weights& w, Arm& a, unsigned T, int c) {
    hc_pre_stage<<<T, 1024>>>(in.streams.p, w.norm(c), a.normed.p, H, HC, EPS);
}
static void old_down(const Weights& w, Arm& a, unsigned T, int c) {
    const unsigned dsplit = std::min(std::max(48u / T, 1u), 10u);
    hc_pre_down<<<dim3(T, dsplit), 1024, HCD * 4>>>(a.normed.p, w.down(c), a.low.p, H, HC, RANK, T);
}
static void old_finish(const Weights& w, Arm& a, unsigned T, int c, bool inject) {
    hc_pre_finish_x4<<<dim3(T, H / 32), 128, RANK * 4>>>(
        a.normed.p, a.low.p, w.up(c), inject ? w.inject(c) : nullptr, (bf*)a.y.p, a.inj.p, H, HC, RANK);
}
static void mt_down(const Weights& w, Arm& a, unsigned T, int c) {
    hc_pre_down_mt<<<RANK / 4, 128>>>(a.normed.p, w.down(c), a.low.p, H, HC, RANK, T);
}
static void mt_finish(const Weights& w, Arm& a, unsigned T, int c, bool inject) {
    hc_pre_finish_x4_mt<<<H / 32, 128, T * RANK * 4>>>(
        a.normed.p, a.low.p, w.up(c), inject ? w.inject(c) : nullptr, (bf*)a.y.p, a.inj.p, H, HC, RANK, T);
}
static void old_post(const Inputs& in, Arm& a, unsigned T) {
    hc_post<<<T, 256>>>((const bf*)in.block_out.p, in.streams.p, in.post_inj.p, a.post_out.p, H, HC);
}

static void vec_stage(const Inputs& in, const Weights& w, Arm& a, unsigned T, int c) {
    hc_pre_stage_vec<<<dim3(T, g_stage_split), 1024>>>(in.streams.p, w.norm(c), a.normed.p, H, HC, EPS);
}
static void vec_down(const Weights& w, Arm& a, unsigned T, int c, bool inject) {
    launch_down(T <= 4 ? hc_pre_down_vec : hc_pre_down_vec8, HC_V_DOWN_CPT, g_down_block, w, a, T, c,
                inject);
}
static void vec_finish(const Weights& w, Arm& a, unsigned T, int c) {
    launch_fin(T <= 4 ? hc_pre_finish_vec : hc_pre_finish_vec8, HC_V_FIN_DPT, g_fin_block, w, a, T, c);
}
static void vec_post(const Inputs& in, Arm& a, unsigned T) {
    hc_post_vec<<<dim3(T, (H / 4 + g_post_block - 1) / g_post_block), g_post_block>>>(
        (const bf*)in.block_out.p, in.streams.p, in.post_inj.p, a.post_out.p, H, HC);
}

enum Chain { OLD, MT, VEC };
static void site(const Inputs& in, const Weights& w, Arm& a, unsigned T, int c, Chain ch, bool inject) {
    if (ch == VEC) {
        vec_stage(in, w, a, T, c);
        vec_down(w, a, T, c, inject);
        vec_finish(w, a, T, c);
    } else {
        old_stage(in, w, a, T, c);
        if (ch == MT) { mt_down(w, a, T, c); mt_finish(w, a, T, c, inject); }
        else { old_down(w, a, T, c); old_finish(w, a, T, c, inject); }
    }
}

template <typename T>
static bool same(const char* what, unsigned rows, const Dev<T>& a, const Dev<T>& b, size_t per_row, bool quiet = false) {
    std::vector<T> ha = a.get(), hb = b.get();
    const size_t n = rows * per_row;
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&ha[i], &hb[i], sizeof(T)) != 0) { if (first == (size_t)-1) first = i; bad++; }
    }
    if (!quiet || bad) {
        printf("  bitwise %-28s %s", what, bad ? "MISMATCH" : "ok");
        if (bad) printf("  (%zu of %zu differ, first at %zu)", bad, n, first);
        printf("\n");
    }
    return bad == 0;
}

extern "C" __global__ void bench_spin(unsigned int* sink, unsigned int rounds) {
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < rounds; i++) x = x * 1664525u + 1013904223u;
    if (threadIdx.x == 0) sink[0] = x;
}
extern "C" __global__ void bench_empty() {}

// Bandwidth reference: stream `n16` 16-byte words once (the down_w footprint)
// with a grid-stride loop, 8 loads in flight per thread.
extern "C" __global__ void bench_stream(const uint4* __restrict__ p, size_t n16, unsigned int* sink) {
    unsigned int x = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 7 * stride < n16; i += 8 * stride) {
        uint4 v[8];
        #pragma unroll
        for (int u = 0; u < 8; u++) v[u] = __ldcs(p + i + u * stride);
        #pragma unroll
        for (int u = 0; u < 8; u++) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n16; i += stride) { const uint4 v = p[i]; x ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (x == 0x9E3779B9u) sink[0] = x;
}

// Median GPU time of one launch (or chain) in microseconds over `groups` x 8
// per rep, stream events around every 8.
template <typename F>
static double time_us(int groups, int reps, unsigned int* sink, F&& chain) {
    const int per = 8;
    static std::vector<cudaEvent_t> ev;
    while ((int)ev.size() < 2 * groups) { cudaEvent_t event; CK(cudaEventCreate(&event)); ev.push_back(event); }
    std::vector<float> us;
    for (int r = 0; r < reps; r++) {
        CK(cudaDeviceSynchronize());
        bench_spin<<<1, 32>>>(sink, 4000000u);
        for (int g = 0; g < groups; g++) {
            CK(cudaEventRecord(ev[2 * g]));
            for (int i = 0; i < per; i++) chain((r * groups + g) * per + i);
            CK(cudaEventRecord(ev[2 * g + 1]));
        }
        CK(cudaDeviceSynchronize());
        for (int g = 0; g < groups; g++) {
            float ms;
            CK(cudaEventElapsedTime(&ms, ev[2 * g], ev[2 * g + 1]));
            us.push_back(ms * 1000.f / per);
        }
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

// Fresh random inputs for one trial: streams at `scale` with 0.2% zeros and
// 0.2% -0.0, injection scalars in [0, 2), a unit-normal block output.
static void make_inputs(Inputs& in, std::mt19937& rng, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> hs((size_t)MAXT * HCD);
    for (size_t i = 0; i < hs.size(); i++) {
        const float u = ud(rng);
        hs[i] = u < 0.002f ? 0.f : u < 0.004f ? -0.f : scale * nd(rng);
    }
    in.streams.put(hs);
    std::vector<float> hi(MAXT * HC);
    for (auto& v : hi) v = 2.f * ud(rng);
    in.post_inj.put(hi);
    std::vector<unsigned short> hb((size_t)MAXT * H);
    for (auto& v : hb) v = tobf(nd(rng));
    in.block_out.put(hb);
}

static const float SCALES[3] = {1.f, 30.f, 1e-3f};

int main(int argc, char** argv) {
    const bool sweep = argc > 1 && std::string(argv[1]) == "sweep";
    const int copies = argc > 2 ? atoi(argv[2]) : 8;
    const int groups = argc > 3 ? atoi(argv[3]) : 48;
    const int reps = argc > 4 ? atoi(argv[4]) : 3;
    if (argc > 5) g_down_block = (unsigned)atoi(argv[5]);
    if (argc > 6) g_fin_block = (unsigned)atoi(argv[6]);
    if (argc > 7) g_stage_split = (unsigned)atoi(argv[7]);
    if (argc > 8) g_post_block = (unsigned)atoi(argv[8]);
    printf("copies %d  down cpt %u u %u dn %u/%u block %u  fin dpt %u u %u/%u block %u  stage_split %u  "
           "post_block %u\n", copies, HC_V_DOWN_CPT, HC_V_DOWN_UNROLL, HC_V_DOWN_DN, HC_V_DOWN_DN8,
           g_down_block, HC_V_FIN_DPT, HC_V_FIN_UNROLL, HC_V_FIN_UNROLL8, g_fin_block, g_stage_split,
           g_post_block);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    Weights w;
    w.copies = copies;
    w.norm_w.alloc((size_t)copies * HCD);
    w.down_w.alloc((size_t)copies * RANK * HCD);
    w.up_w.alloc((size_t)copies * RANK * HCD);
    w.inject_w.alloc((size_t)copies * HC * HCD);
    {
        // Checkpoint-like magnitudes; a few exact zeros and -0.0 in the weights.
        auto gen = [&](size_t n, float sd) {
            std::vector<unsigned short> h(n);
            for (size_t i = 0; i < n; i++) {
                const float u = ud(rng);
                h[i] = u < 0.001f ? 0 : u < 0.002f ? 0x8000 : tobf(sd * nd(rng));
            }
            return h;
        };
        w.norm_w.put(gen(w.norm_w.n, 0.1f));
        w.down_w.put(gen(w.down_w.n, 0.02f));
        w.up_w.put(gen(w.up_w.n, 0.05f));
        w.inject_w.put(gen(w.inject_w.n, 0.02f));
    }

    Inputs in;
    in.streams.alloc((size_t)MAXT * HCD);
    in.post_inj.alloc(MAXT * HC);
    in.block_out.alloc((size_t)MAXT * H);
    Arm arm[3];
    for (auto& a : arm) a.alloc();
    unsigned int* sink;
    CK(cudaMalloc(&sink, 4));

    if (sweep) {
        // Bit-check every shape against the default chain, then time it.
        bool ok = true;
        const unsigned blocks[2] = {64, 128};
        printf("\n%-16s %5s |", "down", "block");
        for (unsigned T = 1; T <= MAXT; T++) printf("   T=%u   ", T);
        printf("  (us per launch)\n");
        for (const DownCfg& cfg : DOWN_CFGS) {
            for (unsigned b : blocks) {
                for (int trial = 0; trial < 3; trial++) {
                    make_inputs(in, rng, SCALES[trial]);
                    for (unsigned T = 1; T <= MAXT; T++) {
                        for (int inject = 1; inject >= 0; inject--) {
                            arm[0].poison();
                            arm[2].poison();
                            old_stage(in, w, arm[0], T, trial);
                            old_stage(in, w, arm[2], T, trial);
                            old_down(w, arm[0], T, trial);
                            old_finish(w, arm[0], T, trial, inject);
                            launch_down(cfg.k, cfg.cpt, b, w, arm[2], T, trial, inject);
                            CK(cudaDeviceSynchronize());
                            ok &= same(cfg.name, T, arm[0].low, arm[2].low, RANK, true);
                            if (inject) ok &= same(cfg.name, T, arm[0].inj, arm[2].inj, HC, true);
                        }
                    }
                }
                printf("%-16s %5u |", cfg.name, b);
                for (unsigned T = 1; T <= MAXT; T++) {
                    printf(" %8.1f", time_us(groups, reps, sink, [&](int i) {
                        launch_down(cfg.k, cfg.cpt, b, w, arm[2], T, i, true); }));
                }
                printf("\n");
            }
        }
        printf("\n%-16s %5s |", "finish", "block");
        for (unsigned T = 1; T <= MAXT; T++) printf("   T=%u   ", T);
        printf("\n");
        for (const FinCfg& cfg : FIN_CFGS) {
            for (unsigned b : blocks) {
                for (int trial = 0; trial < 3; trial++) {
                    make_inputs(in, rng, SCALES[trial]);
                    for (unsigned T = 1; T <= MAXT; T++) {
                        arm[0].poison();
                        old_stage(in, w, arm[0], T, trial);
                        old_down(w, arm[0], T, trial);
                        CK(cudaMemcpy(arm[2].normed.p, arm[0].normed.p, arm[0].normed.n * 4, cudaMemcpyDeviceToDevice));
                        CK(cudaMemcpy(arm[2].low.p, arm[0].low.p, arm[0].low.n * 4, cudaMemcpyDeviceToDevice));
                        arm[2].y.fill(0x7F);
                        old_finish(w, arm[0], T, trial, true);
                        launch_fin(cfg.k, cfg.dpt, b, w, arm[2], T, trial);
                        CK(cudaDeviceSynchronize());
                        ok &= same(cfg.name, T, arm[0].y, arm[2].y, H, true);
                    }
                }
                printf("%-16s %5u |", cfg.name, b);
                for (unsigned T = 1; T <= MAXT; T++) {
                    printf(" %8.1f", time_us(groups, reps, sink, [&](int i) {
                        launch_fin(cfg.k, cfg.dpt, b, w, arm[2], T, i); }));
                }
                printf("\n");
            }
        }
        printf("empty kernel launch: %.1f us\n", time_us(groups, reps, sink, [&](int) { bench_empty<<<1, 32>>>(); }));
        for (unsigned grid : {48u, 96u, 192u, 384u, 768u}) {
            const double us = time_us(groups, reps, sink, [&](int i) {
                bench_stream<<<grid, 256>>>((const uint4*)w.down(i), (size_t)RANK * HCD / 8, sink); });
            printf("stream 6.55 MB, grid %4u x 256: %6.1f us  %6.1f GB/s\n", grid, us, RANK * HCD * 2 / us / 1e3);
        }
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    bool ok = true;
    const char* names[3] = {"default", "mt", "vec"};
    for (int trial = 0; trial < 3; trial++) {
        make_inputs(in, rng, SCALES[trial]);
        for (unsigned T = 1; T <= MAXT; T++) {
            for (int inject = 1; inject >= 0; inject--) {
                printf("trial %d (scale %g)  T=%u  %s\n", trial, SCALES[trial], T, inject ? "hc_pre" : "hc_head");
                const int c = trial % copies;
                for (int ch = 0; ch < 3; ch++) {
                    arm[ch].poison();
                    if (ch != MT || T <= MT_MAXT) site(in, w, arm[ch], T, c, (Chain)ch, inject);
                }
                old_post(in, arm[0], T);
                vec_post(in, arm[2], T);
                CK(cudaDeviceSynchronize());
                for (int ch = T <= MT_MAXT ? 1 : 2; ch < 3; ch++) {
                    char what[64];
                    snprintf(what, sizeof what, "normed  %s", names[ch]);
                    ok &= same(what, T, arm[0].normed, arm[ch].normed, HCD);
                    snprintf(what, sizeof what, "low     %s", names[ch]);
                    ok &= same(what, T, arm[0].low, arm[ch].low, RANK);
                    if (inject) {
                        snprintf(what, sizeof what, "inj     %s", names[ch]);
                        ok &= same(what, T, arm[0].inj, arm[ch].inj, HC);
                    }
                    snprintf(what, sizeof what, "y       %s", names[ch]);
                    ok &= same(what, T, arm[0].y, arm[ch].y, H);
                }
                ok &= same("post    vec", T, arm[0].post_out, arm[2].post_out, HCD);
            }
        }
    }
    // Row t of a T-row vec launch against the T = 1 vec kernels on row t
    // alone (what single-sequence decode runs), every output byte.
    for (int trial = 0; trial < 3; trial++) {
        make_inputs(in, rng, SCALES[trial]);
        for (unsigned T = 2; T <= MAXT; T++) {
            for (int inject = 1; inject >= 0; inject--) {
                arm[2].poison();
                site(in, w, arm[2], T, trial, VEC, inject);
                const auto nT = arm[2].normed.get();
                const auto lT = arm[2].low.get();
                const auto iT = arm[2].inj.get();
                const auto yT = arm[2].y.get();
                size_t bad = 0;
                for (unsigned t = 0; t < T; t++) {
                    Inputs one = in;
                    one.streams.p = in.streams.p + (size_t)t * HCD;
                    arm[1].poison();
                    site(one, w, arm[1], 1, trial, VEC, inject);
                    const auto n1 = arm[1].normed.get();
                    const auto l1 = arm[1].low.get();
                    const auto i1 = arm[1].inj.get();
                    const auto y1 = arm[1].y.get();
                    bad += memcmp(&nT[(size_t)t * HCD], &n1[0], HCD * 4) != 0;
                    bad += memcmp(&lT[(size_t)t * RANK], &l1[0], RANK * 4) != 0;
                    if (inject) bad += memcmp(&iT[(size_t)t * HC], &i1[0], HC * 4) != 0;
                    bad += memcmp(&yT[(size_t)t * H], &y1[0], H * 2) != 0;
                }
                printf("  bitwise rows of T=%u vs T=1 per row (%s)  %s\n", T, inject ? "hc_pre" : "hc_head",
                       bad ? "MISMATCH" : "ok");
                ok &= bad == 0;
            }
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    printf("\nus per launch (median of %d x %d groups of 8, %d weight copies; empty launch %.1f us)\n",
           reps, groups, copies, time_us(groups, reps, sink, [&](int) { bench_empty<<<1, 32>>>(); }));
    printf("%-3s %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %8s %8s\n", "T",
           "stage", "down", "finish", "post", "stage_v", "down_v", "fin_v", "post_v",
           "site", "site_mt", "site_v", "speedup");
    for (unsigned T = 1; T <= MAXT; T++) {
        Arm& o = arm[0];
        Arm& v = arm[2];
        const double s0 = time_us(groups, reps, sink, [&](int i) { old_stage(in, w, o, T, i); });
        const double d0 = time_us(groups, reps, sink, [&](int i) { old_down(w, o, T, i); });
        const double f0 = time_us(groups, reps, sink, [&](int i) { old_finish(w, o, T, i, true); });
        const double p0 = time_us(groups, reps, sink, [&](int) { old_post(in, o, T); });
        const double s1 = time_us(groups, reps, sink, [&](int i) { vec_stage(in, w, v, T, i); });
        const double d1 = time_us(groups, reps, sink, [&](int i) { vec_down(w, v, T, i, true); });
        const double f1 = time_us(groups, reps, sink, [&](int i) { vec_finish(w, v, T, i); });
        const double p1 = time_us(groups, reps, sink, [&](int) { vec_post(in, v, T); });
        const double c0 = time_us(groups, reps, sink, [&](int i) { site(in, w, o, T, i, OLD, true); old_post(in, o, T); });
        const double cm = T > MT_MAXT ? 0.0 : time_us(groups, reps, sink, [&](int i) { site(in, w, arm[1], T, i, MT, true); old_post(in, arm[1], T); });
        const double c1 = time_us(groups, reps, sink, [&](int i) { site(in, w, v, T, i, VEC, true); vec_post(in, v, T); });
        printf("%-3u %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f %7.2fx\n",
               T, s0, d0, f0, p0, s1, d1, f1, p1, c0, cm, c1, c0 / c1);
    }
    return 0;
}
