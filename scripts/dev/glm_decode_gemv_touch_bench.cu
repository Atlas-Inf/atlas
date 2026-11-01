// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the GLM verify-step NVFP4 / MXFP8 GEMVs with and without
// the pre-wait weight touch (ATLAS_GLM_DECODE_GEMV_BATCH=1):
//   w4a16_gemv_tc{8,16,32} / _ld      vs  w4a16_gemv_tc{8,16,32}_touch
//   mxfp8_gemv_tc{8,16,32}            vs  mxfp8_gemv_tc{8,16,32}_touch
//   w4a16_gemv_batch2 / _batch3       vs  w4a16_gemv_batch2_touch / _batch3_touch
//   w4a16_gemv_batch5_qkv             vs  w4a16_gemv_batch5_qkv_touch
//   shared gate + up tc{8,16,32}      vs  w4a16_gemv_tc{8,16,32}_pair_touch (one launch)
//
// 1. Bitwise: every output of every projection shape of a KDA layer (q/k/v/o
//    4096x4096, shared gate/up 1024x4096, the K-slice shared down 4096x1024)
//    and of an MLA layer (q_a 1536x4096, kv_a 512x4096, q_b 8192x1536,
//    o 4096x8192), rows M = 1..32 (tc8/16/32 tiers), three touch geometries. Must print
//    "bitwise: 0 of N outputs differ".
// 2. Roofline: each shape alone, back to back, cold weights: us and GB/s.
// 3. Layer time: one emulated layer = its GEMVs in production order and launch
//    mode, each group behind a latency-bound `spin` standing in for the small
//    kernels it waits on, and a plain launch where production restarts the
//    chain after a collective. Windows are the median pre-wait windows of the
//    2026-09-30 nsys profiles (rank 0, c1): up to 5 rows the prose ones (45 us
//    before KDA q, 28 before o, 31 before the shared gate, 50 before MLA q_a),
//    from 6 rows the code ones (8 rows: 30 / 37 / 31 / 30). The spin makes no
//    DRAM traffic; in production the o window overlaps the KDA recurrence,
//    which streams ~1 MB of state, so the o saving here is an upper bound.
//    Weights cycle over `layers` distinct copies (L2 is 24 MiB), so every read
//    comes from DRAM as in decode. Arms per layer: base (production
//    launches), pdl (the twins with touch_rows = 0: TOUCH_MB=0), touch (the
//    flag: shared gate/up as one pair launch) and, for KDA, "unmerged" (the
//    flag with gate and up as two twins). "saved" = base - arm; the last
//    columns scale it to a step (34 KDA + 11 MLA layers). Pass a 5th
//    argument for a per-segment table.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_decode_gemv_touch_bench.cu -o gemv_touch_bench
//   ./gemv_touch_bench [layers=24] [reps=9] [touch_mb=12] [touch_ctas=32]
// Device memory: ~1.3 GB at the defaults; about a minute on a shared GPU.
// Exit 0 iff the bitwise check passes.
#include "w4a16_gemv.cu"
#include "mxfp8_gemv.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef unsigned char u8;

// Stand-in for the small kernels a GEMV waits on: a chain of `iters` dependent
// loads from a 4 KiB table. Latency-bound like those kernels (the nsys profile
// shows them no slower with a waiting GEMV resident); an ALU-bound loop here
// ran 1.7x slower next to the waiting CTAs and skewed the comparison.
extern "C" __global__ void spin(unsigned int* tab, unsigned int iters) {
    atlas_pdl_enter();
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < iters; i++) x = tab[(x + i) & 1023u];
    if (x == 0xFFFFFFFFu) tab[0] = 0u;  // never: the table holds indices below 1024
}

// Diagnostic: an empty PDL kernel (launch floor) of any grid/block.
extern "C" __global__ void nop(unsigned int* tab) {
    atlas_pdl_enter();
    if (tab == nullptr) return;
}

// Random weight bytes. kind 0: any byte (E2M1 nibbles); 1: E4M3 scales in a
// moderate positive range; 2: E4M3 values without the NaN codes; 3: E8M0.
extern "C" __global__ void fill(u8* p, unsigned long long n, unsigned int kind, unsigned int seed) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        u8 b = (u8)h;
        if (kind == 1) b = 0x20 + (b & 0x1F);
        if (kind == 2 && (b & 0x7F) == 0x7F) b ^= 1;
        if (kind == 3) b = 120 + (b & 7);
        p[i] = b;
    }
}

// `pdl` false is a plain launch (the kernels production launches without PDL).
template <typename... Args>
static void launch(bool pdl, void (*kern)(Args...), dim3 grid, unsigned int block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

// One projection: weight bytes, scale bytes, shape, row strides.
struct Proj { u8 *w, *s; unsigned int n, k, ldw, lds; bool mx; double mb; };

static unsigned int g_touch_ctas = 32;
static unsigned long long g_touch_bytes = 12ull << 20;
static const float SCALE2 = 0.37f;
static bool g_pair = true;  // layer emulation: gate/up as one pair launch

static unsigned int touch_rows(const Proj& p) {
    const unsigned long long row = p.mx ? p.k + p.k / 32 : p.k / 2 + p.k / 16;
    return (unsigned int)std::min<unsigned long long>(p.n, g_touch_bytes / row);
}

// The tensor-core tier for `m` rows (tc8 / tc16 / tc32; production takes it
// at 4..32 rows for KDA, at every row count for the shared expert and MLA).
// Production launches tc8 with PDL and tc16 / tc32 without; the twins all use it.
static void tc(const Proj& p, bool touch, const bf* a, bf* c, unsigned int m) {
    const unsigned int grid = (p.n + 15) / 16;
    const unsigned int ctas = std::min(grid, g_touch_ctas);
    const int tier = m <= 8 ? 0 : (m <= 16 ? 1 : 2);
    if (p.mx) {
        typedef void (*Mx)(const bf*, const u8*, const u8*, bf*, unsigned, unsigned, unsigned, unsigned);
        typedef void (*MxT)(const bf*, const u8*, const u8*, bf*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
        const Mx base[3] = {mxfp8_gemv_tc8, mxfp8_gemv_tc16, mxfp8_gemv_tc32};
        const MxT twin[3] = {mxfp8_gemv_tc8_touch, mxfp8_gemv_tc16_touch, mxfp8_gemv_tc32_touch};
        if (touch) launch(true, twin[tier], dim3(grid), 256u, a, (const u8*)p.w, (const u8*)p.s, c, m, p.n, p.k, p.n, touch_rows(p), ctas);
        else launch(tier == 0, base[tier], dim3(grid), 256u, a, (const u8*)p.w, (const u8*)p.s, c, m, p.n, p.k, p.n);
        return;
    }
    typedef void (*W4)(const bf*, const u8*, const u8*, float, bf*, unsigned, unsigned, unsigned);
    typedef void (*W4Ld)(const bf*, const u8*, const u8*, float, bf*, unsigned, unsigned, unsigned, unsigned, unsigned);
    typedef void (*W4T)(const bf*, const u8*, const u8*, float, bf*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
    const W4 base[3] = {w4a16_gemv_tc8, w4a16_gemv_tc16, w4a16_gemv_tc32};
    const W4Ld base_ld[3] = {w4a16_gemv_tc8_ld, w4a16_gemv_tc16_ld, w4a16_gemv_tc32_ld};
    const W4T twin[3] = {w4a16_gemv_tc8_touch, w4a16_gemv_tc16_touch, w4a16_gemv_tc32_touch};
    if (touch) launch(true, twin[tier], dim3(grid), 256u, a, (const u8*)p.w, (const u8*)p.s, SCALE2, c, m, p.n, p.k, p.ldw, p.lds, touch_rows(p), ctas);
    else if (p.ldw == p.k / 2) launch(tier == 0, base[tier], dim3(grid), 256u, a, (const u8*)p.w, (const u8*)p.s, SCALE2, c, m, p.n, p.k);
    else launch(tier == 0, base_ld[tier], dim3(grid), 256u, a, (const u8*)p.w, (const u8*)p.s, SCALE2, c, m, p.n, p.k, p.ldw, p.lds);
}

// The scalar tier production uses for a KDA projection at 2 or 3 rows.
// mode 0: production (plain launch); 1: PDL twin, no touch; 2: PDL twin + touch.
static void scalar23(const Proj& p, int mode, const bf* a, bf* c, unsigned int m) {
    const dim3 grid((p.n + 3) / 4);
    const unsigned int rows = mode == 2 ? touch_rows(p) : 0u;
    if (mode == 0 && m == 2) launch(false, w4a16_gemv_batch2, grid, 256u, a, (const u8*)p.w, (const u8*)p.s, SCALE2, c, p.n, p.k);
    else if (mode == 0) launch(false, w4a16_gemv_batch3, grid, 256u, a, (const u8*)p.w, (const u8*)p.s, SCALE2, c, p.n, p.k);
    else launch(true, m == 2 ? w4a16_gemv_batch2_touch : w4a16_gemv_batch3_touch, grid, 256u, a, (const u8*)p.w,
                (const u8*)p.s, SCALE2, c, m, p.n, p.k, rows, std::min(grid.x, g_touch_ctas));
}

// p[0] and p[1] (same shape, whole weights) of one input in one pair launch;
// plane i writes c + i*m*n.
static void pair(const Proj* p, const bf* a, bf* c, unsigned int m) {
    typedef void (*Pr)(const bf*, const u8*, const u8*, float, bf*, const u8*, const u8*, float, bf*,
                       unsigned, unsigned, unsigned, unsigned, unsigned);
    const Pr kern[3] = {w4a16_gemv_tc8_pair_touch, w4a16_gemv_tc16_pair_touch, w4a16_gemv_tc32_pair_touch};
    const unsigned int grid = (p[0].n + 15) / 16;
    launch(true, kern[m <= 8 ? 0 : (m <= 16 ? 1 : 2)], dim3(grid, 1, 2), 256u, a, (const u8*)p[0].w,
           (const u8*)p[0].s, SCALE2, c, (const u8*)p[1].w, (const u8*)p[1].s, SCALE2, c + (size_t)m * p[0].n,
           m, p[0].n, p[0].k, touch_rows(p[0]), std::min(grid, g_touch_ctas));
}

// The five-row fused Q/K/V (ATLAS_GLM_K5_FUSED_QKV=1): production is a plain launch.
static void qkv5(const Proj* q, bool touch, const bf* a, bf* c) {
    const dim3 grid((q[0].n + 3) / 4, 1, 3);
    if (touch) launch(true, w4a16_gemv_batch5_qkv_touch, grid, 256u, a, (const u8*)q[0].w, (const u8*)q[0].s, SCALE2,
                      (const u8*)q[1].w, (const u8*)q[1].s, SCALE2, (const u8*)q[2].w, (const u8*)q[2].s, SCALE2,
                      c, 5u, q[0].n, q[0].k, touch_rows(q[0]), std::min(grid.x, g_touch_ctas));
    else launch(false, w4a16_gemv_batch5_qkv, grid, 256u, a, (const u8*)q[0].w, (const u8*)q[0].s, SCALE2,
                (const u8*)q[1].w, (const u8*)q[1].s, SCALE2, (const u8*)q[2].w, (const u8*)q[2].s, SCALE2,
                c, 5u, q[0].n, q[0].k);
}

static Proj make(unsigned int n, unsigned int k, unsigned int ld_k, bool mx, unsigned int seed) {
    Proj p; p.n = n; p.k = k; p.mx = mx;
    p.ldw = mx ? ld_k : ld_k / 2; p.lds = mx ? ld_k / 32 : ld_k / 16;
    p.mb = (double)n * (mx ? k + k / 32 : k / 2 + k / 16) / 1e6;
    const unsigned long long wb = (unsigned long long)n * p.ldw, sb = (unsigned long long)n * p.lds;
    CK(cudaMalloc(&p.w, wb)); CK(cudaMalloc(&p.s, sb));
    fill<<<1024, 256>>>(p.w, wb, mx ? 2u : 0u, seed);
    fill<<<1024, 256>>>(p.s, sb, mx ? 3u : 1u, seed + 17u);
    return p;
}

enum { Q, KP, V, O, GATE, UP, DOWN };   // KDA layer projections
enum { QA, KVA, QB, MO };               // MLA layer projections

int main(int argc, char** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 24;
    const int reps = argc > 2 ? atoi(argv[2]) : 9;
    if (argc > 3) g_touch_bytes = (unsigned long long)atoi(argv[3]) << 20;
    if (argc > 4) g_touch_ctas = (unsigned int)atoi(argv[4]);
    const unsigned int H = 4096, KMAX = 8192, NMAX = 3 * 4096, MAXM = 32;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> hA((size_t)MAXM * KMAX);
    for (auto& x : hA) { bf b = __float2bfloat16(nd(rng)); memcpy(&x, &b, 2); }
    bf *dA, *dC[2]; unsigned int* dSpin;
    CK(cudaMalloc(&dA, hA.size() * 2));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    for (auto& c : dC) CK(cudaMalloc(&c, (size_t)MAXM * NMAX * 2));
    std::vector<unsigned int> hTab(1024);
    for (auto& x : hTab) x = rng() & 1023u;
    CK(cudaMalloc(&dSpin, 1024 * 4));
    CK(cudaMemcpy(dSpin, hTab.data(), 1024 * 4, cudaMemcpyHostToDevice));

    const int mla_layers = std::max(2, layers / 3);
    std::vector<std::vector<Proj>> kda(layers), mla(mla_layers);
    for (int l = 0; l < layers; l++) {
        for (int i = 0; i < 4; i++) kda[l].push_back(make(H, H, H, false, 1000u * l + i));
        for (int i = 0; i < 2; i++) kda[l].push_back(make(1024, H, H, false, 1000u * l + 10 + i));
        // Shared down: this rank's columns [1024, 2048) of a 2048-wide weight.
        Proj down = make(H, 1024, 2048, false, 1000u * l + 20);
        down.w += 512; down.s += 64;
        kda[l].push_back(down);
    }
    for (int l = 0; l < mla_layers; l++) {
        mla[l].push_back(make(1536, H, H, true, 5000u * l + 1));
        mla[l].push_back(make(512, H, H, true, 5000u * l + 2));
        mla[l].push_back(make(8192, 1536, 1536, true, 5000u * l + 3));
        mla[l].push_back(make(H, 8192, 8192, true, 5000u * l + 4));
    }
    CK(cudaDeviceSynchronize());

    // ── 1. Bitwise ──
    std::vector<unsigned short> o[2] = {std::vector<unsigned short>((size_t)MAXM * NMAX),
                                        std::vector<unsigned short>((size_t)MAXM * NMAX)};
    size_t diff = 0, checked = 0;
    auto compare = [&](size_t elems, const std::function<void(int)>& fire) {
        for (int v = 0; v < 2; v++) {
            CK(cudaMemset(dC[v], v ? 0xEE : 0x11, (size_t)MAXM * NMAX * 2));
            fire(v);
        }
        CK(cudaDeviceSynchronize());
        for (int v = 0; v < 2; v++) CK(cudaMemcpy(o[v].data(), dC[v], elems * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; i++, checked++) diff += o[0][i] != o[1][i];
    };
    const unsigned long long saved_bytes = g_touch_bytes;
    const unsigned int saved_ctas = g_touch_ctas;
    const unsigned long long geo_bytes[3] = {saved_bytes, 3ull << 20, 64ull << 20};
    const unsigned int geo_ctas[3] = {saved_ctas, 7, 100000};
    for (int geo = 0; geo < 3; geo++) {
        g_touch_bytes = geo_bytes[geo]; g_touch_ctas = geo_ctas[geo];
        for (unsigned int m : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 16u, 17u, 25u, 32u}) {
            for (int fam = 0; fam < 2; fam++)
                for (const Proj& p : fam ? mla[m % mla_layers] : kda[m % layers])
                    compare((size_t)m * p.n, [&](int v) { tc(p, v == 1, dA, dC[v], m); });
            {
                const Proj* p = &kda[m % layers][GATE];
                compare((size_t)2 * m * p->n, [&](int v) {
                    if (v) pair(p, dA, dC[v], m);
                    else for (unsigned int i = 0; i < 2; i++) tc(p[i], false, dA, dC[v] + (size_t)i * m * p->n, m);
                });
            }
            const Proj& q = kda[m % layers][Q];
            if (m == 2 || m == 3)
                for (int mode = 1; mode <= 2; mode++)
                    compare((size_t)m * q.n, [&](int v) { scalar23(q, v ? mode : 0, dA, dC[v], m); });
        }
        for (int l = 0; l < 2; l++)
            compare((size_t)3 * 5 * H, [&](int v) { qkv5(&kda[l][Q], v == 1, dA, dC[v]); });
    }
    g_touch_bytes = saved_bytes; g_touch_ctas = saved_ctas;
    printf("bitwise: %zu of %zu outputs differ\n", diff, checked);

    // ── Timing harness ──
    // A pass is queued behind a long `gate` spin, so the GPU runs a pre-queued
    // PDL chain as it does in a verify step (the host is ~9 ms ahead there),
    // and an event after every layer times the GPU alone. Reported: the lower
    // decile over all layers and reps, which survives a time-sliced GPU.
    std::vector<cudaEvent_t> ev(layers + 1);
    for (auto& event : ev) CK(cudaEventCreate(&event));
    auto spin_us = [&](unsigned int iters) { if (iters) launch(true, spin, dim3(64), 128u, dSpin, iters); };
    const unsigned int gate_iters = 200000u;
    auto time_us = [&](int n, const std::function<void(int)>& layer) {
        std::vector<float> us;
        for (int r = 0; r < reps; r++) {
            spin_us(gate_iters);
            CK(cudaEventRecord(ev[0]));
            for (int l = 0; l < n; l++) { layer(l); CK(cudaEventRecord(ev[l + 1])); }
            CK(cudaEventSynchronize(ev[n]));
            for (int l = 0; l < n; l++) {
                float ms; CK(cudaEventElapsedTime(&ms, ev[l], ev[l + 1]));
                us.push_back(ms * 1000.f);
            }
        }
        std::sort(us.begin(), us.end());
        return us[us.size() / 10];
    };

    // ── 2. Roofline ──
    const char* kda_names[] = {"q/k/v/o 4096x4096", "", "", "", "shared gate/up 1024x4096", "", "shared down 4096x1024"};
    const char* mla_names[] = {"q_a 1536x4096", "kv_a 512x4096", "q_b 8192x1536", "mla o 4096x8192"};
    printf("alone, back to back, cold weights (M=5):\n");
    for (int fam = 0; fam < 2; fam++)
        for (int i : fam ? std::vector<int>{QA, KVA, QB, MO} : std::vector<int>{Q, GATE, DOWN}) {
            const auto& net = fam ? mla : kda;
            const int n = (int)net.size();
            const float b = time_us(n, [&](int l) { tc(net[l][i], false, dA, dC[0], 5); });
            const float t = time_us(n, [&](int l) { tc(net[l][i], true, dA, dC[0], 5); });
            printf("  %-26s %6.2f MB  tc8 %7.1f us %6.1f GB/s   touch twin %7.1f us\n",
                   fam ? mla_names[i] : kda_names[i], net[0][i].mb, b, net[0][i].mb / b * 1e3, t);
        }
    {
        const double mb = kda[0][Q].mb;
        const float b3 = time_us(layers, [&](int l) { scalar23(kda[l][Q], 0, dA, dC[0], 3); });
        const float b5 = time_us(layers, [&](int l) { qkv5(&kda[l][Q], false, dA, dC[0]); });
        printf("  %-26s %6.2f MB  batch3 %5.1f us %6.1f GB/s\n", "q 4096x4096 (M=3)", mb, b3, mb / b3 * 1e3);
        printf("  %-26s %6.2f MB  batch5_qkv %5.1f us %6.1f GB/s\n", "q+k+v 3x4096x4096 (M=5)", 3 * mb, b5, 3 * mb / b5 * 1e3);
    }

    // ── 3. Layer time ──
    // Spin iterations whose wait, with a tc8 GEMV resident behind it, is 35 us.
    unsigned int unit = 2000;
    for (int pass = 0; pass < 5; pass++) {
        const float alone = time_us(layers, [&](int l) { tc(kda[l][Q], false, dA, dC[0], 5); });
        const float with = time_us(layers, [&](int l) { spin_us(unit); tc(kda[l][Q], false, dA, dC[0], 5); });
        unit = (unsigned int)(unit * 35.f / std::max(with - alone, 2.f));
    }
    {
        const float solo = time_us(layers, [&](int) { spin_us(unit); });
        const float alone = time_us(layers, [&](int l) { tc(kda[l][Q], false, dA, dC[0], 5); });
        const float with = time_us(layers, [&](int l) { spin_us(unit); tc(kda[l][Q], false, dA, dC[0], 5); });
        printf("35 us stand-in: %u loads; %.1f us by itself, %.1f us in front of a waiting tc8\n",
               unit, solo, with - alone);
    }
    // A plain launch where production has one after a collective (the KDA
    // input add, hc_post_add): the chain restarts from an idle bus there.
    auto barrier = [&] { launch(false, nop, dim3(8), 256u, dSpin); };
    auto sp = [&](unsigned int us) { spin_us(unit * us / 35u); };
    if (argc > 5) {  // diagnostic: segments of the KDA layer, by touch set and CTA count
        const unsigned int m = 8;
        for (unsigned int ctas : {32u, 64u}) {
            g_touch_ctas = ctas;
            printf("touch by %u CTAs\n", ctas);
            for (int mask = 0; mask < 8; mask++) {
                const float t = time_us(layers, [&](int l) { sp(45); tc(kda[l][Q], mask & 1, dA, dC[0], m);
                    tc(kda[l][KP], mask & 2, dA, dC[0], m); tc(kda[l][V], mask & 4, dA, dC[0], m); });
                printf("  [45us] q k v  touch q=%d k=%d v=%d : %6.1f\n", mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, t);
            }
            for (int mask = 0; mask < 2; mask++) {
                const float t = time_us(layers, [&](int l) { tc(kda[l][V], false, dA, dC[0], m); sp(28); tc(kda[l][O], mask & 1, dA, dC[0], m); });
                printf("  v [28us] o    touch o=%d : %6.1f\n", mask & 1, t);
            }
            for (int mask = 0; mask < 8; mask++) {
                const float t = time_us(layers, [&](int l) { tc(kda[l][O], false, dA, dC[0], m); barrier(); sp(31); tc(kda[l][GATE], mask & 1, dA, dC[0], m);
                    tc(kda[l][UP], mask & 2, dA, dC[0], m); sp(1); tc(kda[l][DOWN], mask & 4, dA, dC[0], m); });
                printf("  o (plain nop) [31us] gate up [1] down  touch gate=%d up=%d down=%d : %6.1f\n", mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, t);
            }
            for (int mask = 0; mask < 2; mask++) {
                const float t = time_us(mla_layers, [&](int l) { barrier(); sp(50); tc(mla[l][QA], mask & 1, dA, dC[0], m); sp(2);
                    tc(mla[l][KVA], mask & 1, dA, dC[0], m); });
                printf("  (plain nop) [50us] q_a [2] kv_a  touch both=%d : %6.1f\n", mask & 1, t);
            }
            for (int mask = 0; mask < 2; mask++) {
                const float t = time_us(layers, [&](int l) { barrier(); sp(45); qkv5(&kda[l][Q], mask & 1, dA, dC[0]); });
                printf("  (plain nop) [45us] batch5_qkv  touch=%d : %6.1f\n", mask & 1, t);
            }
            for (int mode = 0; mode < 3; mode++) {
                const float t = time_us(layers, [&](int l) { barrier(); sp(45); scalar23(kda[l][Q], mode == 0 ? 0 : 2, dA, dC[0], 3);
                    for (int i : {KP, V}) scalar23(kda[l][i], mode, dA, dC[0], 3); });
                printf("  (plain nop) [45us] batch3 q k v  mode=%d (0 base, 1 q touch + PDL k v, 2 all touch) : %6.1f\n", mode, t);
            }
        }
        g_touch_ctas = saved_ctas;
        for (int merged = 0; merged < 2; merged++) {
            const float t = time_us(layers, [&](int l) { barrier(); sp(31);
                if (merged) pair(&kda[l][GATE], dA, dC[0], m);
                else { tc(kda[l][GATE], true, dA, dC[0], m); tc(kda[l][UP], true, dA, dC[0], m); } });
            printf("  (plain nop) [31us] gate up at M=8, both touched: %s %6.1f\n",
                   merged ? "one pair launch" : "two tc8 twins  ", t);
        }
        return 0;
    }
    // Pre-wait windows (us) by row count: prose medians up to 5 rows, code
    // (8-row) medians above.
    struct Win { unsigned int q, o, gate, qa; };
    auto win = [](unsigned int m) { return m <= 5 ? Win{45, 28, 31, 50} : Win{30, 37, 31, 30}; };
    // One KDA layer's GEMVs, production launches (touch = false) or the flag's.
    auto kda_layer = [&](int l, unsigned int m, bool touch) {
        const std::vector<Proj>& p = kda[l];
        const Win w = win(m);
        barrier();
        sp(w.q);  // hc_post, HC partial, finalize, norm
        if (m == 5) {
            qkv5(&p[Q], touch, dA, dC[0]);
        } else {
            for (int i : {Q, KP, V}) {
                if (m <= 3) scalar23(p[i], touch ? 2 : 0, dA, dC[0], m);
                else tc(p[i], touch, dA, dC[0], m);
            }
        }
        sp(w.o);  // (beta/f/g projections), pack, conv, recurrent, gated norm
        if (m <= 3) scalar23(p[O], touch ? 2 : 0, dA, dC[0], m);
        else tc(p[O], touch, dA, dC[0], m);
        barrier();
        sp(w.gate);  // HC partial, finalize, norm
        if (touch && g_pair) {
            pair(&p[GATE], dA, dC[0], m);
        } else {
            tc(p[GATE], touch, dA, dC[0], m);
            tc(p[UP], touch, dA, dC[0], m);
        }
        sp(1);  // silu
        tc(p[DOWN], touch, dA, dC[0], m);
    };
    // One MLA layer: q_a and kv_a behind the HC chain; q_b and o (plain
    // predecessors in production, no wait to fill) stay on the tc8 kernels.
    auto mla_layer = [&](int l, unsigned int m, bool touch) {
        const std::vector<Proj>& p = mla[l];
        barrier();
        sp(win(m).qa);
        touch = touch && m <= 16;  // the flag leaves 17..32-row MLA q_a/kv_a alone
        tc(p[QA], touch, dA, dC[0], m);
        sp(2);  // q_a norm
        tc(p[KVA], touch, dA, dC[0], m);
        barrier();
        tc(p[QB], false, dA, dC[0], m);
        barrier();
        tc(p[MO], false, dA, dC[0], m);
    };
    printf("us per emulated layer (lower decile, %d reps); touch %llu MiB by %u CTAs\n",
           reps, g_touch_bytes >> 20, g_touch_ctas);
    const unsigned long long touch_bytes = g_touch_bytes;
    for (unsigned int m : {2u, 3u, 5u, 8u, 16u, 32u}) {
        float k[4], a[3];  // base, pdl (touch_rows = 0), touch, KDA touch with gate/up unmerged
        for (int arm = 0; arm < 4; arm++) {
            g_touch_bytes = arm == 1 ? 0ull : touch_bytes;
            g_pair = arm != 3;
            k[arm] = time_us(layers, [&](int l) { kda_layer(l, m, arm > 0); });
            if (arm < 3) a[arm] = time_us(mla_layers, [&](int l) { mla_layer(l, m, arm > 0); });
        }
        g_touch_bytes = touch_bytes;
        g_pair = true;
        const float step_pdl = (34.f * (k[0] - k[1]) + 11.f * (a[0] - a[1])) / 1e3f;
        const float step_touch = (34.f * (k[0] - k[2]) + 11.f * (a[0] - a[2])) / 1e3f;
        printf("M=%2u  KDA base %6.1f pdl %6.1f touch %6.1f (unmerged %6.1f) | MLA base %6.1f pdl %6.1f touch %6.1f"
               " | saved per step (34 KDA + 11 MLA): pdl %5.2f ms, touch %5.2f ms\n",
               m, k[0], k[1], k[2], k[3], a[0], a[1], a[2], step_pdl, step_touch);
    }
    return diff == 0 ? 0 : 2;
}
