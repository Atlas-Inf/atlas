// SPDX-License-Identifier: AGPL-3.0-only
// Per-row bit parity and cost of the qwen4_exp exact batching kernels
// (ATLAS_QWEN4EXP_BATCH_FAST, crates/spark-model/src/model/qwen4exp_batch_fast.rs)
// against the single-row kernels serial decode runs. Real shapes (hidden 2560,
// routed/shared intermediate 640, top-10 of 512 experts), synthetic NVFP4
// codes/scales and BF16 activations at three scales with zeros and -0.0.
//
//   MoE rows pair   qwen4exp_moe_rows_{plan,gate_up,silu_down} over R rows
//                   vs R launches each of moe_expert_{gate_up,silu_down}_shared
//                   (R = 1..8, 12, 16; independent, overlapping and identical
//                   routes; TP1 tables and EP2-style half-NULL tables; every
//                   output byte, routed and shared; `sweep` adds the silu/down
//                   unit kernel's other shapes, QU_SWEEP)
//   BF16 GEMV       dense_gemv_bf16_batchm M = 2..8 vs dense_gemv_bf16 per row
//                   (GDN qkvz/out_proj shards, LM-head vocab shard)
//   MoE router      w4a16_gemv_batch4 / batch8 M = 2..8 vs w4a16_gemv_sw per row
//   FP8 GDN proj    w8a16_gemv_batch4 / batch16 M = 2..8 vs w8a16_gemv per row
//   shared blend    moe_batched_blend T = R vs T = 1 per row
//
// Then GPU time per launch and per MoE layer, R rows, per-row pair loop vs one
// rows pair, routes drawn over a 512-expert pool (2.76 MB an expert, ~1.4 GB:
// every launch streams from DRAM as serving does), launch gaps included, with
// GB/s of the unique expert bytes each projection must read.
//
// Build/run: scripts/dev/qwen4exp_batch_exact_bench.sh [check|time|sweep]
// (repo root, GB10, inside atlas-release-builder:1.93.1 for the runtime's
// CUDA 13.0).
//
// Measured 2026-10-05 on GB10 (ennspark03): PASS. EP2 tables, us per launch
// (overlap = half of each row's picks drawn from earlier rows, as a verify
// window's rows do), the (row, slot) silu/down of c44a7cbc -> the unit
// silu/down (qwen4exp_moe_rows.cu), and the MoE layer vs the per-row loop:
//                 silu/down            layer: per-row loop -> rows pair
//   R=4 indep   136 ->  93 (209 GB/s)     342 -> 283  1.21x  (was 1.04x)
//   R=4 overlap 124 ->  78               288 -> 213  1.35x  (was 1.15x)
//   R=8 indep   275 -> 174 (201 GB/s)     650 -> 513  1.27x  (was 1.08x)
//   R=8 overlap 236 -> 130               580 -> 362  1.60x  (was 1.27x)
// gate/up is unchanged at 200-249 GB/s (DRAM-bound; see qwen4exp_moe_rows.cu).
// TP1 R=8: loop 1221 -> 960 indep, 1158 -> 631 overlap.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s = nullptr; \
    cuGetErrorString(r, &s); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s ? s : "?"); exit(1); } } while (0)

typedef unsigned long long u64;
static const unsigned H = 2560, I = 640, TOPK = 10, NE = 512, MAXR = 8, MOE_MAXR = 16;
static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(4321);

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
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, 0, args.data(), nullptr));
}

static unsigned short tobf(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
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
static std::vector<unsigned short> rand_bf16(size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<unsigned short> v(n);
    for (auto& x : v) x = tobf(d(g_rng));
    for (size_t i = 0; i < n; i += 97) v[i] = (i / 97) % 2 ? 0x8000 : 0x0000;
    return v;
}
static std::vector<unsigned char> rand_codes(size_t n) {
    std::vector<unsigned char> v(n);
    for (auto& x : v) x = (unsigned char)g_rng();
    return v;
}
// Finite E4M3 scale bytes over the format's range (no NaN).
static std::vector<unsigned char> rand_scales(size_t n) {
    std::vector<unsigned char> v(n);
    for (auto& x : v) {
        unsigned char b = (unsigned char)(g_rng() & 0x7F);
        if (b == 0x7F) b = 0x7E;
        x = b;  // positive scales, as the checkpoint's
    }
    return v;
}

static bool same(const char* what, const std::vector<unsigned char>& a,
                 const std::vector<unsigned char>& b) {
    size_t diff = 0;
    for (size_t i = 0; i + 1 < a.size(); i += 2)
        if (a[i] != b[i] || a[i + 1] != b[i + 1]) diff++;
    if (diff) {
        printf("  MISMATCH %-48s %zu of %zu values\n", what, diff, a.size() / 2);
        g_fail++;
    }
    return diff == 0;
}

// One NVFP4 projection: [N, K/2] packed, [N, K/16] scales, scale2.
struct Proj { unsigned char* packed; unsigned char* scale; float s2; };
static Proj make_proj(unsigned n, unsigned k) {
    Proj p;
    p.packed = dput(rand_codes((size_t)n * k / 2));
    p.scale = dput(rand_scales((size_t)n * k / 16));
    std::uniform_real_distribution<float> d(0.5f, 2.0f);
    p.s2 = d(g_rng) / 448.f;
    return p;
}

struct Tables { u64 *gp, *gs, *upk, *us, *dp, *ds; float *g2, *u2, *d2; };

static Tables make_tables(const std::vector<Proj>& gate, const std::vector<Proj>& up,
                          const std::vector<Proj>& down, bool ep_half) {
    std::vector<u64> gp(NE), gs(NE), upk(NE), us(NE), dp(NE), ds(NE);
    std::vector<float> g2(NE), u2(NE), d2(NE);
    for (unsigned e = 0; e < NE; e++) {
        const unsigned c = e % gate.size();
        const bool remote = ep_half && (e % 2 == 1);
        gp[e] = remote ? 0 : (u64)gate[c].packed;  gs[e] = remote ? 0 : (u64)gate[c].scale;  g2[e] = gate[c].s2;
        upk[e] = remote ? 0 : (u64)up[c].packed;   us[e] = remote ? 0 : (u64)up[c].scale;    u2[e] = up[c].s2;
        dp[e] = remote ? 0 : (u64)down[c].packed;  ds[e] = remote ? 0 : (u64)down[c].scale;  d2[e] = down[c].s2;
    }
    return {dput(gp), dput(gs), dput(upk), dput(us), dput(dp), dput(ds), dput(g2), dput(u2), dput(d2)};
}

// R rows of top-10 distinct experts; `overlap` 1 reuses earlier rows' picks
// half the time, 2 gives every row row 0's picks (shuffled): one unit per
// expert holds all R rows.
static std::vector<unsigned> routes(unsigned rows, int overlap) {
    std::vector<unsigned> ids;
    std::uniform_int_distribution<unsigned> d(0, NE - 1);
    for (unsigned r = 0; r < rows; r++) {
        std::vector<unsigned> row;
        if (overlap == 2 && r > 0) {
            row.assign(ids.begin(), ids.begin() + TOPK);
            std::shuffle(row.begin(), row.end(), g_rng);
        }
        while (row.size() < TOPK) {
            unsigned e = d(g_rng);
            if (overlap == 1 && r > 0 && (g_rng() % 2)) e = ids[(g_rng() % r) * TOPK + g_rng() % TOPK];
            bool dup = false;
            for (unsigned x : row) dup |= (x == e);
            if (!dup) row.push_back(e);
        }
        ids.insert(ids.end(), row.begin(), row.end());
    }
    return ids;
}

struct MoeBufs {
    void *A, *gate, *up, *shg, *shu, *down, *shd, *ids;
};

static void per_row_pair(CUfunction gu, CUfunction sd, const Tables& t, const Proj& sg,
                         const Proj& su, const Proj& sdn, MoeBufs b, unsigned rows) {
    unsigned n_i = I, k_h = H, n_h = H, k_i = I, topk = TOPK;
    for (unsigned r = 0; r < rows; r++) {
        void* A = (char*)b.A + (size_t)r * H * 2;
        void* go = (char*)b.gate + (size_t)r * TOPK * I * 2;
        void* uo = (char*)b.up + (size_t)r * TOPK * I * 2;
        void* ids = (char*)b.ids + (size_t)r * TOPK * 4;
        void* shg = (char*)b.shg + (size_t)r * I * 2;
        void* shu = (char*)b.shu + (size_t)r * I * 2;
        float s2g = sg.s2, s2u = su.s2, s2d = sdn.s2;
        launch(gu, dim3((I + 7) / 8, TOPK + 1, 2), dim3(128),
               {&A, (void*)&t.gp, (void*)&t.gs, (void*)&t.g2, &go, (void*)&t.upk, (void*)&t.us,
                (void*)&t.u2, &uo, &ids, (void*)&sg.packed, (void*)&sg.scale, &s2g, &shg,
                (void*)&su.packed, (void*)&su.scale, &s2u, &shu, &n_i, &k_h, &topk});
        void* dn = (char*)b.down + (size_t)r * TOPK * H * 2;
        void* shd = (char*)b.shd + (size_t)r * H * 2;
        launch(sd, dim3((H + 7) / 8, TOPK + 1, 1), dim3(128),
               {&go, &uo, (void*)&t.dp, (void*)&t.ds, (void*)&t.d2, &dn, &ids, &shg, &shu,
                (void*)&sdn.packed, (void*)&sdn.scale, &s2d, &shd, &n_h, &k_i, &topk},
               I * 4);
    }
}

// The rows pair: plan, gate/up over (row, slot) CTAs, silu/down over unit
// tiles of `sd_tile` outputs (`sd_block` threads, RC rows a chunk) -- the
// production shape, or a QU_SWEEP shape of the down kernel.
struct RowsImpl {
    const char* name;
    CUfunction plan, gu, sd;
    unsigned sd_tile, sd_block, sd_rc;
};
static void* g_order = nullptr;
static void rows_plan(const RowsImpl& m, MoeBufs b, unsigned rows) {
    unsigned slots = rows * TOPK;
    if (!g_order) g_order = dzero(4096);
    launch(m.plan, dim3(1), dim3(256), {(void*)&b.ids, &g_order, &slots});
}
static void rows_gate_up(const RowsImpl& m, const Tables& t, const Proj& sg, const Proj& su,
                         MoeBufs b, unsigned rows) {
    unsigned n_i = I, k_h = H, topk = TOPK, R = rows;
    float s2g = sg.s2, s2u = su.s2;
    launch(m.gu, dim3(I / 8, rows * TOPK + rows, 2), dim3(128),
           {&b.A, (void*)&t.gp, (void*)&t.gs, (void*)&t.g2, (void*)&b.gate, (void*)&t.upk,
            (void*)&t.us, (void*)&t.u2, (void*)&b.up, (void*)&b.ids, &g_order, (void*)&sg.packed,
            (void*)&sg.scale, &s2g, (void*)&b.shg, (void*)&su.packed, (void*)&su.scale, &s2u,
            (void*)&b.shu, &n_i, &k_h, &topk, &R});
}
static void rows_silu_down(const RowsImpl& m, const Tables& t, const Proj& sdn, MoeBufs b,
                           unsigned rows) {
    unsigned n_h = H, k_i = I, topk = TOPK, R = rows;
    float s2d = sdn.s2;
    // Units: one per 8 rows of the shared expert, then at most one per entry.
    launch(m.sd, dim3(H / m.sd_tile, (rows + 7) / 8 + rows * TOPK, 1), dim3(m.sd_block),
           {(void*)&b.gate, (void*)&b.up, (void*)&t.dp, (void*)&t.ds, (void*)&t.d2,
            (void*)&b.down, (void*)&b.ids, &g_order, (void*)&b.shg, (void*)&b.shu,
            (void*)&sdn.packed, (void*)&sdn.scale, &s2d, (void*)&b.shd, &n_h, &k_i, &topk, &R},
           m.sd_tile * (I / 2 + I / 16) + m.sd_rc * I * 4);
}
static void rows_pair(const RowsImpl& m, const Tables& t, const Proj& sg, const Proj& su,
                      const Proj& sdn, MoeBufs b, unsigned rows) {
    rows_plan(m, b, rows);
    rows_gate_up(m, t, sg, su, b, rows);
    rows_silu_down(m, t, sdn, b, rows);
}

// The production pair (QU_SD_TILE / QU_SD_WARPS / QU_SD_RC defaults), then
// the down kernel's QU_SWEEP shapes: tile, warps, rows per chunk.
static std::vector<RowsImpl> rows_impls(bool sweep) {
    const char* M = "qwen4exp_moe_rows";
    CUfunction plan = load(M, "qwen4exp_moe_rows_plan"), gu = load(M, "qwen4exp_moe_rows_gate_up");
    std::vector<RowsImpl> v;
    v.push_back({"rows", plan, gu, load(M, "qwen4exp_moe_rows_silu_down"), 64, 256, 2});
    if (!sweep) return v;
    static const unsigned SD_SHAPES[][3] = {{16, 4, 2}, {32, 4, 2}, {64, 4, 2},
                                            {32, 8, 1}, {32, 8, 4}, {32, 8, 2}};
    for (auto& sh : SD_SHAPES) {
        char fn[64], nm[64];
        snprintf(fn, sizeof fn, "qu_sweep_silu_down_t%u_w%u_r%u", sh[0], sh[1], sh[2]);
        snprintf(nm, sizeof nm, "down %u.%u.%u", sh[0], sh[1], sh[2]);
        v.push_back({strdup(nm), plan, gu, load(M, fn), sh[0], sh[1] * 32, sh[2]});
    }
    return v;
}

static MoeBufs alloc_bufs() {
    MoeBufs b;
    b.A = dzero((size_t)MOE_MAXR * H * 2);
    b.gate = dzero((size_t)MOE_MAXR * TOPK * I * 2);
    b.up = dzero((size_t)MOE_MAXR * TOPK * I * 2);
    b.shg = dzero((size_t)MOE_MAXR * I * 2);
    b.shu = dzero((size_t)MOE_MAXR * I * 2);
    b.down = dzero((size_t)MOE_MAXR * TOPK * H * 2);
    b.shd = dzero((size_t)MOE_MAXR * H * 2);
    b.ids = dzero((size_t)MOE_MAXR * TOPK * 4);
    return b;
}

static void moe_check(const Tables& t, const Proj& sg, const Proj& su, const Proj& sdn,
                      const char* tag) {
    CUfunction gu1 = load("moe_shared_expert_fused", "moe_expert_gate_up_shared");
    CUfunction sd1 = load("moe_shared_expert_fused", "moe_expert_silu_down_shared");
    const std::vector<RowsImpl> impls = rows_impls(true);
    MoeBufs ref = alloc_bufs(), got = alloc_bufs();
    std::vector<int> bad(impls.size(), 0);
    static const char* ROUTES[3] = {"indep", "overlap", "same"};
    for (unsigned rows = 1; rows <= MOE_MAXR; rows += rows < 8 ? 1 : 4) {
        for (int overlap = 0; overlap < 3; overlap++) {
            for (float scale : {0.02f, 1.0f, 30.0f}) {
                auto a = rand_bf16((size_t)rows * H, scale);
                auto ids = routes(rows, overlap);
                CK(cudaMemcpy(ref.A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(ref.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemset(ref.gate, 0x55, (size_t)MOE_MAXR * TOPK * I * 2));
                CK(cudaMemset(ref.down, 0x55, (size_t)MOE_MAXR * TOPK * H * 2));
                per_row_pair(gu1, sd1, t, sg, su, sdn, ref, rows);
                for (const RowsImpl& m : impls) {
                    CK(cudaMemcpy(got.A, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(got.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemset(got.gate, 0x55, (size_t)MOE_MAXR * TOPK * I * 2));
                    CK(cudaMemset(got.up, 0x55, (size_t)MOE_MAXR * TOPK * I * 2));
                    CK(cudaMemset(got.down, 0x55, (size_t)MOE_MAXR * TOPK * H * 2));
                    CK(cudaMemset(got.shg, 0x55, (size_t)MOE_MAXR * I * 2));
                    CK(cudaMemset(got.shu, 0x55, (size_t)MOE_MAXR * I * 2));
                    CK(cudaMemset(got.shd, 0x55, (size_t)MOE_MAXR * H * 2));
                    rows_pair(m, t, sg, su, sdn, got, rows);
                    char what[160];
                    const size_t rk = (size_t)rows * TOPK;
                    struct { const char* n; void* r; void* g; size_t bytes; } outs[] = {
                        {"gate_out", ref.gate, got.gate, rk * I * 2},
                        {"up_out", ref.up, got.up, rk * I * 2},
                        {"shared gate", ref.shg, got.shg, (size_t)rows * I * 2},
                        {"shared up", ref.shu, got.shu, (size_t)rows * I * 2},
                        {"down", ref.down, got.down, rk * H * 2},
                        {"shared down", ref.shd, got.shd, (size_t)rows * H * 2},
                    };
                    for (auto& o : outs) {
                        snprintf(what, sizeof what, "%s %s R=%u %s x%g %s", tag, m.name, rows,
                                 ROUTES[overlap], scale, o.n);
                        bad[&m - &impls[0]] += !same(what, dget(o.r, o.bytes), dget(o.g, o.bytes));
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < impls.size(); i++)
        printf("  %s  %s %-24s R = 1..8, 12, 16; indep / overlap / same routes; 3 scales; "
               "every routed + shared output byte\n", bad[i] ? "BAD" : "ok ", tag, impls[i].name);
}

static void gemv_check() {
    CUfunction g1 = load("dense_gemv_bf16", "dense_gemv_bf16");
    CUfunction gm = load("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2", 8192, 2560}, {"GDN out_proj TP2", 2560, 3072},
        {"LM head shard (4096 cols)", 4096, 2560}};
    for (auto& s : shapes) {
        auto w = rand_bf16((size_t)s.N * s.K, 0.05f);
        unsigned short* W = dput(w);
        for (unsigned m = 2; m <= MAXR; m++) {
            auto a = rand_bf16((size_t)m * s.K, 1.0f);
            unsigned short* A = dput(a);
            void* ref = dzero((size_t)m * s.N * 2);
            void* got = dzero((size_t)m * s.N * 2);
            unsigned N = s.N, K = s.K, M = m, stride = s.N;
            for (unsigned r = 0; r < m; r++) {
                void* Ar = A + (size_t)r * s.K;
                void* Cr = (char*)ref + (size_t)r * s.N * 2;
                launch(g1, dim3((N + 3) / 4), dim3(256), {&Ar, &W, &Cr, &N, &K});
            }
            launch(gm, dim3((N + 3) / 4), dim3(256), {&A, &W, &got, &M, &N, &K, &stride});
            char what[96];
            snprintf(what, sizeof what, "dense_gemv_bf16_batchm M=%u %s", m, s.n);
            if (same(what, dget(ref, (size_t)m * s.N * 2), dget(got, (size_t)m * s.N * 2)) && m == MAXR)
                printf("  ok  %s, M=2..8\n", s.n);
            CK(cudaFree(A)); CK(cudaFree(ref)); CK(cudaFree(got));
        }
        CK(cudaFree(W));
    }
}

static void router_check() {
    CUfunction sw = load("w4a16_gemv", "w4a16_gemv_sw");
    const unsigned N = NE, K = H;
    Proj w = make_proj(N, K);
    for (const char* tier : {"w4a16_gemv_batch4", "w4a16_gemv_batch8"}) {
        CUfunction bm = load("w4a16_gemv", tier);
        const unsigned lo = tier[16] == '4' ? 2 : 5, hi = tier[16] == '4' ? 4 : 8;
        for (unsigned m = lo; m <= hi; m++) {
            auto a = rand_bf16((size_t)m * K, 1.0f);
            unsigned short* A = dput(a);
            void* ref = dzero((size_t)m * N * 2);
            void* got = dzero((size_t)m * N * 2);
            unsigned n = N, k = K, M = m;
            float s2 = w.s2;
            for (unsigned r = 0; r < m; r++) {
                void* Ar = A + (size_t)r * K;
                void* Cr = (char*)ref + (size_t)r * N * 2;
                launch(sw, dim3((N + 7) / 8), dim3(256),
                       {&Ar, (void*)&w.packed, (void*)&w.scale, &s2, &Cr, &n, &k});
            }
            launch(bm, dim3((N + 3) / 4), dim3(256),
                   {&A, (void*)&w.packed, (void*)&w.scale, &s2, &got, &M, &n, &k});
            char what[96];
            snprintf(what, sizeof what, "router %s M=%u vs w4a16_gemv_sw", tier, m);
            if (same(what, dget(ref, (size_t)m * N * 2), dget(got, (size_t)m * N * 2)) && m == hi)
                printf("  ok  router (512 x 2560 NVFP4) %s M=%u..%u vs w4a16_gemv_sw per row\n",
                       tier, lo, hi);
            CK(cudaFree(A)); CK(cudaFree(ref)); CK(cudaFree(got));
        }
    }
}

// FP8 GDN projections (ATLAS_QWEN4EXP_FP8_GDN): w8a16_gemv_batch4 (M<=4) and
// w8a16_gemv_batch16 (M=5..8) rows vs w8a16_gemv per row, 128x128 FP32 block
// scales, E4M3 codes over the finite range.
static void fp8_check() {
    CUfunction g1 = load("w8a16_gemv", "w8a16_gemv");
    CUfunction b4 = load("w8a16_gemv_batch4", "w8a16_gemv_batch4");
    CUfunction b16 = load("w8a16_gemv_batch4", "w8a16_gemv_batch16");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2", 8192, 2560}, {"GDN out_proj TP2", 2560, 3072}};
    for (auto& s : shapes) {
        auto codes = rand_codes((size_t)s.N * s.K);
        for (auto& c : codes) if ((c & 0x7F) == 0x7F) c ^= 0x01;  // no NaN
        unsigned char* W = dput(codes);
        std::uniform_real_distribution<float> d(0.001f, 0.01f);
        std::vector<float> sc((size_t)((s.N + 127) / 128) * ((s.K + 127) / 128));
        for (auto& x : sc) x = d(g_rng);
        float* S = dput(sc);
        for (unsigned m = 2; m <= MAXR; m++) {
            auto a = rand_bf16((size_t)m * s.K, 1.0f);
            unsigned short* A = dput(a);
            void* ref = dzero((size_t)m * s.N * 2);
            void* got = dzero((size_t)m * s.N * 2);
            unsigned N = s.N, K = s.K, M = m;
            for (unsigned r = 0; r < m; r++) {
                void* Ar = A + (size_t)r * s.K;
                void* Cr = (char*)ref + (size_t)r * s.N * 2;
                launch(g1, dim3((N + 3) / 4), dim3(256), {&Ar, &W, &S, &Cr, &N, &K});
            }
            launch(m <= 4 ? b4 : b16, dim3((N + 3) / 4), dim3(256), {&A, &W, &S, &got, &M, &N, &K});
            char what[96];
            snprintf(what, sizeof what, "%s M=%u %s", m <= 4 ? "w8a16_gemv_batch4" : "w8a16_gemv_batch16", m, s.n);
            if (same(what, dget(ref, (size_t)m * s.N * 2), dget(got, (size_t)m * s.N * 2)) && m == MAXR)
                printf("  ok  FP8 %s: batch4 M=2..4, batch16 M=5..8 vs w8a16_gemv per row\n", s.n);
            CK(cudaFree(A)); CK(cudaFree(ref)); CK(cudaFree(got));
        }
        CK(cudaFree(W)); CK(cudaFree(S));
    }
}

static void blend_check() {
    CUfunction bl = load("moe_permute", "moe_batched_blend");
    for (unsigned rows = 2; rows <= MAXR; rows++) {
        auto out0 = rand_bf16((size_t)rows * H, 1.0f), sh = rand_bf16((size_t)rows * H, 1.0f);
        auto x = rand_bf16((size_t)rows * H, 1.0f), g = rand_bf16(H, 0.05f);
        unsigned short *ref = dput(out0), *got = dput(out0), *S = dput(sh), *X = dput(x), *G = dput(g);
        unsigned h = H, one = 1, R = rows;
        for (unsigned r = 0; r < rows; r++) {
            void* o = ref + (size_t)r * H; void* s = S + (size_t)r * H; void* xr = X + (size_t)r * H;
            launch(bl, dim3(1), dim3(256), {&o, &s, &xr, &G, &h, &one});
        }
        launch(bl, dim3(rows), dim3(256), {&got, &S, &X, &G, &h, &R});
        char what[64];
        snprintf(what, sizeof what, "moe_batched_blend T=%u", rows);
        if (same(what, dget(ref, (size_t)rows * H * 2), dget(got, (size_t)rows * H * 2)) && rows == MAXR)
            printf("  ok  moe_batched_blend T=2..8 vs T=1 per row\n");
    }
}

static void moe_time(const Tables& t, const Proj& sg, const Proj& su, const Proj& sdn, bool ep2,
                     bool sweep) {
    CUfunction gu1 = load("moe_shared_expert_fused", "moe_expert_gate_up_shared");
    CUfunction sd1 = load("moe_shared_expert_fused", "moe_expert_silu_down_shared");
    const std::vector<RowsImpl> impls = rows_impls(sweep);
    MoeBufs b = alloc_bufs();
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int iters = 100;
    // Unique expert bytes a launch must read (each local expert picked, + shared).
    const double gu_bytes = 2.0 * (I * H / 2 + I * H / 16), sd_bytes = H * I / 2 + H * I / 16;
    printf("\n  %s: us per launch, routes over a %u-expert pool (GB/s of unique expert bytes)\n",
           ep2 ? "EP2 tables" : "TP1 tables", NE);
    printf("  rows  routes   %-22s %6s  %-17s  %-17s  %7s\n", "", "plan", "gate_up (GB/s)", "silu_down (GB/s)", "layer");
    for (unsigned rows : {1u, 2u, 4u, 6u, 8u}) {
        for (int overlap = 0; overlap < 2; overlap++) {
            std::vector<unsigned*> dev;
            double uniq = 0;
            for (int i = 0; i < iters; i++) {
                auto p = routes(rows, overlap);
                std::vector<unsigned> local;
                for (unsigned e : p) if (!(ep2 && e % 2 == 1)) local.push_back(e);
                std::sort(local.begin(), local.end());
                uniq += std::unique(local.begin(), local.end()) - local.begin() + 1;
                dev.push_back(dput(p));
            }
            uniq /= iters;
            auto timed = [&](auto&& body) {
                float ms;
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; i++) body(i);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms, e0, e1));
                return ms * 1000.0 / iters;
            };
            const double loop = timed([&](int i) {
                b.ids = dev[i];
                per_row_pair(gu1, sd1, t, sg, su, sdn, b, rows);
            });
            printf("  %4u  %-7s  %-22s %6s  %-17s  %-17s  %7.1f  (%.1f unique experts)\n", rows,
                   overlap ? "overlap" : "indep", "per-row loop", "", "", "", loop, uniq);
            for (const RowsImpl& m : impls) {
                // Each route set's plan into its own order buffer, so the
                // projections can be timed alone.
                std::vector<void*> orders;
                for (int i = 0; i < iters; i++) {
                    g_order = dzero(4096);
                    b.ids = dev[i];
                    rows_plan(m, b, rows);
                    orders.push_back(g_order);
                }
                const double pl = timed([&](int i) { b.ids = dev[i]; g_order = orders[i]; rows_plan(m, b, rows); });
                const double gu = timed([&](int i) { b.ids = dev[i]; g_order = orders[i]; rows_gate_up(m, t, sg, su, b, rows); });
                const double sd = timed([&](int i) { b.ids = dev[i]; g_order = orders[i]; rows_silu_down(m, t, sdn, b, rows); });
                const double all = timed([&](int i) { b.ids = dev[i]; g_order = orders[i]; rows_pair(m, t, sg, su, sdn, b, rows); });
                printf("  %4u  %-7s  %-22s %6.1f  %7.1f (%5.0f)  %7.1f (%5.0f)  %7.1f  %.2fx\n", rows,
                       overlap ? "overlap" : "indep", m.name, pl, gu, uniq * gu_bytes / gu / 1e3, sd,
                       uniq * sd_bytes / sd / 1e3, all, loop / all);
                for (void* o : orders) CK(cudaFree(o));
                g_order = nullptr;
            }
            for (auto* p : dev) CK(cudaFree(p));
        }
    }
}

// GPU time of the exact BF16 batched GEMV against M single-row GEMVs, over a
// ring of weight copies (>= 4x the 24 MiB L2) so each launch streams from DRAM.
static void gemv_time() {
    CUfunction g1 = load("dense_gemv_bf16", "dense_gemv_bf16");
    CUfunction gm = load("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    struct { const char* n; unsigned N, K; } shapes[] = {
        {"GDN qkvz TP2 8192x2560", 8192, 2560}, {"GDN out_proj TP2 2560x3072", 2560, 3072},
        {"LM head half 124160x2560", 124160, 2560}};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("\n  us per projection, BF16: M x dense_gemv_bf16 vs one dense_gemv_bf16_batchm\n");
    for (auto& s : shapes) {
        const size_t wbytes = (size_t)s.N * s.K * 2;
        const int copies = (int)std::max<size_t>(1, (size_t)(128u << 20) / wbytes + 1);
        std::vector<unsigned short*> W;
        for (int c = 0; c < copies; c++) W.push_back((unsigned short*)dzero(wbytes));
        unsigned short* A = dput(rand_bf16((size_t)MAXR * s.K, 1.0f));
        void* C = dzero((size_t)MAXR * s.N * 2);
        const int iters = s.N > 100000 ? 20 : 100;
        printf("  %-28s", s.n);
        for (unsigned m : {1u, 2u, 4u, 8u}) {
            float ms[2];
            for (int which = 0; which < 2; which++) {
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; i++) {
                    void* w = W[i % copies];
                    unsigned N = s.N, K = s.K, M = m, stride = s.N;
                    if (which == 0) {
                        for (unsigned r = 0; r < m; r++) {
                            void* Ar = A + (size_t)r * s.K;
                            void* Cr = (char*)C + (size_t)r * s.N * 2;
                            launch(g1, dim3((N + 3) / 4), dim3(256), {&Ar, &w, &Cr, &N, &K});
                        }
                    } else {
                        launch(gm, dim3((N + 3) / 4), dim3(256), {&A, &w, &C, &M, &N, &K, &stride});
                    }
                }
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[which], e0, e1));
            }
            printf("  M=%u %7.1f -> %7.1f", m, ms[0] * 1000 / iters, ms[1] * 1000 / iters);
        }
        printf("\n");
        for (auto* w : W) CK(cudaFree(w));
        CK(cudaFree(A));
        CK(cudaFree(C));
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    CU(cuInit(0));
    CK(cudaFree(0));
    // A pool of distinct experts (DRAM-resident: ~1.4 GB) plus the shared one.
    const unsigned pool = mode == "check" ? 32 : NE;
    std::vector<Proj> gate, up, down;
    for (unsigned e = 0; e < pool; e++) {
        gate.push_back(make_proj(I, H));
        up.push_back(make_proj(I, H));
        down.push_back(make_proj(H, I));
    }
    Proj sg = make_proj(I, H), su = make_proj(I, H), sdn = make_proj(H, I);
    Tables tp1 = make_tables(gate, up, down, false), ep2 = make_tables(gate, up, down, true);
    if (mode == "check") {
        printf("per-row parity:\n");
        moe_check(tp1, sg, su, sdn, "TP1");
        moe_check(ep2, sg, su, sdn, "EP2");
        gemv_check();
        router_check();
        fp8_check();
        blend_check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    const bool sweep = mode == "sweep";
    moe_time(ep2, sg, su, sdn, true, sweep);
    if (sweep) return 0;
    moe_time(tp1, sg, su, sdn, false, false);
    gemv_time();
    return 0;
}
