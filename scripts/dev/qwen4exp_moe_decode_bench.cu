// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check and A/B timing of the Qwen3.8-Flash-Next unified-
// layout MoE decode kernels (ATLAS_QWEN4EXP_MOE_FAST) against the kernels they
// replace, at the real shapes (hidden 2560, routed + shared intermediate 640,
// top-10 of 512 NVFP4 experts, [K/2, N] transposed tables) for T = 1..4 rows:
//
//   old  moe_expert_{gate_up,silu_down}_shared_t   T=1, and once per row for
//                                                  T=4 (as forward_batched runs it)
//        ..._batch2_t / ..._batch3_t               T=2 / T=3
//   new  qwen4exp_moe_{gate_up,silu_down}_t        every T, one launch
//
// Every kernel is compiled to PTX with the target's production flags and
// JIT-loaded through the driver API, exactly as Atlas loads it:
//
//   scripts/dev/qwen4exp_moe_decode_bench.sh check|time|sweep [pool] [reps]
//
// `check`: every output byte (routed gate/up/down rows and the shared rows)
// of new against old, T = 1..4, with and without the shared expert, all 10
// experts local (TP1) and EP2-style tables whose upper half is NULL (TP2),
// rows routed independently and with heavy overlap (so the one-read-per-
// expert path runs), at three input scales including zeros and -0.0. One
// line per case, then PASS or FAIL (exit 1).
// `time`: median GPU microseconds per call, old vs new, gate_up, down and the
// pair. Weights come from a pool of `pool` distinct experts (2.76 MB each),
// routes are random per launch, and each batch of 48 calls is queued behind a
// spin kernel, so every launch streams from DRAM and the numbers include the
// back-to-back launch gap. QX_SH_COPIES=1 with a pool of 4 keeps everything
// L2-resident, which times the compute side instead.
// `sweep`: `time` over the templated shapes (values per thread, ring depth,
// L2 prefetch distance, CTA width), each bit-checked first; QX_SWEEP_ONLY
// filters by tag.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s = nullptr; \
    cuGetErrorString(r, &s); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s ? s : "?"); exit(1); } } while (0)

typedef unsigned long long u64;
static const unsigned H = 2560, I = 640, TOPK = 10, NE = 512, MAXT = 4;
static unsigned SH_COPIES = 8;  // shared-expert copies cycled per launch
static size_t g_pool = 256;
static int g_reps = 7;

// ── device helpers (bench-only) ─────────────────────────────────────────
__global__ void fill_bytes(unsigned char* p, size_t n, u64 seed, int kind) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        u64 z = seed + i * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        unsigned char b = (unsigned char)z;
        if (kind == 1) {
            // E4M3 block scale: positive normals 2^-6 .. 2^3, 1/64 zeros.
            unsigned e = 1 + (unsigned)((z >> 8) % 10), m = (unsigned)((z >> 16) & 7);
            b = ((z >> 24) & 63) == 0 ? 0 : (unsigned char)((e << 3) | m);
        }
        p[i] = b;
    }
}
__global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
__global__ void stream_read(const uint4* p, size_t n, uint4* sink) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0x12345678u) sink[0] = acc;
}

// ── weights: [K/2, N] packed + [K/16, N] scales per expert ──────────────
struct Proj {
    unsigned char* packed = nullptr;
    unsigned char* scale = nullptr;
    size_t packed_bytes = 0, scale_bytes = 0;
};
struct Weights {
    Proj gate, up, down, sh_gate, sh_up, sh_down;
    u64* tab[2][3][2] = {};  // [tp: 0 all local, 1 ids >= NE/2 NULL][proj][packed/scale]
    float* s2[3] = {};
    std::vector<float> sh_s2;  // [proj * SH_COPIES + copy]
};

static void alloc_proj(Proj& p, size_t n_out, size_t k, size_t count, u64 seed) {
    p.packed_bytes = n_out * k / 2;
    p.scale_bytes = n_out * k / 16;
    CK(cudaMalloc(&p.packed, p.packed_bytes * count));
    CK(cudaMalloc(&p.scale, p.scale_bytes * count));
    fill_bytes<<<1024, 256>>>(p.packed, p.packed_bytes * count, seed, 0);
    fill_bytes<<<1024, 256>>>(p.scale, p.scale_bytes * count, seed ^ 0xABCDEF, 1);
    CK(cudaGetLastError());
}

static void build_weights(Weights& W, std::mt19937& rng) {
    alloc_proj(W.gate, I, H, g_pool, 1);
    alloc_proj(W.up, I, H, g_pool, 2);
    alloc_proj(W.down, H, I, g_pool, 3);
    alloc_proj(W.sh_gate, I, H, SH_COPIES, 4);
    alloc_proj(W.sh_up, I, H, SH_COPIES, 5);
    alloc_proj(W.sh_down, H, I, SH_COPIES, 6);
    std::uniform_real_distribution<float> u(1e-3f, 5e-3f);
    Proj* pr[3] = {&W.gate, &W.up, &W.down};
    for (int p = 0; p < 3; p++) {
        std::vector<float> s2(NE);
        for (auto& v : s2) v = u(rng);
        CK(cudaMalloc(&W.s2[p], NE * 4));
        CK(cudaMemcpy(W.s2[p], s2.data(), NE * 4, cudaMemcpyHostToDevice));
        for (int tp = 0; tp < 2; tp++) {
            std::vector<u64> pk(NE), sc(NE);
            for (unsigned e = 0; e < NE; e++) {
                const bool local = tp == 0 || e < NE / 2;
                const size_t slot = e % g_pool;
                pk[e] = local ? (u64)(pr[p]->packed + slot * pr[p]->packed_bytes) : 0;
                sc[e] = local ? (u64)(pr[p]->scale + slot * pr[p]->scale_bytes) : 0;
            }
            for (int k = 0; k < 2; k++) CK(cudaMalloc(&W.tab[tp][p][k], NE * 8));
            CK(cudaMemcpy(W.tab[tp][p][0], pk.data(), NE * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(W.tab[tp][p][1], sc.data(), NE * 8, cudaMemcpyHostToDevice));
        }
    }
    W.sh_s2.resize(3 * SH_COPIES);
    for (auto& v : W.sh_s2) v = u(rng);
}

// ── modules ─────────────────────────────────────────────────────────────
static std::string g_ptx_dir;
static std::map<std::string, CUmodule> g_mods;
static CUfunction fn(const std::string& stem, const std::string& name) {
    if (!g_mods.count(stem)) {
        CUmodule m;
        const std::string path = g_ptx_dir + "/" + stem + ".ptx";
        CU(cuModuleLoad(&m, path.c_str()));
        g_mods[stem] = m;
    }
    CUfunction f;
    if (cuModuleGetFunction(&f, g_mods[stem], name.c_str()) != CUDA_SUCCESS) {
        fprintf(stderr, "missing %s::%s\n", stem.c_str(), name.c_str());
        exit(1);
    }
    return f;
}

struct Args {
    std::vector<std::vector<unsigned char>> store;
    std::vector<void*> ptrs;
    template <typename T> Args& add(T v) {
        store.emplace_back(sizeof(T));
        memcpy(store.back().data(), &v, sizeof(T));
        return *this;
    }
    void** get() {
        ptrs.clear();
        for (auto& s : store) ptrs.push_back(s.data());
        return ptrs.data();
    }
};
static void launch(CUfunction f, dim3 g, dim3 b, unsigned smem, Args& a) {
    if (smem > 48 * 1024) CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, 0, a.get(), nullptr));
}

// ── one MoE decode site's buffers ───────────────────────────────────────
struct Site {
    __nv_bfloat16 *A, *gate_out, *up_out, *sh_gate, *sh_up, *down_out, *sh_down;
    void alloc() {
        CK(cudaMalloc(&A, MAXT * H * 2));
        CK(cudaMalloc(&gate_out, MAXT * TOPK * I * 2));
        CK(cudaMalloc(&up_out, MAXT * TOPK * I * 2));
        CK(cudaMalloc(&sh_gate, MAXT * I * 2));
        CK(cudaMalloc(&sh_up, MAXT * I * 2));
        CK(cudaMalloc(&down_out, MAXT * TOPK * H * 2));
        CK(cudaMalloc(&sh_down, MAXT * H * 2));
    }
    void poison() {
        CK(cudaMemset(gate_out, 0x7F, MAXT * TOPK * I * 2));
        CK(cudaMemset(up_out, 0x7F, MAXT * TOPK * I * 2));
        CK(cudaMemset(sh_gate, 0x7F, MAXT * I * 2));
        CK(cudaMemset(sh_up, 0x7F, MAXT * I * 2));
        CK(cudaMemset(down_out, 0x7F, MAXT * TOPK * H * 2));
        CK(cudaMemset(sh_down, 0x7F, MAXT * H * 2));
    }
};

// What one call reads: routes and weight copies.
struct Call {
    const unsigned* idx;  // [T*TOPK]
    int tp;               // 0 TP1, 1 TP2 (half NULL)
    int sh;               // shared copy, -1 = no shared expert (NULL)
    unsigned T;
};

static const unsigned char* shp(const Proj& p, int c, bool scale) {
    if (c < 0) return nullptr;
    return scale ? p.scale + (size_t)c * p.scale_bytes : p.packed + (size_t)c * p.packed_bytes;
}
static float sh_s2(const Weights& W, int proj, int c) { return c < 0 ? 0.f : W.sh_s2[proj * SH_COPIES + c]; }

// Argument lists shared by old and new (new appends `rows`).
static Args gate_up_args(const Weights& W, const Call& c, const __nv_bfloat16* A, __nv_bfloat16* go,
                         __nv_bfloat16* uo, const unsigned* idx, __nv_bfloat16* sgo, __nv_bfloat16* suo) {
    Args a;
    a.add(A).add(W.tab[c.tp][0][0]).add(W.tab[c.tp][0][1]).add(W.s2[0]).add(go)
     .add(W.tab[c.tp][1][0]).add(W.tab[c.tp][1][1]).add(W.s2[1]).add(uo).add(idx)
     .add(shp(W.sh_gate, c.sh, false)).add(shp(W.sh_gate, c.sh, true)).add(sh_s2(W, 0, c.sh)).add(sgo)
     .add(shp(W.sh_up, c.sh, false)).add(shp(W.sh_up, c.sh, true)).add(sh_s2(W, 1, c.sh)).add(suo)
     .add(I).add(H).add(TOPK);
    return a;
}
static Args down_args(const Weights& W, const Call& c, const __nv_bfloat16* go, const __nv_bfloat16* uo,
                      __nv_bfloat16* dout, const unsigned* idx, const __nv_bfloat16* sgi,
                      const __nv_bfloat16* sui, __nv_bfloat16* sdo) {
    Args a;
    a.add(go).add(uo).add(W.tab[c.tp][2][0]).add(W.tab[c.tp][2][1]).add(W.s2[2]).add(dout).add(idx)
     .add(sgi).add(sui).add(shp(W.sh_down, c.sh, false)).add(shp(W.sh_down, c.sh, true))
     .add(sh_s2(W, 2, c.sh)).add(sdo).add(H).add(I).add(TOPK);
    return a;
}

// OLD, exactly as serving launches it (ops/fp8_moe.rs, fp8_moe_batch_a.rs:
// T_BLOCK = 32, grid (N/32, T*(top_k+1), 2 | 1), down smem K*4).
static void old_gate_up(const Weights& W, const Site& s, const Call& c) {
    if (c.T == 2 || c.T == 3) {
        const std::string sfx = c.T == 2 ? "batch2_t" : "batch3_t";
        Args a = gate_up_args(W, c, s.A, s.gate_out, s.up_out, c.idx, s.sh_gate, s.sh_up);
        launch(fn("moe_shared_expert_fused_" + sfx, "moe_expert_gate_up_shared_" + sfx),
               dim3(I / 32, c.T * (TOPK + 1), 2), dim3(32), 0, a);
        return;
    }
    for (unsigned t = 0; t < c.T; t++) {
        Args a = gate_up_args(W, c, s.A + t * H, s.gate_out + t * TOPK * I, s.up_out + t * TOPK * I,
                              c.idx + t * TOPK, s.sh_gate + t * I, s.sh_up + t * I);
        launch(fn("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t"), dim3(I / 32, TOPK + 1, 2), dim3(32),
               0, a);
    }
}
static void old_silu_down(const Weights& W, const Site& s, const Call& c) {
    if (c.T == 2 || c.T == 3) {
        const std::string sfx = c.T == 2 ? "batch2_t" : "batch3_t";
        Args a = down_args(W, c, s.gate_out, s.up_out, s.down_out, c.idx, s.sh_gate, s.sh_up, s.sh_down);
        launch(fn("moe_shared_expert_fused_" + sfx, "moe_expert_silu_down_shared_" + sfx),
               dim3(H / 32, c.T * (TOPK + 1)), dim3(32), I * 4, a);
        return;
    }
    for (unsigned t = 0; t < c.T; t++) {
        Args a = down_args(W, c, s.gate_out + t * TOPK * I, s.up_out + t * TOPK * I, s.down_out + t * TOPK * H,
                           c.idx + t * TOPK, s.sh_gate + t * I, s.sh_up + t * I, s.sh_down + t * H);
        launch(fn("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t"), dim3(H / 32, TOPK + 1), dim3(32),
               I * 4, a);
    }
}

// NEW. Production shape (crates/spark-model/src/layers/ops/qwen4exp_moe.rs):
// 160 threads x 2 outputs = 320 columns a CTA; the sweep overrides it.
struct Shape {
    std::string gu = "qwen4exp_moe_gate_up_t", dn = "qwen4exp_moe_silu_down_t";
    unsigned block = 160, cols = 320;
};
static void new_gate_up(const Shape& n, const Weights& W, const Site& s, const Call& c) {
    Args a = gate_up_args(W, c, s.A, s.gate_out, s.up_out, c.idx, s.sh_gate, s.sh_up);
    a.add(c.T);
    launch(fn("qwen4exp_moe_decode", n.gu), dim3(I / n.cols, c.T * TOPK + 1, 2), dim3(n.block), c.T * H * 4, a);
}
static void new_silu_down(const Shape& n, const Weights& W, const Site& s, const Call& c) {
    Args a = down_args(W, c, s.gate_out, s.up_out, s.down_out, c.idx, s.sh_gate, s.sh_up, s.sh_down);
    a.add(c.T);
    launch(fn("qwen4exp_moe_decode", n.dn), dim3(H / n.cols, c.T * TOPK + 1), dim3(n.block), c.T * I * 4, a);
}

// ── routes ──────────────────────────────────────────────────────────────
// Row 0 picks TOPK distinct ids out of NE; each later row reuses each of
// row 0's picks with probability `overlap`, the rest fresh (distinct in row).
static std::vector<unsigned> make_route(std::mt19937& rng, unsigned T, double overlap) {
    std::vector<unsigned> idx(T * TOPK);
    std::uniform_int_distribution<unsigned> pick(0, NE - 1);
    std::uniform_real_distribution<double> coin(0, 1);
    for (unsigned t = 0; t < T; t++) {
        std::vector<unsigned> row;
        for (unsigned s = 0; s < TOPK; s++) {
            unsigned e;
            do {
                e = (t > 0 && coin(rng) < overlap) ? idx[s] : pick(rng);
            } while (std::find(row.begin(), row.end(), e) != row.end());
            row.push_back(e);
        }
        std::shuffle(row.begin(), row.end(), rng);
        for (unsigned s = 0; s < TOPK; s++) idx[t * TOPK + s] = row[s];
    }
    return idx;
}

static void fill_input(std::mt19937& rng, __nv_bfloat16* dst, float scale) {
    std::normal_distribution<float> nd(0.f, scale);
    std::uniform_int_distribution<int> z(0, 31);
    std::vector<__nv_bfloat16> h(MAXT * H);
    for (auto& v : h) {
        const int k = z(rng);
        v = __float2bfloat16(k == 0 ? 0.0f : k == 1 ? -0.0f : nd(rng));
    }
    CK(cudaMemcpy(dst, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
}

static bool same(const char* what, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n, std::string& msg) {
    std::vector<unsigned short> x(n), y(n);
    CK(cudaMemcpy(x.data(), a, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, n * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (x[i] != y[i] && bad++ == 0) first = i;
    if (bad) {
        char buf[256];
        snprintf(buf, sizeof buf, " %s:%zu diff (first @%zu: %04x vs %04x)", what, bad, first, x[first], y[first]);
        msg += buf;
    }
    return bad == 0;
}

// Old and new on the same inputs into two sites; compare every byte. Down
// runs on the OLD gate/up in both, so a gate/up diff cannot hide one there.
static bool check_case(const Shape& n, const Weights& W, Site& so, Site& sn, const Call& c, std::string& msg) {
    so.poison();
    sn.poison();
    CK(cudaMemcpy(sn.A, so.A, MAXT * H * 2, cudaMemcpyDeviceToDevice));
    old_gate_up(W, so, c);
    new_gate_up(n, W, sn, c);
    CK(cudaDeviceSynchronize());
    bool ok = true;
    ok &= same("gate", so.gate_out, sn.gate_out, c.T * TOPK * I, msg);
    ok &= same("up", so.up_out, sn.up_out, c.T * TOPK * I, msg);
    ok &= same("sh_gate", so.sh_gate, sn.sh_gate, c.T * I, msg);
    ok &= same("sh_up", so.sh_up, sn.sh_up, c.T * I, msg);
    CK(cudaMemcpy(sn.gate_out, so.gate_out, MAXT * TOPK * I * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(sn.up_out, so.up_out, MAXT * TOPK * I * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(sn.sh_gate, so.sh_gate, MAXT * I * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(sn.sh_up, so.sh_up, MAXT * I * 2, cudaMemcpyDeviceToDevice));
    old_silu_down(W, so, c);
    new_silu_down(n, W, sn, c);
    CK(cudaDeviceSynchronize());
    ok &= same("down", so.down_out, sn.down_out, c.T * TOPK * H, msg);
    ok &= same("sh_down", so.sh_down, sn.sh_down, c.T * H, msg);
    return ok;
}

static bool check_all(const Shape& n, const Weights& W, Site& so, Site& sn, std::mt19937& rng, bool verbose) {
    bool all = true;
    unsigned* idx_dev;
    CK(cudaMalloc(&idx_dev, MAXT * TOPK * 4));
    for (unsigned T = 1; T <= MAXT; T++)
        for (int tp = 0; tp < 2; tp++)
            for (int sh = 0; sh < 2; sh++)
                for (double ov : {0.0, 0.7})
                    for (float sc : {1.0f, 0.02f, 30.0f}) {
                        const auto route = make_route(rng, T, ov);
                        CK(cudaMemcpy(idx_dev, route.data(), route.size() * 4, cudaMemcpyHostToDevice));
                        fill_input(rng, so.A, sc);
                        const Call c{idx_dev, tp, sh ? (int)(rng() % SH_COPIES) : -1, T};
                        std::string msg;
                        const bool ok = check_case(n, W, so, sn, c, msg);
                        all &= ok;
                        if (verbose || !ok)
                            printf("  bitwise T=%u %s %s overlap=%.1f scale=%g: %s%s\n", T, tp ? "tp2" : "tp1",
                                   sh ? "shared" : "no-shared", ov, sc, ok ? "ok" : "DIFF", msg.c_str());
                    }
    CK(cudaFree(idx_dev));
    return all;
}

// ── timing ──────────────────────────────────────────────────────────────
static const int BATCH = 48, NROUTES = 64;

template <typename F> static float time_us(F&& body) {
    static cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (!e0) { CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); }
    std::vector<float> v;
    for (int r = 0; r < g_reps + 1; r++) {
        spin<<<1, 1>>>(4000000);
        CK(cudaEventRecord(e0));
        for (int j = 0; j < BATCH; j++) body(j);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (r) v.push_back(ms * 1000.f / BATCH);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Routes {
    unsigned* dev = nullptr;
    std::vector<std::vector<unsigned>> host;
    void build(std::mt19937& rng, unsigned T, double ov) {
        host.clear();
        std::vector<unsigned> flat;
        for (int r = 0; r < NROUTES; r++) {
            host.push_back(make_route(rng, T, ov));
            host.back().resize(MAXT * TOPK, 0);
            flat.insert(flat.end(), host.back().begin(), host.back().end());
        }
        if (!dev) CK(cudaMalloc(&dev, NROUTES * MAXT * TOPK * 4));
        CK(cudaMemcpy(dev, flat.data(), flat.size() * 4, cudaMemcpyHostToDevice));
    }
    const unsigned* at(int j) const { return dev + (size_t)(j % NROUTES) * MAXT * TOPK; }
    // MB a call streams: distinct local experts in the route, plus shared.
    double mb(unsigned T, int tp, bool gate_up) const {
        double tot = 0;
        for (auto& r : host) {
            std::vector<unsigned> d;
            for (unsigned i = 0; i < T * TOPK; i++)
                if ((tp == 0 || r[i] < NE / 2) && std::find(d.begin(), d.end(), r[i]) == d.end()) d.push_back(r[i]);
            tot += (d.size() + 1) * (gate_up ? 2.0 : 1.0) * I * H * 9 / 16;
        }
        return tot / host.size() / 1e6;
    }
};

static void time_all(const Weights& W, Site& s, std::mt19937& rng) {
    const Shape n;
    for (int tp = 0; tp < 2; tp++)
        for (unsigned T = 1; T <= MAXT; T++)
            for (double ov : {0.0, 0.7}) {
                if (T == 1 && ov > 0) continue;
                Routes R;
                R.build(rng, T, ov);
                auto call = [&](int j) { return Call{R.at(j), tp, (int)(j % SH_COPIES), T}; };
                fill_input(rng, s.A, 1.0f);
                const float og = time_us([&](int j) { old_gate_up(W, s, call(j)); });
                const float ng = time_us([&](int j) { new_gate_up(n, W, s, call(j)); });
                const float od = time_us([&](int j) { old_silu_down(W, s, call(j)); });
                const float nd = time_us([&](int j) { new_silu_down(n, W, s, call(j)); });
                const double mg = R.mb(T, tp, true), md = R.mb(T, tp, false);
                printf("%s T=%u ov=%.1f  gate_up %6.1f -> %6.1f us (%4.0f -> %4.0f GB/s)  "
                       "down %6.1f -> %6.1f us (%4.0f -> %4.0f GB/s)  pair %6.1f -> %6.1f (%.2fx)\n",
                       tp ? "tp2" : "tp1", T, ov, og, ng, mg * 1e3 / og, mg * 1e3 / ng, od, nd, md * 1e3 / od,
                       md * 1e3 / nd, og + od, ng + nd, (og + od) / (ng + nd));
                fflush(stdout);
            }
}

static void sweep(const Weights& W, Site& so, Site& sn, std::mt19937& rng) {
    // (V, ring depth, L2 prefetch distance): QX_T_VARIANT in the kernel file.
    const int tv[][3] = {{1, 4, 0}, {1, 4, 8}, {1, 4, 16}, {1, 8, 16}, {2, 4, 0}, {2, 4, 8},
                         {2, 4, 16}, {4, 2, 8}, {4, 4, 0}, {4, 4, 8}, {4, 4, 16}};
    const char* only = getenv("QX_SWEEP_ONLY");
    for (auto& t : tv)
        for (unsigned block : {32u, 64u, 128u, 160u}) {
            const unsigned cols = block * t[0];
            if (I % cols) continue;
            const std::string sfx =
                "_v" + std::to_string(t[0]) + "_d" + std::to_string(t[1]) + "_pf" + std::to_string(t[2]);
            const Shape n{"qx_sweep_gate_up_t" + sfx, "qx_sweep_silu_down_t" + sfx, block, cols};
            const std::string tag = sfx.substr(1) + " cols" + std::to_string(cols);
            if (only && tag.find(only) == std::string::npos) continue;
            bool ok = true;
            unsigned* idx_dev;
            CK(cudaMalloc(&idx_dev, MAXT * TOPK * 4));
            for (unsigned T = 1; T <= MAXT; T++) {
                const auto route = make_route(rng, T, 0.5);
                CK(cudaMemcpy(idx_dev, route.data(), route.size() * 4, cudaMemcpyHostToDevice));
                fill_input(rng, so.A, 1.0f);
                std::string msg;
                ok &= check_case(n, W, so, sn, Call{idx_dev, 1, 0, T}, msg);
            }
            CK(cudaFree(idx_dev));
            printf("== %s bitwise %s\n", tag.c_str(), ok ? "ok" : "DIFF");
            if (!ok) continue;
            for (int tp = 0; tp < 2; tp++)
                for (unsigned T : {1u, 2u, 4u}) {
                    Routes R;
                    R.build(rng, T, 0.0);
                    auto call = [&](int j) { return Call{R.at(j), tp, (int)(j % SH_COPIES), T}; };
                    const float ng = time_us([&](int j) { new_gate_up(n, W, so, call(j)); });
                    const float nd = time_us([&](int j) { new_silu_down(n, W, so, call(j)); });
                    printf("   %s T=%u gate_up %6.1f us (%4.0f GB/s)  down %6.1f us (%4.0f GB/s)\n",
                           tp ? "tp2" : "tp1", T, ng, R.mb(T, tp, true) * 1e3 / ng, nd,
                           R.mb(T, tp, false) * 1e3 / nd);
                    fflush(stdout);
                }
        }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ptx dir> check|time|sweep [pool=256] [reps=7]\n", argv[0]);
        return 2;
    }
    g_ptx_dir = argv[1];
    const std::string mode = argv[2];
    if (argc > 3) g_pool = strtoul(argv[3], nullptr, 10);
    if (argc > 4) g_reps = atoi(argv[4]);
    if (getenv("QX_SH_COPIES")) SH_COPIES = strtoul(getenv("QX_SH_COPIES"), nullptr, 10);
    CK(cudaFree(0));
    CU(cuInit(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, %d SMs, L2 %d MB, pool %zu experts (%.0f MB)\n", prop.name, prop.multiProcessorCount,
           prop.l2CacheSize >> 20, g_pool, g_pool * 3.0 * I * H * 9 / 16 / 1e6);

    std::mt19937 rng(20261005);
    Weights W;
    build_weights(W, rng);
    Site so, sn;
    so.alloc();
    sn.alloc();
    CK(cudaDeviceSynchronize());

    if (mode == "check") {
        const bool ok = check_all(Shape{}, W, so, sn, rng, true);
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (mode == "time") {
        const bool ok = check_all(Shape{}, W, so, sn, rng, false);
        printf("bitwise: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
        const size_t bytes = 20u << 20;
        uint4 *buf, *sink;
        CK(cudaMalloc(&buf, bytes * 8));
        CK(cudaMalloc(&sink, 16));
        const float us = time_us([&](int j) {
            stream_read<<<prop.multiProcessorCount * 8, 256>>>(buf + (j % 8) * (bytes / 16), bytes / 16, sink);
        });
        printf("reference: contiguous %.1f MB read %.1f us = %.0f GB/s\n", bytes / 1e6, us, bytes / 1e3 / us);
        CK(cudaFree(buf));
        time_all(W, so, rng);
        return 0;
    }
    if (mode == "sweep") {
        sweep(W, so, sn, rng);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
