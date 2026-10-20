// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check and A/B timing of the Qwen3.8-Flash-Next exact
// fused decode tier (ATLAS_QWEN4EXP_DECODE_FUSE) and of PDL
// (ATLAS_QWEN4EXP_PDL) on the decode chains, at the TP2 rank shapes (8 key
// heads, 24 value heads of 128, hidden 2560, hc 4) and at TP1 (16 / 48):
//
//   gdn   dense_gemv_ba_gates -> causal_conv1d_update_l2norm_f32
//         -> gated_delta_rule_decode_f32 -> gated_rms_norm_f32_input_sigmoid
//    vs   qwen4exp_gdn_decode_fused
//   seam  hc_post_vec -> hc_pre_stage_vec
//    vs   hc_post_stage_vec
//   moe   memset -> moe_weighted_sum_blend -> (EP all-reduce) -> moe_batched_blend
//         -> hc_post_vec
//    vs   moe_weighted_sum_blend (null shared row) -> moe_blend_hc_post
//
// The check runs `steps` recurrent decode steps on both arms from the same
// state, with new random inputs each step (zeros, -0.0 and large values
// included), and compares every byte each arm leaves behind: the recurrence
// and conv states, gates, betas, the normed output, the highway and normed
// rows, the MoE output. It checks with and without PDL launches. Then it times each chain as serving launches it, cycling `copies`
// layer-sized weight and state sets (DRAM-streamed, like 36 distinct layers),
// eagerly and inside a CUDA graph, with and without programmatic dependent
// launch: stream events around `groups` chains, queued behind a spin kernel
// so the GPU never waits for the host.
//
// Build (sources compiled as separate units, as the runtime loads them):
//   K=kernels/gb10; Q=$K/qwen3.8-flash-next/nvfp4; F="-arch=sm_121a -O3 --fmad=false -std=c++17"
//   for s in $K/common/causal_conv1d.cu $K/common/ssm_preprocess.cu $K/common/rms_norm.cu \
//            $K/common/moe_expert_gemv.cu $K/common/moe_permute.cu $Q/gated_delta_rule.cu \
//            $Q/hyper_connection.cu $Q/qwen4exp_decode_fuse.cu; do
//     nvcc $F -c $s -o $(basename $s .cu).o; done
//   nvcc $F scripts/dev/qwen4exp_decode_fuse_bench.cu *.o -o qwen4exp_decode_fuse_bench
//   ./qwen4exp_decode_fuse_bench [copies=36] [groups=64] [reps=5] [steps=6]
// Prints one "bitwise" line per comparison, then PASS or FAIL (exit 1), then
// the median us per chain for each arm.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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
typedef unsigned int u32;

extern "C" __global__ void dense_gemv_ba_gates(const bf*, const bf*, const float*, const float*, float*,
                                               float*, u32, u32, u32);
extern "C" __global__ void causal_conv1d_update_l2norm_f32(float*, const bf*, const bf*, const float*,
                                                           float*, u32, u32, u32, u32, u32, float);
extern "C" __global__ void gated_delta_rule_decode_f32(float*, const float*, const float*, const float*,
                                                       const float*, const float*, float*, u32, u32, u32,
                                                       u32, u32);
extern "C" __global__ void gated_rms_norm_f32_input_sigmoid(const float*, const bf*, const bf*, bf*, u32,
                                                            float, u32, u32);
extern "C" __global__ void qwen4exp_gdn_decode_fused(float*, float*, const bf*, const bf*, const bf*,
                                                     const bf*, const float*, const float*, float*, float*,
                                                     const bf*, const bf*, bf*, u32, u32, u32, u32, float, float);
extern "C" __global__ void hc_post_vec(const bf*, const float*, const float*, float*, u32, u32);
extern "C" __global__ void hc_pre_stage_vec(const float*, const bf*, float*, u32, u32, float);
extern "C" __global__ void hc_post_stage_vec(const bf*, float*, const float*, const bf*, float*, u32, u32,
                                             float);
extern "C" __global__ void moe_weighted_sum_blend(bf*, const bf*, const float*, const bf*, const bf*, const bf*,
                                                  u32, u32, u32);
extern "C" __global__ void moe_batched_blend(bf*, const bf*, const bf*, const bf*, u32, u32);
extern "C" __global__ void moe_blend_hc_post(bf*, const bf*, const bf*, const bf*, float*, const float*, u32,
                                             u32);

static bool g_pdl = false;

// Launch as serving does: with programmatic stream serialization under PDL.
template <typename... Args>
static void launch(void (*kernel)(Args...), dim3 grid, unsigned block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = g_pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

__global__ void spin(long long ns) {
    long long t0 = clock64();
    while (clock64() - t0 < ns) {}
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
    void copy_from(const Dev<T>& o) { CK(cudaMemcpy(p, o.p, n * sizeof(T), cudaMemcpyDeviceToDevice)); }
    void fill(unsigned char b) { CK(cudaMemset(p, b, n * sizeof(T))); }
};

static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 1234);

static unsigned short tobf(float f) {
    bf b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}

// Random values with zeros, -0.0 and a few large entries mixed in.
static float rnd(float scale) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const unsigned k = g_rng() % 64;
    if (k == 0) return 0.0f;
    if (k == 1) return -0.0f;
    if (k == 2) return 30.0f * scale * u(g_rng);
    return scale * u(g_rng);
}
static std::vector<unsigned short> rbf(size_t n, float scale) {
    std::vector<unsigned short> v(n);
    for (auto& x : v) x = tobf(rnd(scale));
    return v;
}
static std::vector<float> rf(size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = rnd(scale);
    return v;
}

template <typename T>
static bool same(const char* what, const Dev<T>& a, const Dev<T>& b) {
    std::vector<T> ha = a.get(), hb = b.get();
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < ha.size(); i++) {
        if (memcmp(&ha[i], &hb[i], sizeof(T)) != 0) { if (first == (size_t)-1) first = i; bad++; }
    }
    printf("  bitwise %-34s %s", what, bad ? "MISMATCH" : "ok");
    if (bad) printf("  (%zu of %zu differ, first at %zu)", bad, ha.size(), first);
    printf("\n");
    return bad == 0;
}

// ── timing: median us per chain over `reps` runs of `groups` chains ──
template <typename F>
static double time_chain(F&& chain, int groups, int reps, bool graph) {
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < groups; i++) chain(s, i);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&exec, g, 0));
        CK(cudaGraphDestroy(g));
    }
    std::vector<double> v;
    for (int r = 0; r < reps + 1; r++) {
        spin<<<1, 1, 0, s>>>(200000000LL / 100);
        CK(cudaEventRecord(a, s));
        if (graph) CK(cudaGraphLaunch(exec, s));
        else for (int i = 0; i < groups; i++) chain(s, i);
        CK(cudaEventRecord(b, s));
        CK(cudaStreamSynchronize(s));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (r) v.push_back(1000.0 * ms / groups);
    }
    if (exec) CK(cudaGraphExecDestroy(exec));
    CK(cudaStreamDestroy(s));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static cudaStream_t g_s = 0;  // the stream the arms launch on (set per chain)
template <typename... Args>
static void slaunch(void (*kernel)(Args...), dim3 grid, unsigned block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cfg.stream = g_s;
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = g_pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

// ══ GDN ══════════════════════════════════════════════════════════════════
struct GdnShape { u32 nk, nv; };
static const u32 D = 128, DCONV = 4, HID = 2560;

struct GdnLayer {  // one layer's weights and state
    Dev<float> h, conv, a_log, dt_bias;
    Dev<unsigned short> conv_w, ba_w, norm_w;
};
struct GdnIo {  // per-step activations and scratch, shared by the layers
    Dev<unsigned short> qkvz, ba_in, out;
    Dev<float> conv_out, gdn_out, gates;
};

static void gdn_alloc(const GdnShape& sh, GdnLayer& l, GdnIo* io) {
    const size_t conv_dim = 2 * sh.nk * D + sh.nv * D;
    l.h.alloc((size_t)sh.nv * D * D);
    l.conv.alloc(conv_dim * DCONV);
    l.a_log.alloc(sh.nv);
    l.dt_bias.alloc(sh.nv);
    l.conv_w.alloc(conv_dim * DCONV);
    l.ba_w.alloc((size_t)2 * sh.nv * HID);
    l.norm_w.alloc(D);
    l.h.put(rf(l.h.n, 0.05f));
    l.conv.put(rf(l.conv.n, 1.0f));
    l.a_log.put(rf(l.a_log.n, 1.0f));
    l.dt_bias.put(rf(l.dt_bias.n, 1.0f));
    l.conv_w.put(rbf(l.conv_w.n, 0.5f));
    l.ba_w.put(rbf(l.ba_w.n, 0.05f));
    l.norm_w.put(rbf(l.norm_w.n, 1.0f));
    if (io) {
        io->qkvz.alloc(conv_dim + sh.nv * D);
        io->ba_in.alloc(HID);
        io->out.alloc(sh.nv * D);
        io->conv_out.alloc(conv_dim + sh.nv * D);  // [conv | gdn out in the Z region]
        io->gdn_out.alloc(sh.nv * D);
        io->gates.alloc(2 * sh.nv);
    }
}

static void gdn_old(const GdnShape& sh, GdnLayer& l, GdnIo& io) {
    const u32 conv_dim = 2 * sh.nk * D + sh.nv * D, key_dim = sh.nk * D, n_ba = 2 * sh.nv;
    float* gates = io.gates.p;
    float* beta = io.gates.p + sh.nv;
    const bf* z = (const bf*)io.qkvz.p + conv_dim;
    slaunch(dense_gemv_ba_gates, dim3((n_ba + 3) / 4), 256, (const bf*)io.ba_in.p, (const bf*)l.ba_w.p,
            (const float*)l.a_log.p, (const float*)l.dt_bias.p, gates, beta, n_ba, HID, sh.nv / sh.nk);
    slaunch(causal_conv1d_update_l2norm_f32, dim3((conv_dim + 255) / 256), 256, l.conv.p, (const bf*)io.qkvz.p,
            (const bf*)l.conv_w.p, (const float*)nullptr, io.conv_out.p, 1u, conv_dim, DCONV, 2 * key_dim, D, 1e-6f);
    float* gdn_out = io.conv_out.p + conv_dim;
    slaunch(gated_delta_rule_decode_f32, dim3(sh.nv), 128, l.h.p, (const float*)io.conv_out.p,
            (const float*)io.conv_out.p + key_dim, (const float*)io.conv_out.p + 2 * key_dim, (const float*)gates,
            (const float*)beta, gdn_out, 1u, sh.nk, sh.nv, D, D);
    slaunch(gated_rms_norm_f32_input_sigmoid, dim3(sh.nv), 128, (const float*)gdn_out, z, (const bf*)l.norm_w.p,
            (bf*)io.out.p, D, 1e-6f, D, D);
}

static void gdn_new(const GdnShape& sh, GdnLayer& l, GdnIo& io) {
    const u32 conv_dim = 2 * sh.nk * D + sh.nv * D;
    const bf* z = (const bf*)io.qkvz.p + conv_dim;
    slaunch(qwen4exp_gdn_decode_fused, dim3(sh.nv), 128, l.h.p, l.conv.p, (const bf*)io.qkvz.p,
            (const bf*)l.conv_w.p, (const bf*)io.ba_in.p, (const bf*)l.ba_w.p, (const float*)l.a_log.p,
            (const float*)l.dt_bias.p, io.gates.p, io.gates.p + sh.nv, z, (const bf*)l.norm_w.p, (bf*)io.out.p,
            sh.nk, sh.nv, HID, D, 1e-6f, 1e-6f);
}

static bool gdn_check(const GdnShape& sh, int steps) {
    printf("gdn nk=%u nv=%u, %d steps\n", sh.nk, sh.nv, steps);
    GdnLayer la, lb;
    GdnIo ia, ib;
    gdn_alloc(sh, la, &ia);
    gdn_alloc(sh, lb, &ib);
    lb.h.copy_from(la.h); lb.conv.copy_from(la.conv); lb.a_log.copy_from(la.a_log);
    lb.dt_bias.copy_from(la.dt_bias); lb.conv_w.copy_from(la.conv_w); lb.ba_w.copy_from(la.ba_w);
    lb.norm_w.copy_from(la.norm_w);
    bool ok = true;
    for (int st = 0; st < steps; st++) {
        auto q = rbf(ia.qkvz.n, st == 0 ? 3.0f : 1.0f);
        auto x = rbf(ia.ba_in.n, 1.0f);
        ia.qkvz.put(q); ib.qkvz.put(q);
        ia.ba_in.put(x); ib.ba_in.put(x);
        ia.out.fill(0x7F); ib.out.fill(0x7F);
        ia.gates.fill(0x7F); ib.gates.fill(0x7F);
        gdn_old(sh, la, ia);
        gdn_new(sh, lb, ib);
        CK(cudaDeviceSynchronize());
        if (!same("  normed out", ia.out, ib.out)) {
            ok = false;
            auto ho = ia.out.get(), hn = ib.out.get();
            auto z = ib.qkvz.get();
            const size_t conv_dim = 2 * sh.nk * D + sh.nv * D;
            for (size_t i = 0; i < ho.size(); i++)
                if (ho[i] != hn[i])
                    printf("    [%zu] old %04x new %04x  z %04x\n", i, ho[i], hn[i], z[conv_dim + i]);
        }
        ok &= same("  gate+beta", ia.gates, ib.gates);
        ok &= same("  conv state", la.conv, lb.conv);
        ok &= same("  recurrence state", la.h, lb.h);
    }
    return ok;
}

// ══ mHC seam ═════════════════════════════════════════════════════════════
static const u32 HC = 4, HCD = HC * HID, SPLIT = 8, POST_BLOCK = 64;
struct Seam {
    Dev<float> streams, inj, normed;
    Dev<unsigned short> block_out, norm_w;
};
static void seam_alloc(Seam& s, unsigned T) {
    s.streams.alloc((size_t)T * HCD);
    s.inj.alloc(T * HC);
    s.normed.alloc((size_t)T * HCD);
    s.block_out.alloc((size_t)T * HID);
    s.norm_w.alloc(HCD);
}
static void seam_old(Seam& s, unsigned T) {
    slaunch(hc_post_vec, dim3(T, (HID / 4 + POST_BLOCK - 1) / POST_BLOCK), POST_BLOCK, (const bf*)s.block_out.p,
            (const float*)s.streams.p, (const float*)s.inj.p, s.streams.p, HID, HC);
    slaunch(hc_pre_stage_vec, dim3(T, SPLIT), 1024, (const float*)s.streams.p, (const bf*)s.norm_w.p, s.normed.p,
            HID, HC, 1e-6f);
}
static void seam_new(Seam& s, unsigned T) {
    slaunch(hc_post_stage_vec, dim3(T, SPLIT), 1024, (const bf*)s.block_out.p, s.streams.p,
            (const float*)s.inj.p, (const bf*)s.norm_w.p, s.normed.p, HID, HC, 1e-6f);
}
static bool seam_check(unsigned T, int steps) {
    printf("seam T=%u, %d steps\n", T, steps);
    Seam a, b;
    seam_alloc(a, T);
    seam_alloc(b, T);
    a.streams.put(rf(a.streams.n, 2.0f));
    b.streams.copy_from(a.streams);
    a.norm_w.put(rbf(a.norm_w.n, 0.5f));
    b.norm_w.copy_from(a.norm_w);
    bool ok = true;
    for (int st = 0; st < steps; st++) {
        auto x = rbf(a.block_out.n, 1.0f);
        auto w = rf(a.inj.n, 2.0f);
        a.block_out.put(x); b.block_out.put(x);
        a.inj.put(w); b.inj.put(w);
        a.normed.fill(0x7F); b.normed.fill(0x7F);
        seam_old(a, T);
        seam_new(b, T);
        CK(cudaDeviceSynchronize());
        ok &= same("  highway", a.streams, b.streams);
        ok &= same("  normed", a.normed, b.normed);
    }
    return ok;
}

// ══ MoE (EP) tail ════════════════════════════════════════════════════════
// The routed sum over the local slots, the expert all-reduce (not run here:
// both arms see the same reduced row), the gated shared-expert blend and the
// layer's mHC post. Old: memset zero row -> moe_weighted_sum_blend ->
// moe_batched_blend -> hc_post_vec. New: moe_weighted_sum_blend (null shared
// row) -> moe_blend_hc_post.
static const u32 TOPK = 10;
struct Moe {
    Dev<unsigned short> out, expert_out, shared_out, input, gate_w, zero;
    Dev<float> w, streams, inj;
};
static void moe_alloc(Moe& m) {
    m.out.alloc(HID); m.expert_out.alloc(TOPK * HID); m.shared_out.alloc(HID); m.input.alloc(HID);
    m.gate_w.alloc(HID); m.zero.alloc(HID); m.w.alloc(TOPK); m.streams.alloc(HCD); m.inj.alloc(HC);
}
static void moe_old(Moe& m) {
    CK(cudaMemsetAsync(m.zero.p, 0, HID * 2, g_s));
    slaunch(moe_weighted_sum_blend, dim3(HID / 256), 256, (bf*)m.out.p, (const bf*)m.expert_out.p,
            (const float*)m.w.p, (const bf*)m.zero.p, (const bf*)m.input.p, (const bf*)m.gate_w.p, HID, TOPK, HID);
    slaunch(moe_batched_blend, dim3(1), 256, (bf*)m.out.p, (const bf*)m.shared_out.p, (const bf*)m.input.p,
            (const bf*)m.gate_w.p, HID, 1u);
    slaunch(hc_post_vec, dim3(1, (HID / 4 + POST_BLOCK - 1) / POST_BLOCK), POST_BLOCK, (const bf*)m.out.p,
            (const float*)m.streams.p, (const float*)m.inj.p, m.streams.p, HID, HC);
}
static void moe_new(Moe& m) {
    slaunch(moe_weighted_sum_blend, dim3(HID / 256), 256, (bf*)m.out.p, (const bf*)m.expert_out.p,
            (const float*)m.w.p, (const bf*)nullptr, (const bf*)m.input.p, (const bf*)m.gate_w.p, HID, TOPK, HID);
    slaunch(moe_blend_hc_post, dim3(1, (HID + 1023) / 1024), 256, (bf*)m.out.p, (const bf*)m.shared_out.p,
            (const bf*)m.input.p, (const bf*)m.gate_w.p, m.streams.p, (const float*)m.inj.p, HID, HC);
}
static bool moe_check(int steps) {
    printf("moe EP tail, %d steps\n", steps);
    Moe a, b;
    moe_alloc(a);
    moe_alloc(b);
    a.streams.put(rf(a.streams.n, 2.0f));
    b.streams.copy_from(a.streams);
    a.gate_w.put(rbf(HID, 0.05f));
    b.gate_w.copy_from(a.gate_w);
    bool ok = true;
    for (int st = 0; st < steps; st++) {
        auto eo = rbf(TOPK * HID, 1.0f), so = rbf(HID, 1.0f), in = rbf(HID, 1.0f);
        auto w = rf(TOPK, 0.3f), inj = rf(HC, 2.0f);
        // Rows that sum to -0.0 in the routed part: all-(-0.0) expert outputs.
        if (st == 1) for (u32 j = 0; j < 64; j++) for (u32 e = 0; e < TOPK; e++) eo[e * HID + j] = 0x8000;
        a.expert_out.put(eo); b.expert_out.put(eo);
        a.shared_out.put(so); b.shared_out.put(so);
        a.input.put(in); b.input.put(in);
        a.w.put(w); b.w.put(w);
        a.inj.put(inj); b.inj.put(inj);
        a.zero.fill(0x7F); b.zero.fill(0x7F);
        a.out.fill(0x7F); b.out.fill(0x7F);
        g_s = 0;
        moe_old(a);
        moe_new(b);
        CK(cudaDeviceSynchronize());
        ok &= same("  moe output", a.out, b.out);
        ok &= same("  highway", a.streams, b.streams);
    }
    return ok;
}

// ══ GDN layer segment ════════════════════════════════════════════════════
// What a TP2 GDN layer launches from its QKVZ projection to the MoE site's
// collapse, as serving does (BF16 GDN projections, ATLAS_QWEN4EXP_HC_FAST):
//   dense_gemv_bf16 (qkvz) -> [4 small kernels | fused] -> dense_gemv_bf16
//   (out_proj; the TP all-reduce is not run) -> [hc_post_vec + hc_pre_stage_vec
//   | hc_post_stage_vec] -> hc_pre_down_vec -> hc_pre_finish_vec
// `pdl_small`: whether the four small GDN kernels launch with PDL when the
// rest do (to price listing them).
extern "C" __global__ void dense_gemv_bf16(const bf*, const bf*, bf*, u32, u32);
extern "C" __global__ void hc_pre_down_vec(const float*, const bf*, const bf*, float*, float*, u32, u32, u32, u32);
extern "C" __global__ void hc_pre_finish_vec(const float*, const float*, const bf*, bf*, u32, u32, u32);
static const u32 RANK = 320;
struct Seg {
    GdnLayer g;
    Dev<unsigned short> qkvz_w, out_w, down_w, up_w, inj_w, hc_norm_w;
};
struct SegIo {
    GdnIo g;
    Dev<unsigned short> x, ssm_out, y;
    Dev<float> streams, post, normed, low;
};
static const GdnShape SEG_SH = {8, 24};
static void seg_alloc(Seg& s) {
    gdn_alloc(SEG_SH, s.g, nullptr);
    const size_t qkvz_n = 2 * SEG_SH.nk * D + 2 * SEG_SH.nv * D;
    s.qkvz_w.alloc(qkvz_n * HID); s.qkvz_w.put(rbf(s.qkvz_w.n, 0.02f));
    s.out_w.alloc((size_t)HID * SEG_SH.nv * D); s.out_w.put(rbf(s.out_w.n, 0.02f));
    s.down_w.alloc((size_t)RANK * HCD); s.down_w.put(rbf(s.down_w.n, 0.01f));
    s.up_w.alloc((size_t)RANK * HCD); s.up_w.put(rbf(s.up_w.n, 0.01f));
    s.inj_w.alloc((size_t)HC * HCD); s.inj_w.put(rbf(s.inj_w.n, 0.01f));
    s.hc_norm_w.alloc(HCD); s.hc_norm_w.put(rbf(HCD, 0.5f));
}
static void seg_io_alloc(SegIo& io) {
    gdn_alloc(SEG_SH, *new GdnLayer, &io.g);
    io.x.alloc(HID); io.x.put(rbf(HID, 1.0f));
    io.ssm_out.alloc(HID); io.y.alloc(HID);
    io.streams.alloc(HCD); io.streams.put(rf(HCD, 1.0f));
    io.post.alloc(HC); io.post.put(rf(HC, 0.001f));
    io.normed.alloc(HCD); io.low.alloc(RANK);
}
static void seg_run(Seg& s, SegIo& io, bool fuse, bool pdl, bool pdl_small) {
    const u32 conv_dim = 2 * SEG_SH.nk * D + SEG_SH.nv * D;
    const u32 qkvz_n = conv_dim + SEG_SH.nv * D;
    g_pdl = pdl;
    slaunch(dense_gemv_bf16, dim3(qkvz_n / 4), 256, (const bf*)io.x.p, (const bf*)s.qkvz_w.p, (bf*)io.g.qkvz.p,
            qkvz_n, HID);
    io.g.ba_in.p = io.x.p;  // the BA GEMV reads the mixer input
    if (fuse) {
        gdn_new(SEG_SH, s.g, io.g);
    } else {
        g_pdl = pdl && pdl_small;
        gdn_old(SEG_SH, s.g, io.g);
        g_pdl = pdl;
    }
    slaunch(dense_gemv_bf16, dim3(HID / 4), 256, (const bf*)io.g.out.p, (const bf*)s.out_w.p, (bf*)io.ssm_out.p,
            HID, SEG_SH.nv * D);
    if (fuse) {
        slaunch(hc_post_stage_vec, dim3(1, SPLIT), 1024, (const bf*)io.ssm_out.p, io.streams.p,
                (const float*)io.post.p, (const bf*)s.hc_norm_w.p, io.normed.p, HID, HC, 1e-6f);
    } else {
        slaunch(hc_post_vec, dim3(1, (HID / 4 + POST_BLOCK - 1) / POST_BLOCK), POST_BLOCK, (const bf*)io.ssm_out.p,
                (const float*)io.streams.p, (const float*)io.post.p, io.streams.p, HID, HC);
        slaunch(hc_pre_stage_vec, dim3(1, SPLIT), 1024, (const float*)io.streams.p, (const bf*)s.hc_norm_w.p,
                io.normed.p, HID, HC, 1e-6f);
    }
    slaunch(hc_pre_down_vec, dim3(((RANK + HC) * 16 + 127) / 128), 128, (const float*)io.normed.p,
            (const bf*)s.down_w.p, (const bf*)s.inj_w.p, io.low.p, io.post.p, HID, HC, RANK, 1u);
    cudaLaunchConfig_t cfg = {};  // dynamic shared: the finish kernel stages `low`
    cfg.gridDim = dim3((HCD / 4 + 127) / 128);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = RANK * 4;
    cfg.stream = g_s;
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, hc_pre_finish_vec, (const float*)io.normed.p, (const float*)io.low.p,
                          (const bf*)s.up_w.p, (bf*)io.y.p, HID, RANK, 1u));
}

int main(int argc, char** argv) {
    const int copies = argc > 1 ? atoi(argv[1]) : 36;
    const int groups = argc > 2 ? atoi(argv[2]) : 64;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    const int steps = argc > 4 ? atoi(argv[4]) : 6;
    const GdnShape shapes[] = {{8, 24}, {16, 48}};

    bool ok = true;
    for (int pdl = 0; pdl < 2; pdl++) {
        g_pdl = pdl;
        printf("== check, pdl=%d\n", pdl);
        for (const auto& sh : shapes) ok &= gdn_check(sh, steps);
        for (unsigned T = 1; T <= 4; T++) ok &= seam_check(T, steps);
        ok &= moe_check(steps);
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    // ── timing ──
    for (const auto& sh : shapes) {
        std::vector<GdnLayer> ls(copies);
        GdnIo io;
        for (int c = 0; c < copies; c++) gdn_alloc(sh, ls[c], c == 0 ? &io : nullptr);
        io.qkvz.put(rbf(io.qkvz.n, 1.0f));
        io.ba_in.put(rbf(io.ba_in.n, 1.0f));
        for (int graph = 0; graph < 2; graph++) {
            for (int pdl = 0; pdl < 2; pdl++) {
                g_pdl = pdl;
                const double o = time_chain([&](cudaStream_t s, int i) { g_s = s; gdn_old(sh, ls[i % copies], io); },
                                            groups, reps, graph);
                const double n = time_chain([&](cudaStream_t s, int i) { g_s = s; gdn_new(sh, ls[i % copies], io); },
                                            groups, reps, graph);
                printf("time gdn  nk=%2u nv=%2u %-5s pdl=%d  4 kernels %7.2f us   fused %7.2f us\n", sh.nk, sh.nv,
                       graph ? "graph" : "eager", pdl, o, n);
            }
        }
    }
    for (unsigned T = 1; T <= 2; T++) {
        std::vector<Seam> ss(copies);
        for (auto& s : ss) {
            seam_alloc(s, T);
            s.streams.put(rf(s.streams.n, 1.0f));
            s.norm_w.put(rbf(s.norm_w.n, 0.5f));
            s.block_out.put(rbf(s.block_out.n, 1.0f));
            s.inj.put(rf(s.inj.n, 0.001f));
        }
        for (int graph = 0; graph < 2; graph++) {
            for (int pdl = 0; pdl < 2; pdl++) {
                g_pdl = pdl;
                const double o = time_chain([&](cudaStream_t s, int i) { g_s = s; seam_old(ss[i % copies], T); },
                                            groups, reps, graph);
                const double n = time_chain([&](cudaStream_t s, int i) { g_s = s; seam_new(ss[i % copies], T); },
                                            groups, reps, graph);
                printf("time seam T=%u %-5s pdl=%d  post+stage %7.2f us   fused %7.2f us\n", T,
                       graph ? "graph" : "eager", pdl, o, n);
            }
        }
    }
    {
        std::vector<Moe> ms(copies);
        for (auto& m : ms) {
            moe_alloc(m);
            m.expert_out.put(rbf(m.expert_out.n, 1.0f));
            m.shared_out.put(rbf(HID, 1.0f));
            m.input.put(rbf(HID, 1.0f));
            m.gate_w.put(rbf(HID, 0.05f));
            m.w.put(rf(TOPK, 0.1f));
            m.streams.put(rf(m.streams.n, 1.0f));
            m.inj.put(rf(HC, 0.001f));
        }
        for (int graph = 0; graph < 2; graph++) {
            for (int pdl = 0; pdl < 2; pdl++) {
                g_pdl = pdl;
                const double o = time_chain([&](cudaStream_t s, int i) { g_s = s; moe_old(ms[i % copies]); },
                                            groups, reps, graph);
                const double n = time_chain([&](cudaStream_t s, int i) { g_s = s; moe_new(ms[i % copies]); },
                                            groups, reps, graph);
                printf("time moe  %-5s pdl=%d  memset+wsum+blend+post %7.2f us   wsum+fused %7.2f us\n",
                       graph ? "graph" : "eager", pdl, o, n);
            }
        }
    }
    {
        const int seg_copies = std::min(copies, 8);
        std::vector<Seg> ss(seg_copies);
        for (auto& s : ss) seg_alloc(s);
        SegIo io;
        seg_io_alloc(io);
        struct Arm { const char* name; bool fuse, pdl, pdl_small; };
        const Arm arms[] = {{"base", false, false, false}, {"pdl", false, true, true},
                            {"pdl (small gdn eager)", false, true, false}, {"fuse", true, false, false},
                            {"fuse+pdl", true, true, true}};
        for (int graph = 0; graph < 2; graph++) {
            for (const auto& a : arms) {
                const double t = time_chain(
                    [&](cudaStream_t s, int i) { g_s = s; seg_run(ss[i % seg_copies], io, a.fuse, a.pdl, a.pdl_small); },
                    groups, reps, graph);
                printf("time layer segment %-5s %-22s %8.2f us\n", graph ? "graph" : "eager", a.name, t);
            }
        }
        g_pdl = false;
    }
    return 0;
}
