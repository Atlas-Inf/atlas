// SPDX-License-Identifier: AGPL-3.0-only
// Per-row bit parity and cost of the qwen4_exp exact small-kernel batching
// (ATLAS_QWEN4EXP_BATCH_SMALL, crates/spark-model/src/layers/ops/qwen4exp_batch_small.rs):
// one launch over every row of a batched decode or verify step where the
// batched step ran the single-row kernel once per row.
//
//   MoE top-k     moe_topk_softmax_rows (R blocks)        vs R x moe_topk_softmax
//                 (512 BF16 logits, top-10, normalized; ties planted inside a
//                 warp, across warps and across a thread's two experts;
//                 moe_topk_softmax_batched is shown breaking them differently)
//   MoE blend     moe_weighted_sum_blend_rows (x, R)      vs R x moe_weighted_sum_blend
//                 (per-row shared rows, the EP zero row at stride 0, a null
//                 shared row; with and without the shared-expert gate)
//   GDN step      qwen4exp_gdn_decode_fused_rows (vh, R)  vs per sequence the
//                 four-kernel chain dense_gemv_ba_gates -> causal_conv1d_update_
//                 l2norm_f32 -> gated_delta_rule_decode_f32 -> gated_rms_norm_
//                 f32_input_sigmoid, and vs qwen4exp_gdn_decode_fused; each
//                 sequence on its own recurrence and conv state, `steps`
//                 recurrent steps, every byte left behind (normed rows, gates,
//                 betas, conv windows, recurrence states). TP2 (8 key / 24
//                 value heads) and TP1 (16 / 48).
//   GDN verify    qwen4exp_gdn_verify_fused_rows (vh, seqs) vs the exact MTP
//                 verify arm per sequence and token: causal_conv1d_update_
//                 l2norm_f32, conv state -> rollback slot, gated_delta_rule_
//                 decode_f32, gated_rms_norm_f32_input_sigmoid, H -> rollback
//                 slot (slots for tokens 0..k-2); 1..12 sequences, k = 2..4
//                 and ragged 1..4; normed rows, final states and every slot.
//   mHC / MoE     the decode-fuse seam hc_post_stage_vec and EP tail
//                 moe_blend_hc_post at T = 1..8 rows vs hc_post_vec +
//                 hc_pre_stage_vec and moe_batched_blend + hc_post_vec.
//
// R = 1..8, 16, 32 (16 and 32 as 8-row launches for the GDN step, as the
// runtime chunks them). Then GPU time per layer at R rows, the old launches
// against the new one, inside a CUDA graph (as serving replays decode) over
// 36 layer copies of weights and per-sequence states (DRAM-streamed).
//
// Build/run: scripts/dev/qwen4exp_batch_exact_bench.sh small-check|small-time
// (repo root, GB10, inside atlas-release-builder:1.93.1 for the runtime's
// CUDA 13.0). PTX built with the production flags, JIT-loaded as the server
// loads it (qwen4exp_ptx_harness.h).
#include "qwen4exp_ptx_harness.h"

#include <random>

typedef unsigned int u32;
typedef unsigned short bf;

static std::string g_dir = ".";
static int g_fail = 0;
static std::mt19937 g_rng(getenv("SEED") ? atoi(getenv("SEED")) : 2718);
static cudaStream_t g_s = nullptr;

static PtxModule& mod(const char* stem) {
    static std::vector<std::pair<std::string, PtxModule*>> mods;
    for (auto& m : mods) if (m.first == stem) return *m.second;
    PtxModule* m = new PtxModule;
    m->load(g_dir + "/" + stem + ".ptx");
    mods.push_back({stem, m});
    return *m;
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
static std::vector<bf> rbf(size_t n, float scale) {
    std::vector<bf> v(n);
    for (auto& x : v) x = f2bf(rnd(scale));
    return v;
}
static std::vector<float> rf(size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = rnd(scale);
    return v;
}

template <typename T>
static bool same(const std::string& what, const Buf<T>& a, const Buf<T>& b, size_t count) {
    std::vector<T> ha = a.get(), hb = b.get();
    ha.resize(count);
    hb.resize(count);
    const size_t bad = diff_bytes(ha, hb);
    if (bad) {
        printf("  MISMATCH %-58s %zu bytes differ\n", what.c_str(), bad);
        g_fail++;
    }
    return bad == 0;
}

static const u32 ROWS[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32};

// ══ MoE router top-k ═════════════════════════════════════════════════════
static const u32 NE = 512, TOPK = 10, H = 2560;

// Logits with planted exact ties among the leaders: equal values at experts
// i and i+1 (same warp, neighbouring lanes), i and i+32*k (other warps), and
// i and i+256 (one thread's two experts).
static std::vector<bf> tied_logits(u32 rows) {
    std::vector<bf> v = rbf((size_t)rows * NE, 2.0f);
    std::uniform_int_distribution<u32> e(0, NE - 1);
    for (u32 r = 0; r < rows; r++) {
        bf* row = v.data() + (size_t)r * NE;
        for (int t = 0; t < 6; t++) {
            const u32 a = e(g_rng);
            static const u32 OFF[] = {1, 32, 96, 256, 255, 33};
            const u32 b = (a + OFF[t]) % NE;
            const bf top = f2bf(8.0f + 0.25f * (g_rng() % 4));
            row[a] = top;
            row[b] = top;
        }
    }
    return v;
}

static void topk_check() {
    CUfunction one = mod("moe_topk").fn("moe_topk_softmax");
    CUfunction rows_k = mod("moe_topk").fn("moe_topk_softmax_rows");
    CUfunction batched = mod("moe_topk").fn("moe_topk_softmax_batched");
    int batched_diff = 0, bad = 0;
    for (u32 rows : ROWS) {
        for (int rep = 0; rep < 8; rep++) {
            for (u32 norm : {1u, 0u}) {
                Buf<bf> logits;
                logits.alloc((size_t)rows * NE);
                logits.put(tied_logits(rows));
                Buf<u32> ri, gi, bi;
                Buf<float> rw, gw, bw;
                for (auto* b : {&ri, &gi, &bi}) { b->alloc(rows * TOPK); b->fill(0x55); }
                for (auto* b : {&rw, &gw, &bw}) { b->alloc(rows * TOPK); b->fill(0x55); }
                u32 ne = NE, k = TOPK, stride = NE;
                for (u32 r = 0; r < rows; r++) {
                    Args a;
                    a.add(logits.p + (size_t)r * NE).add(ri.p + r * TOPK).add(rw.p + r * TOPK)
                        .add(ne).add(k).add(norm);
                    launch(one, dim3(1), dim3(256), 0, a);
                }
                Args a;
                a.add(logits.p).add(gi.p).add(gw.p).add(ne).add(k).add(norm).add(stride);
                launch(rows_k, dim3(rows), dim3(256), 0, a);
                Args b;
                b.add(logits.p).add(bi.p).add(bw.p).add(ne).add(k).add(norm);
                launch(batched, dim3(rows), dim3(256), 0, b);
                CK(cudaDeviceSynchronize());
                char what[96];
                snprintf(what, sizeof what, "moe_topk_softmax_rows R=%u norm=%u indices", rows, norm);
                bad += !same(what, ri, gi, rows * TOPK);
                snprintf(what, sizeof what, "moe_topk_softmax_rows R=%u norm=%u weights", rows, norm);
                bad += !same(what, rw, gw, rows * TOPK);
                batched_diff += diff_bytes(ri.get(), bi.get()) != 0;
                for (auto* b : {&ri, &gi, &bi}) b->free_();
                for (auto* b : {&rw, &gw, &bw}) b->free_();
                logits.free_();
            }
        }
    }
    printf("  %s moe_topk_softmax_rows: R = 1..8, 16, 32 x 8 tie-planted draws x norm 0/1, "
           "indices + weights byte-equal to R x moe_topk_softmax\n", bad ? "BAD" : "ok ");
    printf("       (for contrast: moe_topk_softmax_batched picked other experts in %d of %zu "
           "of those draws)\n", batched_diff, sizeof(ROWS) / sizeof(ROWS[0]) * 16);
}

// ══ MoE weighted sum + shared blend ══════════════════════════════════════
static void blend_check() {
    CUfunction one = mod("moe_expert_gemv").fn("moe_weighted_sum_blend");
    CUfunction rows_k = mod("moe_expert_gemv").fn("moe_weighted_sum_blend_rows");
    int bad = 0;
    // shared: 0 per-row rows, 1 one zero row at stride 0 (EP), 2 null.
    for (u32 rows : ROWS) {
        for (int shared = 0; shared < 3; shared++) {
            for (bool gated : {true, false}) {
                Buf<bf> eo, sh, x, gw, ref, got;
                Buf<float> w;
                eo.alloc((size_t)rows * TOPK * H); eo.put(rbf(eo.n, 1.0f));
                sh.alloc((size_t)rows * H);
                sh.put(shared == 0 ? rbf(sh.n, 1.0f) : std::vector<bf>(sh.n, 0));
                x.alloc((size_t)rows * H); x.put(rbf(x.n, 1.0f));
                gw.alloc(H); gw.put(rbf(H, 0.05f));
                w.alloc(rows * TOPK); w.put(rf(w.n, 0.3f));
                ref.alloc((size_t)rows * H); ref.fill(0x55);
                got.alloc((size_t)rows * H); got.fill(0x55);
                bf* shp = shared == 2 ? nullptr : sh.p;
                bf* g = gated ? gw.p : nullptr;
                u32 h = H, k = TOPK, kk = H, stride = shared == 0 ? H : 0;
                for (u32 r = 0; r < rows; r++) {
                    Args a;
                    a.add(ref.p + (size_t)r * H).add(eo.p + (size_t)r * TOPK * H).add(w.p + r * TOPK)
                        .add(shp ? shp + (size_t)r * stride : shp).add(x.p + (size_t)r * H).add(g)
                        .add(h).add(k).add(kk);
                    launch(one, dim3((H + 255) / 256), dim3(256), 0, a);
                }
                Args a;
                a.add(got.p).add(eo.p).add(w.p).add(shp).add(x.p).add(g).add(h).add(k).add(kk).add(stride);
                launch(rows_k, dim3((H + 255) / 256, rows), dim3(256), 0, a);
                CK(cudaDeviceSynchronize());
                char what[96];
                snprintf(what, sizeof what, "moe_weighted_sum_blend_rows R=%u shared=%d gated=%d", rows,
                         shared, gated);
                bad += !same(what, ref, got, (size_t)rows * H);
                for (auto* b : {&eo, &sh, &x, &gw, &ref, &got}) b->free_();
                w.free_();
            }
        }
    }
    printf("  %s moe_weighted_sum_blend_rows: R = 1..8, 16, 32 x shared rows / EP zero row / null x "
           "gate on/off, every output byte equal to R x moe_weighted_sum_blend\n",
           bad ? "BAD" : "ok ");
}

// ══ GDN decode step ══════════════════════════════════════════════════════
static const u32 D = 128, DCONV = 4, ROWS_MAX = 8;
struct GdnShape { u32 nk, nv; const char* name; };
static u32 conv_dim(const GdnShape& s) { return 2 * s.nk * D + s.nv * D; }
static u32 qkvz_dim(const GdnShape& s) { return conv_dim(s) + s.nv * D; }

struct GdnWeights { Buf<bf> conv_w, ba_w, norm_w; Buf<float> a_log, dt_bias; };
struct GdnSeq { Buf<float> h, conv; };

static void gdn_weights(const GdnShape& s, GdnWeights& w) {
    w.conv_w.alloc((size_t)conv_dim(s) * DCONV); w.conv_w.put(rbf(w.conv_w.n, 0.5f));
    w.ba_w.alloc((size_t)2 * s.nv * H); w.ba_w.put(rbf(w.ba_w.n, 0.05f));
    w.norm_w.alloc(D); w.norm_w.put(rbf(D, 1.0f));
    w.a_log.alloc(s.nv); w.a_log.put(rf(s.nv, 1.0f));
    w.dt_bias.alloc(s.nv); w.dt_bias.put(rf(s.nv, 1.0f));
}
static void gdn_seq(const GdnShape& s, GdnSeq& q, bool init) {
    q.h.alloc((size_t)s.nv * D * D);
    q.conv.alloc((size_t)conv_dim(s) * DCONV);
    if (init) { q.h.put(rf(q.h.n, 0.05f)); q.conv.put(rf(q.conv.n, 1.0f)); }
}

// Step activations: rows of the mixer input, the qkvz projection, and the
// outputs (gates [rows, 2nv], normed [rows, nv*128]); conv_out/gdn_out are
// the chain's FP32 scratch.
struct GdnIo { Buf<bf> ba_in, qkvz, out; Buf<float> gates, conv_out; };
static void gdn_io(const GdnShape& s, GdnIo& io, u32 rows) {
    io.ba_in.alloc((size_t)rows * H);
    io.qkvz.alloc((size_t)rows * qkvz_dim(s));
    io.out.alloc((size_t)rows * s.nv * D);
    io.gates.alloc((size_t)rows * 2 * s.nv);
    io.conv_out.alloc(qkvz_dim(s));
}

struct GdnKernels { CUfunction ba, conv, rec, norm, fused, rows; };
static GdnKernels gdn_kernels() {
    return {mod("ssm_preprocess").fn("dense_gemv_ba_gates"),
            mod("causal_conv1d").fn("causal_conv1d_update_l2norm_f32"),
            mod("gated_delta_rule").fn("gated_delta_rule_decode_f32"),
            mod("rms_norm").fn("gated_rms_norm_f32_input_sigmoid"),
            mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_decode_fused"),
            mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_decode_fused_rows")};
}

// Sequence r's four-kernel chain (ssm_batched_recurrent.rs's per-sequence loop).
static void gdn_chain(const GdnKernels& k, const GdnShape& s, const GdnWeights& w, GdnSeq& q,
                      GdnIo& io, u32 r) {
    const u32 cd = conv_dim(s), key_dim = s.nk * D, n_ba = 2 * s.nv, hid = H, vpg = s.nv / s.nk;
    const u32 one = 1, dconv = DCONV, qk = 2 * key_dim, d = D;
    const float l2 = 1e-6f, eps = 1e-6f;
    bf* qkv = io.qkvz.p + (size_t)r * qkvz_dim(s);
    float* gates = io.gates.p + (size_t)r * 2 * s.nv;
    float* beta = gates + s.nv;
    float* conv_out = io.conv_out.p;
    float* gdn_out = conv_out + cd;
    { Args a; a.add(io.ba_in.p + (size_t)r * H).add(w.ba_w.p).add(w.a_log.p).add(w.dt_bias.p)
          .add(gates).add(beta).add(n_ba).add(hid).add(vpg);
      launch(k.ba, dim3((n_ba + 3) / 4), dim3(256), 0, a, g_s); }
    { Args a; a.add(q.conv.p).add(qkv).add(w.conv_w.p).add((const float*)nullptr).add(conv_out)
          .add(one).add(cd).add(dconv).add(qk).add(d).add(l2);
      launch(k.conv, dim3((cd + 255) / 256), dim3(256), 0, a, g_s); }
    { Args a; a.add(q.h.p).add(conv_out).add(conv_out + key_dim).add(conv_out + 2 * key_dim)
          .add(gates).add(beta).add(gdn_out).add(one).add(s.nk).add(s.nv).add(d).add(d);
      launch(k.rec, dim3(s.nv), dim3(128), 0, a, g_s); }
    { Args a; a.add(gdn_out).add(qkv + cd).add(w.norm_w.p).add(io.out.p + (size_t)r * s.nv * D)
          .add(d).add(eps).add(d).add(d);
      launch(k.norm, dim3(s.nv), dim3(128), 0, a, g_s); }
}

// Sequence r through the single-sequence fused kernel.
static void gdn_fused(const GdnKernels& k, const GdnShape& s, const GdnWeights& w, GdnSeq& q,
                      GdnIo& io, u32 r) {
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 hid = H, d = D;
    bf* qkv = io.qkvz.p + (size_t)r * qkvz_dim(s);
    float* gates = io.gates.p + (size_t)r * 2 * s.nv;
    Args a;
    a.add(q.h.p).add(q.conv.p).add(qkv).add(w.conv_w.p).add(io.ba_in.p + (size_t)r * H).add(w.ba_w.p)
        .add(w.a_log.p).add(w.dt_bias.p).add(gates).add(gates + s.nv).add(qkv + conv_dim(s))
        .add(w.norm_w.p).add(io.out.p + (size_t)r * s.nv * D).add(s.nk).add(s.nv).add(hid).add(d)
        .add(l2).add(eps);
    launch(k.fused, dim3(s.nv), dim3(128), 0, a, g_s);
}

struct RowStates { float* h[ROWS_MAX]; float* conv[ROWS_MAX]; };

// Rows [first, first + count) in one launch of the rows kernel.
static void gdn_rows(const GdnKernels& k, const GdnShape& s, const GdnWeights& w,
                     std::vector<GdnSeq>& seqs, GdnIo& io, u32 first, u32 count) {
    RowStates st = {};
    for (u32 i = 0; i < count; i++) { st.h[i] = seqs[first + i].h.p; st.conv[i] = seqs[first + i].conv.p; }
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 hid = H, d = D, stride = qkvz_dim(s);
    float* gates = io.gates.p + (size_t)first * 2 * s.nv;
    Args a;
    a.add(st).add(io.qkvz.p + (size_t)first * stride).add(w.conv_w.p).add(io.ba_in.p + (size_t)first * H)
        .add(w.ba_w.p).add(w.a_log.p).add(w.dt_bias.p).add(gates).add(w.norm_w.p)
        .add(io.out.p + (size_t)first * s.nv * D).add(s.nk).add(s.nv).add(hid).add(d).add(stride)
        .add(l2).add(eps);
    launch(k.rows, dim3(s.nv, count), dim3(128), 0, a, g_s);
}
static void gdn_rows_all(const GdnKernels& k, const GdnShape& s, const GdnWeights& w,
                         std::vector<GdnSeq>& seqs, GdnIo& io, u32 rows) {
    for (u32 first = 0; first < rows; first += ROWS_MAX)
        gdn_rows(k, s, w, seqs, io, first, std::min(ROWS_MAX, rows - first));
}

static bool gdn_check(const GdnShape& s, int steps) {
    const GdnKernels k = gdn_kernels();
    GdnWeights w;
    gdn_weights(s, w);
    int bad = 0;
    for (u32 rows : ROWS) {
        // Three arms on identical state: chain, per-sequence fused, rows.
        std::vector<GdnSeq> sa(rows), sb(rows), sc(rows);
        for (u32 r = 0; r < rows; r++) {
            gdn_seq(s, sa[r], true);
            gdn_seq(s, sb[r], false);
            gdn_seq(s, sc[r], false);
            for (GdnSeq* o : {&sb[r], &sc[r]}) {
                CK(cudaMemcpy(o->h.p, sa[r].h.p, sa[r].h.n * 4, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(o->conv.p, sa[r].conv.p, sa[r].conv.n * 4, cudaMemcpyDeviceToDevice));
            }
        }
        GdnIo ia, ib, ic;
        gdn_io(s, ia, rows); gdn_io(s, ib, rows); gdn_io(s, ic, rows);
        for (int st = 0; st < steps; st++) {
            auto q = rbf(ia.qkvz.n, st == 0 ? 3.0f : 1.0f);
            auto x = rbf(ia.ba_in.n, 1.0f);
            for (GdnIo* io : {&ia, &ib, &ic}) {
                io->qkvz.put(q); io->ba_in.put(x); io->out.fill(0x7F); io->gates.fill(0x7F);
            }
            for (u32 r = 0; r < rows; r++) gdn_chain(k, s, w, sa[r], ia, r);
            for (u32 r = 0; r < rows; r++) gdn_fused(k, s, w, sb[r], ib, r);
            gdn_rows_all(k, s, w, sc, ic, rows);
            CK(cudaDeviceSynchronize());
            char what[128];
            for (int arm = 0; arm < 2; arm++) {
                GdnIo& io = arm ? ib : ia;
                std::vector<GdnSeq>& ss = arm ? sb : sa;
                const char* vs = arm ? "fused" : "chain";
                snprintf(what, sizeof what, "%s R=%u step %d vs %s: normed rows", s.name, rows, st, vs);
                bad += !same(what, io.out, ic.out, io.out.n);
                snprintf(what, sizeof what, "%s R=%u step %d vs %s: gates+betas", s.name, rows, st, vs);
                bad += !same(what, io.gates, ic.gates, io.gates.n);
                for (u32 r = 0; r < rows; r++) {
                    snprintf(what, sizeof what, "%s R=%u step %d vs %s: seq %u conv", s.name, rows, st, vs, r);
                    bad += !same(what, ss[r].conv, sc[r].conv, sc[r].conv.n);
                    snprintf(what, sizeof what, "%s R=%u step %d vs %s: seq %u recurrence", s.name, rows, st, vs, r);
                    bad += !same(what, ss[r].h, sc[r].h, sc[r].h.n);
                }
            }
        }
        for (auto* v : {&sa, &sb, &sc}) for (auto& q : *v) { q.h.free_(); q.conv.free_(); }
        for (GdnIo* io : {&ia, &ib, &ic}) {
            io->ba_in.free_(); io->qkvz.free_(); io->out.free_(); io->gates.free_(); io->conv_out.free_();
        }
    }
    printf("  %s qwen4exp_gdn_decode_fused_rows %s: R = 1..8, 16, 32 x %d steps, every normed / "
           "gate / beta / conv / recurrence byte equal to the per-sequence four-kernel chain and "
           "to qwen4exp_gdn_decode_fused\n", bad ? "BAD" : "ok ", s.name, steps);
    return bad == 0;
}

// ══ GDN exact MTP verify ═════════════════════════════════════════════════
// Sequence s owns rows row0..row0+k-1 of the step (ragged k). The exact arm
// per token: conv kernel on the live conv state, conv state -> rollback slot
// t (t < k-1), recurrence on the live H, gated norm, H -> rollback slot t.
static const u32 KMAX = 4;
struct VerifySeq {
    GdnSeq st;
    Buf<float> h_snap[KMAX - 1], conv_snap[KMAX - 1];
    u32 row0 = 0, k = 1;
};
static void vseq_alloc(const GdnShape& s, VerifySeq& v, bool init) {
    gdn_seq(s, v.st, init);
    for (u32 t = 0; t + 1 < KMAX; t++) {
        v.h_snap[t].alloc((size_t)s.nv * D * D); v.h_snap[t].fill(0x7F);
        v.conv_snap[t].alloc((size_t)conv_dim(s) * DCONV); v.conv_snap[t].fill(0x7F);
    }
}
static void vseq_free(VerifySeq& v) {
    v.st.h.free_(); v.st.conv.free_();
    for (u32 t = 0; t + 1 < KMAX; t++) { v.h_snap[t].free_(); v.conv_snap[t].free_(); }
}
static void vseq_copy(VerifySeq& dst, const VerifySeq& src) {
    CK(cudaMemcpy(dst.st.h.p, src.st.h.p, src.st.h.n * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dst.st.conv.p, src.st.conv.p, src.st.conv.n * 4, cudaMemcpyDeviceToDevice));
    dst.row0 = src.row0;
    dst.k = src.k;
}

// The exact arm for sequence v (trait_decode_batched_conv_gdn_exact.rs,
// FP32 conv, no fused verify conv on this target).
static void verify_chain(const GdnKernels& k, const GdnShape& s, const GdnWeights& w, VerifySeq& v,
                         GdnIo& io) {
    const u32 cd = conv_dim(s), key_dim = s.nk * D;
    const u32 one = 1, dconv = DCONV, qk = 2 * key_dim, d = D;
    const float l2 = 1e-6f, eps = 1e-6f;
    for (u32 t = 0; t < v.k; t++) {
        const size_t row = v.row0 + t;
        bf* qkv = io.qkvz.p + row * qkvz_dim(s);
        float* gates = io.gates.p + row * 2 * s.nv;
        float* conv_out = io.conv_out.p;
        float* gdn_out = conv_out + cd;
        { Args a; a.add(v.st.conv.p).add(qkv).add(w.conv_w.p).add((const float*)nullptr).add(conv_out)
              .add(one).add(cd).add(dconv).add(qk).add(d).add(l2);
          launch(k.conv, dim3((cd + 255) / 256), dim3(256), 0, a, g_s); }
        if (t + 1 < v.k)
            CK(cudaMemcpyAsync(v.conv_snap[t].p, v.st.conv.p, v.st.conv.n * 4, cudaMemcpyDeviceToDevice, g_s));
        { Args a; a.add(v.st.h.p).add(conv_out).add(conv_out + key_dim).add(conv_out + 2 * key_dim)
              .add(gates).add(gates + s.nv).add(gdn_out).add(one).add(s.nk).add(s.nv).add(d).add(d);
          launch(k.rec, dim3(s.nv), dim3(128), 0, a, g_s); }
        { Args a; a.add(gdn_out).add(qkv + cd).add(w.norm_w.p).add(io.out.p + row * s.nv * D)
              .add(d).add(eps).add(d).add(d);
          launch(k.norm, dim3(s.nv), dim3(128), 0, a, g_s); }
        if (t + 1 < v.k)
            CK(cudaMemcpyAsync(v.h_snap[t].p, v.st.h.p, v.st.h.n * 4, cudaMemcpyDeviceToDevice, g_s));
    }
}

struct VerifyRowsArg {
    struct {
        float* h; float* conv; float* h_snap[KMAX - 1]; float* conv_snap[KMAX - 1];
        u32 row0, k;
    } seq[ROWS_MAX];
};

static void verify_rows(CUfunction fn, const GdnShape& s, const GdnWeights& w,
                        std::vector<VerifySeq>& seqs, GdnIo& io) {
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 d = D, stride = qkvz_dim(s);
    for (size_t first = 0; first < seqs.size(); first += ROWS_MAX) {
        const u32 count = (u32)std::min<size_t>(ROWS_MAX, seqs.size() - first);
        VerifyRowsArg t = {};
        for (u32 i = 0; i < count; i++) {
            VerifySeq& v = seqs[first + i];
            t.seq[i].h = v.st.h.p; t.seq[i].conv = v.st.conv.p;
            for (u32 j = 0; j + 1 < KMAX; j++) { t.seq[i].h_snap[j] = v.h_snap[j].p; t.seq[i].conv_snap[j] = v.conv_snap[j].p; }
            t.seq[i].row0 = v.row0; t.seq[i].k = v.k;
        }
        Args a;
        a.add(t).add(io.qkvz.p).add(w.conv_w.p).add(io.gates.p).add(w.norm_w.p).add(io.out.p)
            .add(s.nk).add(s.nv).add(d).add(stride).add(l2).add(eps);
        launch(fn, dim3(s.nv, count), dim3(128), 0, a, g_s);
    }
}

// Gate / beta rows as the BA kernels leave them: decay in (0, 1], beta in
// (0, 1), with exact 0 / 1 and out-of-range values mixed in (the recurrence
// clamps the decay).
static std::vector<float> gate_rows(size_t n) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) {
        const unsigned k = g_rng() % 32;
        x = k == 0 ? 0.0f : k == 1 ? 1.0f : k == 2 ? 1.5f : u(g_rng);
    }
    return v;
}

static bool verify_check(const GdnShape& s, int steps) {
    const GdnKernels k = gdn_kernels();
    CUfunction vk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_fused_rows");
    GdnWeights w;
    gdn_weights(s, w);
    int bad = 0;
    for (u32 nseq : {1u, 2u, 3u, 5u, 8u, 12u}) {
        for (int ragged = 0; ragged < 2; ragged++) {
            std::vector<VerifySeq> a(nseq), b(nseq);
            u32 rows = 0;
            for (u32 i = 0; i < nseq; i++) {
                vseq_alloc(s, a[i], true);
                vseq_alloc(s, b[i], false);
                a[i].k = ragged ? 1 + (g_rng() % KMAX) : 2 + (i % (KMAX - 1));
                a[i].row0 = rows;
                rows += a[i].k;
                vseq_copy(b[i], a[i]);
            }
            GdnIo ia, ib;
            gdn_io(s, ia, rows); gdn_io(s, ib, rows);
            for (int st = 0; st < steps; st++) {
                auto q = rbf(ia.qkvz.n, st == 0 ? 3.0f : 1.0f);
                auto g = gate_rows(ia.gates.n);
                for (GdnIo* io : {&ia, &ib}) { io->qkvz.put(q); io->gates.put(g); io->out.fill(0x7F); }
                for (auto& v : a) verify_chain(k, s, w, v, ia);
                verify_rows(vk, s, w, b, ib);
                CK(cudaDeviceSynchronize());
                char what[160];
                snprintf(what, sizeof what, "%s verify n=%u %s step %d: normed rows", s.name, nseq,
                         ragged ? "ragged" : "k=2..4", st);
                bad += !same(what, ia.out, ib.out, ia.out.n);
                for (u32 i = 0; i < nseq; i++) {
                    snprintf(what, sizeof what, "%s verify n=%u step %d seq %u (k=%u): recurrence", s.name, nseq, st, i, a[i].k);
                    bad += !same(what, a[i].st.h, b[i].st.h, a[i].st.h.n);
                    snprintf(what, sizeof what, "%s verify n=%u step %d seq %u (k=%u): conv", s.name, nseq, st, i, a[i].k);
                    bad += !same(what, a[i].st.conv, b[i].st.conv, a[i].st.conv.n);
                    for (u32 t = 0; t + 1 < a[i].k; t++) {
                        snprintf(what, sizeof what, "%s verify n=%u step %d seq %u: h snapshot %u", s.name, nseq, st, i, t);
                        bad += !same(what, a[i].h_snap[t], b[i].h_snap[t], a[i].h_snap[t].n);
                        snprintf(what, sizeof what, "%s verify n=%u step %d seq %u: conv snapshot %u", s.name, nseq, st, i, t);
                        bad += !same(what, a[i].conv_snap[t], b[i].conv_snap[t], a[i].conv_snap[t].n);
                    }
                }
            }
            for (auto& v : a) vseq_free(v);
            for (auto& v : b) vseq_free(v);
            for (GdnIo* io : {&ia, &ib}) {
                io->ba_in.free_(); io->qkvz.free_(); io->out.free_(); io->gates.free_(); io->conv_out.free_();
            }
        }
    }
    printf("  %s qwen4exp_gdn_verify_fused_rows %s: n = 1, 2, 3, 5, 8, 12 sequences x k = 2..4 and "
           "ragged 1..4 x %d steps, every normed row and every recurrence / conv / rollback-slot byte "
           "equal to the exact arm's per-token chain\n", bad ? "BAD" : "ok ", s.name, steps);
    return bad == 0;
}

// ══ timing ═══════════════════════════════════════════════════════════════
// Median us per layer of `body(layer)` over `layers` distinct layer sets,
// captured into one CUDA graph (serving replays decode as graphs).
template <typename F>
static double graph_us(F&& body, int layers, int reps = 7) {
    cudaGraph_t g;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < layers; l++) body(l);
    CK(cudaStreamEndCapture(g_s, &g));
    CK(cudaGraphInstantiate(&exec, g, 0));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> v;
    for (int r = 0; r < reps + 1; r++) {
        CK(cudaEventRecord(a, g_s));
        CK(cudaGraphLaunch(exec, g_s));
        CK(cudaEventRecord(b, g_s));
        CK(cudaStreamSynchronize(g_s));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (r) v.push_back(1000.0 * ms / layers);
    }
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(g));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static const int LAYERS = 36;

static void moe_small_time() {
    CUfunction t1 = mod("moe_topk").fn("moe_topk_softmax"), tr = mod("moe_topk").fn("moe_topk_softmax_rows");
    CUfunction b1 = mod("moe_expert_gemv").fn("moe_weighted_sum_blend");
    CUfunction br = mod("moe_expert_gemv").fn("moe_weighted_sum_blend_rows");
    const u32 R = 32;
    Buf<bf> logits, eo, x, gw, out;
    Buf<u32> idx;
    Buf<float> w;
    logits.alloc((size_t)R * NE); logits.put(rbf(logits.n, 2.0f));
    eo.alloc((size_t)R * TOPK * H); eo.put(rbf(eo.n, 1.0f));
    x.alloc((size_t)R * H); x.put(rbf(x.n, 1.0f));
    gw.alloc(H); gw.put(rbf(H, 0.05f));
    out.alloc((size_t)R * H);
    idx.alloc(R * TOPK);
    w.alloc(R * TOPK); w.put(rf(w.n, 0.3f));
    u32 ne = NE, k = TOPK, norm = 1, h = H, zero = 0;
    printf("\n  MoE glue, us per layer in a graph: R single-row launches -> one rows launch\n");
    printf("  %5s  %-24s  %-24s\n", "rows", "top-k", "weighted sum + blend (EP)");
    for (u32 rows : ROWS) {
        const double ta = graph_us([&](int) {
            for (u32 r = 0; r < rows; r++) {
                Args a; a.add(logits.p + (size_t)r * NE).add(idx.p + r * TOPK).add(w.p + r * TOPK)
                    .add(ne).add(k).add(norm);
                launch(t1, dim3(1), dim3(256), 0, a, g_s);
            }
        }, LAYERS);
        const double tb = graph_us([&](int) {
            Args a; a.add(logits.p).add(idx.p).add(w.p).add(ne).add(k).add(norm).add(ne);
            launch(tr, dim3(rows), dim3(256), 0, a, g_s);
        }, LAYERS);
        const double ba = graph_us([&](int) {
            for (u32 r = 0; r < rows; r++) {
                Args a; a.add(out.p + (size_t)r * H).add(eo.p + (size_t)r * TOPK * H).add(w.p + r * TOPK)
                    .add((bf*)nullptr).add(x.p + (size_t)r * H).add(gw.p).add(h).add(k).add(h);
                launch(b1, dim3((H + 255) / 256), dim3(256), 0, a, g_s);
            }
        }, LAYERS);
        const double bb = graph_us([&](int) {
            Args a; a.add(out.p).add(eo.p).add(w.p).add((bf*)nullptr).add(x.p).add(gw.p).add(h).add(k)
                .add(h).add(zero);
            launch(br, dim3((H + 255) / 256, rows), dim3(256), 0, a, g_s);
        }, LAYERS);
        printf("  %5u  %7.1f -> %6.1f (%4.1fx)  %7.1f -> %6.1f (%4.1fx)\n", rows, ta, tb, ta / tb, ba, bb,
               ba / bb);
    }
}

static void gdn_time(const GdnShape& s) {
    const GdnKernels k = gdn_kernels();
    const u32 RMAX = 8;
    std::vector<GdnWeights> ws(LAYERS);
    std::vector<std::vector<GdnSeq>> seqs(LAYERS, std::vector<GdnSeq>(RMAX));
    for (int l = 0; l < LAYERS; l++) {
        gdn_weights(s, ws[l]);
        for (auto& q : seqs[l]) gdn_seq(s, q, true);
    }
    GdnIo io;
    gdn_io(s, io, RMAX);
    io.qkvz.put(rbf(io.qkvz.n, 1.0f));
    io.ba_in.put(rbf(io.ba_in.n, 1.0f));
    printf("\n  GDN step %s, us per layer in a graph (%d layers, per-sequence states):\n", s.name, LAYERS);
    printf("  %5s  %10s  %10s  %10s  %s\n", "rows", "4-chain", "fused/seq", "rows", "chain->rows");
    for (u32 rows = 1; rows <= RMAX; rows++) {
        const double a = graph_us([&](int l) { for (u32 r = 0; r < rows; r++) gdn_chain(k, s, ws[l], seqs[l][r], io, r); }, LAYERS);
        const double b = graph_us([&](int l) { for (u32 r = 0; r < rows; r++) gdn_fused(k, s, ws[l], seqs[l][r], io, r); }, LAYERS);
        const double c = graph_us([&](int l) { gdn_rows_all(k, s, ws[l], seqs[l], io, rows); }, LAYERS);
        printf("  %5u  %10.1f  %10.1f  %10.1f  %.2fx  (%.0f GB/s of recurrence state)\n", rows, a, b, c,
               a / c, rows * 2.0 * s.nv * D * D * 4 / c / 1e3);
    }
}

static void verify_time(const GdnShape& s) {
    const GdnKernels k = gdn_kernels();
    CUfunction vk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_fused_rows");
    const u32 NMAX = 8;
    std::vector<GdnWeights> ws(LAYERS);
    std::vector<std::vector<VerifySeq>> seqs(LAYERS, std::vector<VerifySeq>(NMAX));
    for (int l = 0; l < LAYERS; l++) {
        gdn_weights(s, ws[l]);
        for (auto& v : seqs[l]) vseq_alloc(s, v, true);
    }
    GdnIo io;
    gdn_io(s, io, NMAX * KMAX);
    io.qkvz.put(rbf(io.qkvz.n, 1.0f));
    io.gates.put(gate_rows(io.gates.n));
    printf("\n  GDN exact verify %s, us per layer in a graph (%d layers): per-token chain -> one launch\n",
           s.name, LAYERS);
    printf("  %5s", "seqs");
    for (u32 kk = 2; kk <= KMAX; kk++) printf("  %22s", (std::string("k=") + std::to_string(kk)).c_str());
    printf("\n");
    for (u32 n : {1u, 2u, 4u, 8u}) {
        printf("  %5u", n);
        for (u32 kk = 2; kk <= KMAX; kk++) {
            for (auto& L : seqs) for (u32 i = 0; i < NMAX; i++) { L[i].k = kk; L[i].row0 = i * kk; }
            const double a = graph_us([&](int l) { for (u32 i = 0; i < n; i++) verify_chain(k, s, ws[l], seqs[l][i], io); }, LAYERS);
            const double b = graph_us([&](int l) {
                std::vector<VerifySeq> sub(seqs[l].begin(), seqs[l].begin() + n);  // shallow
                verify_rows(vk, s, ws[l], sub, io);
            }, LAYERS);
            printf("  %8.1f -> %6.1f %4.1fx", a, b, a / b);
        }
        printf("\n");
    }
}

// ══ mHC seam and the EP blend + post over T rows ═════════════════════════
// The decode-fuse tier's seam (hc_post_stage_vec) and MoE tail
// (moe_blend_hc_post) at T = 1..8 rows against the kernels the batched step
// runs: hc_post_vec + hc_pre_stage_vec, moe_batched_blend + hc_post_vec.
static const u32 HC = 4, HC_SPLIT = 8, POST_BLOCK = 64, BLEND_BLOCK = 256;
struct HcKernels { CUfunction post, stage, post_stage, blend, blend_post; };
static HcKernels hc_kernels() {
    return {mod("hyper_connection").fn("hc_post_vec"), mod("hyper_connection").fn("hc_pre_stage_vec"),
            mod("hyper_connection").fn("hc_post_stage_vec"), mod("moe_permute").fn("moe_batched_blend"),
            mod("qwen4exp_decode_fuse").fn("moe_blend_hc_post")};
}
static void hc_post(const HcKernels& k, bf* block, float* streams, float* inj, u32 T) {
    u32 h = H, hc = HC;
    Args a; a.add(block).add(streams).add(inj).add(streams).add(h).add(hc);
    launch(k.post, dim3(T, (H / 4 + POST_BLOCK - 1) / POST_BLOCK), dim3(POST_BLOCK), 0, a, g_s);
}
static void seam_old(const HcKernels& k, bf* block, float* streams, float* inj, bf* norm_w, float* normed, u32 T) {
    hc_post(k, block, streams, inj, T);
    u32 h = H, hc = HC; float eps = 1e-6f;
    Args a; a.add(streams).add(norm_w).add(normed).add(h).add(hc).add(eps);
    launch(k.stage, dim3(T, HC_SPLIT), dim3(1024), 0, a, g_s);
}
static void seam_new(const HcKernels& k, bf* block, float* streams, float* inj, bf* norm_w, float* normed, u32 T) {
    u32 h = H, hc = HC; float eps = 1e-6f;
    Args a; a.add(block).add(streams).add(inj).add(norm_w).add(normed).add(h).add(hc).add(eps);
    launch(k.post_stage, dim3(T, HC_SPLIT), dim3(1024), 0, a, g_s);
}
static void tail_old(const HcKernels& k, bf* out, bf* shared, bf* x, bf* gw, float* streams, float* inj, u32 T) {
    u32 h = H;
    Args a; a.add(out).add(shared).add(x).add(gw).add(h).add(T);
    launch(k.blend, dim3(T), dim3(BLEND_BLOCK), 0, a, g_s);
    hc_post(k, out, streams, inj, T);
}
static void tail_new(const HcKernels& k, bf* out, bf* shared, bf* x, bf* gw, float* streams, float* inj, u32 T) {
    u32 h = H, hc = HC;
    Args a; a.add(out).add(shared).add(x).add(gw).add(streams).add(inj).add(h).add(hc);
    launch(k.blend_post, dim3(T, (H + 4 * BLEND_BLOCK - 1) / (4 * BLEND_BLOCK)), dim3(BLEND_BLOCK), 0, a, g_s);
}

static void hc_rows_check() {
    const HcKernels k = hc_kernels();
    int bad = 0;
    for (u32 T = 1; T <= 8; T++) {
        for (int rep = 0; rep < 4; rep++) {
            auto sv = rf((size_t)T * HC * H, 1.0f), iv = rf((size_t)T * HC, 1.0f);
            auto bv = rbf((size_t)T * H, 1.0f), nw = rbf((size_t)HC * H, 1.0f);
            auto ov = rbf((size_t)T * H, 1.0f), shv = rbf((size_t)T * H, 1.0f), gv = rbf(H, 0.05f);
            Buf<float> s1, s2, n1, n2, inj; Buf<bf> blk, w, o1, o2, sh, gw;
            for (auto* b : {&s1, &s2}) { b->alloc(sv.size()); b->put(sv); }
            for (auto* b : {&n1, &n2}) { b->alloc((size_t)T * HC * H); b->fill(0x7F); }
            inj.alloc(iv.size()); inj.put(iv);
            blk.alloc(bv.size()); blk.put(bv);
            w.alloc(nw.size()); w.put(nw);
            for (auto* b : {&o1, &o2}) { b->alloc(ov.size()); b->put(ov); }
            sh.alloc(shv.size()); sh.put(shv);
            gw.alloc(gv.size()); gw.put(gv);
            char what[96];
            seam_old(k, blk.p, s1.p, inj.p, w.p, n1.p, T);
            seam_new(k, blk.p, s2.p, inj.p, w.p, n2.p, T);
            CK(cudaDeviceSynchronize());
            snprintf(what, sizeof what, "hc_post_stage_vec T=%u streams", T);
            bad += !same(what, s1, s2, s1.n);
            snprintf(what, sizeof what, "hc_post_stage_vec T=%u staged rows", T);
            bad += !same(what, n1, n2, n1.n);
            for (bf* g : {gw.p, (bf*)nullptr}) {
                s1.put(sv); s2.put(sv); o1.put(ov); o2.put(ov);
                tail_old(k, o1.p, sh.p, blk.p, g, s1.p, inj.p, T);
                tail_new(k, o2.p, sh.p, blk.p, g, s2.p, inj.p, T);
                CK(cudaDeviceSynchronize());
                snprintf(what, sizeof what, "moe_blend_hc_post T=%u gate=%d output", T, g != nullptr);
                bad += !same(what, o1, o2, o1.n);
                snprintf(what, sizeof what, "moe_blend_hc_post T=%u gate=%d streams", T, g != nullptr);
                bad += !same(what, s1, s2, s1.n);
            }
            for (auto* b : {&s1, &s2, &n1, &n2, &inj}) b->free_();
            for (auto* b : {&blk, &w, &o1, &o2, &sh, &gw}) b->free_();
        }
    }
    printf("  %s hc_post_stage_vec and moe_blend_hc_post at T = 1..8 rows: highway, staged rows and "
           "MoE output bytes equal to hc_post_vec + hc_pre_stage_vec / moe_batched_blend + hc_post_vec\n",
           bad ? "BAD" : "ok ");
}

static void hc_rows_time() {
    const HcKernels k = hc_kernels();
    std::vector<Buf<float>> streams(LAYERS), normed(LAYERS), inj(LAYERS);
    Buf<bf> blk, w, out, sh, gw;
    for (int l = 0; l < LAYERS; l++) {
        streams[l].alloc((size_t)8 * HC * H); streams[l].put(rf(streams[l].n, 1.0f));
        normed[l].alloc((size_t)8 * HC * H);
        inj[l].alloc(8 * HC); inj[l].put(rf(inj[l].n, 1.0f));
    }
    blk.alloc((size_t)8 * H); blk.put(rbf(blk.n, 1.0f));
    w.alloc((size_t)HC * H); w.put(rbf(w.n, 1.0f));
    out.alloc((size_t)8 * H); out.put(rbf(out.n, 1.0f));
    sh.alloc((size_t)8 * H); sh.put(rbf(sh.n, 1.0f));
    gw.alloc(H); gw.put(rbf(H, 0.05f));
    printf("\n  mHC seam / EP MoE tail, us per layer in a graph: two launches -> one\n");
    printf("  %5s  %-22s  %-22s\n", "rows", "seam (post + stage)", "tail (blend + post)");
    for (u32 T : {1u, 2u, 4u, 8u}) {
        const double a = graph_us([&](int l) { seam_old(k, blk.p, streams[l].p, inj[l].p, w.p, normed[l].p, T); }, LAYERS);
        const double b = graph_us([&](int l) { seam_new(k, blk.p, streams[l].p, inj[l].p, w.p, normed[l].p, T); }, LAYERS);
        const double c = graph_us([&](int l) { tail_old(k, out.p, sh.p, blk.p, gw.p, streams[l].p, inj[l].p, T); }, LAYERS);
        const double d = graph_us([&](int l) { tail_new(k, out.p, sh.p, blk.p, gw.p, streams[l].p, inj[l].p, T); }, LAYERS);
        printf("  %5u  %7.1f -> %6.1f (%3.1fx)   %7.1f -> %6.1f (%3.1fx)\n", T, a, b, a / b, c, d, c / d);
    }
}

// The recurrence alone, R sequences: R launches of gated_delta_rule_decode_f32
// (one per sequence, as the batched decode's loop) against one launch of the
// strided multi-sequence twin gated_delta_rule_decode_f32_strided (the
// non-exact-lane batched recurrent arm; contiguous state slots), to answer
// why that twin measured slower than the loop. Answer (ptxas -v, sm_121a):
// both spill H_reg (255 registers, ~1.4 KB of spill a thread, ~176 KB a
// block). A launch per sequence keeps 24 blocks resident (~4 MB of spill,
// cache-resident); the strided grid keeps up to two blocks an SM resident
// (~17 MB of spill beside the streamed state), which plausibly spills to
// DRAM: 172 -> 293 us at R = 8, TP2. The fused step spills ~0.3 KB a thread,
// and its rows grid runs as fast as, or faster than, a launch per sequence
// (150 -> 145 us).
static void rec_time(const GdnShape& s) {
    CUfunction rec = mod("gated_delta_rule").fn("gated_delta_rule_decode_f32");
    CUfunction strided = mod("gated_delta_rule").fn("gated_delta_rule_decode_f32_strided");
    const u32 R = 8, cd = conv_dim(s), key_dim = s.nk * D, vdim = s.nv * D;
    std::vector<Buf<float>> h(LAYERS);
    for (auto& b : h) { b.alloc((size_t)R * s.nv * D * D); b.put(rf(b.n, 0.05f)); }
    Buf<float> conv, gates, out;
    conv.alloc((size_t)R * cd); conv.put(rf(conv.n, 0.1f));
    gates.alloc((size_t)R * 2 * s.nv); gates.put(gate_rows(gates.n));
    out.alloc((size_t)R * vdim);
    printf("\n  recurrence only %s, us per layer in a graph: R x decode_f32 vs 1 x decode_f32_strided\n", s.name);
    for (u32 rows : {1u, 2u, 4u, 8u}) {
        const double a = graph_us([&](int l) {
            for (u32 r = 0; r < rows; r++) {
                u32 one = 1, d = D;
                float* c = conv.p + (size_t)r * cd;
                float* g = gates.p + (size_t)r * 2 * s.nv;
                Args x;
                x.add(h[l].p + (size_t)r * s.nv * D * D).add(c).add(c + key_dim).add(c + 2 * key_dim).add(g)
                    .add(g + s.nv).add(out.p + (size_t)r * vdim).add(one).add(s.nk).add(s.nv).add(d).add(d);
                launch(rec, dim3(s.nv), dim3(128), 0, x, g_s);
            }
        }, LAYERS);
        const double b = graph_us([&](int l) {
            u32 d = D, qks = cd, vs = cd, gbs = 2 * s.nv, os = vdim;
            Args x;
            x.add(h[l].p).add(conv.p).add(conv.p + key_dim).add(conv.p + 2 * key_dim).add(gates.p)
                .add(gates.p + s.nv).add(out.p).add(rows).add(s.nk).add(s.nv).add(d).add(d).add(qks).add(vs)
                .add(gbs).add(os);
            launch(strided, dim3(s.nv, rows), dim3(128), 0, x, g_s);
        }, LAYERS);
        printf("  %5u  %7.1f -> %7.1f\n", rows, a, b);
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    const int steps = argc > 3 ? atoi(argv[3]) : 4;
    init_driver();
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    const GdnShape tp2 = {8, 24, "TP2 8k/24v"}, tp1 = {16, 48, "TP1 16k/48v"};
    if (mode == "check") {
        printf("per-row parity (batch-small):\n");
        topk_check();
        blend_check();
        gdn_check(tp2, steps);
        gdn_check(tp1, steps);
        verify_check(tp2, steps);
        verify_check(tp1, steps);
        hc_rows_check();
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    moe_small_time();
    gdn_time(tp2);
    gdn_time(tp1);
    verify_time(tp2);
    hc_rows_time();
    rec_time(tp2);
    return 0;
}
