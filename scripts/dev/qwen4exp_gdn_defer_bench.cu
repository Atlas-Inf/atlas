// SPDX-License-Identifier: AGPL-3.0-only
// Parity and cost of the exact deferred GDN commit (ATLAS_QWEN4EXP_EXACT_DEFER,
// kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_decode_fuse.cu):
//
//   qwen4exp_gdn_verify_defer_rows   the exact verify with no state stores,
//                                    the step's conv inputs and gates staged
//   qwen4exp_gdn_commit_layers       the accepted prefix replayed from H0,
//                                    one launch over a sequence's GDN layers
//
// against qwen4exp_gdn_verify_fused_rows, which stores the state after every
// token (rollback slots t < k - 1) and the final state, so that the commit is
// a copy of slot n - 1 (or nothing when n == k).
//
// check: for 1..12 sequences, k = 2..8 and ragged 1..8, three GDN layers each
//   with its own weights, inputs and states, over `steps` steps: the normed
//   rows of the two verifies are equal; then, for EVERY sequence and EVERY
//   accepted length n = 1..k (a rejection at each position, and the full
//   accept), one commit launch over the three layers from H0 lands the bytes
//   of the storing kernel's slot n - 1 / final state, recurrence and conv
//   windows. The step then commits a random n in place and carries on.
// time: GPU us in a graph, TP2 shapes (8 key / 24 value heads, 36 GDN
//   layers): storing vs deferred verify per layer, the commit of a sequence
//   over 36 layers, and the pitched-copy commit it replaces.
//
// Build/run: scripts/dev/qwen4exp_batch_exact_bench.sh defer-check|defer-time
// (repo root, GB10, inside atlas-release-builder:1.93.1).
#define QB_SMALL_NO_MAIN
#include "qwen4exp_batch_small_bench.cu"

struct DeferStage { Buf<bf> qkv; Buf<float> gb; };
static void stage_alloc(const GdnShape& s, DeferStage& st) {
    st.qkv.alloc((size_t)KMAX * conv_dim(s)); st.qkv.fill(0x7F);
    st.gb.alloc((size_t)KMAX * 2 * s.nv); st.gb.fill(0x7F);
}

struct DeferRowsArg {
    struct { float* h; float* conv; bf* sq; float* sg; u32 row0, k; const u32* fuse_n; } seq[ROWS_MAX];
};
static_assert(sizeof(DeferRowsArg) == 48 * ROWS_MAX, "QdfDeferRows layout");

// `fuse`: each sequence's pending-commit word (ATLAS_QWEN4EXP_GDN_COMMIT_FUSE),
// or none (the unfused kernel).
static void verify_defer(CUfunction fn, const GdnShape& s, const GdnWeights& w,
                         std::vector<VerifySeq>& seqs, std::vector<DeferStage>& stg, GdnIo& io,
                         const std::vector<Buf<u32>>* fuse = nullptr) {
    const float l2 = 1e-6f, eps = 1e-6f;
    const u32 d = D, stride = qkvz_dim(s);
    for (size_t first = 0; first < seqs.size(); first += ROWS_MAX) {
        const u32 count = (u32)std::min<size_t>(ROWS_MAX, seqs.size() - first);
        DeferRowsArg t = {};
        for (u32 i = 0; i < count; i++) {
            VerifySeq& v = seqs[first + i];
            const u32* fw = fuse ? (*fuse)[first + i].p : nullptr;
            t.seq[i] = {v.st.h.p, v.st.conv.p, stg[first + i].qkv.p, stg[first + i].gb.p, v.row0, v.k, fw};
        }
        Args a;
        a.add(t).add(io.qkvz.p).add(w.conv_w.p).add(io.gates.p).add(w.norm_w.p).add(io.out.p)
            .add(s.nk).add(s.nv).add(d).add(stride).add(l2).add(eps);
        launch(fn, dim3(s.nv, count), dim3(128), 0, a, g_s);
    }
}

static const u32 COMMIT_LAYERS = 48;
struct CommitEntry { float* h; float* conv; const bf* sq; const float* sg; const bf* cw; };
struct CommitArg { CommitEntry layer[COMMIT_LAYERS]; };
static_assert(sizeof(CommitArg) == 40 * COMMIT_LAYERS, "QdfCommitLayers layout");

static void commit(CUfunction fn, const GdnShape& s, const std::vector<CommitEntry>& e, u32 n) {
    const u32 d = D;
    const float l2 = 1e-6f;
    static const size_t per = getenv("QB_COMMIT_PER") ? atoi(getenv("QB_COMMIT_PER")) : COMMIT_LAYERS;
    for (size_t first = 0; first < e.size(); first += per) {
        const u32 count = (u32)std::min<size_t>(per, e.size() - first);
        CommitArg t = {};
        for (u32 i = 0; i < count; i++) t.layer[i] = e[first + i];
        Args a;
        a.add(t).add(n).add(s.nk).add(s.nv).add(d).add(l2);
        launch(fn, dim3(s.nv, count), dim3(128), 0, a, g_s);
    }
}

// On the kernels' stream: a D2D cudaMemcpy does not order against the
// non-blocking `g_s`.
static void copy_state(GdnSeq& dst, const Buf<float>& h, const Buf<float>& conv) {
    CK(cudaMemcpyAsync(dst.h.p, h.p, h.n * 4, cudaMemcpyDeviceToDevice, g_s));
    CK(cudaMemcpyAsync(dst.conv.p, conv.p, conv.n * 4, cudaMemcpyDeviceToDevice, g_s));
}

// The storing kernel's state after `n` of `v.k` tokens.
static const Buf<float>& ref_h(const VerifySeq& v, u32 n) { return n < v.k ? v.h_snap[n - 1] : v.st.h; }
static const Buf<float>& ref_conv(const VerifySeq& v, u32 n) { return n < v.k ? v.conv_snap[n - 1] : v.st.conv; }

static bool defer_check(const GdnShape& s, int steps) {
    const u32 NL = 3;  // GDN layers of every sequence, one commit launch over them
    CUfunction vk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_fused_rows");
    CUfunction dk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_defer_rows");
    CUfunction ck = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_commit_layers");
    std::vector<GdnWeights> w(NL);
    for (auto& x : w) gdn_weights(s, x);
    int bad = 0, commits = 0, fuse_commits = 0;
    for (u32 nseq : {1u, 2u, 3u, 5u, 8u, 12u}) {
        for (int ragged = 0; ragged < 2; ragged++) {
            // a: storing kernel (reference); b: deferred; c: commit scratch;
            // f: deferred with the commit fused into the next verify (`fw`:
            // one pending-commit word a sequence, shared by its layers).
            std::vector<std::vector<VerifySeq>> a(NL, std::vector<VerifySeq>(nseq)), b = a, f = a;
            std::vector<std::vector<DeferStage>> stg(NL, std::vector<DeferStage>(nseq)), stgf = stg;
            std::vector<Buf<u32>> fw(nseq);
            for (auto& x : fw) x.alloc(1);
            std::vector<std::vector<GdnSeq>> c(NL, std::vector<GdnSeq>(nseq));
            u32 rows = 0;
            std::vector<u32> ks(nseq), row0(nseq);
            for (u32 i = 0; i < nseq; i++) {
                ks[i] = ragged ? 1 + (g_rng() % KMAX) : 2 + (i % (KMAX - 1));
                row0[i] = rows;
                rows += ks[i];
            }
            std::vector<GdnIo> ia(NL), ib(NL), iff(NL);
            for (u32 l = 0; l < NL; l++) {
                gdn_io(s, ia[l], rows); gdn_io(s, ib[l], rows); gdn_io(s, iff[l], rows);
                for (u32 i = 0; i < nseq; i++) {
                    vseq_alloc(s, a[l][i], true);
                    gdn_seq(s, b[l][i].st, false);  // no rollback slots: the arm stores none
                    a[l][i].k = ks[i]; a[l][i].row0 = row0[i];
                    vseq_copy(b[l][i], a[l][i]);
                    gdn_seq(s, f[l][i].st, false);
                    vseq_copy(f[l][i], a[l][i]);
                    stage_alloc(s, stgf[l][i]);
                    stage_alloc(s, stg[l][i]);
                    gdn_seq(s, c[l][i], false);
                }
            }
            CK(cudaDeviceSynchronize());  // the setup copies ran on the legacy stream
            for (int st = 0; st < steps; st++) {
                for (u32 l = 0; l < NL; l++) {
                    auto q = rbf(ia[l].qkvz.n, st == 0 ? 3.0f : 1.0f);
                    auto g = gate_rows(ia[l].gates.n);
                    for (GdnIo* io : {&ia[l], &ib[l], &iff[l]}) { io->qkvz.put(q); io->gates.put(g); io->out.fill(0x7F); }
                    verify_rows(vk, s, w[l], a[l], ia[l]);
                    verify_defer(dk, s, w[l], b[l], stg[l], ib[l]);
                    verify_defer(dk, s, w[l], f[l], stgf[l], iff[l], &fw);
                }
                CK(cudaDeviceSynchronize());
                char what[192];
                for (u32 l = 0; l < NL; l++) {
                    snprintf(what, sizeof what, "%s defer n=%u %s step %d layer %u: normed rows", s.name, nseq,
                             ragged ? "ragged" : "k=2..8", st, l);
                    bad += !same(what, ia[l].out, ib[l].out, ia[l].out.n);
                    snprintf(what, sizeof what, "%s fuse n=%u %s step %d layer %u: normed rows", s.name, nseq,
                             ragged ? "ragged" : "k=2..8", st, l);
                    bad += !same(what, ia[l].out, iff[l].out, ia[l].out.n);
                    // The fused verify stored the previous step's commit: the
                    // state the deferred arm committed in place last step.
                    for (u32 i = 0; i < nseq; i++) {
                        snprintf(what, sizeof what, "%s fuse step %d seq %u layer %u: fused commit", s.name,
                                 st, i, l);
                        bad += !same(what, f[l][i].st.h, b[l][i].st.h, b[l][i].st.h.n);
                        bad += !same(what, f[l][i].st.conv, b[l][i].st.conv, b[l][i].st.conv.n);
                    }
                }
                // Every accepted length of every sequence, from H0.
                for (u32 i = 0; i < nseq; i++) {
                    for (u32 n = 1; n <= ks[i]; n++) {
                        std::vector<CommitEntry> e;
                        for (u32 l = 0; l < NL; l++) {
                            copy_state(c[l][i], b[l][i].st.h, b[l][i].st.conv);
                            e.push_back({c[l][i].h.p, c[l][i].conv.p, stg[l][i].qkv.p, stg[l][i].gb.p, w[l].conv_w.p});
                        }
                        commit(ck, s, e, n);
                        CK(cudaDeviceSynchronize());
                        commits++;
                        for (u32 l = 0; l < NL; l++) {
                            snprintf(what, sizeof what, "%s defer step %d seq %u layer %u k=%u n=%u: recurrence",
                                     s.name, st, i, l, ks[i], n);
                            bad += !same(what, c[l][i].h, ref_h(a[l][i], n), c[l][i].h.n);
                            snprintf(what, sizeof what, "%s defer step %d seq %u layer %u k=%u n=%u: conv",
                                     s.name, st, i, l, ks[i], n);
                            bad += !same(what, c[l][i].conv, ref_conv(a[l][i], n), c[l][i].conv.n);
                        }
                    }
                }
                // Commit a random prefix in place (the production form) and
                // put the reference arm on the same state for the next step.
                for (u32 i = 0; i < nseq; i++) {
                    const u32 n = 1 + g_rng() % ks[i];
                    std::vector<CommitEntry> e;
                    for (u32 l = 0; l < NL; l++)
                        e.push_back({b[l][i].st.h.p, b[l][i].st.conv.p, stg[l][i].qkv.p, stg[l][i].gb.p, w[l].conv_w.p});
                    commit(ck, s, e, n);
                    fw[i].put({n});  // the fused arm commits it in its next verify
                    fuse_commits++;
                    CK(cudaDeviceSynchronize());
                    for (u32 l = 0; l < NL; l++) {
                        snprintf(what, sizeof what, "%s defer step %d seq %u layer %u n=%u: in-place commit",
                                 s.name, st, i, l, n);
                        bad += !same(what, b[l][i].st.h, ref_h(a[l][i], n), b[l][i].st.h.n);
                        bad += !same(what, b[l][i].st.conv, ref_conv(a[l][i], n), b[l][i].st.conv.n);
                        if (n < ks[i]) copy_state(a[l][i].st, a[l][i].h_snap[n - 1], a[l][i].conv_snap[n - 1]);
                    }
                }
            }
            // The last step's commit is still pending on the fused arm: the
            // standalone commit (a flush) lands it.
            for (u32 i = 0; i < nseq; i++) {
                const u32 n = fw[i].get()[0];
                std::vector<CommitEntry> e;
                for (u32 l = 0; l < NL; l++)
                    e.push_back({f[l][i].st.h.p, f[l][i].st.conv.p, stgf[l][i].qkv.p, stgf[l][i].gb.p, w[l].conv_w.p});
                commit(ck, s, e, n);
                CK(cudaDeviceSynchronize());
                char what[192];
                for (u32 l = 0; l < NL; l++) {
                    snprintf(what, sizeof what, "%s fuse seq %u layer %u n=%u: flush", s.name, i, l, n);
                    bad += !same(what, f[l][i].st.h, b[l][i].st.h, b[l][i].st.h.n);
                    bad += !same(what, f[l][i].st.conv, b[l][i].st.conv, b[l][i].st.conv.n);
                }
            }
            for (auto& x : fw) x.free_();
            for (u32 l = 0; l < NL; l++) {
                for (auto& v : a[l]) vseq_free(v);
                for (auto& v : b[l]) vseq_free(v);
                for (auto& v : f[l]) { v.st.h.free_(); v.st.conv.free_(); }
                for (auto& x : stgf[l]) { x.qkv.free_(); x.gb.free_(); }
                for (auto& x : stg[l]) { x.qkv.free_(); x.gb.free_(); }
                for (auto& x : c[l]) { x.h.free_(); x.conv.free_(); }
                for (GdnIo* io : {&ia[l], &ib[l], &iff[l]}) {
                    io->ba_in.free_(); io->qkvz.free_(); io->out.free_(); io->gates.free_(); io->conv_out.free_();
                }
            }
        }
    }
    printf("  %s qwen4exp_gdn_verify_defer_rows + qwen4exp_gdn_commit_layers %s: n = 1, 2, 3, 5, 8, 12 "
           "sequences x k = 2..%u and ragged 1..%u x %d steps x %u layers: normed rows equal, and %d "
           "commits (every accepted length 1..k of every sequence, from H0, plus one in place a step) "
           "land the storing kernel's recurrence and conv bytes; fused into the next verify (%d "
           "commits + a flush each): normed rows and committed state equal\n",
           bad ? "BAD" : "ok ", s.name, KMAX, KMAX, steps, NL, commits, fuse_commits);
    for (auto& x : w) { x.conv_w.free_(); x.ba_w.free_(); x.norm_w.free_(); x.a_log.free_(); x.dt_bias.free_(); }
    return bad == 0;
}

// GPU us in a graph. Footprint stays small (03 is shared): 8 layers of
// 8 sequences for the per-layer verify numbers, which DRAM-stream (each
// layer's states are 12.6 MB, the set well past L2).
static void defer_time(const GdnShape& s) {
    const int NLT = 8;
    const u32 NS = 8;
    CUfunction vk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_fused_rows");
    CUfunction dk = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_verify_defer_rows");
    CUfunction ck = mod("qwen4exp_decode_fuse").fn("qwen4exp_gdn_commit_layers");
    std::vector<GdnWeights> ws(NLT);
    std::vector<std::vector<VerifySeq>> seqs(NLT, std::vector<VerifySeq>(NS));
    std::vector<std::vector<DeferStage>> stg(NLT, std::vector<DeferStage>(NS));
    for (int l = 0; l < NLT; l++) {
        gdn_weights(s, ws[l]);
        for (auto& v : seqs[l]) vseq_alloc(s, v, true);
        for (auto& x : stg[l]) stage_alloc(s, x);
    }
    GdnIo io;
    gdn_io(s, io, NS * KMAX);
    io.qkvz.put(rbf(io.qkvz.n, 1.0f));
    io.gates.put(gate_rows(io.gates.n));
    printf("\n  GDN verify %s, %u sequences, us per layer in a graph (%d layers): storing -> deferred\n",
           s.name, NS, NLT);
    for (u32 kk : {2u, 4u, 8u}) {
        for (auto& L : seqs) for (u32 i = 0; i < NS; i++) { L[i].k = kk; L[i].row0 = i * kk; }
        const double a = graph_us([&](int l) { verify_rows(vk, s, ws[l], seqs[l], io); }, NLT);
        const double b = graph_us([&](int l) { verify_defer(dk, s, ws[l], seqs[l], stg[l], io); }, NLT);
        printf("  k=%u  %7.1f -> %7.1f us/layer  (x36 layers: %.2f -> %.2f ms/step)\n", kk, a, b,
               36 * a / 1e3, 36 * b / 1e3);
    }
    // ATLAS_QWEN4EXP_GDN_COMMIT_FUSE: the deferred verify that first lands the
    // previous step's commit of n tokens (and stores the state once).
    std::vector<Buf<u32>> fw(NS);
    for (auto& x : fw) x.alloc(1);
    for (auto& L : seqs) for (u32 i = 0; i < NS; i++) { L[i].k = 4; L[i].row0 = i * 4; }
    printf("  fused commit + verify, k=4 (us/layer, x36 ms/step):");
    for (u32 n : {0u, 1u, 3u, 4u}) {
        for (auto& x : fw) x.put({n});
        const double c = graph_us([&](int l) { verify_defer(dk, s, ws[l], seqs[l], stg[l], io, &fw); }, NLT);
        printf("  n=%u %.1f (%.2f)", n, c, 36 * c / 1e3);
    }
    printf("\n");
    for (auto& x : fw) x.free_();
    // Commit of one sequence over 36 GDN layers: the replay (one launch) vs
    // the pitched copies of the storing path (H then conv, 36 rows each).
    const u32 L36 = 36;
    std::vector<GdnSeq> st(L36);
    for (u32 l = 0; l < L36; l++) gdn_seq(s, st[l], true);
    std::vector<CommitEntry> e;
    for (u32 l = 0; l < L36; l++)
        e.push_back({st[l].h.p, st[l].conv.p, stg[l % NLT][0].qkv.p, stg[l % NLT][0].gb.p, ws[l % NLT].conv_w.p});
    printf("  commit of one sequence over %u layers (us): ", L36);
    for (u32 n : {1u, 3u, 4u, 7u}) {
        const double c = graph_us([&](int) { commit(ck, s, e, n); }, 1);
        printf(" replay n=%u %.0f", n, c);
    }
    // The storing path's commit as it runs: one pitched copy per family over
    // the 36 per-layer regions (contiguous here, pitch = blob).
    Buf<float> hs, hd, cs, cd;
    const size_t hb = (size_t)s.nv * D * D, cb = (size_t)conv_dim(s) * DCONV;
    hs.alloc(hb * L36); hd.alloc(hb * L36); cs.alloc(cb * L36); cd.alloc(cb * L36);
    const double cp = graph_us([&](int) {
        CK(cudaMemcpy2DAsync(hd.p, hb * 4, hs.p, hb * 4, hb * 4, L36, cudaMemcpyDeviceToDevice, g_s));
        CK(cudaMemcpy2DAsync(cd.p, cb * 4, cs.p, cb * 4, cb * 4, L36, cudaMemcpyDeviceToDevice, g_s));
    }, 1);
    printf("  | pitched copies %.0f\n", cp);
    for (auto* b : {&hs, &hd, &cs, &cd}) b->free_();
    for (auto& q : st) { q.h.free_(); q.conv.free_(); }
    for (int l = 0; l < NLT; l++) {
        for (auto& v : seqs[l]) vseq_free(v);
        for (auto& x : stg[l]) { x.qkv.free_(); x.gb.free_(); }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "check";
    const int steps = argc > 3 ? atoi(argv[3]) : 3;
    init_driver();
    CK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
    const GdnShape tp2 = {8, 24, "TP2 8k/24v"}, tp1 = {16, 48, "TP1 16k/48v"};
    if (mode == "check") {
        printf("exact deferred GDN commit parity:\n");
        defer_check(tp2, steps);
        defer_check(tp1, steps);
        printf("%s\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }
    defer_time(tp2);
    return 0;
}
