// SPDX-License-Identifier: AGPL-3.0-only
// Bitwise check and timing of the GDN prefill state spine at the qwen4_exp
// TP2 rank shape (24 v-heads, 8 k-heads, k = v = 128, 64-token chunks):
// `gated_delta_rule_chunk_delta_h_pipe` (the shipped spine, one CTA a head)
// against `gated_delta_rule_chunk_delta_h_pipe_dv64` (two column blocks a
// head). S_out, uc_out and the final state must match byte for byte; so must
// the `_cap` twins' (and the state they capture at chunk c must equal the
// final state of a c-chunk run).
//
// Build (repo root, GB10):
//   D=$(mktemp -d)
//   nvcc --ptx -arch=sm_121f -O3 --fmad=false -o $D/gdn_fla.ptx kernels/gb10/common/gated_delta_rule_fla.cu
//   (optionally the pre-change file as $D/old_fla.ptx, to pin the default)
//   nvcc -O3 -std=c++17 -o $D/bench scripts/dev/qwen4exp_gdn_spine_bench.cu -lcuda
//   $D/bench $D [tokens=16000] [v_heads=24] [k_heads=8]
#include "qwen4exp_ptx_harness.h"
#include <random>

static const unsigned KD = 128, VD = 128, C = 64;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [tokens] [nv] [nk]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 16000;
    const unsigned NV = argc > 3 ? atoi(argv[3]) : 24;
    const unsigned NK = argc > 4 ? atoi(argv[4]) : 8;
    const unsigned NC = (T + C - 1) / C;
    const unsigned QK = 2 * NK * KD + NV * VD;   // packed q | k | v row
    init_driver();
    PtxModule m;
    m.load(dir + "/gdn_fla.ptx");
    const unsigned smem_pipe = 2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4;
    const unsigned smem_dv = 2 * C * KD * 2 * 2 + 2 * C * 64 * 2 + 2 * C * 4 + 2 * (C + 1) * 4;
    CUfunction k_pipe = m.fn("gated_delta_rule_chunk_delta_h_pipe", smem_pipe);
    CUfunction k_dv = m.fn("gated_delta_rule_chunk_delta_h_pipe_dv64", smem_dv);

    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t blocks = (size_t)NC * NV;
    std::vector<unsigned short> w(blocks * C * KD), u(blocks * C * VD), qkv((size_t)T * QK);
    for (auto& x : w) x = f2bf(0.05f * nd(rng));
    for (auto& x : u) x = f2bf(0.5f * nd(rng));
    for (auto& x : qkv) x = f2bf(0.1f * nd(rng));
    std::vector<float> gc(blocks * C), h0((size_t)NV * KD * VD);
    for (size_t b = 0; b < blocks; ++b) {
        float acc = 0.0f;
        for (unsigned i = 0; i < C; ++i) {
            acc -= 0.002f + 0.01f * fabsf(nd(rng));
            gc[b * C + i] = acc;
        }
    }
    for (auto& x : h0) x = 0.01f * nd(rng);

    Buf<unsigned short> d_w, d_u, d_qkv, d_s1, d_s2, d_uc1, d_uc2;
    Buf<float> d_gc, d_h1, d_h2, d_gate;
    d_w.alloc(w.size()); d_w.put(w);
    d_u.alloc(u.size()); d_u.put(u);
    d_qkv.alloc(qkv.size()); d_qkv.put(qkv);
    d_gc.alloc(gc.size()); d_gc.put(gc);
    d_gate.alloc((size_t)T * 2 * NV);
    d_h1.alloc(h0.size()); d_h2.alloc(h0.size());
    d_s1.alloc(blocks * KD * VD); d_s2.alloc(blocks * KD * VD);
    d_uc1.alloc(blocks * C * VD); d_uc2.alloc(blocks * C * VD);
    const unsigned short* key = d_qkv.p + NK * KD;   // the k region of each packed row

    auto run = [&](CUfunction k, dim3 grid, unsigned threads, unsigned smem, float* h,
                   unsigned short* s_out, unsigned short* uc_out) {
        Args a;
        a.add(h).add(d_w.p).add(d_u.p).add(key).add(d_gate.p).add(d_gc.p).add(s_out).add(uc_out)
         .add(1u).add(T).add(NC).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV).add(0u)
         .add((int*)nullptr).add((int*)nullptr).add(0u);
        launch(k, grid, dim3(threads), smem, a);
    };
    auto pipe = [&] { run(k_pipe, dim3(NV, 1), 256, smem_pipe, d_h1.p, d_s1.p, d_uc1.p); };
    auto dv = [&] { run(k_dv, dim3(NV, 2), 128, smem_dv, d_h2.p, d_s2.p, d_uc2.p); };

    d_h1.put(h0); d_h2.put(h0);
    // Same fill on both sides: rows past a partial chunk's end are written by neither.
    d_s1.fill(0x11); d_s2.fill(0x11); d_uc1.fill(0x33); d_uc2.fill(0x33);
    pipe();
    dv();
    CK(cudaDeviceSynchronize());
    size_t ds = diff_bytes(d_s1.get(), d_s2.get());
    size_t du = diff_bytes(d_uc1.get(), d_uc2.get());
    size_t dh = diff_bytes(d_h1.get(), d_h2.get());
    printf("bitwise pipe vs pipe_dv64 (T=%u, %u v-heads, %u k-heads, %u chunks): S_out %zu, uc_out %zu, state %zu differing bytes\n",
           T, NV, NK, NC, ds, du, dh);
    const bool ok = ds == 0 && du == 0 && dh == 0;
    float t1 = time_ms([&] { d_h1.put(h0); pipe(); }, 3, 3);
    float t2 = time_ms([&] { d_h2.put(h0); dv(); }, 3, 3);
    printf("spine: pipe %.3f ms -> pipe_dv64 %.3f ms (%.2fx)  [each includes a %zu KB state upload]\n",
           t1, t2, t1 / t2, h0.size() * 4 / 1024);
    // ── the `_cap` twins (qwen4_exp mid-chunk checkpoint) ──
    // Same S_out / uc_out / final state as their plain kernels, and the state
    // they capture at chunk `cc` equals the final state of the same spine
    // run over only the first cc chunks.
    bool ok_cap = true;
    {
        CUfunction k_pipe_cap = m.fn("gated_delta_rule_chunk_delta_h_pipe_cap", smem_pipe);
        CUfunction k_dv_cap = m.fn("gated_delta_rule_chunk_delta_h_pipe_dv64_cap", smem_dv);
        Buf<unsigned short> d_s3, d_uc3;
        Buf<float> d_h3, d_cap, d_hpre;
        d_s3.alloc(blocks * KD * VD); d_uc3.alloc(blocks * C * VD);
        d_h3.alloc(h0.size()); d_cap.alloc(h0.size()); d_hpre.alloc(h0.size());
        for (unsigned cc : {1u, NC / 2, NC - 1}) {
            for (int dvk = 0; dvk < 2; ++dvk) {
                CUfunction kp = dvk ? k_dv : k_pipe, kc = dvk ? k_dv_cap : k_pipe_cap;
                const dim3 grid = dvk ? dim3(NV, 2) : dim3(NV, 1);
                const unsigned thr = dvk ? 128 : 256, sm = dvk ? smem_dv : smem_pipe;
                // Reference: plain kernel over all T, and over the first cc chunks.
                d_h1.put(h0); d_s1.fill(0x11); d_uc1.fill(0x33);
                run(kp, grid, thr, sm, d_h1.p, d_s1.p, d_uc1.p);
                d_hpre.put(h0);
                {
                    Args a;
                    a.add(d_hpre.p).add(d_w.p).add(d_u.p).add(key).add(d_gate.p).add(d_gc.p).add(d_s2.p).add(d_uc2.p)
                     .add(1u).add(cc * C).add(cc).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV).add(0u)
                     .add((int*)nullptr).add((int*)nullptr).add(0u);
                    launch(kp, grid, dim3(thr), sm, a);
                }
                d_h3.put(h0); d_s3.fill(0x11); d_uc3.fill(0x33); d_cap.fill(0x77);
                {
                    Args a;
                    a.add(d_h3.p).add(d_w.p).add(d_u.p).add(key).add(d_gate.p).add(d_gc.p).add(d_s3.p).add(d_uc3.p)
                     .add(1u).add(T).add(NC).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV).add(0u)
                     .add((int*)nullptr).add((int*)nullptr).add(0u).add(d_cap.p).add(cc);
                    launch(kc, grid, dim3(thr), sm, a);
                }
                CK(cudaDeviceSynchronize());
                const size_t a1 = diff_bytes(d_s1.get(), d_s3.get()), a2 = diff_bytes(d_uc1.get(), d_uc3.get());
                const size_t a3 = diff_bytes(d_h1.get(), d_h3.get()), a4 = diff_bytes(d_hpre.get(), d_cap.get());
                printf("bitwise %s_cap at chunk %u: S_out %zu, uc_out %zu, state %zu; captured vs %u-chunk final state %zu differing bytes\n",
                       dvk ? "pipe_dv64" : "pipe", cc, a1, a2, a3, cc, a4);
                ok_cap = ok_cap && a1 == 0 && a2 == 0 && a3 == 0 && a4 == 0;
            }
        }
        float t_c = time_ms([&] {
            d_h3.put(h0);
            Args a;
            a.add(d_h3.p).add(d_w.p).add(d_u.p).add(key).add(d_gate.p).add(d_gc.p).add(d_s3.p).add(d_uc3.p)
             .add(1u).add(T).add(NC).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV).add(0u)
             .add((int*)nullptr).add((int*)nullptr).add(0u).add(d_cap.p).add(NC / 2);
            launch(k_dv_cap, dim3(NV, 2), dim3(128), smem_dv, a);
        }, 3, 3);
        printf("pipe_dv64_cap %.3f ms (pipe_dv64 above)\n", t_c);
    }
    // ── chunk_fwd_o vs chunk_fwd_o_wide on the spine's S_out / uc_out ──
    // (`old_fla.ptx`, when present, is the module before the wide twin landed:
    // its chunk_fwd_o must also equal this build's.)
    bool ok2 = true;
    {
        const unsigned smem_fo = C * KD * 2 + C * KD * 2 + C * C * 4 + C * VD * 2 + KD * VD * 2 + 2 * C * 4;
        CUfunction k_fo = m.fn("gated_delta_rule_chunk_fwd_o", smem_fo);
        CUfunction k_fow = m.fn("gated_delta_rule_chunk_fwd_o_wide", smem_fo);
        PtxModule m_old;
        CUfunction k_fo_old = nullptr;
        if (m_old.try_load(dir + "/old_fla.ptx")) k_fo_old = m_old.fn("gated_delta_rule_chunk_fwd_o", smem_fo);
        Buf<unsigned short> d_o1, d_o2, d_o3;
        d_o1.alloc((size_t)T * NV * VD); d_o2.alloc((size_t)T * NV * VD); d_o3.alloc((size_t)T * NV * VD);
        const unsigned short* query = d_qkv.p;
        auto fo = [&](CUfunction k, unsigned short* out) {
            Args a;
            a.add(query).add(key).add(d_gate.p).add(d_gc.p).add(d_s1.p).add(d_uc1.p).add(out)
             .add(1u).add(T).add(NC).add(NK).add(NV).add(KD).add(VD).add(QK).add(2 * NV)
             .add((int*)nullptr).add((int*)nullptr).add(0u);
            launch(k, dim3(NC, NV, 1), dim3(512), smem_fo, a);
        };
        d_o1.fill(0x55); d_o2.fill(0x55); d_o3.fill(0x55);
        fo(k_fo, d_o1.p);
        fo(k_fow, d_o2.p);
        if (k_fo_old) fo(k_fo_old, d_o3.p);
        CK(cudaDeviceSynchronize());
        size_t dw = diff_bytes(d_o1.get(), d_o2.get());
        size_t dold = k_fo_old ? diff_bytes(d_o1.get(), d_o3.get()) : 0;
        printf("bitwise chunk_fwd_o vs chunk_fwd_o_wide: %zu differing bytes%s\n", dw,
               k_fo_old ? (dold == 0 ? "; this build's chunk_fwd_o == the previous build's" : "; DEFAULT CHANGED") : "");
        ok2 = dw == 0 && dold == 0;
        float t1 = time_ms([&] { fo(k_fo, d_o1.p); }, 3, 3);
        float t2 = time_ms([&] { fo(k_fow, d_o2.p); }, 3, 3);
        printf("chunk_fwd_o %.3f ms -> chunk_fwd_o_wide %.3f ms (%.2fx)\n", t1, t2, t1 / t2);
    }
    printf("%s\n", ok && ok2 && ok_cap ? "PASS" : "FAIL");
    return ok && ok2 && ok_cap ? 0 : 1;
}
