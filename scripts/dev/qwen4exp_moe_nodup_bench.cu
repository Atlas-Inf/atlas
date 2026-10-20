// SPDX-License-Identifier: AGPL-3.0-only
// ATLAS_QWEN4EXP_PREFILL_MOE_NODUP: the q38 routed-MoE prefill chain on the
// checkpoint's N-major expert planes (moe_q38{,w}n_*) against the same chain
// on the K-major duplicate (moe_q38{,w}_*), at the TP=EP=2 shape: 256 local
// experts, hidden 2560 / intermediate 640, NVFP4, a chunk routed top-10 over
// 512 experts with a lognormal skew (qwen4exp_moe_prefill_bench.cu's routing).
//
//   bitwise   act8 (gate_up + SiLU, E4M3) and down output, N-major vs K-major,
//             for q38 and W2; the K-major chain also against the PTX of
//             `<base>` (the kernel file before the NM change), if given
//   rowinv    the N-major W2 chain over the first T/3+5 tokens only: every
//             (token, expert) row equals the whole chunk's, byte for byte
//   timing    the three launches per layer, each arm; plus the on-the-fly
//             alternative's cost: `moe_transpose_u8_batched` rebuilding one
//             layer's K-major gate/up/down (packed + scales) from N-major
//
// Build + run (repo root, GB10): scripts/dev/qwen4exp_moe_nodup_bench.sh
#include "qwen4exp_ptx_harness.h"
#include <cmath>
#include <random>

static const unsigned H = 2560, I = 640, E_ALL = 512, E = 256, TOPK = 10;

static unsigned char e4m3(float f) {
    int ex;
    float m = frexpf(f, &ex);
    int e = ex - 1 + 7;
    int mant = (int)lrintf((m * 2.0f - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; ++e; }
    e = std::min(15, std::max(1, e));
    return (unsigned char)((e << 3) | mant);
}

// One projection of every local expert, in both layouts: N-major (the
// checkpoint's [N, K/2] / [N, K/16]) and K-major ([K/2, N] / [K/16, N]).
struct Proj {
    Buf<unsigned char> pn, sn, pk, sk;               // slabs
    Buf<unsigned long long> tpn, tsn, tpk, tsk;      // per-expert tables
    Buf<float> s2;
    size_t pb = 0, sb = 0;
    void make(std::mt19937& rng, unsigned K, unsigned N) {
        pb = (size_t)K / 2 * N;
        sb = (size_t)K / 16 * N;
        std::vector<unsigned char> p(pb * E), s(sb * E), pt(pb * E), st(sb * E);
        for (auto& x : p) x = (unsigned char)(rng() & 0xFF);
        std::uniform_real_distribution<float> u(0.004f, 0.06f);
        for (auto& x : s) x = e4m3(u(rng));
        for (unsigned e = 0; e < E; ++e)
            for (unsigned n = 0; n < N; ++n) {
                for (unsigned c = 0; c < K / 2; ++c)
                    pt[e * pb + (size_t)c * N + n] = p[e * pb + (size_t)n * (K / 2) + c];
                for (unsigned c = 0; c < K / 16; ++c)
                    st[e * sb + (size_t)c * N + n] = s[e * sb + (size_t)n * (K / 16) + c];
            }
        auto up = [&](Buf<unsigned char>& b, const std::vector<unsigned char>& h) { b.alloc(h.size()); b.put(h); };
        up(pn, p); up(sn, s); up(pk, pt); up(sk, st);
        auto table = [&](Buf<unsigned long long>& t, const Buf<unsigned char>& b, size_t each) {
            std::vector<unsigned long long> v(E);
            for (unsigned e = 0; e < E; ++e) v[e] = (unsigned long long)(b.p + e * each);
            t.alloc(E); t.put(v);
        };
        table(tpn, pn, pb); table(tsn, sn, sb); table(tpk, pk, pb); table(tsk, sk, sb);
        std::vector<float> v(E);
        std::uniform_real_distribution<float> u2(0.5f, 2.0f);
        for (auto& x : v) x = u2(rng);
        s2.alloc(E); s2.put(v);
    }
};

struct Routing {
    std::vector<int> offsets, sorted;
    unsigned R = 0, grid_m = 0;
};

static Routing route(const std::vector<int>& ids, unsigned tokens) {
    std::vector<std::vector<int>> rows_of(E);
    for (unsigned t = 0; t < tokens; ++t)
        for (unsigned k = 0; k < TOPK; ++k)
            if (ids[(size_t)t * TOPK + k] < (int)E) rows_of[ids[(size_t)t * TOPK + k]].push_back((int)t);
    Routing r;
    r.offsets.assign(E + 1, 0);
    for (unsigned e = 0; e < E; ++e) {
        r.offsets[e + 1] = r.offsets[e] + (int)rows_of[e].size();
        r.sorted.insert(r.sorted.end(), rows_of[e].begin(), rows_of[e].end());
    }
    r.R = r.offsets[E];
    const unsigned avg = (r.R + E - 1) / E;
    r.grid_m = std::max(1u, (avg * 2 + 127) / 128);   // try_q38_routed_prefill's grid
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx dir> [tokens=16000]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const unsigned T = argc > 2 ? atoi(argv[2]) : 16000;
    init_driver();
    PtxModule mq, mbase, mtr;
    mq.load(dir + "/moe_prefill_q38.ptx");
    const bool have_base = mbase.try_load(dir + "/moe_prefill_q38_base.ptx");
    const bool have_tr = mtr.try_load(dir + "/moe_transpose_batched.ptx");

    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<double> pop(E_ALL);
    for (auto& p : pop) p = exp(0.8 * nd(rng));
    std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::vector<int> ids((size_t)T * TOPK);
    for (unsigned t = 0; t < T; ++t)
        for (unsigned k = 0; k < TOPK; ++k) {
            int e;
            do { e = pick(rng); } while (std::find(&ids[(size_t)t * TOPK], &ids[(size_t)t * TOPK + k], e) !=
                                         &ids[(size_t)t * TOPK + k]);
            ids[(size_t)t * TOPK + k] = e;
        }
    const Routing full = route(ids, T);
    printf("tokens %u, local routed rows %u, grid.y %u\n", T, full.R, full.grid_m);

    std::vector<unsigned short> a((size_t)T * H);
    for (auto& x : a) x = f2bf(nd(rng));
    Buf<unsigned short> d_a;
    d_a.alloc(a.size()); d_a.put(a);
    Buf<unsigned char> d_a8;
    d_a8.alloc((size_t)T * H);
    Proj gate, up, down;
    gate.make(rng, H, I);
    up.make(rng, H, I);
    down.make(rng, I, H);
    Buf<int> d_off, d_sorted;
    d_off.alloc(E + 1); d_off.put(full.offsets);
    d_sorted.alloc(full.R); d_sorted.put(full.sorted);

    CUfunction k_a8 = mq.fn("moe_q38_a_to_e4m3");
    {
        Args x;
        x.add(d_a.p).add(d_a8.p).add(T * H);
        launch(k_a8, dim3((T * H / 4 + 255) / 256), dim3(256), 0, x);
    }
    // One chain: gate_up+silu -> act8, down -> out, on (gu, dn) entries, the
    // N-major (nm) or K-major tables, over routing (off, sorted, R, grid_m).
    struct Out { Buf<unsigned char> act8; Buf<unsigned short> out; };
    auto chain = [&](CUfunction gu, CUfunction dn, bool nm, const int* off, const int* sorted, unsigned R,
                     unsigned gm, Out& o, int which) {
        auto& gp = nm ? gate.tpn : gate.tpk; auto& gs = nm ? gate.tsn : gate.tsk;
        auto& upp = nm ? up.tpn : up.tpk;    auto& us = nm ? up.tsn : up.tsk;
        auto& dp = nm ? down.tpn : down.tpk; auto& ds = nm ? down.tsn : down.tsk;
        if (which & 1) {
            Args x;
            x.add(d_a8.p).add(gp.p).add(gs.p).add(gate.s2.p).add(upp.p).add(us.p).add(up.s2.p)
             .add(o.act8.p).add(off).add(sorted).add(E).add(I).add(H);
            launch(gu, dim3(I / 64, gm, E), dim3(256), 0, x);
        }
        if (which & 2) {
            Args y;
            y.add(o.act8.p).add(dp.p).add(ds.p).add(down.s2.p).add(o.out.p).add(off).add(E).add(H).add(I);
            launch(dn, dim3(H / 128, gm, E), dim3(256), 0, y);
        }
        (void)R;
    };
    auto mk_out = [&](Out& o, unsigned R) { o.act8.alloc((size_t)R * I); o.out.alloc((size_t)R * H);
                                            o.act8.fill(0x5A); o.out.fill(0x6B); };

    bool ok = true;
    const double gu_tf = 2.0 * full.R * H * 2.0 * I / 1e12, dn_tf = 2.0 * full.R * I * H / 1e12;
    struct Arm { const char* name; PtxModule* m; const char* gu; const char* dn; bool nm; };
    std::vector<Arm> arms = {
        {"K-major q38w (dup)", &mq, "moe_q38w_gate_up_silu", "moe_q38w_down", false},
        {"N-major q38wn", &mq, "moe_q38wn_gate_up_silu", "moe_q38wn_down", true},
        {"K-major q38 (dup)", &mq, "moe_q38_gate_up_silu", "moe_q38_down", false},
        {"N-major q38n", &mq, "moe_q38n_gate_up_silu", "moe_q38n_down", true},
    };
    if (have_base) arms.insert(arms.begin(), {"base q38w (HEAD PTX)", &mbase, "moe_q38w_gate_up_silu", "moe_q38w_down", false});
    std::vector<Out> outs(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        CUfunction gu = arms[i].m->fn(arms[i].gu), dn = arms[i].m->fn(arms[i].dn);
        mk_out(outs[i], full.R);
        chain(gu, dn, arms[i].nm, d_off.p, d_sorted.p, full.R, full.grid_m, outs[i], 3);
        CK(cudaDeviceSynchronize());
        const auto act = outs[i].act8.get();
        const auto out = outs[i].out.get();
        size_t da = 0, dd = 0;
        if (i > 0) { da = diff_bytes(outs[0].act8.get(), act); dd = diff_bytes(outs[0].out.get(), out); }
        ok = ok && da == 0 && dd == 0;
        const float t1 = time_ms([&] { chain(gu, dn, arms[i].nm, d_off.p, d_sorted.p, full.R, full.grid_m, outs[i], 1); });
        const float t2 = time_ms([&] { chain(gu, dn, arms[i].nm, d_off.p, d_sorted.p, full.R, full.grid_m, outs[i], 2); });
        printf("%-22s gate_up+silu %6.3f ms (%5.1f TF/s)  down %6.3f ms (%5.1f TF/s)  sum %6.3f ms  "
               "vs %s: act %zu, out %zu differing bytes\n",
               arms[i].name, t1, gu_tf / t1 * 1e3, t2, dn_tf / t2 * 1e3, t1 + t2, arms[0].name, da, dd);
    }
    const float t_a8 = time_ms([&] {
        Args x;
        x.add(d_a.p).add(d_a8.p).add(T * H);
        launch(k_a8, dim3((T * H / 4 + 255) / 256), dim3(256), 0, x);
    });
    printf("a->e4m3 (shared by every arm) %.3f ms\n", t_a8);

    // ── row invariance: the first T2 tokens alone ──
    {
        const unsigned T2 = T / 3 + 5;
        const Routing sub = route(ids, T2);
        Buf<int> s_off, s_sorted;
        s_off.alloc(E + 1); s_off.put(sub.offsets);
        s_sorted.alloc(sub.R); s_sorted.put(sub.sorted);
        Out o;
        mk_out(o, sub.R);
        chain(mq.fn("moe_q38wn_gate_up_silu"), mq.fn("moe_q38wn_down"), true, s_off.p, s_sorted.p, sub.R,
              sub.grid_m, o, 3);
        CK(cudaDeviceSynchronize());
        // Sub rows of expert e are the prefix of the whole chunk's (tokens ascend).
        const size_t wi = have_base ? 2 : 1;   // the N-major W2 arm
        const auto fa = outs[wi].act8.get(), sa = o.act8.get();
        const auto fo = outs[wi].out.get(), so = o.out.get();
        size_t bad = 0;
        for (unsigned e = 0; e < E; ++e)
            for (int j = 0; j < sub.offsets[e + 1] - sub.offsets[e]; ++j) {
                const size_t rs = sub.offsets[e] + j, rf = full.offsets[e] + j;
                bad += memcmp(&sa[rs * I], &fa[rf * I], I) != 0;
                bad += memcmp(&so[rs * H], &fo[rf * H], (size_t)H * 2) != 0;
            }
        printf("rowinv: %u of %u tokens (%u rows) alone vs the whole chunk: %zu differing rows\n", T2, T, sub.R, bad);
        ok = ok && bad == 0;
    }

    // ── on-the-fly alternative: rebuild one layer's K-major copy ──
    if (have_tr) {
        CUfunction k_tr = mtr.fn("moe_transpose_u8_batched");
        auto tr = [&](Buf<unsigned long long>& src, Buf<unsigned long long>& dst, unsigned rows, unsigned cols) {
            Args x;
            x.add(src.p).add(dst.p).add(rows).add(cols);
            launch(k_tr, dim3((cols + 31) / 32, (rows + 31) / 32, E), dim3(32, 8), 0, x);
        };
        auto layer = [&] {
            tr(gate.tpn, gate.tpk, I, H / 2); tr(gate.tsn, gate.tsk, I, H / 16);
            tr(up.tpn, up.tpk, I, H / 2);     tr(up.tsn, up.tsk, I, H / 16);
            tr(down.tpn, down.tpk, H, I / 2); tr(down.tsn, down.tsk, H, I / 16);
        };
        const float t = time_ms(layer);
        const double bytes = 2.0 * E * 3 * (gate.pb + gate.sb);
        printf("on-the-fly transpose of one layer (%.0f MB read + written): %.3f ms (%.0f GB/s)\n",
               bytes / 2e6, t, bytes / t / 1e6);
        // The kernels must still agree after the rebuild (it wrote the same bytes).
        Out o;
        mk_out(o, full.R);
        chain(mq.fn("moe_q38w_gate_up_silu"), mq.fn("moe_q38w_down"), false, d_off.p, d_sorted.p, full.R,
              full.grid_m, o, 3);
        CK(cudaDeviceSynchronize());
        const size_t d = diff_bytes(outs[0].out.get(), o.out.get());
        printf("bitwise after the rebuild: %zu differing bytes\n", d);
        ok = ok && d == 0;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
