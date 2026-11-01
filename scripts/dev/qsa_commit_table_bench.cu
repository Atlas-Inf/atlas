// SPDX-License-Identifier: AGPL-3.0-only
// Parity and cost of qsa_commit_table (ATLAS_QWEN4EXP_QSA_COMMIT_TABLE,
// kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu) against what it
// replaces: per commit, the pitched copy of `count` staged raw keys into the
// raw window, then qsa_block_pool of the blocks they complete.
//
// check: E entries (a layer x sequence each, own staging, window, norm
// weight and pooled-key buffer), random ingested position, row count 1..8,
// staging row; both forms on copies of the same buffers; raw windows and
// pooled keys compared byte for byte. Repeated over seeds and E.
// time: GPU us of the two forms at E = 96 (12 layers x 8 sequences).
//
// Build/run (repo root, GB10, in atlas-release-builder:1.93.1):
//   nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr \
//     kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu -o OUT/qsa_indexer.ptx
//   nvcc -O3 -std=c++17 -arch=sm_121a scripts/dev/qsa_commit_table_bench.cu -lcuda -o OUT/b
//   OUT/b OUT
#include "qwen4exp_ptx_harness.h"
#include <random>

typedef unsigned short bf;
typedef unsigned int u32;
static const u32 HD = 128, RATIO = 4, ROT = 64, QKW = 640, STAGE_ROWS = 16, CAP = 4096,
                 MAXB = CAP / RATIO;
static const float THETA = 10000.0f, EPS = 1e-6f;
static std::mt19937 rng(7);

struct Entry { const bf* src; bf* dst; const bf* raw; const bf* w; bf* bk; u32 pitch, count, first, n_new; };
static_assert(sizeof(Entry) == 56, "QsaCommitEntry layout");

struct Side { Buf<bf> raw, bk; };
struct Case { Buf<bf> stage, w; Side a, b; u32 pos, row, count; };

static std::vector<bf> rbf(size_t n) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<bf> v(n);
    for (auto& x : v) x = f2bf(u(rng));
    return v;
}

static void setup(Case& c) {
    c.stage.alloc((size_t)STAGE_ROWS * QKW); c.stage.put(rbf(c.stage.n));
    c.w.alloc(HD); c.w.put(rbf(HD));
    auto raw = rbf((size_t)CAP * HD);
    for (Side* s : {&c.a, &c.b}) {
        s->raw.alloc((size_t)CAP * HD); s->raw.put(raw);
        s->bk.alloc((size_t)MAXB * HD); s->bk.fill(0x7F);
    }
    c.count = 1 + rng() % 8;
    c.row = rng() % (STAGE_ROWS - c.count);
    c.pos = rng() % (CAP - 64);
}

static Entry entry(const Case& c, const Side& s) {
    const u32 pooled = c.pos / RATIO, complete = (c.pos + c.count) / RATIO;
    return {c.stage.p + (size_t)c.row * QKW, s.raw.p + (size_t)c.pos * HD, s.raw.p, c.w.p, s.bk.p,
            QKW, c.count, pooled, complete - pooled};
}

static void per_commit(CUfunction pool, std::vector<Case>& cs, cudaStream_t st) {
    for (auto& c : cs) {
        Entry e = entry(c, c.a);
        CK(cudaMemcpy2DAsync(e.dst, HD * 2, e.src, QKW * 2, HD * 2, c.count, cudaMemcpyDeviceToDevice, st));
        if (e.n_new) {
            Args a;
            a.add(e.raw).add(e.w).add(e.bk).add(e.first).add(RATIO).add(HD).add(ROT).add(THETA).add(EPS);
            launch(pool, dim3(e.n_new), dim3(HD), (HD + 32) * 4, a, st);
        }
    }
}

static void table(CUfunction tk, std::vector<Case>& cs, Buf<unsigned char>& tb, cudaStream_t st) {
    std::vector<Entry> t;
    for (auto& c : cs) t.push_back(entry(c, c.b));
    CK(cudaMemcpyAsync(tb.p, t.data(), t.size() * sizeof(Entry), cudaMemcpyHostToDevice, st));
    Args a;
    a.add((const Entry*)tb.p).add(RATIO).add(HD).add(ROT).add(THETA).add(EPS);
    launch(tk, dim3((u32)t.size()), dim3(HD), (HD + 32) * 4, a, st);
}

static size_t diff(const Buf<bf>& x, const Buf<bf>& y) {
    auto a = x.get(), b = y.get();
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i++) n += a[i] != b[i];
    return n;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    init_driver();
    PtxModule m;
    m.load(dir + "/qsa_indexer.ptx");
    CUfunction pool = m.fn("qsa_block_pool"), tk = m.fn("qsa_commit_table");
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    Buf<unsigned char> tb;
    tb.alloc(128 * sizeof(Entry));
    int bad = 0, entries = 0, blocks = 0;
    for (u32 E : {1u, 2u, 7u, 24u, 96u, 128u}) {
        for (int rep = 0; rep < 4; rep++) {
            std::vector<Case> cs(E);
            for (auto& c : cs) setup(c);
            CK(cudaDeviceSynchronize());
            per_commit(pool, cs, st);
            table(tk, cs, tb, st);
            CK(cudaStreamSynchronize(st));
            for (auto& c : cs) {
                entries++;
                blocks += entry(c, c.a).n_new;
                const size_t r = diff(c.a.raw, c.b.raw), k = diff(c.a.bk, c.b.bk);
                if (r || k) {
                    bad++;
                    printf("  MISMATCH E=%u pos=%u count=%u: raw %zu, pooled %zu elements\n", E, c.pos, c.count, r, k);
                }
            }
            for (auto& c : cs) { c.stage.free_(); c.w.free_(); for (Side* s : {&c.a, &c.b}) { s->raw.free_(); s->bk.free_(); } }
        }
    }
    printf("%s qsa_commit_table vs pitched copy + qsa_block_pool: %d entries, %d pooled blocks: "
           "raw windows and pooled keys byte-equal\n", bad ? "BAD" : "ok ", entries, blocks);
    // Time at E = 96: events around 20 repetitions of each form.
    std::vector<Case> cs(96);
    for (auto& c : cs) setup(c);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int form = 0; form < 2; form++) {
        float best = 1e9f;
        for (int r = 0; r < 20; r++) {
            CK(cudaEventRecord(e0, st));
            if (form == 0) per_commit(pool, cs, st); else table(tk, cs, tb, st);
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        printf("  E=96 %s: %.1f us (best of 20, eager launches)\n", form ? "table" : "per-commit", best * 1e3f);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
