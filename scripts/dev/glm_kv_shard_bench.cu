// SPDX-License-Identifier: AGPL-3.0-only
// Standalone check of the ATLAS_GLM_KV_SHARD merge-form attention kernels
// (one verify owner of `rows` rows x 32 heads over 2051 selected tokens of a
// `ctx`-token fp8_g128 history), three ways:
//
//   U   unsharded: the split kernel over the full pool + the BF16 merge;
//   S0  sharded (ATLAS_GLM_KV_SHARD=1): both ranks localize the selection
//       (the peer's tokens become -1), attend both ranks' heads over the
//       tokens they store, FP32-merge the peer's heads' partial, and merge;
//   S1  S0 with ATLAS_GLM_KV_SHARD_COMPACT=1: compacted ids + per-row
//       counts, counted split kernels, and the peer's partial merged where
//       it landed (no copies).
//
// Both ranks are simulated on one GPU (two half pools); the exchanges are
// plain device copies and are NOT part of what is timed as "kernels". Checks,
// for BOTH ranks' heads (exit status = failed checks):
//   - the localized ids of glm_kv_shard_localize against the ownership rule
//     on the CPU, and glm_kv_shard_localize_compact against them (the same
//     owned ids in the same order at the front, their count, -1 behind);
//   - each pipeline against a double-precision CPU softmax attention over
//     the same dequantized latents (the exact merge of its FP32 partials, and
//     its BF16 output), and S0/S1 against U and each other;
//   - the in-place merge against copy + merge of the same partials, bitwise.
// Then times one rank's kernels per layer: `reps` layers are enqueued per
// synchronize (the GPU on the bench host is time-sliced with live services,
// so a single launch mostly measures the slice), the three pipelines are
// interleaved, and the minimum and median over `iters` batches are printed
// (`iters` = 0 skips the timing). Also times the copy-engine staging copies
// of the two exchanged payloads into device-mapped pinned host memory.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_kv_shard_bench.cu -o kv_shard_bench
//   ./kv_shard_bench [rows=8] [ctx=65536] [iters=40] [skew=50] [reps=50] [causal=-1]
// `skew` = percent of the selection stored by rank 0. `causal` >= 0 takes the
// dense-exact form of sequences up to 2048 tokens instead: no selection (the
// shard kernels generate causal ids, row r = tokens [0, causal + r + 1));
// `causal` + rows <= 16 leaves rank 1 owning nothing. Device memory: the
// history twice (full + two halves), ~70 MB per 64K tokens.
#include "glm_sparse_prefill_kv_reuse.cu"
#include "glm_sparse_decode_split_merge.cu"
#define GLM_KV_SHARD_BODIES_INCLUDED  // the unsharded entry points too
#include "glm_kv_shard.cu"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t ck_ = (x); if (ck_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(ck_)); exit(1); } } while (0)

static const unsigned HEADS = 32, DIM = 512, WIDTH = 2051, BS = 16, SMEM = 69376;
static const size_t BLOCK_BYTES = (size_t)BS * 528;

static unsigned short f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float bf2f(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }
static float e4m3(unsigned char c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? ldexpf((float)m / 8.0f, -6) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (c & 0x80) ? -v : v;
}

// sparse_split_count (crates/spark-model/src/layers/ops/glm_sparse_prefill_tc.rs).
static unsigned split_count(unsigned rows) {
    unsigned best = 1, best_cost = ~0u;
    for (unsigned s = 1; s <= 16; ++s) {
        const unsigned cost = ((rows * s + 47) / 48) * ((65 + s - 1) / s);
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    return best;
}
// merge_splits (crates/spark-model/src/layers/glm_kv_shard.rs).
static unsigned merge_splits(unsigned rows) {
    return std::min(std::min(split_count(rows), 15u), std::max(192u / rows, 1u));
}

template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }

struct Rank {
    unsigned char* pool;            // this rank's half pool
    __nv_bfloat16 *q_own, *q_peer;  // [rows, 32, 512]
    int* ids;                       // [rows, 2051] localized
    unsigned* counts;               // [rows]
    float *own_o, *own_lse, *peer_o, *peer_lse, *send, *recv, *out_lse;
    __nv_bfloat16* out;             // [rows, 32, 512]
};

struct Env {
    unsigned rows, splits, rank_id;
    const int* selected;    // null: causal ids from `causal_start`
    unsigned causal_start;
    const unsigned *table, *identity;
    size_t part, lse;
};

static void partials(bool counted, const __nv_bfloat16* q, const void* pool, const int* ids,
                     const unsigned* table, const unsigned* counts, unsigned rows, unsigned splits,
                     float* po, float* pl) {
    dim3 grid(1, rows, splits);
    if (counted)
        glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted<<<grid, 256, SMEM>>>(
            q, pool, pool, ids, nullptr, table, rows, HEADS, DIM, WIDTH, BS, 0.0625f, po, pl, counts);
    else
        glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split<<<grid, 256, SMEM>>>(
            q, pool, pool, ids, nullptr, table, rows, HEADS, DIM, WIDTH, BS, 0.0625f, po, pl);
}

// One rank's launches up to the partial it sends (the peer's heads over this
// rank's tokens), as glm_shard_merge_attention orders them.
static void shard_first_half(const Env& e, Rank& r, bool compact) {
    if (compact)
        glm_kv_shard_localize_compact<<<e.rows, 256>>>(e.selected, r.ids, r.counts, e.table,
            e.rows, WIDTH, BS, e.rank_id, 2, e.causal_start);
    else
        glm_kv_shard_localize<<<dim3((WIDTH + 255) / 256, e.rows), 256>>>(e.selected, r.ids,
            e.table, e.rows, WIDTH, BS, e.rank_id, 2, e.causal_start);
    if (e.splits == 1) {
        partials(compact, r.q_peer, r.pool, r.ids, e.identity, r.counts, e.rows, 1, r.send,
                 r.send + e.part / 4);
    } else {
        partials(compact, r.q_peer, r.pool, r.ids, e.identity, r.counts, e.rows, e.splits,
                 r.peer_o, r.peer_lse);
        glm_sparse_decode_split_merge_f32<<<e.rows * HEADS, 256>>>(r.peer_o, r.peer_lse, r.send,
            r.send + e.part / 4, e.rows, HEADS, DIM, e.splits);
    }
}

// ...and from the peer's partial (already in `recv`) to the BF16 output.
static void shard_second_half(const Env& e, Rank& r, bool compact) {
    partials(compact, r.q_own, r.pool, r.ids, e.identity, r.counts, e.rows, e.splits, r.own_o,
             r.own_lse);
    if (compact) {
        glm_sparse_decode_split_merge_extra<<<e.rows * HEADS, 256>>>(r.own_o, r.own_lse, r.out,
            r.out_lse, e.rows, HEADS, DIM, e.splits, r.recv);
    } else {
        CK(cudaMemcpyAsync((char*)r.own_o + e.splits * e.part, r.recv, e.part,
                           cudaMemcpyDeviceToDevice, 0));
        CK(cudaMemcpyAsync((char*)r.own_lse + e.splits * e.lse, (char*)r.recv + e.part, e.lse,
                           cudaMemcpyDeviceToDevice, 0));
        glm_sparse_decode_split_merge<<<e.rows * HEADS, 256>>>(r.own_o, r.own_lse, r.out,
            r.out_lse, e.rows, HEADS, DIM, e.splits + 1);
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double lowest(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

// Max abs difference of two BF16 tensors (NaN if any element's is) and how
// many elements differ.
struct Diff { double max; size_t count; };
static Diff diff(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b) {
    Diff d{0, 0};
    bool nan = false;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = fabsf(bf2f(a[i]) - bf2f(b[i]));
        nan = nan || std::isnan(e);
        d.max = std::max(d.max, e);
        d.count += a[i] != b[i];
    }
    if (nan) d.max = NAN;
    return d;
}

// The exact (double) LSE merge of `n` FP32 partials `[n, rows*heads, dim]`
// with LSEs `[n, rows*heads]`, against `ref`: max abs error before any BF16
// rounding (NaN if the merge has one).
static double merged_error(const std::vector<const float*>& o, const std::vector<const float*>& l,
                           size_t rh, const std::vector<double>& ref) {
    double worst = 0;
    for (size_t i = 0; i < rh; ++i) {
        double mx = -INFINITY;
        for (auto p : l) mx = std::max(mx, (double)p[i]);
        double z = 0;
        for (auto p : l) z += exp((double)p[i] - mx);
        for (unsigned d = 0; d < DIM; ++d) {
            double v = 0;
            for (size_t s = 0; s < o.size(); ++s)
                v += exp((double)l[s][i] - mx) / z * (double)o[s][i * DIM + d];
            if (std::isnan(v)) return NAN;
            worst = std::max(worst, fabs(v - ref[i * DIM + d]));
        }
    }
    return worst;
}

int main(int argc, char** argv) {
    const unsigned rows = argc > 1 ? atoi(argv[1]) : 8;
    const unsigned ctx = argc > 2 ? atoi(argv[2]) : 65536;
    const unsigned iters = argc > 3 ? atoi(argv[3]) : 40;
    const unsigned skew = argc > 4 ? atoi(argv[4]) : 50;
    const unsigned reps = argc > 5 ? atoi(argv[5]) : 50;
    const int causal = argc > 6 ? atoi(argv[6]) : -1;
    if (causal >= 0 && (unsigned)causal + rows > std::min(WIDTH, ctx)) {
        fprintf(stderr, "causal + rows must fit %u ids and the context\n", WIDTH);
        return 2;
    }
    const unsigned blocks = ctx / BS, local_blocks = (blocks + 1) / 2;
    const unsigned splits_u = split_count(rows), splits = merge_splits(rows);
    std::mt19937 rng(1234);

    // History: random E4M3 codes (no NaN codes) and per-128 scales, in a
    // shuffled table that keeps the shard invariant table[l] % 2 == l % 2.
    std::vector<unsigned char> full(blocks * BLOCK_BYTES);
    for (unsigned b = 0; b < blocks; ++b) {
        unsigned char* p = &full[b * BLOCK_BYTES];
        for (unsigned i = 0; i < BS * 512; ++i) {
            unsigned char c = (unsigned char)(rng() & 0xff);
            if ((c & 0x7f) == 0x7f) c ^= 1;
            p[i] = c;
        }
        float* scales = (float*)(p + BS * 512);
        for (unsigned i = 0; i < BS * 4; ++i) scales[i] = (1.0f + (rng() % 1000) / 250.0f) / 448.0f;
    }
    std::vector<unsigned> table(blocks), even, odd;
    for (unsigned b = 0; b < blocks; ++b) (b % 2 ? odd : even).push_back(b);
    std::shuffle(even.begin(), even.end(), rng);
    std::shuffle(odd.begin(), odd.end(), rng);
    for (unsigned l = 0; l < blocks; ++l) table[l] = (l % 2 ? odd : even)[l / 2];
    std::vector<unsigned char> half[2];
    for (auto& h : half) h.resize(local_blocks * BLOCK_BYTES);
    for (unsigned b = 0; b < blocks; ++b)
        memcpy(&half[b % 2][(b / 2) * BLOCK_BYTES], &full[b * BLOCK_BYTES], BLOCK_BYTES);
    std::vector<unsigned> identity(std::max(blocks, local_blocks));
    for (unsigned i = 0; i < identity.size(); ++i) identity[i] = i;

    // Selection: distinct tokens per row, `skew` percent stored by rank 0; or
    // the causal ids, which only the unsharded pipeline and the CPU are given.
    std::vector<int> selected(rows * WIDTH);
    for (unsigned r = 0; r < rows && causal >= 0; ++r)
        for (unsigned j = 0; j < WIDTH; ++j)
            selected[r * WIDTH + j] = j < (unsigned)causal + r + 1 ? (int)j : -1;
    for (unsigned r = 0; r < rows && causal < 0; ++r) {
        std::vector<int> t0, t1;
        for (unsigned t = 0; t < ctx; ++t) ((t / BS) % 2 ? t1 : t0).push_back((int)t);
        std::shuffle(t0.begin(), t0.end(), rng);
        std::shuffle(t1.begin(), t1.end(), rng);
        const unsigned n0 = std::min<size_t>(WIDTH * skew / 100, t0.size());
        std::vector<int> row(t0.begin(), t0.begin() + n0);
        row.insert(row.end(), t1.begin(), t1.begin() + (WIDTH - n0));
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), selected.begin() + r * WIDTH);
    }
    std::normal_distribution<float> nd(0.0f, 0.3f);
    std::vector<unsigned short> q[2];
    for (auto& v : q) { v.resize(rows * HEADS * DIM); for (auto& x : v) x = f2bf(nd(rng)); }

    unsigned char* d_full = dalloc<unsigned char>(full.size());
    CK(cudaMemcpy(d_full, full.data(), full.size(), cudaMemcpyHostToDevice));
    unsigned* d_table = dalloc<unsigned>(blocks);
    CK(cudaMemcpy(d_table, table.data(), blocks * 4, cudaMemcpyHostToDevice));
    unsigned* d_identity = dalloc<unsigned>(identity.size());
    CK(cudaMemcpy(d_identity, identity.data(), identity.size() * 4, cudaMemcpyHostToDevice));
    int* d_selected = dalloc<int>(selected.size());
    CK(cudaMemcpy(d_selected, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));

    const size_t part = (size_t)rows * HEADS * DIM * 4, lse = (size_t)rows * HEADS * 4;
    const size_t qn = (size_t)rows * HEADS * DIM;
    Rank rk[2];
    __nv_bfloat16* d_q[2];
    for (int i = 0; i < 2; ++i) {
        d_q[i] = dalloc<__nv_bfloat16>(qn);
        CK(cudaMemcpy(d_q[i], q[i].data(), qn * 2, cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < 2; ++i) {
        Rank& r = rk[i];
        r.pool = dalloc<unsigned char>(half[i].size());
        CK(cudaMemcpy(r.pool, half[i].data(), half[i].size(), cudaMemcpyHostToDevice));
        r.q_own = d_q[i];
        r.q_peer = dalloc<__nv_bfloat16>(qn);  // the exchanged copy
        CK(cudaMemcpy(r.q_peer, d_q[1 - i], qn * 2, cudaMemcpyDeviceToDevice));
        r.ids = dalloc<int>(rows * WIDTH);
        r.counts = dalloc<unsigned>(rows);
        r.own_o = dalloc<float>((splits + 1) * part / 4);
        r.own_lse = dalloc<float>((splits + 1) * lse / 4);
        r.peer_o = dalloc<float>(splits * part / 4);
        r.peer_lse = dalloc<float>(splits * lse / 4);
        r.send = dalloc<float>((part + lse) / 4);
        r.recv = dalloc<float>((part + lse) / 4);
        r.out_lse = dalloc<float>(lse / 4);
        r.out = dalloc<__nv_bfloat16>(qn);
    }
    float* u_o = dalloc<float>(splits_u * part / 4);
    float* u_lse = dalloc<float>(splits_u * lse / 4);
    float* u_out_lse = dalloc<float>(lse / 4);
    __nv_bfloat16* u_out = dalloc<__nv_bfloat16>(qn);

    CK(cudaFuncSetAttribute(glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    auto unsharded = [&](int rank) {
        partials(false, d_q[rank], d_full, d_selected, d_table, nullptr, rows, splits_u, u_o, u_lse);
        glm_sparse_decode_split_merge<<<rows * HEADS, 256>>>(u_o, u_lse, u_out, u_out_lse, rows,
            HEADS, DIM, splits_u);
    };
    Env env[2];
    for (unsigned i = 0; i < 2; ++i)
        env[i] = Env{rows, splits, i, causal >= 0 ? nullptr : d_selected,
                     (unsigned)std::max(causal, 0), d_table, d_identity, part, lse};
    // Both ranks of one sharded layer, with the partial exchange as a copy.
    auto sharded = [&](bool compact) {
        for (int i = 0; i < 2; ++i) shard_first_half(env[i], rk[i], compact);
        for (int i = 0; i < 2; ++i)
            CK(cudaMemcpyAsync(rk[i].recv, rk[1 - i].send, part + lse, cudaMemcpyDeviceToDevice, 0));
        for (int i = 0; i < 2; ++i) shard_second_half(env[i], rk[i], compact);
    };
    auto fetch = [&](const __nv_bfloat16* p) {
        std::vector<unsigned short> h(qn);
        CK(cudaMemcpy(h.data(), p, qn * 2, cudaMemcpyDeviceToHost));
        return h;
    };

    // FP32 partials of one pipeline for one rank's heads: its own `n` splits
    // and (sharded) the peer's merged partial.
    struct Parts { std::vector<float> o, l; };
    auto fetch_f = [&](const float* p, size_t floats) {
        std::vector<float> h(floats);
        CK(cudaMemcpy(h.data(), p, floats * 4, cudaMemcpyDeviceToHost));
        return h;
    };
    auto fetch_ids = [&](const int* p) {
        std::vector<int> h(rows * WIDTH);
        CK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
        return h;
    };
    const size_t rh = (size_t)rows * HEADS;

    // ---- Correctness: both ranks' heads, all three pipelines vs the CPU. ----
    std::vector<unsigned short> out_u[2], out_s0[2], out_s1[2];
    Parts pu[2], ps0[2], ps1[2];
    std::vector<float> peer_s0[2], peer_s1[2];
    std::vector<int> ids_s0[2], ids_s1[2];
    std::vector<unsigned> counts[2];
    for (int i = 0; i < 2; ++i) {
        unsharded(i); CK(cudaDeviceSynchronize());
        out_u[i] = fetch(u_out);
        pu[i] = Parts{fetch_f(u_o, splits_u * rh * DIM), fetch_f(u_lse, splits_u * rh)};
    }
    sharded(false); CK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i) {
        out_s0[i] = fetch(rk[i].out);
        ps0[i] = Parts{fetch_f(rk[i].own_o, splits * rh * DIM), fetch_f(rk[i].own_lse, splits * rh)};
        peer_s0[i] = fetch_f(rk[i].recv, rh * (DIM + 1));
        ids_s0[i] = fetch_ids(rk[i].ids);
    }
    sharded(true); CK(cudaDeviceSynchronize());
    bool extra_same = true;
    for (int i = 0; i < 2; ++i) {
        out_s1[i] = fetch(rk[i].out);
        ps1[i] = Parts{fetch_f(rk[i].own_o, splits * rh * DIM), fetch_f(rk[i].own_lse, splits * rh)};
        peer_s1[i] = fetch_f(rk[i].recv, rh * (DIM + 1));
        ids_s1[i] = fetch_ids(rk[i].ids);
        counts[i].resize(rows);
        CK(cudaMemcpy(counts[i].data(), rk[i].counts, rows * 4, cudaMemcpyDeviceToHost));
        // The in-place merge against copying the partial behind the own
        // partitions and merging splits + 1 (what S0 does): same bits.
        CK(cudaMemcpy((char*)rk[i].own_o + splits * part, rk[i].recv, part, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy((char*)rk[i].own_lse + splits * lse, (char*)rk[i].recv + part, lse,
                      cudaMemcpyDeviceToDevice));
        glm_sparse_decode_split_merge<<<rows * HEADS, 256>>>(rk[i].own_o, rk[i].own_lse, u_out,
            u_out_lse, rows, HEADS, DIM, splits + 1);
        CK(cudaDeviceSynchronize());
        extra_same = extra_same && fetch(u_out) == out_s1[i];
    }
    CK(cudaGetLastError());

    // The localized ids: glm_kv_shard_localize against the ownership rule,
    // and glm_kv_shard_localize_compact against glm_kv_shard_localize.
    bool localize_ok = true, compact_ok = true;
    unsigned empty_rows = 0;
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned r = 0; r < rows; ++r) {
            std::vector<int> owned;
            for (unsigned j = 0; j < WIDTH; ++j) {
                const int t = selected[r * WIDTH + j];
                const unsigned block = t >= 0 ? table[(unsigned)t / BS] : ~0u;
                const int want = t >= 0 && block % 2 == i ? (int)((block / 2) * BS + (unsigned)t % BS) : -1;
                localize_ok = localize_ok && ids_s0[i][r * WIDTH + j] == want;
                if (want >= 0) owned.push_back(want);
            }
            const int* got = &ids_s1[i][r * WIDTH];
            compact_ok = compact_ok && counts[i][r] == owned.size()
                && std::equal(owned.begin(), owned.end(), got)
                && std::all_of(got + owned.size(), got + WIDTH, [](int v) { return v == -1; });
            empty_rows += owned.empty();
        }
    }
    printf("rows=%u ctx=%u splits U=%u shard=%u skew=%u%% causal=%d counts[0]: rank0=%u rank1=%u "
           "(of %u); rows a rank owns nothing of: %u\n", rows, ctx, splits_u, splits, skew, causal,
           counts[0][0], counts[1][0], WIDTH, empty_rows);
    printf("localized ids vs the ownership rule: %s; compact vs localize (stable pack + counts): %s\n",
           localize_ok ? "ok" : "MISMATCH", compact_ok ? "ok" : "MISMATCH");
    unsigned failed = !localize_ok + !compact_ok + !extra_same;

    for (unsigned i = 0; i < 2; ++i) {
        std::vector<unsigned short> ref(qn);
        std::vector<double> ref_d(qn);
        double ref_rms = 0, ref_max = 0;
        std::vector<float> k((size_t)WIDTH * DIM);
        std::vector<double> logit(WIDTH), acc(DIM);
        for (unsigned r = 0; r < rows; ++r) {
            unsigned n = 0;  // the row's selected tokens, -1 skipped
            for (unsigned j = 0; j < WIDTH; ++j) {
                if (selected[r * WIDTH + j] < 0) continue;
                const unsigned t = (unsigned)selected[r * WIDTH + j];
                const unsigned char* blk = &full[(size_t)table[t / BS] * BLOCK_BYTES];
                const float* scales = (const float*)(blk + BS * 512) + (t % BS) * 4;
                for (unsigned d = 0; d < DIM; ++d)
                    k[(size_t)n * DIM + d] = bf2f(f2bf(e4m3(blk[(t % BS) * 512 + d]) * scales[d / 128]));
                ++n;
            }
            for (unsigned h = 0; h < HEADS; ++h) {
                const unsigned short* qh = &q[i][((size_t)r * HEADS + h) * DIM];
                double mx = -1e300;
                for (unsigned j = 0; j < n; ++j) {
                    double sum = 0;
                    for (unsigned d = 0; d < DIM; ++d) sum += (double)bf2f(qh[d]) * k[(size_t)j * DIM + d];
                    logit[j] = sum * 0.0625;
                    mx = std::max(mx, logit[j]);
                }
                double z = 0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (unsigned j = 0; j < n; ++j) {
                    const double p = exp(logit[j] - mx);
                    z += p;
                    for (unsigned d = 0; d < DIM; ++d) acc[d] += p * k[(size_t)j * DIM + d];
                }
                for (unsigned d = 0; d < DIM; ++d) {
                    const double v = acc[d] / z;
                    ref_d[((size_t)r * HEADS + h) * DIM + d] = v;
                    ref[((size_t)r * HEADS + h) * DIM + d] = f2bf((float)v);
                    ref_rms += v * v;
                    ref_max = std::max(ref_max, fabs(v));
                }
            }
        }
        ref_rms = sqrt(ref_rms / qn);
        auto parts_error = [&](const Parts& p, unsigned n, const std::vector<float>* peer) {
            std::vector<const float*> o, l;
            for (unsigned s = 0; s < n; ++s) {
                o.push_back(&p.o[s * rh * DIM]);
                l.push_back(&p.l[s * rh]);
            }
            if (peer) { o.push_back(peer->data()); l.push_back(peer->data() + rh * DIM); }
            return merged_error(o, l, rh, ref_d);
        };
        const double eu = parts_error(pu[i], splits_u, nullptr);
        const double e0 = parts_error(ps0[i], splits, &peer_s0[i]);
        const double e1 = parts_error(ps1[i], splits, &peer_s1[i]);
        printf("rank %u heads: FP32 partials, exact merge, max abs error vs CPU double attention "
               "(output rms %.4f): U=%.3e S0=%.3e S1=%.3e\n", i, ref_rms, eu, e0, e1);
        const Diff du = diff(out_u[i], ref), d0 = diff(out_s0[i], ref), d1 = diff(out_s1[i], ref);
        printf("rank %u heads: BF16 output vs bf16(CPU double): max abs / differing of %zu: "
               "U=%.3e/%zu S0=%.3e/%zu S1=%.3e/%zu\n", i, qn, du.max, du.count, d0.max, d0.count,
               d1.max, d1.count);
        const Diff a0 = diff(out_s0[i], out_u[i]), a1 = diff(out_s1[i], out_u[i]),
                   a2 = diff(out_s1[i], out_s0[i]);
        printf("rank %u heads: BF16 output pairs: S0 vs U=%.3e/%zu  S1 vs U=%.3e/%zu  "
               "S1 vs S0=%.3e/%zu\n", i, a0.max, a0.count, a1.max, a1.count, a2.max, a2.count);
        // The kernels round each probability to BF16, so allow 2% of the
        // output rms before the output's own rounding and two BF16 ulps of
        // the largest value after it. `!(x <= tol)` also fails a NaN.
        const double tol = 0.02 * ref_rms, tol_bf16 = tol + ref_max / 64.0;
        for (double e : {eu, e0, e1}) failed += !(e <= tol);
        for (double e : {du.max, d0.max, d1.max}) failed += !(e <= tol_bf16);
    }
    printf("merge_extra vs copy + merge of the same partials (both ranks): %s\n",
           extra_same ? "bitwise equal" : "DIFFERS");
    printf("CHECK %s (%u failed)\n", failed ? "FAIL" : "PASS", failed);
    if (iters == 0) return (int)failed;

    // ---- Timing: one rank's kernels per MLA layer, pipelines interleaved. ----
    auto rank_layer = [&](bool compact) {
        shard_first_half(env[0], rk[0], compact);
        shard_second_half(env[0], rk[0], compact);
    };
    auto timed = [&](auto&& launches) {
        const double t0 = now_us();
        for (unsigned i = 0; i < reps; ++i) launches();
        CK(cudaDeviceSynchronize());
        return (now_us() - t0) / reps;
    };
    std::vector<double> tu, ts0, ts1;
    for (unsigned i = 0; i < iters + 2; ++i) {
        const double a = timed([&] { unsharded(0); });
        const double b = timed([&] { rank_layer(false); });
        const double c = timed([&] { rank_layer(true); });
        if (i >= 2) { tu.push_back(a); ts0.push_back(b); ts1.push_back(c); }
    }
    printf("kernel us per layer, min (median) of %u x %u: U=%.1f (%.1f)  S0=%.1f (%.1f)  "
           "S1=%.1f (%.1f)\n", iters, reps, lowest(tu), median(tu), lowest(ts0), median(ts0),
           lowest(ts1), median(ts1));
    printf("shard kernel cost per layer over U (min): S0=%+.1f us  S1=%+.1f us\n",
           lowest(ts0) - lowest(tu), lowest(ts1) - lowest(tu));

    // ---- Copy-engine staging of the exchanged payloads (send + land). ----
    const size_t q_bytes = qn * 2, p_bytes = part + lse;
    void* pinned;
    CK(cudaHostAlloc(&pinned, 2 * p_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    void* pinned_dev;
    CK(cudaHostGetDevicePointer(&pinned_dev, pinned, 0));
    for (size_t bytes : {q_bytes, p_bytes}) {
        std::vector<double> t;
        for (unsigned i = 0; i < iters + 2; ++i) {
            const double us = timed([&] {
                CK(cudaMemcpyAsync(pinned_dev, rk[0].send, bytes, cudaMemcpyDeviceToDevice, 0));
                CK(cudaMemcpyAsync(rk[0].recv, (char*)pinned_dev + p_bytes, bytes,
                                   cudaMemcpyDeviceToDevice, 0));
            });
            if (i >= 2) t.push_back(us);
        }
        printf("stage + land copies of %zu bytes: min %.1f us (median %.1f)\n", bytes, lowest(t),
               median(t));
    }
    return (int)failed;
}
