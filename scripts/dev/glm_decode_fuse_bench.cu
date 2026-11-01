// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check and A/B timing of the GLM fused decode tier
// (ATLAS_GLM_DECODE_FUSE) against the launch chains it replaces, on the real
// verify shapes (hidden 4096, hc_mult 4, 24 mix rows, top-8 of 288 experts)
// at 1..8 rows and at 32:
//
//   hc     hc_post_bf16 | hc_post_bf16_add_bf16 -> glm_hc_decode_partial_bf16
//      vs  glm_hc_decode_post_bf16 -> glm_hc_decode_partial_rows_bf16
//   seam   glm_hc_decode_post_partial_bf16
//      vs  glm_hc_decode_post_partial_rows_bf16
//          (both then glm_hc_decode_finalize_bf16 -> rms_norm_vanilla)
//   moe    moe_unpermute_reduce_indexed_ep_vec8 -> moe_batched_blend
//      vs  moe_unpermute_blend_ep_vec8
//   sort   moe_sort_by_expert -> moe_build_tile_worklist (after moe_topk_sigmoid_batched)
//      vs  moe_sort_by_expert_scan -> moe_build_tile_worklist_scan
//   norm   rms_norm_vanilla vs rms_norm_vanilla_regs (hidden 4096, 1536, 512;
//          the odd 4095 at one row, the only odd shape either kernel can read)
//
// Every output byte (highway, partial sums, collapsed row, post, comb,
// normed row, MoE output) must match, including rows holding zeros, -0.0,
// large values, Inf and NaN. It then times each chain as serving launches it
// (PDL where the runtime lists the kernel), cycling `copies` weight sets
// (hc_fn is 1.5 MiB a site, so 32 copies do not fit the 24 MiB L2).
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_decode_fuse_bench.cu -o glm_decode_fuse_bench
//   ./glm_decode_fuse_bench [copies=32] [groups=64] [reps=3] [fused=31]
// Prints one "bitwise" line per chain and width, then PASS or FAIL (exit 1),
// then the median GPU time per chain in microseconds: stream events around
// every 8 chains, each batch queued behind a spin kernel so the GPU never
// waits for the host. `fused` selects the fused groups of the new arm
// (1 post, 2 partial, 4 MoE unpermute+blend, 8 MoE sort, 16 RMS norm). The
// sort's rows within an expert group are unordered by contract (atomics), so
// it compares expert_offsets bytes and checks both permutations route every
// slot to its token and expert. Device memory: ~60 MB.
#include "hyper_connection.cu"
#include "glm_hc_prefill_vec.cu"
#include "moe_permute.cu"
#include "glm_rms_norm_regs.cu"
#include "../../common/moe_topk_sigmoid.cu"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned H = 4096, HC = 4, MIX = 24, SPLIT = 64, TOPK = 8, EXPERTS = 288, MAXT = 32;

// Launch as serving does under ATLAS_PDL=1: with programmatic stream
// serialization when the kernel is on the runtime's PDL list (`pdl`).
template <typename... Args>
static void launch_as(bool pdl, void (*kernel)(Args...), dim3 grid, unsigned block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

template <typename... Args>
static void launch(void (*kernel)(Args...), dim3 grid, unsigned block, Args... args) {
    launch_as(true, kernel, grid, block, args...);
}

static unsigned short tobf(float f) {
    bf b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    void alloc(size_t count) { n = count; CK(cudaMalloc(&p, n * sizeof(T))); }
    void put(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

// One arm's mutable state: the highway and every output buffer.
struct Arm {
    Dev<unsigned short> streams, hidden, normed, moe_out;
    Dev<float> partial, post, comb;
    void alloc() {
        streams.alloc((size_t)MAXT * HC * H); hidden.alloc((size_t)MAXT * H); normed.alloc((size_t)MAXT * H);
        moe_out.alloc((size_t)MAXT * H); partial.alloc((size_t)SPLIT * MAXT * 25);
        post.alloc(MAXT * HC); comb.alloc(MAXT * HC * HC);
    }
};

// MoE routing state of one arm (top-k then the sort).
struct Route {
    Dev<unsigned int> ids;
    Dev<float> weights;
    Dev<int> sorted_tok, sorted_exp, offsets, perm, total;
    Dev<unsigned int> worklist;
    void alloc() {
        ids.alloc(MAXT * 8); weights.alloc(MAXT * 8); sorted_tok.alloc(MAXT * 8); sorted_exp.alloc(MAXT * 8);
        offsets.alloc(289); perm.alloc(MAXT * 8); total.alloc(1); worklist.alloc(2 * MAXT * 8 * 4);
    }
};

struct Inputs {
    Dev<unsigned short> block_out, peer, norm_w, expert_out, shared, gate_w;
    Dev<float> hc_fn, hc_scale, hc_base, topk_w;
    Dev<int> perm, ids;
    Dev<unsigned short> logits, quant_in;
    Dev<float> bias;
    Dev<unsigned long long> expert_ptrs;
    int copies;
};

static const unsigned SINK = 20;
static const float NORM_EPS = 1e-5f, HC_EPS = 1e-6f;
static unsigned g_fused = 31;

// One mHC site: an optional post (`peer`: fold the other rank's block output
// in), the next site's pre-mix partials, then finalize and the RMS norm.
static void hc_site(const Inputs& in, Arm& a, unsigned T, int copy, bool post, bool peer, bool seam, bool fused) {
    const float* fn = in.hc_fn.p + (size_t)copy * MIX * HC * H;
    bf* st = (bf*)a.streams.p;
    const bool fpost = fused && (g_fused & 1), fpartial = fused && (g_fused & 2);
    const dim3 pgrid(SPLIT, (T + 3) / 4);
    if (seam) {
        launch(fpartial ? glm_hc_decode_post_partial_rows_bf16 : glm_hc_decode_post_partial_bf16, pgrid, 128,
               (const bf*)in.block_out.p, st, (const float*)a.post.p, (const float*)a.comb.p, fn, a.partial.p, T);
    } else {
        if (post && fpost)
            launch(glm_hc_decode_post_bf16, dim3(2 * T), 256, (const bf*)in.block_out.p,
                   peer ? (const bf*)in.peer.p : (const bf*)nullptr, st, (const float*)a.post.p, (const float*)a.comb.p, T);
        else if (post && peer)  // not on the runtime's PDL list
            launch_as(false, hc_post_bf16_add_bf16, dim3(T), 256, (const bf*)in.block_out.p, (const bf*)in.peer.p,
                      (const bf*)st, (const float*)a.post.p, (const float*)a.comb.p, st, H, HC);
        else if (post)
            launch(hc_post_bf16, dim3(T), 256, (const bf*)in.block_out.p, (const bf*)st,
                   (const float*)a.post.p, (const float*)a.comb.p, st, H, HC);
        launch(fpartial ? glm_hc_decode_partial_rows_bf16 : glm_hc_decode_partial_bf16, pgrid, 128, (const bf*)nullptr,
               st, (const float*)a.post.p, (const float*)a.comb.p, fn, a.partial.p, T);
    }
    launch(glm_hc_decode_finalize_bf16, dim3(T), 256, (const bf*)st, (const float*)a.partial.p,
           (const float*)in.hc_scale.p, (const float*)in.hc_base.p, (bf*)a.hidden.p, a.post.p, a.comb.p,
           T, SINK, NORM_EPS, HC_EPS);
    launch(rms_norm_vanilla, dim3(T), 1024, (const bf*)a.hidden.p, (const bf*)(in.norm_w.p + (size_t)copy * H),
           (bf*)a.normed.p, H, NORM_EPS);
}

static void moe_post(const Inputs& in, Arm& a, unsigned T, int copy, bool fused) {
    const bf* gate = (const bf*)(in.gate_w.p + (size_t)copy * H);
    if (fused && (g_fused & 4)) {
        launch(moe_unpermute_blend_ep_vec8, dim3(T), 256, (const bf*)in.expert_out.p, (bf*)a.moe_out.p,
               (const int*)in.perm.p, (const int*)in.ids.p, (const float*)in.topk_w.p, (const bf*)in.shared.p,
               (const bf*)in.block_out.p, gate, T, TOPK, 0u, EXPERTS);
        return;
    }
    launch(moe_unpermute_reduce_indexed_ep_vec8, dim3(T), 256, (const bf*)in.expert_out.p, (bf*)a.moe_out.p,
           (const int*)in.perm.p, (const int*)in.ids.p, (const float*)in.topk_w.p, H, T, TOPK, 0u, EXPERTS);
    launch(moe_batched_blend, dim3(T), 256, (bf*)a.moe_out.p, (const bf*)in.shared.p, (const bf*)in.block_out.p,
           gate, H, T);
}

template <typename T>
static size_t diff(const Dev<T>& a, const Dev<T>& b, size_t count) {
    const std::vector<T> x = a.get(), y = b.get();
    return memcmp(x.data(), y.data(), count * sizeof(T)) == 0 ? 0 : 1 + std::mismatch(x.begin(), x.begin() + count, y.begin(),
        [](T p, T q) { return memcmp(&p, &q, sizeof(T)) == 0; }).first - x.begin();
}

static void route(const Inputs& in, Route& r, unsigned T, int copy, bool fused) {
    const bool f = fused && (g_fused & 8);
    const unsigned te = T * TOPK;
    launch(moe_topk_sigmoid_batched, dim3(T), 256,
           (const bf*)(in.logits.p + (size_t)(copy % 4) * MAXT * EXPERTS), (const float*)in.bias.p, r.ids.p,
           r.weights.p, EXPERTS, TOPK, 1u, 2.5f);
    launch(f ? moe_sort_by_expert_scan : moe_sort_by_expert, dim3(1), 256, (const unsigned*)r.ids.p, r.sorted_tok.p,
           r.sorted_exp.p, r.offsets.p, r.perm.p, te, EXPERTS, TOPK);
    // Serving's M16 grid (one N tile, M64) with every third expert remote.
    launch(f ? moe_build_tile_worklist_scan : moe_build_tile_worklist, dim3(1), 256, (const int*)r.offsets.p,
           (const unsigned long long*)in.expert_ptrs.p, r.worklist.p, r.total.p, EXPERTS, 1u + (copy % 2) * 3, 64u);
}

// RMS norm of T rows of width `h` (block min(h, 1024)) into the arm's normed buffer.
static void norm(const Inputs& in, Arm& a, unsigned T, unsigned h, int copy, bool fused) {
    const bool f = fused && (g_fused & 16);
    launch(f ? rms_norm_vanilla_regs : rms_norm_vanilla, dim3(T), std::min(h, 1024u),
           (const bf*)(in.quant_in.p + (size_t)(copy % 4) * MAXT * H), (const bf*)(in.norm_w.p + (size_t)(copy % 32) * H),
           (bf*)a.normed.p, h, NORM_EPS);
}

// Both sorts must route slot i of token i / topk to its expert through
// token_to_perm, with the same expert_offsets.
static bool route_same(const Route& a, const Route& b, unsigned T) {
    const unsigned te = T * TOPK;
    const auto ids = a.ids.get(), ids_b = b.ids.get();
    bool ok = memcmp(ids.data(), ids_b.data(), te * 4) == 0
        && diff(a.weights, b.weights, te) == 0 && diff(a.offsets, b.offsets, EXPERTS + 1) == 0
        && diff(a.total, b.total, 1) == 0;
    const int tiles = a.total.get()[0];
    ok = ok && tiles >= 0 && tiles <= (int)a.worklist.n / 2 && diff(a.worklist, b.worklist, 2 * (size_t)tiles) == 0;
    for (const Route* r : {&a, &b}) {
        const auto tok = r->sorted_tok.get(), exp = r->sorted_exp.get(), perm = r->perm.get(), off = r->offsets.get();
        for (unsigned i = 0; i < te; i++) {
            const int p = perm[i];
            ok = ok && p >= 0 && p < (int)te && tok[p] == (int)(i / TOPK) && exp[p] == (int)ids[i]
                && p >= off[ids[i]] && p < off[ids[i] + 1];
        }
    }
    return ok;
}

// Hold the stream busy while a batch of chains is queued behind it, so the
// GPU never waits for the host between launches.
extern "C" __global__ void bench_spin(unsigned int* sink, unsigned int rounds) {
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < rounds; i++) x = x * 1664525u + 1013904223u;
    if (threadIdx.x == 0) sink[0] = x;
}

// Median GPU time of one chain in microseconds over `groups` x 8 chains per
// rep, stream events around every 8. A group that another process's work
// lands in is an outlier the median ignores.
template <typename F>
static double time_us(int groups, int reps, unsigned int* sink, F&& chain) {
    const int per = 8;
    static std::vector<cudaEvent_t> ev;
    while ((int)ev.size() < 2 * groups) { cudaEvent_t event; CK(cudaEventCreate(&event)); ev.push_back(event); }
    std::vector<float> us;
    for (int r = 0; r < reps; r++) {
        CK(cudaDeviceSynchronize());
        bench_spin<<<1, 32>>>(sink, 4000000u);
        for (int g = 0; g < groups; g++) {
            CK(cudaEventRecord(ev[2 * g]));
            for (int i = 0; i < per; i++) chain((r * groups + g) * per + i);
            CK(cudaEventRecord(ev[2 * g + 1]));
        }
        CK(cudaDeviceSynchronize());
        for (int g = 0; g < groups; g++) {
            float ms;
            CK(cudaEventElapsedTime(&ms, ev[2 * g], ev[2 * g + 1]));
            us.push_back(ms * 1000.f / per);
        }
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

int main(int argc, char** argv) {
    const int copies = argc > 1 ? atoi(argv[1]) : 32;
    const int groups = argc > 2 ? atoi(argv[2]) : 64;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    if (argc > 4) g_fused = (unsigned)atoi(argv[4]);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    Inputs in;
    in.copies = copies;
    std::vector<unsigned short> h_streams((size_t)MAXT * HC * H), h_block((size_t)MAXT * H), h_peer((size_t)MAXT * H),
        h_norm((size_t)copies * H), h_expert((size_t)MAXT * TOPK * H), h_shared((size_t)MAXT * H), h_gate((size_t)copies * H);
    for (auto& x : h_streams) x = tobf(nd(rng) * 3.f);
    for (auto& x : h_block) x = tobf(nd(rng));
    for (auto& x : h_peer) x = tobf(nd(rng));
    for (auto& x : h_norm) x = tobf(1.f + 0.2f * nd(rng));
    for (auto& x : h_expert) x = tobf(nd(rng) * 0.5f);
    for (auto& x : h_shared) x = tobf(nd(rng) * 0.5f);
    for (auto& x : h_gate) x = tobf(nd(rng) * 0.02f);
    // Row 1: zeros and -0.0; row 2: large magnitudes.
    for (unsigned d = 0; d < H; d++) {
        h_block[1 * H + d] = d & 1 ? 0x8000 : 0;
        h_streams[(size_t)1 * HC * H + d] = d & 2 ? 0x8000 : 0;
        h_block[2 * H + d] = tobf(nd(rng) * 3.0e4f);
        h_expert[(size_t)3 * H + d] = d & 1 ? 0x8000 : 0;
    }
    std::vector<float> h_fn((size_t)copies * MIX * HC * H), h_scale(3), h_base(MIX), h_post(MAXT * HC), h_comb(MAXT * HC * HC),
        h_w(MAXT * TOPK);
    for (auto& x : h_fn) x = nd(rng) * 0.02f;
    for (auto& x : h_scale) x = 0.05f + 0.1f * ud(rng);
    for (auto& x : h_base) x = nd(rng);
    for (auto& x : h_post) x = 2.f * ud(rng);
    for (auto& x : h_comb) x = 0.5f * ud(rng);
    for (auto& x : h_w) x = ud(rng) * 0.3f;
    std::vector<int> h_perm(MAXT * TOPK), h_ids(MAXT * TOPK);
    for (unsigned i = 0; i < MAXT * TOPK; i++) h_perm[i] = (int)i;
    std::shuffle(h_perm.begin(), h_perm.end(), rng);
    for (unsigned t = 0; t < MAXT; t++) {
        std::vector<int> e(EXPERTS);
        for (unsigned i = 0; i < EXPERTS; i++) e[i] = (int)i;
        std::shuffle(e.begin(), e.end(), rng);
        for (unsigned k = 0; k < TOPK; k++) h_ids[t * TOPK + k] = e[k];
    }
    in.block_out.alloc(h_block.size()); in.peer.alloc(h_peer.size()); in.norm_w.alloc(h_norm.size());
    in.expert_out.alloc(h_expert.size()); in.shared.alloc(h_shared.size()); in.gate_w.alloc(h_gate.size());
    in.hc_fn.alloc(h_fn.size()); in.hc_scale.alloc(3); in.hc_base.alloc(MIX); in.topk_w.alloc(h_w.size());
    in.perm.alloc(h_perm.size()); in.ids.alloc(h_ids.size());
    in.peer.put(h_peer); in.norm_w.put(h_norm); in.expert_out.put(h_expert);
    in.shared.put(h_shared); in.gate_w.put(h_gate); in.hc_fn.put(h_fn); in.hc_scale.put(h_scale);
    in.hc_base.put(h_base); in.topk_w.put(h_w); in.perm.put(h_perm); in.ids.put(h_ids);
    // Router logits (4 sets): row 0 all ties, row 1 with NaNs, row 2 zeros.
    std::vector<unsigned short> h_logits((size_t)4 * MAXT * EXPERTS), h_qin((size_t)4 * MAXT * H);
    std::vector<float> h_bias(EXPERTS), zero_bias(EXPERTS, 0.f);
    for (auto& x : h_logits) x = tobf(nd(rng) * 2.f);
    for (auto& x : h_bias) x = nd(rng) * 0.05f;
    for (unsigned e = 0; e < EXPERTS; e++) {
        h_logits[e] = tobf(0.5f);
        if (e % 3 == 0) h_logits[EXPERTS + e] = 0x7FC0;
        h_logits[2 * EXPERTS + e] = e & 1 ? 0x8000 : 0;
    }
    // Quantizer input: rare large values, a row of +-0, a row with Inf.
    for (auto& x : h_qin) x = tobf(nd(rng) * (ud(rng) < 0.01f ? 1000.f : 1.f));
    for (unsigned d = 0; d < H; d++) {
        h_qin[d] = d & 1 ? 0x8000 : 0;
        if (d % 97 == 0) h_qin[H + d] = 0x7F80;
    }
    in.logits.alloc(h_logits.size()); in.logits.put(h_logits);
    in.quant_in.alloc(h_qin.size()); in.quant_in.put(h_qin);
    in.bias.alloc(EXPERTS); in.bias.put(h_bias);
    std::vector<unsigned long long> h_ptrs(EXPERTS);
    for (unsigned e = 0; e < EXPERTS; e++) h_ptrs[e] = e % 3 == 2 ? 0 : 0x10000ull * (e + 1);
    in.expert_ptrs.alloc(EXPERTS); in.expert_ptrs.put(h_ptrs);
    Route rt[2];
    for (auto& r : rt) r.alloc();

    Arm arm[2];
    for (auto& a : arm) a.alloc();
    const auto reset = [&](const std::vector<unsigned short>& streams, const std::vector<unsigned short>& block) {
        in.block_out.put(block);
        for (auto& a : arm) {
            a.streams.put(streams); a.post.put(h_post); a.comb.put(h_comb);
            CK(cudaMemset(a.hidden.p, 0xAB, a.hidden.n * 2)); CK(cudaMemset(a.normed.p, 0xAB, a.normed.n * 2));
            CK(cudaMemset(a.moe_out.p, 0xAB, a.moe_out.n * 2)); CK(cudaMemset(a.partial.p, 0xAB, a.partial.n * 4));
        }
    };

    bool ok = true;
    const unsigned widths[] = {1, 2, 3, 4, 5, 6, 7, 8, 32};
    // Poisoned inputs: an Inf and a NaN in the block output and the highway.
    std::vector<unsigned short> p_streams = h_streams, p_block = h_block;
    p_block[5] = 0x7F80; p_block[H + 9] = 0x7FC1; p_streams[3 * H + 17] = 0xFF80; p_streams[(size_t)HC * H + 40] = 0x7FC2;
    struct Case { const char* name; bool post, peer, seam; };
    const Case cases[] = {{"hc pre only", false, false, false}, {"hc post+pre", true, false, false},
                          {"hc post(+peer)+pre", true, true, false}, {"hc seam", true, false, true}};
    for (int poison = 0; poison < 2; poison++) {
        for (unsigned T : widths) {
            for (const Case& c : cases) {
                reset(poison ? p_streams : h_streams, poison ? p_block : h_block);
                // Two consecutive sites, so the second consumes the first's post/comb.
                for (int site = 0; site < 2; site++) {
                    hc_site(in, arm[0], T, site, c.post, c.peer, c.seam, false);
                    hc_site(in, arm[1], T, site, c.post, c.peer, c.seam, true);
                }
                CK(cudaDeviceSynchronize());
                const size_t bad = diff(arm[0].streams, arm[1].streams, (size_t)T * HC * H)
                    + diff(arm[0].partial, arm[1].partial, (size_t)SPLIT * T * 25)
                    + diff(arm[0].hidden, arm[1].hidden, (size_t)T * H) + diff(arm[0].normed, arm[1].normed, (size_t)T * H)
                    + diff(arm[0].post, arm[1].post, T * HC) + diff(arm[0].comb, arm[1].comb, T * HC * HC);
                if (bad) ok = false;
                printf("bitwise %-20s rows=%-2u %s: %s\n", c.name, T, poison ? "inf/nan" : "finite ", bad ? "FAIL" : "same");
                if (bad)
                    printf("  first mismatch (1-based; 0 = same): streams %zu hidden %zu normed %zu post %zu comb %zu\n",
                           diff(arm[0].streams, arm[1].streams, (size_t)T * HC * H), diff(arm[0].hidden, arm[1].hidden, (size_t)T * H),
                           diff(arm[0].normed, arm[1].normed, (size_t)T * H), diff(arm[0].post, arm[1].post, T * HC),
                           diff(arm[0].comb, arm[1].comb, T * HC * HC));
            }
            reset(poison ? p_streams : h_streams, poison ? p_block : h_block);
            moe_post(in, arm[0], T, 1, false);
            moe_post(in, arm[1], T, 1, true);
            CK(cudaDeviceSynchronize());
            const size_t bad = diff(arm[0].moe_out, arm[1].moe_out, (size_t)T * H);
            if (bad) ok = false;
            printf("bitwise %-20s rows=%-2u %s: %s\n", "moe unpermute+blend", T, poison ? "inf/nan" : "finite ", bad ? "FAIL" : "same");
        }
    }
    for (unsigned T : widths) {
        for (int set = 0; set < 4; set++) {
            // Set 0 has the bias, which breaks the all-ties row; set 1 runs it without.
            in.bias.put(set == 1 ? zero_bias : h_bias);
            route(in, rt[0], T, set, false);
            route(in, rt[1], T, set, true);
            CK(cudaDeviceSynchronize());
            const bool same = route_same(rt[0], rt[1], T);
            ok = ok && same;
            printf("bitwise %-20s rows=%-2u set %d : %s\n", "moe sort+worklist", T, set, same ? "same" : "FAIL");
        }
    }
    in.bias.put(h_bias);
    for (unsigned T : widths) {
        for (unsigned h : {4096u, 1536u, 512u, 4095u}) {
            if (h % 2 && T > 1) continue;  // odd rows are 4-byte loads off alignment in both kernels
            for (auto& a : arm) CK(cudaMemset(a.normed.p, 0xAB, a.normed.n * 2));
            for (int set = 0; set < 4; set++) {
                norm(in, arm[0], T, h, set, false);
                norm(in, arm[1], T, h, set, true);
                CK(cudaDeviceSynchronize());
                const bool same = diff(arm[0].normed, arm[1].normed, arm[0].normed.n) == 0;
                ok = ok && same;
                if (!same || set == 3)
                    printf("bitwise %-20s rows=%-2u h=%-4u set %d: %s\n", "rms norm", T, h, set, same ? "same" : "FAIL");
            }
        }
    }
    printf("%s\n", ok ? "PASS: every fused chain matches its unfused chain bit for bit" : "FAIL");
    if (!ok) return 1;

    Dev<unsigned int> sink;
    sink.alloc(1);
    printf("\nus per chain (median of %d x %d groups of 8, %d weight copies, fused groups %u)\n", reps, groups, copies, g_fused);
    printf("%-22s %4s %9s %9s %8s\n", "chain", "rows", "unfused", "fused", "saved");
    for (unsigned T : {3u, 4u, 5u, 8u}) {
        for (const Case& c : cases) {
            reset(h_streams, h_block);
            const double a = time_us(groups, reps, sink.p, [&](int i) { hc_site(in, arm[0], T, i % copies, c.post, c.peer, c.seam, false); });
            const double b = time_us(groups, reps, sink.p, [&](int i) { hc_site(in, arm[1], T, i % copies, c.post, c.peer, c.seam, true); });
            printf("%-22s %4u %9.2f %9.2f %8.2f\n", c.name, T, a, b, a - b);
        }
        const double a = time_us(groups, reps, sink.p, [&](int i) { moe_post(in, arm[0], T, i % copies, false); });
        const double b = time_us(groups, reps, sink.p, [&](int i) { moe_post(in, arm[1], T, i % copies, true); });
        printf("%-22s %4u %9.2f %9.2f %8.2f\n", "moe unpermute+blend", T, a, b, a - b);
        const double c = time_us(groups, reps, sink.p, [&](int i) { route(in, rt[0], T, i, false); });
        const double d = time_us(groups, reps, sink.p, [&](int i) { route(in, rt[1], T, i, true); });
        printf("%-22s %4u %9.2f %9.2f %8.2f\n", "moe sort+worklist", T, c, d, c - d);
        const double e = time_us(groups, reps, sink.p, [&](int i) { norm(in, arm[0], T, H, i, false); });
        const double f = time_us(groups, reps, sink.p, [&](int i) { norm(in, arm[1], T, H, i, true); });
        printf("%-22s %4u %9.2f %9.2f %8.2f\n", "rms norm", T, e, f, e - f);
    }
    return 0;
}
