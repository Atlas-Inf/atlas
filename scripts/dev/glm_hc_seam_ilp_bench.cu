// SPDX-License-Identifier: AGPL-3.0-only
// Standalone bitwise check and A/B timing of the decode seam's load-batched
// twins (ATLAS_GLM_HC_SEAM_ILP) against the kernels serving runs today, on the
// GLM-5.3-Flash shapes (hidden 4096, hc_mult 4, 24 mix rows, 64 partial slices)
// at every decode width 1..32:
//
//   base  glm_hc_decode_{post_,}partial_rows_bf16 (FP32 hc_fn) -> glm_hc_decode_finalize_bf16
//   orig  glm_hc_decode_{post_,}partial_bf16      (FP32 hc_fn) -> glm_hc_decode_finalize_bf16
//   ilp   glm_hc_decode_{post_,}partial_ilp_bf16  (BF16 hc_fn) -> glm_hc_decode_finalize_ilp_bf16
//
// The FP32 hc_fn is the BF16 one widened on the host exactly as load_hc_f32
// does. Two consecutive sites run per case (the second's post reads the first's
// post/comb), seam (post + pre) and pre-only, with finite inputs and with Inf /
// NaN / -0.0 / subnormals in the highway, block output and weights. Every byte
// of the highway, the partial sums, the collapsed row, post and comb must
// match base and orig.
//
// Timing (as serving launches them, PDL on every kernel): per width, the chain
// pred -> partial -> finalize -> next, where pred is the seam's real
// predecessor (a 48-CTA BF16 add for the seam, glm_hc_decode_post_bf16 for the
// pre-only site) and next is a stand-in for the projection that follows:
// 1024 CTAs reading an 8.5 MiB weight, whose first 32 CTAs touch it into L2
// before their PDL wait as the *_touch GEMVs do (`next=0` drops it). It cycles
// `copies` weight sets so nothing stays in the 24 MiB L2. Arms: base, ilp
// partial only, ilp finalize only, both.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_hc_seam_ilp_bench.cu -o glm_hc_seam_ilp_bench
//   ./glm_hc_seam_ilp_bench [copies=32] [groups=64] [reps=3] [next=1]
//
// Prints one "bitwise" line per case and width, then
//   PASS: ILP seam matches base and orig bit for bit at rows 1..32
// (or FAIL, exit 1), then the median GPU time per chain in microseconds (stream
// events around every 8 chains, each batch queued behind a spin kernel).
// Device memory: about 120 MB.
//
// The finalizer's split into helpers (glm_hc_vec_gates / _sinkhorn / _collapse4)
// must leave every pre-existing kernel's code unchanged. Check the PTX of each
// against the parent commit (expect no DIFF line):
//   d=kernels/gb10/glm-5.3-flash/nvfp4; git show daa94067:$d/glm_hc_prefill_vec.cu > $d/hc_old.cu
//   for f in hc_old glm_hc_prefill_vec; do
//     nvcc --ptx -arch=sm_121a -O3 --fmad=false -std=c++17 $d/$f.cu -o /tmp/$f.ptx; done; rm $d/hc_old.cu
//   for k in $(grep -o 'entry [a-z0-9_]*' /tmp/hc_old.ptx | cut -d' ' -f2); do
//     cmp -s <(sed -n "/entry $k(/,/^}/p" /tmp/hc_old.ptx) \
//            <(sed -n "/entry $k(/,/^}/p" /tmp/glm_hc_prefill_vec.ptx) || echo "DIFF $k"; done
#include "glm_hc_prefill_vec.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned H = 4096, HC = 4, MIX = 24, SPLIT = 64, MAXT = 32, SINK = 20;
static const float NORM_EPS = 1e-5f, HC_EPS = 1e-6f;
static const size_t FN = (size_t)MIX * HC * H;          // mix weight elements
static const size_t NEXT_BYTES = (size_t)17 << 19;      // 8.5 MiB, a KDA projection
static const unsigned NEXT_SETS = 4;

template <typename... Args>
static void launch(void (*kernel)(Args...), dim3 grid, unsigned block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at;
    cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

// The seam's predecessor (bf16_add_inplace's shape: 48 CTAs over T rows).
extern "C" __global__ void bench_add(bf* __restrict__ x, const bf* __restrict__ y, unsigned n) {
    atlas_pdl_enter();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __hadd(x[i], y[i]);
}

// Stand-in for the projection after the seam: touch, wait, then stream the weight.
extern "C" __global__ void bench_next(const unsigned char* __restrict__ w, const unsigned char* __restrict__ ws,
                                      unsigned rows, unsigned* sink) {
    atlas_pdl_enter_touch({w, 4096u, 4096ull}, {ws, 0u, 0ull}, rows, blockIdx.x, 32u);
    const uint4* p = (const uint4*)w;
    const size_t n = (size_t)rows * 4096 / 16;
    unsigned acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

extern "C" __global__ void bench_spin(unsigned int* sink, unsigned int rounds) {
    unsigned int x = threadIdx.x;
    for (unsigned int i = 0; i < rounds; i++) x = x * 1664525u + 1013904223u;
    if (threadIdx.x == 0) sink[0] = x;
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

// Number of the first differing element (1-based), 0 when the first `count` match.
template <typename T>
static size_t diff(const Dev<T>& a, const Dev<T>& b, size_t count) {
    const std::vector<T> x = a.get(), y = b.get();
    for (size_t i = 0; i < count; i++)
        if (memcmp(&x[i], &y[i], sizeof(T)) != 0) return i + 1;
    return 0;
}

struct Arm {
    Dev<unsigned short> streams, hidden;
    Dev<float> partial, post, comb;
    void alloc() {
        streams.alloc((size_t)MAXT * HC * H); hidden.alloc((size_t)MAXT * H);
        partial.alloc((size_t)SPLIT * MAXT * 25); post.alloc(MAXT * HC); comb.alloc(MAXT * HC * HC);
    }
};

struct Inputs {
    Dev<unsigned short> block_out, fn16, add_in;
    Dev<float> fn32, hc_scale, hc_base;
    Dev<unsigned char> next_w;
    Dev<unsigned> sink;
    int copies;
};

enum Kind { BASE, ORIG, ILP };

static void partial(const Inputs& in, Arm& a, unsigned T, int copy, bool post, Kind k) {
    const dim3 grid(SPLIT, (T + 3) / 4);
    const float* f32 = in.fn32.p + (size_t)copy * FN;
    const unsigned char* f16 = (const unsigned char*)(in.fn16.p + (size_t)copy * FN);
    const bf* bo = post ? (const bf*)in.block_out.p : (const bf*)nullptr;
    bf* st = (bf*)a.streams.p;
    const float *p = a.post.p, *c = a.comb.p;
    if (k == ILP)
        launch(post ? glm_hc_decode_post_partial_ilp_bf16 : glm_hc_decode_partial_ilp_bf16, grid, 128, bo, st, p, c, f16,
               a.partial.p, T);
    else if (k == BASE)
        launch(post ? glm_hc_decode_post_partial_rows_bf16 : glm_hc_decode_partial_rows_bf16, grid, 128, bo, st, p, c,
               f32, a.partial.p, T);
    else
        launch(post ? glm_hc_decode_post_partial_bf16 : glm_hc_decode_partial_bf16, grid, 128, bo, st, p, c, f32,
               a.partial.p, T);
}

static void finalize(const Inputs& in, Arm& a, unsigned T, bool ilp) {
    launch(ilp ? glm_hc_decode_finalize_ilp_bf16 : glm_hc_decode_finalize_bf16, dim3(T), 256, (const bf*)a.streams.p,
           (const float*)a.partial.p, (const float*)in.hc_scale.p, (const float*)in.hc_base.p, (bf*)a.hidden.p,
           a.post.p, a.comb.p, T, SINK, NORM_EPS, HC_EPS);
}

// One timed site as serving runs it: predecessor, partial, finalize, next projection.
static void chain(const Inputs& in, Arm& a, unsigned T, int i, bool post, bool ilp_partial, bool ilp_finalize,
                  bool next) {
    const int copy = i % in.copies;
    if (post)
        launch(bench_add, dim3(48), 256, (bf*)in.add_in.p, (const bf*)in.block_out.p, T * H);
    else
        launch(glm_hc_decode_post_bf16, dim3(2 * T), 256, (const bf*)in.block_out.p, (const bf*)nullptr,
               (bf*)a.streams.p, (const float*)a.post.p, (const float*)a.comb.p, T);
    partial(in, a, T, copy, post, ilp_partial ? ILP : BASE);
    finalize(in, a, T, ilp_finalize);
    if (next) {
        const unsigned char* w = in.next_w.p + (size_t)(i % NEXT_SETS) * NEXT_BYTES;
        launch(bench_next, dim3(1024), 256, w, w, (unsigned)(NEXT_BYTES / 4096), in.sink.p);
    }
}

template <typename F>
static double time_us(int groups, int reps, unsigned int* sink, F&& one) {
    const int per = 8;
    static std::vector<cudaEvent_t> ev;
    while ((int)ev.size() < 2 * groups) { cudaEvent_t evt; CK(cudaEventCreate(&evt)); ev.push_back(evt); }
    std::vector<float> us;
    for (int r = 0; r < reps; r++) {
        CK(cudaDeviceSynchronize());
        bench_spin<<<1, 32>>>(sink, 4000000u);
        for (int g = 0; g < groups; g++) {
            CK(cudaEventRecord(ev[2 * g]));
            for (int i = 0; i < per; i++) one((r * groups + g) * per + i);
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
    const int copies = argc > 1 ? std::max(2, atoi(argv[1])) : 32;
    const int groups = argc > 2 ? atoi(argv[2]) : 64;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    const bool with_next = argc > 4 ? atoi(argv[4]) != 0 : true;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    Inputs in;
    in.copies = copies;
    std::vector<unsigned short> h_streams((size_t)MAXT * HC * H), h_block((size_t)MAXT * H), h_fn16((size_t)copies * FN);
    for (auto& x : h_streams) x = tobf(nd(rng) * 3.f);
    for (auto& x : h_block) x = tobf(nd(rng));
    for (auto& x : h_fn16) x = tobf(nd(rng) * 0.02f);
    // Row 1 zeros and -0.0, row 2 large magnitudes; weight set 1 with +-0 and subnormals.
    for (unsigned d = 0; d < H; d++) {
        h_block[1 * H + d] = d & 1 ? 0x8000 : 0;
        h_streams[(size_t)1 * HC * H + d] = d & 2 ? 0x8000 : 0;
        h_block[2 * H + d] = tobf(nd(rng) * 3.0e4f);
    }
    for (size_t i = 0; i < FN; i += 7) h_fn16[FN + i] = i % 3 == 0 ? 0x8000 : (i % 3 == 1 ? 0x0001 : 0x8003);
    std::vector<float> h_scale(3), h_base(MIX), h_post(MAXT * HC), h_comb(MAXT * HC * HC);
    for (auto& x : h_scale) x = 0.05f + 0.1f * ud(rng);
    for (auto& x : h_base) x = nd(rng);
    for (auto& x : h_post) x = 2.f * ud(rng);
    for (auto& x : h_comb) x = 0.5f * ud(rng);
    // load_hc_f32's widening: the BF16 bits become the top half of the FP32 word.
    const auto widen = [](const std::vector<unsigned short>& b) {
        std::vector<float> f(b.size());
        for (size_t i = 0; i < b.size(); i++) {
            const unsigned u = (unsigned)b[i] << 16;
            memcpy(&f[i], &u, 4);
        }
        return f;
    };
    in.block_out.alloc(h_block.size()); in.fn16.alloc(h_fn16.size()); in.fn32.alloc(h_fn16.size());
    in.hc_scale.alloc(3); in.hc_base.alloc(MIX); in.add_in.alloc((size_t)MAXT * H);
    in.next_w.alloc(NEXT_SETS * NEXT_BYTES); in.sink.alloc(1);
    in.hc_scale.put(h_scale); in.hc_base.put(h_base);
    CK(cudaMemset(in.add_in.p, 0, in.add_in.n * 2));
    CK(cudaMemset(in.next_w.p, 0x5A, in.next_w.n));
    const auto put_fn = [&](const std::vector<unsigned short>& f16) { in.fn16.put(f16); in.fn32.put(widen(f16)); };

    Arm arm[3];
    for (auto& a : arm) a.alloc();
    const auto reset = [&](const std::vector<unsigned short>& streams, const std::vector<unsigned short>& block) {
        in.block_out.put(block);
        for (auto& a : arm) {
            a.streams.put(streams); a.post.put(h_post); a.comb.put(h_comb);
            CK(cudaMemset(a.hidden.p, 0xAB, a.hidden.n * 2)); CK(cudaMemset(a.partial.p, 0xAB, a.partial.n * 4));
        }
    };

    bool ok = true;
    std::vector<unsigned short> p_streams = h_streams, p_block = h_block, p_fn16 = h_fn16;
    p_block[5] = 0x7F80; p_block[H + 9] = 0x7FC1; p_streams[3 * H + 17] = 0xFF80; p_streams[(size_t)HC * H + 40] = 0x7FC2;
    p_streams[(size_t)5 * HC * H + 2 * H + 4000] = 0x0001;
    p_fn16[FN + 3 * 16384 + 77] = 0x7F80; p_fn16[FN + 11 * 16384 + 9000] = 0x7FC0;
    for (int poison = 0; poison < 2; poison++) {
        put_fn(poison ? p_fn16 : h_fn16);
        for (unsigned T = 1; T <= MAXT; T++) {
            for (int post = 0; post < 2; post++) {
                reset(poison ? p_streams : h_streams, poison ? p_block : h_block);
                for (int site = 0; site < 2; site++) {
                    partial(in, arm[0], T, site, post, BASE); finalize(in, arm[0], T, false);
                    partial(in, arm[1], T, site, post, ORIG); finalize(in, arm[1], T, false);
                    partial(in, arm[2], T, site, post, ILP); finalize(in, arm[2], T, true);
                }
                CK(cudaDeviceSynchronize());
                for (int ref = 0; ref < 2; ref++) {
                    const Arm &a = arm[ref], &b = arm[2];
                    const size_t s = diff(a.streams, b.streams, (size_t)T * HC * H);
                    // Partial sums: the first T * 25 of each of the 64 slices.
                    size_t pa = 0;
                    const auto x = a.partial.get(), y = b.partial.get();
                    for (unsigned c = 0; c < SPLIT && !pa; c++)
                        if (memcmp(&x[(size_t)c * T * 25], &y[(size_t)c * T * 25], (size_t)T * 25 * 4)) pa = c + 1;
                    const size_t hd = diff(a.hidden, b.hidden, (size_t)T * H), po = diff(a.post, b.post, T * HC),
                                 co = diff(a.comb, b.comb, T * HC * HC);
                    const bool same = !(s || pa || hd || po || co);
                    ok = ok && same;
                    printf("bitwise %-4s vs ilp %-9s rows=%-2u %s: %s\n", ref ? "orig" : "base", post ? "seam" : "pre-only",
                           T, poison ? "inf/nan" : "finite ", same ? "same" : "FAIL");
                    if (!same)
                        printf("  first mismatch (1-based; 0 = same): streams %zu partial slice %zu hidden %zu post %zu comb %zu\n",
                               s, pa, hd, po, co);
                }
            }
        }
    }
    printf("%s\n", ok ? "PASS: ILP seam matches base and orig bit for bit at rows 1..32" : "FAIL");
    if (!ok) return 1;

    put_fn(h_fn16);
    reset(h_streams, h_block);
    printf("\nus per chain (median of %d x %d groups of 8, %d weight copies, next projection %s)\n", reps, groups, copies,
           with_next ? "on" : "off");
    printf("%-9s %4s %9s %9s %9s %9s %8s\n", "site", "rows", "base", "partial", "finalize", "both", "saved");
    for (int post = 1; post >= 0; post--) {
        for (unsigned T : {1u, 2u, 3u, 4u, 5u, 8u, 16u, 32u}) {
            double t[4];
            for (int v = 0; v < 4; v++)
                t[v] = time_us(groups, reps, in.sink.p, [&](int i) { chain(in, arm[0], T, i, post, v & 1, v & 2, with_next); });
            printf("%-9s %4u %9.2f %9.2f %9.2f %9.2f %8.2f\n", post ? "seam" : "pre-only", T, t[0], t[1], t[2], t[3],
                   t[0] - t[3]);
        }
    }
    return 0;
}
