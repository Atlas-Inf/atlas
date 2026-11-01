// SPDX-License-Identifier: AGPL-3.0-only
// Standalone GLM-5.3-Flash routed-MoE prefill microbenchmark (GB10, EP2 rank 0).
//
// One prefill chunk of T tokens, top-8 of 288 experts, the local 144 experts
// holding random NVFP4 weights, launched exactly as serving launches them:
//   moe_mtile_prefix -> gate_up_silu_k128w -> k128w_compact (down)
//   -> moe_unpermute_reduce_indexed_ep
// after roofline probes (mxf4nvf4 m16n8k64 MMA peak, DRAM stream bandwidth).
// Then the ATLAS_GLM_MOE_PREFILL_PERSIST kernels and the
// ATLAS_GLM_MOE_UNPERMUTE_VEC kernel run on the same inputs: their outputs
// must match the production bytes (exit 1 otherwise), and each is timed in
// interleaved pairs against its production twin (minimum times, since the
// GPU may be shared; the median paired ratio).
//
// Build + run on a GB10 host, from this directory (-arch=sm_121a alone does
// not enable the block-scaled MMA):
//   nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//        -o /tmp/moe_prefill_bench moe_prefill_bench.cu
//   /tmp/moe_prefill_bench [tokens=8192] [skew=0.0] [iters=5]
// Adding -DMOE_BENCH_BASE builds the production half alone, for comparing
// the production hashes of another tree's kernels.
// Device memory: about 2.6 GB at 8192 tokens (1 GB more for the probes).

#include "../../kernels/gb10/glm-5.3-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "../../kernels/gb10/glm-5.3-flash/nvfp4/moe_permute.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static const unsigned E = 288, LOCAL = 144, TOPK = 8, H = 4096, I = 2048;

struct Lcg {
    unsigned long long s;
    unsigned next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(s >> 33); }
    float unit() { return (next() & 0xFFFFFF) / 16777216.0f; }
};

// ── Roofline probes ──────────────────────────────────────────────────────
__global__ void __launch_bounds__(256) mma_peak(float* out, int iters) {
    unsigned a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x5a5a5a5au, a2 = a0 + 7, a3 = a0 * 3;
    unsigned b0 = a0 ^ 0x33333333u, b1 = a1 + 11, sf = 0x38383838u;
    float acc[8][4] = {};
    const unsigned short z = 0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};"
                : "=f"(acc[j][0]), "=f"(acc[j][1]), "=f"(acc[j][2]), "=f"(acc[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[j][0]), "f"(acc[j][1]), "f"(acc[j][2]), "f"(acc[j][3]),
                  "r"(sf), "h"(z), "h"(z), "r"(sf), "h"(z), "h"(z));
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    if (s == 12345.f) out[threadIdx.x] = s;
}

__global__ void stream_read(const uint4* __restrict__ src, size_t n, unsigned* out) {
    unsigned x = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x12345678u) out[0] = x;
}

__global__ void stream_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// ── Helpers ──────────────────────────────────────────────────────────────
template <class T> static T* dev(size_t n) { T* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> static T* up(const std::vector<T>& h) { T* p = dev<T>(h.size()); CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static unsigned long long fnv(const std::vector<T>& v) {
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)v.data();
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
template <class T> static std::vector<T> down(const T* p, size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }

static void fill_bytes(std::vector<unsigned char>& v, Lcg& r, bool scale) {
    for (auto& b : v) b = scale ? (unsigned char)(0x28 + r.next() % 24) : (unsigned char)r.next();
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    template <class F> float ms(F f, int iters) {
        f();  // warm
        CK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (int i = 0; i < iters; ++i) {
            CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float x; CK(cudaEventElapsedTime(&x, a, b)); t.push_back(x);
        }
        std::sort(t.begin(), t.end());
        return t[0];   // minimum: the GPU host is shared, contention only adds time
    }
    // Interleaved A/B: minima of each and the median per-pair ratio b/a.
    template <class A, class B> void ab(A fa, B fb, int iters, float* ma, float* mb, float* ratio) {
        std::vector<float> ta, tb, r;
        for (int i = 0; i < iters; ++i) {
            ta.push_back(ms(fa, 1));
            tb.push_back(ms(fb, 1));
            r.push_back(tb.back() / ta.back());
        }
        std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end()); std::sort(r.begin(), r.end());
        *ma = ta[0]; *mb = tb[0]; *ratio = r[iters / 2];
    }
};

struct ExpertTable {  // [K/2, N] packed + [K/16, N] scales + scale2, remote half NULL
    std::vector<unsigned char*> w, s;
    unsigned long long *pp, *sp;
    float* s2;
};

static ExpertTable make_table(Lcg& r, unsigned n, unsigned k, float s2base) {
    ExpertTable t;
    std::vector<unsigned long long> pp(E, 0), sp(E, 0);
    std::vector<float> s2(E, 0.f);
    std::vector<unsigned char> w((size_t)k / 2 * n), s((size_t)k / 16 * n);
    for (unsigned e = 0; e < LOCAL; ++e) {
        fill_bytes(w, r, false);
        fill_bytes(s, r, true);
        t.w.push_back(up(w));
        t.s.push_back(up(s));
        pp[e] = (unsigned long long)t.w.back();
        sp[e] = (unsigned long long)t.s.back();
        s2[e] = s2base * (1.0f + (e % 7) * 0.125f);
    }
    t.pp = up(pp); t.sp = up(sp); t.s2 = up(s2);
    return t;
}

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? atoi(argv[1]) : 8192;
    const double skew = argc > 2 ? atof(argv[2]) : 0.0;
    const int iters = argc > 3 ? atoi(argv[3]) : 5;
    const unsigned TE = T * TOPK;
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    Timer tm;

    // Roofline probes.
    {
        float* o = dev<float>(256);
        const int it = 4096;
        const float ms = tm.ms([&] { mma_peak<<<sms * 4, 256>>>(o, it); }, iters);
        const double flops = 2.0 * 16 * 8 * 64 * 8.0 * it * (sms * 4) * 8;
        printf("probe mma m16n8k64 mxf4nvf4: %.1f TFLOPS (%d SMs)\n", flops / ms / 1e9, sms);
        const size_t bytes = (size_t)1 << 30;
        uint4* a = dev<uint4>(bytes / 16);
        uint4* b = dev<uint4>(bytes / 16);
        CK(cudaMemset(a, 1, bytes));
        unsigned* u = dev<unsigned>(1);
        const float rd = tm.ms([&] { stream_read<<<sms * 8, 512>>>(a, bytes / 16, u); }, iters);
        const float cp = tm.ms([&] { stream_copy<<<sms * 8, 512>>>(a, b, bytes / 16); }, iters);
        printf("probe DRAM read %.1f GB/s, copy (r+w) %.1f GB/s\n", bytes / rd / 1e6, 2.0 * bytes / cp / 1e6);
        CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(o)); CK(cudaFree(u));
    }

    // Routing: T tokens x 8 distinct experts, popularity ~ 1/(1+rank)^skew
    // over a fixed random permutation (skew 0 = uniform).
    Lcg r{0x5eed};
    std::vector<unsigned> perm(E);
    for (unsigned e = 0; e < E; ++e) perm[e] = e;
    for (unsigned e = E - 1; e > 0; --e) std::swap(perm[e], perm[r.next() % (e + 1)]);
    std::vector<double> cdf(E);
    double acc = 0;
    for (unsigned e = 0; e < E; ++e) { acc += 1.0 / std::pow(1.0 + e, skew); cdf[e] = acc; }
    std::vector<int> ids(TE);
    std::vector<float> wts(TE);
    for (unsigned t = 0; t < T; ++t) {
        for (unsigned k = 0; k < TOPK; ++k) {
            int e;
            bool dup;
            do {
                const double u = r.unit() * acc;
                e = perm[std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()];
                dup = false;
                for (unsigned j = 0; j < k; ++j) dup |= ids[t * TOPK + j] == e;
            } while (dup);
            ids[t * TOPK + k] = e;
            wts[t * TOPK + k] = 0.05f + r.unit() * 0.3f;
        }
    }
    // Counting sort as moe_sort_by_expert (token order within an expert).
    std::vector<int> offs(E + 1, 0), sorted(TE), t2p(TE);
    for (int e : ids) offs[e + 1]++;
    for (unsigned e = 0; e < E; ++e) offs[e + 1] += offs[e];
    {
        std::vector<int> fill(offs.begin(), offs.end() - 1);
        for (unsigned i = 0; i < TE; ++i) { const int p = fill[ids[i]]++; sorted[p] = i / TOPK; t2p[i] = p; }
    }
    const unsigned local_rows = offs[LOCAL];
    unsigned tiles = 0, maxrows = 0;
    for (unsigned e = 0; e < LOCAL; ++e) {
        const unsigned m = offs[e + 1] - offs[e];
        tiles += (m + 63) / 64;
        maxrows = std::max(maxrows, m);
    }
    const unsigned bound = (TE + 63) / 64 + E;
    printf("routing T=%u skew=%.2f: local rows %u, M64 tiles %u (pad %.1f%%), max rows/expert %u, grid bound %u\n",
           T, skew, local_rows, tiles, 100.0 * (tiles * 64.0 - local_rows) / local_rows, maxrows, bound);

    // Activations (token-major NVFP4, as quantize_bf16_to_nvfp4 leaves them).
    std::vector<unsigned char> a((size_t)T * H / 2), as((size_t)T * H / 16);
    fill_bytes(a, r, false); fill_bytes(as, r, true);
    unsigned char *d_a = up(a), *d_as = up(as);
    ExpertTable gate = make_table(r, I, H, 1.0f / 256), upt = make_table(r, I, H, 1.0f / 128),
                dn = make_table(r, H, I, 1.0f / 64);
    int *d_off = up(offs), *d_sorted = up(sorted), *d_t2p = up(t2p), *d_ids = up(ids);
    float* d_w = up(wts);
    int* d_prefix = dev<int>(E + 1);
    unsigned char* d_q = dev<unsigned char>((size_t)TE * I / 2 + (size_t)TE * I / 16);
    unsigned char* d_qs = d_q + (size_t)TE * I / 2;
    __nv_bfloat16* d_c = dev<__nv_bfloat16>((size_t)TE * H);
    __nv_bfloat16* d_out = dev<__nv_bfloat16>((size_t)T * H);
    __nv_bfloat16* d_out2 = dev<__nv_bfloat16>((size_t)T * H);

    auto prefix = [&] { moe_mtile_prefix<<<1, 1024>>>(d_off, gate.pp, d_prefix, E); };
    auto gate_up = [&] {
        moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w<<<dim3(I / 128, bound), 256>>>(
            d_a, d_as, gate.pp, gate.sp, gate.s2, nullptr, d_off, d_sorted, E, I, H, d_prefix,
            upt.pp, upt.sp, upt.s2, d_q, d_qs);
    };
    auto down_proj = [&] {
        moe_w4a4_grouped_gemm_prequant_t_k128w_compact<<<dim3(H / 256, bound), 256>>>(
            d_q, d_qs, dn.pp, dn.sp, dn.s2, d_c, d_off, nullptr, E, H, I, d_prefix);
    };
    auto unperm = [&](__nv_bfloat16* o) {
        moe_unpermute_reduce_indexed_ep<<<T, 256>>>(d_c, o, d_t2p, d_ids, d_w, H, T, TOPK, 0, LOCAL);
    };
    // Production pipeline: output hashes (compare a base-tree build with this
    // one) and roofline-relative timings.
    const size_t qb = (size_t)local_rows * I / 2, sb = (size_t)local_rows * I / 16, cb = (size_t)local_rows * H;
    CK(cudaMemset(d_q, 0, (size_t)TE * I / 2 + (size_t)TE * I / 16));
    CK(cudaMemset(d_c, 0, (size_t)TE * H * 2));
    prefix(); gate_up(); down_proj(); unperm(d_out);
    CK(cudaDeviceSynchronize());
    const auto ref_q = down(d_q, qb), ref_s = down(d_qs, sb);
    const auto ref_c = down(d_c, cb), ref_o = down(d_out, (size_t)T * H);
    printf("production hashes: q %016llx s %016llx c %016llx out %016llx\n", fnv(ref_q), fnv(ref_s),
           fnv(ref_c), fnv(ref_o));

    const double wbytes_gu = LOCAL * 2.0 * ((double)H / 2 * I + (double)H / 16 * I);
    const double wbytes_dn = LOCAL * ((double)I / 2 * H + (double)I / 16 * H);
    const double f_gu = 2.0 * local_rows * 2 * I * H, f_dn = 2.0 * local_rows * H * I;
    const float t_gu = tm.ms(gate_up, iters), t_dn = tm.ms(down_proj, iters);
    const float t_un = tm.ms([&] { unperm(d_out); }, iters);
    const double un_bytes = (double)local_rows * H * 2 + (double)T * H * 2;
    printf("gate_up_silu_k128w : %7.3f ms  %6.1f TFLOPS  weights %.2f GB -> %6.1f GB/s\n",
           t_gu, f_gu / t_gu / 1e9, wbytes_gu / 1e9, wbytes_gu / t_gu / 1e6);
    printf("down k128w_compact : %7.3f ms  %6.1f TFLOPS  weights+C %.2f GB -> %6.1f GB/s\n",
           t_dn, f_dn / t_dn / 1e9, (wbytes_dn + local_rows * H * 2.0) / 1e9,
           (wbytes_dn + local_rows * H * 2.0) / t_dn / 1e6);
    printf("unpermute_reduce_ep: %7.3f ms  %.3f GB -> %6.1f GB/s\n", t_un, un_bytes / 1e9, un_bytes / t_un / 1e6);
    printf("moe chunk total    : %7.3f ms\n", t_gu + t_dn + t_un);

#ifndef MOE_BENCH_BASE
    // Candidates (ATLAS_GLM_MOE_PREFILL_PERSIST, ATLAS_GLM_MOE_UNPERMUTE_VEC),
    // gated byte for byte against the production outputs above.
    const unsigned ctas = 2 * sms;
    int* d_ctr = dev<int>(2);
    auto gate_up_p = [&] {
        CK(cudaMemsetAsync(d_ctr, 0, 8));
        moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w_persist<<<ctas, 256>>>(
            d_a, d_as, gate.pp, gate.sp, gate.s2, nullptr, d_off, d_sorted, E, I, H, d_prefix,
            upt.pp, upt.sp, upt.s2, d_q, d_qs, d_ctr);
    };
    auto down_p = [&] {
        moe_w4a4_grouped_gemm_prequant_t_k128w_compact_persist<<<ctas, 256>>>(
            d_q, d_qs, dn.pp, dn.sp, dn.s2, d_c, d_off, nullptr, E, H, I, d_prefix, d_ctr + 1);
    };
    auto unperm_v = [&](__nv_bfloat16* o) {
        moe_unpermute_reduce_indexed_ep_vec8<<<T, 256>>>(d_c, o, d_t2p, d_ids, d_w, H, T, TOPK, 0, LOCAL);
    };
    CK(cudaMemset(d_q, 0, (size_t)TE * I / 2 + (size_t)TE * I / 16));
    CK(cudaMemset(d_c, 0, (size_t)TE * H * 2));
    CK(cudaMemset(d_out2, 0, (size_t)T * H * 2));
    gate_up_p(); down_p(); unperm_v(d_out2);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const auto q = down(d_q, qb), sc = down(d_qs, sb);
    const auto c = down(d_c, cb), o = down(d_out2, (size_t)T * H);
    size_t bad_q = 0, bad_c = 0, bad_o = 0;
    float diff_c = 0.f, diff_o = 0.f;
    for (size_t i = 0; i < qb; ++i) bad_q += q[i] != ref_q[i];
    for (size_t i = 0; i < sb; ++i) bad_q += sc[i] != ref_s[i];
    for (size_t i = 0; i < cb; ++i) {
        bad_c += memcmp(&c[i], &ref_c[i], 2) != 0;
        diff_c = std::max(diff_c, std::fabs(__bfloat162float(c[i]) - __bfloat162float(ref_c[i])));
    }
    for (size_t i = 0; i < (size_t)T * H; ++i) {
        bad_o += memcmp(&o[i], &ref_o[i], 2) != 0;
        diff_o = std::max(diff_o, std::fabs(__bfloat162float(o[i]) - __bfloat162float(ref_o[i])));
    }
    printf("candidates (%u persistent CTAs): gate_up NVFP4 bytes differing %zu; down bf16 differing %zu, "
           "max abs diff %g; moe out (vec8) bf16 differing %zu, max abs diff %g\n",
           ctas, bad_q, bad_c, diff_c, bad_o, diff_o);
    float tp, tc, ratio;
    tm.ab(gate_up, gate_up_p, 2 * iters + 1, &tp, &tc, &ratio);
    printf("persist gate_up : %7.3f ms vs production %7.3f ms, paired %+.1f%%\n", tc, tp, 100.0 * (ratio - 1.0));
    tm.ab(down_proj, [&] { CK(cudaMemsetAsync(d_ctr + 1, 0, 4)); down_p(); }, 2 * iters + 1, &tp, &tc, &ratio);
    printf("persist down    : %7.3f ms vs production %7.3f ms, paired %+.1f%%\n", tc, tp, 100.0 * (ratio - 1.0));
    tm.ab([&] { unperm(d_out); }, [&] { unperm_v(d_out2); }, 2 * iters + 1, &tp, &tc, &ratio);
    printf("unpermute vec8  : %7.3f ms vs production %7.3f ms, paired %+.1f%%\n", tc, tp, 100.0 * (ratio - 1.0));
    if (bad_q || bad_c || bad_o) return 1;
#endif
    return 0;
}
