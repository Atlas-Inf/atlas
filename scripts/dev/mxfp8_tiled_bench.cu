// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the MXFP8 tensor-core GEMV on its row-major weights
// (mxfp8_gemv_tc{8,16,32}) against the chunk-tiled twin
// (mxfp8_gemv_tc{8,16,32}_tiled on mxfp8_tile16's layout; ATLAS_GLM_MXFP8_TILED),
// at the GLM-5.3-Flash rank-local shapes: MLA q_a 1536x4096, kv_a 512x4096,
// q_b 8192x1536, o 4096x8192 and the vocab-split LM head 77440x4096.
//
// 1. Bitwise: every output of every shape at M = 1..32 rows, tiled vs
//    row-major, and every tiled row at M rows against the same row at 32
//    rows (row invariance). Must print "bitwise: 0 of N outputs differ".
// 2. Time: each shape alone, launches back to back (tc8 with PDL as in
//    production), weights cycling over enough copies that every read comes
//    from DRAM (L2 is 24 MiB). Arms interleave per repetition; the table
//    shows the min and median per-launch us and the GB/s of the min, plus a
//    plain streaming read of the same bytes ("floor").
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/mxfp8_tiled_bench.cu -o mxfp8_tiled_bench
//   ./mxfp8_tiled_bench [rows=3] [reps=15]
// Exit 0 iff the bitwise check passes.
//
// ennspark03, 2026-10-12, 3 rows, GPU shared with vLLM serving (relative only),
// min us row-major / tiled / floor: q_b 58.8 / 57.2 / 55.7, o 167.7 / 150.4 /
// 143.2, lm_head 3165 / 2827 / 2815. Lowering registers for 5-6 CTAs/SM (the
// wave-quantization fix) spilled and lost on every shape (o 180 / 198).
#include "mxfp8_gemv.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef unsigned char u8;

// kind 2: E4M3 values without the NaN codes; 3: E8M0 scales near 2^-4.
extern "C" __global__ void fill(u8* p, unsigned long long n, unsigned int kind, unsigned int seed) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        u8 b = (u8)h;
        if (kind == 2 && (b & 0x7F) == 0x7F) b ^= 1;
        if (kind == 3) b = 120 + (b & 7);
        p[i] = b;
    }
}

// The bandwidth floor: read `n16` uint4s once, fold them into one word.
extern "C" __global__ void stream_read(const uint4* __restrict__ p, unsigned long long n16,
                                       unsigned int* __restrict__ sink) {
    atlas_pdl_enter();
    unsigned int x = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9E3779B9u) sink[0] = x;
}

template <typename... Args>
static void launch(bool pdl, void (*kern)(Args...), dim3 grid, unsigned int block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

typedef void (*Mx)(const bf*, const u8*, const u8*, bf*, unsigned, unsigned, unsigned, unsigned);
static const Mx BASE[3] = {mxfp8_gemv_tc8, mxfp8_gemv_tc16, mxfp8_gemv_tc32};
static const Mx TILED[3] = {mxfp8_gemv_tc8_tiled, mxfp8_gemv_tc16_tiled, mxfp8_gemv_tc32_tiled};

struct Shape { const char* name; unsigned int n, k, copies; };
struct Copy { u8 *w, *s, *wt, *st; };

static void gemv(bool tiled, const Shape& sh, const Copy& c, const bf* a, bf* out, unsigned int m) {
    const int tier = m <= 8 ? 0 : (m <= 16 ? 1 : 2);
    const Mx k = (tiled ? TILED : BASE)[tier];
    launch(tier == 0, k, dim3(sh.n / 16), 256u, a, (const u8*)(tiled ? c.wt : c.w),
           (const u8*)(tiled ? c.st : c.s), out, m, sh.n, sh.k, sh.n);
}

int main(int argc, char** argv) {
    const unsigned int rows = argc > 1 ? (unsigned int)atoi(argv[1]) : 3u;
    const int reps = argc > 2 ? atoi(argv[2]) : 15;
    const unsigned int MAXM = 32, KMAX = 8192, NMAX = 77440;
    const Shape shapes[] = {
        {"q_a", 1536, 4096, 12}, {"kv_a", 512, 4096, 32}, {"q_b", 8192, 1536, 11},
        {"o", 4096, 8192, 4}, {"lm_head", 77440, 4096, 1},
    };

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<unsigned short> hA((size_t)MAXM * KMAX);
    for (auto& x : hA) { bf b = __float2bfloat16(nd(rng)); memcpy(&x, &b, 2); }
    bf *dA, *dC[2], *dR;
    unsigned int* dSink;
    CK(cudaMalloc(&dA, hA.size() * 2));
    CK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    for (auto& c : dC) CK(cudaMalloc(&c, (size_t)MAXM * NMAX * 2));
    CK(cudaMalloc(&dR, (size_t)MAXM * NMAX * 2));
    CK(cudaMalloc(&dSink, 4));
    std::vector<unsigned short> h0((size_t)MAXM * NMAX), h1(h0.size()), hr(h0.size());

    unsigned long long checked = 0, differ = 0;
    std::vector<std::vector<Copy>> all;
    for (const Shape& sh : shapes) {
        std::vector<Copy> copies(sh.copies);
        const unsigned long long wb = (unsigned long long)sh.n * sh.k, sb = wb / 32;
        for (unsigned int i = 0; i < sh.copies; i++) {
            Copy& c = copies[i];
            CK(cudaMalloc(&c.w, wb)); CK(cudaMalloc(&c.s, sb));
            CK(cudaMalloc(&c.wt, wb)); CK(cudaMalloc(&c.st, sb));
            fill<<<1024, 256>>>(c.w, wb, 2u, 11u + i);
            fill<<<1024, 256>>>(c.s, sb, 3u, 29u + i);
            const unsigned long long threads = (unsigned long long)(sh.n / 16) * (sh.k / 64) * 32;
            mxfp8_tile16<<<(unsigned int)((threads + 255) / 256), 256>>>(c.w, c.s, c.wt, c.st, sh.n, sh.k);
        }
        CK(cudaDeviceSynchronize());
        // Bitwise: tiled vs row-major at every M; tiled rows at M vs at 32.
        gemv(true, sh, copies[0], dA, dR, MAXM);
        CK(cudaMemcpy(hr.data(), dR, (size_t)MAXM * sh.n * 2, cudaMemcpyDeviceToHost));
        unsigned long long d0 = differ;
        for (unsigned int m = 1; m <= MAXM; m++) {
            CK(cudaMemset(dC[0], 0xFF, (size_t)m * sh.n * 2));
            CK(cudaMemset(dC[1], 0xEE, (size_t)m * sh.n * 2));
            gemv(false, sh, copies[0], dA, dC[0], m);
            gemv(true, sh, copies[0], dA, dC[1], m);
            CK(cudaMemcpy(h0.data(), dC[0], (size_t)m * sh.n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1.data(), dC[1], (size_t)m * sh.n * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < (size_t)m * sh.n; i++) {
                checked += 2;
                differ += (h0[i] != h1[i]) + (h1[i] != hr[i]);
            }
        }
        printf("%-8s %5ux%-5u bitwise: %llu differ\n", sh.name, sh.n, sh.k, differ - d0);
        all.push_back(copies);
    }
    printf("bitwise: %llu of %llu outputs differ\n", differ, checked);

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("\n%u rows, %d reps; per launch: min / median us (GB/s at min)\n", rows, reps);
    printf("%-8s %8s | %22s | %22s | %22s | %s\n", "shape", "MB", "row-major", "tiled", "floor", "tiled saves");
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const Shape& sh = shapes[si];
        const auto& copies = all[si];
        const double mb = (double)sh.n * sh.k * (1.0 + 1.0 / 32) / 1e6;
        const unsigned int launches = std::max(8u, sh.copies * 2u);
        std::vector<float> t[3];
        for (int r = 0; r < reps + 1; r++) {
            for (int arm = 0; arm < 3; arm++) {
                CK(cudaEventRecord(e0, st));
                for (unsigned int l = 0; l < launches; l++) {
                    const Copy& c = copies[l % sh.copies];
                    if (arm == 2) {
                        const unsigned long long n16 = (unsigned long long)sh.n * sh.k / 16;
                        cudaLaunchConfig_t cfg = {};
                        cfg.gridDim = dim3(48 * 8); cfg.blockDim = dim3(256); cfg.stream = st;
                        CK(cudaLaunchKernelEx(&cfg, stream_read, (const uint4*)c.w, n16, dSink));
                        continue;
                    }
                    const int tier = rows <= 8 ? 0 : (rows <= 16 ? 1 : 2);
                    const Mx k = (arm ? TILED : BASE)[tier];
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = dim3(sh.n / 16); cfg.blockDim = dim3(256); cfg.stream = st;
                    cudaLaunchAttribute at;
                    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at.val.programmaticStreamSerializationAllowed = 1;
                    cfg.attrs = &at; cfg.numAttrs = tier == 0 ? 1 : 0;
                    CK(cudaLaunchKernelEx(&cfg, k, (const bf*)dA, (const u8*)(arm ? c.wt : c.w),
                                          (const u8*)(arm ? c.st : c.s), dC[0], rows, sh.n, sh.k, sh.n));
                }
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                if (r > 0) t[arm].push_back(ms * 1000.f / launches);
            }
        }
        double mn[3], md[3];
        for (int a = 0; a < 3; a++) {
            std::sort(t[a].begin(), t[a].end());
            mn[a] = t[a].front(); md[a] = t[a][t[a].size() / 2];
        }
        printf("%-8s %8.1f | %7.1f %7.1f (%4.0f) | %7.1f %7.1f (%4.0f) | %7.1f %7.1f (%4.0f) | %+.1f us min, %+.1f us median\n",
               sh.name, mb, mn[0], md[0], mb / mn[0] * 1e3, mn[1], md[1], mb / mn[1] * 1e3, mn[2],
               md[2], mb / mn[2] * 1e3,
               mn[0] - mn[1], md[0] - md[1]);
    }
    return differ == 0 ? 0 : 1;
}
