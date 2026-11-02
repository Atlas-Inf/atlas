// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of one rank's GLM dense FFN (layers 0..=2: gate/up 12288 x
// 4096, down 4096 x 12288, NVFP4) at a canonical verify's row counts,
// unsplit vs ATLAS_GLM_DENSE_TP_SPLIT's half (dense_ffn_tp_split.rs):
//
//   full   w4a16_gemv_tc8 gate, up (N 12288) -> moe_silu_mul -> down (N 4096)
//   split  w4a16_gemv_tc8 gate, up (N 6144) -> moe_silu_mul -> 2 pitched
//          copies (the join) -> down (N 2048, K 12288) -> 2 pitched copies
//
// The split's two pair exchanges (rows x 12 KiB, then rows x 4 KiB) need the
// pair; they are not here (add the rdma_oneshot copy time of a live profile).
// Bitwise: each split column must equal the unsplit column (checked here on
// the gate and down outputs at every row count). Weights cycle over
// `layers` copies (84 MB each), so they are cold as in a step. Every kernel
// launches with PDL as serving does.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -std=c++17 -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_dense_split_bench.cu -o glm_dense_split_bench
//   ./glm_dense_split_bench [layers=3] [reps=9]
#include "w4a16_gemv.cu"
#include "moe_silu_mul.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
typedef unsigned char u8;

static const unsigned H = 4096, I = 12288;

extern "C" __global__ void fill(u8* p, unsigned long long n, unsigned int kind, unsigned int seed) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int h = (unsigned int)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        u8 v = (u8)h;
        if (kind == 1) v = 0x20 + (v & 0x1F);              // E4M3 scales
        if (kind == 2) v = (u8)((h >> 8) & 0x3F) ^ 0x3C;   // moderate BF16 high byte
        p[i] = v;
    }
}

template <typename... Args>
static void launch(void (*kern)(Args...), dim3 grid, unsigned int block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block, 1, 1);
    cudaLaunchAttribute at;
    at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}

struct W { u8 *w, *s; };
struct Layer { W gate, up, down; };

static W weight(unsigned n, unsigned k, unsigned seed) {
    W w;
    CK(cudaMalloc(&w.w, (size_t)n * k / 2));
    CK(cudaMalloc(&w.s, (size_t)n * k / 16));
    fill<<<512, 256>>>(w.w, (unsigned long long)n * k / 2, 0, seed);
    fill<<<512, 256>>>(w.s, (unsigned long long)n * k / 16, 1, seed + 1);
    return w;
}

static W rows_from(W w, unsigned first, unsigned k) {
    return {w.w + (size_t)first * k / 2, w.s + (size_t)first * k / 16};
}

static void gemv(W w, const bf* a, bf* c, unsigned m, unsigned n, unsigned k) {
    launch(w4a16_gemv_tc8, dim3((n + 15) / 16), 256u, a, (const u8*)w.w, (const u8*)w.s, 0.37f, c, m, n, k);
}

static void silu(bf* g, const bf* u, unsigned n) {
    launch(moe_silu_mul, dim3((n + 255) / 256), 256u, (const bf*)g, u, g, n);
}

// Pitched join of two [m, n/2] halves into [m, n] (rank 0's first).
static void join(bf* dst, const bf* own, const bf* peer, unsigned m, unsigned n) {
    const size_t half = (size_t)n / 2 * 2;
    CK(cudaMemcpy2DAsync(dst, n * 2, own, half, half, m, cudaMemcpyDeviceToDevice, 0));
    CK(cudaMemcpy2DAsync((u8*)dst + half, n * 2, peer, half, half, m, cudaMemcpyDeviceToDevice, 0));
}

int main(int argc, char** argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 3;
    const int reps = argc > 2 ? atoi(argv[2]) : 9;
    std::vector<Layer> ls;
    for (int l = 0; l < layers; l++)
        ls.push_back({weight(I, H, 10 * l), weight(I, H, 10 * l + 3), weight(H, I, 10 * l + 6)});
    bf *x, *g, *u, *act, *out, *g2, *u2, *out2;
    for (bf** p : {&x, &g, &u, &act, &out, &g2, &u2, &out2}) CK(cudaMalloc(p, 32ull * I * 2));
    fill<<<64, 256>>>((u8*)x, 32ull * H * 2, 2, 9);
    CK(cudaDeviceSynchronize());

    // Bitwise: split columns vs unsplit, rank 0 and rank 1 halves.
    unsigned long long bad = 0, total = 0;
    for (unsigned m : {1u, 3u, 8u, 9u, 32u}) {
        const Layer& L = ls[0];
        gemv(L.gate, x, g, m, I, H);
        gemv(L.down, x /* any [m, I] activation */, out, m, H, I);
        for (unsigned r = 0; r < 2; r++) {
            gemv(rows_from(L.gate, r * I / 2, H), x, g2, m, I / 2, H);
            gemv(rows_from(L.down, r * H / 2, I), x, out2, m, H / 2, I);
            CK(cudaDeviceSynchronize());
            std::vector<unsigned short> a(m * I), b(m * I / 2), c(m * H), d(m * H / 2);
            CK(cudaMemcpy(a.data(), g, a.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), g2, b.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(c.data(), out, c.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d.data(), out2, d.size() * 2, cudaMemcpyDeviceToHost));
            for (unsigned row = 0; row < m; row++) {
                for (unsigned j = 0; j < I / 2; j++, total++)
                    bad += a[row * I + r * I / 2 + j] != b[row * I / 2 + j];
                for (unsigned j = 0; j < H / 2; j++, total++)
                    bad += c[row * H + r * H / 2 + j] != d[row * H / 2 + j];
            }
        }
    }
    printf("bitwise: %llu of %llu split outputs differ\n", bad, total);

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto run = [&](bool split, unsigned m) {
        std::vector<float> t;
        for (int r = 0; r < reps + 1; r++) {
            CK(cudaEventRecord(e0));
            for (const Layer& L : ls) {
                if (!split) {
                    gemv(L.gate, x, g, m, I, H);
                    gemv(L.up, x, u, m, I, H);
                    silu(g, u, m * I);
                    gemv(L.down, g, out, m, H, I);
                } else {
                    gemv(rows_from(L.gate, 0, H), x, g, m, I / 2, H);
                    gemv(rows_from(L.up, 0, H), x, u, m, I / 2, H);
                    silu(g, u, m * I / 2);
                    join(act, g, g + (size_t)m * I / 2, m, I);
                    gemv(rows_from(L.down, 0, I), act, g, m, H / 2, I);
                    join(out, g, g + (size_t)m * H / 2, m, H);
                }
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (r > 0) t.push_back(ms * 1000.0f / ls.size());
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    printf("rows  full_us/layer  split_us/layer (no exchanges)  saved_us/layer  saved_ms/step(3)\n");
    for (unsigned m : {1u, 3u, 5u, 8u, 16u, 32u}) {
        const float f = run(false, m), s = run(true, m);
        printf("%4u  %13.1f  %30.1f  %14.1f  %16.2f\n", m, f, s, f - s, (f - s) * 3 / 1000);
    }
    return bad == 0 ? 0 : 1;
}
