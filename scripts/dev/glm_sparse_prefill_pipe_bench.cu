// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the fp8_g128 GLM sparse-MLA prefill kernels:
// glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad (production) versus the
// opt-in glm_sparse_mla_prefill_fp8g128_head32_tc_pipe. Realistic shapes: one
// 4096-row prefill piece at `seq_start`, 32 heads, 2051 selected slots (512
// kpool-4 pools picked by a shared importance plus per-row noise, in score
// order, then the 0..3-token causal tail), a shuffled 16-token block table.
// Reports ms per call and the bitwise / max-abs output difference.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_sparse_prefill_pipe_bench.cu -o pipe_bench
//   ./pipe_bench [rows=4096] [seq_start=57344] [iters=5]
// Device memory: ~0.4 GB at the defaults.
#include "glm_sparse_prefill_kv_reuse.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static float bf2f(unsigned short b) { unsigned int u = (unsigned int)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    const unsigned rows = argc > 1 ? atoi(argv[1]) : 4096;
    const unsigned seq_start = argc > 2 ? atoi(argv[2]) : 57344;
    const int iters = argc > 3 ? atoi(argv[3]) : 5;
    const unsigned heads = 32, dim = 512, width = 2051, bs = 16;
    const unsigned tokens = seq_start + rows, blocks = (tokens + bs - 1) / bs;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    // fp8_g128 cache in shuffled physical blocks (glm_fp8g128.cuh layout).
    std::vector<unsigned> table(blocks);
    for (unsigned i = 0; i < blocks; i++) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<unsigned char> cache((size_t)blocks * bs * GLM_FP8G128_TOKEN_BYTES, 0);
    std::vector<float> v(dim);
    for (unsigned t = 0; t < tokens; t++) {
        const float mag = 0.5f + (float)(t % 7) * 0.25f;
        for (unsigned d = 0; d < dim; d++) v[d] = nd(rng) * mag;
        const unsigned phys = table[t / bs], off = t % bs;
        unsigned char* base = cache.data() + (size_t)phys * bs * GLM_FP8G128_TOKEN_BYTES;
        float* sc = reinterpret_cast<float*>(base + bs * 512 + off * 16);
        for (unsigned g = 0; g < 4; g++) {
            float m = 0.f;
            for (unsigned d = 0; d < 128; d++) m = fmaxf(m, fabsf(v[g * 128 + d]));
            if (!(m > 1.0e-4f)) m = 1.0e-4f;
            sc[g] = m * (float)(1.0 / 448.0);
            for (unsigned d = 0; d < 128; d++)
                base[off * 512 + g * 128 + d] =
                    __nv_cvt_float_to_fp8(v[g * 128 + d] / sc[g], __NV_SATFINITE, __NV_E4M3);
        }
    }

    // Selection: pools fully inside [0, p], top 512 by importance + noise.
    std::vector<float> base((tokens + 3) / 4);
    for (auto& b : base) b = nd(rng);
    std::vector<int> idx((size_t)rows * width, -1);
    std::vector<std::pair<float, unsigned>> cand;
    for (unsigned r = 0; r < rows; r++) {
        const unsigned p = seq_start + r, pools = (p + 1) / 4;
        cand.clear();
        for (unsigned q = 0; q < pools; q++) {
            unsigned h = (q * 2654435761u) ^ (r * 40503u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            const float recency = q + 64 >= pools ? 3.f : 0.f;
            cand.push_back({base[q] + recency + 0.3f * ((h & 0xFFFF) / 65535.f - 0.5f), q});
        }
        const unsigned k = std::min(512u, pools);
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        int* row = &idx[(size_t)r * width];
        for (unsigned i = 0; i < k; i++)
            for (unsigned j = 0; j < 4; j++) row[i * 4 + j] = (int)(cand[i].second * 4 + j);
        for (unsigned t = pools * 4, j = 0; t <= p; t++, j++) row[2048 + j] = (int)t;
    }

    std::vector<unsigned short> q((size_t)rows * heads * dim);
    for (auto& x : q) {
        const __nv_bfloat16 b = __float2bfloat16(nd(rng) * 1.5f);
        memcpy(&x, &b, 2);
    }

    void *dq, *dcache, *didx, *dtable, *dout0, *dout1;
    const size_t out_bytes = (size_t)rows * heads * dim * 2;
    CK(cudaMalloc(&dq, q.size() * 2));
    CK(cudaMalloc(&dcache, cache.size()));
    CK(cudaMalloc(&didx, idx.size() * 4));
    CK(cudaMalloc(&dtable, table.size() * 4));
    CK(cudaMalloc(&dout0, out_bytes));
    CK(cudaMalloc(&dout1, out_bytes));
    CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcache, cache.data(), cache.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtable, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout0, 0xFF, out_bytes));
    CK(cudaMemset(dout1, 0xEE, out_bytes));

    typedef void (*Kern)(GLM_KV_PAD_ARGS);
    const Kern kern[2] = {glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad,
                          glm_sparse_mla_prefill_fp8g128_head32_tc_pipe};
    const int smem[2] = {69376, GLM_PIPE_SMEM};
    void* outs[2] = {dout0, dout1};
    for (int k = 0; k < 2; k++)
        CK(cudaFuncSetAttribute(kern[k], cudaFuncAttributeMaxDynamicSharedMemorySize, smem[k]));
    auto run = [&](int k) {
        kern[k]<<<dim3(1, rows, 1), 256, smem[k]>>>(
            (const __nv_bfloat16*)dq, dcache, dcache, (const int*)didx, (__nv_bfloat16*)outs[k],
            (const unsigned*)dtable, rows, heads, dim, width, bs, 0.0625f);
    };
    run(0); run(1);
    CK(cudaDeviceSynchronize());

    std::vector<unsigned short> o0((size_t)rows * heads * dim), o1(o0.size());
    CK(cudaMemcpy(o0.data(), dout0, out_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o1.data(), dout1, out_bytes, cudaMemcpyDeviceToHost));
    size_t diff = 0, nonfinite = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < o0.size(); i++) {
        const float a = bf2f(o0[i]), b = bf2f(o1[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) nonfinite++;
        if (o0[i] != o1[i]) { diff++; max_abs = std::max(max_abs, (double)fabsf(a - b)); }
    }

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float ms[2] = {0.f, 0.f};
    for (int it = 0; it < iters; it++)
        for (int k = 0; k < 2; k++) {
            float t;
            CK(cudaEventRecord(e0)); run(k); CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&t, e0, e1));
            ms[k] += t;
        }
    const double flop = 2.0 * 2.0 * rows * heads * 2048.0 * dim;
    printf("rows=%u seq_start=%u width=%u iters=%d\n", rows, seq_start, width, iters);
    for (int k = 0; k < 2; k++)
        printf("%-6s %8.3f ms/call  %6.1f TFLOP/s (2048 keys)\n", k ? "pipe" : "kv_pad",
               ms[k] / iters, flop / (ms[k] / iters * 1e-3) / 1e12);
    printf("speedup %.3fx  differing_bf16=%zu/%zu  max_abs_diff=%.3g  nonfinite=%zu\n",
           ms[0] / ms[1], diff, o0.size(), max_abs, nonfinite);
    return diff == 0 && nonfinite == 0 ? 0 : 2;
}
