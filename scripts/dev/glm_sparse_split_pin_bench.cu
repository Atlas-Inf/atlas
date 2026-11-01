// SPDX-License-Identifier: AGPL-3.0-only
// Standalone A/B of the split count of GLM sparse-MLA verify owners
// (ATLAS_GLM_SPARSE_VERIFY_SPLIT_PIN): glm_sparse_mla_prefill_*_kv_pad_split
// + glm_sparse_decode_split_merge at the count the launch's rows pick today
// (`sparse_split_count`) versus the pinned count of the widest verify block
// (8 rows: 6 splits), for 1..8 rows of one sequence at `seq_start`, on an
// fp8_g128 and a BF16 latent cache. A third arm is the unsplit kernel (one CTA
// per row): what ATLAS_GLM_SPARSE_VERIFY_SPLIT=0 runs for verify rows, and
// what prefill pieces of 47 rows or more and fp8_g128 single-row decode run
// today. Its bits do not depend on the row count either; "pinned!=whole"
// counts how far a pinned verify row stays from that arithmetic. On fp8_g128
// the pipelined unsplit kernel (ATLAS_GLM_SPARSE_PREFILL_PIPE=1, the
// production prefill kernel) is timed and compared bit for bit as well.
//
// A launch of R rows verifies the first R of the same eight rows, so the
// bench also counts, per R, the BF16 outputs of those rows that differ from
// the 8-row launch: nonzero today (the bits of a position follow the verify
// width), zero when pinned. Accuracy is the largest |output - reference|
// against a double-precision host attention over the latents the kernel
// reads, for random ("flat") and one-key-dominant ("peak") queries. Timing
// cycles `copies` selections so the latents come from DRAM as in decode.
//
//   nvcc -arch=sm_121a -O3 --fmad=false -I kernels/gb10/glm-5.3-flash/nvfp4 \
//        scripts/dev/glm_sparse_split_pin_bench.cu -o split_pin_bench
//   ./split_pin_bench [seq_start=131064] [iters=240] [reps=7] [alt_splits=12]
// Device memory: ~0.3 GB at the defaults. Exit 2 if a pinned or an unsplit
// row moves with the row count, or the pipelined kernel differs from kv_pad.
#include "glm_sparse_prefill_kv_reuse.cu"
#include "glm_sparse_decode_split_merge.cu"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef __nv_bfloat16 bf;
static const unsigned HEADS = 32, DIM = 512, WIDTH = 2051, BS = 16, MAXR = 8, SMEM = 69376;
static const unsigned PIN = 6, COPIES = 24;
// `splits` values of run(): the unsplit kv_pad kernel and the pipelined one.
static const unsigned WHOLE = 0, PIPED = 99;

static float bf2f(unsigned short b) { unsigned int u = (unsigned int)b << 16; float f; memcpy(&f, &u, 4); return f; }
static unsigned short f2bf(float f) { const bf b = __float2bfloat16(f); unsigned short u; memcpy(&u, &b, 2); return u; }

// `sparse_split_count` (crates/spark-model/src/layers/ops/glm_sparse_prefill_tc.rs).
static unsigned split_count(unsigned rows) {
    unsigned best = 1, best_cost = ~0u;
    for (unsigned s = 1; s <= 16; s++) {
        const unsigned cost = ((rows * s + 47) / 48) * ((65 + s - 1) / s);
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    return best;
}

// The BF16 latents the fp8_g128 kernel attends: its own dequantization.
extern "C" __global__ void dequant_fp8g128(const void* cache, const unsigned* table, bf* out) {
    const unsigned token = blockIdx.x, col = threadIdx.x * 8;
    const unsigned physical = table[token / BS], offset = token % BS;
    const uint2 q = *reinterpret_cast<const uint2*>(glm_fp8g128_values(cache, physical, offset, BS) + col);
    const float scale = glm_fp8g128_scales(cache, physical, offset, BS)[col / 128u];
    *reinterpret_cast<uint4*>(out + (size_t)token * DIM + col) = glm_fp8g128_dequant8(q, scale);
}

int main(int argc, char** argv) {
    const unsigned seq_start = argc > 1 ? atoi(argv[1]) : 131064;
    const int iters = argc > 2 ? atoi(argv[2]) : 240;
    const int reps = argc > 3 ? atoi(argv[3]) : 7;
    const unsigned alt = argc > 4 ? atoi(argv[4]) : 12;
    const unsigned tokens = seq_start + MAXR, blocks = (tokens + BS - 1) / BS;
    std::mt19937 rng(20260930);
    std::normal_distribution<float> nd(0.f, 1.f);

    // One latent per token in shuffled physical blocks, as fp8_g128
    // (glm_fp8g128.cuh layout) and as BF16.
    std::vector<unsigned> table(blocks);
    for (unsigned i = 0; i < blocks; i++) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<unsigned char> fp8((size_t)blocks * BS * GLM_FP8G128_TOKEN_BYTES, 0);
    std::vector<unsigned short> b16((size_t)blocks * BS * DIM, 0);
    std::vector<float> v(DIM);
    for (unsigned t = 0; t < tokens; t++) {
        const float mag = 0.5f + (float)(t % 7) * 0.25f;
        for (unsigned d = 0; d < DIM; d++) v[d] = nd(rng) * mag;
        const size_t slot = (size_t)table[t / BS] * BS;
        unsigned char* base = fp8.data() + slot * GLM_FP8G128_TOKEN_BYTES;
        const unsigned off = t % BS;
        float* sc = reinterpret_cast<float*>(base + BS * 512 + off * 16);
        for (unsigned g = 0; g < 4; g++) {
            float m = 0.f;
            for (unsigned d = 0; d < 128; d++) m = fmaxf(m, fabsf(v[g * 128 + d]));
            if (!(m > 1.0e-4f)) m = 1.0e-4f;
            sc[g] = m * (float)(1.0 / 448.0);
            for (unsigned d = 0; d < 128; d++)
                base[off * 512 + g * 128 + d] =
                    __nv_cvt_float_to_fp8(v[g * 128 + d] / sc[g], __NV_SATFINITE, __NV_E4M3);
        }
        for (unsigned d = 0; d < DIM; d++) b16[(slot + off) * DIM + d] = f2bf(v[d]);
    }

    // COPIES selections of the eight rows: 512 kpool-4 pools by a shared
    // importance plus per-row noise, in score order, then the causal tail.
    const unsigned pools_all = tokens / 4;
    std::vector<int> idx((size_t)COPIES * MAXR * WIDTH, -1);
    std::vector<std::pair<float, unsigned>> cand;
    for (unsigned c = 0; c < COPIES; c++) {
        std::vector<float> imp(pools_all);
        for (auto& x : imp) x = nd(rng);
        for (unsigned r = 0; r < MAXR; r++) {
            const unsigned p = seq_start + r, pools = (p + 1) / 4;
            cand.clear();
            for (unsigned q = 0; q < pools; q++) {
                unsigned h = (q * 2654435761u) ^ (r * 40503u);
                h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
                const float recency = q + 64 >= pools ? 3.f : 0.f;
                cand.push_back({imp[q] + recency + 0.3f * ((h & 0xFFFF) / 65535.f - 0.5f), q});
            }
            const unsigned k = std::min(512u, pools);
            std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            int* row = &idx[((size_t)c * MAXR + r) * WIDTH];
            for (unsigned i = 0; i < k; i++)
                for (unsigned j = 0; j < 4; j++) row[i * 4 + j] = (int)(cand[i].second * 4 + j);
            for (unsigned t = pools * 4, j = 0; t <= p; t++, j++) row[2048 + j] = (int)t;
        }
    }

    void *dfp8, *db16, *dtable, *didx, *ddeq;
    CK(cudaMalloc(&dfp8, fp8.size()));
    CK(cudaMalloc(&db16, b16.size() * 2));
    CK(cudaMalloc(&dtable, table.size() * 4));
    CK(cudaMalloc(&didx, idx.size() * 4));
    CK(cudaMalloc(&ddeq, (size_t)tokens * DIM * 2));
    CK(cudaMemcpy(dfp8, fp8.data(), fp8.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db16, b16.data(), b16.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtable, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    dequant_fp8g128<<<tokens, 64>>>(dfp8, (const unsigned*)dtable, (bf*)ddeq);
    CK(cudaDeviceSynchronize());
    // Host latents by token for the reference: [0] fp8_g128 (as dequantized), [1] BF16.
    std::vector<unsigned short> lat[2];
    lat[0].resize((size_t)tokens * DIM);
    CK(cudaMemcpy(lat[0].data(), ddeq, lat[0].size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaFree(ddeq));
    lat[1].resize((size_t)tokens * DIM);
    for (unsigned t = 0; t < tokens; t++)
        memcpy(&lat[1][(size_t)t * DIM], &b16[((size_t)table[t / BS] * BS + t % BS) * DIM], DIM * 2);

    // Queries of the eight rows: [0] flat, [1] peaked on one selected key of
    // copy 0 (logit about +10 over the rest), per dtype since the keys differ.
    const size_t qn = (size_t)MAXR * HEADS * DIM, rh = (size_t)MAXR * HEADS;
    std::vector<unsigned short> q[2][2];
    for (int dt = 0; dt < 2; dt++) {
        q[dt][0].resize(qn);
        for (auto& x : q[dt][0]) x = f2bf(nd(rng) * 1.5f);
        q[dt][1].resize(qn);
        for (unsigned r = 0; r < MAXR; r++)
            for (unsigned h = 0; h < HEADS; h++) {
                const int token = idx[(size_t)r * WIDTH + rng() % 2048];
                const unsigned short* key = &lat[dt][(size_t)token * DIM];
                double norm = 0.0;
                for (unsigned d = 0; d < DIM; d++) norm += (double)bf2f(key[d]) * bf2f(key[d]);
                const float alpha = (float)(160.0 / norm);
                for (unsigned d = 0; d < DIM; d++)
                    q[dt][1][((size_t)r * HEADS + h) * DIM + d] = f2bf(alpha * bf2f(key[d]) + 0.05f * nd(rng));
            }
    }

    void *dq, *dpo, *dpl, *dlse, *dout;
    CK(cudaMalloc(&dq, qn * 2));
    CK(cudaMalloc(&dpo, 16 * qn * 4));
    CK(cudaMalloc(&dpl, 16 * rh * 4));
    CK(cudaMalloc(&dlse, rh * 4));
    CK(cudaMalloc(&dout, qn * 2));
    typedef void (*Kern)(GLM_KV_PAD_ARGS, float*, float*);
    const Kern kern[2] = {glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split,
                          glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split};
    typedef void (*Whole)(GLM_KV_PAD_ARGS);
    const Whole whole[2] = {glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad,
                            glm_sparse_mla_prefill_bf16_head32_tc_kv_pad};
    const Whole piped = glm_sparse_mla_prefill_fp8g128_head32_tc_pipe;
    const void* cache[2] = {dfp8, db16};
    const char* name[2] = {"fp8_g128", "bf16"};
    for (int dt = 0; dt < 2; dt++) {
        CK(cudaFuncSetAttribute(kern[dt], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        CK(cudaFuncSetAttribute(whole[dt], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    }
    CK(cudaFuncSetAttribute(piped, cudaFuncAttributeMaxDynamicSharedMemorySize, GLM_PIPE_SMEM));
    auto run = [&](int dt, unsigned rows, unsigned splits, unsigned copy) {
        const int* sel = (const int*)didx + (size_t)copy * MAXR * WIDTH;
        if (splits == WHOLE || splits == PIPED) {
            const bool pipe = splits == PIPED;
            (pipe ? piped : whole[dt])<<<dim3(1, rows, 1), 256, pipe ? GLM_PIPE_SMEM : SMEM>>>(
                (const bf*)dq, cache[dt], cache[dt], sel, (bf*)dout, (const unsigned*)dtable, rows,
                HEADS, DIM, WIDTH, BS, 0.0625f);
            return;
        }
        kern[dt]<<<dim3(1, rows, splits), 256, SMEM>>>(
            (const bf*)dq, cache[dt], cache[dt], sel,
            nullptr, (const unsigned*)dtable, rows, HEADS, DIM, WIDTH, BS, 0.0625f,
            (float*)dpo, (float*)dpl);
        glm_sparse_decode_split_merge<<<rows * HEADS, 256>>>(
            (const float*)dpo, (const float*)dpl, (bf*)dout, (float*)dlse, rows, HEADS, DIM, splits);
    };
    auto output = [&](int dt, unsigned rows, unsigned splits) {
        CK(cudaMemset(dout, 0xEE, qn * 2));
        run(dt, rows, splits, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<unsigned short> o((size_t)rows * HEADS * DIM);
        CK(cudaMemcpy(o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost));
        return o;
    };
    auto time_us = [&](int dt, unsigned rows, unsigned splits) {
        std::vector<double> us;
        for (int rep = 0; rep < reps + 1; rep++) {
            CK(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; it++) run(dt, rows, splits, it % COPIES);
            CK(cudaDeviceSynchronize());
            const std::chrono::duration<double, std::micro> d = std::chrono::steady_clock::now() - t0;
            if (rep > 0) us.push_back(d.count() / iters);
        }
        std::sort(us.begin(), us.end());
        return us[us.size() / 2];
    };

    int moved = 0, whole_moved = 0, pipe_differs = 0;
    std::vector<float> keys((size_t)WIDTH * DIM);
    std::vector<double> ref(qn), logit(WIDTH);
    for (int dt = 0; dt < 2; dt++) {
        for (int peak = 0; peak < 2; peak++) {
            CK(cudaMemcpy(dq, q[dt][peak].data(), qn * 2, cudaMemcpyHostToDevice));
            // Double-precision reference of the eight rows (copy 0).
            double scale = 0.0;
            for (unsigned r = 0; r < MAXR; r++) {
                const int* row = &idx[(size_t)r * WIDTH];
                unsigned n = 0;
                for (unsigned j = 0; j < WIDTH; j++) {
                    if (row[j] < 0) continue;
                    const unsigned short* key = &lat[dt][(size_t)row[j] * DIM];
                    for (unsigned d = 0; d < DIM; d++) keys[(size_t)n * DIM + d] = bf2f(key[d]);
                    n++;
                }
                for (unsigned h = 0; h < HEADS; h++) {
                    const unsigned short* qr = &q[dt][peak][((size_t)r * HEADS + h) * DIM];
                    double qd[DIM], mx = -1e300, sum = 0.0;
                    for (unsigned d = 0; d < DIM; d++) qd[d] = bf2f(qr[d]);
                    for (unsigned j = 0; j < n; j++) {
                        double s = 0.0;
                        for (unsigned d = 0; d < DIM; d++) s += qd[d] * keys[(size_t)j * DIM + d];
                        logit[j] = s * 0.0625;
                        mx = std::max(mx, logit[j]);
                    }
                    for (unsigned j = 0; j < n; j++) { logit[j] = exp(logit[j] - mx); sum += logit[j]; }
                    double* o = &ref[((size_t)r * HEADS + h) * DIM];
                    std::fill(o, o + DIM, 0.0);
                    for (unsigned j = 0; j < n; j++) {
                        const double p = logit[j] / sum;
                        for (unsigned d = 0; d < DIM; d++) o[d] += p * keys[(size_t)j * DIM + d];
                    }
                    for (unsigned d = 0; d < DIM; d++) scale = std::max(scale, fabs(o[d]));
                }
            }
            auto err = [&](const std::vector<unsigned short>& o) {
                double e = 0.0;
                for (size_t i = 0; i < o.size(); i++) e = std::max(e, fabs(bf2f(o[i]) - ref[i]));
                return e;
            };
            auto differing = [](const std::vector<unsigned short>& a, const std::vector<unsigned short>& b) {
                size_t n = 0;
                for (size_t i = 0; i < a.size(); i++) n += a[i] != b[i];
                return n;
            };
            const std::vector<unsigned short> wide[3] = {output(dt, MAXR, split_count(MAXR)),
                                                        output(dt, MAXR, PIN), output(dt, MAXR, WHOLE)};
            printf("%s %s queries, seq_start=%u, max |reference| %.3g\n", name[dt],
                   peak ? "peak" : "flat", seq_start, scale);
            printf("rows  splits  err_today  err_pinned  err_whole   today!=pinned   today!=8row  "
                   "pinned!=8row  whole!=8row  pinned!=whole\n");
            for (unsigned rows = 1; rows <= MAXR; rows++) {
                const unsigned today = split_count(rows);
                const auto a = output(dt, rows, today), b = output(dt, rows, PIN), w = output(dt, rows, WHOLE);
                const size_t vs8_pinned = differing(b, wide[1]), vs8_whole = differing(w, wide[2]);
                moved += vs8_pinned != 0;
                whole_moved += vs8_whole != 0;
                if (dt == 0) pipe_differs += differing(output(dt, rows, PIPED), w) != 0;
                printf("%4u  %2u->%u   %.3e  %.3e   %.3e  %6zu/%-6zu  %6zu       %6zu        %6zu       %6zu\n",
                       rows, today, PIN, err(a), err(b), err(w), differing(a, b), a.size(),
                       differing(a, wide[0]), vs8_pinned, vs8_whole, differing(b, w));
            }
        }
        // The kernels' work does not depend on the values: time on the
        // queries the loop left.
        printf("%s split + merge, median us per launch (%d iters x %d reps, %u selections)\n",
               name[dt], iters, reps, COPIES);
        printf("rows  today        pinned %u         unsplit%s     alt %u\n", PIN,
               dt == 0 ? "           unsplit pipe" : "", alt);
        for (unsigned rows = 1; rows <= MAXR; rows++) {
            const unsigned today = split_count(rows);
            const double t = time_us(dt, rows, today), p = time_us(dt, rows, PIN);
            const double w = time_us(dt, rows, WHOLE);
            printf("%4u  S%-2u %7.1f   %7.1f (%+5.1f)   %7.1f (%+7.1f)", rows, today, t, p, p - t, w, w - t);
            if (dt == 0) {
                const double pp = time_us(dt, rows, PIPED);
                printf("   %7.1f (%+7.1f)", pp, pp - t);
            }
            if (alt > 0) printf("   %7.1f", time_us(dt, rows, alt));
            printf("\n");
        }
    }
    printf("pinned rows that differ from the 8-row launch: %d cases\n", moved);
    printf("unsplit rows that differ from the 8-row launch: %d cases\n", whole_moved);
    printf("fp8_g128 pipelined outputs that differ from kv_pad: %d cases\n", pipe_differs);
    return moved == 0 && whole_moved == 0 && pipe_differs == 0 ? 0 : 2;
}
