// SPDX-License-Identifier: AGPL-3.0-only
// Per-row bit parity and cost of the qwen4_exp (Qwen3.8-Flash-Next) K=2/3/4
// verify kernels against the single-row kernels serial decode runs
// (ATLAS_QWEN4EXP_EXACT_VERIFY, `model/qwen4exp_exact_verify.rs`).
//
// Row i of a batched launch over M rows must equal, byte for byte, a
// single-row launch on row i alone. Real shapes, synthetic data (random
// inputs at three scales, NVFP4/FP8 codes and scales drawn over their
// finite ranges), M = 2, 3 and 4:
//
//   BF16 GEMV      dense_gemv_bf16  vs  _batch2 / _batchm        (GDN qkvz/out_proj, LM head)
//   FP8 GEMV       w8a16_gemv       vs  w8a16_gemv_batch4         (ATLAS_QWEN4EXP_FP8_GDN)
//   NVFP4 GEMV     w4a16_gemv[_sw]  vs  w4a16_gemv_batch2 / 3 / 4 (router, attention o_proj)
//   NVFP4 Q+gate   w4a16_gemv_qg    vs  w4a16_gemv_qg_batch2/3/4  (attention Q)
//   NVFP4 K/V      w4a16_gemv_dual  vs  w4a16_gemv_dual_batch2/3  (attention K/V: the default
//                                       verify arm) and vs w4a16_gemv_batch2/3/4 per projection
//   BF16 tile GEMM dense_gemm_bf16_pipelined M=1 vs M=2/3/4   (PLE key/value projections)
//   mHC collapse   hc_pre_stage/down/finish_x4, hc_post (default, T=2..8) and the
//                  _vec kernels (ATLAS_QWEN4EXP_HC_FAST): T=2/3/4 vs T=1 per row
//   BF16 LM head   dense_gemv_bf16 vs dense_gemm_bf16 M=3/4 (the default K=3/4 head;
//                  informational — the exact verify projects `_batchm` rows)
//
// Then the cost of what the exact switch changes, GPU time per launch over a
// ring of weight copies (>= 4x the 24 MiB L2, so weights stream from DRAM as
// serving's distinct layers do):
//   K/V    two w4a16_gemv_dual launches  vs  one w4a16_gemv_dual_batch2  vs  2 x w4a16_gemv_batch2
//   GDN    2 x gated_delta_rule_decode_f32 + h snapshot copy  vs  gated_delta_rule_wy2
//
// Build (each kernel source is its own module, loaded through the driver API
// as Atlas loads them; --fmad=false as KERNEL.toml builds this target):
//   K=kernels/gb10/common Q=kernels/gb10/qwen3.8-flash-next/nvfp4
//   for f in dense_gemv_bf16 dense_gemv_bf16_batch2 dense_gemv_bf16_batchm w8a16_gemv \
//            w8a16_gemv_batch4 w4a16_gemv w4a16_gemv_fused dense_gemm_bf16 gated_delta_rule_wy; do
//     nvcc -cubin -arch=sm_121a -O3 --fmad=false $K/$f.cu -o $f.cubin; done
//   nvcc -cubin -arch=sm_121a -O3 --fmad=false $Q/hyper_connection.cu -o hyper_connection.cubin
//   nvcc -cubin -arch=sm_121a -O3 --fmad=false $Q/gated_delta_rule.cu -o gated_delta_rule.cubin
//   nvcc -arch=sm_121a -O3 -std=c++17 scripts/dev/qwen4exp_exact_verify_bench.cu -o exact_bench -lcuda
//   ./exact_bench [cubin_dir=.] [iters=200]
// Prints one line per comparison (ok / MISMATCH with the differing-element
// count; the default K/V arm the switch replaces prints "differs" without
// failing), PASS or FAIL, then the timings.
//
// Measured 2026-10-05 on GB10 (ennspark03): PASS; dual_batch2/3 differ from
// serial in 1 of 1024 / 1536 outputs at the TP2 K/V shape (x0.02 inputs).
// K/V at K=2: 2 x dual 14.5 us | dual_batch2 7.9 | 2 x batch2 12.9 (TP2),
// 20.7 | 10.0 | 16.0 (TP1). GDN recurrence at K=2: exact 50.6 us vs wy2
// 28.2 (TP2, 1.5 MB h a layer), 141.1 vs 50.8 (TP1).
//
// M=4 added 2026-10-05 (ennspark03): PASS, incl. w4a16_gemv_qg_batch4 vs
// w4a16_gemv_qg and w4a16_gemv_batch4 vs w4a16_gemv_dual per projection.
// The default BF16 LM head at 3/4 rows (scalar dense_gemm_bf16) differs from
// dense_gemv_bf16 in 1 of 12288 / 16384 outputs, so the exact verify now
// projects `_batchm` rows at K=3 too. Draft head (one dense_gemv_bf16 over
// [n, 2560]): 5210.7 us at the full 248320 vocab, 1360.8 at 65536, 988.3
// at 47149, 686.8 at 32768 (~244 GB/s each).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s = nullptr; \
    cuGetErrorString(r, &s); fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, s ? s : "?"); exit(1); } } while (0)

static std::string g_dir = ".";
static int g_iters = 200;
static int g_fail = 0;
static std::mt19937 g_rng(1234);

static CUfunction load(const char* module, const char* fn) {
    static std::vector<std::pair<std::string, CUmodule>> mods;
    CUmodule m = nullptr;
    for (auto& p : mods) if (p.first == module) m = p.second;
    if (!m) {
        CU(cuModuleLoad(&m, (g_dir + "/" + module + ".cubin").c_str()));
        mods.push_back({module, m});
    }
    CUfunction f;
    CU(cuModuleGetFunction(&f, m, fn));
    return f;
}

static void launch(CUfunction f, dim3 g, dim3 b, std::vector<void*> args, unsigned smem = 0) {
    if (smem > 48 * 1024)
        CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, smem, 0, args.data(), nullptr));
}

static unsigned short tobf(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    unsigned short u;
    memcpy(&u, &b, 2);
    return u;
}

template <typename T>
static T* dput(const std::vector<T>& h) {
    T* p;
    CK(cudaMalloc(&p, h.size() * sizeof(T) + 256));
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static void* dzero(size_t bytes) {
    void* p;
    CK(cudaMalloc(&p, bytes + 256));
    CK(cudaMemset(p, 0, bytes + 256));
    return p;
}
static std::vector<unsigned char> dget(const void* p, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    return h;
}

static std::vector<unsigned short> rand_bf16(size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<unsigned short> v(n);
    for (auto& x : v) x = tobf(d(g_rng));
    // Signed zeros and exact zeros are where reassociation shows first.
    for (size_t i = 0; i < n; i += 97) v[i] = (i / 97) % 2 ? 0x8000 : 0x0000;
    return v;
}
static std::vector<float> rand_f32(size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}
static std::vector<unsigned char> rand_bytes(size_t n) {
    std::uniform_int_distribution<int> d(0, 255);
    std::vector<unsigned char> v(n);
    for (auto& x : v) x = (unsigned char)d(g_rng);
    return v;
}
// E4M3 codes over the finite range, positive (NVFP4 block scales) or any sign
// (FP8 weights); 0x7f / 0xff are NaN.
static std::vector<unsigned char> rand_e4m3(size_t n, bool positive, int lo = 0x20, int hi = 0x48) {
    std::uniform_int_distribution<int> d(lo, hi), s(0, 1);
    std::vector<unsigned char> v(n);
    for (auto& x : v) x = (unsigned char)(d(g_rng) | (!positive && s(g_rng) ? 0x80 : 0));
    return v;
}

// `gate` = a comparison the exact verify relies on (a mismatch FAILs the
// run); otherwise informational (the default verify arm the switch replaces).
static void report(const char* what, const std::vector<unsigned char>& a,
                   const std::vector<unsigned char>& b, size_t elem, bool gate = true) {
    size_t bad = 0;
    for (size_t i = 0; i + elem <= a.size(); i += elem)
        if (memcmp(&a[i], &b[i], elem) != 0) bad++;
    printf("  %-58s %s", what, bad ? (gate ? "MISMATCH" : "differs (not used exact)") : "ok");
    if (bad) printf(" (%zu of %zu elements)", bad, a.size() / elem);
    printf("\n");
    if (bad && gate) g_fail++;
}

// GPU time per launch over `g_iters` back-to-back calls.
template <typename F>
static double time_us(F f) {
    for (int i = 0; i < 10; i++) f(i);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, 0));
    for (int i = 0; i < g_iters; i++) f(i);
    CK(cudaEventRecord(b, 0));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    return 1000.0 * ms / g_iters;
}

static const float SCALES[3] = {0.02f, 1.0f, 30.0f};

// ── BF16 GEMV ────────────────────────────────────────────────────────────
static void bf16_gemv(unsigned N, unsigned K, const char* tag) {
    CUfunction f1 = load("dense_gemv_bf16", "dense_gemv_bf16");
    CUfunction f2 = load("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2");
    CUfunction fm = load("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    auto* B = dput(rand_bf16((size_t)N * K, 0.05f));
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(4 * (size_t)K, sc));
        auto* C1 = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* C2 = (unsigned short*)dzero(4 * (size_t)N * 2);
        for (unsigned r = 0; r < 4; r++) {
            void* a = A + (size_t)r * K;
            void* c = C1 + (size_t)r * N;
            launch(f1, dim3((N + 3) / 4), dim3(256), {&a, &B, &c, &N, &K});
        }
        auto ref = dget(C1, 4 * (size_t)N * 2);
        char name[128];
        unsigned stride = N;
        launch(f2, dim3((N + 3) / 4), dim3(256), {&A, &B, &C2, &N, &K, &stride});
        snprintf(name, sizeof name, "%s bf16 batch2 x%g", tag, sc);
        report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + 2 * N * 2),
               dget(C2, 2 * (size_t)N * 2), 2);
        for (unsigned M = 2; M <= 4; M++) {
            launch(fm, dim3((N + 3) / 4), dim3(256), {&A, &B, &C2, &M, &N, &K, &stride});
            snprintf(name, sizeof name, "%s bf16 batchm M=%u x%g", tag, M, sc);
            report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
                   dget(C2, (size_t)M * N * 2), 2);
        }
        cudaFree(A); cudaFree(C1); cudaFree(C2);
    }
    cudaFree(B);
}

// ── FP8 (128x128 block-scaled) GEMV ──────────────────────────────────────
static void fp8_gemv(unsigned N, unsigned K, const char* tag) {
    CUfunction f1 = load("w8a16_gemv", "w8a16_gemv");
    CUfunction f4 = load("w8a16_gemv_batch4", "w8a16_gemv_batch4");
    auto* B = dput(rand_e4m3((size_t)N * K, false, 0x08, 0x5f));
    std::vector<float> s(((N + 127) / 128) * ((K + 127) / 128));
    for (auto& x : s) x = 0.002f + 0.01f * (float)(g_rng() % 1000) / 1000.f;
    auto* S = dput(s);
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(4 * (size_t)K, sc));
        auto* C1 = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* C2 = (unsigned short*)dzero(4 * (size_t)N * 2);
        for (unsigned r = 0; r < 4; r++) {
            void* a = A + (size_t)r * K;
            void* c = C1 + (size_t)r * N;
            launch(f1, dim3((N + 3) / 4), dim3(256), {&a, &B, &S, &c, &N, &K});
        }
        auto ref = dget(C1, 4 * (size_t)N * 2);
        for (unsigned M = 2; M <= 4; M++) {
            launch(f4, dim3((N + 3) / 4), dim3(256), {&A, &B, &S, &C2, &M, &N, &K});
            char name[128];
            snprintf(name, sizeof name, "%s fp8 w8a16_gemv_batch4 M=%u x%g", tag, M, sc);
            report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
                   dget(C2, (size_t)M * N * 2), 2);
        }
        cudaFree(A); cudaFree(C1); cudaFree(C2);
    }
    cudaFree(B); cudaFree(S);
}

// ── NVFP4 GEMVs ──────────────────────────────────────────────────────────
struct Fp4 {
    void* w;   // [N, K/2] packed E2M1
    void* s;   // [N, K/16] E4M3 block scales
    float s2;
};
static Fp4 rand_fp4(unsigned N, unsigned K) {
    return {dput(rand_bytes((size_t)N * K / 2)), dput(rand_e4m3((size_t)N * K / 16, true)), 0.0021f};
}

// Q+gate (interleaved per head, written deinterleaved) and plain GEMVs.
static void fp4_gemv(unsigned N, unsigned K, unsigned nh, unsigned hd, const char* tag) {
    CUfunction g1 = load("w4a16_gemv", "w4a16_gemv");
    CUfunction gsw = load("w4a16_gemv", "w4a16_gemv_sw");
    CUfunction gb[3] = {load("w4a16_gemv", "w4a16_gemv_batch2"), load("w4a16_gemv", "w4a16_gemv_batch3"),
                        load("w4a16_gemv", "w4a16_gemv_batch4")};
    CUfunction q1 = load("w4a16_gemv", "w4a16_gemv_qg");
    CUfunction qb[3] = {load("w4a16_gemv", "w4a16_gemv_qg_batch2"), load("w4a16_gemv", "w4a16_gemv_qg_batch3"),
                        load("w4a16_gemv", "w4a16_gemv_qg_batch4")};
    Fp4 W = rand_fp4(N, K);
    // The Q+gate kernels deinterleave [nh x (Q|gate) x hd] rows; only shapes
    // that are such a projection run them.
    const bool qg = nh * hd * 2 == N;
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(4 * (size_t)K, sc));
        auto* R = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* Rs = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* Rq = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* O = (unsigned short*)dzero(4 * (size_t)N * 2);
        for (unsigned r = 0; r < 4; r++) {
            void* a = A + (size_t)r * K;
            void* c = R + (size_t)r * N;
            void* cs = Rs + (size_t)r * N;
            void* cq = Rq + (size_t)r * N;
            launch(g1, dim3((N + 3) / 4), dim3(256), {&a, &W.w, &W.s, &W.s2, &c, &N, &K});
            launch(gsw, dim3((N + 7) / 8), dim3(256), {&a, &W.w, &W.s, &W.s2, &cs, &N, &K});
            if (qg) launch(q1, dim3((N + 3) / 4), dim3(256), {&a, &W.w, &W.s, &W.s2, &cq, &N, &K, &nh, &hd});
        }
        auto ref = dget(R, 4 * (size_t)N * 2);
        auto refq = dget(Rq, 4 * (size_t)N * 2);
        char name[128];
        snprintf(name, sizeof name, "%s nvfp4 w4a16_gemv_sw vs w4a16_gemv x%g", tag, sc);
        report(name, ref, dget(Rs, 4 * (size_t)N * 2), 2);
        for (unsigned M = 2; M <= 4; M++) {
            if (M == 4)  // batch4 takes M (the M<=4 template entry point)
                launch(gb[2], dim3((N + 3) / 4), dim3(256), {&A, &W.w, &W.s, &W.s2, &O, &M, &N, &K});
            else
                launch(gb[M - 2], dim3((N + 3) / 4), dim3(256), {&A, &W.w, &W.s, &W.s2, &O, &N, &K});
            snprintf(name, sizeof name, "%s nvfp4 w4a16_gemv_batch%u x%g", tag, M, sc);
            report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
                   dget(O, (size_t)M * N * 2), 2);
            if (!qg) continue;
            launch(qb[M - 2], dim3((N + 3) / 4), dim3(256), {&A, &W.w, &W.s, &W.s2, &O, &N, &K, &nh, &hd});
            snprintf(name, sizeof name, "%s nvfp4 w4a16_gemv_qg_batch%u x%g", tag, M, sc);
            report(name, std::vector<unsigned char>(refq.begin(), refq.begin() + M * N * 2),
                   dget(O, (size_t)M * N * 2), 2);
        }
        cudaFree(A); cudaFree(R); cudaFree(Rs); cudaFree(Rq); cudaFree(O);
    }
    cudaFree(W.w); cudaFree(W.s);
}

// The 5..8-row tiers a batched multi-sequence step runs (batch5..8, the
// strided batch4_os/batch8_os the attention QKV writes with) vs the serial
// GEMV per row (ATLAS_QWEN4EXP_BATCH_FAST, padded widths 5..8).
static void fp4_tiers(unsigned N, unsigned K, unsigned nh, unsigned hd, const char* tag) {
    CUfunction g1 = load("w4a16_gemv", "w4a16_gemv");
    const char* tiers[] = {"w4a16_gemv_batch5", "w4a16_gemv_batch6", "w4a16_gemv_batch7",
                           "w4a16_gemv_batch8", "w4a16_gemv_batch16"};
    Fp4 W = rand_fp4(N, K);
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(8 * (size_t)K, sc));
        auto* R = (unsigned short*)dzero(8 * (size_t)N * 2);
        auto* O = (unsigned short*)dzero(16 * (size_t)N * 2 + 4096);
        for (unsigned r = 0; r < 8; r++) {
            void* a = A + (size_t)r * K;
            void* c = R + (size_t)r * N;
            launch(g1, dim3((N + 3) / 4), dim3(256), {&a, &W.w, &W.s, &W.s2, &c, &N, &K});
        }
        auto ref = dget(R, 8 * (size_t)N * 2);
        char name[160];
        for (unsigned t = 0; t < 5; t++) {
            CUfunction f = load("w4a16_gemv", tiers[t]);
            const unsigned lo = t < 4 ? 5 + t : 5, hi = t < 4 ? 5 + t : 8;
            for (unsigned M = (t == 3 ? 5 : lo); M <= hi; M++) {
                CK(cudaMemset(O, 0x55, (size_t)M * N * 2));
                launch(f, dim3((N + 3) / 4), dim3(256), {&A, &W.w, &W.s, &W.s2, &O, &M, &N, &K});
                snprintf(name, sizeof name, "%s %s M=%u vs w4a16_gemv x%g", tag, tiers[t], M, sc);
                report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
                       dget(O, (size_t)M * N * 2), 2);
            }
        }
        // Q+gate: serial decode runs `w4a16_gemv_qg` (one accumulator, the
        // scale folded into the weight), NOT the scalar template the default
        // 4+-row QKV arm (`ms_qkv_batchn`) runs then deinterleaves. Shown
        // here (informational); the exact lanes take `qg_batchN` instead.
        if (nh) {
            CUfunction qg = load("w4a16_gemv", "w4a16_gemv_qg");
            auto* Q = (unsigned short*)dzero(8 * (size_t)N * 2);
            for (unsigned r = 0; r < 8; r++) {
                void* a = A + (size_t)r * K;
                void* c = Q + (size_t)r * N;
                launch(qg, dim3((N + 3) / 4), dim3(256), {&a, &W.w, &W.s, &W.s2, &c, &N, &K, &nh, &hd});
            }
            // Deinterleave the template rows on the host: [Q_h G_h]... -> [Q... | G...].
            std::vector<unsigned char> de(ref.size());
            for (unsigned r = 0; r < 8; r++)
                for (unsigned head = 0; head < nh; head++)
                    for (unsigned part = 0; part < 2; part++)
                        memcpy(&de[((size_t)r * N + (size_t)part * nh * hd + (size_t)head * hd) * 2],
                               &ref[((size_t)r * N + (size_t)(2 * head + part) * hd) * 2], (size_t)hd * 2);
            snprintf(name, sizeof name, "%s scalar template (batchN/_os) vs w4a16_gemv_qg x%g", tag, sc);
            report(name, dget(Q, 8 * (size_t)N * 2), de, 2, false);
            cudaFree(Q);
        }
        for (const char* os : {"w4a16_gemv_batch4_os", "w4a16_gemv_batch8_os"}) {
            const unsigned top = os[16] == '4' ? 4 : 8;
            CUfunction f = load("w4a16_gemv", os);
            // Strided rows, as the QKV path writes them (stride > N).
            const unsigned stride = N + 64;
            for (unsigned M = (top == 4 ? 2 : 5); M <= top; M++) {
                CK(cudaMemset(O, 0x55, (size_t)M * stride * 2));
                launch(f, dim3((N + 3) / 4), dim3(256), {&A, &W.w, &W.s, &W.s2, &O, &M, &N, &K, (void*)&stride});
                auto got = dget(O, (size_t)M * stride * 2);
                std::vector<unsigned char> rows;
                for (unsigned r = 0; r < M; r++)
                    rows.insert(rows.end(), got.begin() + (size_t)r * stride * 2,
                                got.begin() + (size_t)r * stride * 2 + (size_t)N * 2);
                snprintf(name, sizeof name, "%s %s M=%u (strided) vs w4a16_gemv x%g", tag, os, M, sc);
                report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2), rows, 2);
            }
        }
        cudaFree(A); cudaFree(R); cudaFree(O);
    }
    cudaFree(W.w); cudaFree(W.s);
}

// K/V: serial runs `w4a16_gemv_dual` (one launch, blockIdx.z picks K or V).
static void fp4_dual(unsigned N, unsigned K, const char* tag) {
    CUfunction d1 = load("w4a16_gemv_fused", "w4a16_gemv_dual");
    CUfunction db[2] = {load("w4a16_gemv", "w4a16_gemv_dual_batch2"), load("w4a16_gemv", "w4a16_gemv_dual_batch3")};
    CUfunction gb[3] = {load("w4a16_gemv", "w4a16_gemv_batch2"), load("w4a16_gemv", "w4a16_gemv_batch3"),
                        load("w4a16_gemv", "w4a16_gemv_batch4")};
    Fp4 Wk = rand_fp4(N, K), Wv = rand_fp4(N, K);
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(4 * (size_t)K, sc));
        auto* Rk = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* Rv = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* Ok = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* Ov = (unsigned short*)dzero(4 * (size_t)N * 2);
        for (unsigned r = 0; r < 4; r++) {
            void* a = A + (size_t)r * K;
            void* ck = Rk + (size_t)r * N;
            void* cv = Rv + (size_t)r * N;
            launch(d1, dim3((N + 3) / 4, 1, 2), dim3(256),
                   {&a, &Wk.w, &Wk.s, &Wk.s2, &ck, &Wv.w, &Wv.s, &Wv.s2, &cv, &N, &K});
        }
        auto rk = dget(Rk, 4 * (size_t)N * 2), rv = dget(Rv, 4 * (size_t)N * 2);
        char name[128];
        for (unsigned M = 2; M <= 4; M++) {
            size_t b = (size_t)M * N * 2;
            std::vector<unsigned char> r(rk.begin(), rk.begin() + b);
            r.insert(r.end(), rv.begin(), rv.begin() + b);
            std::vector<unsigned char> ok, ov;
            if (M <= 3) {  // no dual_batch4: the default K=4 arm is the batch4 template
                launch(db[M - 2], dim3((N + 3) / 4, 1, 2), dim3(256),
                       {&A, &Wk.w, &Wk.s, &Wk.s2, &Ok, &Wv.w, &Wv.s, &Wv.s2, &Ov, &N, &K});
                snprintf(name, sizeof name, "%s K/V w4a16_gemv_dual_batch%u (default verify) x%g", tag, M, sc);
                ok = dget(Ok, b); ov = dget(Ov, b);
                ok.insert(ok.end(), ov.begin(), ov.end());
                report(name, r, ok, 2, false);
                launch(gb[M - 2], dim3((N + 3) / 4), dim3(256), {&A, &Wk.w, &Wk.s, &Wk.s2, &Ok, &N, &K});
                launch(gb[M - 2], dim3((N + 3) / 4), dim3(256), {&A, &Wv.w, &Wv.s, &Wv.s2, &Ov, &N, &K});
            } else {
                launch(gb[2], dim3((N + 3) / 4), dim3(256), {&A, &Wk.w, &Wk.s, &Wk.s2, &Ok, &M, &N, &K});
                launch(gb[2], dim3((N + 3) / 4), dim3(256), {&A, &Wv.w, &Wv.s, &Wv.s2, &Ov, &M, &N, &K});
            }
            snprintf(name, sizeof name, "%s K/V 2 x w4a16_gemv_batch%u x%g", tag, M, sc);
            ok = dget(Ok, b); ov = dget(Ov, b);
            ok.insert(ok.end(), ov.begin(), ov.end());
            report(name, r, ok, 2);
        }
        cudaFree(A); cudaFree(Rk); cudaFree(Rv); cudaFree(Ok); cudaFree(Ov);
    }

    // Cost at this shape: what K=2 verify launches per attention layer.
    const int copies = std::max(8, (int)((256u << 20) / ((size_t)N * K / 2 * 2 + 1)));
    std::vector<Fp4> ks, vs;
    for (int i = 0; i < copies; i++) { ks.push_back(rand_fp4(N, K)); vs.push_back(rand_fp4(N, K)); }
    auto* A = dput(rand_bf16(3 * (size_t)K, 1.f));
    auto* Ok = (unsigned short*)dzero(3 * (size_t)N * 2);
    auto* Ov = (unsigned short*)dzero(3 * (size_t)N * 2);
    double per_row = time_us([&](int i) {
        Fp4& k = ks[i % copies];
        Fp4& v = vs[i % copies];
        for (unsigned r = 0; r < 2; r++) {
            void* a = A + (size_t)r * K;
            void* ck = Ok + (size_t)r * N;
            void* cv = Ov + (size_t)r * N;
            launch(d1, dim3((N + 3) / 4, 1, 2), dim3(256),
                   {&a, &k.w, &k.s, &k.s2, &ck, &v.w, &v.s, &v.s2, &cv, &N, &K});
        }
    });
    double dual_b2 = time_us([&](int i) {
        Fp4& k = ks[i % copies];
        Fp4& v = vs[i % copies];
        launch(db[0], dim3((N + 3) / 4, 1, 2), dim3(256),
               {&A, &k.w, &k.s, &k.s2, &Ok, &v.w, &v.s, &v.s2, &Ov, &N, &K});
    });
    double two_b2 = time_us([&](int i) {
        Fp4& k = ks[i % copies];
        Fp4& v = vs[i % copies];
        launch(gb[0], dim3((N + 3) / 4), dim3(256), {&A, &k.w, &k.s, &k.s2, &Ok, &N, &K});
        launch(gb[0], dim3((N + 3) / 4), dim3(256), {&A, &v.w, &v.s, &v.s2, &Ov, &N, &K});
    });
    printf("  cost %s K/V at K=2: 2 x dual %.1f us | dual_batch2 %.1f us | 2 x batch2 %.1f us\n",
           tag, per_row, dual_b2, two_b2);
    for (auto& w : ks) { cudaFree(w.w); cudaFree(w.s); }
    for (auto& w : vs) { cudaFree(w.w); cudaFree(w.s); }
    cudaFree(A); cudaFree(Ok); cudaFree(Ov); cudaFree(Wk.w); cudaFree(Wk.s); cudaFree(Wv.w); cudaFree(Wv.s);
}

// ── BF16 tile GEMM (PLE projections): M=1 vs M=2/3 per row ───────────────
static void bf16_gemm(unsigned N, unsigned K, const char* tag) {
    CUfunction g = load("dense_gemm_bf16", "dense_gemm_bf16_pipelined");
    const unsigned T = 128, TH = 256;
    auto* B = dput(rand_bf16((size_t)N * K, 0.05f));
    for (float sc : SCALES) {
        auto* A = dput(rand_bf16(4 * (size_t)K, sc));
        auto* R = (unsigned short*)dzero(4 * (size_t)N * 2);
        auto* O = (unsigned short*)dzero(4 * (size_t)N * 2);
        unsigned one = 1;
        for (unsigned r = 0; r < 4; r++) {
            void* a = A + (size_t)r * K;
            void* c = R + (size_t)r * N;
            launch(g, dim3((N + T - 1) / T, 1), dim3(TH), {&a, &B, &c, &one, &N, &K});
        }
        auto ref = dget(R, 4 * (size_t)N * 2);
        for (unsigned M = 2; M <= 4; M++) {
            launch(g, dim3((N + T - 1) / T, (M + T - 1) / T), dim3(TH), {&A, &B, &O, &M, &N, &K});
            char name[128];
            snprintf(name, sizeof name, "%s dense_gemm_bf16_pipelined M=%u x%g", tag, M, sc);
            report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
                   dget(O, (size_t)M * N * 2), 2);
        }
        cudaFree(A); cudaFree(R); cudaFree(O);
    }
    cudaFree(B);
}

// ── mHC collapse + post: T=2/3 vs T=1 per row ────────────────────────────
static const unsigned H = 2560, HC = 4, RANK = 320, HCD = HC * H;
struct HcOut {
    std::vector<unsigned char> normed, low, y, inj, post;
};
// One chain over `T` rows starting at row `r0` of the inputs. `vec` = the
// ATLAS_QWEN4EXP_HC_FAST kernels.
static HcOut hc_chain(bool vec, unsigned T, unsigned r0, float* streams, void* norm_w, void* down_w,
                      void* up_w, void* inject_w, void* block_out) {
    float* normed = (float*)dzero((size_t)T * HCD * 4);
    float* low = (float*)dzero((size_t)T * RANK * 4);
    void* y = dzero((size_t)T * H * 2);
    float* inj = (float*)dzero((size_t)T * HC * 4);
    float* post = (float*)dzero((size_t)T * HCD * 4);
    float* s = streams + (size_t)r0 * HCD;
    void* bo = (unsigned short*)block_out + (size_t)r0 * H;
    unsigned h = H, hc = HC, rank = RANK;
    float eps = 1e-6f;
    if (!vec) {
        launch(load("hyper_connection", "hc_pre_stage"), dim3(T), dim3(1024), {&s, &norm_w, &normed, &h, &hc, &eps});
        unsigned dsplit = std::min(10u, std::max(1u, 48u / T));
        launch(load("hyper_connection", "hc_pre_down"), dim3(T, dsplit), dim3(1024),
               {&normed, &down_w, &low, &h, &hc, &rank, &T}, HCD * 4);
        launch(load("hyper_connection", "hc_pre_finish_x4"), dim3(T, H / 32), dim3(128),
               {&normed, &low, &up_w, &inject_w, &y, &inj, &h, &hc, &rank}, RANK * 4);
        launch(load("hyper_connection", "hc_post"), dim3(T), dim3(256), {&bo, &s, &inj, &post, &h, &hc});
    } else {
        launch(load("hyper_connection", "hc_pre_stage_vec"), dim3(T, 8), dim3(1024), {&s, &norm_w, &normed, &h, &hc, &eps});
        unsigned threads = (RANK + HC) * (32 / 2);
        launch(load("hyper_connection", "hc_pre_down_vec"), dim3((threads + 127) / 128), dim3(128),
               {&normed, &down_w, &inject_w, &low, &inj, &h, &hc, &rank, &T});
        launch(load("hyper_connection", "hc_pre_finish_vec"), dim3((HCD / 4 + 127) / 128), dim3(128),
               {&normed, &low, &up_w, &y, &h, &rank, &T}, T * RANK * 4);
        launch(load("hyper_connection", "hc_post_vec"), dim3(T, (H / 4 + 63) / 64), dim3(64),
               {&bo, &s, &inj, &post, &h, &hc});
    }
    HcOut o{dget(normed, (size_t)T * HCD * 4), dget(low, (size_t)T * RANK * 4), dget(y, (size_t)T * H * 2),
            dget(inj, (size_t)T * HC * 4), dget(post, (size_t)T * HCD * 4)};
    cudaFree(normed); cudaFree(low); cudaFree(y); cudaFree(inj); cudaFree(post);
    return o;
}
static void hc_rows() {
    void* norm_w = dput(rand_bf16(HCD, 0.3f));
    void* down_w = dput(rand_bf16((size_t)RANK * HCD, 0.02f));
    void* up_w = dput(rand_bf16((size_t)RANK * HCD, 0.02f));  // [rank, hc*H] (transposed at load)
    void* inject_w = dput(rand_bf16((size_t)HC * HCD, 0.02f));
    for (int vec = 0; vec < 2; vec++) {
        for (float sc : SCALES) {
            // T up to 8: ATLAS_QWEN4EXP_BATCH_FAST runs batches of up to 8 rows
            // (and wider ones in 8-row chunks) on the split path; the _vec
            // kernels serve T <= 4.
            const unsigned tmax = vec ? 4 : 8;
            float* streams = dput(rand_f32(tmax * (size_t)HCD, sc));
            void* bo = dput(rand_bf16(tmax * (size_t)H, sc));
            std::vector<HcOut> one;
            for (unsigned r = 0; r < tmax; r++) one.push_back(hc_chain(vec, 1, r, streams, norm_w, down_w, up_w, inject_w, bo));
            for (unsigned T = 2; T <= tmax; T++) {
                HcOut b = hc_chain(vec, T, 0, streams, norm_w, down_w, up_w, inject_w, bo);
                auto cat = [&](std::vector<unsigned char> HcOut::*f) {
                    std::vector<unsigned char> v;
                    for (unsigned r = 0; r < T; r++) v.insert(v.end(), (one[r].*f).begin(), (one[r].*f).end());
                    return v;
                };
                char name[128];
                const char* k = vec ? "hc _vec (HC_FAST)" : "hc default";
                snprintf(name, sizeof name, "%s T=%u normed x%g", k, T, sc); report(name, cat(&HcOut::normed), b.normed, 4);
                snprintf(name, sizeof name, "%s T=%u low x%g", k, T, sc); report(name, cat(&HcOut::low), b.low, 4);
                snprintf(name, sizeof name, "%s T=%u y x%g", k, T, sc); report(name, cat(&HcOut::y), b.y, 2);
                snprintf(name, sizeof name, "%s T=%u inj x%g", k, T, sc); report(name, cat(&HcOut::inj), b.inj, 4);
                snprintf(name, sizeof name, "%s T=%u post x%g", k, T, sc); report(name, cat(&HcOut::post), b.post, 4);
            }
            cudaFree(streams); cudaFree(bo);
        }
    }
    cudaFree(norm_w); cudaFree(down_w); cudaFree(up_w); cudaFree(inject_w);
}

// ── mHC: the default chain vs the _vec chain, row by row at T=1 ──
// A batched step past the _vec kernels' T <= 4 runs the default chain, while
// single-row decode under ATLAS_QWEN4EXP_HC_FAST runs _vec: the two must be
// the same bytes for a 5..8-row batch to equal C1.
static void hc_cross() {
    void* norm_w = dput(rand_bf16(HCD, 0.3f));
    void* down_w = dput(rand_bf16((size_t)RANK * HCD, 0.02f));
    void* up_w = dput(rand_bf16((size_t)RANK * HCD, 0.02f));
    void* inject_w = dput(rand_bf16((size_t)HC * HCD, 0.02f));
    for (float sc : SCALES) {
        float* streams = dput(rand_f32(4 * (size_t)HCD, sc));
        void* bo = dput(rand_bf16(4 * (size_t)H, sc));
        for (unsigned r = 0; r < 4; r++) {
            HcOut d = hc_chain(false, 1, r, streams, norm_w, down_w, up_w, inject_w, bo);
            HcOut v = hc_chain(true, 1, r, streams, norm_w, down_w, up_w, inject_w, bo);
            char name[128];
            snprintf(name, sizeof name, "hc default vs _vec T=1 row %u normed x%g", r, sc); report(name, d.normed, v.normed, 4);
            snprintf(name, sizeof name, "hc default vs _vec T=1 row %u low x%g", r, sc); report(name, d.low, v.low, 4);
            snprintf(name, sizeof name, "hc default vs _vec T=1 row %u y x%g", r, sc); report(name, d.y, v.y, 2);
            snprintf(name, sizeof name, "hc default vs _vec T=1 row %u inj x%g", r, sc); report(name, d.inj, v.inj, 4);
            snprintf(name, sizeof name, "hc default vs _vec T=1 row %u post x%g", r, sc); report(name, d.post, v.post, 4);
        }
        cudaFree(streams); cudaFree(bo);
    }
    cudaFree(norm_w); cudaFree(down_w); cudaFree(up_w); cudaFree(inject_w);
}

// ── BF16 LM head: serial dense_gemv_bf16 rows vs the default K=3/4 head ──
// `lm_head_batched` on a BF16 head runs the scalar tile GEMM at 3+ rows; the
// exact verify projects `dense_gemv_bf16_batchm` rows instead (checked in
// bf16_gemv). Informational: how far the default head is from serial.
static void head_gemm(unsigned N, unsigned K) {
    CUfunction f1 = load("dense_gemv_bf16", "dense_gemv_bf16");
    CUfunction g = load("dense_gemm_bf16", "dense_gemm_bf16");
    auto* B = dput(rand_bf16((size_t)N * K, 0.05f));
    auto* A = dput(rand_bf16(4 * (size_t)K, 1.0f));
    auto* R = (unsigned short*)dzero(4 * (size_t)N * 2);
    auto* O = (unsigned short*)dzero(4 * (size_t)N * 2);
    for (unsigned r = 0; r < 4; r++) {
        void* a = A + (size_t)r * K;
        void* c = R + (size_t)r * N;
        launch(f1, dim3((N + 3) / 4), dim3(256), {&a, &B, &c, &N, &K});
    }
    auto ref = dget(R, 4 * (size_t)N * 2);
    for (unsigned M = 3; M <= 4; M++) {
        launch(g, dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16), {&A, &B, &O, &M, &N, &K});
        char name[128];
        snprintf(name, sizeof name, "lm head dense_gemm_bf16 M=%u vs dense_gemv_bf16 (default)", M);
        report(name, std::vector<unsigned char>(ref.begin(), ref.begin() + M * N * 2),
               dget(O, (size_t)M * N * 2), 2, false);
    }
    cudaFree(A); cudaFree(B); cudaFree(R); cudaFree(O);
}

// ── Draft head cost: full-vocab vs reduced-vocab BF16 GEMV (one draft) ──
// The MTP drafter's argmax head (`ATLAS_QWEN4EXP_DRAFT_VOCAB`): one
// dense_gemv_bf16 over [n, 2560] per draft. Weights stream from DRAM (the
// head is far larger than L2), so the cost is ~bytes / bandwidth.
static void draft_head_cost(unsigned K) {
    CUfunction f1 = load("dense_gemv_bf16", "dense_gemv_bf16");
    const unsigned full = 248320;
    auto* B = dput(rand_bf16((size_t)full * K, 0.05f));
    auto* A = dput(rand_bf16(K, 1.0f));
    auto* C = (unsigned short*)dzero((size_t)full * 2);
    for (unsigned n : {248320u, 65536u, 47149u, 32768u}) {
        double us = time_us([&](int) { launch(f1, dim3((n + 3) / 4), dim3(256), {&A, &B, &C, &n, &K}); });
        printf("  cost draft head dense_gemv_bf16 [%u x %u]: %.1f us (%.1f GB/s)\n", n, K, us,
               (double)n * K * 2 / us / 1e3);
    }
    cudaFree(A); cudaFree(B); cudaFree(C);
}

// ── GDN recurrence cost at K=2: exact (2 x decode_f32 + h snapshot) vs wy2 ──
static void gdn_cost(unsigned nk, unsigned nv, const char* tag) {
    CUfunction f32 = load("gated_delta_rule", "gated_delta_rule_decode_f32");
    CUfunction wy2 = load("gated_delta_rule_wy", "gated_delta_rule_wy2");
    const unsigned kd = 128, vd = 128;
    const size_t hb = (size_t)nv * kd * vd * 4;
    const int copies = std::max(4, (int)((256u << 20) / (hb * 2)));
    std::vector<float*> hs, his;
    for (int i = 0; i < copies; i++) {
        hs.push_back((float*)dput(rand_f32(hb / 4, 0.1f)));
        his.push_back((float*)dzero(hb));
    }
    // Row layout of the f32 chain: q|k|v FP32 rows, gate|beta FP32.
    const unsigned qk = nk * kd, vdim = nv * vd;
    float* qkv32 = (float*)dput(rand_f32(2 * (size_t)(2 * qk + vdim), 0.1f));
    unsigned short* qkv16 = (unsigned short*)dput(rand_bf16(2 * (size_t)(2 * qk + vdim), 0.1f));
    std::vector<float> gbh(4 * nv);
    for (size_t i = 0; i < gbh.size(); i++) gbh[i] = (i % (2 * nv)) < nv ? 0.97f : 0.4f;
    float* gb = (float*)dput(gbh);
    float* out32 = (float*)dzero(2 * (size_t)vdim * 4);
    void* out16 = dzero(2 * (size_t)vdim * 2);
    unsigned one = 1, row = 2 * qk + vdim, gbs = 2 * nv, tab = 0;
    double exact = time_us([&](int i) {
        float* h = hs[i % copies];
        for (unsigned t = 0; t < 2; t++) {
            float* q = qkv32 + (size_t)t * row;
            float* k = q + qk;
            float* v = k + qk;
            float* g = gb + (size_t)t * gbs;
            float* b = g + nv;
            float* o = out32 + (size_t)t * vdim;
            launch(f32, dim3(nv, 1), dim3(128), {&h, &q, &k, &v, &g, &b, &o, &one, &nk, &nv, (void*)&kd, (void*)&vd});
            if (t == 0) CK(cudaMemcpyAsync(his[i % copies], h, hb, cudaMemcpyDeviceToDevice, 0));
        }
    });
    double wy = time_us([&](int i) {
        float* h = hs[i % copies];
        float* hi = his[i % copies];
        unsigned short* q = qkv16;
        unsigned short* k = q + qk;
        unsigned short* v = k + qk;
        float* g = gb;
        float* b = g + nv;
        launch(wy2, dim3(nv, 1), dim3(128),
               {&h, &q, &k, &v, &g, &b, &out16, &hi, &one, &nk, &nv, (void*)&kd, (void*)&vd, &row, &row, &gbs, &tab});
    });
    printf("  cost %s GDN recurrence at K=2 (h %.1f MB/layer): exact %.1f us | wy2 %.1f us\n",
           tag, hb / 1048576.0, exact, wy);
    for (auto* p : hs) cudaFree(p);
    for (auto* p : his) cudaFree(p);
    cudaFree(qkv32); cudaFree(qkv16); cudaFree(gb); cudaFree(out32); cudaFree(out16);
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (argc > 2) g_iters = atoi(argv[2]);
    CU(cuInit(0));
    CK(cudaFree(0));
    printf("Per-row bit parity, batched verify launch vs single-row launches:\n");
    // GDN projections (TP2 shard, TP1) and the LM head row shape.
    bf16_gemv(8192, 2560, "qkvz TP2");
    bf16_gemv(2560, 3072, "out_proj TP2");
    bf16_gemv(16384, 2560, "qkvz TP1");
    bf16_gemv(2560, 6144, "out_proj TP1");
    fp8_gemv(8192, 2560, "qkvz TP2");
    fp8_gemv(2560, 3072, "out_proj TP2");
    // Router [512, 2560]; attention Q+gate [24 heads x 256 x 2] and o_proj.
    fp4_gemv(512, 2560, 0, 0, "router");
    fp4_tiers(512, 2560, 0, 0, "router");
    fp4_tiers(6144, 2560, 12, 256, "attn q+gate TP2");
    fp4_tiers(256, 2560, 0, 0, "attn k/v TP2");
    fp4_tiers(2560, 3072, 0, 0, "attn o_proj TP2");
    fp4_gemv(12288, 2560, 24, 256, "attn q+gate TP1");
    fp4_gemv(6144, 2560, 12, 256, "attn q+gate TP2");
    fp4_gemv(2560, 6144, 0, 0, "attn o_proj TP1");
    fp4_dual(512, 2560, "attn TP1");
    fp4_dual(256, 2560, "attn TP2");
    bf16_gemm(2560, 2560, "PLE value");
    bf16_gemm(10240, 2560, "PLE key");
    hc_rows();
    hc_cross();
    head_gemm(4096, 2560);
    printf("%s (%d mismatching comparison%s)\n\nCost:\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    // K/V cost is printed inside fp4_dual; GDN here.
    gdn_cost(8, 24, "TP2");
    gdn_cost(16, 48, "TP1");
    draft_head_cost(2560);
    return g_fail ? 1 : 0;
}
