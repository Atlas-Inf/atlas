// SPDX-License-Identifier: AGPL-3.0-only
// kbench: w8a16_gemv_batch8_dyn_vl2_n2 (the 27B C8 decode's largest kernel, 35% of a step on strix s210) against
// candidate rewrites that keep every output's accumulation order, so each must be BIT-IDENTICAL to the baseline.
// Shapes are the 27B FP8 projections at M = 8 decode rows. Reports us/call, weight GB/s, and a bytewise compare.
//
//   n2        the shipped kernel (included from kernels/strix-hip/common/w8a16_gemv_batch4.cu)
//   n2_bit    e4m3 decoded with integer ops instead of the LDS LUT (same function the LUT is built from; no barrier)
//   n2_pf     next-iteration weight words + scales issued before this iteration's FMAs (more bytes in flight)
//   n4        4 adjacent outputs per thread (activation loads/unpacks shared four ways), grid ceil(N/32)
#include "w8a16_gemv_batch4.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void decode16(uint4 p, float sc, float w[16]) {
    const unsigned int ws[4] = {p.x, p.y, p.z, p.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j*4]   = atlas_e4m3_to_f32((unsigned char)(ws[j] & 0xffu)) * sc;
        w[j*4+1] = atlas_e4m3_to_f32((unsigned char)((ws[j] >> 8) & 0xffu)) * sc;
        w[j*4+2] = atlas_e4m3_to_f32((unsigned char)((ws[j] >> 16) & 0xffu)) * sc;
        w[j*4+3] = atlas_e4m3_to_f32((unsigned char)(ws[j] >> 24)) * sc;
    }
}

// Shared body for the N-outputs-per-thread family. Per output: logical lanes p (k16 = lane, lane+64, ...) and p+32
// (k16 = lane+32, ...), the same k16 walk, accumulate order and two-half 32-lane reduce as the shipped n2 kernel.
template <int NO, bool BITDEC, bool PREFETCH>
__device__ __forceinline__ void gemv8_body(const __nv_bfloat16* A, const unsigned char* B, const float* block_scale,
                                           __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K,
                                           const float* lut) {
    const unsigned int local_out = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int nbase = blockIdx.x * (8 * NO) + local_out * NO;
    const unsigned int k16_count = K / 16, k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    float acc[NO][2][8] = {};
    auto dec = [&](uint4 p, float sc, float w[16]) {
        if (BITDEC) { decode16(p, sc, w); return; }
        const unsigned int ws[4] = {p.x, p.y, p.z, p.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j*4] = lut[ws[j] & 0xffu] * sc; w[j*4+1] = lut[(ws[j] >> 8) & 0xffu] * sc;
            w[j*4+2] = lut[(ws[j] >> 16) & 0xffu] * sc; w[j*4+3] = lut[ws[j] >> 24] * sc;
        }
    };
    auto ldw = [&](unsigned int o, unsigned int k16, uint4& p, float& sc) {
        const unsigned int n = nbase + o;
        if (n < N && k16 < k16_count) {
            sc = block_scale[(n / FP8_BLOCK) * k_blocks + (k16 * 16) / FP8_BLOCK];
            p = reinterpret_cast<const uint4*>(B + (unsigned long long)n * K)[k16];
        } else { sc = 0.f; p = make_uint4(0, 0, 0, 0); }
    };
    uint4 pn[NO][2]; float sn[NO][2];
    if (PREFETCH) {
        #pragma unroll
        for (int o = 0; o < NO; ++o) { ldw(o, lane, pn[o][0], sn[o][0]); ldw(o, lane + 32, pn[o][1], sn[o][1]); }
    }
    for (unsigned int k16 = lane; k16 < k16_count; k16 += 64) {
        uint4 pc[NO][2]; float sc[NO][2];
        #pragma unroll
        for (int o = 0; o < NO; ++o) {
            if (PREFETCH) { pc[o][0] = pn[o][0]; sc[o][0] = sn[o][0]; pc[o][1] = pn[o][1]; sc[o][1] = sn[o][1]; }
            else { ldw(o, k16, pc[o][0], sc[o][0]); ldw(o, k16 + 32, pc[o][1], sc[o][1]); }
        }
        if (PREFETCH) {
            #pragma unroll
            for (int o = 0; o < NO; ++o) { ldw(o, k16 + 64, pn[o][0], sn[o][0]); ldw(o, k16 + 96, pn[o][1], sn[o][1]); }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int kk = k16 + half * 32;
            if (half == 1 && kk >= k16_count) break;
            float w[NO][16];
            #pragma unroll
            for (int o = 0; o < NO; ++o) dec(pc[o][half], sc[o][half], w[o]);
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                if (row >= M) continue;
                const __nv_bfloat16* arow = A + (unsigned long long)row * K;
                const uint4 lo = reinterpret_cast<const uint4*>(arow)[kk * 2];
                const uint4 hi = reinterpret_cast<const uint4*>(arow)[kk * 2 + 1];
                const unsigned int aw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const float a_lo = atlas_bf16_bits_to_f32(aw[pair] & 0xffffu);
                    const float a_hi = atlas_bf16_bits_to_f32(aw[pair] >> 16);
                    #pragma unroll
                    for (int o = 0; o < NO; ++o) acc[o][half][row] += a_lo * w[o][pair * 2] + a_hi * w[o][pair * 2 + 1];
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        if (row >= M) continue;
        #pragma unroll
        for (int o = 0; o < NO; ++o) {
            float x0 = acc[o][0][row], x1 = acc[o][1][row];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                x0 += __shfl_down_sync(0xffffffffu, x0, off);
                x1 += __shfl_down_sync(0xffffffffu, x1, off);
            }
            const unsigned int n = nbase + o;
            if (lane == 0 && n < N) C[(unsigned long long)row * N + n] = __float2bfloat16(x0 + x1);
        }
    }
}

#define LUT_PROLOGUE __shared__ float lut[256]; lut[threadIdx.x] = atlas_e4m3_to_f32((unsigned char)threadIdx.x); __syncthreads();
extern "C" __global__ void __launch_bounds__(256) cand_n2_bit(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { gemv8_body<2, true, false>(A, B, s, C, M, N, K, nullptr); }
extern "C" __global__ void __launch_bounds__(256) cand_n2_pf(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { LUT_PROLOGUE gemv8_body<2, false, true>(A, B, s, C, M, N, K, lut); }
extern "C" __global__ void __launch_bounds__(256) cand_n2_bitpf(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { gemv8_body<2, true, true>(A, B, s, C, M, N, K, nullptr); }
extern "C" __global__ void __launch_bounds__(256) cand_n2_rw(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { LUT_PROLOGUE gemv8_body<2, false, false>(A, B, s, C, M, N, K, lut); }
extern "C" __global__ void __launch_bounds__(256, 2) cand_n2_lb2(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { LUT_PROLOGUE gemv8_body<2, false, false>(A, B, s, C, M, N, K, lut); }
extern "C" __global__ void __launch_bounds__(256, 3) cand_n2_lb3(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { LUT_PROLOGUE gemv8_body<2, false, false>(A, B, s, C, M, N, K, lut); }
extern "C" __global__ void __launch_bounds__(256) cand_n4(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { LUT_PROLOGUE gemv8_body<4, false, false>(A, B, s, C, M, N, K, lut); }
extern "C" __global__ void __launch_bounds__(256) cand_n4_bit(const __nv_bfloat16* A, const unsigned char* B, const float* s, __nv_bfloat16* C, unsigned int M, unsigned int N, unsigned int K) { gemv8_body<4, true, false>(A, B, s, C, M, N, K, nullptr); }

typedef void (*KFn)(const __nv_bfloat16*, const unsigned char*, const float*, __nv_bfloat16*, unsigned int, unsigned int, unsigned int);
struct Cand { const char* name; KFn fn; int outs_per_block; };

int main() {
    // 27B FP8 projections at M = 8 (Qwen3.8-27B: H 5120; GDN qkv 10240 + z 6144; attn q+gate 12288; out/o_proj K 6144)
    const unsigned int shapes[][2] = {{16384, 5120}, {5120, 6144}};
    // Loop #2: DRAM-realistic timing (weights rotated across >256 MB of copies so the 32 MB MALL cannot serve them)
    // and occupancy candidates. Loop #1's bit-decode / prefetch / 4-output arms all lost and are dropped here.
    const Cand cands[] = {{"n2 (shipped)", w8a16_gemv_batch8_dyn_vl2_n2, 16}, {"n2_rw", cand_n2_rw, 16},
                          {"n2_lb2", cand_n2_lb2, 16}, {"n2_lb3", cand_n2_lb3, 16}, {"n2_pf", cand_n2_pf, 16}};
    for (unsigned int M : {8u, 7u, 6u, 5u})   // loop #8: hoist bit-identity at every M the n2 tier serves (MTP verify runs M=5)
    for (auto& sh : shapes) {
        const unsigned int N = sh[0], K = sh[1], kb = (K + 127) / 128, nb = (N + 127) / 128;
        std::vector<unsigned char> hB((size_t)N * K); std::vector<float> hS((size_t)nb * kb); std::vector<unsigned short> hA((size_t)M * K);
        srand(1234);
        for (auto& b : hB) { b = (unsigned char)(rand() & 0xff); if ((b & 0x7f) == 0x7f) b &= 0xfe; }   // no NaN codes
        for (auto& s : hS) s = 0.001f + (rand() % 1000) * 1e-5f;
        for (auto& a : hA) { float f = ((rand() % 2001) - 1000) * 1e-3f; unsigned int u; memcpy(&u, &f, 4); a = (unsigned short)(u >> 16); }
        unsigned char* dB; float* dS; __nv_bfloat16 *dA, *dC;
        const int NROT = (int)((256e6 + hB.size() - 1) / hB.size()) + 1;
        std::vector<unsigned char*> rot(NROT);
        for (int r = 0; r < NROT; ++r) { CHECK(hipMalloc(&rot[r], hB.size())); CHECK(hipMemcpy(rot[r], hB.data(), hB.size(), hipMemcpyHostToDevice)); }
        CHECK(hipMalloc(&dB, hB.size())); CHECK(hipMalloc(&dS, hS.size() * 4)); CHECK(hipMalloc(&dA, hA.size() * 2)); CHECK(hipMalloc(&dC, (size_t)M * N * 2));
        CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(dS, hS.data(), hS.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
        std::vector<unsigned short> ref((size_t)M * N), out((size_t)M * N);
        const double bytes = (double)N * K + hS.size() * 4.0;
        printf("-- N=%u K=%u M=%u (weights %.1f MB)\n", N, K, M, bytes / 1e6);
        for (int ci = 0; ci < (int)(sizeof(cands) / sizeof(cands[0])); ++ci) {
            const Cand& c = cands[ci]; dim3 grid((N + c.outs_per_block - 1) / c.outs_per_block), block(256);
            CHECK(hipMemset(dC, 0, (size_t)M * N * 2));
            hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dB, dS, dC, M, N, K); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(ci == 0 ? ref.data() : out.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost));
            for (int w = 0; w < 5; ++w) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, dB, dS, dC, M, N, K);
            hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
            const int iters = 50; CHECK(hipEventRecord(e0));
            for (int it = 0; it < iters; ++it) hipLaunchKernelGGL(c.fn, grid, block, 0, 0, dA, rot[it % NROT], dS, dC, M, N, K);
            CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            size_t diff = 0; if (ci > 0) for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
            printf("   %-14s %8.1f us  %6.1f GB/s  %s\n", c.name, us, bytes / us / 1e3,
                   ci == 0 ? "(reference)" : (diff == 0 ? "BIT-IDENTICAL" : "DIFFERS"));
            if (ci > 0 && diff) printf("      %zu of %zu outputs differ\n", diff, out.size());
        }
        hipFree(dB); hipFree(dS); hipFree(dA); hipFree(dC); for (auto r : rot) hipFree(r);
    }
    return 0;
}
