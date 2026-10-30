// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next routed-MoE PREFILL on native FP4 tensor cores, at the
// numerics of the engine NVIDIA's FP4-Local card names (eugr/spark-vllm-b12x:
// VLLM_B12X_MOE_FP4_FORCE_A16=1, B12X_W4A16_A4_PREFILL=1,
// VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 -> flashinfer b12x
// `moe/_shared/kernels/w4a16/prefill_a4.py`). Prompt rows contract NVFP4
// activations with the checkpoint's NVFP4 weights, both as stored:
//
//  1. moe_w4a4_quant_input: the token-major BF16 input to NVFP4 once, under
//     ONE global scale for the layer, g1 = 1 / max_e w13 input_scale (b12x's
//     amin of 1 / w13_input_scale); `nvfp4_block_quant.cuh`.
//  2. moe_w4a4_gate_up_silu: per expert, gate and up of a 64-column
//     intermediate slice in one CTA on mma.sync kind::mxf4nvf4 m16n8k64
//     (E2M1 x E2M1, the activation's and the checkpoint's E4M3 block scales);
//     gate = bf16(acc * alpha_gate[e]), up likewise (alpha = weight_scale_2 /
//     g1); act = bf16(bf16(gate * sigmoid(gate)) * up) with the reference's
//     approximate exp / reciprocal and no clamp (the checkpoint declares no
//     swiglu_limit); act to NVFP4 under the expert's g2 = 1 / w2 input_scale.
//  3. moe_w4a4_down: act x down on the same MMA, C = bf16(acc * alpha_down[e])
//     (weight_scale_2 / g2) into the expert-sorted rows of `expert_down_out`.
//
// The weights are read in the checkpoint's own planes -- packed [N, K/2]
// (low nibble = even k) and scales [N, K/16] -- which is the m16n8k64 B
// fragment's order (eight consecutive k per register, four group scales per
// k64), so no prefill copy of the experts exists. The activations use the
// same layout ([rows, K/2] + [rows, K/16]).
//
// ROW INVARIANT: the quantization is per row (16-value groups along K, a
// layer- or expert-constant global scale) and every output is one FP32
// accumulator over the k64 MMAs in increasing k, so a row's bytes do not
// depend on which rows share its launch.
//
// Rows of a remote expert (EP: null tables) are skipped, as in the other
// prefill MoE kernels; the local unpermute never reads them.
//
// MEASURED (GB10 clock-capped, 256 local experts of 512, top-10 lognormal,
// 16000 tokens = 81K local rows; scripts/dev/qwen4exp_moe_w4a4_bench.sh):
//   W4A4 8.36 ms = quant 0.48 + gate_up+silu 4.09 + down 3.79 (96 TFLOP/s)
//   the E4M3 chain it replaces (moe_prefill_q38 W2, N-major) 14.71 ms
//   the BF16 TC chain (qwen4exp_moe_tcp) 20.25 ms
// BK = 64 with 3 / 4 / 5 stages measured 11.0 / 10.8 / 10.5 ms.

#include <cuda_bf16.h>
#include "../../common/nvfp4_block_quant.cuh"

#define W4_BM 128                       // rows per CTA (2 warp rows x 4 m16 tiles)
#ifndef W4_BK
#define W4_BK 128                       // K per pipeline stage (two k64 MMAs)
#endif
#ifndef W4_STAGES
#define W4_STAGES 2                     // cp.async ring depth
#endif
#define W4_ROW (W4_BK / 2)              // packed bytes of one row per stage
#define W4_CHUNKS (W4_ROW / 16)         // 16-byte chunks of one row per stage
#define W4_SROW (W4_BK / 16)            // scale bytes of one row per stage
static_assert(W4_BK == 64 || W4_BK == 128, "W4_BK: 64 or 128");

struct W4Smem {
    unsigned char a[W4_STAGES][W4_BM][W4_ROW];   // activation rows (chunks swizzled)
    unsigned char b[W4_STAGES][128][W4_ROW];     // weight columns (chunks swizzled)
    unsigned char as[W4_STAGES][W4_BM][W4_SROW];
    unsigned char bs[W4_STAGES][128][W4_SROW];
    int tok[W4_BM];
};

// Physical 16-byte chunk of logical chunk `j` in row `r`: the eight rows a
// fragment load spans land on eight distinct bank groups.
__device__ __forceinline__ unsigned int w4_swz(unsigned int r, unsigned int j) {
    return W4_CHUNKS == 4 ? (j ^ ((r >> 1) & 3u)) : (j ^ ((r >> 2) & 1u));
}

__device__ __forceinline__ void w4_cp16(void* dst, const void* src, bool pred) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(pred ? 16u : 0u));
}
__device__ __forceinline__ void w4_cps(void* dst, const void* src, bool pred) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(d), "l"(src), "n"(W4_SROW),
                 "r"(pred ? (unsigned int)W4_SROW : 0u));
}

__device__ __forceinline__ void w4_mma(float (&c)[4], unsigned int a0, unsigned int a1, unsigned int a2,
                                       unsigned int a3, unsigned int b0, unsigned int b1, unsigned int sfa,
                                       unsigned int sfb) {
    const unsigned short z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
          "h"(z));
}

// The reference's BF16 rounding (cvt.rn.satfinite.bf16x2.f32), back to float.
__device__ __forceinline__ float w4_bf16(float x) {
    unsigned int p;
    asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(0.0f), "f"(x));
    return __uint_as_float(p << 16);
}
__device__ __forceinline__ unsigned int w4_bf16x2(float lo, float hi) {
    unsigned int p;
    asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(hi), "f"(lo));
    return p;
}

// silu(g) * u as the reference's epilogue forms it from BF16 g and u:
// bf16(bf16(g * rcp.approx(1 + ex2.approx(-g * log2 e))) * u).
__device__ __forceinline__ float w4_silu_mul(float g, float u) {
    float e, r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(-g, 1.4426950408889634f)));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fadd_rn(1.0f, e)));
    return w4_bf16(__fmul_rn(w4_bf16(__fmul_rn(g, r)), u));
}

// n-tile (of 16, 8 columns each) j (0..3) of warp column wn (0..3): two of
// the first 64 columns and the same two of the second, so a gate value and
// its up value share a thread and a warp's gate columns are one 16-group.
__device__ __forceinline__ unsigned int w4_nt(unsigned int wn, unsigned int j) {
    return (j < 2u ? 0u : 8u) + 2u * wn + (j & 1u);
}

// One 128-row x 128-column tile over K: rows tok[r] of (A, As) -- [*, K/2]
// packed and [*, K/16] scales -- against columns n0 .. n0+63 of (B0, S0)
// and n1 .. n1+63 of (B1, S1), each [N, K/2] / [N, K/16]. Warp (wm, wn) =
// (warp & 1, warp >> 1) owns rows wm*64 .. +63 (four m16 tiles) and n-tiles
// w4_nt(wn, 0..3); acc[mt * 4 + j].
__device__ __forceinline__ void w4_tile(
    W4Smem& sm, float (&acc)[16][4],
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As,
    unsigned int K, unsigned int rows_left,
    const unsigned char* B0, const unsigned char* S0, unsigned int n0,
    const unsigned char* B1, const unsigned char* S1, unsigned int n1
) {
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int q = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    const unsigned int steps = K / W4_BK, kh = K / 2, ks = K / 16;

    auto issue = [&](unsigned int s, unsigned int buf) {
        if (s < steps) {
            const unsigned int kb = s * W4_BK;
            #pragma unroll
            for (unsigned int i = tid; i < W4_BM * W4_CHUNKS; i += 256) {
                const unsigned int r = i / W4_CHUNKS, j = i % W4_CHUNKS;
                const bool valid = r < rows_left;
                const unsigned char* src = A + (size_t)(valid ? sm.tok[r] : 0) * kh + kb / 2 + 16 * j;
                w4_cp16(&sm.a[buf][r][16 * w4_swz(r, j)], src, valid);
            }
            #pragma unroll
            for (unsigned int i = tid; i < 128 * W4_CHUNKS; i += 256) {
                const unsigned int col = i / W4_CHUNKS, j = i % W4_CHUNKS;
                const unsigned char* src = (col < 64 ? B0 + (size_t)(n0 + col) * kh : B1 + (size_t)(n1 + col - 64) * kh);
                w4_cp16(&sm.b[buf][col][16 * w4_swz(col, j)], src + kb / 2 + 16 * j, true);
            }
            if (tid < W4_BM) {
                const bool valid = tid < rows_left;
                w4_cps(&sm.as[buf][tid][0], As + (size_t)(valid ? sm.tok[tid] : 0) * ks + kb / 16, valid);
            } else {
                const unsigned int col = tid - W4_BM;
                const unsigned char* src = col < 64 ? S0 + (size_t)(n0 + col) * ks : S1 + (size_t)(n1 + col - 64) * ks;
                w4_cps(&sm.bs[buf][col][0], src + kb / 16, true);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    #pragma unroll
    for (unsigned int i = 0; i + 1 < W4_STAGES; ++i) issue(i, i);
    for (unsigned int s = 0; s < steps; ++s) {
        const unsigned int buf = s % W4_STAGES;
        issue(s + W4_STAGES - 1, (s + W4_STAGES - 1) % W4_STAGES);
        asm volatile("cp.async.wait_group %0;" ::"n"(W4_STAGES - 1));
        __syncthreads();
        #pragma unroll
        for (unsigned int kk = 0; kk < W4_BK / 64; ++kk) {
            const unsigned int j0 = 2 * kk, j1 = j0 + 1;
            unsigned int b[4][2], sfb[4];
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                const unsigned int nc = w4_nt(wn, j) * 8 + q;
                b[j][0] = *(const unsigned int*)&sm.b[buf][nc][16 * w4_swz(nc, j0) + 4 * c];
                b[j][1] = *(const unsigned int*)&sm.b[buf][nc][16 * w4_swz(nc, j1) + 4 * c];
                sfb[j] = *(const unsigned int*)&sm.bs[buf][nc][4 * kk];
            }
            #pragma unroll
            for (unsigned int mt = 0; mt < 4; ++mt) {
                const unsigned int ra = wm * 64 + mt * 16 + q, rc = ra + 8;
                const unsigned int a0 = *(const unsigned int*)&sm.a[buf][ra][16 * w4_swz(ra, j0) + 4 * c];
                const unsigned int a1 = *(const unsigned int*)&sm.a[buf][rc][16 * w4_swz(rc, j0) + 4 * c];
                const unsigned int a2 = *(const unsigned int*)&sm.a[buf][ra][16 * w4_swz(ra, j1) + 4 * c];
                const unsigned int a3 = *(const unsigned int*)&sm.a[buf][rc][16 * w4_swz(rc, j1) + 4 * c];
                // Scale lanes: rows 0-7 on lanes 4i, rows 8-15 on lanes 4i + 1.
                const unsigned int sfa = *(const unsigned int*)&sm.as[buf][ra + 8 * (c & 1u)][4 * kk];
                #pragma unroll
                for (unsigned int j = 0; j < 4; ++j)
                    w4_mma(acc[mt * 4 + j], a0, a1, a2, a3, b[j][0], b[j][1], sfa, sfb[j]);
            }
        }
        __syncthreads();   // every warp done with `buf` before it is refilled
    }
    asm volatile("cp.async.wait_group 0;");
}

// Gather rows [cta_m, cta_m + 128) of an expert: token ids through
// `sorted_token_ids`, or the sorted rows themselves when it is null.
__device__ __forceinline__ void w4_rows(W4Smem& sm, const int* __restrict__ sorted_token_ids,
                                        unsigned int row0, unsigned int rows_left) {
    __syncthreads();   // the previous tile's readers of sm.tok are done
    if (threadIdx.x < W4_BM) {
        const unsigned int r = threadIdx.x;
        sm.tok[r] = r < rows_left ? (sorted_token_ids ? sorted_token_ids[row0 + r] : (int)(row0 + r)) : 0;
    }
    __syncthreads();
}

// ── 1. BF16 -> NVFP4, one 16-value group a thread ──
// Grid: ceil(groups / 256), Block 256; groups = rows * K / 16.
extern "C" __global__ void moe_w4a4_quant_input(
    const __nv_bfloat16* __restrict__ x,     // [rows, K]
    unsigned char* __restrict__ packed,      // [rows, K/2]
    unsigned char* __restrict__ scale,       // [rows, K/16]
    const float g,                           // global scale
    const unsigned int groups
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const uint4* src = reinterpret_cast<const uint4*>(x) + 2 * (size_t)i;
    const uint4 w0 = src[0], w1 = src[1];
    const unsigned int w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    float v[16];
    float amax = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[2 * k] = __uint_as_float(w[k] << 16);
        v[2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * k]), fabsf(v[2 * k + 1])));
    }
    Nvfp4qGrid grid;
    scale[i] = (unsigned char)nvfp4q_block_scale(amax, g, &grid);
    unsigned int lo = 0, hi = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        lo |= nvfp4q_pack2(v[2 * k], v[2 * k + 1], grid) << (8 * k);
        hi |= nvfp4q_pack2(v[8 + 2 * k], v[9 + 2 * k], grid) << (8 * k);
    }
    reinterpret_cast<uint2*>(packed)[i] = make_uint2(lo, hi);
}

// ── 2. gate + up + SiLU*mul -> NVFP4 ──
// Grid: (N / 64, row tiles (strided), num_experts), Block 256.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_gate_up_silu(
    const unsigned char* __restrict__ A, const unsigned char* __restrict__ As,   // [tokens, K/2], [tokens, K/16]
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_alpha,                                       // weight_scale_2 / g1
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_alpha,
    const float* __restrict__ act_gscale,                                       // g2[e]
    unsigned char* __restrict__ act, unsigned char* __restrict__ act_scale,     // [rows, N/2], [rows, N/16]
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const unsigned int num_experts,
    const unsigned int N,
    const unsigned int K
) {
    __shared__ __align__(16) W4Smem sm;
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e], M = expert_offsets[e + 1] - m_start;
    const unsigned char* Bg = (const unsigned char*)gate_packed_ptrs[e];
    const unsigned char* Bu = (const unsigned char*)up_packed_ptrs[e];
    if (M <= 0 || Bg == nullptr || Bu == nullptr) return;
    const unsigned char* Sg = (const unsigned char*)gate_scale_ptrs[e];
    const unsigned char* Su = (const unsigned char*)up_scale_ptrs[e];
    const float ag = gate_alpha[e], au = up_alpha[e], g2 = act_gscale[e];
    const unsigned int n0 = blockIdx.x * 64;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int q = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
    for (unsigned int m0 = blockIdx.y * W4_BM; m0 < (unsigned int)M; m0 += gridDim.y * W4_BM) {
        const unsigned int rows_left = (unsigned int)M - m0;
        w4_rows(sm, sorted_token_ids, (unsigned int)m_start + m0, rows_left);
        float acc[16][4];
        w4_tile(sm, acc, A, As, K, rows_left, Bg, Sg, n0, Bu, Su, n0);
        // This warp's gate columns n0 + 16 wn .. +15 are one NVFP4 group; the
        // quad's four lanes hold its 16 values of a row (j = 0, 1 x two).
        const unsigned int grp = (n0 + 16 * wn) / 16;
        #pragma unroll
        for (unsigned int mt = 0; mt < 4; ++mt) {
            #pragma unroll
            for (unsigned int h = 0; h < 2; ++h) {
                float v[4];
                float amax = 0.0f;
                #pragma unroll
                for (unsigned int j = 0; j < 2; ++j) {
                    #pragma unroll
                    for (unsigned int t = 0; t < 2; ++t) {
                        const float gv = w4_bf16(__fmul_rn(acc[mt * 4 + j][2 * h + t], ag));
                        const float uv = w4_bf16(__fmul_rn(acc[mt * 4 + j + 2][2 * h + t], au));
                        v[2 * j + t] = w4_silu_mul(gv, uv);
                        amax = fmaxf(amax, fabsf(v[2 * j + t]));
                    }
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                Nvfp4qGrid gridq;
                const unsigned int sf = nvfp4q_block_scale(amax, g2, &gridq);
                // Lane c holds group bytes c (columns 2c, 2c+1) and 4 + c.
                const unsigned int mine = nvfp4q_pack2(v[0], v[1], gridq) | (nvfp4q_pack2(v[2], v[3], gridq) << 16);
                const unsigned int pair = mine | (__shfl_xor_sync(0xffffffffu, mine, 1) << 8);   // c even: c, c+1
                const unsigned int other = __shfl_xor_sync(0xffffffffu, pair, 2);                // from c + 2
                const unsigned int lr = wm * 64 + mt * 16 + q + 8 * h;
                if (c == 0 && lr < rows_left) {
                    const size_t row = (size_t)m_start + m0 + lr;
                    const unsigned int lo = (pair & 0xFFFFu) | (other << 16);
                    const unsigned int hi = (pair >> 16) | (other & 0xFFFF0000u);
                    *reinterpret_cast<uint2*>(act + row * (N / 2) + grp * 8) = make_uint2(lo, hi);
                    act_scale[row * (N / 16) + grp] = (unsigned char)sf;
                }
            }
        }
    }
}

// ── 3. down ──
// Grid: (N / 128, row tiles (strided), num_experts), Block 256.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w4a4_down(
    const unsigned char* __restrict__ act, const unsigned char* __restrict__ act_scale,   // [rows, K/2], [rows, K/16]
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ alpha,                                                    // weight_scale_2 / g2[e]
    __nv_bfloat16* __restrict__ C,                                                      // [rows, N]
    const int* __restrict__ expert_offsets,
    const unsigned int num_experts,
    const unsigned int N,
    const unsigned int K
) {
    __shared__ __align__(16) W4Smem sm;
    const unsigned int e = blockIdx.z;
    if (e >= num_experts) return;
    const int m_start = expert_offsets[e], M = expert_offsets[e + 1] - m_start;
    const unsigned char* B = (const unsigned char*)packed_ptrs[e];
    if (M <= 0 || B == nullptr) return;
    const unsigned char* S = (const unsigned char*)scale_ptrs[e];
    const float a = alpha[e];
    const unsigned int n0 = blockIdx.x * 128;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int q = lane >> 2, c = lane & 3u, wm = warp & 1u, wn = warp >> 1;
    for (unsigned int m0 = blockIdx.y * W4_BM; m0 < (unsigned int)M; m0 += gridDim.y * W4_BM) {
        const unsigned int rows_left = (unsigned int)M - m0;
        w4_rows(sm, nullptr, (unsigned int)m_start + m0, rows_left);
        float acc[16][4];
        w4_tile(sm, acc, act, act_scale, K, rows_left, B, S, n0, B, S, n0 + 64);
        #pragma unroll
        for (unsigned int i = 0; i < 16; ++i) {
            const unsigned int col = n0 + w4_nt(wn, i & 3u) * 8 + 2 * c;
            #pragma unroll
            for (unsigned int h = 0; h < 2; ++h) {
                const unsigned int lr = wm * 64 + (i >> 2) * 16 + q + 8 * h;
                if (lr >= rows_left) continue;
                const size_t row = (size_t)m_start + m0 + lr;
                *reinterpret_cast<unsigned int*>(C + row * N + col) =
                    w4_bf16x2(__fmul_rn(acc[i][2 * h], a), __fmul_rn(acc[i][2 * h + 1], a));
            }
        }
    }
}
