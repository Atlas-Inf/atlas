// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next (qwen4_exp) mHC collapse, down and finish on TENSOR
// CORES (ATLAS_QWEN4EXP_HC_MMA=1): the `hc_pre_down_vec*` / `hc_pre_finish_vec*`
// pair of hyper_connection.cu as two bf16 mma.sync GEMMs, for 1..32 rows.
//
// WHY. At ~32 rows the FP32 kernels cost ~55 us each a site (vLLM's cuBLAS
// GEMMs ~37). ncu: ~8.6M warp instructions each (two FP32 ops a multiply-
// accumulate under --fmad=false and the bit identity, plus widening and
// operand loads) at ~2 warps a scheduler, stalled on L1TEX; the mma pair
// issues 1.25M + 0.87M. Tensor cores leave only the weight stream.
//
// NUMERICS (contract (b): a new, row-invariant baseline that serial decode
// takes too). NOT bit-identical to the FP32 kernels -- an mma sums its 16
// products in hardware order -- but every output of every row is the same
// instruction sequence whatever the row count or the row's position:
//
//   * activations (`normed` FP32, `low` FP32) enter as hi + lo bf16 pairs,
//     hi = bf16(x), lo = bf16(x - hi) (~16 significant bits), the weights as
//     their exact bf16; hi and lo products go to separate FP32 accumulators,
//     summed hi + lo once at the end;
//   * k is visited in a fixed permuted order; the K split (warps of a CTA,
//     then the CTAs of a cluster) is reduced in fixed rank order through
//     shared / distributed shared memory: no atomics, no global partials;
//   * a row's token is its own mma column (n), and an mma column never sees
//     another column, so a row's bits do not depend on its neighbours. Rows
//     past T read a clamped copy of row T-1 and are never stored.
//   * the epilogues are the FP32 kernels' formulas: low = silu(v / hc),
//     inj = 2 sigmoid(v / hc), y = bf16((0 + p0 + p1 + p2 + p3) * 0.25),
//     p_s = sigmoid(up_s) * normed_s.
//
// scripts/dev/qwen4exp_hc_mma_bench.cu proves the row invariance (every row
// of every T = 1..32 batch byte-equal to that row alone), measures the error
// against an FP64 reference beside the FP32 kernels' error, and times both.
// Error vs FP64 at T = 32: low 5.0e-6 (FP32 kernels 4.8e-7), y off the
// correctly rounded bf16 in 0.054% of outputs (FP32 0.009%), worst 0.90 ulp
// (FP32 1.84). us per launch, weights cold, a GB10:
//
//     T                 1     4     8    16    24    32
//     down   FP32    31.8  34.3  42.5  49.5  64.1  64.2
//            mma     33.2  33.2  33.7  34.2  35.1  36.2
//     finish FP32    33.1  38.9  49.1  47.9  47.9  50.9
//            mma     33.2  33.4  33.8  35.5  36.2  38.3
//
// against ~27 us for a bare 6.55 MB streaming read. L2-resident the mma pair
// takes 10 + 10 us at T = 1 and 20 + 18 at T = 32: what is left is the DRAM
// stream, not the math. (Swept: CTA tiles, K split 2..16, rings 2..20, L2
// hints, an L2 prefetch of up_w from the down kernel -- it moves time from
// the finish to the down -- and DW 16/32/64; the defaults below won.)
//
// K PERMUTATION. mma.m16n8k16 gives lane (g = lane / 4, q = lane % 4) the A
// k-slots {2q, 2q+1, 2q+8, 2q+9} of rows g and g+8 and the same B k-slots of
// column g. Any k <-> slot map shared by A and B is the same dot product, so
// lane q of the down GEMM takes physical k = 8q .. 8q+7 of a 32-k chunk with
// ONE 16-byte load (two k-steps), and lane q of the up GEMM takes the four
// rank rows 4q .. 4q+3 of a 16-row k-step.
//
// DOWN: [rows = rank (+ hc injection), K = 4H] x [K, T] -> [rows, T]. Weights
// are the A operand straight from DRAM (row-major, k contiguous), `normed`
// the B operand (16-byte FP32 loads, L1-shared by the CTA's WM row tiles).
// Grid (ceil(m16 tiles / WM), CL), cluster (1, CL, 1); CTA = WM x WK warps.
//
// UP + MIX: [4H, rank] x [rank, T]. `up_w` is stored [rank, 4H] (the decode
// layout), so A's k pairs are two rank rows: a lane loads DW/8 consecutive
// output dims of each of its four rank rows and byte-permutes pairs; m16 tile
// j of the warp is the lane's dims {2j, 2j+1} (rows g, g+8). `low` is staged
// once per CTA as ready-made hi/lo B fragments. CTA = 4 warps (one per
// stream) x WK, DW output dims; the stream mean runs in the CTA.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"

// The host mirrors the HCM_DN_* / HCM_FN_* shape (grids, cluster dims,
// shared memory) in `layers/ops/hyper_connection_lowrank_mma.rs`: change
// both, never these alone through -D.
#ifndef HCM_DN_WM
#define HCM_DN_WM 3u  // m16 weight tiles a CTA (rank + hc = 324 rows = 21 tiles = 7 x 3)
#endif
#ifndef HCM_DN_WK
#define HCM_DN_WK 1u  // warps splitting the CTA's K slice
#endif
#ifndef HCM_DN_CL
#define HCM_DN_CL 8u  // CTAs of a cluster splitting K (portable max)
#endif
#ifndef HCM_DN_U
#define HCM_DN_U 4u   // weight ring, 32-k chunks
#endif
#ifndef HCM_DN_DN
#define HCM_DN_DN 2u  // normed ring, 32-k chunks
#endif
#ifndef HCM_FN_DW
#define HCM_FN_DW 32u // output dims a warp (DW / 16 m16 tiles)
#endif
#ifndef HCM_FN_WK
#define HCM_FN_WK 1u  // warps splitting rank
#endif
#ifndef HCM_FN_U
#define HCM_FN_U 5u   // up_w ring, 16-row k-steps
#endif
#define HCM_MAX_T 32u

__device__ __forceinline__ float hcm_silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float hcm_sigmoid(float v) { return 1.0f / (1.0f + __expf(-v)); }

// d += a x b, m16n8k16, bf16 in, FP32 accumulate.
__device__ __forceinline__ void hcm_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// (x0, x1) -> packed bf16 hi pair and lo pair; element 0 in the low half.
__device__ __forceinline__ void hcm_split(float x0, float x1, unsigned& hi, unsigned& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    const float2 hf = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(x0 - hf.x, x1 - hf.y);
    hi = *reinterpret_cast<const unsigned*>(&h);
    lo = *reinterpret_cast<const unsigned*>(&l);
}

// W words (2W streamed bf16): read-only, no L1 allocation, L2 prefetch hint.
#ifndef HCM_LDW_HINT
#define HCM_LDW_HINT ".L2::256B"
#endif
template <unsigned W>
__device__ __forceinline__ void hcm_ldw(const __nv_bfloat16* p, unsigned (&w)[W]) {
    if constexpr (W == 1) {
        asm("ld.global.nc.L1::no_allocate" HCM_LDW_HINT ".u32 %0, [%1];" : "=r"(w[0]) : "l"(p));
    } else if constexpr (W == 2) {
        asm("ld.global.nc.L1::no_allocate" HCM_LDW_HINT ".v2.u32 {%0, %1}, [%2];" : "=r"(w[0]), "=r"(w[1]) : "l"(p));
    } else {
        asm("ld.global.nc.L1::no_allocate" HCM_LDW_HINT ".v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p));
    }
}

// The predecessor's outputs (`normed`, `low`), as volatile asm loads: a plain
// read through a `const __restrict__` pointer is an invariant load the
// compiler may move above `griddepcontrol.wait`, which under PDL reads the
// previous kernel's buffer before it is written. (Measured while the weight
// ring was issued ahead of the wait: the normed loads moved with it and the
// row-invariance check failed under PDL. The wait is now first, which also
// keeps the PDL source contract of cuda_backend/gpu_impl/pdl.rs; issuing the
// ring ahead of it measured ~2 us a site better.)
__device__ __forceinline__ float4 hcm_ld4(const float* p) {
    float4 v;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float2 hcm_ld2(const float* p) {
    float2 v;
    asm volatile("ld.global.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

// ── down + injection rows ────────────────────────────────────────────────
// WM (m16 weight tiles a CTA) and DN (normed ring depth) are launch
// geometry: a row's chain and the K reduction order do not depend on them.
template <unsigned NT, unsigned WM = HCM_DN_WM, unsigned DN = HCM_DN_DN>
__device__ __forceinline__ void hcm_down(
    const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ inject_w, float* __restrict__ low_out,
    float* __restrict__ inj_out, const unsigned K, const unsigned hc, const unsigned rank,
    const unsigned T, float* s_red, float* s_wk
) {
    constexpr unsigned WK = HCM_DN_WK, CL = HCM_DN_CL, U = HCM_DN_U;
    constexpr unsigned E = WM * 16u * NT * 8u;  // CTA partial: [WM*16 rows][NT*8 tokens]
    static_assert(DN >= 1 && U % DN == 0, "normed ring must tile the weight ring");

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const unsigned crank = cluster.block_rank();
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned g = lane >> 2, q = lane & 3u;
    const unsigned wm = warp % WM, wk = warp / WM;
    const unsigned rows = rank + (inject_w != nullptr ? hc : 0u);
    const unsigned rcta = blockIdx.x * WM * 16u;
    const unsigned row0 = rcta + wm * 16u;
    const unsigned kw = K / (CL * WK);
    const unsigned kbeg = (crank * WK + wk) * kw + 8u * q;
    auto wrow = [&](unsigned r) -> const __nv_bfloat16* {
        return r < rank ? down_w + (size_t)r * K
             : r < rows ? inject_w + (size_t)(r - rank) * K
                        : down_w;  // padding row: read, never stored
    };
    const __nv_bfloat16* wa = wrow(row0 + g) + kbeg;
    const __nv_bfloat16* wb = wrow(row0 + g + 8u) + kbeg;
    const float* nx[NT];
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) nx[nt] = normed + (size_t)min(nt * 8u + g, T - 1u) * K + kbeg;
    const unsigned nc = kw / 32u;

    float ah[NT][4], al[NT][4];
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) ah[nt][c] = al[nt][c] = 0.0f;
    }
    unsigned wr[U][2][4];
    float4 nr[DN][NT][2];
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) {
        hcm_ldw<4>(wa + (size_t)u * 32u, wr[u][0]);
        hcm_ldw<4>(wb + (size_t)u * 32u, wr[u][1]);
    }
    #pragma unroll
    for (unsigned d = 0; d < DN; ++d) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            nr[d][nt][0] = hcm_ld4(nx[nt] + d * 32u);
            nr[d][nt][1] = hcm_ld4(nx[nt] + d * 32u + 4u);
        }
    }
    auto step = [&](const unsigned u, const unsigned c, const bool rw, const bool rn) {
        unsigned a0[4] = {wr[u][0][0], wr[u][1][0], wr[u][0][1], wr[u][1][1]};
        unsigned a1[4] = {wr[u][0][2], wr[u][1][2], wr[u][0][3], wr[u][1][3]};
        if (rw) {
            hcm_ldw<4>(wa + (size_t)(c + U) * 32u, wr[u][0]);
            hcm_ldw<4>(wb + (size_t)(c + U) * 32u, wr[u][1]);
        }
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            const float4 f0 = nr[u % DN][nt][0], f1 = nr[u % DN][nt][1];
            if (rn) {
                nr[u % DN][nt][0] = hcm_ld4(nx[nt] + (c + DN) * 32u);
                nr[u % DN][nt][1] = hcm_ld4(nx[nt] + (c + DN) * 32u + 4u);
            }
            unsigned h0, l0, h1, l1;
            hcm_split(f0.x, f0.y, h0, l0);
            hcm_split(f0.z, f0.w, h1, l1);
            hcm_mma(ah[nt], a0, h0, h1);
            hcm_mma(al[nt], a0, l0, l1);
            hcm_split(f1.x, f1.y, h0, l0);
            hcm_split(f1.z, f1.w, h1, l1);
            hcm_mma(ah[nt], a1, h0, h1);
            hcm_mma(al[nt], a1, l0, l1);
        }
    };
    unsigned c0 = 0;
    for (; c0 + U < nc; c0 += U) {
        #pragma unroll
        for (unsigned u = 0; u < U; ++u) step(u, c0 + u, true, true);
    }
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) step(u, c0 + u, false, u + DN < U);

    // Fixed-order K reduction: hi + lo, then warps wk = 0, 1, .. of the CTA,
    // then cluster ranks 0, 1, ...
    auto at = [&](unsigned nt, unsigned c) {  // this lane's (row, token) in a partial
        return (wm * 16u + g + (c >> 1) * 8u) * (NT * 8u) + nt * 8u + 2u * q + (c & 1u);
    };
    float v[NT][4];
    #pragma unroll
    for (unsigned nt = 0; nt < NT; ++nt) {
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) v[nt][c] = ah[nt][c] + al[nt][c];
    }
    if (WK > 1 && wk > 0) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) s_wk[(wk - 1u) * E + at(nt, c)] = v[nt][c];
        }
    }
    if (WK > 1) __syncthreads();
    if (wk == 0) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
                float x = v[nt][c];
                for (unsigned k = 1; k < WK; ++k) x += s_wk[(k - 1u) * E + at(nt, c)];
                s_red[at(nt, c)] = x;
            }
        }
    }
    cluster.sync();
    const float inv_hc = 1.0f / (float)hc;
    for (unsigned e = crank * blockDim.x + threadIdx.x; e < E; e += CL * blockDim.x) {
        float x = cluster.map_shared_rank(s_red, 0)[e];
        #pragma unroll
        for (unsigned k = 1; k < CL; ++k) x += cluster.map_shared_rank(s_red, k)[e];
        const unsigned r = rcta + e / (NT * 8u), t = e % (NT * 8u);
        if (t < T && r < rows) {
            if (r < rank) low_out[(size_t)t * rank + r] = hcm_silu(x * inv_hc);
            else inj_out[(size_t)t * hc + (r - rank)] = 2.0f * hcm_sigmoid(x * inv_hc);
        }
    }
    cluster.sync();  // peers' shared memory stays live until every read is done
}

// Grid (ceil(ceil(rows / 16) / HCM_DN_WM), HCM_DN_CL), block 32 * WM * WK.
// Host checks K % (32 * CL * WK * U) == 0 and T in 1..32.
extern "C" __global__ void __cluster_dims__(1, HCM_DN_CL, 1)
__launch_bounds__(32 * HCM_DN_WM * HCM_DN_WK) hc_mma_down(
    const float* __restrict__ normed,          // [>= T, hc*H] FP32
    const __nv_bfloat16* __restrict__ down_w,  // [rank, hc*H]
    const __nv_bfloat16* __restrict__ inject_w,// [hc, hc*H] or null (the head)
    float* __restrict__ low_out,               // [T, rank]
    float* __restrict__ inj_out,               // [T, hc]
    const unsigned hidden_size, const unsigned hc, const unsigned rank, const unsigned T
) {
    atlas_pdl_enter();
    __shared__ float s_red[HCM_DN_WM * 16u * HCM_MAX_T];
    __shared__ float s_wk[(HCM_DN_WK > 1 ? HCM_DN_WK - 1 : 1) * HCM_DN_WM * 16u * HCM_MAX_T];
    const unsigned K = hc * hidden_size;
#define HCM_DOWN(NT_) hcm_down<NT_>(normed, down_w, inject_w, low_out, inj_out, K, hc, rank, T, s_red, s_wk)
    switch ((T + 7u) / 8u) {
    case 1: HCM_DOWN(1); break;
    case 2: HCM_DOWN(2); break;
    case 3: HCM_DOWN(3); break;
    default: HCM_DOWN(4); break;
    }
#undef HCM_DOWN
}

// ── up + gate + stream mean (hc == 4) ─────────────────────────────────────
// DIRECT: each warp builds its `low` B fragments from global memory at each
// k-step instead of staging them all in shared memory first -- the same
// values (the same load and split per fragment), so the same bytes; the
// prefill rows twin takes it so a 64-row group needs no 80 KB of staging.
template <unsigned NT, bool DIRECT = false>
__device__ __forceinline__ void hcm_finish(
    const float* __restrict__ normed, const float* __restrict__ low,
    const __nv_bfloat16* __restrict__ up_w, __nv_bfloat16* __restrict__ y_out,
    const unsigned H, const unsigned rank, const unsigned T, unsigned char* smem
) {
    constexpr unsigned DW = HCM_FN_DW, MT = DW / 16u, WK = HCM_FN_WK, U = HCM_FN_U;
    constexpr unsigned TOK = NT * 8u;
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned g = lane >> 2, q = lane & 3u;
    const unsigned s = warp & 3u, wk = warp >> 2;
    const unsigned hc_dim = 4u * H;
    const unsigned d0 = blockIdx.x * DW;
    const unsigned nks = rank / 16u / WK;
    const unsigned ks0 = wk * nks;
    const __nv_bfloat16* ub = up_w + (size_t)(16u * ks0 + 4u * q) * hc_dim + s * H + d0 + 2u * MT * g;

    unsigned ur[U][4][MT];
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) {
        #pragma unroll
        for (unsigned e = 0; e < 4; ++e) hcm_ldw<MT>(ub + (size_t)(16u * u + e) * hc_dim, ur[u][e]);
    }

    // `low` as B fragments: s_b[(ks * NT + nt) * 32 + lane] = {hi(r0,r1),
    // hi(r2,r3), lo(r0,r1), lo(r2,r3)} of token nt*8 + lane/4, rank rows
    // r = 16 ks + 4 (lane % 4) + 0..3.
    uint4* s_b = reinterpret_cast<uint4*>(smem);
    auto frag = [&](unsigned ks, unsigned nt, unsigned l) {
        const unsigned t = min(nt * 8u + (l >> 2), T - 1u);
        const float4 f = hcm_ld4(low + (size_t)t * rank + 16u * ks + 4u * (l & 3u));
        uint4 b;
        hcm_split(f.x, f.y, b.x, b.z);
        hcm_split(f.z, f.w, b.y, b.w);
        return b;
    };
    if constexpr (!DIRECT) {
        const unsigned nb = (rank / 16u) * NT * 32u;
        for (unsigned i = threadIdx.x; i < nb; i += blockDim.x) {
            s_b[i] = frag((i >> 5) / NT, (i >> 5) % NT, i & 31u);
        }
        __syncthreads();
    }

    float ah[MT][NT][4], al[MT][NT][4];
    #pragma unroll
    for (unsigned j = 0; j < MT; ++j) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) ah[j][nt][c] = al[j][nt][c] = 0.0f;
        }
    }
    auto step = [&](const unsigned u, const unsigned k, const bool reload) {
        unsigned w[4][MT];
        #pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            #pragma unroll
            for (unsigned j = 0; j < MT; ++j) w[e][j] = ur[u][e][j];
        }
        if (reload) {
            #pragma unroll
            for (unsigned e = 0; e < 4; ++e) hcm_ldw<MT>(ub + (size_t)(16u * (k + U) + e) * hc_dim, ur[u][e]);
        }
        uint4 b[NT];
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt)
            b[nt] = DIRECT ? frag(ks0 + k, nt, lane) : s_b[((ks0 + k) * NT + nt) * 32u + lane];
        #pragma unroll
        for (unsigned j = 0; j < MT; ++j) {
            const unsigned a[4] = {__byte_perm(w[0][j], w[1][j], 0x5410), __byte_perm(w[0][j], w[1][j], 0x7632),
                                   __byte_perm(w[2][j], w[3][j], 0x5410), __byte_perm(w[2][j], w[3][j], 0x7632)};
            #pragma unroll
            for (unsigned nt = 0; nt < NT; ++nt) {
                hcm_mma(ah[j][nt], a, b[nt].x, b[nt].y);
                hcm_mma(al[j][nt], a, b[nt].z, b[nt].w);
            }
        }
    };
    unsigned k0 = 0;
    for (; k0 + U < nks; k0 += U) {
        #pragma unroll
        for (unsigned u = 0; u < U; ++u) step(u, k0 + u, true);
    }
    #pragma unroll
    for (unsigned u = 0; u < U; ++u) step(u, k0 + u, false);
    __syncthreads();  // s_b is reused below

    // Lane (g, q), tile j, token column c of n-tile nt: dims 2MT g + 2j (c0, c1)
    // and + 1 (c2, c3), tokens nt*8 + 2q (c0, c2) and + 1 (c1, c3).
    float v[MT][NT][4];
    #pragma unroll
    for (unsigned j = 0; j < MT; ++j) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) v[j][nt][c] = ah[j][nt][c] + al[j][nt][c];
        }
    }
    float* s_f = reinterpret_cast<float*>(smem);
    constexpr unsigned PW = MT * NT * 4u * 32u;  // one warp's partial, lane-major
    if (WK > 1) {
        if (wk > 0) {
            #pragma unroll
            for (unsigned j = 0; j < MT; ++j) {
                #pragma unroll
                for (unsigned nt = 0; nt < NT; ++nt) {
                    #pragma unroll
                    for (unsigned c = 0; c < 4; ++c)
                        s_f[((wk - 1u) * 4u + s) * PW + ((j * NT + nt) * 4u + c) * 32u + lane] = v[j][nt][c];
                }
            }
        }
        __syncthreads();
        if (wk == 0) {
            #pragma unroll
            for (unsigned j = 0; j < MT; ++j) {
                #pragma unroll
                for (unsigned nt = 0; nt < NT; ++nt) {
                    #pragma unroll
                    for (unsigned c = 0; c < 4; ++c) {
                        for (unsigned k = 1; k < WK; ++k)
                            v[j][nt][c] += s_f[((k - 1u) * 4u + s) * PW + ((j * NT + nt) * 4u + c) * 32u + lane];
                    }
                }
            }
        }
        __syncthreads();
    }
    // p_s = sigmoid(up) * normed -> s_p[s][token][DW], then the stream mean.
    float* s_p = s_f;
    if (wk == 0) {
        #pragma unroll
        for (unsigned nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (unsigned c = 0; c < 2; ++c) {
                const unsigned tok = nt * 8u + 2u * q + c;
                const float* nrow = normed + (size_t)min(tok, T - 1u) * hc_dim + s * H + d0 + 2u * MT * g;
                float n[2 * MT];
                #pragma unroll
                for (unsigned x = 0; x < 2 * MT; x += 2) {
                    const float2 f = hcm_ld2(nrow + x);
                    n[x] = f.x; n[x + 1] = f.y;
                }
                float* dst = s_p + (s * TOK + tok) * DW + 2u * MT * g;
                #pragma unroll
                for (unsigned j = 0; j < MT; ++j) {
                    dst[2 * j] = hcm_sigmoid(v[j][nt][c]) * n[2 * j];
                    dst[2 * j + 1] = hcm_sigmoid(v[j][nt][2 + c]) * n[2 * j + 1];
                }
            }
        }
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < TOK * DW / 2u; i += blockDim.x) {
        const unsigned tok = i / (DW / 2u), dl = 2u * (i % (DW / 2u));
        if (tok >= T) continue;
        float m[2];
        #pragma unroll
        for (unsigned e = 0; e < 2; ++e) {
            float mixed = 0.0f;
            mixed += s_p[(0u * TOK + tok) * DW + dl + e];
            mixed += s_p[(1u * TOK + tok) * DW + dl + e];
            mixed += s_p[(2u * TOK + tok) * DW + dl + e];
            mixed += s_p[(3u * TOK + tok) * DW + dl + e];
            m[e] = mixed * 0.25f;
        }
        *reinterpret_cast<__nv_bfloat162*>(y_out + (size_t)tok * H + d0 + dl) = __floats2bfloat162_rn(m[0], m[1]);
    }
}

// Grid (H / HCM_FN_DW), block 128 * HCM_FN_WK. Dynamic shared:
// hc_mma_finish_smem() bytes. Host checks rank % (16 * WK * U) == 0,
// H % HCM_FN_DW == 0, T in 1..32.
extern "C" __global__ void __launch_bounds__(128 * HCM_FN_WK) hc_mma_finish(
    const float* __restrict__ normed,          // [>= T, 4*H] FP32
    const float* __restrict__ low,             // [T, rank] FP32
    const __nv_bfloat16* __restrict__ up_w,    // [rank, 4*H]
    __nv_bfloat16* __restrict__ y_out,         // [T, H]
    const unsigned hidden_size, const unsigned rank, const unsigned T
) {
    atlas_pdl_enter();
    extern __shared__ __align__(16) unsigned char hcm_smem[];
    switch ((T + 7u) / 8u) {
    case 1: hcm_finish<1>(normed, low, up_w, y_out, hidden_size, rank, T, hcm_smem); break;
    case 2: hcm_finish<2>(normed, low, up_w, y_out, hidden_size, rank, T, hcm_smem); break;
    case 3: hcm_finish<3>(normed, low, up_w, y_out, hidden_size, rank, T, hcm_smem); break;
    default: hcm_finish<4>(normed, low, up_w, y_out, hidden_size, rank, T, hcm_smem); break;
    }
}

// ── prefill rows (ATLAS_QWEN4EXP_PREFILL_ROWINV) ──────────────────────────
// The two kernels above over any number of rows, so a prefill collapse is
// byte for byte the decode collapse of each of its rows: groups of rows on
// blockIdx.z, each the same per-row operation sequence (a row is its own mma
// column; WM, DN and the group size are launch geometry only). Wider CTAs
// (7 weight tiles: `normed` read by 3 CTAs, not 7) and wider groups (96 rows
// down, 64 finish) keep the weight re-reads down at prefill widths.
#define HCM_ROWS_WM 7u
#define HCM_ROWS_DN_T 96u
#define HCM_ROWS_FN_T 64u

// Grid (ceil(ceil(rows / 16) / HCM_ROWS_WM), HCM_DN_CL, ceil(T / 96)),
// block 32 * HCM_ROWS_WM * HCM_DN_WK.
extern "C" __global__ void __cluster_dims__(1, HCM_DN_CL, 1)
__launch_bounds__(32 * HCM_ROWS_WM * HCM_DN_WK) hc_mma_down_rows(
    const float* __restrict__ normed, const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ inject_w, float* __restrict__ low_out,
    float* __restrict__ inj_out, const unsigned hidden_size, const unsigned hc,
    const unsigned rank, const unsigned T
) {
    atlas_pdl_enter();
    __shared__ float s_red[HCM_ROWS_WM * 16u * HCM_ROWS_DN_T];
    __shared__ float s_wk[HCM_DN_WK > 1 ? (HCM_DN_WK - 1) * HCM_ROWS_WM * 16u * HCM_ROWS_DN_T : 1];
    const unsigned K = hc * hidden_size;
    const unsigned g0 = blockIdx.z * HCM_ROWS_DN_T;
    const unsigned Tg = min(HCM_ROWS_DN_T, T - g0);
    normed += (size_t)g0 * K;
    low_out += (size_t)g0 * rank;
    if (inj_out != nullptr) inj_out += (size_t)g0 * hc;
#define HCM_DOWN_R(NT_, DN_) \
    hcm_down<NT_, HCM_ROWS_WM, DN_>(normed, down_w, inject_w, low_out, inj_out, K, hc, rank, Tg, s_red, s_wk)
    switch ((Tg + 7u) / 8u) {
    case 1: HCM_DOWN_R(1, 2); break;
    case 2: HCM_DOWN_R(2, 2); break;
    case 3: HCM_DOWN_R(3, 2); break;
    case 4: HCM_DOWN_R(4, 2); break;
    case 5: HCM_DOWN_R(5, 2); break;
    case 6: HCM_DOWN_R(6, 2); break;
    case 7: HCM_DOWN_R(7, 2); break;
    case 8: HCM_DOWN_R(8, 2); break;
    case 9: HCM_DOWN_R(9, 1); break;
    case 10: HCM_DOWN_R(10, 1); break;
    case 11: HCM_DOWN_R(11, 1); break;
    default: HCM_DOWN_R(12, 1); break;
    }
#undef HCM_DOWN_R
}

// Grid (H / HCM_FN_DW, 1, ceil(T / 64)), block 128 * HCM_FN_WK. Dynamic
// shared: the stream-mean tile, 4 * 8 * ceil(min(T, 64) / 8) * HCM_FN_DW * 4
// bytes (HCM_FN_WK == 1: no K-split partials).
extern "C" __global__ void __launch_bounds__(128 * HCM_FN_WK) hc_mma_finish_rows(
    const float* __restrict__ normed, const float* __restrict__ low,
    const __nv_bfloat16* __restrict__ up_w, __nv_bfloat16* __restrict__ y_out,
    const unsigned hidden_size, const unsigned rank, const unsigned T
) {
    atlas_pdl_enter();
    extern __shared__ __align__(16) unsigned char hcm_smem[];
    const unsigned g0 = blockIdx.z * HCM_ROWS_FN_T;
    const unsigned Tg = min(HCM_ROWS_FN_T, T - g0);
    normed += (size_t)g0 * 4u * hidden_size;
    low += (size_t)g0 * rank;
    y_out += (size_t)g0 * hidden_size;
#define HCM_FIN_R(NT_) hcm_finish<NT_, true>(normed, low, up_w, y_out, hidden_size, rank, Tg, hcm_smem)
    switch ((Tg + 7u) / 8u) {
    case 1: HCM_FIN_R(1); break;
    case 2: HCM_FIN_R(2); break;
    case 3: HCM_FIN_R(3); break;
    case 4: HCM_FIN_R(4); break;
    case 5: HCM_FIN_R(5); break;
    case 6: HCM_FIN_R(6); break;
    case 7: HCM_FIN_R(7); break;
    default: HCM_FIN_R(8); break;
    }
#undef HCM_FIN_R
}
