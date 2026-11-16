// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the tensor-core projection every `dsv41_*gemv_tc*`
// entry and the fused expert kernels (`dsv41_expert.cu`) instantiate. See
// `dsv41_tc.cu` for the arithmetic and the layout. `cta` is which 16 weight
// rows the block computes: `blockIdx.x` when a launch is one projection.
//
// `IN` is how the input reaches the multiply. Every projection of this
// model reads an FP8 activation, and at one token the launch that rounded
// it cost as much as the rounding: F4_QUANT takes the BF16 values as they
// were computed and rounds them here, as `dsv41_fp8_quant_dequant` does (a
// lane holds 16 of a scale block's 32 values and the lane beside it the
// rest). A block pays for it again, which costs about what the launch did.
//
// How the weights reach the block: two forms with the same sums in the same
// order. In `dsv41_gemv_tc_impl` a warp loads its weights as it uses them,
// one chunk ahead. It waits on memory for each load, and on cold weights
// that wait, not the arithmetic, is nine tenths of its time. In
// `dsv41_gemv_staged_impl` a block stages its 16 rows into shared memory
// with asynchronous copies, 512 K values of every row at a time and three
// such stages in flight, its scales whole and first, and the warps multiply
// from there. Staging wins where a row is long (K of 4,096 and 5,120, by a
// quarter), and loses where it is short and the blocks many, for it costs a
// block a barrier a stage and 27 KB of shared memory. Measured on GB10, on
// weights no cache held.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "dsv41_common.cuh"

#define F4_ROUNDED 0
#define F4_QUANT 1

#define F4_WARPS 8
#define F4_WARP 32
#define F4_BLOCK 32
// Stages of weights in flight, a row's bytes of one (FP8; FP4 packs two
// values a byte), and the scale blocks of the widest row (K = 5120).
#define F4_DEPTH 3
#define F4_STAGE_BYTES 512
#define F4_MAX_BLOCKS 160

// What a block keeps in shared memory: the weight codes and the warps'
// sums; and, when it stages its weights, those and their scales.
template <int NT>
struct F4Shared {
    float red[F4_WARPS][F4_WARP][NT * 4];
    unsigned short code[256];
};

struct F4Staged {
    alignas(16) unsigned char w[F4_DEPTH][16][F4_STAGE_BYTES];
    alignas(16) unsigned char s[16 * F4_MAX_BLOCKS];
};

// Sixteen bytes into shared memory without waiting for them.
__device__ __forceinline__ void f4_cp16(void* dst, const void* src) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" : : "r"(d), "l"(src));
}

template <int NT>
struct F4Chunk {
    uint4 w0, w1;               // 16 values of rows g and g+8 (FP4: 8 bytes each)
    uint4 x[NT][2];             // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;        // E8M0 block scales of rows g, g+8
};

// Chunk `ch` of the lane's rows: values at w{0,1}p + ch * (FP4 ? 32 : 64)
// (8 or 16 bytes each) and scales at s{0,1}p[ch * 2]; activations at k = kb
// = ch * 64 + 16c.
template <int NT, bool FP4>
__device__ __forceinline__ void f4_load(
    F4Chunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p,
    const __nv_bfloat16* A, unsigned int a_stride, unsigned int g, unsigned int M,
    unsigned int K,
    unsigned int ch, unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    const unsigned long long wo = (unsigned long long)ch * (FP4 ? 32ull : 64ull);
    const unsigned long long so = (unsigned long long)ch * 2ull;
    if constexpr (FP4) {
        if (v0) { const uint2 w = *(const uint2*)(w0p + wo); t.w0.x = w.x; t.w0.y = w.y; t.s0 = s0p[so]; }
        if (v1) { const uint2 w = *(const uint2*)(w1p + wo); t.w1.x = w.x; t.w1.y = w.y; t.s1 = s1p[so]; }
    } else {
        if (v0) { t.w0 = *(const uint4*)(w0p + wo); t.s0 = s0p[so]; }
        if (v1) { t.w1 = *(const uint4*)(w1p + wo); t.s1 = s1p[so]; }
    }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * a_stride + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

// A chunk for one lane: its rows' 16 values at `w0` and `w1` (8 or 16 staged
// bytes each), their block scales, and the activations at k = kb.
template <int NT, bool FP4>
__device__ __forceinline__ void f4_take(
    F4Chunk<NT>& t, const unsigned char* w0, const unsigned char* w1, unsigned int s0,
    unsigned int s1, const __nv_bfloat16* A, unsigned int a_stride, unsigned int g, unsigned int M,
    unsigned int K,
    unsigned int kb, bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    if constexpr (FP4) {
        if (v0) { const uint2 w = *(const uint2*)w0; t.w0.x = w.x; t.w0.y = w.y; t.s0 = s0; }
        if (v1) { const uint2 w = *(const uint2*)w1; t.w1.x = w.x; t.w1.y = w.y; t.s1 = s1; }
    } else {
        if (v0) { t.w0 = *(const uint4*)w0; t.s0 = s0; }
        if (v1) { t.w1 = *(const uint4*)w1; t.s1 = s1; }
    }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * a_stride + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

// The chunk's activations as the multiply reads them: see `IN` above. Only
// the lanes that hold a row of `A` (g of them in eight, for one row one)
// have anything to round, and only they pay for it; but every lane of the
// warp exchanges its block's largest magnitude with the lane beside it,
// which holds the other half of the scale block.
template <int NT, int IN>
__device__ __forceinline__ void f4_prepare(F4Chunk<NT>& t, unsigned int g, unsigned int M) {
    if constexpr (IN != F4_ROUNDED) {
        #pragma unroll
        for (int i = 0; i < NT; i++) {
            const bool holds = 8u * i + g < M;
            unsigned int* xw = reinterpret_cast<unsigned int*>(&t.x[i][0]);
            float v[16];
            float amax = 0.0f;
            if (holds) {
                #pragma unroll
                for (int w = 0; w < 8; w++) dsv41_unpack2(xw[w], v[2 * w], v[2 * w + 1]);
                #pragma unroll
                for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(v[j]));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            if (holds) {
                const float scale = dsv41_fp8_scale(amax);
                #pragma unroll
                for (int w = 0; w < 8; w++)
                    xw[w] = dsv41_pack2(dsv41_fp8_value(v[2 * w], scale),
                                        dsv41_fp8_value(v[2 * w + 1], scale));
            }
        }
    }
}

// BF16 2^(s-127) (E8M0 scale; 0 decodes to 0, as does the NaN code 255).
__device__ __forceinline__ __nv_bfloat162 f4_scale(unsigned int s) {
    const unsigned short bits = (s == 0u || s == 255u) ? 0 : (unsigned short)(s << 7);
    const __nv_bfloat16 v = __ushort_as_bfloat16(bits);
    return __halves2bfloat162(v, v);
}

// One m16n8k16 BF16 MMA step into `acc` (FP32).
__device__ __forceinline__ void f4_mma(float (&acc)[4], const unsigned int (&a)[4],
                                       unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Weight code -> BF16 bits (exact), in `s_code`, by the first threads of
// the block. FP4: an E2M1 nibble, 0, 0.5, 1, 1.5, 2, 3, 4, 6, then the same
// negated. FP8: an E4M3 byte. The caller puts a barrier after it.
template <bool FP4>
__device__ __forceinline__ void f4_codes(unsigned short* s_code) {
    if constexpr (FP4) {
        if (threadIdx.x < 16) {
            const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            const float v = mags[threadIdx.x & 7u];
            s_code[threadIdx.x] =
                __bfloat16_as_ushort(__float2bfloat16((threadIdx.x & 8u) ? -v : v));
        }
    } else {
        for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
            __nv_fp8_e4m3 f;
            *reinterpret_cast<unsigned char*>(&f) = (unsigned char)i;
            s_code[i] = __bfloat16_as_ushort(__float2bfloat16((float)f));
        }
    }
}

// One chunk's four MMA steps into `acc`.
template <int NT, bool FP4>
__device__ __forceinline__ void f4_chunk(
    float (&acc)[NT][4], const F4Chunk<NT>& cur, const unsigned short* s_code) {
    const __nv_bfloat162 sc0 = f4_scale(cur.s0), sc1 = f4_scale(cur.s1);
    // Two weight codes -> a BF16 pair, times the block's scale.
    auto pair = [&](unsigned int lo, unsigned int hi, __nv_bfloat162 sc) {
        __nv_bfloat162 v = __halves2bfloat162(__ushort_as_bfloat16(s_code[lo]),
                                              __ushort_as_bfloat16(s_code[hi]));
        v = __hmul2(v, sc);
        return *reinterpret_cast<unsigned int*>(&v);
    };
    const unsigned int wb0[4] = {cur.w0.x, cur.w0.y, cur.w0.z, cur.w0.w};
    const unsigned int wb1[4] = {cur.w1.x, cur.w1.y, cur.w1.z, cur.w1.w};
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned int a[4];
        if constexpr (FP4) {
            // Values 4j..4j+3 of each row's 16-value run = nibbles 4j..4j+3
            // = bytes 2j and 2j+1 = 16 bits at 16j of the run's 64.
            const unsigned int x0 = (wb0[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
            const unsigned int x1 = (wb1[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
            a[0] = pair(x0 & 0xFu, (x0 >> 4) & 0xFu, sc0);
            a[1] = pair(x1 & 0xFu, (x1 >> 4) & 0xFu, sc1);
            a[2] = pair((x0 >> 8) & 0xFu, x0 >> 12, sc0);
            a[3] = pair((x1 >> 8) & 0xFu, x1 >> 12, sc1);
        } else {
            // Values 4j..4j+3 = bytes 4j..4j+3 = word j.
            const unsigned int x0 = wb0[j], x1 = wb1[j];
            a[0] = pair(x0 & 0xFFu, (x0 >> 8) & 0xFFu, sc0);
            a[1] = pair(x1 & 0xFFu, (x1 >> 8) & 0xFFu, sc1);
            a[2] = pair((x0 >> 16) & 0xFFu, x0 >> 24, sc0);
            a[3] = pair((x1 >> 16) & 0xFFu, x1 >> 24, sc1);
        }
        #pragma unroll
        for (int i = 0; i < NT; i++) {
            const uint4 lo = cur.x[i][0], hi = cur.x[i][1];
            const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            f4_mma(acc[i], a, xw[2 * j], xw[2 * j + 1]);
        }
    }
}

// A lane's four sums of tile `i` (rows r0 and r1, activation rows 8i + 2c and
// the one after) into C.
template <bool F32OUT>
__device__ __forceinline__ void f4_store(
    void* C, const float (&v)[4], unsigned int i, unsigned int c, unsigned int r0,
    unsigned int r1, unsigned int M, unsigned int N, unsigned int out_stride) {
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        const unsigned int n = (q < 2) ? r0 : r1;
        const unsigned int m = 8u * i + 2u * c + (q & 1u);
        if (n < N && m < M) {
            const unsigned long long at = (unsigned long long)m * out_stride + n;
            if constexpr (F32OUT) {
                static_cast<float*>(C)[at] = v[q];
            } else {
                static_cast<__nv_bfloat16*>(C)[at] = __float2bfloat16(v[q]);
            }
        }
    }
}

// Where a lane reads rows r0 and r1 of the weights and their scales.
struct F4Rows {
    const unsigned char* w0p;
    const unsigned char* w1p;
    const unsigned char* s0p;
    const unsigned char* s1p;
    bool v0, v1;
};

template <bool FP4>
__device__ __forceinline__ F4Rows f4_rows(
    const unsigned char* W, const unsigned char* S, unsigned int r0, unsigned int r1,
    unsigned int c, unsigned int N, unsigned int K) {
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned long long blocks = K / F4_BLOCK;
    // A row is K bytes, or K/2 packed; the lane's 16 values are 16 or 8 of them.
    const unsigned int row_bytes = FP4 ? K / 2u : K, lane_bytes = FP4 ? 8u : 16u;
    return F4Rows{
        W + (unsigned long long)(v0 ? r0 : 0u) * row_bytes + lane_bytes * c,
        W + (unsigned long long)(v1 ? r1 : 0u) * row_bytes + lane_bytes * c,
        S + (unsigned long long)(v0 ? r0 : 0u) * blocks + (c >> 1),
        S + (unsigned long long)(v1 ? r1 : 0u) * blocks + (c >> 1),
        v0,
        v1,
    };
}

template <int NT, bool FP4, bool F32OUT, int IN = F4_ROUNDED>
__device__ __forceinline__ void dsv41_gemv_tc_impl(
    F4Shared<NT>& sh,
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    void* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int cta,
    unsigned int a_stride
) {
    f4_codes<FP4>(sh.code);
    __syncthreads();

    const unsigned int warp = threadIdx.x / F4_WARP;
    const unsigned int lane = threadIdx.x % F4_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = cta * 16u + g;
    const unsigned int r1 = r0 + 8u;
    const unsigned int chunks = (K + 63u) / 64u;
    const F4Rows r = f4_rows<FP4>(W, S, r0, r1, c, N, K);

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    F4Chunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks) {
        f4_load<NT, FP4>(cur, r.w0p, r.w1p, r.s0p, r.s1p, A, a_stride, g, M, K, ch, ch * 64u + 16u * c,
                         r.v0, r.v1);
        f4_prepare<NT, IN>(cur, g, M);
    }
    for (; ch < chunks; ch += F4_WARPS) {
        const unsigned int next = ch + F4_WARPS;
        if (next < chunks) {
            f4_load<NT, FP4>(nxt, r.w0p, r.w1p, r.s0p, r.s1p, A, a_stride, g, M, K, next,
                             next * 64u + 16u * c, r.v0, r.v1);
            f4_prepare<NT, IN>(nxt, g, M);
        }
        f4_chunk<NT, FP4>(acc, cur, sh.code);
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) sh.red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            v[q] = 0.0f;
            #pragma unroll
            for (int w = 0; w < F4_WARPS; w++) v[q] += sh.red[w][lane][i * 4 + q];
        }
        f4_store<F32OUT>(C, v, i, c, r0, r1, M, N, out_stride);
    }
}

// The staged form: see the head of this file.
template <int NT, bool FP4, bool F32OUT, int IN = F4_ROUNDED>
__device__ __forceinline__ void dsv41_gemv_staged_impl(
    F4Shared<NT>& sh,
    F4Staged& st_sh,
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    void* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int cta,
    unsigned int a_stride
) {
    f4_codes<FP4>(sh.code);
    const unsigned int warp = threadIdx.x / F4_WARP;
    const unsigned int lane = threadIdx.x % F4_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int first = cta * 16u;
    const unsigned int r0 = first + g, r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned int blocks = K / F4_BLOCK;
    const unsigned int chunks = (K + 63u) / 64u;
    const unsigned int stages = (chunks + F4_WARPS - 1u) / F4_WARPS;
    // A row is K bytes, or K/2 packed; a chunk of it 64 or 32, of which the
    // lane's 16 values are 16 or 8.
    const unsigned int row_bytes = FP4 ? K / 2u : K;
    const unsigned int stage_bytes = FP4 ? F4_STAGE_BYTES / 2u : F4_STAGE_BYTES;
    const unsigned int chunk_bytes = FP4 ? 32u : 64u, lane_bytes = FP4 ? 8u : 16u;

    // The block's scales: 16 rows are one run of 16 * blocks bytes, which
    // starts on a 16-byte boundary whatever K is. A piece that would reach
    // past the last row's end comes a byte at a time.
    const unsigned long long scales_at = (unsigned long long)first * blocks;
    const unsigned long long scales_end = (unsigned long long)N * blocks;
    for (unsigned int p = threadIdx.x; p < blocks; p += F4_WARPS * F4_WARP) {
        const unsigned long long at = scales_at + 16ull * p;
        if (at + 16ull <= scales_end) {
            f4_cp16(&st_sh.s[16u * p], S + at);
        } else {
            for (unsigned int b = 0; b < 16u && at + b < scales_end; ++b) st_sh.s[16u * p + b] = S[at + b];
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    // Stage `st` of the block's rows: K values [512 st, 512 st + 512).
    auto stage_in = [&](unsigned int st) {
        const unsigned int pieces = stage_bytes / 16u;
        for (unsigned int p = threadIdx.x; p < 16u * pieces; p += F4_WARPS * F4_WARP) {
            const unsigned int row = p / pieces, off = (p % pieces) * 16u;
            const unsigned long long at = (unsigned long long)st * stage_bytes + off;
            if (first + row < N && at < row_bytes)
                f4_cp16(&st_sh.w[st % F4_DEPTH][row][off],
                        W + (unsigned long long)(first + row) * row_bytes + at);
        }
    };
    #pragma unroll
    for (unsigned int st = 0; st < F4_DEPTH - 1; st++) {
        if (st < stages) stage_in(st);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    for (unsigned int st = 0; st < stages; st++) {
        // Everything but the newest stages has landed; the barrier lets
        // every warp see it, and says the stage before is no longer read.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(F4_DEPTH - 2));
        __syncthreads();
        const unsigned int ch = st * F4_WARPS + warp;
        if (ch < chunks) {
            const unsigned char (*rows)[F4_STAGE_BYTES] = st_sh.w[st % F4_DEPTH];
            const unsigned int at = warp * chunk_bytes + lane_bytes * c;
            const unsigned int block = ch * 2u + (c >> 1);
            F4Chunk<NT> cur;
            f4_take<NT, FP4>(cur, rows[g] + at, rows[g + 8u] + at, st_sh.s[g * blocks + block],
                             st_sh.s[(g + 8u) * blocks + block], A, a_stride, g, M, K, ch * 64u + 16u * c, v0,
                             v1);
            f4_prepare<NT, IN>(cur, g, M);
            f4_chunk<NT, FP4>(acc, cur, sh.code);
        }
        const unsigned int next = st + F4_DEPTH - 1u;
        if (next < stages) stage_in(next);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) sh.red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            v[q] = 0.0f;
            #pragma unroll
            for (int w = 0; w < F4_WARPS; w++) v[q] += sh.red[w][lane][i * 4 + q];
        }
        f4_store<F32OUT>(C, v, i, c, r0, r1, M, N, out_stride);
    }
}
