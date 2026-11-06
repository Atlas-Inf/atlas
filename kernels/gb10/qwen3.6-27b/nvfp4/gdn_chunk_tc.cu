// SPDX-License-Identifier: AGPL-3.0-only

// Fused chunked Gated DeltaNet prefill on BF16 tensor cores
// (lever ATLAS_GDN_CHUNK_TC=1). One CTA per (stream, value head, 64-column
// value block) walks its stream's 64-token chunks in order; every matmul of
// a chunk is mma.sync m16n8k16 with FP32 accumulation, and nothing but the
// output and the final state leaves the CTA (the default FLA path writes W,
// U, the per-chunk states and uc to global and runs three kernels).
//
// Per chunk (gc = in-chunk cumulative log gate, serial in token order as
// the FLA path does; L[i][l] = b_i e^(gc_i-gc_l) <k_i,k_l> for l < i):
//   T  = (I + L)^-1                        FP32, blocked in place
//   R  = V - diag(e^gc) K S                (K S: S split into BF16 hi + lo)
//   uc = T diag(b) R                       (= U - W S of the FLA path)
//   O  = (diag(e^gc) Q S + P uc) / sqrt(d), P[i][l] = e^(gc_i-gc_l) <q_i,k_l>, l <= i
//   S <- e^(gc_last) S + (K diag(e^(gc_last-gc)))^T uc
// The state S stays FP32 in registers across chunks. Where an operand is
// an FP32 intermediate (S, R, T diag(b), P, uc, the decayed K) it is split
// into BF16 hi + lo and multiplied as hi*hi + hi*lo + lo*hi, which keeps
// ~16 mantissa bits; Q, K and V are BF16 inputs and go in as they are. Q S
// uses the hi half of S only (the FLA output pass reads a BF16 state too).
//
// Q and K rows are L2-normalized here, bit for bit as l2_norm_bf16 (same
// pair squares, same reduction tree, eps 1e-6), so that pass is skipped
// wherever this kernel serves every scan (`gdn_chunk_tc_serves`); a path
// that normalized already only re-normalizes unit rows.
//
// Each stream is computed by its own CTAs in a fixed order, so a stream's
// result does not depend on what else is in the launch.
//
// Grid (V_DIM / 64, num_v_heads, batch), block 256, dynamic smem SMEM below.
// Requires k_dim = v_dim = 128 and 16-byte aligned rows (strides % 8 == 0).

#include <cuda_bf16.h>
#include <stdint.h>

namespace gdn_chunk_tc {
constexpr int KD = 128;        // key head dim
constexpr int VD = 128;        // value head dim
constexpr int DV = 64;         // value columns per CTA
constexpr int CH = 64;         // chunk length
constexpr int KP = KD + 8;     // pitch of [*][k] bf16 tiles (conflict-free fragments)
constexpr int VP = DV + 8;     // pitch of the V tile
constexpr int CP = CH + 8;     // pitch of [*][chunk] tiles (bf16 and the fp32 T)
constexpr float GATE_FLOOR = 1e-30f;   // as gated_delta_rule_fla.cu
constexpr float L2_EPS = 1e-6f;        // as the l2_norm_bf16 pass this kernel absorbs

constexpr int OFF_K = 0;                      // bf16 [CH][KP]   K chunk
constexpr int OFF_V = OFF_K + CH * KP * 2;    // bf16 [CH][VP]   V chunk (this block's columns)
constexpr int OFF_SH = OFF_V + CH * VP * 2;   // bf16 [DV][KP]   S^T hi
constexpr int OFF_SL = OFF_SH + DV * KP * 2;  // bf16 [DV][KP]   S^T lo
constexpr int OFF_T = OFF_SL + DV * KP * 2;   // fp32 [CH][CP] T, then bf16 hi/lo [CH][CP] (T diag(b), then P)
constexpr int OFF_R = OFF_T + CH * CP * 4;    // bf16 hi/lo [DV][CP]: R^T, then uc^T
constexpr int OFF_F = OFF_R + 2 * DV * CP * 2;  // fp32 gc, e^gc, decay, beta [CH] each
constexpr int SMEM = OFF_F + 4 * CH * 4;
static_assert(SMEM <= 101376, "GB10 caps a block at 99 KB of shared memory");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ld32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t pack(bf16 lo, bf16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ void split(float x, bf16& h, bf16& l) {
    h = __float2bfloat16_rn(x);
    l = __float2bfloat16_rn(x - __bfloat162float(h));
}
// 16-byte global -> shared copy; zero-filled when !valid.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
// Chunk rows [cs, cs + ce) of a [tokens][stride] tile, `W` columns, into a
// pitch-`P` smem tile; rows past ce are zeros.
template <int W, int P>
__device__ __forceinline__ void stage_rows(bf16* dst, const bf16* src, size_t stride, unsigned int cs, int ce, int tid) {
#pragma unroll
    for (int idx = tid; idx < CH * (W / 8); idx += 256) {
        const int i = idx / (W / 8), c8 = (idx % (W / 8)) * 8;
        cp16(dst + i * P + c8, src + (size_t)(cs + (i < ce ? i : 0)) * stride + c8, i < ce);
    }
}

// A fragment (16 x 16) of a row-major [m][k] tile.
__device__ __forceinline__ void lda(uint32_t (&a)[4], const bf16* A, int P, int m0, int k0, int grp, int q) {
    a[0] = ld32(A + (m0 + grp) * P + k0 + 2 * q);
    a[1] = ld32(A + (m0 + grp + 8) * P + k0 + 2 * q);
    a[2] = ld32(A + (m0 + grp) * P + k0 + 2 * q + 8);
    a[3] = ld32(A + (m0 + grp + 8) * P + k0 + 2 * q + 8);
}
// B fragment (16 k x 8 n) of a tile stored [n][k].
__device__ __forceinline__ void ldb(uint32_t& b0, uint32_t& b1, const bf16* B, int P, int n0, int k0, int grp, int q) {
    b0 = ld32(B + (n0 + grp) * P + k0 + 2 * q);
    b1 = ld32(B + (n0 + grp) * P + k0 + 2 * q + 8);
}
// Accumulator element e of a 16 x 8 tile: row (grp, +8 for e >= 2), column (2q, +1 for odd e).
__device__ __forceinline__ int erow(int e, int grp) { return grp + (e >> 1) * 8; }
__device__ __forceinline__ int ecol(int e, int q) { return 2 * q + (e & 1); }

// Named barriers: 0 is __syncthreads; G0 (warps 0-3) syncs on 2 and hands
// the gate tables to G1 (warps 4-7) on 1.
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
constexpr int BAR_GATES = 1, BAR_G0 = 2;

// acc += T[ra..+16][ka..+16] * T[kb..+16][cb..+8], both FP32 in smem, each
// split into BF16 hi + lo (hi*hi + hi*lo + lo*hi): ~16 mantissa bits, which
// is what T keeps anyway once it is split for the uc product.
__device__ __forceinline__ void mma_f32(float (&acc)[4], const float* T, int ra, int ka, int kb, int cb, int grp, int q) {
    uint32_t ah[4], al[4];
#pragma unroll
    for (int f = 0; f < 4; f++) {
        const float2 x = *reinterpret_cast<const float2*>(T + (ra + grp + (f & 1) * 8) * CP + ka + 2 * q + (f >> 1) * 8);
        bf16 h0, l0, h1, l1;
        split(x.x, h0, l0);
        split(x.y, h1, l1);
        ah[f] = pack(h0, h1);
        al[f] = pack(l0, l1);
    }
    uint32_t bh[2], bl[2];
#pragma unroll
    for (int f = 0; f < 2; f++) {
        const int k = kb + 2 * q + f * 8;
        bf16 h0, l0, h1, l1;
        split(T[k * CP + cb + grp], h0, l0);
        split(T[(k + 1) * CP + cb + grp], h1, l1);
        bh[f] = pack(h0, h1);
        bl[f] = pack(l0, l1);
    }
    mma(acc, ah, bh[0], bh[1]);
    mma(acc, ah, bl[0], bl[1]);
    mma(acc, al, bh[0], bh[1]);
}
// One warp's 16 x 8 tile of a block step: T[ra..][cb..] = (neg ? -1 : 1) *
// T[ra..][ka..ka+16KT] * T[ka-rows kb..][cb..], stored once every G0 warp
// has read its operands.
template <int KT>
__device__ __forceinline__ void block_step(float* T, int ra, int ka, int kb, int cb, bool neg, int grp, int q) {
    float acc[4] = {};
#pragma unroll
    for (int ks = 0; ks < KT; ks++) mma_f32(acc, T, ra, ka + ks * 16, kb + ks * 16, cb, grp, q);
    bar_sync(BAR_G0, 128);
#pragma unroll
    for (int e = 0; e < 4; e++) T[(ra + erow(e, grp)) * CP + cb + ecol(e, q)] = neg ? -acc[e] : acc[e];
    bar_sync(BAR_G0, 128);
}

// T <- (I + T)^-1 for a strictly lower FP32 64 x 64 T (pitch CP), in place,
// by the 128 threads of G0: 16 x 16 diagonal blocks by substitution, then
// the 2 x 2 block formula [A 0; C B]^-1 = [A^-1 0; -B^-1 C A^-1  B^-1] at 32
// and at 64. Entered and left with G0 synchronized.
__device__ __forceinline__ void tri_inverse(float* T, int tid) {
    float x[16], xs[2][4];
    const int r = tid >> 4, j = tid & 15;
    if (tid < 64) {
        const float* B = T + 16 * r * CP + 16 * r;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            float s = 0.f;
#pragma unroll
            for (int l = 0; l < i; l++) s += B[i * CP + l] * x[l];
            x[i] = i == j ? 1.f : (i > j ? -s : 0.f);
        }
    }
    bar_sync(BAR_G0, 128);
    if (tid < 64) {
        float* B = T + 16 * r * CP + 16 * r;
#pragma unroll
        for (int i = 0; i < 16; i++) B[i * CP + j] = x[i];
    }
    bar_sync(BAR_G0, 128);
    // Level 32: blocks (16..31, 0..15) and (48..63, 32..47): warp pair p,
    // one 8-column half each. X = C A^-1, then -B^-1 X.
    const int warp = tid >> 5, grp = (tid & 31) >> 2, q = tid & 3;
    {
        const int c0 = 32 * (warp >> 1), r0 = c0 + 16, cb = c0 + 8 * (warp & 1);
        block_step<1>(T, r0, c0, c0, cb, false, grp, q);
        block_step<1>(T, r0, r0, r0, cb, true, grp, q);
    }
    // Level 64: block (32..63, 0..31), each warp 16 rows x 8 columns, twice.
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ra = 32 + 16 * (warp >> 1), cb = 16 * (warp & 1) + 8 * h;
        float acc[4] = {};
#pragma unroll
        for (int ks = 0; ks < 2; ks++) mma_f32(acc, T, ra, ks * 16, ks * 16, cb, grp, q);
        xs[h][0] = acc[0]; xs[h][1] = acc[1]; xs[h][2] = acc[2]; xs[h][3] = acc[3];
    }
    bar_sync(BAR_G0, 128);
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ra = 32 + 16 * (warp >> 1), cb = 16 * (warp & 1) + 8 * h;
#pragma unroll
        for (int e = 0; e < 4; e++) T[(ra + erow(e, grp)) * CP + cb + ecol(e, q)] = xs[h][e];
    }
    bar_sync(BAR_G0, 128);
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ra = 32 + 16 * (warp >> 1), cb = 16 * (warp & 1) + 8 * h;
        float acc[4] = {};
#pragma unroll
        for (int ks = 0; ks < 2; ks++) mma_f32(acc, T, ra, 32 + ks * 16, 32 + ks * 16, cb, grp, q);
        xs[h][0] = -acc[0]; xs[h][1] = -acc[1]; xs[h][2] = -acc[2]; xs[h][3] = -acc[3];
    }
    bar_sync(BAR_G0, 128);
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ra = 32 + 16 * (warp >> 1), cb = 16 * (warp & 1) + 8 * h;
#pragma unroll
        for (int e = 0; e < 4; e++) T[(ra + erow(e, grp)) * CP + cb + ecol(e, q)] = xs[h][e];
    }
    bar_sync(BAR_G0, 128);
}

// A 128-wide row's sum of squares exactly as l2_norm_bf16 forms it (128
// threads, thread p squaring pair p, a butterfly per warp, then the two warp
// sums), from the quad that holds it: lane q of the quad holds sq[k][f] =
// pair k * 8 + q + 4 f. Every lane of the quad returns the row's total.
__device__ __forceinline__ float l2_row_total(const float (&sq)[8][2], int lane) {
    float w[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {   // warp h of the norm pass: pairs 32 h ..
        float b[2];
#pragma unroll
        for (int f = 0; f < 2; f++) {
            const float a0 = sq[4 * h][f] + sq[4 * h + 2][f];       // xor 16
            const float a1 = sq[4 * h + 1][f] + sq[4 * h + 3][f];
            b[f] = a0 + a1;                                         // xor 8
        }
        const float c = b[0] + b[1];                                // xor 4
        const float d = c + __shfl_xor_sync(0xffffffffu, c, 2);     // xor 2
        const float e = d + __shfl_xor_sync(0xffffffffu, d, 1);     // xor 1
        w[h] = __shfl_sync(0xffffffffu, e, lane & ~3);              // lane 0's
    }
    return w[0] + w[1];
}
// One pair's square, written as l2_norm_bf16 accumulates it.
__device__ __forceinline__ float pair_sq(float x0, float x1) {
    float s = 0.f;
    s += x0 * x0 + x1 * x1;
    return s;
}

// The warp's state rows (16 * warp + grp, +8) x 64 columns, into S^T hi/lo.
__device__ __forceinline__ void stage_state(const float (&S)[8][4], bf16* Sh, bf16* Sl, int warp, int grp, int q) {
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int k = 16 * warp + erow(e, grp), v = nt * 8 + ecol(e, q);
            bf16 h, l;
            split(S[nt][e], h, l);
            Sh[v * KP + k] = h;
            Sl[v * KP + k] = l;
        }
}
}  // namespace gdn_chunk_tc

extern "C" __global__ void __launch_bounds__(256, 1)
gdn_chunk_tc_prefill(
    float* __restrict__ h_state,               // [nv][KD][VD] fp32, or a table of per-stream bases
    const __nv_bfloat16* __restrict__ query,   // row stride qk_stride (elements)
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,   // row stride v_stride
    const float* __restrict__ gate,            // LINEAR decay, row stride gb_stride
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,        // [tokens][nv * VD]
    unsigned int batch_size, unsigned int seq_len, unsigned int num_k_heads,
    unsigned int num_v_heads, unsigned int qk_stride, unsigned int v_stride,
    unsigned int gb_stride, unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, unsigned int is_varlen,
    const float* const* __restrict__ h_init   // optional per-stream entry-state table
) {
    using namespace gdn_chunk_tc;
    const int vb = blockIdx.x, vh = blockIdx.y, b = blockIdx.z;
    if (vh >= (int)num_v_heads || b >= (int)batch_size) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, grp = lane >> 2, q = lane & 3;
    const int mg = warp & 3, nh = warp >> 2;   // 64 x 64 tiles: rows 16 mg, columns 32 nh
    unsigned int tok0, len;
    if (is_varlen) {
        tok0 = (unsigned int)cu_seqlens[b];
        len = (unsigned int)cu_seqlens[b + 1] - tok0;
    } else {
        tok0 = b * seq_len;
        len = seq_len;
    }
    const int kh = vh / (num_v_heads / num_k_heads);
    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (size_t)vh * KD * VD
        : h_state + ((size_t)b * num_v_heads + vh) * KD * VD;
    H += vb * DV;
    // The entry state: h_init[b] when given (a score branch reads its head's
    // final state in place of a copy), else the state being advanced.
    const float* H0 = h_init ? h_init[b] + (size_t)vh * KD * VD + vb * DV : H;
    const bf16* qg = query + (size_t)tok0 * qk_stride + kh * KD;
    const bf16* kg = key + (size_t)tok0 * qk_stride + kh * KD;
    const bf16* vg = value + (size_t)tok0 * v_stride + vh * VD + vb * DV;
    const float* gg = gate + (size_t)tok0 * gb_stride + vh;
    const float* bg = beta + (size_t)tok0 * gb_stride + vh;
    bf16* og = output + (size_t)tok0 * num_v_heads * VD + vh * VD + vb * DV;
    const size_t ostride = (size_t)num_v_heads * VD;
    const float inv_sqrt_d = rsqrtf((float)KD);

    extern __shared__ __align__(16) char smem_gct[];
    bf16* Ks = (bf16*)(smem_gct + OFF_K);
    bf16* Vs = (bf16*)(smem_gct + OFF_V);
    bf16* Sh = (bf16*)(smem_gct + OFF_SH);
    bf16* Sl = (bf16*)(smem_gct + OFF_SL);
    float* Tf = (float*)(smem_gct + OFF_T);
    bf16* Th = (bf16*)(smem_gct + OFF_T);
    bf16* Tl = Th + CH * CP;
    bf16* Rh = (bf16*)(smem_gct + OFF_R);
    bf16* Rl = Rh + DV * CP;
    float* gc = (float*)(smem_gct + OFF_F);
    float* egc = gc + CH;
    float* dec = egc + CH;
    float* bet = dec + CH;

    // State: this warp's rows k = 16 warp + grp (+8), all 64 columns.
    float S[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++)
            S[nt][e] = H0[(16 * warp + erow(e, grp)) * VD + nt * 8 + ecol(e, q)];
    stage_state(S, Sh, Sl, warp, grp, q);
    if (len > 0) {
        const int ce0 = len < (unsigned)CH ? (int)len : CH;
        stage_rows<KD, KP>(Ks, kg, qk_stride, 0, ce0, tid);
        stage_rows<DV, VP>(Vs, vg, v_stride, 0, ce0, tid);
        cp_commit();
    }

    for (unsigned int cs = 0; cs < len; cs += CH) {
        const int ce = (len - cs) < (unsigned)CH ? (int)(len - cs) : CH;
        const unsigned int cn = cs + CH;   // next chunk
        const int cen = cn < len ? ((len - cn) < (unsigned)CH ? (int)(len - cn) : CH) : 0;
        // ── gate, beta (K, V are in flight) ──
        if (tid < CH) {
            float g = 0.f, bt = 0.f;
            if (tid < ce) {
                g = logf(fmaxf(gg[(size_t)(cs + tid) * gb_stride], GATE_FLOOR));
                bt = bg[(size_t)(cs + tid) * gb_stride];
            }
            gc[tid] = g;
            bet[tid] = bt;
        }
        cp_wait_all();
        __syncthreads();   // K, V (this chunk) and the raw log gates are staged
        {   // L2-normalize the K rows in place, bit for bit as l2_norm_bf16: a
            // quad a row, lane q owning pairs k * 8 + q + 4 f (l2_row_total).
            const int i = tid >> 2, qq = tid & 3;
            uint32_t* row = reinterpret_cast<uint32_t*>(Ks + i * KP);
            uint32_t w[8][2];
            float sq[8][2];
#pragma unroll
            for (int k = 0; k < 8; k++)
#pragma unroll
                for (int f = 0; f < 2; f++) {
                    w[k][f] = row[k * 8 + qq + 4 * f];
                    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w[k][f]);
                    sq[k][f] = pair_sq(__low2float(v), __high2float(v));
                }
            const float inv = rsqrtf(l2_row_total(sq, lane) + L2_EPS);
#pragma unroll
            for (int k = 0; k < 8; k++)
#pragma unroll
                for (int f = 0; f < 2; f++) {
                    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w[k][f]);
                    const __nv_bfloat162 o = __floats2bfloat162_rn(__low2float(v) * inv, __high2float(v) * inv);
                    row[k * 8 + qq + 4 * f] = *reinterpret_cast<const uint32_t*>(&o);
                }
        }
        __syncthreads();

        // Two groups run side by side until the join:
        //   G0 (warps 0-3): K K^T -> L -> T = (I + L)^-1 -> T diag(beta) hi/lo
        //   G1 (warps 4-7): K S -> R; Q K^T -> P; Q S -> e^gc Q S (registers)
        // Each warp of either group owns 16 rows x all 64 columns.
        const int g = warp & 3, r0 = 16 * g;
        // A partial chunk's rows and columns past ce are exact zeros (K, Q,
        // V, beta and the decay are), so the 16-wide tiles wholly past ce
        // only ever add zeros: skip them. `live` tiles hold a row below ce.
        const int live = (ce + 15) >> 4;
        const bool rows_live = r0 < ce;
        float o1[8][4] = {}, pk[8][4] = {};   // G1: e^gc Q S, then P (rows r0..)
        if (warp < 4) {
            if (tid == 0) {   // in token order, as the FLA path sums it (gc is 0 past ce)
                float gv[CH];
#pragma unroll
                for (int i = 0; i < CH; i++) gv[i] = gc[i];
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < CH; i++) {
                    acc += gv[i];
                    gc[i] = i < ce ? acc : 0.f;
                }
            }
            float kk[8][4] = {};
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                if (!rows_live) break;
                uint32_t ka[4];
                lda(ka, Ks, KP, r0, ks * 16, grp, q);
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    uint32_t b0, b1;
                    ldb(b0, b1, Ks, KP, nt * 8, ks * 16, grp, q);
                    mma(kk[nt], ka, b0, b1);
                }
            }
            bar_sync(BAR_G0, 128);   // gc final
            const float gl = gc[ce - 1];
            if (tid < CH) {
                egc[tid] = expf(gc[tid]);
                dec[tid] = tid < ce ? expf(gl - gc[tid]) : 0.f;
            }
            bar_arrive(BAR_GATES, 256);   // gc, egc, dec -> G1
            // Row and column gate values to registers first: the stores below
            // may alias them, which would serialize every load behind a store.
            float gr[2], br[2], gcol[8][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                gr[h] = gc[r0 + grp + 8 * h];
                br[h] = bet[r0 + grp + 8 * h];
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int t = 0; t < 2; t++) gcol[nt][t] = gc[nt * 8 + 2 * q + t];
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int i = r0 + erow(e, grp), c = nt * 8 + ecol(e, q);
                    // Branch-free: a divergent branch around the exp costs
                    // ~6x the arithmetic here. Masked entries clamp to e^0.
                    const float v = br[e >> 1] * __expf(fminf(gr[e >> 1] - gcol[nt][e & 1], 0.f)) * kk[nt][e];
                    Tf[i * CP + c] = (c < i && i < ce) ? v : 0.f;
                }
            bar_sync(BAR_G0, 128);
            tri_inverse(Tf, tid);
            {   // T diag(beta) as bf16 hi/lo over the same bytes
                const int i = tid >> 1, l0 = (tid & 1) * 32;
                float t[32];
#pragma unroll
                for (int j = 0; j < 32; j++) t[j] = Tf[i * CP + l0 + j] * bet[l0 + j];
                bar_sync(BAR_G0, 128);
#pragma unroll
                for (int j = 0; j < 32; j++) split(t[j], Th[i * CP + l0 + j], Tl[i * CP + l0 + j]);
            }
        } else {
            uint32_t qa[8][4];
            {
                const bf16* q0 = qg + (size_t)(cs + r0 + grp) * qk_stride + 2 * q;
                const bf16* q1 = q0 + 8 * (size_t)qk_stride;
                const bool v0 = r0 + grp < ce, v1 = r0 + grp + 8 < ce;
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    qa[ks][0] = v0 ? ld32(q0 + ks * 16) : 0u;
                    qa[ks][1] = v1 ? ld32(q1 + ks * 16) : 0u;
                    qa[ks][2] = v0 ? ld32(q0 + ks * 16 + 8) : 0u;
                    qa[ks][3] = v1 ? ld32(q1 + ks * 16 + 8) : 0u;
                }
                // L2-normalize both rows, bit for bit as l2_norm_bf16: lane q
                // of the quad holds pairs k * 8 + q (reg 0 / 1) and + 4 (reg 2 / 3)
                // of rows grp and grp + 8, the layout l2_row_total takes.
                float inv[2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    float sq[8][2];
#pragma unroll
                    for (int ks = 0; ks < 8; ks++)
#pragma unroll
                        for (int f = 0; f < 2; f++) {
                            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&qa[ks][h + 2 * f]);
                            sq[ks][f] = pair_sq(__low2float(v), __high2float(v));
                        }
                    inv[h] = rsqrtf(l2_row_total(sq, lane) + L2_EPS);
                }
#pragma unroll
                for (int ks = 0; ks < 8; ks++)
#pragma unroll
                    for (int f = 0; f < 4; f++) {
                        const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&qa[ks][f]);
                        const __nv_bfloat162 o = __floats2bfloat162_rn(__low2float(v) * inv[f & 1], __high2float(v) * inv[f & 1]);
                        qa[ks][f] = *reinterpret_cast<const uint32_t*>(&o);
                    }
            }
            float kS[8][4] = {};
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                if (!rows_live) break;
                uint32_t ka[4];
                lda(ka, Ks, KP, r0, ks * 16, grp, q);
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    uint32_t h0, h1, l0, l1;
                    ldb(h0, h1, Sh, KP, nt * 8, ks * 16, grp, q);
                    ldb(l0, l1, Sl, KP, nt * 8, ks * 16, grp, q);
                    mma(kS[nt], ka, h0, h1);
                    mma(kS[nt], ka, l0, l1);
                }
            }
#pragma unroll
            for (int ks = 0; ks < 8; ks++)
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    if (!rows_live) break;
                    uint32_t b0, b1, h0, h1;
                    ldb(h0, h1, Sh, KP, nt * 8, ks * 16, grp, q);
                    mma(o1[nt], qa[ks], h0, h1);
                    ldb(b0, b1, Ks, KP, nt * 8, ks * 16, grp, q);
                    mma(pk[nt], qa[ks], b0, b1);
                }
            bar_sync(BAR_GATES, 256);   // gc, egc, dec from G0
            // R = V - e^gc (K S) -> R^T hi/lo; e^gc Q S and P stay in registers.
            // Every operand is loaded before the first store (see G0's L).
            float gr[2], er[2], gcol[8][2], vr[8][4];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                gr[h] = gc[r0 + grp + 8 * h];
                er[h] = egc[r0 + grp + 8 * h];
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int t = 0; t < 2; t++) gcol[nt][t] = gc[nt * 8 + 2 * q + t];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(
                        Vs + (r0 + grp + 8 * h) * VP + nt * 8 + 2 * q);
                    vr[nt][2 * h] = __low2float(v2);
                    vr[nt][2 * h + 1] = __high2float(v2);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int i = r0 + erow(e, grp), c = nt * 8 + ecol(e, q);
                    const float r = vr[nt][e] - er[e >> 1] * kS[nt][e];
                    split(r, Rh[c * CP + i], Rl[c * CP + i]);
                    o1[nt][e] *= er[e >> 1];
                    const float p = __expf(fminf(gr[e >> 1] - gcol[nt][e & 1], 0.f)) * pk[nt][e];
                    pk[nt][e] = c <= i ? p : 0.f;   // branch-free, as G0's L
                }
        }
        __syncthreads();   // join: T diag(beta) (G0), R (G1)
        if (cen > 0) {     // V is read: stage the next chunk's
            stage_rows<DV, VP>(Vs, vg, v_stride, cn, cen, tid);
            cp_commit();
        }

        // ── uc = T diag(beta) R: all warps, rows 16 mg, columns 32 nh ──
        float uc[4][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            if (16 * mg >= ce || ks >= live) break;
            uint32_t th[4], tl[4];
            lda(th, Th, CP, 16 * mg, ks * 16, grp, q);
            lda(tl, Tl, CP, 16 * mg, ks * 16, grp, q);
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                uint32_t h0, h1, l0, l1;
                ldb(h0, h1, Rh, CP, 32 * nh + nt * 8, ks * 16, grp, q);
                ldb(l0, l1, Rl, CP, 32 * nh + nt * 8, ks * 16, grp, q);
                mma(uc[nt], th, h0, h1);
                mma(uc[nt], th, l0, l1);
                mma(uc[nt], tl, h0, h1);
            }
        }
        __syncthreads();
        // uc^T hi/lo over R^T; G1's P hi/lo over T.
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int i = 16 * mg + erow(e, grp), c = 32 * nh + nt * 8 + ecol(e, q);
                split(uc[nt][e], Rh[c * CP + i], Rl[c * CP + i]);
            }
        if (warp >= 4) {
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int i = r0 + erow(e, grp), c = nt * 8 + ecol(e, q);
                    split(pk[nt][e], Th[i * CP + c], Tl[i * CP + c]);
                }
        }
        __syncthreads();

        // ── G1: O = e^gc Q S + P uc (P hi/lo, uc hi: the FLA output pass reads a BF16 uc) ──
        if (warp >= 4) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                if (!rows_live || ks >= live) break;
                uint32_t ph[4], pl[4];
                lda(ph, Th, CP, r0, ks * 16, grp, q);
                lda(pl, Tl, CP, r0, ks * 16, grp, q);
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    uint32_t h0, h1;
                    ldb(h0, h1, Rh, CP, nt * 8, ks * 16, grp, q);
                    mma(o1[nt], ph, h0, h1);
                    mma(o1[nt], pl, h0, h1);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int i = r0 + grp + 8 * h, c = nt * 8 + 2 * q;
                    if (i < ce)
                        *reinterpret_cast<__nv_bfloat162*>(og + (size_t)(cs + i) * ostride + c) =
                            __floats2bfloat162_rn(o1[nt][2 * h] * inv_sqrt_d, o1[nt][2 * h + 1] * inv_sqrt_d);
                }
        }
        const float elast = expf(gc[ce - 1]);

        // ── S = e^gc_last S + (K diag(decay))^T uc ──
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) S[nt][e] *= elast;
        const int k0 = 16 * warp + grp;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            if (ks >= live) break;
            uint32_t ah[4], al[4];
#pragma unroll
            for (int f = 0; f < 4; f++) {
                // a0 (row grp, i 2q..), a1 (row grp+8), a2 (row grp, i 2q+8..), a3 (row grp+8, i+8)
                const int kr = k0 + (f & 1) * 8, i = ks * 16 + 2 * q + (f >> 1) * 8;
                bf16 h0, l0, h1, l1;
                split(__bfloat162float(Ks[i * KP + kr]) * dec[i], h0, l0);
                split(__bfloat162float(Ks[(i + 1) * KP + kr]) * dec[i + 1], h1, l1);
                ah[f] = pack(h0, h1);
                al[f] = pack(l0, l1);
            }
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t h0, h1, l0, l1;
                ldb(h0, h1, Rh, CP, nt * 8, ks * 16, grp, q);
                ldb(l0, l1, Rl, CP, nt * 8, ks * 16, grp, q);
                mma(S[nt], ah, h0, h1);
                mma(S[nt], ah, l0, l1);
                mma(S[nt], al, h0, h1);
            }
        }
        __syncthreads();   // every read of K, S^T and uc^T for this chunk is done
        if (cen > 0) {
            stage_rows<KD, KP>(Ks, kg, qk_stride, cn, cen, tid);
            cp_commit();
        }
        stage_state(S, Sh, Sl, warp, grp, q);
    }

#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++)
            H[(16 * warp + erow(e, grp)) * VD + nt * 8 + ecol(e, q)] = S[nt][e];
}

// causal_conv1d_update_prefill_tp (common/causal_conv1d.cu) over several
// streams in one launch, each stream b over its rows [cu[b], cu[b + 1]) of
// the input / output, reading its incoming window from init[b] and writing
// the outgoing one to state[b] (a score branch starts from its head's window
// with no copy; init[b] == state[b] is the in-place update). Per channel and
// token the arithmetic is the single-stream kernel's.
// Grid (ceil(dim / 32), ceil(max_len / 64), batch), block (32, 8).
extern "C" __global__ void __launch_bounds__(256, 4)
gdn_chunk_tc_conv1d_varlen(
    float* const* __restrict__ state,
    const float* const* __restrict__ init,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ cu_seqlens,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int input_stride,
    unsigned int output_stride,
    unsigned int batch_size
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch_size) return;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    const unsigned int tok0 = (unsigned int)cu_seqlens[b];
    const unsigned int seq_len = (unsigned int)cu_seqlens[b + 1] - tok0;
    const unsigned int t0 = (blockIdx.y * blockDim.y + threadIdx.y) * 8u;
    if (t0 >= seq_len) return;
    input += (unsigned long long)tok0 * input_stride;
    output += (unsigned long long)tok0 * output_stride;

    const float* st_in = init[b] + (unsigned long long)ch * d_conv;
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    float w_reg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (unsigned int k = 0; k < 4; k++)
        if (k < d_conv) w_reg[k] = __bfloat162float(w[k]);

    auto xin = [&](long long t) -> float {
        if (t >= 0) {
            return (t < (long long)seq_len) ? __bfloat162float(input[(unsigned long long)t * input_stride + ch])
                                            : 0.0f;
        }
        const long long idx = (long long)d_conv + t;
        return (idx >= 0) ? st_in[idx] : 0.0f;
    };

    float s0 = xin((long long)t0 - 3);
    float s1 = xin((long long)t0 - 2);
    float s2 = xin((long long)t0 - 1);
#pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        const unsigned int t = t0 + i;
        if (t >= seq_len) break;
        const float s3 = xin((long long)t);
        const float acc = b_val + s0 * w_reg[0] + s1 * w_reg[1] + s2 * w_reg[2] + s3 * w_reg[3];
        const float sig = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sig);
        s0 = s1; s1 = s2; s2 = s3;
    }

    // As the single-stream kernel: the thread that read the incoming window
    // writes the outgoing one, reading before ascending k overwrites (only
    // matters in place).
    if (t0 == 0u) {
        float* st = state[b] + (unsigned long long)ch * d_conv;
#pragma unroll
        for (unsigned int k = 0; k < 4; k++)
            if (k < d_conv) st[k] = xin((long long)seq_len - (long long)d_conv + (long long)k);
    }
}
