// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 on AMD (gfx1151): elementwise decode fusions over the gb10 target's
// kernels. Each fused launch keeps the unfused arithmetic bit for bit — same
// operands in the same order, the same BF16 roundings at the same points — so a
// greedy decode is identical to the unfused chain; what goes away is only the
// launch overhead between them (rows 1..16, C1-C8 and verify).
//
//   inkling2_attn_prologue        k/v short convs + the four per-head epilogue
//                                 norms (q, k, v, rel) + the paged KV write
//   inkling2_mlp_tail             moe_combine + post-MLP conv + layer_out
//   inkling2_block_res_sums_resid the residual + boundary block push folded
//                                 into block_res_sums on its own 9-CTA grid

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define INKF_MAX_HD 128
#define INKF_MAX_K 4
#define INKF_MAX_STATES 9
#define INKF_MAX_Q 4
#define INKF_THREADS 256

// ---------------------------------------------------------------------------
// Attention prologue: one CTA (8 warps) a row.
// Warps 0..kv_heads-1 each take one kv head: its short conv (state roll when the
// row is alone in its sequence — decode), then the head's clip (k) / rms (v)
// norm over the channels the warp just produced, then that head's slice of the
// paged KV write (dst = cache + slot * kv_dim + head * head_dim: block_idx *
// block_size + block_offset collapses back to slot). All 8 warps then split the
// q and rel heads' clip norms, exactly `inkling2_head_norm` mode 0 each.
//
// `inkling2_sconv` with roll=1 has the (sequence, channel) thread roll the
// state it just read; the same thread owns that state here, and a row's conv
// reads only x inputs or its own sequence's state, so the fusion adds no
// cross-CTA dependency. With rows_per_seq > 1 (verify) the conv runs without
// the roll and the caller launches `inkling2_sconv_state` for k and v after, as
// the unfused path does.

// One kv head's conv on this lane's channels: out the BF16 value, xc the fp32
// current input. Bit-identical to inkling2_sconv (same loop, same taps).
__device__ __forceinline__ float inkf_sconv_channel(
    const __nv_bfloat16* __restrict__ xlin, const float* __restrict__ st,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ dyn,
    unsigned int r, unsigned int first, unsigned int c, unsigned int kv_dim,
    unsigned int k, unsigned int dyn_stride, unsigned int head_dim, float* xc) {
    *xc = __bfloat162float(xlin[(unsigned long long)r * kv_dim + c]);
    float acc = 0.0f;
    for (unsigned int lag = 0; lag < k; lag++) {
        const int src = (int)r - (int)lag;
        const float xv = src >= (int)first
            ? __bfloat162float(xlin[(unsigned long long)src * kv_dim + c])
            : st[(int)(k - 1) - ((int)first - src)];
        float tap;
        if (dyn == nullptr) {
            tap = __bfloat162float(w[(unsigned long long)c * k + (k - 1 - lag)]);
        } else {
            tap = __bfloat162float(w[(unsigned long long)lag * kv_dim + c])
                + __bfloat162float(dyn[(unsigned long long)r * dyn_stride + (c / head_dim) * k + lag]);
        }
        acc += tap * xv;
    }
    const float denom = dyn == nullptr ? (float)(k + 1) : (float)(2 * k + 1);
    return (acc + *xc) * rsqrtf(denom);
}

// mode 0 (clip) / mode 1 (rms) over the warp's own channels: the head_norm math
// of inkling2_head_ops.cu on values already in registers.
__device__ __forceinline__ float inkf_head_factor(float ss, unsigned int head_dim,
                                                  unsigned int mode, float value) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    return mode == 0 ? fminf(value / (sqrtf(ss) + 1e-6f), 1.0f)
                     : rsqrtf(ss / (float)head_dim + value);
}

extern "C" __global__ void __launch_bounds__(256) inkling2_attn_prologue(
    const __nv_bfloat16* __restrict__ k_lin,   // [rows, kv_dim]
    const __nv_bfloat16* __restrict__ v_lin,
    const __nv_bfloat16* __restrict__ q_in,    // [rows, heads * head_dim]
    const __nv_bfloat16* __restrict__ r_in,    // [rows, heads * d_rel]
    __nv_bfloat16* __restrict__ kc,
    __nv_bfloat16* __restrict__ vc,
    __nv_bfloat16* __restrict__ qn,
    __nv_bfloat16* __restrict__ kn,
    __nv_bfloat16* __restrict__ vn,
    __nv_bfloat16* __restrict__ rn,
    float* const* __restrict__ states,         // [num_seqs] state bases
    const unsigned int state_off_k,
    const unsigned int state_off_v,
    const __nv_bfloat16* __restrict__ k_w,     // static [C, k] / dynamic [k, C]
    const __nv_bfloat16* __restrict__ v_w,
    const __nv_bfloat16* __restrict__ k_dyn,   // dynamic taps or NULL
    const __nv_bfloat16* __restrict__ v_dyn,
    const __nv_bfloat16* __restrict__ q_scale,     // [heads, head_dim]
    const __nv_bfloat16* __restrict__ rel_scale,   // [heads, d_rel]
    const int* __restrict__ positions,             // [rows] or NULL (SWA)
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const long long* __restrict__ slots,           // [rows] slot_mapping
    const unsigned int rows_per_seq,
    const unsigned int heads,
    const unsigned int kv_heads,
    const unsigned int head_dim,
    const unsigned int d_rel,
    const unsigned int conv_k,
    const unsigned int dyn_stride,
    const float qk_clip,
    const float rel_clip,
    const float eps,
    const float n_floor,
    const float alpha
) {
    atlas_pdl_enter();
    const unsigned int r = blockIdx.x;
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int kv_dim = kv_heads * head_dim;
    const unsigned int s = r / rows_per_seq;
    const unsigned int first = s * rows_per_seq;
    const unsigned int roll = (rows_per_seq == 1);
    const long long slot = slots[r];

    if (warp < kv_heads) {
        // k then v: conv channels head*head_dim + lane + 32i, norm, cache write
        for (int t = 0; t < 2; t++) {
            const __nv_bfloat16* xlin = t ? v_lin : k_lin;
            const __nv_bfloat16* cw = t ? v_w : k_w;
            const __nv_bfloat16* dyn = t ? v_dyn : k_dyn;
            __nv_bfloat16* convo = t ? vc : kc;
            __nv_bfloat16* normed = t ? vn : kn;
            __nv_bfloat16* cache = t ? v_cache : k_cache;
            const unsigned int off = t ? state_off_v : state_off_k;
            float v[INKF_MAX_HD / 32], xc[INKF_MAX_HD / 32];
#pragma unroll
            for (int i = 0; i < INKF_MAX_HD / 32; i++) {
                const unsigned int e = lane + 32 * i;
                v[i] = 0.0f;
                xc[i] = 0.0f;
                if (e < head_dim) {
                    const unsigned int c = warp * head_dim + e;
                    float* st = states[s] + off + (unsigned long long)c * (conv_k - 1);
                    const float o = inkf_sconv_channel(xlin, st, cw, dyn, r, first, c, kv_dim,
                                                       conv_k, dyn_stride, head_dim, &xc[i]);
                    const __nv_bfloat16 ob = __float2bfloat16(o);
                    convo[(unsigned long long)r * kv_dim + c] = ob;
                    v[i] = __bfloat162float(ob);
                    if (roll) {
                        for (unsigned int j = 0; j + 2 < conv_k; j++) st[j] = st[j + 1];
                        st[conv_k - 2] = xc[i];
                    }
                }
            }
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < INKF_MAX_HD / 32; i++) ss += v[i] * v[i];
            // k: mode 0 with qk_clip; v: mode 1 with eps
            const float f = inkf_head_factor(ss, head_dim, t, t ? eps : qk_clip);
            __nv_bfloat16* cd = slot >= 0 ? cache + (unsigned long long)slot * kv_dim + warp * head_dim
                                        : nullptr;
#pragma unroll
            for (int i = 0; i < INKF_MAX_HD / 32; i++) {
                const unsigned int e = lane + 32 * i;
                if (e >= head_dim) continue;
                const __nv_bfloat16 o = __float2bfloat16(v[i] * f);
                normed[(unsigned long long)r * kv_dim + warp * head_dim + e] = o;
                if (cd != nullptr) cd[e] = o;
            }
        }
    }

    // q and rel heads: mode 0 clip norms, positions giving the log-length scale
    for (unsigned int t = warp; t < 2 * heads; t += 8) {
        const bool is_q = t < heads;
        const unsigned int head = is_q ? t : t - heads;
        const unsigned int hd = is_q ? head_dim : d_rel;
        const unsigned int stride = heads * hd;
        const __nv_bfloat16* xp = (is_q ? q_in : r_in) + (unsigned long long)r * stride + head * hd;
        const __nv_bfloat16* scale = is_q ? q_scale : rel_scale;
        const float thr = is_q ? qk_clip : rel_clip;
        float v[INKF_MAX_HD / 32], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < INKF_MAX_HD / 32; i++) {
            const unsigned int e = lane + 32 * i;
            float val = 0.0f;
            if (e < hd) {
                val = __bfloat162float(xp[e]);
                if (scale != nullptr) val *= __bfloat162float(scale[head * hd + e]);
            }
            v[i] = val;
            ss += val * val;
        }
        const float f = inkf_head_factor(ss, hd, 0, thr);
        float tau = 1.0f;
        if (positions != nullptr)
            tau = 1.0f + alpha * logf(fmaxf((float)(positions[r] + 1) / n_floor, 1.0f));
        __nv_bfloat16* yp = (is_q ? qn : rn) + (unsigned long long)r * stride + head * hd;
#pragma unroll
        for (int i = 0; i < INKF_MAX_HD / 32; i++) {
            const unsigned int e = lane + 32 * i;
            if (e >= hd) continue;
            __nv_bfloat16 o = __float2bfloat16(v[i] * f);
            if (positions != nullptr) o = __float2bfloat16(__bfloat162float(o) * tau);
            yp[e] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// MLP tail, decode rows only (rows_per_seq == 1: the conv's state roll touches
// only this row's state and its lag inputs all come from that state). One
// thread a (row, channel): the top-k weighted combine (inkling2_moe_combine),
// then the static post-MLP conv with its roll (inkling2_sconv), then
// layer_out's bf16(resid + conv). Bit-identical to the three launches.
// top_k == 0: moe_out is already combined (the batched-expert unpermute wrote
// it); the conv reads it directly and only conv + layer_out fuse.
extern "C" __global__ void __launch_bounds__(256) inkling2_mlp_tail(
    const __nv_bfloat16* __restrict__ y,       // [rows, top_k, H]
    const float* __restrict__ w,             // [rows, top_k]
    __nv_bfloat16* __restrict__ moe_out,     // [rows, H] (combine out)
    __nv_bfloat16* __restrict__ mlp_conv,    // [rows, H] (conv out)
    float* const* __restrict__ states,       // [num_seqs]
    const unsigned int state_off,
    const __nv_bfloat16* __restrict__ conv_w, // [H, k] static taps
    const float* __restrict__ resid,         // [rows, H]
    __nv_bfloat16* __restrict__ hidden,      // [rows, H]
    const unsigned int H,
    const unsigned int top_k,
    const unsigned int conv_k
) {
    atlas_pdl_enter();
    const unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int r = blockIdx.y;
    if (e >= H) return;
    __nv_bfloat16 comb;
    if (top_k > 0) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < top_k; s++)
            acc += __bfloat162float(y[((unsigned long long)r * top_k + s) * H + e]) * w[r * top_k + s];
        comb = __float2bfloat16(acc);
        moe_out[(unsigned long long)r * H + e] = comb;
    } else {
        comb = moe_out[(unsigned long long)r * H + e];
    }

    // the conv over this channel: rows_per_seq == 1, so every lag reads the
    // state (newest last) and the roll is this thread's own
    float* st = states[r] + state_off + (unsigned long long)e * (conv_k - 1);
    const float xc = __bfloat162float(comb);
    float cacc = 0.0f;
    for (unsigned int lag = 0; lag < conv_k; lag++) {
        const float xv = lag == 0 ? xc : st[conv_k - 1 - lag];
        cacc += __bfloat162float(conv_w[(unsigned long long)e * conv_k + (conv_k - 1 - lag)]) * xv;
    }
    const __nv_bfloat16 co = __float2bfloat16((cacc + xc) * rsqrtf((float)(conv_k + 1)));
    mlp_conv[(unsigned long long)r * H + e] = co;
    for (unsigned int j = 0; j + 2 < conv_k; j++) st[j] = st[j + 1];
    st[conv_k - 2] = xc;

    hidden[(unsigned long long)r * H + e] = __float2bfloat16(resid[(unsigned long long)r * H + e] + __bfloat162float(co));
}

// ---------------------------------------------------------------------------
// Block-residual sums with the MLP residual and the boundary block push folded
// in, on inkling2_block_res_sums' own (chunks, rows) grid: each CTA covers the
// channels e = chunk * INK_THREADS + tid (mod chunks * INK_THREADS), computes
// resid = (push ? 0 : x) + c for exactly those channels, pushes x's chunk slice
// into block nb-1, then runs the unchanged sums pass. Every state element it
// reads is either an old block or this thread's own just-written value.
#define INKF_RES_PER (INKF_MAX_STATES * (1 + INKF_MAX_Q))
__device__ __forceinline__ float inkf_state(const __nv_bfloat16* __restrict__ blocks,
                                            const float* __restrict__ pf,
                                            unsigned long long row, unsigned int row_stride,
                                            unsigned long long block_stride, unsigned int H,
                                            unsigned int nb, unsigned int j, unsigned int e) {
    if (j < nb) return __bfloat162float(blocks[row * row_stride + (unsigned long long)j * block_stride + e]);
    return pf[row * H + e];
}

extern "C" __global__ void __launch_bounds__(256) inkling2_block_res_sums_resid(
    __nv_bfloat16* __restrict__ blocks,        // gets the pushed slice at block nb-1
    const __nv_bfloat16* __restrict__ resid_x, // [rows, H] residual input x
    const __nv_bfloat16* __restrict__ resid_c, // [rows, H] conv output
    float* __restrict__ resid,                 // [rows, H] fp32 out (= the partial)
    const float* __restrict__ queries,         // [nq, H]
    float* __restrict__ sums_all,              // [rows, chunks, INKF_RES_PER]
    const unsigned int H,
    const unsigned int nb,                     // saved blocks incl. the pushed one
    const unsigned int row_stride,
    const unsigned long long block_stride,
    const unsigned int nq,
    const unsigned int push                    // 1: boundary visit
) {
    atlas_pdl_enter();
    const unsigned long long row = blockIdx.y;
    const unsigned int chunk = blockIdx.x, chunks = gridDim.x;
    for (unsigned int e = chunk * INKF_THREADS + threadIdx.x; e < H; e += chunks * INKF_THREADS) {
        if (push) blocks[row * row_stride + (unsigned long long)(nb - 1) * block_stride + e] =
                      resid_x[row * H + e];
        resid[row * H + e] = (push ? 0.0f : __bfloat162float(resid_x[row * H + e]))
                           + __bfloat162float(resid_c[row * H + e]);
    }

    // ink_res_sums of inkling2_block_res.cu, verbatim
    const unsigned int ns = nb + 1;
    float acc[INKF_MAX_STATES][1 + INKF_MAX_Q];
#pragma unroll
    for (int j = 0; j < INKF_MAX_STATES; j++)
#pragma unroll
        for (int q = 0; q <= INKF_MAX_Q; q++) acc[j][q] = 0.0f;
    for (unsigned int e = chunk * INKF_THREADS + threadIdx.x; e < H; e += chunks * INKF_THREADS) {
        float qv[INKF_MAX_Q];
#pragma unroll
        for (int q = 0; q < INKF_MAX_Q; q++) qv[q] = q < (int)nq ? queries[(unsigned long long)q * H + e] : 0.0f;
#pragma unroll
        for (int j = 0; j < INKF_MAX_STATES; j++) {
            if (j >= (int)ns) break;
            const float sv = inkf_state(blocks, resid, row, row_stride, block_stride, H, nb, j, e);
            acc[j][0] += sv * sv;
#pragma unroll
            for (int q = 0; q < INKF_MAX_Q; q++) acc[j][1 + q] += qv[q] * sv;
        }
    }
    __shared__ float red[INKF_THREADS / 32][INKF_MAX_STATES][1 + INKF_MAX_Q];
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
    for (int j = 0; j < INKF_MAX_STATES; j++)
#pragma unroll
        for (int q = 0; q <= INKF_MAX_Q; q++) {
            float v = acc[j][q];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0) red[warp][j][q] = v;
        }
    __syncthreads();
    float* sums = sums_all + (row * chunks + chunk) * INKF_RES_PER;
    if (threadIdx.x < INKF_RES_PER) {
        const unsigned int j = threadIdx.x / (1 + INKF_MAX_Q), q = threadIdx.x % (1 + INKF_MAX_Q);
        float v = 0.0f;
        for (unsigned int w = 0; w < INKF_THREADS / 32; w++) v += red[w][j][q];
        sums[threadIdx.x] = v;
    }
}
