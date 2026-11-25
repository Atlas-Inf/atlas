// SPDX-License-Identifier: AGPL-3.0-only

// Inkling 2 DSpark drafter (`mtp.safetensors`) pieces around the DFlash head:
// the per-layer 4-tap feature mix, the block's anchor/mask input, and the
// recurrent (`rnn_hidden`) Markov head. Numerics follow vLLM PR #69028
// (`models/inkling2/nvidia/ops/dspark.py`, Apache-2.0): fp32 math, BF16
// rounding where the reference's torch ops round.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>

#define MIX_THREADS 256
#define MIX_MAX_H 4096
#define RNN_RANK 256

__device__ __forceinline__ float ink_round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float ink_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const unsigned int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (unsigned int i = 0; i < blockDim.x / 32; i++) t += red[i];
    return t;
}

// out[r] = BF16(rms(sum_t alpha[t] * x[r, t]) * w) over `hidden` (fp32 math). When `tap3_norm`
// is set, tap 3 (the target's last visit, before its final norm) first becomes the target's
// final-norm output BF16(rms(x3) * tap3_norm), the feature the reference captures.
// x row r starts at x + r * row_stride; tap t at + t * hidden. grid (rows), block 256.
extern "C" __global__ void __launch_bounds__(MIX_THREADS) inkling2_dspark_mix(
    const __nv_bfloat16* __restrict__ x, const unsigned int row_stride,
    const float a0, const float a1, const float a2, const float a3,
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ tap3_norm,
    const float eps, const float tap3_eps, __nv_bfloat16* __restrict__ out, const unsigned int hidden
) {
    atlas_pdl_enter();
    __shared__ float mixed[MIX_MAX_H];
    __shared__ float red[MIX_THREADS / 32];
    const __nv_bfloat16* row = x + (unsigned long long)blockIdx.x * row_stride;
    float inv3 = 1.0f;
    if (tap3_norm != nullptr) {
        float ss = 0.0f;
        for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __bfloat162float(row[3 * hidden + i]);
            ss += v * v;
        }
        inv3 = rsqrtf(ink_block_sum(ss, red) / hidden + tap3_eps);
    }
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float t3 = __bfloat162float(row[3 * hidden + i]);
        if (tap3_norm != nullptr) t3 = ink_round_bf16(t3 * inv3 * __bfloat162float(tap3_norm[i]));
        const float m = __bfloat162float(row[i]) * a0 + __bfloat162float(row[hidden + i]) * a1
                      + __bfloat162float(row[2 * hidden + i]) * a2 + t3 * a3;
        mixed[i] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(ink_block_sum(ss, red) / hidden + eps);
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[(unsigned long long)blockIdx.x * hidden + i] = __float2bfloat16(mixed[i] * inv * __bfloat162float(w[i]));
}

// The block input in place over `rows` rows of `hidden`: row 0 holds the anchor's target
// embedding and stays (the target's table is normed at load; a non-NULL `embed_norm`
// norms a raw row here instead: BF16(rms(row 0) * embed_norm)); rows 1.. = mask_embed.
// grid (rows, sequences): sequence y's block starts at row y * rows. block 256.
extern "C" __global__ void __launch_bounds__(MIX_THREADS) inkling2_dspark_block_input(
    __nv_bfloat16* __restrict__ block, const __nv_bfloat16* __restrict__ embed_norm,
    const __nv_bfloat16* __restrict__ mask_embed, const float eps, const unsigned int hidden
) {
    atlas_pdl_enter();
    __shared__ float red[MIX_THREADS / 32];
    __nv_bfloat16* row = block + ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * hidden;
    if (blockIdx.x > 0) {
        for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) row[i] = mask_embed[i];
        return;
    }
    if (embed_norm == nullptr) return;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = __bfloat162float(row[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(ink_block_sum(ss, red) / hidden + eps);
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x)
        row[i] = __float2bfloat16(__bfloat162float(row[i]) * inv * __bfloat162float(embed_norm[i]));
}

// The recurrent head's per-propose context: ctx[s, o] = b[o] + W[o, 2R:] . [h_s ; f] in fp32,
// for draft steps s < steps (h_s = the drafter's final-normed block row s + 1, f = f_rnn).
// W [3R, 2R + 2H] row-major. One warp per output o; grid (3R / 8, sequences), block 256:
// sequence y reads h + y * h_stride and f + y * f_stride, writes ctx + y * ctx_stride.
extern "C" __global__ void __launch_bounds__(256) inkling2_dspark_rnn_ctx(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ f,
    float* __restrict__ ctx, const unsigned int steps, const unsigned int hidden,
    const unsigned int h_stride, const unsigned int f_stride, const unsigned int ctx_stride
) {
    atlas_pdl_enter();
    h += (unsigned long long)blockIdx.y * h_stride;
    f += (unsigned long long)blockIdx.y * f_stride;
    ctx += (unsigned long long)blockIdx.y * ctx_stride;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int o = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int outs = 3 * RNN_RANK, width = 2 * RNN_RANK + 2 * hidden;
    if (o >= outs) return;
    const __nv_bfloat16* wr = W + (unsigned long long)o * width + 2 * RNN_RANK;
    float fpart = 0.0f;
    for (unsigned int k = lane; k < hidden; k += 32) fpart += __bfloat162float(wr[hidden + k]) * __bfloat162float(f[k]);
    for (int off = 16; off > 0; off >>= 1) fpart += __shfl_xor_sync(0xffffffff, fpart, off);
    for (unsigned int s = 0; s < steps; s++) {
        const __nv_bfloat16* hs = h + (unsigned long long)(s + 1) * hidden;
        float acc = 0.0f;
        for (unsigned int k = lane; k < hidden; k += 32) acc += __bfloat162float(wr[k]) * __bfloat162float(hs[k]);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) ctx[s * outs + o] = __bfloat162float(bias[o]) + fpart + acc;
    }
}

// One recurrence step's raw gates: raw[o] = ctx[o] + W[o, :R] . state + W[o, R:2R] . w1[prev]
// (fp32). `prev` is a device token id. One warp per output; grid (3R / 8, sequences), block
// 256: sequence y's ctx, state, prev and raw are its own, at y times their strides.
extern "C" __global__ void __launch_bounds__(256) inkling2_dspark_rnn_raw(
    const __nv_bfloat16* __restrict__ W, const float* __restrict__ ctx,
    const __nv_bfloat16* __restrict__ state, const __nv_bfloat16* __restrict__ w1,
    const unsigned int* __restrict__ prev, float* __restrict__ raw, const unsigned int hidden,
    const unsigned int ctx_stride, const unsigned int state_stride, const unsigned int prev_stride,
    const unsigned int raw_stride
) {
    atlas_pdl_enter();
    ctx += (unsigned long long)blockIdx.y * ctx_stride;
    state += (unsigned long long)blockIdx.y * state_stride;
    prev += (unsigned long long)blockIdx.y * prev_stride;
    raw += (unsigned long long)blockIdx.y * raw_stride;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int o = blockIdx.x * 8 + threadIdx.x / 32;
    if (o >= 3 * RNN_RANK) return;
    const __nv_bfloat16* wr = W + (unsigned long long)o * (2 * RNN_RANK + 2 * hidden);
    const __nv_bfloat16* emb = w1 + (unsigned long long)(*prev) * RNN_RANK;
    float acc = 0.0f;
    for (unsigned int k = lane; k < RNN_RANK; k += 32)
        acc += __bfloat162float(wr[k]) * __bfloat162float(state[k])
             + __bfloat162float(wr[RNN_RANK + k]) * __bfloat162float(emb[k]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0) raw[o] = ctx[o] + acc;
}

// The gating, as the reference's BF16 torch ops (each op rounds): g = sigmoid(bf16(raw_g)),
// state = bf16(g * state) + bf16(bf16(1 - g) * tanh(bf16(raw_c))), latent = tanh(bf16(raw_o)).
// `latent` gets [tanh(o) (R), 1.0, 0 ...] up to `latent_len`: the fold table's bias column.
// grid (sequences), block R: sequence x's raw, state and latent at x times their strides.
extern "C" __global__ void __launch_bounds__(RNN_RANK) inkling2_dspark_rnn_gate(
    const float* __restrict__ raw, __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ latent, const unsigned int latent_len,
    const unsigned int raw_stride, const unsigned int state_stride, const unsigned int latent_stride
) {
    atlas_pdl_enter();
    raw += (unsigned long long)blockIdx.x * raw_stride;
    state += (unsigned long long)blockIdx.x * state_stride;
    latent += (unsigned long long)blockIdx.x * latent_stride;
    const unsigned int c = threadIdx.x;
    const float g = ink_round_bf16(raw[c]);
    const float cand = ink_round_bf16(raw[RNN_RANK + c]);
    const float o = ink_round_bf16(raw[2 * RNN_RANK + c]);
    const float s = __bfloat162float(state[c]);
    const float gate = ink_round_bf16(__frcp_rn(1.0f + expf(-g)));
    const float keep = ink_round_bf16(gate * s);
    const float update = ink_round_bf16(ink_round_bf16(1.0f - gate) * ink_round_bf16(tanhf(cand)));
    state[c] = __float2bfloat16(keep + update);
    latent[c] = __float2bfloat16(tanhf(o));
    for (unsigned int i = RNN_RANK + c; i < latent_len; i += RNN_RANK) latent[i] = __float2bfloat16(i == RNN_RANK ? 1.0f : 0.0f);
}

// A draft row's pick and the drafter's confidence in it: token = argmax(x) (the lowest index
// among equal maxima, as `argmax_bf16`) of x = logits, or with `bias` of x = BF16(logits + bias)
// (as `bf16_residual_add` would leave the row); conf = log softmax(softcap(x))[token] with
// softcap(x) = cap * tanh(x / cap) (cap 0: none), the reference's `markov_logits`, written only
// with `conf`. NaN reads as the lowest value. grid (sequences), block 1024: sequence s reads
// logits + s * logits_stride and bias + s * bias_stride, writes token + s * token_stride and
// conf + s * conf_stride.
extern "C" __global__ void __launch_bounds__(1024) inkling2_dspark_pick(
    const __nv_bfloat16* __restrict__ logits, const unsigned long long logits_stride,
    const __nv_bfloat16* __restrict__ bias, const unsigned int bias_stride,
    unsigned int* __restrict__ token, const unsigned int token_stride,
    float* __restrict__ conf, const unsigned int conf_stride, const unsigned int n, const float cap
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    const unsigned int tid = threadIdx.x, stride = blockDim.x, seq = blockIdx.x;
    logits += seq * logits_stride;
    if (bias != nullptr) bias += (unsigned long long)seq * bias_stride;
    auto value = [&](unsigned int i) {
        const float l = __bfloat162float(logits[i]);
        return bias == nullptr ? l : ink_round_bf16(l + __bfloat162float(bias[i]));
    };
    float best = -1e30f;
    unsigned int at = 0;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = value(i);
        if (v > best) { best = v; at = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = at;
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s && (s_val[tid + s] > s_val[tid]
                        || (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid]))) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    const float top_raw = s_val[0];
    const unsigned int pick = s_idx[0];
    if (conf == nullptr) {
        if (tid == 0) token[(unsigned long long)seq * token_stride] = pick;
        return;
    }
    const float top = cap > 0.0f ? cap * tanhf(top_raw / cap) : top_raw;
    __syncthreads();
    // sum over the row of exp(softcap(x) - softcap(max)): at least 1 (the pick itself)
    float mass = 0.0f;
    for (unsigned int i = tid; i < n; i += stride) {
        const float v = value(i);
        if (v == v) mass += expf((cap > 0.0f ? cap * tanhf(v / cap) : v) - top);
    }
    s_val[tid] = mass;
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) s_val[tid] += s_val[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        token[(unsigned long long)seq * token_stride] = pick;
        conf[(unsigned long long)seq * conf_stride] = -logf(fmaxf(s_val[0], 1.0f));
    }
}
