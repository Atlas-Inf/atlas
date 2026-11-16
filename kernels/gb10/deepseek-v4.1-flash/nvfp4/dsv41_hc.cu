// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash Hyper-Connections: the residual stream is `hc` copies,
// BF16, and every sublayer sits between a collapse to one vector and an
// expansion back. The reference is `Block.hc_mixes / hc_pre / hc_post` in the
// release's `inference/model.py` and `hc_split_sinkhorn` in its `kernel.py`.
//
// V4.1 differs from DeepSeek-V4 in which coefficients a collapse uses: a
// sublayer is collapsed with the `pre` the sublayer BEFORE it computed
// (attention with the previous block's FFN site, the FFN with this block's
// attention site, the head with the last FFN site), and layer 0's attention
// with the one-hot (1, 0, 0, 0). So the mix, the collapse and the expansion
// are separate kernels here, and `pre` is handed from site to site.
//
// Layout. A thread owns one value index d of a token, in all its copies; a
// block is 256 consecutive d, so a token is H / 256 blocks and uses that many
// SMs. (One block a token keeps the 2 MB of mix weights on a single SM, which
// is what a token then waits for.) The mix is the one step that sums over
// the stream: a block leaves its share of every sum in `partials`, and one
// small block per token adds the shares in block order. Every sum has an
// order fixed by these two counts, so a token's bytes do not depend on the
// tokens beside it, on the rank, or on which kernel below computed them.
//
// `dsv41_hc_site_sums` + `dsv41_hc_site_finish` are a whole site in two
// launches, for the block that runs a token at a time: the sublayer before
// it into the stream, the mix, the collapse, the sublayer's norm and its FP8
// form. They call the lines of the plain kernels and leave their bytes.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

#define DSV41_HC_THREADS 256
#define DSV41_HC_WARPS (DSV41_HC_THREADS / 32)
#define DSV41_HC_MAX 4
// The rows of the mix: pre, post and comb; and, last, the sum of squares.
#define DSV41_MIX_ROWS ((2 + DSV41_HC_MAX) * DSV41_HC_MAX)
#define DSV41_MIX_SUMS (DSV41_MIX_ROWS + 1)

__device__ __forceinline__ float dsv41_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// One value of a collapse: the copies `v` weighted by `pre`.
__device__ __forceinline__ float dsv41_hc_collapse_one(
    const float* pre, const float* v, unsigned int hc) {
    float acc = pre[0] * v[0];
    for (unsigned int j = 1; j < hc; ++j) acc += pre[j] * v[j];
    return acc;
}

// One value of copy `j` of an expansion: the sublayer's `x` by `post`, plus
// the copies `rv` of the stream through column j of `comb`.
__device__ __forceinline__ float dsv41_hc_expand_one(
    const float* comb, const float* rv, float post, float x, unsigned int j, unsigned int hc) {
    float mixed = comb[j] * rv[0];
    for (unsigned int i = 1; i < hc; ++i) mixed += comb[i * hc + j] * rv[i];
    return post * x + mixed;
}

// A block's share of a token's mix sums, into `partial` [DSV41_MIX_SUMS]:
// thread `tid` adds its value index `d` of every copy (`v`, in copy order)
// into the sum of squares and every row's dot product; the warps reduce by
// shuffles, and the warps' sums are added in warp order.
__device__ __forceinline__ void dsv41_hc_block_sums(
    const float* v, const float* hc_fn, float* partial, unsigned int d, unsigned int H,
    unsigned int hc) {
    __shared__ float part[DSV41_HC_WARPS][DSV41_MIX_SUMS];
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int hc_dim = hc * H, mix_hc = (2 + hc) * hc;
    float acc[DSV41_MIX_SUMS];
    #pragma unroll
    for (unsigned int m = 0; m < DSV41_MIX_SUMS; ++m) acc[m] = 0.0f;
    if (d < H) {
        for (unsigned int j = 0; j < hc; ++j) {
            const unsigned int k = j * H + d;
            acc[DSV41_MIX_ROWS] += v[j] * v[j];
            #pragma unroll
            for (unsigned int m = 0; m < DSV41_MIX_ROWS; ++m)
                if (m < mix_hc) acc[m] += hc_fn[(size_t)m * hc_dim + k] * v[j];
        }
    }
    #pragma unroll
    for (unsigned int m = 0; m < DSV41_MIX_SUMS; ++m) {
        float sum = acc[m];
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) part[warp][m] = sum;
    }
    __syncthreads();
    if (tid < DSV41_MIX_SUMS) {
        float sum = 0.0f;
        for (unsigned int w = 0; w < DSV41_HC_WARPS; ++w) sum += part[w][tid];
        partial[tid] = sum;
    }
}

// One token's coefficients from its blocks' `partials` [blocks,
// DSV41_MIX_SUMS], on the lanes of one warp: the shares are added in block
// order; pre and post are a sigmoid each, comb a softmax by row and then the
// Sinkhorn rounds. `mix` and `comb` are shared scratch.
__device__ __forceinline__ void dsv41_hc_coefficients(
    const float* partials, unsigned int blocks, float* mix, volatile float* comb,
    unsigned int lane, unsigned int H, unsigned int hc, unsigned int sinkhorn_iters,
    float norm_eps, float hc_eps, const float* hc_scale, const float* hc_base, float* pre_out,
    float* post_out, float* comb_out) {
    if (lane < DSV41_MIX_SUMS) {
        float sum = 0.0f;
        for (unsigned int b = 0; b < blocks; ++b) sum += partials[b * DSV41_MIX_SUMS + lane];
        mix[lane] = sum;
    }
    __syncwarp();
    const float rsqrt = rsqrtf(mix[DSV41_MIX_ROWS] / (float)(hc * H) + norm_eps);
    if (lane < hc) {
        pre_out[lane] = dsv41_sigmoid(mix[lane] * rsqrt * hc_scale[0] + hc_base[lane]) + hc_eps;
        post_out[lane] =
            2.0f * dsv41_sigmoid(mix[lane + hc] * rsqrt * hc_scale[1] + hc_base[lane + hc]);
        // comb = softmax over this lane's row, plus eps.
        float row[DSV41_HC_MAX];
        float top = -INFINITY;
        for (unsigned int k = 0; k < hc; ++k) {
            const unsigned int i = lane * hc + k + hc * 2;
            row[k] = mix[i] * rsqrt * hc_scale[2] + hc_base[i];
            top = fmaxf(top, row[k]);
        }
        float sum = 0.0f;
        for (unsigned int k = 0; k < hc; ++k) {
            row[k] = expf(row[k] - top);
            sum += row[k];
        }
        for (unsigned int k = 0; k < hc; ++k) comb[lane * hc + k] = row[k] / sum + hc_eps;
    }
    __syncwarp();
    // Columns once, then `sinkhorn_iters - 1` rounds of rows and columns,
    // each dividing by (sum + eps). A lane owns a row in the row pass and a
    // column in the column pass.
    for (unsigned int round = 0; round < sinkhorn_iters; ++round) {
        if (round > 0) {
            if (lane < hc) {
                float sum = 0.0f;
                for (unsigned int k = 0; k < hc; ++k) sum += comb[lane * hc + k];
                for (unsigned int k = 0; k < hc; ++k)
                    comb[lane * hc + k] = comb[lane * hc + k] / (sum + hc_eps);
            }
            __syncwarp();
        }
        if (lane < hc) {
            float sum = 0.0f;
            for (unsigned int j = 0; j < hc; ++j) sum += comb[j * hc + lane];
            for (unsigned int j = 0; j < hc; ++j)
                comb[j * hc + lane] = comb[j * hc + lane] / (sum + hc_eps);
        }
        __syncwarp();
    }
    if (lane < hc * hc) comb_out[lane] = comb[lane];
}

// The mix of streams [T, hc, H] BF16, first launch: every block's share of
// the sums, partials [T, H / 256, DSV41_MIX_SUMS] FP32. hc_fn is
// [(2 + hc) * hc, hc * H] FP32. Grid (H / 256, T), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_mix_sums(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ hc_fn,
    float* __restrict__ partials,
    unsigned int H,
    unsigned int hc) {
    const unsigned int t = blockIdx.y, d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    const __nv_bfloat16* x = streams + (size_t)t * hc * H;
    float v[DSV41_HC_MAX];
    for (unsigned int j = 0; j < hc; ++j) v[j] = d < H ? __bfloat162float(x[j * H + d]) : 0.0f;
    dsv41_hc_block_sums(
        v, hc_fn, partials + ((size_t)t * gridDim.x + blockIdx.x) * DSV41_MIX_SUMS, d, H, hc);
}

// The mix, second launch: pre [T, hc], post [T, hc] and comb [T, hc, hc]
// from the partials. hc_scale [3], hc_base [(2 + hc) * hc], FP32. Grid (T),
// block 32.
extern "C" __global__ void __launch_bounds__(32) dsv41_hc_mix_finish(
    const float* __restrict__ partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    unsigned int H,
    unsigned int hc,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps) {
    const unsigned int t = blockIdx.x, blocks = H / DSV41_HC_THREADS;
    __shared__ float mix[DSV41_MIX_SUMS];
    __shared__ volatile float comb[DSV41_HC_MAX * DSV41_HC_MAX];
    dsv41_hc_coefficients(
        partials + (size_t)t * blocks * DSV41_MIX_SUMS, blocks, mix, comb, threadIdx.x, H, hc,
        sinkhorn_iters, norm_eps, hc_eps, hc_scale, hc_base, pre_out + (size_t)t * hc,
        post_out + (size_t)t * hc, comb_out + (size_t)t * hc * hc);
}

// y[t, d] = sum_j pre[t, j] * streams[t, j, d], rounded to BF16.
// Grid (ceil(H / 256), T), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_collapse(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__ pre,
    __nv_bfloat16* __restrict__ y,
    unsigned int H,
    unsigned int hc) {
    const unsigned int t = blockIdx.y, d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    if (d >= H) return;
    const __nv_bfloat16* x = streams + (size_t)t * hc * H;
    float v[DSV41_HC_MAX];
    for (unsigned int j = 0; j < hc; ++j) v[j] = __bfloat162float(x[j * H + d]);
    y[(size_t)t * H + d] =
        __float2bfloat16(dsv41_hc_collapse_one(pre + (size_t)t * hc, v, hc));
}

// out[t, j, d] = post[t, j] * x[t, d] + sum_i comb[t, i, j] * residual[t, i, d],
// rounded to BF16. `out` may be `residual`. Grid (ceil(H / 256), T), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_expand(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* out,
    unsigned int H,
    unsigned int hc) {
    const unsigned int t = blockIdx.y, d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    if (d >= H) return;
    const __nv_bfloat16* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    __nv_bfloat16* o = out + (size_t)t * hc * H;
    const float xd = __bfloat162float(x[(size_t)t * H + d]);
    float rv[DSV41_HC_MAX];
    for (unsigned int i = 0; i < hc; ++i) rv[i] = __bfloat162float(res[i * H + d]);
    for (unsigned int j = 0; j < hc; ++j)
        o[j * H + d] = __float2bfloat16(dsv41_hc_expand_one(c, rv, p[j], xd, j, hc));
}

// The same expansion of one token from the sublayer's output as its four
// parts (`dsv41_parts.cu`): x_parts [4, slots, H] FP32, combined and rounded
// here, in place into the stream. Grid (ceil(H / 256)), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_expand_parts(
    const float* __restrict__ x_parts,
    __nv_bfloat16* streams,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    unsigned int H,
    unsigned int hc,
    unsigned int slots) {
    const unsigned int d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    if (d >= H) return;
    const float xd = dsv41_parts_combine(x_parts, slots, H, d);
    float rv[DSV41_HC_MAX];
    for (unsigned int i = 0; i < hc; ++i) rv[i] = __bfloat162float(streams[i * H + d]);
    for (unsigned int j = 0; j < hc; ++j)
        streams[j * H + d] = __float2bfloat16(dsv41_hc_expand_one(comb, rv, post[j], xd, j, hc));
}

// A site of one token, first launch. Thread d:
//
//   1. With `x_parts` (the parts [4, slots, H] of the sublayer before this
//      site), that sublayer's output into the stream, under the `post` and
//      `comb` the site before left: `dsv41_hc_expand_parts`.
//   2. Its share of this site's mix of the stream: `dsv41_hc_mix_sums`.
//   3. The stream collapsed with `pre_in`, the coefficients of the site
//      before: `dsv41_hc_collapse`, into `collapsed`.
//
// Grid (H / 256), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_site_sums(
    const float* __restrict__ x_parts,
    __nv_bfloat16* streams,
    const float* __restrict__ hc_fn,
    const float* __restrict__ pre_in,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ collapsed,
    float* __restrict__ partials,
    unsigned int H,
    unsigned int hc,
    unsigned int slots) {
    const unsigned int d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    float v[DSV41_HC_MAX];
    if (d < H) {
        float rv[DSV41_HC_MAX];
        for (unsigned int j = 0; j < hc; ++j) rv[j] = __bfloat162float(streams[j * H + d]);
        if (x_parts) {
            const float xd = dsv41_parts_combine(x_parts, slots, H, d);
            for (unsigned int j = 0; j < hc; ++j) {
                const __nv_bfloat16 out =
                    __float2bfloat16(dsv41_hc_expand_one(comb, rv, post[j], xd, j, hc));
                streams[j * H + d] = out;
                v[j] = __bfloat162float(out);
            }
        } else {
            for (unsigned int j = 0; j < hc; ++j) v[j] = rv[j];
        }
        collapsed[d] = __float2bfloat16(dsv41_hc_collapse_one(pre_in, v, hc));
    }
    dsv41_hc_block_sums(v, hc_fn, partials + (size_t)blockIdx.x * DSV41_MIX_SUMS, d, H, hc);
}

#define DSV41_FINISH_THREADS 1024

// A site of one token, second launch, one block of 1,024 threads:
//
//   4. `normed` = the RMS norm of `collapsed` under `norm_weight`
//      (`rms_norm_vanilla` on a block of 1,024), and `quantized` its FP8
//      form (`dsv41_fp8_quant_dequant`).
//   5. The site's coefficients from the partials: `pre_out`, and `post` and
//      `comb` replaced (`dsv41_hc_mix_finish`).
extern "C" __global__ void __launch_bounds__(DSV41_FINISH_THREADS) dsv41_hc_site_finish(
    const float* __restrict__ partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ pre_out,
    float* __restrict__ post,
    float* __restrict__ comb_out,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ collapsed,
    __nv_bfloat16* normed,
    __nv_bfloat16* __restrict__ quantized,
    unsigned int H,
    unsigned int hc,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps) {
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    __shared__ float mix[DSV41_MIX_SUMS];
    __shared__ volatile float comb[DSV41_HC_MAX * DSV41_HC_MAX];
    __shared__ float squares[DSV41_FINISH_THREADS / 32];

    // The norm of the collapsed vector, two values a word.
    const unsigned int half = H / 2;
    const unsigned int* y32 = (const unsigned int*)collapsed;
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < half; i += DSV41_FINISH_THREADS) {
        float v0, v1;
        dsv41_unpack2(y32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, o);
    if (lane == 0) squares[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float total = squares[lane];
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
        if (lane == 0) squares[0] = total;
    }
    __syncthreads();
    const float rms = rsqrtf(squares[0] / (float)H + norm_eps);
    const unsigned int* w32 = (const unsigned int*)norm_weight;
    unsigned int* n32 = (unsigned int*)normed;
    for (unsigned int i = tid; i < half; i += DSV41_FINISH_THREADS) {
        float xv0, xv1, wv0, wv1;
        dsv41_unpack2(y32[i], xv0, xv1);
        dsv41_unpack2(w32[i], wv0, wv1);
        n32[i] = dsv41_pack2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    __syncthreads();
    if (tid < H / 32) {
        float values[32];
        for (int i = 0; i < 32; ++i) values[i] = __bfloat162float(normed[tid * 32 + i]);
        dsv41_fp8_block(values, quantized + tid * 32);
    }
    if (warp != 0) return;
    dsv41_hc_coefficients(
        partials, H / DSV41_HC_THREADS, mix, comb, lane, H, hc, sinkhorn_iters, norm_eps, hc_eps,
        hc_scale, hc_base, pre_out, post, comb_out);
}
