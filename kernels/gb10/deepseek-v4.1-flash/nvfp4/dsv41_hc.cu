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
    atlas_pdl_enter();
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
    atlas_pdl_enter();
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
    atlas_pdl_enter();
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
    atlas_pdl_enter();
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

// Value d of token t of a sublayer's output as the ranks gathered it: four
// parts, each a quarter of the H values of every token, with room for
// `part_rows` tokens.
__device__ __forceinline__ float dsv41_hc_gathered(
    const __nv_bfloat16* x, unsigned int t, unsigned int d, unsigned int H,
    unsigned int part_rows) {
    const unsigned int n = H / DSV41_PARTS;
    return __bfloat162float(x[((size_t)(d / n) * part_rows + t) * n + d % n]);
}

// The same expansion from the sublayer's output in its four parts
// (`dsv41_hc_gathered`), in place into the stream. Grid (ceil(H / 256), T),
// block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_expand_parts(
    const __nv_bfloat16* __restrict__ x_parts,
    __nv_bfloat16* streams,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    unsigned int H,
    unsigned int hc,
    unsigned int part_rows) {
    atlas_pdl_enter();
    const unsigned int t = blockIdx.y, d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    if (d >= H) return;
    const float xd = dsv41_hc_gathered(x_parts, t, d, H, part_rows);
    streams += (size_t)t * hc * H;
    post += (size_t)t * hc;
    comb += (size_t)t * hc * hc;
    float rv[DSV41_HC_MAX];
    for (unsigned int i = 0; i < hc; ++i) rv[i] = __bfloat162float(streams[i * H + d]);
    for (unsigned int j = 0; j < hc; ++j)
        streams[j * H + d] = __float2bfloat16(dsv41_hc_expand_one(comb, rv, post[j], xd, j, hc));
}

// A site, first launch, a token a row of the grid. Thread d:
//
//   1. With `x_parts` (the parts of the sublayer before this site, as
//      `dsv41_hc_expand_parts` takes them), that sublayer's output into the
//      stream, under the `post` and `comb` the site before left.
//   2. Its share of this site's mix of the stream: `dsv41_hc_mix_sums`.
//   3. The stream collapsed with `pre_in`, the coefficients of the site
//      before: `dsv41_hc_collapse`, into `collapsed`.
//
// Grid (H / 256, T), block 256.
extern "C" __global__ void __launch_bounds__(DSV41_HC_THREADS) dsv41_hc_site_sums(
    const __nv_bfloat16* __restrict__ x_parts,
    __nv_bfloat16* streams,
    const unsigned char* __restrict__ hc_fn_bytes,
    const float* __restrict__ pre_in,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ collapsed,
    float* __restrict__ partials,
    unsigned int H,
    unsigned int hc,
    unsigned int part_rows) {
    atlas_pdl_enter_touch({hc_fn_bytes, 4u * hc * H, 4ull * hc * H}, {hc_fn_bytes, 0u, 0ull},
                          (2u + hc) * hc, blockIdx.y * gridDim.x + blockIdx.x,
                          dsv41_touchers(gridDim.x));
    const float* hc_fn = (const float*)hc_fn_bytes;
    const unsigned int t = blockIdx.y, d = blockIdx.x * DSV41_HC_THREADS + threadIdx.x;
    streams += (size_t)t * hc * H;
    pre_in += (size_t)t * hc;
    post += (size_t)t * hc;
    comb += (size_t)t * hc * hc;
    collapsed += (size_t)t * H;
    partials += (size_t)t * gridDim.x * DSV41_MIX_SUMS;
    float v[DSV41_HC_MAX];
    if (d < H) {
        float rv[DSV41_HC_MAX];
        for (unsigned int j = 0; j < hc; ++j) rv[j] = __bfloat162float(streams[j * H + d]);
        if (x_parts) {
            const float xd = dsv41_hc_gathered(x_parts, t, d, H, part_rows);
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

// A site, second launch, two blocks of 1,024 threads a token:
//
//   4. Block 0: `normed` = the RMS norm of `collapsed` under `norm_weight`
//      (`rms_norm_vanilla` on a block of 1,024), and `quantized` its FP8
//      form (`dsv41_fp8_quant_dequant`). A thread rounds the two values it
//      has just normed: the 32 values of a scale block are the pairs of 16
//      neighbouring lanes, which find the block's largest by shuffles.
//   5. Block 1, on one warp: the site's coefficients from the partials:
//      `pre_out`, and `post` and `comb` replaced (`dsv41_hc_mix_finish`).
//      Twenty Sinkhorn rounds are a chain nothing shortens, so they run
//      beside the norm, not after it.
//
// Grid (2, T).
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
    atlas_pdl_enter();
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int t = blockIdx.y, blocks = H / DSV41_HC_THREADS;
    __shared__ float mix[DSV41_MIX_SUMS];
    __shared__ volatile float comb[DSV41_HC_MAX * DSV41_HC_MAX];
    __shared__ float squares[DSV41_FINISH_THREADS / 32];
    if (blockIdx.x == 1) {
        if (warp == 0)
            dsv41_hc_coefficients(
                partials + (size_t)t * blocks * DSV41_MIX_SUMS, blocks, mix, comb, lane, H, hc,
                sinkhorn_iters, norm_eps, hc_eps, hc_scale, hc_base, pre_out + (size_t)t * hc,
                post + (size_t)t * hc, comb_out + (size_t)t * hc * hc);
        return;
    }
    collapsed += (size_t)t * H;
    normed += (size_t)t * H;
    quantized += (size_t)t * H;

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
    unsigned int* q32 = (unsigned int*)quantized;
    // Every thread of a warp takes as many turns: 1,024 divides the pairs
    // by whole warps.
    for (unsigned int i = tid; i < half; i += DSV41_FINISH_THREADS) {
        float xv0, xv1, wv0, wv1;
        dsv41_unpack2(y32[i], xv0, xv1);
        dsv41_unpack2(w32[i], wv0, wv1);
        const unsigned int pair = dsv41_pack2(xv0 * rms * wv0, xv1 * rms * wv1);
        n32[i] = pair;
        float v0, v1;
        dsv41_unpack2(pair, v0, v1);
        float amax = fmaxf(fabsf(v0), fabsf(v1));
        #pragma unroll
        for (unsigned int o = 8; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float scale = dsv41_fp8_scale(amax);
        q32[i] = dsv41_pack2(dsv41_fp8_value(v0, scale), dsv41_fp8_value(v1, scale));
    }
}

// A site of a batch, before a GEMM.
//
// The mix is 24 dot products over a token's 20,480 stream values against
// FP32 weights, and for a batch it is most of a site. A tensor core
// multiplies BF16, and an FP32 weight is exactly the sum of three BF16
// values (its eight leading bits, the next eight and the last), so the
// same products come out of a BF16 GEMM against the three: sums [T, 3 * R]
// FP32, R the mix's rows, a row's three shares R columns apart, which
// `dsv41_hc_mix_rows` adds. The GEMM reads the stream this launch leaves, a
// block of 1,024 threads a token:
//
//   with `x_parts` (the parts of the sublayer before this site, as
//   `dsv41_hc_expand_parts` reads them, under `post` and `comb`, that
//   site's), the stream expanded in place, in the pass that reads it: a
//   stream is 40 KB a token, and a launch of its own for the expansion read
//   and wrote every byte of it once more;
//   the stream collapsed with `pre_in`, the coefficients of the site before
//   (`dsv41_hc_collapse`), into `collapsed`; its RMS norm under
//   `norm_weight` and that in FP8, as `dsv41_hc_site_finish` leaves them;
//   and the stream's sum of squares, partials [T, DSV41_MIX_SUMS] at
//   DSV41_MIX_ROWS.
//
// The expansion, the collapse, the norm and the rounding are the
// row-at-a-time site's, value for value. The mix's sums have another order.
extern "C" __global__ void __launch_bounds__(DSV41_FINISH_THREADS) dsv41_hc_site_rows(
    __nv_bfloat16* __restrict__ streams,
    const __nv_bfloat16* __restrict__ x_parts,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    unsigned int part_rows,
    const float* __restrict__ pre_in,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ collapsed,
    __nv_bfloat16* __restrict__ normed,
    __nv_bfloat16* __restrict__ quantized,
    float* __restrict__ partials,
    unsigned int H,
    unsigned int hc,
    float norm_eps) {
    const unsigned int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const unsigned int t = blockIdx.x, half = H / 2;
    __shared__ float squares[2][DSV41_FINISH_THREADS / 32];
    unsigned int* s32 = (unsigned int*)(streams + (size_t)t * hc * H);
    post += (size_t)t * hc;
    comb += (size_t)t * hc * hc;
    const float* pre = pre_in + (size_t)t * hc;
    unsigned int* y32 = (unsigned int*)(collapsed + (size_t)t * H);
    // Of the collapsed vector, and of the stream.
    float sum_sq = 0.0f, stream_sq = 0.0f;
    for (unsigned int i = tid; i < half; i += DSV41_FINISH_THREADS) {
        float v0[DSV41_HC_MAX], v1[DSV41_HC_MAX];
        for (unsigned int j = 0; j < hc; ++j) dsv41_unpack2(s32[(size_t)j * half + i], v0[j], v1[j]);
        if (x_parts) {
            const float x0 = dsv41_hc_gathered(x_parts, t, 2u * i, H, part_rows);
            const float x1 = dsv41_hc_gathered(x_parts, t, 2u * i + 1u, H, part_rows);
            unsigned int pair[DSV41_HC_MAX];
            for (unsigned int j = 0; j < hc; ++j)
                pair[j] = dsv41_pack2(dsv41_hc_expand_one(comb, v0, post[j], x0, j, hc),
                                      dsv41_hc_expand_one(comb, v1, post[j], x1, j, hc));
            for (unsigned int j = 0; j < hc; ++j) {
                s32[(size_t)j * half + i] = pair[j];
                dsv41_unpack2(pair[j], v0[j], v1[j]);
            }
        }
        float a0 = 0.0f, a1 = 0.0f;
        for (unsigned int j = 0; j < hc; ++j) {
            a0 = j == 0 ? pre[0] * v0[0] : a0 + pre[j] * v0[j];
            a1 = j == 0 ? pre[0] * v1[0] : a1 + pre[j] * v1[j];
            stream_sq += v0[j] * v0[j] + v1[j] * v1[j];
        }
        const unsigned int pair = dsv41_pack2(a0, a1);
        y32[i] = pair;
        float c0, c1;
        dsv41_unpack2(pair, c0, c1);
        sum_sq += c0 * c0 + c1 * c1;
    }
    #pragma unroll
    for (unsigned int o = 16; o > 0; o >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, o);
        stream_sq += __shfl_xor_sync(0xffffffffu, stream_sq, o);
    }
    if (lane == 0) {
        squares[0][warp] = sum_sq;
        squares[1][warp] = stream_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float total = squares[0][lane], of_stream = squares[1][lane];
        #pragma unroll
        for (unsigned int o = 16; o > 0; o >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, o);
            of_stream += __shfl_xor_sync(0xffffffffu, of_stream, o);
        }
        if (lane == 0) {
            squares[0][0] = total;
            partials[(size_t)t * DSV41_MIX_SUMS + DSV41_MIX_ROWS] = of_stream;
        }
    }
    __syncthreads();
    const float rms = rsqrtf(squares[0][0] / (float)H + norm_eps);
    const unsigned int* w32 = (const unsigned int*)norm_weight;
    unsigned int* n32 = (unsigned int*)(normed + (size_t)t * H);
    unsigned int* q32 = (unsigned int*)(quantized + (size_t)t * H);
    for (unsigned int i = tid; i < half; i += DSV41_FINISH_THREADS) {
        float xv0, xv1, wv0, wv1;
        dsv41_unpack2(y32[i], xv0, xv1);
        dsv41_unpack2(w32[i], wv0, wv1);
        const unsigned int pair = dsv41_pack2(xv0 * rms * wv0, xv1 * rms * wv1);
        n32[i] = pair;
        float v0, v1;
        dsv41_unpack2(pair, v0, v1);
        float amax = fmaxf(fabsf(v0), fabsf(v1));
        #pragma unroll
        for (unsigned int o = 8; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float scale = dsv41_fp8_scale(amax);
        q32[i] = dsv41_pack2(dsv41_fp8_value(v0, scale), dsv41_fp8_value(v1, scale));
    }
}

// `dsv41_hc_mix_finish` from one set of sums a token: partials [T,
// DSV41_MIX_SUMS], of which `dsv41_hc_site_rows` left the stream's sum of
// squares and this fills the mix's rows, the three shares of each added from
// the GEMM's sums [T, 3 * R]. Grid (T), block 32.
extern "C" __global__ void __launch_bounds__(32) dsv41_hc_mix_rows(
    const float* __restrict__ sums,
    float* __restrict__ partials,
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
    const unsigned int t = blockIdx.x, rows = (2 + hc) * hc;
    __shared__ float mix[DSV41_MIX_SUMS];
    __shared__ volatile float comb[DSV41_HC_MAX * DSV41_HC_MAX];
    if (threadIdx.x < rows) {
        const float* mine = sums + (size_t)t * 3u * rows + threadIdx.x;
        partials[(size_t)t * DSV41_MIX_SUMS + threadIdx.x] =
            (mine[0] + mine[rows]) + mine[2u * rows];
    }
    __syncthreads();
    dsv41_hc_coefficients(
        partials + (size_t)t * DSV41_MIX_SUMS, 1, mix, comb, threadIdx.x, H, hc, sinkhorn_iters,
        norm_eps, hc_eps, hc_scale, hc_base, pre_out + (size_t)t * hc, post_out + (size_t)t * hc,
        comb_out + (size_t)t * hc * hc);
}

// Where a pass starts: every copy of each token's stream its embedding, and
// each token's coefficients the identity mix, which collapses to the first
// copy. tokens [rows, hidden] BF16, streams [rows, copies, hidden] BF16, pre
// [rows, copies] FP32. hidden % 8 == 0. One thread 8 values of a token.
extern "C" __global__ void dsv41_hc_start(
    const uint4* __restrict__ tokens,
    uint4* __restrict__ streams,
    float* __restrict__ pre,
    unsigned int rows,
    unsigned int hidden,
    unsigned int copies) {
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per = hidden / 8u;
    if (i >= (size_t)rows * per) return;
    const size_t row = i / per, at = i % per;
    const uint4 v = tokens[i];
    for (unsigned int copy = 0; copy < copies; ++copy)
        streams[(row * copies + copy) * per + at] = v;
    if (at < copies) pre[row * copies + at] = at == 0 ? 1.0f : 0.0f;
}
