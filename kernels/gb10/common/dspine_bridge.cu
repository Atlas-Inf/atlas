// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DSpine drafter kernels (Tencent DSpine, arXiv 2609.36173; checkpoint and
// reference code Apache-2.0, github.com/Tencent/DSpine). Written for Atlas from
// the paper and the reference's arithmetic; no reference code is copied.
//
// A DSpine drafter is a causal DFlash block drafter whose every layer ends in a
// code bridge. Rows are blocks of `gamma` (row 0 of a block = the anchor):
//   read    rms = sqrt(max(mean(x^2), FLT_MIN)); n = bf16(x / (rms + eps))
//   [gemv]  p = n @ read_l^T                                   [rows, C]
//   codes   c = bf16(sqrt(C) * p / (|p| + eps)); anchor rows take
//           bf16(sqrt(C) * codebook[anchor])
//   [gemv]  k = c @ [gate; predecessor_gate; prev]^T           [rows, 3R]
//   message m_t = bf16(sigmoid(q_t + s_{t-1} + b) * v_{t-1}), 0 on anchor rows
//   [gemv]  d = m @ write_l^T                                  [rows, H]
//   write   x = bf16(x + rms * d) past the anchor
// The final rows' proposals come from a serial table over each row's top-16
// head candidates (`dspine_serial_table`, `dspine_serial_walk`): candidate k of
// row t scores base_k + rms_t * <sigmoid(q_t + TSV_g[s] + b) * TSV_v[s], TG[k]>
// for every candidate s of row t-1 (the anchor for the first row), and the
// chain walks from the anchor. Proposals only steer the drafts; the target's
// exact verify decides every emitted token.
//
// Every kernel works on `rows = blocks * gamma`; the anchor of block b is
// `anchors[b * anchor_stride]`. Built with --fmad=false (common/), so each
// rounding point is the one written.

#define DSP_THREADS 256
#define DSP_K 16
#define DSP_MAX_RANK 512
#define DSP_TAIL_THREADS 512

__device__ __forceinline__ float dsp_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (blockDim.x >> 5) ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float s = red[0];
    __syncthreads();
    return s;
}

// Grid (rows), block DSP_THREADS.
extern "C" __global__ void dspine_bridge_read(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ normalized,
    float* __restrict__ rms_out,
    unsigned int hidden,
    float eps
) {
    __shared__ float red[32];
    const unsigned long long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * hidden;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float f = __bfloat162float(xr[i]);
        ss += f * f;
    }
    const float mean = dsp_block_sum(ss, red) / (float)hidden;
    const float rms = sqrtf(fmaxf(mean, 1.1754943508222875e-38f));
    if (threadIdx.x == 0) rms_out[row] = rms;
    const float denom = rms + eps;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normalized[row * hidden + i] = __float2bfloat16(__bfloat162float(xr[i]) / denom);
    }
}

// Grid (rows), block DSP_THREADS.
extern "C" __global__ void dspine_bridge_codes(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ codebook,
    const unsigned int* __restrict__ anchors,
    unsigned int anchor_stride,
    __nv_bfloat16* __restrict__ codes,
    unsigned int gamma,
    unsigned int code_size,
    unsigned int vocab,
    float eps
) {
    __shared__ float red[32];
    const unsigned long long row = blockIdx.x;
    const float sqrt_c = sqrtf((float)code_size);
    __nv_bfloat16* out = codes + row * code_size;
    if (row % gamma == 0) {
        unsigned int a = anchors[(row / gamma) * anchor_stride];
        if (a >= vocab) a = 0;
        const __nv_bfloat16* cb = codebook + (unsigned long long)a * code_size;
        for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
            out[i] = __float2bfloat16(sqrt_c * __bfloat162float(cb[i]));
        }
        return;
    }
    const __nv_bfloat16* p = projected + row * code_size;
    float ss = 0.0f;
    for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
        const float f = __bfloat162float(p[i]);
        ss += f * f;
    }
    const float denom = sqrtf(dsp_block_sum(ss, red)) + eps;
    for (unsigned int i = threadIdx.x; i < code_size; i += blockDim.x) {
        out[i] = __float2bfloat16(sqrt_c * __bfloat162float(p[i]) / denom);
    }
}

// Grid (rows), block DSP_THREADS. `packed` rows are [q | s | v], R each.
extern "C" __global__ void dspine_bridge_message(
    const __nv_bfloat16* __restrict__ packed,
    const __nv_bfloat16* __restrict__ gate_bias,
    __nv_bfloat16* __restrict__ message,
    unsigned int gamma,
    unsigned int rank
) {
    const unsigned long long row = blockIdx.x;
    __nv_bfloat16* out = message + row * rank;
    if (row % gamma == 0) {
        for (unsigned int i = threadIdx.x; i < rank; i += blockDim.x) out[i] = __float2bfloat16(0.0f);
        return;
    }
    const __nv_bfloat16* own = packed + row * 3 * rank;
    const __nv_bfloat16* pred = packed + (row - 1) * 3 * rank;
    for (unsigned int i = threadIdx.x; i < rank; i += blockDim.x) {
        const float z = __bfloat162float(own[i]) + __bfloat162float(pred[rank + i])
            + __bfloat162float(gate_bias[i]);
        const float gate = 1.0f / (1.0f + expf(-z));
        out[i] = __float2bfloat16(gate * __bfloat162float(pred[2 * rank + i]));
    }
}

// Grid (rows), block DSP_THREADS. In place on `x`; `stash` (nullable)
// receives the pre-write rows (the serial table's raw state).
extern "C" __global__ void dspine_bridge_write(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ delta,
    __nv_bfloat16* __restrict__ stash,
    unsigned int gamma,
    unsigned int hidden
) {
    const unsigned long long row = blockIdx.x;
    __nv_bfloat16* xr = x + row * hidden;
    if (stash) {
        for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) stash[row * hidden + i] = xr[i];
    }
    if (row % gamma == 0) return;
    const float r = rms[row];
    const __nv_bfloat16* d = delta + row * hidden;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float t = r * __bfloat162float(d[i]);
        xr[i] = __float2bfloat16(__bfloat162float(xr[i]) + t);
    }
}

// dst = bf16(src * (col ? col[i % cols] : scalar)) over `n` elements: the
// serial tables' load-time operands. Grid-stride, any launch shape.
extern "C" __global__ void dspine_scale(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    unsigned long long n,
    unsigned int cols,
    float scalar,
    const __nv_bfloat16* __restrict__ col
) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float s = col ? __bfloat162float(col[i % cols]) : scalar;
        dst[i] = __float2bfloat16(__bfloat162float(src[i]) * s);
    }
}

__device__ __constant__ float DSP_E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float dsp_e4m3(unsigned char b) {
    const unsigned int sign = (b >> 7) & 1u, exp = (b >> 3) & 0xFu, man = b & 0x7u;
    float v;
    if (exp == 0u) v = (float)man * 0.001953125f;
    else if (exp == 15u && man == 7u) v = 0.0f;
    else v = (1.0f + (float)man * 0.125f) * exp2f((float)((int)exp - 7));
    return sign ? -v : v;
}

// Head row `id`, element `i`: BF16 when `head_bf16`, else NVFP4 (row-major
// packed E2M1, E4M3 scales per 16, per-row or per-tensor global multiplier).
struct DspHead {
    const __nv_bfloat16* bf16;
    const unsigned char* packed;
    const unsigned char* scales;
    const float* scale2_rows;
    float scale2;
    unsigned int hidden;
    __device__ __forceinline__ float at(unsigned int id, unsigned int i) const {
        if (bf16) return __bfloat162float(bf16[(unsigned long long)id * hidden + i]);
        const unsigned long long r = id;
        const unsigned char byte = packed[r * (hidden / 2) + (i >> 1)];
        const unsigned int nib = (i & 1u) ? (byte >> 4) : (byte & 0xFu);
        const float g = scale2_rows ? scale2_rows[id] : scale2;
        return DSP_E2M1[nib] * (dsp_e4m3(scales[r * (hidden / 16) + i / 16]) * g);
    }
};

// The 16 candidate ids of `lists` row `row` (`dflash2_selector_topk`
// layout: f32 values [rows, 16], then u32 ids [rows, 16]) sorted ascending —
// the reference orders a row's candidates by id, so a score tie goes to the
// lower id. Thread-0 insertion sort; caller syncs.
__device__ __forceinline__ void dsp_sorted_ids(
    const float* lists, unsigned int rows, unsigned int row, unsigned int* out
) {
    if (threadIdx.x != 0) return;
    const unsigned int* ids = reinterpret_cast<const unsigned int*>(lists + rows * DSP_K) + row * DSP_K;
    for (int k = 0; k < DSP_K; k++) {
        const unsigned int v = ids[k];
        int j = k;
        while (j > 0 && out[j - 1] > v) {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = v;
    }
}

// Grid (rows), block DSP_TAIL_THREADS (16 warps). Row t > 0 of each block
// writes its 16 ascending candidates to `cand` and, for every sender slot j,
// the best candidate slot to `idx` ([rows, 16] u32 each). Rows 1..=ban of
// block b (`ban_depth[b * ban_stride]`) never pick end0..end3 (a null
// ban_depth bans nothing).
extern "C" __global__ void dspine_serial_table(
    const float* __restrict__ lists,
    const __nv_bfloat16* __restrict__ raw,
    const float* __restrict__ rms,
    const __nv_bfloat16* __restrict__ packed,
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ gate_bias,
    const __nv_bfloat16* __restrict__ tg,
    const __nv_bfloat16* __restrict__ tsv,
    const unsigned int* __restrict__ anchors,
    unsigned int anchor_stride,
    const unsigned int* __restrict__ ban_depth,
    unsigned int ban_stride,
    const __nv_bfloat16* __restrict__ head_bf16,
    const unsigned char* __restrict__ head_packed,
    const unsigned char* __restrict__ head_scales,
    const float* __restrict__ head_scale2_rows,
    float head_scale2,
    unsigned int hidden,
    unsigned int* __restrict__ cand,
    unsigned int* __restrict__ idx,
    unsigned int rows,
    unsigned int gamma,
    unsigned int rank,
    unsigned int vocab,
    unsigned int end0,
    unsigned int end1,
    unsigned int end2,
    unsigned int end3
) {
    __shared__ unsigned int s_cand[DSP_K];
    __shared__ unsigned int s_send[DSP_K];
    __shared__ float s_base[DSP_K];
    __shared__ float s_score[DSP_K][DSP_K];
    __shared__ __nv_bfloat16 s_tg[DSP_K * DSP_MAX_RANK];

    const unsigned int row = blockIdx.x;
    const unsigned int t = row % gamma;
    if (t == 0 || rank > DSP_MAX_RANK || blockDim.x != DSP_TAIL_THREADS) return;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const DspHead head{head_bf16, head_packed, head_scales, head_scale2_rows, head_scale2, hidden};

    dsp_sorted_ids(lists, rows, row, s_cand);
    if (t > 1) {
        dsp_sorted_ids(lists, rows, row - 1, s_send);
    } else if (threadIdx.x < DSP_K) {
        s_send[threadIdx.x] = anchors[(row / gamma) * anchor_stride];
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < DSP_K * rank; i += blockDim.x) {
        unsigned int id = s_cand[i / rank];
        s_tg[i] = tg[(unsigned long long)(id < vocab ? id : 0) * rank + i % rank];
    }

    // base_k = <raw * norm_w, head[cand_k]>: warp k.
    {
        unsigned int id = s_cand[warp];
        if (id >= vocab) id = 0;
        const __nv_bfloat16* xr = raw + (unsigned long long)row * hidden;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hidden; i += 32) {
            const float a = __bfloat162float(xr[i]) * __bfloat162float(norm_w[i]);
            acc += a * head.at(id, i);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) s_base[warp] = acc;
    }
    __syncthreads();

    // Sender slot j = warp: m = sigmoid(q + TSV_g[s] + b) * TSV_v[s] on the
    // lanes' rank slice, then s_jk = <m, TG[cand_k]>.
    {
        unsigned int s = s_send[warp];
        if (s >= vocab) s = 0;
        const __nv_bfloat16* q = packed + (unsigned long long)row * 3 * rank;
        const __nv_bfloat16* ts = tsv + (unsigned long long)s * 2 * rank;
        float m[DSP_MAX_RANK / 32];
#pragma unroll
        for (int u = 0; u < DSP_MAX_RANK / 32; u++) {
            const unsigned int r = lane + 32 * u;
            m[u] = 0.0f;
            if (r < rank) {
                const float z = __bfloat162float(q[r]) + __bfloat162float(ts[r]) + __bfloat162float(gate_bias[r]);
                m[u] = (1.0f / (1.0f + expf(-z))) * __bfloat162float(ts[rank + r]);
            }
        }
        const float rr = rms[row];
        const unsigned int banned_to = ban_depth ? ban_depth[(row / gamma) * ban_stride] : 0u;
        for (unsigned int k = 0; k < DSP_K; k++) {
            float dot = 0.0f;
#pragma unroll
            for (int u = 0; u < DSP_MAX_RANK / 32; u++) {
                const unsigned int r = lane + 32 * u;
                if (r < rank) dot += m[u] * __bfloat162float(s_tg[k * rank + r]);
            }
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) {
                const unsigned int c = s_cand[k];
                const bool banned = t <= banned_to && (c == end0 || c == end1 || c == end2 || c == end3);
                s_score[warp][k] = c < vocab && !banned ? s_base[k] + rr * dot : -INFINITY;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < DSP_K) {
        const unsigned int j = threadIdx.x;
        unsigned int best = 0;
        float bv = -INFINITY;
        for (unsigned int k = 0; k < DSP_K; k++) {
            const float v = s_score[j][k];
            if (v > bv) {
                bv = v;
                best = k;
            }
        }
        cand[row * DSP_K + j] = s_cand[j] < vocab ? s_cand[j] : 0u;
        idx[row * DSP_K + j] = best;
    }
}

// Grid (blocks), block 32. Block b's drafts: row 0 takes its top-1 (unused by
// the scheduler), then j = idx[1][0] and j = idx[t][j] down the rows.
extern "C" __global__ void dspine_serial_walk(
    const float* __restrict__ lists,
    const unsigned int* __restrict__ cand,
    const unsigned int* __restrict__ idx,
    unsigned int* __restrict__ out_tokens,
    unsigned int rows,
    unsigned int gamma
) {
    if (threadIdx.x != 0) return;
    const unsigned int r0 = blockIdx.x * gamma;
    const unsigned int* ids = reinterpret_cast<const unsigned int*>(lists + rows * DSP_K);
    out_tokens[r0] = ids[r0 * DSP_K];
    unsigned int j = 0;
    for (unsigned int t = 1; t < gamma; t++) {
        j = idx[(r0 + t) * DSP_K + j];
        out_tokens[r0 + t] = cand[(r0 + t) * DSP_K + j];
    }
}
