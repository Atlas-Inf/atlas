// SPDX-License-Identifier: AGPL-3.0-only
//
// Ling MLA over the compressed latent cache.
//
// Each cached token is one 576-wide BF16 row: the normalized 512-channel
// latent (kv_a_layernorm output) followed by the rotated 64-channel RoPE key
// shared by all 32 heads. The V side of the paged pool aliases the K side.
//
// Decode, batched decode and the MTP verify rows attend in the absorbed form:
//   q_abs[h]   = [W_kb[h]^T q_nope[h] | q_rope[h]]            (576)
//   score[h,t] = q_abs[h] . cache[t] * scale
//   lat[h]     = sum_t softmax(score)[h,t] cache[t][0:512]
//   out[h]     = W_vb[h] lat[h]                                  (128)
// where kv_b_proj rows h*256 + [0,128) are W_kb[h] and h*256 + [128,256) are
// W_vb[h]. Every row's arithmetic depends only on that row and its sequence
// length, never on how many rows share a launch, so a two-row verify
// reproduces serial decode byte for byte.
//
// Prompts rebuild full per-head keys and values from the latent in bounded
// tiles (ling_latent_gather + kv_b_proj + ling_latent_expand), attend each
// tile with the paged prefill kernel, and merge tiles by their softmax state.

#include <cuda_bf16.h>
#include <math_constants.h>

#define LING_HEADS 32u
#define LING_LATENT 512u
#define LING_ROPE 64u
#define LING_ROW 576u        // LING_LATENT + LING_ROPE
#define LING_QK 192u         // per-head query width in q_full: nope 128 | rope 64
#define LING_NOPE 128u
#define LING_V 128u
#define LING_KVB 256u        // kv_b_proj rows per head: nope 128 | v 128

// ---- Cache write ------------------------------------------------------------

// One row per block: [kv_norm (512) | k_rope (64)] at the row's slot.
extern "C" __global__ void ling_latent_cache_write(
    const __nv_bfloat16* __restrict__ kv_norm,   // [rows][512]
    const __nv_bfloat16* __restrict__ k_rope,    // [rows][64]
    __nv_bfloat16* __restrict__ pool,            // [slots][576]
    const long long* __restrict__ slots) {
    const unsigned row = blockIdx.x;
    const long long slot = slots[row];
    if (slot < 0) return;
    __nv_bfloat16* dst = pool + static_cast<unsigned long long>(slot) * LING_ROW;
    for (unsigned i = threadIdx.x; i < LING_ROW; i += blockDim.x)
        dst[i] = i < LING_LATENT ? kv_norm[row * LING_LATENT + i]
                                 : k_rope[row * LING_ROPE + i - LING_LATENT];
}

// ---- Query absorption -------------------------------------------------------

#define ABSORB_ROWS 8u

// grid (heads, ceil(rows / ABSORB_ROWS)), block 512: thread c owns latent
// channel c of every row in its group.
extern "C" __global__ void __launch_bounds__(512) ling_q_absorb(
    const __nv_bfloat16* __restrict__ q_full,    // [rows][32][192], rope rotated
    const __nv_bfloat16* __restrict__ kv_b,      // [32*256][512]
    __nv_bfloat16* __restrict__ q_abs,           // [rows][32][576]
    const unsigned rows) {
    const unsigned h = blockIdx.x, r0 = blockIdx.y * ABSORB_ROWS, c = threadIdx.x;
    const unsigned n = min(ABSORB_ROWS, rows - r0);
    __shared__ float q[ABSORB_ROWS][LING_NOPE];
    for (unsigned i = threadIdx.x; i < ABSORB_ROWS * LING_NOPE; i += blockDim.x) {
        const unsigned r = i / LING_NOPE, d = i % LING_NOPE;
        q[r][d] = r < n ? __bfloat162float(q_full[((r0 + r) * LING_HEADS + h) * LING_QK + d]) : 0.f;
    }
    __syncthreads();
    float acc[ABSORB_ROWS];
#pragma unroll
    for (unsigned r = 0; r < ABSORB_ROWS; ++r) acc[r] = 0.f;
    const __nv_bfloat16* w = kv_b + (h * LING_KVB) * LING_LATENT + c;
    for (unsigned d = 0; d < LING_NOPE; ++d) {
        const float wv = __bfloat162float(w[d * LING_LATENT]);
#pragma unroll
        for (unsigned r = 0; r < ABSORB_ROWS; ++r) acc[r] += q[r][d] * wv;
    }
    for (unsigned r = 0; r < n; ++r) {
        __nv_bfloat16* out = q_abs + ((r0 + r) * LING_HEADS + h) * LING_ROW;
        out[c] = __float2bfloat16(acc[r]);
        if (c < LING_ROPE)
            out[LING_LATENT + c] = q_full[((r0 + r) * LING_HEADS + h) * LING_QK + LING_NOPE + c];
    }
}

// ---- Absorbed paged decode ----------------------------------------------------

#define DEC_SPLITS 64u
#define DEC_MIN_CHUNK 256u
#define DEC_TILE 32u
#define DEC_WORDS (LING_ROW / 2u)          // 288 bf16x2 words per cached row
#define DEC_SLICES 8u                      // threads per head
#define DEC_QWORDS (DEC_WORDS / DEC_SLICES)  // 36 words of the row per thread
#define DEC_VWORDS (LING_LATENT / 2u / DEC_SLICES)  // 32 latent words per thread

// Keys per split, from the sequence length alone.
__device__ __forceinline__ unsigned ling_decode_chunk(unsigned len) {
    unsigned chunk = (len + DEC_SPLITS - 1) / DEC_SPLITS;
    chunk = chunk < DEC_MIN_CHUNK ? DEC_MIN_CHUNK : chunk;
    return (chunk + DEC_TILE - 1) / DEC_TILE * DEC_TILE;
}

__device__ __forceinline__ float2 bf2(unsigned w) {
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

// grid (DEC_SPLITS, rows), block 256 = 32 heads x 8 slices. A slice owns the
// row's 32-bit words j*8 + slice, so the eight slices of a head read eight
// different shared-memory banks.
extern "C" __global__ void __launch_bounds__(256) ling_latent_decode(
    const __nv_bfloat16* __restrict__ q_abs,     // [rows][32][576]
    const __nv_bfloat16* __restrict__ pool,      // [blocks][block_size][576]
    const int* __restrict__ block_table,         // [rows][bt_stride] or shared
    const int* __restrict__ seq_lens,            // [rows]
    const unsigned bt_stride,                    // 0: every row shares row 0's table
    const unsigned block_size,
    const float scale,
    float* __restrict__ part_o,                  // [rows][SPLITS][32][512]
    float* __restrict__ part_ml) {               // [rows][SPLITS][32][2]
    const unsigned split = blockIdx.x, row = blockIdx.y;
    const unsigned len = static_cast<unsigned>(seq_lens[row]);
    const unsigned chunk = ling_decode_chunk(len);
    const unsigned begin = split * chunk;
    if (begin >= len) return;
    const unsigned end = min(begin + chunk, len);
    const unsigned head = threadIdx.x / DEC_SLICES, slice = threadIdx.x % DEC_SLICES;
    const int* table = block_table + row * bt_stride;

    __shared__ unsigned keys[DEC_TILE][DEC_WORDS];
    __shared__ float scores[LING_HEADS][DEC_TILE];

    unsigned q[DEC_QWORDS];
    const unsigned* qrow = reinterpret_cast<const unsigned*>(q_abs + (row * LING_HEADS + head) * LING_ROW);
#pragma unroll
    for (unsigned j = 0; j < DEC_QWORDS; ++j) q[j] = qrow[j * DEC_SLICES + slice];

    float acc[2 * DEC_VWORDS];
#pragma unroll
    for (unsigned i = 0; i < 2 * DEC_VWORDS; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, l = 0.f;

    for (unsigned t0 = begin; t0 < end; t0 += DEC_TILE) {
        const unsigned n = min(DEC_TILE, end - t0);
        for (unsigned i = threadIdx.x; i < n * DEC_WORDS; i += blockDim.x) {
            const unsigned k = i / DEC_WORDS, w = i % DEC_WORDS, t = t0 + k;
            const unsigned long long slot =
                static_cast<unsigned long long>(table[t / block_size]) * block_size + t % block_size;
            keys[k][w] = reinterpret_cast<const unsigned*>(pool + slot * LING_ROW)[w];
        }
        __syncthreads();
        float tile_max = -CUDART_INF_F;
        for (unsigned k = 0; k < n; ++k) {
            float s = 0.f;
#pragma unroll
            for (unsigned j = 0; j < DEC_QWORDS; ++j) {
                const float2 a = bf2(q[j]), b = bf2(keys[k][j * DEC_SLICES + slice]);
                s += a.x * b.x;
                s += a.y * b.y;
            }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            s *= scale;
            if (slice == 0) scores[head][k] = s;
            tile_max = fmaxf(tile_max, s);
        }
        __syncwarp();
        const float next = fmaxf(m, tile_max);
        const float correction = expf(m - next);
        l *= correction;
#pragma unroll
        for (unsigned i = 0; i < 2 * DEC_VWORDS; ++i) acc[i] *= correction;
        m = next;
        for (unsigned k = 0; k < n; ++k) {
            const float p = expf(scores[head][k] - m);
            l += p;
#pragma unroll
            for (unsigned j = 0; j < DEC_VWORDS; ++j) {
                const float2 v = bf2(keys[k][j * DEC_SLICES + slice]);
                acc[2 * j] += p * v.x;
                acc[2 * j + 1] += p * v.y;
            }
        }
        __syncthreads();
    }

    const unsigned base = (row * DEC_SPLITS + split) * LING_HEADS + head;
    float2* out = reinterpret_cast<float2*>(part_o + base * LING_LATENT);
#pragma unroll
    for (unsigned j = 0; j < DEC_VWORDS; ++j)
        out[j * DEC_SLICES + slice] = make_float2(acc[2 * j], acc[2 * j + 1]);
    if (slice == 0) {
        part_ml[base * 2] = m;
        part_ml[base * 2 + 1] = l;
    }
}

// grid (heads, rows), block 128: thread i owns latent channels 4i..4i+3.
extern "C" __global__ void __launch_bounds__(128) ling_latent_decode_merge(
    const float* __restrict__ part_o,
    const float* __restrict__ part_ml,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ latent) {        // [rows][32][512]
    const unsigned head = blockIdx.x, row = blockIdx.y;
    const unsigned len = static_cast<unsigned>(seq_lens[row]);
    const unsigned chunk = ling_decode_chunk(len);
    const unsigned splits = (len + chunk - 1) / chunk;
    float top = -CUDART_INF_F;
    for (unsigned s = 0; s < splits; ++s)
        top = fmaxf(top, part_ml[((row * DEC_SPLITS + s) * LING_HEADS + head) * 2]);
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    float total = 0.f;
    for (unsigned s = 0; s < splits; ++s) {
        const unsigned base = (row * DEC_SPLITS + s) * LING_HEADS + head;
        const float w = expf(part_ml[base * 2] - top);
        total += part_ml[base * 2 + 1] * w;
        const float4 v = reinterpret_cast<const float4*>(part_o + base * LING_LATENT)[threadIdx.x];
        o.x += v.x * w;
        o.y += v.y * w;
        o.z += v.z * w;
        o.w += v.w * w;
    }
    const float inv = total > 0.f ? 1.f / total : 0.f;
    __nv_bfloat16* out = latent + (row * LING_HEADS + head) * LING_LATENT + 4 * threadIdx.x;
    out[0] = __float2bfloat16(o.x * inv);
    out[1] = __float2bfloat16(o.y * inv);
    out[2] = __float2bfloat16(o.z * inv);
    out[3] = __float2bfloat16(o.w * inv);
}

// ---- Value up-projection ------------------------------------------------------

#define VUP_ROWS 8u

// grid (heads, ceil(rows / VUP_ROWS)), block 256: warp w produces value
// channels w, w + 8, ...; lane l reduces latent channels 16l..16l+15.
extern "C" __global__ void __launch_bounds__(256) ling_v_up(
    const __nv_bfloat16* __restrict__ latent,    // [rows][32][512]
    const __nv_bfloat16* __restrict__ kv_b,      // [32*256][512]
    __nv_bfloat16* __restrict__ out,             // [rows][32*128]
    const unsigned rows) {
    const unsigned h = blockIdx.x, r0 = blockIdx.y * VUP_ROWS;
    const unsigned n = min(VUP_ROWS, rows - r0);
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float lat[VUP_ROWS][LING_LATENT];
    for (unsigned i = threadIdx.x; i < VUP_ROWS * LING_LATENT; i += blockDim.x) {
        const unsigned r = i / LING_LATENT, c = i % LING_LATENT;
        lat[r][c] = r < n ? __bfloat162float(latent[((r0 + r) * LING_HEADS + h) * LING_LATENT + c]) : 0.f;
    }
    __syncthreads();
    for (unsigned v = warp; v < LING_V; v += 8) {
        const uint4* w = reinterpret_cast<const uint4*>(
            kv_b + (h * LING_KVB + LING_NOPE + v) * LING_LATENT + 16 * lane);
        const uint4 w0 = w[0], w1 = w[1];
        const unsigned words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float acc[VUP_ROWS];
#pragma unroll
        for (unsigned r = 0; r < VUP_ROWS; ++r) acc[r] = 0.f;
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            const float2 wv = bf2(words[j]);
#pragma unroll
            for (unsigned r = 0; r < VUP_ROWS; ++r) {
                acc[r] += wv.x * lat[r][16 * lane + 2 * j];
                acc[r] += wv.y * lat[r][16 * lane + 2 * j + 1];
            }
        }
#pragma unroll
        for (unsigned r = 0; r < VUP_ROWS; ++r) {
            float s = acc[r];
            s += __shfl_xor_sync(0xffffffffu, s, 16);
            s += __shfl_xor_sync(0xffffffffu, s, 8);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            if (lane == 0 && r < n)
                out[(r0 + r) * LING_HEADS * LING_V + h * LING_V + v] = __float2bfloat16(s);
        }
    }
}

// ---- Prompt expansion -----------------------------------------------------------

// Rows [first, first + n) of one sequence's cache into a contiguous latent
// [n][512] and RoPE key [n][64]. grid n, block 64.
extern "C" __global__ void ling_latent_gather(
    const __nv_bfloat16* __restrict__ pool,
    const int* __restrict__ block_table,
    const unsigned block_size,
    const unsigned first,
    __nv_bfloat16* __restrict__ latent,
    __nv_bfloat16* __restrict__ rope) {
    const unsigned k = blockIdx.x, t = first + k;
    const unsigned long long slot =
        static_cast<unsigned long long>(block_table[t / block_size]) * block_size + t % block_size;
    const uint4* src = reinterpret_cast<const uint4*>(pool + slot * LING_ROW);
    // 72 sixteen-byte words per row: 64 latent, 8 RoPE.
    for (unsigned i = threadIdx.x; i < LING_ROW / 8; i += blockDim.x) {
        const uint4 v = src[i];
        if (i < LING_LATENT / 8)
            reinterpret_cast<uint4*>(latent + k * LING_LATENT)[i] = v;
        else
            reinterpret_cast<uint4*>(rope + k * LING_ROPE)[i - LING_LATENT / 8] = v;
    }
}

// Expanded rows in the layout ling_mla_prefill reads (ling_cache_layout.cuh):
// K = 32 x 128 NoPE channels then the shared 64-channel RoPE tail, V = 32 x
// 128 channels then a zero tail (written once at allocation, never here).
// grid n, block 256; row k lands at scratch row `first + k`.
extern "C" __global__ void ling_latent_expand(
    const __nv_bfloat16* __restrict__ kv_b,      // [n][32*256]
    const __nv_bfloat16* __restrict__ rope,      // [n][64]
    __nv_bfloat16* __restrict__ k_rows,
    __nv_bfloat16* __restrict__ v_rows,
    const unsigned first) {
    const unsigned k = blockIdx.x;
    constexpr unsigned wide = LING_HEADS * LING_NOPE, stride = wide + LING_ROPE;
    const unsigned long long dst = static_cast<unsigned long long>(first + k) * stride;
    const __nv_bfloat16* src = kv_b + static_cast<unsigned long long>(k) * LING_HEADS * LING_KVB;
    for (unsigned i = threadIdx.x; i < wide; i += blockDim.x) {
        const unsigned h = i / LING_NOPE, d = i % LING_NOPE;
        k_rows[dst + i] = src[h * LING_KVB + d];
        v_rows[dst + i] = src[h * LING_KVB + LING_NOPE + d];
    }
    if (threadIdx.x < LING_ROPE) k_rows[dst + wide + threadIdx.x] = rope[k * LING_ROPE + threadIdx.x];
}

// Fold one key tile's normalized output and (max, sum) into the running
// result. The last tile writes the BF16 [rows][32][192] attention output.
// grid (heads, rows), block 64: thread i owns channels 2i, 2i + 1 of 128.
extern "C" __global__ void ling_prompt_tile_merge(
    const float* __restrict__ tile_o,            // [rows][32][192]
    const float* __restrict__ tile_ml,           // [rows][32][2]
    float* __restrict__ acc_o,
    float* __restrict__ acc_ml,
    __nv_bfloat16* __restrict__ out,             // [rows][32][192]
    const unsigned first,
    const unsigned last) {
    const unsigned head = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
    const unsigned at = row * LING_HEADS + head;
    const float2 t = reinterpret_cast<const float2*>(tile_o + at * LING_QK)[i];
    float2 o = t;
    float m = tile_ml[at * 2], l = tile_ml[at * 2 + 1];
    if (!first) {
        const float2 a = reinterpret_cast<const float2*>(acc_o + at * LING_QK)[i];
        const float am = acc_ml[at * 2], al = acc_ml[at * 2 + 1];
        const float top = fmaxf(am, m);
        const float wa = al * expf(am - top), wt = l * expf(m - top);
        const float inv = 1.f / (wa + wt);
        o = make_float2((a.x * wa + t.x * wt) * inv, (a.y * wa + t.y * wt) * inv);
        m = top;
        l = wa + wt;
    }
    if (last) {
        __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(out + at * LING_QK);
        dst[i] = __floats2bfloat162_rn(o.x, o.y);
        return;
    }
    reinterpret_cast<float2*>(acc_o + at * LING_QK)[i] = o;
    __syncthreads();
    if (i == 0) {
        acc_ml[at * 2] = m;
        acc_ml[at * 2 + 1] = l;
    }
}
