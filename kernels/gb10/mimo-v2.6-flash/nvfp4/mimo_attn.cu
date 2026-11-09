// SPDX-License-Identifier: AGPL-3.0-only

// MiMo-V2 attention kernels (Q/K head dim 192, V head dim 128).
//
// MiMo-V2.6-Flash mixes 9 full-attention layers (4 KV heads, RoPE theta 1e7)
// with 39 sliding-window layers (8 KV heads, window 128, theta 1e4, per-head
// attention-sink logit). Only the first 64 of the 192 Q/K dims are rotated
// (NeoX rotate_half). None of the shared paged-attention kernels handle a
// V head dim different from the Q/K head dim, so this file carries:
//
//   mimo_qkv_rope_cache      split the chunked fused-QKV GEMM output, RoPE
//                            Q and K, write Q contiguous and K/V to the
//                            paged cache.
//   mimo_paged_decode_attn   one query token per sequence, GQA, optional
//                            window + sink, optional split-K.
//   mimo_paged_decode_reduce split-K merge (+ sink) for the above.
//   mimo_prefill_attn        causal prefill of one sequence chunk against
//                            the paged cache (history + current chunk).
//
// Layouts
//   fused QKV row:  tp_chunks × [Q_c | K_c | V_c | pad], Q_c = nq/tp × 192,
//                   K_c = nkv/tp × 192, V_c = nkv/tp × 128, row stride
//                   `chunk_stride` per chunk (padded to the FP8 128-row
//                   block grid by the loader).
//   paged cache:    NHD [num_blocks, block_size, nkv, 192] BF16 for both K
//                   and V; V only uses the first 128 dims of each head.
//   q (contiguous): [T, nq, 192] BF16 (row stride `q_stride` elements).
//   out:            [T, nq, 128] BF16 (row stride `out_stride` elements).
//
// Scores use `scale` (1/sqrt(192)). The sink is a raw logit appended to the
// softmax row (HF: torch.cat([attn_weights, sinks]) after scaling) and
// contributes only to the denominator. `out_scale` (= attention_value_scale)
// multiplies the normalized output, which equals scaling V beforehand.
//
// Attention core: tiles of 32 keys are staged in shared memory; each warp
// owns one (query, head) row, every lane scores one key of the tile, and
// each lane accumulates 4 of the 128 V dims. Online softmax in FP32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#define MIMO_QK 192
#define MIMO_VD 128
#define MIMO_TILE 32
// Mirrors MIMO_SPLIT_TOKENS / MIMO_MAX_SPLITS in qwen3_attention/mimo.rs.
#define MIMO_SPLIT_TOKENS 512u
#define MIMO_MAX_SPLITS 16u
// +2 bf16 per K row: 388-byte stride maps lane j to bank (j*97) % 32 = j.
#define MIMO_KROW (MIMO_QK + 2)
#define MIMO_VROW MIMO_VD

static __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// FP8 KV (KvCacheDtype::Fp8): the paged pools store one E4M3 byte per element
// in the SAME element layout as the BF16 cache (192-stride head slots), so only
// the per-element width changes; scales are per-layer per-tensor, passed in.
static __device__ __forceinline__ float mimo_fp8_f32(__nv_fp8_storage_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

// ---------------------------------------------------------------------------
// QKV split + partial RoPE + paged cache write.
// Grid: (T, 1, 1). Block: (256, 1, 1).
// ---------------------------------------------------------------------------
extern "C" __global__ void mimo_qkv_rope_cache(
    const __nv_bfloat16* __restrict__ qkv,   // [T, tp_chunks * chunk_stride]
    __nv_bfloat16* __restrict__ q_out,       // [T, nq*192]
    __nv_bfloat16* __restrict__ k_cache,     // [num_blocks, bs, nkv, 192]
    __nv_bfloat16* __restrict__ v_cache,     // [num_blocks, bs, nkv, 192]
    const unsigned int* __restrict__ positions, // [T]
    const long long* __restrict__ slot_mapping, // [T]  (< 0 → no cache write)
    __nv_bfloat16* __restrict__ k_contig,    // [T, nkv, 192] roped K, or null
    __nv_bfloat16* __restrict__ v_contig,    // [T, nkv, 128] V, or null
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int tp_chunks,
    const unsigned int chunk_stride,
    const unsigned int rotary_dim,
    const float rope_theta,
    const unsigned int block_size,
    const unsigned int kv_write_floor,       // tokens [0, floor) skip the cache write
    const unsigned int* __restrict__ swa_map, // [num_blocks] block -> SWA slot, or null
    const unsigned int kv_fp8,               // 1 = paged cache holds FP8 E4M3
    const float k_inv_scale,                 // 1 / per-layer K scale (FP8 only)
    const float v_inv_scale                  // 1 / per-layer V scale (FP8 only)
) {
    const unsigned int t = blockIdx.x;
    const unsigned int nq_c = nq / tp_chunks;
    const unsigned int nkv_c = nkv / tp_chunks;
    const unsigned int row_stride = tp_chunks * chunk_stride;
    const __nv_bfloat16* row = qkv + (unsigned long long)t * row_stride;
    const float pos = (float)positions[t];
    const unsigned int half = rotary_dim / 2;
    const long long slot = slot_mapping[t];
    bool write_kv = slot >= 0 && t >= kv_write_floor;
    unsigned long long swa_blk = 0;
    if (write_kv && swa_map) {
        const unsigned int s = swa_map[slot / block_size];
        write_kv = s != 0xFFFFFFFFu;
        swa_blk = s;
    }

    __nv_bfloat16* k_dst = nullptr;
    __nv_bfloat16* v_dst = nullptr;
    __nv_fp8_storage_t* k_dst8 = nullptr;
    __nv_fp8_storage_t* v_dst8 = nullptr;
    if (write_kv) {
        const unsigned long long blk =
            swa_map ? swa_blk : (unsigned long long)(slot / block_size);
        const unsigned long long off = (unsigned long long)(slot % block_size);
        const unsigned long long base = (blk * block_size + off) * nkv * MIMO_QK;
        if (kv_fp8) {
            // FP8 pool: same element layout as BF16, one byte per element.
            k_dst8 = (__nv_fp8_storage_t*)k_cache + base;
            v_dst8 = (__nv_fp8_storage_t*)v_cache + base;
        } else {
            k_dst = k_cache + base;
            v_dst = v_cache + base;
        }
    }

    // Rows to RoPE: nq Q heads then nkv K heads. One (head, rotary pair) per item.
    const unsigned int n_heads = nq + nkv;
    for (unsigned int i = threadIdx.x; i < n_heads * MIMO_QK; i += blockDim.x) {
        const unsigned int h = i / MIMO_QK;
        const unsigned int d = i % MIMO_QK;
        const bool is_q = h < nq;
        const unsigned int hh = is_q ? h : h - nq;
        const unsigned int per_c = is_q ? nq_c : nkv_c;
        const unsigned int c = hh / per_c;
        const unsigned int local = hh % per_c;
        const unsigned int src_head_off =
            c * chunk_stride + (is_q ? 0u : nq_c * MIMO_QK) + local * MIMO_QK;
        const __nv_bfloat16* src = row + src_head_off;
        float x = __bfloat162float(src[d]);
        if (d < rotary_dim) {
            const unsigned int j = d < half ? d : d - half;
            const float inv_freq = powf(rope_theta, -(float)(2 * j) / (float)rotary_dim);
            float s, co;
            sincosf(pos * inv_freq, &s, &co);
            const float other = __bfloat162float(src[d < half ? d + half : d - half]);
            x = d < half ? x * co - other * s : x * co + other * s;
        }
        const __nv_bfloat16 y = __float2bfloat16(x);
        if (is_q) {
            q_out[(unsigned long long)t * nq * MIMO_QK + h * MIMO_QK + d] = y;
        } else {
            if (write_kv) {
                if (kv_fp8) {
                    // Quantize the roped FP32 value directly (single rounding).
                    k_dst8[hh * MIMO_QK + d] = __nv_cvt_float_to_fp8(
                        x * k_inv_scale, __NV_SATFINITE, __NV_E4M3);
                } else {
                    k_dst[hh * MIMO_QK + d] = y;
                }
            }
            if (k_contig) k_contig[((unsigned long long)t * nkv + hh) * MIMO_QK + d] = y;
        }
    }
    if (write_kv || v_contig) {
        for (unsigned int i = threadIdx.x; i < nkv * MIMO_VD; i += blockDim.x) {
            const unsigned int hh = i / MIMO_VD;
            const unsigned int d = i % MIMO_VD;
            const unsigned int c = hh / nkv_c;
            const unsigned int local = hh % nkv_c;
            const unsigned int src =
                c * chunk_stride + nq_c * MIMO_QK + nkv_c * MIMO_QK + local * MIMO_VD + d;
            if (write_kv) {
                if (kv_fp8) {
                    v_dst8[hh * MIMO_QK + d] = __nv_cvt_float_to_fp8(
                        __bfloat162float(row[src]) * v_inv_scale, __NV_SATFINITE, __NV_E4M3);
                } else {
                    v_dst[hh * MIMO_QK + d] = row[src];
                }
            }
            if (v_contig) v_contig[((unsigned long long)t * nkv + hh) * MIMO_VD + d] = row[src];
        }
    }
}

// ---------------------------------------------------------------------------
// Shared attention core.
// ---------------------------------------------------------------------------

// Physical pool index of logical block `i`: the block id, or its SWA slot
// when the layer's pool is slotted (`smap`, see kv_cache/swa_slots.rs).
static __device__ __forceinline__ unsigned long long mimo_block(
    const int* __restrict__ bt, const unsigned int* __restrict__ smap, unsigned int i) {
    const unsigned int b = (unsigned int)bt[i];
    return smap ? (unsigned long long)smap[b] : (unsigned long long)b;
}

// Stage keys [k0, k0+32) of one KV head into shared memory. Positions at or
// beyond `k_end` are zero-filled (masked later). Called by all threads.
// bt == null selects contiguous mode: k_cache/v_cache are [T, nkv, 192] /
// [T, nkv, 128] buffers indexed by (p - pos_base).
static __device__ __forceinline__ void mimo_load_tile(
    __nv_bfloat16* __restrict__ ks, __nv_bfloat16* __restrict__ vs,
    const __nv_bfloat16* __restrict__ k_cache, const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ bt, const unsigned int* __restrict__ smap,
    unsigned int k0, unsigned int k_end,
    unsigned int kvh, unsigned int nkv, unsigned int block_size, unsigned int pos_base,
    unsigned int kv_fp8, float k_scale, float v_scale) {
    const unsigned int nthreads = blockDim.x;
    // K: 32 rows × 96 bf16x2
    for (unsigned int i = threadIdx.x; i < MIMO_TILE * (MIMO_QK / 2); i += nthreads) {
        const unsigned int r = i / (MIMO_QK / 2);
        const unsigned int c2 = i % (MIMO_QK / 2);
        const unsigned int p = k0 + r;
        __nv_bfloat162 val = __floats2bfloat162_rn(0.f, 0.f);
        if (p < k_end) {
            unsigned long long base;
            if (bt) {
                const unsigned long long blk = mimo_block(bt, smap, p / block_size);
                base = ((blk * block_size + p % block_size) * nkv + kvh) * MIMO_QK;
            } else {
                base = ((unsigned long long)(p - pos_base) * nkv + kvh) * MIMO_QK;
            }
            if (kv_fp8 && bt) {
                // FP8 paged read: one E4M3 byte per element, same element layout.
                const __nv_fp8_storage_t* src8 = (const __nv_fp8_storage_t*)k_cache + base;
                val = __floats2bfloat162_rn(
                    mimo_fp8_f32(src8[c2 * 2]) * k_scale,
                    mimo_fp8_f32(src8[c2 * 2 + 1]) * k_scale);
            } else {
                val = reinterpret_cast<const __nv_bfloat162*>(k_cache + base)[c2];
            }
        }
        reinterpret_cast<__nv_bfloat162*>(ks + r * MIMO_KROW)[c2] = val;
    }
    // V: 32 rows × 64 bf16x2 (first 128 dims of each 192-wide head slot)
    for (unsigned int i = threadIdx.x; i < MIMO_TILE * (MIMO_VD / 2); i += nthreads) {
        const unsigned int r = i / (MIMO_VD / 2);
        const unsigned int c2 = i % (MIMO_VD / 2);
        const unsigned int p = k0 + r;
        __nv_bfloat162 val = __floats2bfloat162_rn(0.f, 0.f);
        if (p < k_end) {
            unsigned long long base;
            if (bt) {
                const unsigned long long blk = mimo_block(bt, smap, p / block_size);
                base = ((blk * block_size + p % block_size) * nkv + kvh) * MIMO_QK;
            } else {
                base = ((unsigned long long)(p - pos_base) * nkv + kvh) * MIMO_VD;
            }
            if (kv_fp8 && bt) {
                const __nv_fp8_storage_t* src8 = (const __nv_fp8_storage_t*)v_cache + base;
                val = __floats2bfloat162_rn(
                    mimo_fp8_f32(src8[c2 * 2]) * v_scale,
                    mimo_fp8_f32(src8[c2 * 2 + 1]) * v_scale);
            } else {
                val = reinterpret_cast<const __nv_bfloat162*>(v_cache + base)[c2];
            }
        }
        reinterpret_cast<__nv_bfloat162*>(vs + r * MIMO_VROW)[c2] = val;
    }
}

struct MimoRowState {
    float m;
    float l;
    float acc[4];
};

// One warp consumes the staged tile for a query row. `lo`/`hi` is the
// inclusive-exclusive key-position range this row may attend to.
static __device__ __forceinline__ void mimo_consume_tile(
    MimoRowState& st, const float* __restrict__ qs, const __nv_bfloat16* __restrict__ ks,
    const __nv_bfloat16* __restrict__ vs, unsigned int k0, unsigned int lo, unsigned int hi,
    float scale) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int p = k0 + lane;
    float s = -CUDART_INF_F;
    if (p >= lo && p < hi) {
        const __nv_bfloat162* kr = reinterpret_cast<const __nv_bfloat162*>(ks + lane * MIMO_KROW);
        float dot = 0.f;
#pragma unroll 8
        for (unsigned int d2 = 0; d2 < MIMO_QK / 2; ++d2) {
            const float2 kf = __bfloat1622float2(kr[d2]);
            dot += qs[2 * d2] * kf.x + qs[2 * d2 + 1] * kf.y;
        }
        s = dot * scale;
    }
    const float tmax = warp_max(s);
    if (tmax == -CUDART_INF_F) return;
    const float m_new = fmaxf(st.m, tmax);
    const float corr = __expf(st.m - m_new);
    const float pr = __expf(s - m_new);
    st.l = st.l * corr + warp_sum(pr);
#pragma unroll
    for (int i = 0; i < 4; ++i) st.acc[i] *= corr;
    for (unsigned int j = 0; j < MIMO_TILE; ++j) {
        const float pj = __shfl_sync(0xffffffffu, pr, j);
        if (pj == 0.f) continue;
        const __nv_bfloat162* vr =
            reinterpret_cast<const __nv_bfloat162*>(vs + j * MIMO_VROW + lane * 4);
        const float2 a = __bfloat1622float2(vr[0]);
        const float2 b = __bfloat1622float2(vr[1]);
        st.acc[0] += pj * a.x;
        st.acc[1] += pj * a.y;
        st.acc[2] += pj * b.x;
        st.acc[3] += pj * b.y;
    }
    st.m = m_new;
}

static __device__ __forceinline__ void mimo_store_row(
    const MimoRowState& st, float sink_logit, bool has_sink, float out_scale,
    __nv_bfloat16* __restrict__ out) {
    const unsigned int lane = threadIdx.x & 31;
    float denom = st.l;
    float m = st.m;
    if (has_sink) {
        if (m == -CUDART_INF_F) {
            m = sink_logit;
            denom = 1.f;
        } else {
            denom += __expf(sink_logit - m);
        }
    }
    const float inv = denom > 0.f ? out_scale / denom : 0.f;
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + lane * 4);
    o[0] = __floats2bfloat162_rn(st.acc[0] * inv, st.acc[1] * inv);
    o[1] = __floats2bfloat162_rn(st.acc[2] * inv, st.acc[3] * inv);
}

static __device__ __forceinline__ void mimo_load_q(float* __restrict__ qs,
                                                   const __nv_bfloat16* __restrict__ q) {
    const unsigned int lane = threadIdx.x & 31;
    for (unsigned int d = lane; d < MIMO_QK; d += 32) qs[d] = __bfloat162float(q[d]);
    __syncwarp();
}

// ---------------------------------------------------------------------------
// Split-K partitions of one row, from its own attended length alone, so a
// row's arithmetic does not depend on the batch it rides in (a speculative
// verify row matches the serial decode of the same token). `grid` is the
// launched split count: an upper bound for every row of the batch.
static __device__ __forceinline__ unsigned int mimo_row_splits(
    unsigned int span, unsigned int grid) {
    const unsigned int want = (span + MIMO_SPLIT_TOKENS - 1) / MIMO_SPLIT_TOKENS;
    return min(grid, max(1u, min(MIMO_MAX_SPLITS, want)));
}

// Paged decode. Grid: (nkv, num_splits, num_seqs). Block: (32 * group, 1, 1)
// where group = nq / nkv (≤ 32). Warp w handles Q head kvh*group + w.
//
// Each row uses mimo_row_splits(span) <= num_splits partitions; splits past
// that exit. One partition: writes the final normalized row to `out`.
// More: writes (m, l) to ws_ml[seq][head][split][2] and the
//                  unnormalized accumulator to ws_acc[seq][head][split][128];
//                  mimo_paged_decode_reduce finishes the row.
// ---------------------------------------------------------------------------
extern "C" __global__ void mimo_paged_decode_attn(
    const __nv_bfloat16* __restrict__ q,      // [num_seqs, q_stride]
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,         // [num_seqs], includes the query token
    const __nv_bfloat16* __restrict__ sinks,  // [nq] or null
    __nv_bfloat16* __restrict__ out,          // [num_seqs, out_stride]
    float* __restrict__ ws_ml,
    float* __restrict__ ws_acc,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int block_size,
    const unsigned int max_blocks_per_seq,
    const unsigned int window,                // 0 = full attention
    const float scale,
    const float out_scale,
    const unsigned int q_stride,
    const unsigned int out_stride,
    const unsigned int* __restrict__ swa_map, // block -> SWA slot, or null
    const unsigned int kv_fp8,               // 1 = paged cache holds FP8 E4M3
    const float k_scale,                     // per-layer K scale (FP8 only)
    const float v_scale                      // per-layer V scale (FP8 only)
) {
    __shared__ __align__(16) __nv_bfloat16 ks[MIMO_TILE * MIMO_KROW];
    __shared__ __align__(16) __nv_bfloat16 vs[MIMO_TILE * MIMO_VROW];
    __shared__ float qs_all[32 * MIMO_QK];

    const unsigned int kvh = blockIdx.x;
    const unsigned int split = blockIdx.y;
    const unsigned int num_splits = gridDim.y;
    const unsigned int seq = blockIdx.z;
    const unsigned int group = nq / nkv;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int head = kvh * group + warp;

    const int len_i = seq_lens[seq];
    const unsigned int len = len_i > 0 ? (unsigned int)len_i : 0u;
    const unsigned int lo = (window > 0 && len > window) ? len - window : 0u;
    // Split the attended range into 32-aligned chunks.
    const unsigned int span = len - lo;
    const unsigned int rs = mimo_row_splits(span, num_splits);
    if (split >= rs) return;
    const unsigned int per = ((span + rs - 1) / rs + MIMO_TILE - 1) / MIMO_TILE * MIMO_TILE;
    const unsigned int s_lo = lo + split * per;
    const unsigned int s_hi = min(len, s_lo + per);

    float* qs = qs_all + warp * MIMO_QK;
    mimo_load_q(qs, q + (unsigned long long)seq * q_stride + head * MIMO_QK);

    MimoRowState st;
    st.m = -CUDART_INF_F;
    st.l = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) st.acc[i] = 0.f;

    const int* bt = block_table + (unsigned long long)seq * max_blocks_per_seq;
    for (unsigned int k0 = s_lo; k0 < s_hi; k0 += MIMO_TILE) {
        __syncthreads();
        mimo_load_tile(ks, vs, k_cache, v_cache, bt, swa_map, k0, s_hi, kvh, nkv, block_size, 0u, kv_fp8, k_scale, v_scale);
        __syncthreads();
        mimo_consume_tile(st, qs, ks, vs, k0, s_lo, s_hi, scale);
    }

    if (rs == 1) {
        const bool has_sink = sinks != nullptr;
        const float sink = has_sink ? __bfloat162float(sinks[head]) : 0.f;
        mimo_store_row(st, sink, has_sink, out_scale, out + (unsigned long long)seq * out_stride + head * MIMO_VD);
        return;
    }
    const unsigned long long idx = ((unsigned long long)seq * nq + head) * num_splits + split;
    if (lane == 0) {
        ws_ml[idx * 2] = st.m;
        ws_ml[idx * 2 + 1] = st.l;
    }
    float4* a = reinterpret_cast<float4*>(ws_acc + idx * MIMO_VD + lane * 4);
    *a = make_float4(st.acc[0], st.acc[1], st.acc[2], st.acc[3]);
}

// Grid: (nq, num_seqs). Block: (128, 1, 1). Full attention only (window 0).
// Rows that took one partition were finished by the decode kernel.
extern "C" __global__ void mimo_paged_decode_reduce(
    const float* __restrict__ ws_ml,
    const float* __restrict__ ws_acc,
    const __nv_bfloat16* __restrict__ sinks,  // [nq] or null
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ seq_lens,         // [num_seqs]
    const unsigned int nq,
    const unsigned int num_splits,
    const float out_scale,
    const unsigned int out_stride
) {
    const unsigned int head = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const int len_i = seq_lens[seq];
    const unsigned int rs = mimo_row_splits(len_i > 0 ? (unsigned int)len_i : 0u, num_splits);
    if (rs == 1) return;
    const unsigned long long base = ((unsigned long long)seq * nq + head) * num_splits;
    float m = -CUDART_INF_F;
    for (unsigned int s = 0; s < rs; ++s) m = fmaxf(m, ws_ml[(base + s) * 2]);
    if (sinks != nullptr) m = fmaxf(m, __bfloat162float(sinks[head]));
    float l = 0.f;
    float acc = 0.f;
    for (unsigned int s = 0; s < rs; ++s) {
        const float ms = ws_ml[(base + s) * 2];
        if (ms == -CUDART_INF_F) continue;
        const float w = __expf(ms - m);
        l += ws_ml[(base + s) * 2 + 1] * w;
        acc += ws_acc[(base + s) * MIMO_VD + d] * w;
    }
    if (sinks != nullptr) l += __expf(__bfloat162float(sinks[head]) - m);
    if (d < MIMO_VD) {
        out[(unsigned long long)seq * out_stride + head * MIMO_VD + d] =
            __float2bfloat16(l > 0.f ? acc * out_scale / l : 0.f);
    }
}

// ---------------------------------------------------------------------------
// Prefill (single sequence chunk). Query i sits at absolute position
// pos0 + i and attends to cache positions [max(0, pos-window+1), pos].
// K/V for the chunk must already be in the cache (mimo_qkv_rope_cache), or,
// with block_table == null (chunk 0, no history), k_cache/v_cache are the
// contiguous k_contig/v_contig outputs of mimo_qkv_rope_cache.
// Grid: (ceil(T / 16), nq). Block: (512, 1, 1) — warp w owns query tile*16+w.
// ---------------------------------------------------------------------------
#define MIMO_PREFILL_Q 16

extern "C" __global__ void mimo_prefill_attn(
    const __nv_bfloat16* __restrict__ q,      // [T, q_stride]
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,      // [max_blocks] for this sequence, or null
    const __nv_bfloat16* __restrict__ sinks,  // [nq] or null
    __nv_bfloat16* __restrict__ out,          // [T, out_stride]
    const unsigned int num_tokens,
    const unsigned int pos0,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int block_size,
    const unsigned int window,
    const float scale,
    const float out_scale,
    const unsigned int q_stride,
    const unsigned int out_stride,
    const unsigned int* __restrict__ swa_map, // block -> SWA slot, or null
    const unsigned int kv_fp8,               // 1 = paged cache holds FP8 E4M3
    const float k_scale,                     // per-layer K scale (FP8 only)
    const float v_scale                      // per-layer V scale (FP8 only)
) {
    __shared__ __align__(16) __nv_bfloat16 ks[MIMO_TILE * MIMO_KROW];
    __shared__ __align__(16) __nv_bfloat16 vs[MIMO_TILE * MIMO_VROW];
    __shared__ float qs_all[MIMO_PREFILL_Q * MIMO_QK];

    const unsigned int head = blockIdx.y;
    const unsigned int kvh = head / (nq / nkv);
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int t0 = blockIdx.x * MIMO_PREFILL_Q;
    const unsigned int t = t0 + warp;
    const bool active = t < num_tokens;
    const unsigned int t_last = min(num_tokens, t0 + MIMO_PREFILL_Q) - 1;

    const unsigned int pos = pos0 + (active ? t : t_last);
    const unsigned int lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0u;
    const unsigned int hi = pos + 1;
    const unsigned int first_pos = pos0 + t0;
    const unsigned int tile_lo_raw =
        (window > 0 && first_pos + 1 > window) ? first_pos + 1 - window : 0u;
    unsigned int tile_lo = tile_lo_raw / MIMO_TILE * MIMO_TILE;
    if (block_table == nullptr && tile_lo < pos0) tile_lo = pos0;
    const unsigned int tile_hi = pos0 + t_last + 1;

    float* qs = qs_all + warp * MIMO_QK;
    if (active) mimo_load_q(qs, q + (unsigned long long)t * q_stride + head * MIMO_QK);

    MimoRowState st;
    st.m = -CUDART_INF_F;
    st.l = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) st.acc[i] = 0.f;

    for (unsigned int k0 = tile_lo; k0 < tile_hi; k0 += MIMO_TILE) {
        __syncthreads();
        mimo_load_tile(ks, vs, k_cache, v_cache, block_table, swa_map, k0, tile_hi, kvh, nkv, block_size, pos0, kv_fp8, k_scale, v_scale);
        __syncthreads();
        if (active && k0 < hi && k0 + MIMO_TILE > lo) {
            mimo_consume_tile(st, qs, ks, vs, k0, lo, hi, scale);
        }
    }
    if (!active) return;
    const bool has_sink = sinks != nullptr;
    const float sink = has_sink ? __bfloat162float(sinks[head]) : 0.f;
    mimo_store_row(st, sink, has_sink, out_scale, out + (unsigned long long)t * out_stride + head * MIMO_VD);
}
