// SPDX-License-Identifier: AGPL-3.0-only
//
// strix-hip (gfx1151) port of kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4exp_decode_fuse.cu:
// the fused Flash-Next GDN decode step (`qwen4exp_gdn_decode_fused`, ATLAS_QWEN4EXP_DECODE_FUSE),
// its multi-sequence form (`qwen4exp_gdn_decode_fused_rows`, ATLAS_QWEN4EXP_BATCH_SMALL), and
// `moe_blend_hc_post` (copied unchanged).
//
// gb10 runs a value head per QDF_D-thread block and groups the QDF_REPEAT value heads of one key
// head into a thread-block CLUSTER: every block evolves the key head's q/k conv window from the
// old one, a cluster barrier waits until all of them have read it, then the cluster's first block
// stores the new window. gfx1151 has no clusters and the gb10 file did not build here, so both
// switches silently kept the per-row four-kernel chain (framework f130 / f159 "neutral" measured
// that fallback; f154: gated_delta_rule_decode_f32 at 73,476 launches, 7.4% of a C8 step).
//
// Here one block IS the cluster: QDF_REPEAT * QDF_D threads, sub-block s = threadIdx.x / QDF_D
// owning value head blockIdx.x * QDF_REPEAT + s, as gb10 block s of the cluster does. The cluster
// barrier becomes __syncthreads (all three sub-blocks have read the old window), and every piece
// of shared memory a gb10 block owns is indexed by s (static __shared__ in a device function is
// ONE instance per block, so the helpers take it explicitly). Each thread runs the same IEEE
// operations in the same order as the gb10 thread with the same (head, tid), so the bytes are the
// gb10 kernels', which are the four-kernel chain's.
//
// Grid: (num_v_heads / QDF_REPEAT, rows, 1)   Block: (QDF_REPEAT * QDF_D, 1, 1); the launchers pick
// that geometry under cfg(atlas_hip). The verify / defer / commit kernels stay gb10-only: their
// lookups resolve to 0 here and the callers keep their per-token arms, as before.

#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"

#define QDF_D 128
#define QDF_REPEAT 3
#define QDF_DCONV 4
#define QDF_BLOCK (QDF_REPEAT * QDF_D)

__device__ __forceinline__ float qdf_conv_silu(const float (&win)[QDF_DCONV],
                                               const __nv_bfloat16* __restrict__ w) {
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int k = 0; k < QDF_DCONV; k++) acc += win[k] * (float)w[k];
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    return acc * sigmoid_acc;
}

// One sub-block's L2 norm over its 128 lanes (4 wave32 warps). `ws` is the sub-block's own 4 floats.
// The barriers are block-wide: every sub-block calls this in lockstep.
__device__ __forceinline__ float qdf_l2(float silu, float* ws, unsigned int tid, float l2_eps) {
    float sq = silu * silu;
    for (int offset = 16; offset >= 1; offset >>= 1)
        sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
    if ((tid & 31u) == 0) ws[tid >> 5] = sq;
    __syncthreads();
    if (tid == 0) {
        float total = ws[0] + ws[1] + ws[2] + ws[3];
        ws[0] = rsqrtf(total + l2_eps);
    }
    __syncthreads();
    float r = silu;
    r *= ws[0];
    __syncthreads();  // ws is reused by the next head
    return r;
}

__device__ __forceinline__ void qdf_load_window(const float* p, float (&old)[QDF_DCONV]) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    old[0] = v.x; old[1] = v.y; old[2] = v.z; old[3] = v.w;
}

__device__ __forceinline__ void qdf_store_window(float* p, const float (&win)[QDF_DCONV]) {
    *reinterpret_cast<float4*>(p) = make_float4(win[0], win[1], win[2], win[3]);
}

__device__ __forceinline__ float qdf_conv_step(float (&wq)[QDF_DCONV], float (&wk)[QDF_DCONV],
                                               float (&wv)[QDF_DCONV],
                                               const __nv_bfloat16* __restrict__ qkv,
                                               const unsigned int cq, const unsigned int ck,
                                               const unsigned int cv,
                                               const __nv_bfloat16* __restrict__ conv_w,
                                               float* smem_q, float* smem_k, float* ws,
                                               const unsigned int tid, const float l2_eps) {
    wq[0] = wq[1]; wq[1] = wq[2]; wq[2] = wq[3]; wq[3] = (float)qkv[cq];
    wk[0] = wk[1]; wk[1] = wk[2]; wk[2] = wk[3]; wk[3] = (float)qkv[ck];
    wv[0] = wv[1]; wv[1] = wv[2]; wv[2] = wv[3]; wv[3] = (float)qkv[cv];
    const float q_silu = qdf_conv_silu(wq, conv_w + (unsigned long long)cq * QDF_DCONV);
    const float k_silu = qdf_conv_silu(wk, conv_w + (unsigned long long)ck * QDF_DCONV);
    const float v_i = qdf_conv_silu(wv, conv_w + (unsigned long long)cv * QDF_DCONV);
    smem_q[tid] = qdf_l2(q_silu, ws, tid, l2_eps);
    smem_k[tid] = qdf_l2(k_silu, ws, tid, l2_eps);
    return v_i;
}

__device__ __forceinline__ float qdf_recur(float (&H_reg)[QDF_D], const float* smem_k,
                                           const float* smem_q, const float gate,
                                           const float bt, const float v_i,
                                           const unsigned int head_dim) {
    float g = fminf(fmaxf(gate, 1e-6f), 1.0f - 1e-6f);

    float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < QDF_D; j += 4) {
        hk0 += H_reg[j]     * smem_k[j];
        hk1 += H_reg[j + 1] * smem_k[j + 1];
        hk2 += H_reg[j + 2] * smem_k[j + 2];
        hk3 += H_reg[j + 3] * smem_k[j + 3];
    }
    float hk_dot = (hk0 + hk1) + (hk2 + hk3);

    float v_new = (v_i - g * hk_dot) * bt;

    float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < QDF_D; j += 4) {
        float h0 = g * H_reg[j]     + smem_k[j]     * v_new;
        float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new;
        float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new;
        float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new;
        H_reg[j]     = h0;
        H_reg[j + 1] = h1;
        H_reg[j + 2] = h2;
        H_reg[j + 3] = h3;
        qd0 += h0 * smem_q[j];
        qd1 += h1 * smem_q[j + 1];
        qd2 += h2 * smem_q[j + 2];
        qd3 += h3 * smem_q[j + 3];
    }
    float q_dot = (qd0 + qd1) + (qd2 + qd3);

    // `rsqrtf` of the runtime head dim, as the recurrence kernel computes it:
    // a compile-time `rsqrtf(128.0f)` folds to the correctly rounded value,
    // one ulp off the hardware approximation the original issues, which
    // flips the BF16 rounding of a few outputs per thousand steps.
    float inv_sqrt_d = rsqrtf((float)head_dim);
    return q_dot * inv_sqrt_d;
}

__device__ __forceinline__ void qdf_gated_norm(const float x,
                                               const __nv_bfloat16* __restrict__ z_head,
                                               const __nv_bfloat16* __restrict__ norm_w,
                                               __nv_bfloat16* __restrict__ out_head,
                                               const unsigned int head_dim, const float eps,
                                               const unsigned int tid, float* x_cache,
                                               float* warp_sums) {
    x_cache[tid] = x;
    float sum_sq = 0.0f;
    sum_sq += x * x;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (QDF_D + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);

    const unsigned long long* g64 = (const unsigned long long*)z_head;
    const unsigned long long* w64 = (const unsigned long long*)norm_w;
    unsigned long long* out64 = (unsigned long long*)out_head;
    const unsigned int quad_size = QDF_D / 4;
    for (unsigned int i = tid; i < quad_size; i += QDF_D) {
        unsigned int base = i * 4;
        float f0 = x_cache[base];
        float f1 = x_cache[base + 1];
        float f2 = x_cache[base + 2];
        float f3 = x_cache[base + 3];

        unsigned long long wv64 = w64[i];
        float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wv64 & 0xFFFF)));
        float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wv64 >> 16) & 0xFFFF)));
        float w2 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wv64 >> 32) & 0xFFFF)));
        float w3 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wv64 >> 48)));

        unsigned long long gv = g64[i];
        float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gv & 0xFFFF)));
        float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((gv >> 16) & 0xFFFF)));
        float g2 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((gv >> 32) & 0xFFFF)));
        float g3 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(gv >> 48)));

        float s0 = 1.0f / (1.0f + expf(-g0));
        float s1 = 1.0f / (1.0f + expf(-g1));
        float s2 = 1.0f / (1.0f + expf(-g2));
        float s3 = 1.0f / (1.0f + expf(-g3));

        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(f0 * rms * w0 * s0))
                        | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(f1 * rms * w1 * s1)) << 16);
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(f2 * rms * w2 * s2))
                        | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(f3 * rms * w3 * s3)) << 16);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

// One QDF_BLOCK-thread block = gb10's cluster of QDF_REPEAT blocks (see the head of this file).
__device__ __forceinline__ void qdf_gdn_block(
    float* __restrict__ h_state,               // [nv, 128, 128] FP32
    float* __restrict__ conv_state,            // [conv_dim, 4] FP32
    const __nv_bfloat16* __restrict__ qkv,     // [conv_dim]: Q nk*128 | K nk*128 | V nv*128
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const __nv_bfloat16* __restrict__ ba_in,   // [ba_k] (the mixer input row)
    const __nv_bfloat16* __restrict__ ba_w,    // [2 * nv, ba_k], groups of 2 * vpg rows
    const float* __restrict__ a_log,           // [nv]
    const float* __restrict__ dt_bias,         // [nv]
    float* __restrict__ gate_out,              // [nv]
    float* __restrict__ beta_out,              // [nv]
    const __nv_bfloat16* __restrict__ z_gate,  // [nv, 128]
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [nv, 128]
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int ba_k,
    const unsigned int head_dim,               // 128 (host checks); a RUNTIME value
                                               // on purpose, see below
    const float l2_eps,
    const float eps
) {
    const unsigned int sub = threadIdx.x / QDF_D;              // gb10: block rank in the cluster
    const unsigned int vh = blockIdx.x * QDF_REPEAT + sub;     // gb10: blockIdx.x
    const unsigned int tid = threadIdx.x % QDF_D;              // gb10: the whole thread index
    const unsigned int vpg = num_v_heads / num_k_heads;
    const unsigned int kh = vh / vpg;
    const unsigned int key_dim = num_k_heads * QDF_D;
    float* H_global = h_state + (unsigned long long)vh * QDF_D * QDF_D;

    // The recurrence state is this block's alone and the bulk of its traffic:
    // issue it first so it streams while the gates and the conv run.
    float H_reg[QDF_D];
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_reg[j] = H_global[j * QDF_D + tid];
    }

    // ── BA gates: half 0 the beta row, half 1 the alpha row of v-head vh ──
    __shared__ float smem_ba_all[QDF_REPEAT][4];
    __shared__ float smem_gb_all[QDF_REPEAT][2];
    float* smem_ba = smem_ba_all[sub];
    float* smem_gb = smem_gb_all[sub];
    {
        const unsigned int which = tid >> 6;
        const unsigned int lane = tid & 63u;
        const unsigned int group = vh / vpg;
        const unsigned int within = vh % vpg;
        const unsigned int n = group * 2u * vpg + (which ? vpg : 0u) + within;
        float acc = 0.0f;
        const unsigned int K_VEC = ba_k / 8;
        const uint4* A_vec = (const uint4*)ba_in;
        const uint4* B_vec = (const uint4*)(ba_w + (unsigned long long)n * ba_k);
        for (unsigned int kv = lane; kv < K_VEC; kv += 64u) {
            uint4 a_data = A_vec[kv];
            uint4 b_data = B_vec[kv];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
                *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if ((tid & 31u) == 0) smem_ba[which * 2 + (lane / 32)] = acc;
        __syncthreads();
        if (lane == 0) {
            float result = smem_ba[which * 2] + smem_ba[which * 2 + 1];
            if (which == 0) {
                const float b = 1.0f / (1.0f + __expf(-result));
                beta_out[vh] = b;
                smem_gb[1] = b;
            } else {
                float a_log_val = a_log[vh];
                float dt_b = dt_bias[vh];
                float A_val = __expf(fminf(a_log_val, 20.0f));
                float dt = __logf(1.0f + __expf(fminf(result + dt_b, 20.0f)));
                const float gv = __expf(-A_val * dt);
                gate_out[vh] = gv;
                smem_gb[0] = gv;
            }
        }
    }

    // ── Conv: q/k of key head kh (from the old window), v of v-head vh ──
    const unsigned int cq = kh * QDF_D + tid;
    const unsigned int ck = key_dim + kh * QDF_D + tid;
    const unsigned int cv = 2u * key_dim + vh * QDF_D + tid;
    float wq[QDF_DCONV], wk[QDF_DCONV], wv[QDF_DCONV];
    qdf_load_window(conv_state + (unsigned long long)cq * QDF_DCONV, wq);
    qdf_load_window(conv_state + (unsigned long long)ck * QDF_DCONV, wk);
    qdf_load_window(conv_state + (unsigned long long)cv * QDF_DCONV, wv);
    __shared__ float smem_k_all[QDF_REPEAT][QDF_D];
    __shared__ float smem_q_all[QDF_REPEAT][QDF_D];
    __shared__ float ws_all[QDF_REPEAT][4];
    float* smem_k = smem_k_all[sub];
    float* smem_q = smem_q_all[sub];
    const float v_i = qdf_conv_step(wq, wk, wv, qkv, cq, ck, cv, conv_w, smem_q, smem_k, ws_all[sub],
                                    tid, l2_eps);

    // Every block of the key head has read the old q/k window (and used it):
    // only now may the cluster's first block overwrite it.
    __syncthreads();  // gb10: cluster barrier
    if (sub == 0) {   // gb10: cluster block rank 0
        qdf_store_window(conv_state + (unsigned long long)cq * QDF_DCONV, wq);
        qdf_store_window(conv_state + (unsigned long long)ck * QDF_DCONV, wk);
    }
    qdf_store_window(conv_state + (unsigned long long)cv * QDF_DCONV, wv);

    // ── Recurrence: gated_delta_rule_decode_f32 ──
    const float x = qdf_recur(H_reg, smem_k, smem_q, smem_gb[0], smem_gb[1], v_i, head_dim);

    // ── gated_rms_norm_f32_input_sigmoid over this head (block = 128) ──
    __shared__ float x_cache_all[QDF_REPEAT][QDF_D];
    __shared__ float warp_sums_all[QDF_REPEAT][32];
    qdf_gated_norm(x, z_gate + vh * QDF_D, norm_w, out + vh * QDF_D, head_dim, eps, tid,
                   x_cache_all[sub], warp_sums_all[sub]);

    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_global[j * QDF_D + tid] = H_reg[j];
    }
}

extern "C" __global__ void __launch_bounds__(QDF_BLOCK, 1)
qwen4exp_gdn_decode_fused(
    float* __restrict__ h_state,               // [nv, 128, 128] FP32
    float* __restrict__ conv_state,            // [conv_dim, 4] FP32
    const __nv_bfloat16* __restrict__ qkv,     // [conv_dim]: Q nk*128 | K nk*128 | V nv*128
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const __nv_bfloat16* __restrict__ ba_in,   // [ba_k] (the mixer input row)
    const __nv_bfloat16* __restrict__ ba_w,    // [2 * nv, ba_k], groups of 2 * vpg rows
    const float* __restrict__ a_log,           // [nv]
    const float* __restrict__ dt_bias,         // [nv]
    float* __restrict__ gate_out,              // [nv]
    float* __restrict__ beta_out,              // [nv]
    const __nv_bfloat16* __restrict__ z_gate,  // [nv, 128]
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [nv, 128]
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int ba_k,
    const unsigned int head_dim,               // 128 (host checks); a RUNTIME value
                                               // on purpose, see below
    const float l2_eps,
    const float eps
) {
    atlas_pdl_enter();
    qdf_gdn_block(h_state, conv_state, qkv, conv_w, ba_in, ba_w, a_log, dt_bias, gate_out,
                  beta_out, z_gate, norm_w, out, num_k_heads, num_v_heads, ba_k, head_dim,
                  l2_eps, eps);
}

// ── qwen4exp_gdn_decode_fused_rows (ATLAS_QWEN4EXP_BATCH_SMALL) ──
//
// The fused step for up to QDF_ROWS_MAX independent sequences in one launch,
// one token each (a batched decode step): block (vh, r) is block vh of
// `qwen4exp_gdn_decode_fused` on sequence r, against that sequence's own
// recurrence and conv state (`states`, by value: graph-stable, no table to
// upload). Every byte is what a launch per sequence writes, which is what the
// four-kernel chain writes per sequence (see the head of this file).
//
// Rows: qkvz `qkvz_stride` BF16 apart (Q|K|V then the Z gate at conv_dim),
// ba_in `ba_k` apart, gates `2 * nv` FP32 apart (gate | beta), out
// `nv * 128` apart. A block (gb10: a cluster) is three heads of one sequence, so the conv
// window handoff is the single-sequence one.
//
// Grid: (num_v_heads / QDF_REPEAT, rows, 1); block QDF_BLOCK (see the head of this file).
#define QDF_ROWS_MAX 8

struct QdfRowStates {
    float* h[QDF_ROWS_MAX];     // [nv, 128, 128] FP32 per sequence
    float* conv[QDF_ROWS_MAX];  // [conv_dim, 4] FP32 per sequence
};

extern "C" __global__ void __launch_bounds__(QDF_BLOCK, 1)
qwen4exp_gdn_decode_fused_rows(
    const QdfRowStates states,
    const __nv_bfloat16* __restrict__ qkvz,    // [rows, qkvz_stride]
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const __nv_bfloat16* __restrict__ ba_in,   // [rows, ba_k]
    const __nv_bfloat16* __restrict__ ba_w,    // [2 * nv, ba_k]
    const float* __restrict__ a_log,           // [nv]
    const float* __restrict__ dt_bias,         // [nv]
    float* __restrict__ gates,                 // [rows, 2 * nv]: gate | beta
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [rows, nv * 128]
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int ba_k,
    const unsigned int head_dim,
    const unsigned int qkvz_stride,
    const float l2_eps,
    const float eps
) {
    const unsigned int r = blockIdx.y;
    const unsigned long long conv_dim = (2ull * num_k_heads + num_v_heads) * QDF_D;
    const __nv_bfloat16* qkv = qkvz + (unsigned long long)r * qkvz_stride;
    float* g = gates + 2ull * r * num_v_heads;
    qdf_gdn_block(states.h[r], states.conv[r], qkv, conv_w, ba_in + (unsigned long long)r * ba_k,
                  ba_w, a_log, dt_bias, g, g + num_v_heads, qkv + conv_dim, norm_w,
                  out + (unsigned long long)r * num_v_heads * QDF_D, num_k_heads, num_v_heads,
                  ba_k, head_dim, l2_eps, eps);
}

// ── moe_blend_hc_post ──
//
// Under EP the MoE adds its gated shared expert after the expert all-reduce
// (`moe_batched_blend`, common/moe_permute.cu), and the layer then injects
// the MoE output into the highway (`hc_post_vec`, hyper_connection.cu). This
// kernel does both: each block recomputes the token's shared-expert gate
// exactly as `moe_batched_blend`'s one block does (256 threads, i += 256,
// shfl_down tree, the eight warp partials summed in order, sigmoid via
// __expf), blends its slice of the MoE output and stores it (the buffer holds
// what the blend leaves), then injects that stored BF16 value into every
// stream with `hc_post_vec`'s expression. Bit-identical (--fmad=false).
//
// Grid: (T, ceil(H / (4 * 256))), block 256 (the gate reduction is only
// bit-identical at that width). `streams` in place.
// Host checks: H % 4 == 0, 16-byte aligned highway.
#define QDF_BLEND_BLOCK 256

extern "C" __global__ void __launch_bounds__(QDF_BLEND_BLOCK)
moe_blend_hc_post(
    __nv_bfloat16* __restrict__ output,            // [T, H] routed sum in, blended out
    const __nv_bfloat16* __restrict__ shared_out,  // [T, H]
    const __nv_bfloat16* __restrict__ normed,      // [T, H] the MoE input
    const __nv_bfloat16* __restrict__ gate_weight, // [H] or null (gate 1)
    float* streams,                                // [T, hc, H]
    const float* __restrict__ inj,                 // [T, hc]
    const unsigned int hidden_size,
    const unsigned int hc
) {
    atlas_pdl_enter();
    __shared__ float s_dot_partial[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int H = hidden_size;

    const __nv_bfloat16* my_normed = normed + token * hidden_size;
    const __nv_bfloat16* my_shared = shared_out + token * hidden_size;
    __nv_bfloat16* my_output = output + token * hidden_size;

    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            float n = __bfloat162float(my_normed[i]);
            float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();

    float gate_scalar;
    if (tid == 0) {
        if (gate_weight == 0) {
            gate_scalar = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) {
                total += s_dot_partial[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    gate_scalar = s_dot_partial[0];

    const unsigned int d = 4u * (blockIdx.y * blockDim.x + tid);
    if (d >= H) return;
    float x[4];
    #pragma unroll
    for (unsigned int j = 0; j < 4; j++) {
        float o = __bfloat162float(my_output[d + j]);
        float s = __bfloat162float(my_shared[d + j]);
        const __nv_bfloat16 b = __float2bfloat16(o + gate_scalar * s);
        my_output[d + j] = b;
        x[j] = (float)b;
    }
    float* res = streams + (size_t)token * hc * H;
    for (unsigned int s = 0; s < hc; ++s) {
        const float wv = inj[(size_t)token * hc + s];
        const float4 r = *reinterpret_cast<const float4*>(res + (size_t)s * H + d);
        float4 v;
        v.x = r.x + x[0] * wv;
        v.y = r.y + x[1] * wv;
        v.z = r.z + x[2] * wv;
        v.w = r.w + x[3] * wv;
        *reinterpret_cast<float4*>(res + (size_t)s * H + d) = v;
    }
}
