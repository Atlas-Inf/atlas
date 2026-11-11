// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next exact fused decode kernels (ATLAS_QWEN4EXP_DECODE_FUSE=1):
// the GDN mixer's small kernels as one launch, and the EP MoE shared-expert
// blend with the mHC post (`moe_blend_hc_post`, at the end). The mHC seam
// twin (`hc_post_stage_vec`) lives in hyper_connection.cu with its helpers.
// The same step over the rows of a batched decode and of an exact MTP verify
// (`qwen4exp_gdn_decode_fused_rows`, `qwen4exp_gdn_verify_fused_rows`) serves
// ATLAS_QWEN4EXP_BATCH_SMALL=1.
//
// ── qwen4exp_gdn_decode_fused ──
//
// One launch for the four small kernels a single-token GDN mixer runs between
// its QKVZ and out projections:
//
//   causal_conv1d_update_l2norm_f32    (common/causal_conv1d.cu)
//   dense_gemv_ba_gates                (common/ssm_preprocess.cu)
//   gated_delta_rule_decode_f32        (gated_delta_rule.cu, this dir)
//   gated_rms_norm_f32_input_sigmoid   (common/rms_norm.cu)
//
// BIT-EXACT BY CONSTRUCTION. Every output element is produced by the same
// IEEE operations in the same order as in the four kernels (this target
// builds with --fmad=false, so no contraction differs either):
//
//   * BA gates: rows are split over 64 lanes with the same `kv += 64` walk,
//     the same 32-lane shfl_down tree and the same two-warp sum, exactly as a
//     64-thread quarter of `dense_gemv_ba_gates`'s 256-thread block does.
//     Block `vh` computes its own two rows (beta, alpha) in its two 64-thread
//     halves.
//   * Conv: the shifted window is held in registers and contracted in the
//     same k order; SiLU and the per-head L2 norm (shfl_down tree, then the
//     four warp sums left to right, rsqrtf(total + eps)) are the 128-thread
//     slice of the conv kernel's 256-thread block that holds this head.
//   * Recurrence and gated norm: `gated_delta_rule_decode_f32` and
//     `gated_rms_norm_f32_input_sigmoid` verbatim, the FP32 output kept in
//     shared instead of a global round trip (the same float either way).
//
// THE ONE STRUCTURAL CHANGE is the conv state. The conv kernel updates every
// channel's window in place, one thread per channel. Here block `vh` needs
// the q/k channels of key head vh / 3, which the other two v-heads of that
// key head (head_repeat = 48 / 16 = 3) need too. Each block therefore
// computes the q/k conv from the OLD window, and the three blocks of a key
// head form a thread-block cluster: after a cluster barrier (every block has
// consumed the old window) the first block of the cluster stores the shifted
// q/k windows. V channels belong to one v-head and are stored by their
// block. The stored windows equal the conv kernel's.
//
// Grid: (num_v_heads, 1, 1) with clusters of QDF_REPEAT; block QDF_D.
// Host checks: k_dim == v_dim == 128, head_repeat == 3, d_conv == 4,
// ba_k % 8 == 0, 16-byte alignment of the vector-loaded buffers, batch 1.
//
// Prior art: the conv+recurrence+norm chain fusion follows this tree's own
// `gated_delta_rule_decode_f32_conv_norm` (per-key-head grid, not exact); the
// cluster barrier for the shared window is ours.

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "../../common/atlas_pdl.cuh"

#define QDF_D 128
#define QDF_REPEAT 3
#define QDF_DCONV 4

// The conv kernel's SiLU and the conv window contraction, for one channel.
// `win` is the window AFTER the shift (old[1..3], new).
__device__ __forceinline__ float qdf_conv_silu(const float (&win)[QDF_DCONV],
                                               const __nv_bfloat16* __restrict__ w) {
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int k = 0; k < QDF_DCONV; k++) acc += win[k] * (float)w[k];
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    return acc * sigmoid_acc;
}

// The conv kernel's per-head L2 norm over 128 channels held one per thread.
// `ws` is 4 floats of shared.
__device__ __forceinline__ float qdf_l2(float silu, float* ws, float l2_eps) {
    const unsigned int tid = threadIdx.x;
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

// One token's conv on windows held in registers: shift in the token's
// channel inputs, contract and SiLU (`qdf_conv_silu`), L2-normalize q and k
// over the head into `smem_q` / `smem_k` (`qdf_l2`). Returns v (SiLU only).
// The windows are left holding what the conv kernel stores.
__device__ __forceinline__ float qdf_conv_step(float (&wq)[QDF_DCONV], float (&wk)[QDF_DCONV],
                                               float (&wv)[QDF_DCONV],
                                               const __nv_bfloat16* __restrict__ qkv,
                                               const unsigned int cq, const unsigned int ck,
                                               const unsigned int cv,
                                               const __nv_bfloat16* __restrict__ conv_w,
                                               float* smem_q, float* smem_k, const float l2_eps) {
    wq[0] = wq[1]; wq[1] = wq[2]; wq[2] = wq[3]; wq[3] = (float)qkv[cq];
    wk[0] = wk[1]; wk[1] = wk[2]; wk[2] = wk[3]; wk[3] = (float)qkv[ck];
    wv[0] = wv[1]; wv[1] = wv[2]; wv[2] = wv[3]; wv[3] = (float)qkv[cv];
    const float q_silu = qdf_conv_silu(wq, conv_w + (unsigned long long)cq * QDF_DCONV);
    const float k_silu = qdf_conv_silu(wk, conv_w + (unsigned long long)ck * QDF_DCONV);
    const float v_i = qdf_conv_silu(wv, conv_w + (unsigned long long)cv * QDF_DCONV);
    __shared__ float ws[4];
    smem_q[threadIdx.x] = qdf_l2(q_silu, ws, l2_eps);
    smem_k[threadIdx.x] = qdf_l2(k_silu, ws, l2_eps);
    return v_i;
}

// `gated_delta_rule_decode_f32`'s step for this block's value head (thread
// = column of H, held in `H_reg`): the clamped gate, the hk and q dots, the
// state update. Returns the FP32 output `x` (before the gated norm).
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

// `gated_rms_norm_f32_input_sigmoid` over one head (block = 128): `x` per
// thread, the head's Z gate and output rows.
__device__ __forceinline__ void qdf_gated_norm(const float x,
                                               const __nv_bfloat16* __restrict__ z_head,
                                               const __nv_bfloat16* __restrict__ norm_w,
                                               __nv_bfloat16* __restrict__ out_head,
                                               const unsigned int head_dim, const float eps) {
    const unsigned int tid = threadIdx.x;
    __shared__ float x_cache[QDF_D];
    x_cache[tid] = x;
    float sum_sq = 0.0f;
    sum_sq += x * x;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
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
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
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

// One block (value head blockIdx.x) of the fused step for one token: the
// body of `qwen4exp_gdn_decode_fused` and, a token per blockIdx.y, of
// `qwen4exp_gdn_decode_fused_rows`. The caller is a (QDF_REPEAT, 1, 1)
// cluster of QDF_D-thread blocks.
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
    namespace cg = cooperative_groups;

    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
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
    __shared__ float smem_ba[4];
    __shared__ float smem_gb[2];
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
    __shared__ float smem_k[QDF_D];
    __shared__ float smem_q[QDF_D];
    const float v_i = qdf_conv_step(wq, wk, wv, qkv, cq, ck, cv, conv_w, smem_q, smem_k, l2_eps);

    // Every block of the key head has read the old q/k window (and used it):
    // only now may the cluster's first block overwrite it.
    cg::this_cluster().sync();
    if (cg::this_cluster().block_rank() == 0) {
        qdf_store_window(conv_state + (unsigned long long)cq * QDF_DCONV, wq);
        qdf_store_window(conv_state + (unsigned long long)ck * QDF_DCONV, wk);
    }
    qdf_store_window(conv_state + (unsigned long long)cv * QDF_DCONV, wv);

    // ── Recurrence: gated_delta_rule_decode_f32 ──
    const float x = qdf_recur(H_reg, smem_k, smem_q, smem_gb[0], smem_gb[1], v_i, head_dim);

    // ── gated_rms_norm_f32_input_sigmoid over this head (block = 128) ──
    qdf_gated_norm(x, z_gate + vh * QDF_D, norm_w, out + vh * QDF_D, head_dim, eps);

    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_global[j * QDF_D + tid] = H_reg[j];
    }
}

extern "C" __global__ void __cluster_dims__(QDF_REPEAT, 1, 1) __launch_bounds__(QDF_D, 1)
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
// `nv * 128` apart. A cluster is three heads of one sequence, so the conv
// window handoff is the single-sequence one.
//
// Grid: (num_v_heads, rows, 1) with clusters of QDF_REPEAT; block QDF_D.
#define QDF_ROWS_MAX 8

struct QdfRowStates {
    float* h[QDF_ROWS_MAX];     // [nv, 128, 128] FP32 per sequence
    float* conv[QDF_ROWS_MAX];  // [conv_dim, 4] FP32 per sequence
};

extern "C" __global__ void __cluster_dims__(QDF_REPEAT, 1, 1) __launch_bounds__(QDF_D, 1)
qwen4exp_gdn_decode_fused_rows(
    const __grid_constant__ QdfRowStates states,
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

// ── qwen4exp_gdn_verify_fused_rows (ATLAS_QWEN4EXP_BATCH_SMALL) ──
//
// The exact MTP verify's GDN chain for up to QDF_ROWS_MAX sequences, each
// with its own `k` <= QDF_VERIFY_KMAX consecutive tokens (rows row0..row0+k-1
// of the step), in one launch. Per token the exact arm
// (trait_decode_batched_conv_gdn_exact.rs) runs causal_conv1d_update_l2norm_f32,
// a copy of the conv state to its rollback slot, gated_delta_rule_decode_f32,
// gated_rms_norm_f32_input_sigmoid and a copy of the recurrence state to its
// slot (the copies for tokens 0..k-2). Block (vh, s) does all of that for
// value head vh of sequence s with H and the conv windows in registers across
// the tokens: each token is the same IEEE operations in the same order (H
// stored and reloaded in FP32 is the same float), the rollback slots get the
// bytes the copies give, and the final state is the k-th token's. Gates and
// betas come from the step's gate rows, as the exact arm reads them.
//
// The q/k windows of a key head are evolved redundantly by its three blocks
// (the same values); the cluster's first block stores them, once every block
// has loaded the old ones.
//
// Grid: (num_v_heads, sequences, 1) with clusters of QDF_REPEAT; block QDF_D.
// 8 = the exact lane's deepest verify window (7 drafts, verify_rows.rs). The
// token loop runs to the sequence's own k, so a k <= 4 sequence executes the
// same instructions it did under KMAX 4; only the by-value table grows
// (8 x 136 B).
#define QDF_VERIFY_KMAX 8

struct QdfVerifySeq {
    float* h;                                // [nv, 128, 128] FP32
    float* conv;                             // [conv_dim, 4] FP32
    float* h_snap[QDF_VERIFY_KMAX - 1];      // after token t, t < k - 1
    float* conv_snap[QDF_VERIFY_KMAX - 1];
    unsigned int row0;                       // first row of the step
    unsigned int k;                          // tokens, 1..QDF_VERIFY_KMAX
};

struct QdfVerifyRows {
    QdfVerifySeq seq[QDF_ROWS_MAX];
};

extern "C" __global__ void __cluster_dims__(QDF_REPEAT, 1, 1) __launch_bounds__(QDF_D, 1)
qwen4exp_gdn_verify_fused_rows(
    const __grid_constant__ QdfVerifyRows rows,
    const __nv_bfloat16* __restrict__ qkvz,    // [rows, qkvz_stride]: Q|K|V|Z
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const float* __restrict__ gates,           // [rows, 2 * nv]: gate | beta
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [rows, nv * 128]
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const unsigned int qkvz_stride,
    const float l2_eps,
    const float eps
) {
    namespace cg = cooperative_groups;
    const QdfVerifySeq& sq = rows.seq[blockIdx.y];
    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int kh = vh / (num_v_heads / num_k_heads);
    const unsigned int key_dim = num_k_heads * QDF_D;
    const unsigned long long conv_dim = 2ull * key_dim + num_v_heads * QDF_D;
    const unsigned long long head = (unsigned long long)vh * QDF_D * QDF_D;
    const bool lead = cg::this_cluster().block_rank() == 0;

    float H_reg[QDF_D];
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_reg[j] = sq.h[head + j * QDF_D + tid];
    }
    const unsigned int cq = kh * QDF_D + tid;
    const unsigned int ck = key_dim + kh * QDF_D + tid;
    const unsigned int cv = 2u * key_dim + vh * QDF_D + tid;
    float wq[QDF_DCONV], wk[QDF_DCONV], wv[QDF_DCONV];
    qdf_load_window(sq.conv + (unsigned long long)cq * QDF_DCONV, wq);
    qdf_load_window(sq.conv + (unsigned long long)ck * QDF_DCONV, wk);
    qdf_load_window(sq.conv + (unsigned long long)cv * QDF_DCONV, wv);

    __shared__ float smem_k[QDF_D];
    __shared__ float smem_q[QDF_D];
    for (unsigned int t = 0; t < sq.k; t++) {
        const unsigned long long row = sq.row0 + t;
        const __nv_bfloat16* qkv = qkvz + row * qkvz_stride;
        const float v_i = qdf_conv_step(wq, wk, wv, qkv, cq, ck, cv, conv_w, smem_q, smem_k, l2_eps);
        __syncthreads();  // every smem_k / smem_q entry before the dots read them
        const bool snap = t + 1 < sq.k;
        if (snap) {
            if (lead) {
                qdf_store_window(sq.conv_snap[t] + (unsigned long long)cq * QDF_DCONV, wq);
                qdf_store_window(sq.conv_snap[t] + (unsigned long long)ck * QDF_DCONV, wk);
            }
            qdf_store_window(sq.conv_snap[t] + (unsigned long long)cv * QDF_DCONV, wv);
        }
        const float* g = gates + row * 2u * num_v_heads;
        const float x = qdf_recur(H_reg, smem_k, smem_q, g[vh], g[num_v_heads + vh], v_i, head_dim);
        if (snap) {
            #pragma unroll
            for (int j = 0; j < QDF_D; j++) {
                sq.h_snap[t][head + j * QDF_D + tid] = H_reg[j];
            }
        }
        qdf_gated_norm(x, qkv + conv_dim + vh * QDF_D, norm_w,
                       out + row * num_v_heads * QDF_D + vh * QDF_D, head_dim, eps);
    }

    // Every block of the key head has loaded the old q/k windows.
    cg::this_cluster().sync();
    if (lead) {
        qdf_store_window(sq.conv + (unsigned long long)cq * QDF_DCONV, wq);
        qdf_store_window(sq.conv + (unsigned long long)ck * QDF_DCONV, wk);
    }
    qdf_store_window(sq.conv + (unsigned long long)cv * QDF_DCONV, wv);
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        sq.h[head + j * QDF_D + tid] = H_reg[j];
    }
}

// ── qwen4exp_gdn_verify_defer_rows + qwen4exp_gdn_commit_layers ──
// (ATLAS_QWEN4EXP_EXACT_DEFER)
//
// The exact verify above writes, per sequence and layer, the recurrence state
// after every token but the last (k - 1 rollback slots of nv x 128 x 128 FP32)
// plus the final state, so the commit of an `n`-token accepted prefix is a
// copy of slot n - 1. At C8 x K=4 that is 3 + 1 state writes per sequence and
// layer inside the verify and a 56.6 MB copy per partially accepted sequence
// after it -- most of the verify step's DRAM traffic.
//
// Deferred, the verify writes no state: `qwen4exp_gdn_verify_defer_rows` is
// the kernel above with the slot and final stores removed and the token's
// conv inputs (the raw BF16 Q|K|V channels) and its gate and beta staged per
// sequence instead (a few KiB). The state stays at H0 and the windows before
// the step. After the verdict `qwen4exp_gdn_commit_layers` replays the
// accepted `n` tokens from there -- one launch per sequence over all its GDN
// layers -- with the same functions the verify ran (`qdf_conv_step`,
// `qdf_recur`) on the same inputs, so the state it stores is, byte for byte,
// slot n - 1 (or the final state when n == k) of the storing kernel. Only the
// gated norm, whose output the verify already produced, is skipped; it does
// not feed the state. Both kernels are checked against the storing one by
// scripts/dev/qwen4exp_batch_small_bench.cu (defer-check).
//
// Commit fused into the next verify (ATLAS_QWEN4EXP_GDN_COMMIT_FUSE): with
// `fuse_n` set and the word it points at n > 0, the previous step's commit of
// n accepted tokens is still pending -- `h` and `conv` hold the state before
// that step and the staging holds its inputs. The verify first runs
// `qwen4exp_gdn_commit_layers`' token loop on them (the same calls, the same
// barriers, from the same loaded state), stores the committed recurrence and
// windows exactly as that kernel does, then verifies its own rows from the
// state still in registers -- the floats the unfused verify reloads from the
// stored state. One state read and one write a step instead of the commit's
// read and write plus the verify's read. The cluster barrier after the replay
// also orders every block's reads of the old staging before the lead block
// overwrites the q/k staging with this step's rows. A NULL `fuse_n` (switch
// off) or n == 0 skips all of it: the kernel above, unchanged.
struct QdfDeferSeq {
    float* h;                      // [nv, 128, 128] FP32: H0 (written only by a fused commit)
    float* conv;                   // [conv_dim, 4] FP32: the step's input windows (idem)
    __nv_bfloat16* stage_qkv;      // [k, conv_dim] BF16: token t's raw Q | K | V channels
    float* stage_gb;               // [k, 2 * nv] FP32: token t's gate | beta
    unsigned int row0;             // first row of the step
    unsigned int k;                // tokens, 1..QDF_VERIFY_KMAX
    const unsigned int* fuse_n;    // the pending commit's accepted tokens, or NULL
};

struct QdfDeferRows {
    QdfDeferSeq seq[QDF_ROWS_MAX];
};

extern "C" __global__ void __cluster_dims__(QDF_REPEAT, 1, 1) __launch_bounds__(QDF_D, 1)
qwen4exp_gdn_verify_defer_rows(
    const __grid_constant__ QdfDeferRows rows,
    const __nv_bfloat16* __restrict__ qkvz,    // [rows, qkvz_stride]: Q|K|V|Z
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const float* __restrict__ gates,           // [rows, 2 * nv]: gate | beta
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [rows, nv * 128]
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const unsigned int qkvz_stride,
    const float l2_eps,
    const float eps
) {
    namespace cg = cooperative_groups;
    const QdfDeferSeq& sq = rows.seq[blockIdx.y];
    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int kh = vh / (num_v_heads / num_k_heads);
    const unsigned int key_dim = num_k_heads * QDF_D;
    const unsigned long long conv_dim = 2ull * key_dim + num_v_heads * QDF_D;
    const unsigned long long head = (unsigned long long)vh * QDF_D * QDF_D;
    const bool lead = cg::this_cluster().block_rank() == 0;

    float H_reg[QDF_D];
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_reg[j] = sq.h[head + j * QDF_D + tid];
    }
    const unsigned int cq = kh * QDF_D + tid;
    const unsigned int ck = key_dim + kh * QDF_D + tid;
    const unsigned int cv = 2u * key_dim + vh * QDF_D + tid;
    float wq[QDF_DCONV], wk[QDF_DCONV], wv[QDF_DCONV];
    qdf_load_window(sq.conv + (unsigned long long)cq * QDF_DCONV, wq);
    qdf_load_window(sq.conv + (unsigned long long)ck * QDF_DCONV, wk);
    qdf_load_window(sq.conv + (unsigned long long)cv * QDF_DCONV, wv);

    __shared__ float smem_k[QDF_D];
    __shared__ float smem_q[QDF_D];
    const unsigned int n_prev = sq.fuse_n ? *sq.fuse_n : 0u;
    if (n_prev > QDF_VERIFY_KMAX) __trap();  // the host commits at most k <= KMAX
    if (n_prev) {
        // `qwen4exp_gdn_commit_layers`, verbatim, on the previous step's staging.
        for (unsigned int t = 0; t < n_prev; t++) {
            const __nv_bfloat16* sqkv = sq.stage_qkv + (unsigned long long)t * conv_dim;
            const float* sg = sq.stage_gb + (unsigned long long)t * 2u * num_v_heads;
            const float v_i = qdf_conv_step(wq, wk, wv, sqkv, cq, ck, cv, conv_w, smem_q, smem_k, l2_eps);
            __syncthreads();
            (void)qdf_recur(H_reg, smem_k, smem_q, sg[vh], sg[num_v_heads + vh], v_i, head_dim);
            __syncthreads();
        }
        cg::this_cluster().sync();
        if (lead) {
            qdf_store_window(sq.conv + (unsigned long long)cq * QDF_DCONV, wq);
            qdf_store_window(sq.conv + (unsigned long long)ck * QDF_DCONV, wk);
        }
        qdf_store_window(sq.conv + (unsigned long long)cv * QDF_DCONV, wv);
        #pragma unroll
        for (int j = 0; j < QDF_D; j++) {
            sq.h[head + j * QDF_D + tid] = H_reg[j];
        }
    }
    for (unsigned int t = 0; t < sq.k; t++) {
        const unsigned long long row = sq.row0 + t;
        const __nv_bfloat16* qkv = qkvz + row * qkvz_stride;
        const float* g = gates + row * 2u * num_v_heads;
        // Stage what the commit replays: the cluster's lead the key head's
        // q/k channels (all three blocks read the same), every block its v.
        __nv_bfloat16* st = sq.stage_qkv + (unsigned long long)t * conv_dim;
        if (lead) {
            st[cq] = qkv[cq];
            st[ck] = qkv[ck];
        }
        st[cv] = qkv[cv];
        if (tid == 0) {
            float* gb = sq.stage_gb + (unsigned long long)t * 2u * num_v_heads;
            gb[vh] = g[vh];
            gb[num_v_heads + vh] = g[num_v_heads + vh];
        }
        const float v_i = qdf_conv_step(wq, wk, wv, qkv, cq, ck, cv, conv_w, smem_q, smem_k, l2_eps);
        __syncthreads();  // every smem_k / smem_q entry before the dots read them
        const float x = qdf_recur(H_reg, smem_k, smem_q, g[vh], g[num_v_heads + vh], v_i, head_dim);
        qdf_gated_norm(x, qkv + conv_dim + vh * QDF_D, norm_w,
                       out + row * num_v_heads * QDF_D + vh * QDF_D, head_dim, eps);
    }
}

// One GDN layer of one sequence for the commit: its live state (H0 and the
// windows, updated in place) and what the deferred verify staged for it.
struct QdfCommitLayer {
    float* h;                           // [nv, 128, 128] FP32
    float* conv;                        // [conv_dim, 4] FP32
    const __nv_bfloat16* stage_qkv;     // [n, conv_dim] BF16
    const float* stage_gb;              // [n, 2 * nv] FP32
    const __nv_bfloat16* conv_w;        // this layer's [conv_dim, 4]
};

#define QDF_COMMIT_LAYERS 48

struct QdfCommitLayers {
    QdfCommitLayer layer[QDF_COMMIT_LAYERS];
};

// Grid: (num_v_heads, layers, 1) with clusters of QDF_REPEAT; block QDF_D.
extern "C" __global__ void __cluster_dims__(QDF_REPEAT, 1, 1) __launch_bounds__(QDF_D, 1)
qwen4exp_gdn_commit_layers(
    const __grid_constant__ QdfCommitLayers layers,
    const unsigned int n_tokens,               // the accepted prefix, 1..k
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const float l2_eps
) {
    namespace cg = cooperative_groups;
    const QdfCommitLayer& L = layers.layer[blockIdx.y];
    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int kh = vh / (num_v_heads / num_k_heads);
    const unsigned int key_dim = num_k_heads * QDF_D;
    const unsigned long long conv_dim = 2ull * key_dim + num_v_heads * QDF_D;
    const unsigned long long head = (unsigned long long)vh * QDF_D * QDF_D;
    const bool lead = cg::this_cluster().block_rank() == 0;

    float H_reg[QDF_D];
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        H_reg[j] = L.h[head + j * QDF_D + tid];
    }
    const unsigned int cq = kh * QDF_D + tid;
    const unsigned int ck = key_dim + kh * QDF_D + tid;
    const unsigned int cv = 2u * key_dim + vh * QDF_D + tid;
    float wq[QDF_DCONV], wk[QDF_DCONV], wv[QDF_DCONV];
    qdf_load_window(L.conv + (unsigned long long)cq * QDF_DCONV, wq);
    qdf_load_window(L.conv + (unsigned long long)ck * QDF_DCONV, wk);
    qdf_load_window(L.conv + (unsigned long long)cv * QDF_DCONV, wv);

    __shared__ float smem_k[QDF_D];
    __shared__ float smem_q[QDF_D];
    for (unsigned int t = 0; t < n_tokens; t++) {
        const __nv_bfloat16* qkv = L.stage_qkv + (unsigned long long)t * conv_dim;
        const float* g = L.stage_gb + (unsigned long long)t * 2u * num_v_heads;
        const float v_i = qdf_conv_step(wq, wk, wv, qkv, cq, ck, cv, L.conv_w, smem_q, smem_k, l2_eps);
        __syncthreads();
        (void)qdf_recur(H_reg, smem_k, smem_q, g[vh], g[num_v_heads + vh], v_i, head_dim);
        __syncthreads();  // the dots have read smem_k / smem_q before the next token's conv
    }

    // Every block of the key head has loaded the old q/k windows.
    cg::this_cluster().sync();
    if (lead) {
        qdf_store_window(L.conv + (unsigned long long)cq * QDF_DCONV, wq);
        qdf_store_window(L.conv + (unsigned long long)ck * QDF_DCONV, wk);
    }
    qdf_store_window(L.conv + (unsigned long long)cv * QDF_DCONV, wv);
    #pragma unroll
    for (int j = 0; j < QDF_D; j++) {
        L.h[head + j * QDF_D + tid] = H_reg[j];
    }
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
