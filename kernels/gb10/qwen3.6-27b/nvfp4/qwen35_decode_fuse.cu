// SPDX-License-Identifier: AGPL-3.0-only
//
// Dense hybrid Qwen3.6/3.8-27B exact fused GDN decode (ATLAS_QWEN38_DECODE_FUSE=1):
// the single-token GDN mixer's four small kernels between its QKVZ and out
// projections as one launch, `qwen35_gdn_decode_fused`:
//
//   dense_gemv_ba_gates                (common/ssm_preprocess.cu)
//   causal_conv1d_update_l2norm_f32    (common/causal_conv1d.cu)
//   gated_delta_rule_decode_f32        (gated_delta_rule.cu, THIS dir)
//   gated_rms_norm_f32_input           (common/rms_norm.cu, SiLU gate)
//
// The port of qwen3.8-flash-next/nvfp4/qwen4exp_decode_fuse.cu's
// `qwen4exp_gdn_decode_fused` to this tree. The block layout, the BA gates,
// the conv + L2 norm and the cluster handoff of the shared q/k conv window are
// that kernel's, unchanged (its BA, conv and norm sources are the same
// common/ files this tree serves). Two things differ, because the chain they
// reproduce differs:
//
//   * Recurrence: this tree's `gated_delta_rule_decode_f32` sums each pass
//     with ONE accumulator, four products a step
//     (`acc += h0*x0 + h1*x1 + h2*x2 + h3*x3`, i.e. acc + (((p0+p1)+p2)+p3)),
//     where Flash-Next's copy keeps four accumulators. `q35f_recur` is this
//     tree's loop, H held in registers instead of re-read from global (the
//     same floats).
//   * Gate: SiLU (`g / (1 + expf(-g))`, `gated_rms_norm_f32_input`) instead
//     of the sigmoid twin.
//
// BIT-EXACT BY CONSTRUCTION. Every output element (gate, beta, conv window,
// recurrence state, normed output) is produced by the same IEEE operations in
// the same order as by the four kernels; this target builds with
// --fmad=false, so no contraction differs either. So serial decode with the
// lever on computes what it computes without it, and the ATLAS_EXACT_VERIFY
// GDN chain (which runs these four kernels' arithmetic per verify row)
// remains its exact twin. `gdn_decode_fuse_check` (crates/spark-model/
// examples) compares every byte against the four kernels.
//
// Grid: (num_v_heads, 1, 1) with clusters of Q35F_REPEAT; block Q35F_D.
// Host checks (qwen4exp_decode_fuse.rs `gdn_geometry_fits`): k_dim == v_dim
// == 128, head_repeat == 3, d_conv == 4, ba_k % 8 == 0, 16-byte alignment of
// the vector-loaded buffers, batch 1, no conv bias.

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "../../common/atlas_pdl.cuh"

#define Q35F_D 128
#define Q35F_REPEAT 3
#define Q35F_DCONV 4

// The conv kernel's contraction (bias null: acc starts at 0) and SiLU for one
// channel. `win` is the window AFTER the shift (old[1..3], new).
__device__ __forceinline__ float q35f_conv_silu(const float (&win)[Q35F_DCONV],
                                                const __nv_bfloat16* __restrict__ w) {
    float acc = 0.0f;
    #pragma unroll
    for (unsigned int k = 0; k < Q35F_DCONV; k++) acc += win[k] * (float)w[k];
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    return acc * sigmoid_acc;
}

// The conv kernel's per-head L2 norm over 128 channels held one per thread:
// the shfl_down tree, the four warp sums left to right, rsqrtf(total + eps).
// `ws` is 4 floats of shared.
__device__ __forceinline__ float q35f_l2(float silu, float* ws, float l2_eps) {
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

__device__ __forceinline__ void q35f_load_window(const float* p, float (&old)[Q35F_DCONV]) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    old[0] = v.x; old[1] = v.y; old[2] = v.z; old[3] = v.w;
}

__device__ __forceinline__ void q35f_store_window(float* p, const float (&win)[Q35F_DCONV]) {
    *reinterpret_cast<float4*>(p) = make_float4(win[0], win[1], win[2], win[3]);
}

// This tree's `gated_delta_rule_decode_f32` for one value head (thread =
// column of H, held in `H_reg`): the clamped gate, the single-accumulator hk
// and q dots, the state update. Returns the FP32 output (before the norm).
__device__ __forceinline__ float q35f_recur(float (&H_reg)[Q35F_D], const float* smem_k,
                                            const float* smem_q, const float gate,
                                            const float bt, const float v_i,
                                            const unsigned int head_dim) {
    const float g = fminf(fmaxf(gate, 1e-6f), 1.0f - 1e-6f);

    float hk_dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < Q35F_D; j += 4) {
        float h0 = H_reg[j];     float h1 = H_reg[j + 1];
        float h2 = H_reg[j + 2]; float h3 = H_reg[j + 3];
        hk_dot += h0*smem_k[j] + h1*smem_k[j+1] + h2*smem_k[j+2] + h3*smem_k[j+3];
    }
    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
    #pragma unroll
    for (int j = 0; j < Q35F_D; j += 4) {
        float h0 = H_reg[j];     float h1 = H_reg[j + 1];
        float h2 = H_reg[j + 2]; float h3 = H_reg[j + 3];
        h0 = g*h0 + smem_k[j]*v_new_i;     h1 = g*h1 + smem_k[j+1]*v_new_i;
        h2 = g*h2 + smem_k[j+2]*v_new_i;   h3 = g*h3 + smem_k[j+3]*v_new_i;
        H_reg[j] = h0; H_reg[j + 1] = h1;
        H_reg[j + 2] = h2; H_reg[j + 3] = h3;
        q_dot += h0*smem_q[j] + h1*smem_q[j+1] + h2*smem_q[j+2] + h3*smem_q[j+3];
    }

    // `rsqrtf` of the RUNTIME head dim, as the recurrence kernel computes it:
    // a folded `rsqrtf(128.0f)` is the correctly rounded value, one ulp off
    // the hardware approximation (see qwen4exp_decode_fuse.cu).
    float inv_sqrt_d = rsqrtf((float)head_dim);
    return q_dot * inv_sqrt_d;
}

// `gated_rms_norm_f32_input` over one head (block = 128 threads, one element
// each): `x` per thread, the head's Z gate (SiLU) and output rows.
__device__ __forceinline__ void q35f_gated_norm(const float x,
                                                const __nv_bfloat16* __restrict__ z_head,
                                                const __nv_bfloat16* __restrict__ norm_w,
                                                __nv_bfloat16* __restrict__ out_head,
                                                const unsigned int head_dim, const float eps) {
    const unsigned int tid = threadIdx.x;
    __shared__ float x_cache[Q35F_D];
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
    const unsigned int quad_size = Q35F_D / 4;
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

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(f0 * rms * w0 * s0))
                        | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(f1 * rms * w1 * s1)) << 16);
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(f2 * rms * w2 * s2))
                        | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(f3 * rms * w3 * s3)) << 16);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}

extern "C" __global__ void __cluster_dims__(Q35F_REPEAT, 1, 1) __launch_bounds__(Q35F_D, 1)
qwen35_gdn_decode_fused(
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
    const float l2_eps,
    const float eps
) {
    atlas_pdl_enter();
    namespace cg = cooperative_groups;

    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int vpg = num_v_heads / num_k_heads;
    const unsigned int kh = vh / vpg;
    const unsigned int key_dim = num_k_heads * Q35F_D;
    float* H_global = h_state + (unsigned long long)vh * Q35F_D * Q35F_D;

    // The recurrence state is this block's alone and the bulk of its traffic:
    // issue it first so it streams while the gates and the conv run.
    float H_reg[Q35F_D];
    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        H_reg[j] = H_global[j * Q35F_D + tid];
    }

    // ── BA gates: half 0 the beta row, half 1 the alpha row of v-head vh ──
    // A 64-thread quarter of `dense_gemv_ba_gates`'s block per row: the same
    // `kv += 64` walk, 32-lane shfl_down tree and two-warp sum.
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
    const unsigned int cq = kh * Q35F_D + tid;
    const unsigned int ck = key_dim + kh * Q35F_D + tid;
    const unsigned int cv = 2u * key_dim + vh * Q35F_D + tid;
    float wq[Q35F_DCONV], wk[Q35F_DCONV], wv[Q35F_DCONV];
    q35f_load_window(conv_state + (unsigned long long)cq * Q35F_DCONV, wq);
    q35f_load_window(conv_state + (unsigned long long)ck * Q35F_DCONV, wk);
    q35f_load_window(conv_state + (unsigned long long)cv * Q35F_DCONV, wv);
    wq[0] = wq[1]; wq[1] = wq[2]; wq[2] = wq[3]; wq[3] = (float)qkv[cq];
    wk[0] = wk[1]; wk[1] = wk[2]; wk[2] = wk[3]; wk[3] = (float)qkv[ck];
    wv[0] = wv[1]; wv[1] = wv[2]; wv[2] = wv[3]; wv[3] = (float)qkv[cv];
    const float q_silu = q35f_conv_silu(wq, conv_w + (unsigned long long)cq * Q35F_DCONV);
    const float k_silu = q35f_conv_silu(wk, conv_w + (unsigned long long)ck * Q35F_DCONV);
    const float v_i = q35f_conv_silu(wv, conv_w + (unsigned long long)cv * Q35F_DCONV);
    __shared__ float smem_k[Q35F_D];
    __shared__ float smem_q[Q35F_D];
    __shared__ float ws[4];
    smem_q[tid] = q35f_l2(q_silu, ws, l2_eps);
    smem_k[tid] = q35f_l2(k_silu, ws, l2_eps);

    // Every block of the key head has read the old q/k window (and used it):
    // only now may the cluster's first block overwrite it.
    cg::this_cluster().sync();
    if (cg::this_cluster().block_rank() == 0) {
        q35f_store_window(conv_state + (unsigned long long)cq * Q35F_DCONV, wq);
        q35f_store_window(conv_state + (unsigned long long)ck * Q35F_DCONV, wk);
    }
    q35f_store_window(conv_state + (unsigned long long)cv * Q35F_DCONV, wv);

    // ── Recurrence: this tree's gated_delta_rule_decode_f32 ──
    const float x = q35f_recur(H_reg, smem_k, smem_q, smem_gb[0], smem_gb[1], v_i, head_dim);

    // ── gated_rms_norm_f32_input (SiLU) over this head (block = 128) ──
    q35f_gated_norm(x, z_gate + vh * Q35F_D, norm_w, out + vh * Q35F_D, head_dim, eps);

    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        H_global[j * Q35F_D + tid] = H_reg[j];
    }
}

// ── qwen35_gdn_verify_tree (ATLAS_QWEN38_VERIFY_FUSE=1) ──
//
// The ATLAS_EXACT_VERIFY GDN chain of ONE sequence's verify rows in one
// launch, H and the conv windows held in registers across the rows. Per row
// the exact arm (trait_decode_batched_conv_gdn_exact.rs, FP32 four-kernel arm
// minus the BA gates, which it computes for every row beforehand) runs
// causal_conv1d_update_l2norm_f32 (or its fused-verify twin), a copy of the
// conv windows to the row's rollback slot, gated_delta_rule_decode_f32,
// gated_rms_norm_f32_input and a copy of H to the row's slot (rows 0..k-2).
// Each row here is the same IEEE operations in the same order as
// `qwen35_gdn_decode_fused`'s step (q35f_* above, H stored and reloaded in
// FP32 is the same float), the rollback slots get the bytes the copies give,
// and the final state is the last row's: the exact arm's bytes, from 2 + 3
// launches and 2 state copies a row down to one launch.
//
// TREES. `parent[t]` names the row whose state row t continues (-1: the state
// before the verify); NULL is a chain (`parent[t] == t - 1`). Rows are in
// depth-first pre-order, so a parent precedes its children. Where a row's
// parent is not the row before it, the block reloads H from the parent's
// rollback slot (its own threads wrote it) and rebuilds each channel's conv
// window from the inputs along the row's ancestor path, the pre-verify
// window filling what the path does not reach: the floats the window would
// hold had the parent been the last row (a window is a shift register of
// the channel's inputs). Every row then runs the chain's step from its
// parent's state, so row t's output and rollback slots are what serial
// decode writes for the path ending at t. Rows that are parents must have
// their slots written, so a tree's last row (a leaf in pre-order) is the only
// row without one, as in the chain.
//
// Grid: (num_v_heads, 1, 1) with clusters of Q35F_REPEAT; block Q35F_D. Host
// checks as `qwen35_gdn_decode_fused`, plus 16-byte alignment of the slots.

// Channel `c`'s conv window after row `p` (-1: the pre-verify window `old`):
// the last Q35F_DCONV of old's entries followed by the inputs along the path
// that ends at p.
__device__ __forceinline__ void q35f_path_window(float (&win)[Q35F_DCONV], const int p,
                                                 const int* __restrict__ parent,
                                                 const __nv_bfloat16* __restrict__ qkvz,
                                                 const unsigned int qkvz_stride,
                                                 const unsigned int c,
                                                 const float* __restrict__ old_state) {
    float old[Q35F_DCONV];
    q35f_load_window(old_state + (unsigned long long)c * Q35F_DCONV, old);
    int a = p;
    unsigned int m = 0;
    #pragma unroll
    for (int i = Q35F_DCONV - 1; i >= 0; i--) {
        if (a >= 0) {
            win[i] = (float)qkvz[(unsigned long long)a * qkvz_stride + c];
            a = parent ? parent[a] : a - 1;
            m++;
        } else {
            win[i] = old[i + m];
        }
    }
}

// Whether row `t` of a deferred verify must keep its state in its rollback
// slot: a later row other than `t + 1` continues it (a branch reloads it).
// A chain (NULL parents) keeps none.
__device__ __forceinline__ bool q35f_branch_parent(const int* __restrict__ parent,
                                                   const unsigned int t, const unsigned int k) {
    if (parent == nullptr) return false;
    for (unsigned int c = t + 2; c < k; c++) {
        if (parent[c] == (int)t) return true;
    }
    return false;
}

// The verify rows' body. DEFER = the deferred twin (`qwen35_gdn_verify_tree_defer`).
template <bool DEFER>
__device__ __forceinline__ void q35f_verify_rows(
    float* __restrict__ h_state,
    float* __restrict__ conv_state,
    float* __restrict__ h_snap,
    float* __restrict__ conv_snap,
    const int* __restrict__ parent,
    const __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ conv_w,
    const float* __restrict__ gates,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ stage_qkv,
    float* __restrict__ stage_gb,
    const unsigned int k,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const unsigned int qkvz_stride,
    const unsigned long long h_snap_stride,
    const unsigned long long conv_snap_stride,
    const float l2_eps,
    const float eps
) {
    namespace cg = cooperative_groups;
    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int kh = vh / (num_v_heads / num_k_heads);
    const unsigned int key_dim = num_k_heads * Q35F_D;
    const unsigned long long conv_dim = 2ull * key_dim + num_v_heads * Q35F_D;
    const unsigned long long head = (unsigned long long)vh * Q35F_D * Q35F_D;
    const bool lead = cg::this_cluster().block_rank() == 0;

    float H_reg[Q35F_D];
    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        H_reg[j] = h_state[head + j * Q35F_D + tid];
    }
    const unsigned int cq = kh * Q35F_D + tid;
    const unsigned int ck = key_dim + kh * Q35F_D + tid;
    const unsigned int cv = 2u * key_dim + vh * Q35F_D + tid;
    float wq[Q35F_DCONV], wk[Q35F_DCONV], wv[Q35F_DCONV];
    q35f_load_window(conv_state + (unsigned long long)cq * Q35F_DCONV, wq);
    q35f_load_window(conv_state + (unsigned long long)ck * Q35F_DCONV, wk);
    q35f_load_window(conv_state + (unsigned long long)cv * Q35F_DCONV, wv);

    __shared__ float smem_k[Q35F_D];
    __shared__ float smem_q[Q35F_D];
    __shared__ float ws[4];
    for (unsigned int t = 0; t < k; t++) {
        const int p = parent ? parent[t] : (int)t - 1;
        if (t > 0 && p != (int)t - 1) {
            // A branch: continue the parent's state, not the previous row's.
            const float* hs = p < 0 ? h_state : h_snap + (unsigned long long)p * h_snap_stride;
            #pragma unroll
            for (int j = 0; j < Q35F_D; j++) {
                H_reg[j] = hs[head + j * Q35F_D + tid];
            }
            q35f_path_window(wq, p, parent, qkvz, qkvz_stride, cq, conv_state);
            q35f_path_window(wk, p, parent, qkvz, qkvz_stride, ck, conv_state);
            q35f_path_window(wv, p, parent, qkvz, qkvz_stride, cv, conv_state);
        }
        const __nv_bfloat16* qkv = qkvz + (unsigned long long)t * qkvz_stride;
        if (DEFER) {
            // What the commit replays: the row's raw conv inputs (the key
            // head's q/k channels by the cluster's lead, every block its v)
            // and its gate and beta.
            __nv_bfloat16* st = stage_qkv + (unsigned long long)t * conv_dim;
            if (lead) {
                st[cq] = qkv[cq];
                st[ck] = qkv[ck];
            }
            st[cv] = qkv[cv];
            if (tid == 0) {
                const float* gr = gates + (unsigned long long)t * 2u * num_v_heads;
                float* gb = stage_gb + (unsigned long long)t * 2u * num_v_heads;
                gb[vh] = gr[vh];
                gb[num_v_heads + vh] = gr[num_v_heads + vh];
            }
        }
        wq[0] = wq[1]; wq[1] = wq[2]; wq[2] = wq[3]; wq[3] = (float)qkv[cq];
        wk[0] = wk[1]; wk[1] = wk[2]; wk[2] = wk[3]; wk[3] = (float)qkv[ck];
        wv[0] = wv[1]; wv[1] = wv[2]; wv[2] = wv[3]; wv[3] = (float)qkv[cv];
        const float q_silu = q35f_conv_silu(wq, conv_w + (unsigned long long)cq * Q35F_DCONV);
        const float k_silu = q35f_conv_silu(wk, conv_w + (unsigned long long)ck * Q35F_DCONV);
        const float v_i = q35f_conv_silu(wv, conv_w + (unsigned long long)cv * Q35F_DCONV);
        smem_q[tid] = q35f_l2(q_silu, ws, l2_eps);
        smem_k[tid] = q35f_l2(k_silu, ws, l2_eps);
        __syncthreads();  // every smem_q / smem_k entry before the dots read them

        // The deferred twin keeps only a branch parent's H (no conv slots: a
        // branch rebuilds its windows from the inputs).
        const bool snap = DEFER ? q35f_branch_parent(parent, t, k) : t + 1 < k;
        if (snap && !DEFER) {
            float* cs = conv_snap + (unsigned long long)t * conv_snap_stride;
            if (lead) {
                q35f_store_window(cs + (unsigned long long)cq * Q35F_DCONV, wq);
                q35f_store_window(cs + (unsigned long long)ck * Q35F_DCONV, wk);
            }
            q35f_store_window(cs + (unsigned long long)cv * Q35F_DCONV, wv);
        }
        const float* g = gates + (unsigned long long)t * 2u * num_v_heads;
        const float x = q35f_recur(H_reg, smem_k, smem_q, g[vh], g[num_v_heads + vh], v_i, head_dim);
        if (snap) {
            float* hs = h_snap + (unsigned long long)t * h_snap_stride;
            #pragma unroll
            for (int j = 0; j < Q35F_D; j++) {
                hs[head + j * Q35F_D + tid] = H_reg[j];
            }
        }
        q35f_gated_norm(x, qkv + conv_dim + vh * Q35F_D, norm_w,
                        out + (unsigned long long)t * num_v_heads * Q35F_D + vh * Q35F_D,
                        head_dim, eps);
    }

    // Deferred: H0 and the windows stay for the commit's replay.
    if (DEFER) return;
    // Every block of the key head has read the pre-verify q/k windows (the
    // row loop and any path rebuild): only now may the lead overwrite them.
    cg::this_cluster().sync();
    if (lead) {
        q35f_store_window(conv_state + (unsigned long long)cq * Q35F_DCONV, wq);
        q35f_store_window(conv_state + (unsigned long long)ck * Q35F_DCONV, wk);
    }
    q35f_store_window(conv_state + (unsigned long long)cv * Q35F_DCONV, wv);
    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        h_state[head + j * Q35F_D + tid] = H_reg[j];
    }
}

extern "C" __global__ void __cluster_dims__(Q35F_REPEAT, 1, 1) __launch_bounds__(Q35F_D, 1)
qwen35_gdn_verify_tree(
    float* __restrict__ h_state,               // [nv, 128, 128] FP32: in before row 0, out after the last row
    float* __restrict__ conv_state,            // [conv_dim, 4] FP32: idem
    float* __restrict__ h_snap,                // row t's H at h_snap + t * h_snap_stride (t < k - 1)
    float* __restrict__ conv_snap,             // row t's windows at conv_snap + t * conv_snap_stride (t < k - 1)
    const int* __restrict__ parent,            // [k] parent row, -1 = before the verify; NULL = a chain
    const __nv_bfloat16* __restrict__ qkvz,    // [k, qkvz_stride]: Q | K | V | Z
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    const float* __restrict__ gates,           // [k, 2 * nv]: gate | beta
    const __nv_bfloat16* __restrict__ norm_w,  // [128]
    __nv_bfloat16* __restrict__ out,           // [k, nv * 128]
    const unsigned int k,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,               // 128 (host checks); a RUNTIME value
    const unsigned int qkvz_stride,            // BF16 elements between rows
    const unsigned long long h_snap_stride,    // FP32 elements between slots
    const unsigned long long conv_snap_stride, // FP32 elements between slots
    const float l2_eps,
    const float eps
) {
    q35f_verify_rows<false>(h_state, conv_state, h_snap, conv_snap, parent, qkvz, conv_w, gates,
                            norm_w, out, nullptr, nullptr, k, num_k_heads, num_v_heads, head_dim,
                            qkvz_stride, h_snap_stride, conv_snap_stride, l2_eps, eps);
}

// ── qwen35_gdn_verify_tree_defer + qwen35_gdn_commit_rows
// (ATLAS_QWEN38_VERIFY_DEFER=1) ──
//
// The storing verify above writes, per row and GDN layer, the 3 MiB H of
// every row but the last (and its conv windows) plus the final state, so the
// commit is a copy of the accepted row's slot: ~0.6 ms a row over 48 layers
// of writes nothing reads unless that row is the one accepted.
//
// Deferred, the verify writes no state: H0 and the windows stay as they
// were, and each row stages its raw Q | K | V conv inputs and its gate and
// beta (~20 KiB a row a layer) in the sequence's deferred-commit staging. A
// tree's branch parents still keep their H (a branch reloads it), nothing
// else. After the verdict `qwen35_gdn_commit_rows` replays the accepted rows
// (a chain's prefix, or a tree's root path) from H0 over all the sequence's
// GDN layers in one launch, through the same conv, L2 norm and recurrence
// calls on the same inputs, so the state it stores is, byte for byte, the
// storing kernel's slot of the last accepted row (its final state when that
// row is the last). The gated norm, which the verify already produced and
// which does not feed the state, is skipped.
extern "C" __global__ void __cluster_dims__(Q35F_REPEAT, 1, 1) __launch_bounds__(Q35F_D, 1)
qwen35_gdn_verify_tree_defer(
    float* __restrict__ h_state,               // [nv, 128, 128] FP32: H0, read only
    float* __restrict__ conv_state,            // [conv_dim, 4] FP32: read only
    float* __restrict__ h_snap,                // a branch parent t's H at h_snap + t * h_snap_stride
    const int* __restrict__ parent,            // [k] parent row; NULL = a chain
    const __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ conv_w,
    const float* __restrict__ gates,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ stage_qkv,     // [k, conv_dim] BF16: row t's raw Q | K | V channels
    float* __restrict__ stage_gb,              // [k, 2 * nv] FP32: row t's gate | beta
    const unsigned int k,
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const unsigned int qkvz_stride,
    const unsigned long long h_snap_stride,
    const float l2_eps,
    const float eps
) {
    q35f_verify_rows<true>(h_state, conv_state, h_snap, nullptr, parent, qkvz, conv_w, gates,
                           norm_w, out, stage_qkv, stage_gb, k, num_k_heads, num_v_heads,
                           head_dim, qkvz_stride, h_snap_stride, 0, l2_eps, eps);
}

#define Q35F_COMMIT_LAYERS 48

struct Q35fCommitLayer {
    float* h;                         // [nv, 128, 128] FP32: H0 in, committed out
    float* conv;                      // [conv_dim, 4] FP32: idem
    const __nv_bfloat16* stage_qkv;   // [k, conv_dim] BF16
    const float* stage_gb;            // [k, 2 * nv] FP32
    const __nv_bfloat16* conv_w;      // [conv_dim, 4]
};

struct Q35fCommitLayers {
    Q35fCommitLayer layer[Q35F_COMMIT_LAYERS];
};

// Grid: (num_v_heads, layers, 1) with clusters of Q35F_REPEAT; block Q35F_D.
// `rows` packs the accepted rows 4 bits each, first row in the low nibble.
extern "C" __global__ void __cluster_dims__(Q35F_REPEAT, 1, 1) __launch_bounds__(Q35F_D, 1)
qwen35_gdn_commit_rows(
    const __grid_constant__ Q35fCommitLayers layers,
    const unsigned long long rows,
    const unsigned int n,                      // accepted rows, 1..16
    const unsigned int num_k_heads,
    const unsigned int num_v_heads,
    const unsigned int head_dim,
    const float l2_eps
) {
    namespace cg = cooperative_groups;
    const Q35fCommitLayer& L = layers.layer[blockIdx.y];
    const unsigned int vh = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int kh = vh / (num_v_heads / num_k_heads);
    const unsigned int key_dim = num_k_heads * Q35F_D;
    const unsigned long long conv_dim = 2ull * key_dim + num_v_heads * Q35F_D;
    const unsigned long long head = (unsigned long long)vh * Q35F_D * Q35F_D;
    const bool lead = cg::this_cluster().block_rank() == 0;

    float H_reg[Q35F_D];
    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        H_reg[j] = L.h[head + j * Q35F_D + tid];
    }
    const unsigned int cq = kh * Q35F_D + tid;
    const unsigned int ck = key_dim + kh * Q35F_D + tid;
    const unsigned int cv = 2u * key_dim + vh * Q35F_D + tid;
    float wq[Q35F_DCONV], wk[Q35F_DCONV], wv[Q35F_DCONV];
    q35f_load_window(L.conv + (unsigned long long)cq * Q35F_DCONV, wq);
    q35f_load_window(L.conv + (unsigned long long)ck * Q35F_DCONV, wk);
    q35f_load_window(L.conv + (unsigned long long)cv * Q35F_DCONV, wv);

    __shared__ float smem_k[Q35F_D];
    __shared__ float smem_q[Q35F_D];
    __shared__ float ws[4];
    for (unsigned int i = 0; i < n; i++) {
        const unsigned int r = (unsigned int)((rows >> (4 * i)) & 0xFull);
        const __nv_bfloat16* qkv = L.stage_qkv + (unsigned long long)r * conv_dim;
        wq[0] = wq[1]; wq[1] = wq[2]; wq[2] = wq[3]; wq[3] = (float)qkv[cq];
        wk[0] = wk[1]; wk[1] = wk[2]; wk[2] = wk[3]; wk[3] = (float)qkv[ck];
        wv[0] = wv[1]; wv[1] = wv[2]; wv[2] = wv[3]; wv[3] = (float)qkv[cv];
        const float q_silu = q35f_conv_silu(wq, L.conv_w + (unsigned long long)cq * Q35F_DCONV);
        const float k_silu = q35f_conv_silu(wk, L.conv_w + (unsigned long long)ck * Q35F_DCONV);
        const float v_i = q35f_conv_silu(wv, L.conv_w + (unsigned long long)cv * Q35F_DCONV);
        smem_q[tid] = q35f_l2(q_silu, ws, l2_eps);
        smem_k[tid] = q35f_l2(k_silu, ws, l2_eps);
        __syncthreads();
        const float* g = L.stage_gb + (unsigned long long)r * 2u * num_v_heads;
        (void)q35f_recur(H_reg, smem_k, smem_q, g[vh], g[num_v_heads + vh], v_i, head_dim);
        __syncthreads();  // the dots read smem_q / smem_k before the next row writes them
    }
    // Every block of the key head has read the old q/k windows.
    cg::this_cluster().sync();
    if (lead) {
        q35f_store_window(L.conv + (unsigned long long)cq * Q35F_DCONV, wq);
        q35f_store_window(L.conv + (unsigned long long)ck * Q35F_DCONV, wk);
    }
    q35f_store_window(L.conv + (unsigned long long)cv * Q35F_DCONV, wv);
    #pragma unroll
    for (int j = 0; j < Q35F_D; j++) {
        L.h[head + j * Q35F_D + tid] = H_reg[j];
    }
}
