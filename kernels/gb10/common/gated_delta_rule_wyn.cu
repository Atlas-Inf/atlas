// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K∈{5..16} verification (wyN).
//
// K-templated generalization of gated_delta_rule_wy17.cu (which itself
// generalizes wy4). One __device__ impl, instantiated for the chain-verify
// widths between the dedicated wy4 and the DFlash wy17 — mirroring the
// w4a16_gemv_batchm_impl<MAX_M> instantiation pattern in w4a16_gemv.cu.
// Removes the serial per-token GDN fallback at chain-verify K=5..16.
//
// Algorithm (identical WY-chunkwise structure — "2 passes over H
// regardless of K"):
//   1. Load q[K], k[K] into SMEM (K KB at k_dim=128).
//   2. Compute K*(K-1)/2 inter-token k-dot products via block reduction.
//   3. PASS 1: read H once, compute hk[K] = pre-update H·k[t] dots.
//   4. WY correction (sequential over K tokens): produce vn[K].
//   5. PASS 2: apply K state updates in single fused loop, writing
//      Hi_t = state after token t for t=0..K-2, and final H = state
//      after token K-1.
//
// SMEM budget @ K=16, k_dim=128 (the largest instantiation):
//   sk[16][128] + sq[16][128] = 16·128·2·4 = 16 KB
//   kdots[120] + gate/beta[32] + warp_sums[4]  < 1 KB
//   (SM_120 cap: 100 KB — trivially fits for every instantiation)
// Register arrays vi/hk/vn/qd are [K_TOKENS] per thread — 16 floats each at
// the max — and kd_flat indexing is t*(t-1)/2 + s, both K-generic.
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// Reduction primitives (gdn_reduce.cuh) match the per-token baseline
// bit-exactly; the gate clamp MUST match per-token gated_delta_rule_decode
// (see gated_delta_rule_wy.cu — drift here flips argmax on long verifies).

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#define BLOCK_SIZE 128

// `h_state_inter_base` is a contiguous pool of (K-1) intermediate H states
// for this (layer, slot). Stride between intermediates is
// `inter_stride_floats` floats. Slot t's intermediate lives at
// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
// `h_state` itself becomes the final (post token K-1) state — same
// pool-layout contract as gated_delta_rule_wy17.
//
// `state_is_table` (trailing arg, wy4 idiom): 0 = contiguous bases indexed by
// (b*num_v_heads+vh)*hv — byte-identical to the original; 1 = `h_state` and
// `h_state_inter_base` are device POINTER TABLES of `batch_size` entries, one
// per sequence — per block b the sequence's base comes from table[b], the
// per-vh offset and the inter_stride_floats distance between Hi_t are
// unchanged. The table form sidesteps the wrong contiguous intermediate
// stride at batch_size>1 (see gated_delta_rule_wy4.cu's comment) AND drops
// the "active sequences occupy contiguous pool slots" assumption.

// MODE: 0 = verify+store (Hi_0..Hi_{K-2} + final H, the legacy contract);
//       1 = verify-defer (same outputs, NO state stores — h_state stays H0
//           so `gated_delta_rule_commit` can re-apply the accepted prefix);
//       2 = commit (runtime token count `accepted_count`, single seq;
//           replays tokens 0..a-1 from live H0 and stores ONLY H_{a-1}).
// `n_tok` is the runtime token bound: K_TOKENS (compile-time constant) for
// MODE 0/1 — identical codegen — and `accepted_count` clamped to K_TOKENS for
// MODE 2, whose loops keep the SAME per-iteration op order.
template <int K_TOKENS, int MODE>
__device__ __forceinline__ void gated_delta_rule_wyn_impl(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int accepted_count,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    // MODE 2 is a single-sequence commit: grid.y == 1 and `accepted` is a
    // scalar (the accepted token count, clamped to the register-cap K_TOKENS).
    const unsigned int n_tok =
        (MODE == 2) ? min(accepted_count, (unsigned int)K_TOKENS) : K_TOKENS;
    if (n_tok == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    // wy4 table idiom: state_is_table=0 keeps the contiguous (b*nv+vh)*hv
    // addressing byte-identically; =1 reads each sequence's base from a
    // device pointer table, then applies the same per-vh head offset.
    const unsigned long long head_off = (unsigned long long)vh * hv;
    const unsigned long long flat_off = (unsigned long long)(b * num_v_heads + vh) * hv;
    float* H = state_is_table ? ((float* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    // Per-(b, vh) offset into the intermediate pool. Each Hi_t base ptr =
    // h_state_inter_base + t * inter_stride_floats + ((b*nv+vh)*hv)
    // (contiguous), or inter_table[b] + vh*hv + t * inter_stride_floats
    // (table — intermediates keep their intra-slot stride).
    float* Hi_base =
        h_state_inter_base
            ? (state_is_table ? ((float* const*)h_state_inter_base)[b] + head_off
                              : h_state_inter_base + flat_off)
            : h_state; // MODE 1/2 pass no pool — never dereferenced

    // MODE 2 (commit) is single-sequence: direct bases, row `t` indexes
    // the per-slot staging directly. Same addresses for MODE != 2 by
    // construction (verify packs tokens contiguously per sequence).
    const __nv_bfloat16* const q_seq = query;
    const __nv_bfloat16* const k_seq = key;
    const __nv_bfloat16* const v_seq = value;
    const float* const g_seq = gate;
    const float* const bt_seq = beta;
    #define WYN_ROW(t) ((MODE == 2) ? (t) : (b * K_TOKENS + (t)))

    // ── Load q, k, gate, beta into SMEM ──
    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];   // gate clamped
    __shared__ float sbt[K_TOKENS];  // beta
    __shared__ float smem_warp[4];

    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < n_tok; t++) {
            const __nv_bfloat16* q_t = q_seq + WYN_ROW(t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = k_seq + WYN_ROW(t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < n_tok) {
        // Gate clamp matches per-token gated_delta_rule_decode (see wy4 comment).
        float g_raw = g_seq[WYN_ROW(tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = bt_seq[WYN_ROW(tid) * gb_stride + vh];
    }
    __syncthreads();

    // ── Compute K*(K-1)/2 k-dot products via block reduction ──
    // kd[t][s] = k_t · k_s for s < t, stored sparsely at tri_idx(t,s) =
    // t*(t-1)/2 + s.
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    #pragma unroll
    for (int t = 1; t < (int)n_tok; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = atlas_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    if (tid < v_dim) {
        // Load v[K] for this thread's v_dim slot.
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < n_tok; t++) {
            const __nv_bfloat16* v_t = v_seq + WYN_ROW(t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        // ── PASS 1: Read H once, compute K dot products hk[t] = H · k_t ──
        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            #pragma unroll
            for (int t = 0; t < n_tok; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // ── WY Correction (sequential over K tokens) ──
        // hk_corrected[t] = product(g[0..t-1]) * hk_raw[t]
        //                 + sum_{s<t} (product(g[s+1..t-1])) * kd[t][s] * vn[s]
        // vn[t]           = (v[t] - g[t] * hk_corrected[t]) * beta[t]
        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < (int)n_tok; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // ── PASS 2: Apply K state updates in fused loop ──
        // After update t: H_new[t] = g[t] * H_prev + k[t] * vn[t].
        // Write Hi_t for t=0..K-2; final H = H_new[K-1].
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            #pragma unroll
            for (int t = 0; t < n_tok; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (MODE == 0 ? t < K_TOKENS - 1 : t < (int)n_tok - 1) {
                    // MODE 0: per-token rollback snapshot; MODE 1/2:
                    // nothing — defer keeps h_state at H0, commit stores
                    // only the accepted prefix's final state.
                    if (MODE == 0) {
                        float* Hi_t = Hi_base + t * inter_stride_floats;
                        Hi_t[(j + 0) * v_dim + tid] = h0;
                        Hi_t[(j + 1) * v_dim + tid] = h1;
                        Hi_t[(j + 2) * v_dim + tid] = h2;
                        Hi_t[(j + 3) * v_dim + tid] = h3;
                    }
                } else if (MODE != 1) {
                    H[(j + 0) * v_dim + tid] = h0;
                    H[(j + 1) * v_dim + tid] = h1;
                    H[(j + 2) * v_dim + tid] = h2;
                    H[(j + 3) * v_dim + tid] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        // ── Write outputs (K rows × v_dim) — commit recomputes the same
        // values but writes only the state, so `output` is null there.
        if (MODE != 2) {
            float s = rsqrtf((float)k_dim);
            #pragma unroll
            for (int t = 0; t < n_tok; t++) {
                output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                    __float2bfloat16(qd[t] * s);
            }
        }
    }
    #undef WYN_ROW
}

// Instantiations for chain-verify K=5..8. The argument list is identical to
// gated_delta_rule_wy17; the Rust side selects the handle by num_tokens.
#define ATLAS_WYN_INSTANTIATE(K)                                              \
    extern "C" __global__ void gated_delta_rule_wy##K(                        \
        float* __restrict__ h_state,                                          \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ h_state_inter_base,                               \
        unsigned int inter_stride_floats,                                     \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride,                                               \
        unsigned int state_is_table                                           \
    ) {                                                                       \
        gated_delta_rule_wyn_impl<K>(                                         \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_floats, batch_size,              \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, state_is_table);                                       \
    }

ATLAS_WYN_INSTANTIATE(5)
ATLAS_WYN_INSTANTIATE(6)
ATLAS_WYN_INSTANTIATE(7)
ATLAS_WYN_INSTANTIATE(8)
ATLAS_WYN_INSTANTIATE(9)
ATLAS_WYN_INSTANTIATE(10)
ATLAS_WYN_INSTANTIATE(11)
ATLAS_WYN_INSTANTIATE(12)
ATLAS_WYN_INSTANTIATE(13)
ATLAS_WYN_INSTANTIATE(14)
ATLAS_WYN_INSTANTIATE(15)
ATLAS_WYN_INSTANTIATE(16)

#undef ATLAS_WYN_INSTANTIATE

ATLAS_WYN_DEFER_INSTANTIATE(5)
ATLAS_WYN_DEFER_INSTANTIATE(6)
ATLAS_WYN_DEFER_INSTANTIATE(7)
ATLAS_WYN_DEFER_INSTANTIATE(8)
ATLAS_WYN_DEFER_INSTANTIATE(9)
ATLAS_WYN_DEFER_INSTANTIATE(10)
ATLAS_WYN_DEFER_INSTANTIATE(11)
ATLAS_WYN_DEFER_INSTANTIATE(12)
ATLAS_WYN_DEFER_INSTANTIATE(13)
ATLAS_WYN_DEFER_INSTANTIATE(14)
ATLAS_WYN_DEFER_INSTANTIATE(15)
ATLAS_WYN_DEFER_INSTANTIATE(16)

#undef ATLAS_WYN_DEFER_INSTANTIATE

// Deferred commit (ATLAS_GDN_DEFERRED_COMMIT): replays tokens 0..a_b-1 of a
// sequence's deferred-verify inputs against the live H0 in `h_state`,
// storing only H_{a_b} in place — byte-identical to the storing verify's
// `Hi_{a_b-1}` (MODE 0) / final H (a_b == K), because the impl runs the same
// per-iteration arithmetic; only the trip count and the stores differ.
// ALL state AND input pointers are per-sequence pointer tables (verify
// staging is per-slot), and `accepted_count` carries the sequence's accepted
// token count (0 → skip, >K clamps to K).
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1).
extern "C" __global__ void gated_delta_rule_commit(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    unsigned int accepted_count,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    gated_delta_rule_wyn_impl<16, 2>(
        h_state, query, key, value, gate, beta,
        nullptr,           // no outputs
        nullptr, 0,        // no intermediate pool
        accepted_count,
        1 /* batch */, num_k_heads, num_v_heads, k_dim, v_dim,
        qk_stride, v_stride, gb_stride,
        0 /* contiguous */);
}
