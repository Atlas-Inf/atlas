// SPDX-License-Identifier: AGPL-3.0-only

// Ling-3.2-air (bailing_hybrid) KDA primitives for GB10, ported from the
// GLM-5.3-flash KDA kernels (kernels/gb10/glm-5.3-flash/nvfp4/kda.cu).
//
// Ground truth encoded (reference: context/modeling_bailing_moe_v3_vl.py,
// class BailingMoeV3KimiDeltaAttention at :1237, forward :1301; tensor
// names per context/ling-nvfp4-index.json):
//
//   num_attention_heads 32, head_dim 128 with head_k_dim == head_dim, so the
//   recurrent state is [32, 128, 128] FP32 per layer.
//   Projections (all BF16, computed by the host GEMM path, not here):
//     q/k/v_proj  2560 -> 4096, then causal short conv, kernel 4, SiLU
//                 (ShortConvolution activation='silu'; the checkpoint has
//                 no conv bias).
//     b_proj      2560 -> 32   (raw beta; sigmoid here).
//     f_proj      2560 -> 4096 (memory gate, one head-dim lane per element).
//     g_proj      2560 -> 4096 (output gate for o_norm).
//     o_proj      4096 -> 2560.
//   A_log [32] FP32, dt_bias [4096] FP32 (per (head, dim)).
//   no_kda_lora = true: f_proj and g_proj are full-rank; there is no fused
//   side projection to unpack, so the GLM kda_split_side_proj kernel has no
//   Ling counterpart.
//   KDA layer count is 35: text layers 0..41 except the 7 gated-MLA layers
//   5, 11, 17, 23, 29, 35, 41. The deepseek-style compressed KV projections
//   (kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj) live on those gated-MLA
//   layers and the MTP layer 42, NOT on KDA layers — per the index and the
//   modeling file, which wins over the campaign brief's phrasing.
//   Recurrence semantics (fla chunk_kda / fused_recurrent_kda calls at
//   modeling :1378/:1395): use_qk_l2norm_in_kernel=True, use_gate_in_kernel=
//   True, safe_gate=True, lower_bound=-5.0, i.e.
//     decay = expf(lower_bound / (1 + expf(-exp(A_log[h]) * (f + dt_bias)))
//   per (head, dim) lane, q normalized by 1/||q|| / sqrt(k_dim), k by 1/||k||,
//   beta = sigmoid(b_proj) per head onto the correction.
//   o_norm = FusedRMSNormGated(128, eps=1e-6, activation='sigmoid') applied
//   per (token, head) row, gated by g_proj (kda_sigmoid_gated_rms_norm).
//
// The recurrent kernels intentionally follow the reference token order; the
// regresident pair is the prefill path. All kernels take heads / dim at
// launch time; the supported Ling geometry is heads=32, dim=128.
//
// The GLM kernels renamed for Ling: glm_kda_recurrent_indexed is ported as
// ling_kda_recurrent_indexed, glm_kda_conv_indexed as ling_kda_conv_indexed
// (the pool-addressing decode pair of the multi-sequence lane; the conv
// twin wraps the common causal_conv1d_update arithmetic). Everything else
// keeps its entry name so the existing runtime KDA layer wrappers bind
// with no code motion.

#include "../../common/atlas_pdl.cuh"
#include <cuda_bf16.h>
#include "kda_recurrent_body.cuh"

extern "C" __global__ void kda_pack_qkv(
    const __nv_bfloat16* __restrict__ planes,
    __nv_bfloat16* __restrict__ packed,
    unsigned int tokens,
    unsigned int dim
) {
    atlas_pdl_enter();
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long count = (unsigned long long)tokens * 3 * dim;
    if (i >= count) return;
    unsigned int c = i % (3 * dim);
    unsigned int t = i / (3 * dim);
    unsigned int plane = c / dim;
    unsigned int d = c % dim;
    packed[i] = planes[((unsigned long long)plane * tokens + t) * dim + d];
}

extern "C" __global__ void kda_recurrent_bf16(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    atlas_kda_recurrent_body(qkv, raw_gate, raw_beta, a_log, dt_bias, state, output, tokens, heads, dim,
        lower_bound);
}

// Independent one-token rows select FP32 pools by explicit device slot IDs.
// Host validates geometry, row strides, capacities and unique live slot IDs.
// Invalid slots leave all state and output untouched. Strides are FP32 elements.
extern "C" __global__ void ling_kda_recurrent_indexed(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state_pool,
    __nv_bfloat16* __restrict__ output,
    unsigned int rows,
    unsigned int heads,
    unsigned int dim,
    float lower_bound,
    const int* __restrict__ state_slots,
    unsigned int slot_count,
    unsigned long long state_stride
) {
    const unsigned int row = blockIdx.y;
    // Host validates the supported 32x128 geometry. Do not prove dim==128
    // here: nvcc can fold rsqrtf(dim) differently from the scalar runtime ABI.
    if (row >= rows) return;
    const int slot = state_slots[row];
    if (slot < 0 || (unsigned int)slot >= slot_count || state_stride < (unsigned long long)heads * dim * dim) return;
    float* state = state_pool + (unsigned long long)slot * state_stride;
    qkv += (unsigned long long)row * 3 * heads * dim;
    raw_gate += (unsigned long long)row * heads * dim;
    raw_beta += (unsigned long long)row * heads;
    output += (unsigned long long)row * heads * dim;
    atlas_kda_recurrent_body(qkv, raw_gate, raw_beta, a_log, dt_bias, state, output, 1, heads, dim,
        lower_bound);
}

// Independent one-token conv rows select the FP32 conv pool by explicit
// device slot IDs — the pool-addressing twin of the common
// causal_conv1d_update (the GLM target's glm_kda_conv_indexed ported for
// the Ling geometry: 3*32*128 = 12288 channels, d_conv 4, SiLU, BF16
// output). Same per-element arithmetic as a one-token causal_conv1d_update
// so the bytes of the indexed decode lane match the per-sequence decode.
// Invalid slots leave all state and output untouched. Strides: rows in
// BF16 elements, pool in FP32 elements.
extern "C" __global__ void ling_kda_conv_indexed(
    float* __restrict__ conv_state_pool,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int rows,
    unsigned int input_stride,
    unsigned int output_stride,
    const int* __restrict__ state_slots,
    unsigned int slot_count,
    unsigned long long state_stride
) {
    const unsigned int row = blockIdx.y;
    if (row >= rows || dim != 12288 || d_conv != 4 || blockDim.x != 256) return;
    const int slot = state_slots[row];
    if (slot < 0 || (unsigned int)slot >= slot_count
        || state_stride < (unsigned long long)dim * d_conv) return;
    float* conv_state = conv_state_pool + (unsigned long long)slot * state_stride;
    input += (unsigned long long)row * input_stride;
    output += (unsigned long long)row * output_stride;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;
    float* state = conv_state + (unsigned long long)ch * d_conv;
    for (unsigned int i = 0; i < d_conv - 1; ++i) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)input[ch];
    const __nv_bfloat16* w = weight + (unsigned long long)ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; ++k) {
        acc += state[k] * (float)w[k];
    }
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;
    output[ch] = __float2bfloat16(silu);
}

// Ling-3.2-air NEXTN verify twin of kda_recurrent_bf16. The recurrence and
// FP32 FMA order are unchanged; after rows 0..tokens-2 it also writes the
// updated H to a contiguous rollback slab. One launch covers a draft window
// (with one NEXTN draft the launch is typically tokens=2) and replaces a
// recurrent launch plus full-state D2D snapshot per draft token.
extern "C" __global__ void kda_recurrent_bf16_verify_snap(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ state_inter,
    unsigned long long inter_stride,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int vrow = threadIdx.x;
    if (head >= heads || dim > 128 || blockDim.x < dim) return;

    __shared__ float qv[128];
    __shared__ float kv[128];
    __shared__ float gate_exp[128];
    __shared__ float red_q[128];
    __shared__ float red_k[128];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float beta;

    float* H = state + (unsigned long long)head * dim * dim;
    const float a = expf(a_log[head]);
    const float scale = rsqrtf((float)dim);

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
        if (vrow < dim) {
            float q = (float)qkv[qbase + (unsigned long long)head * dim + vrow];
            float k = (float)qkv[qbase + (unsigned long long)heads * dim
                              + (unsigned long long)head * dim + vrow];
            qv[vrow] = q;
            kv[vrow] = k;
            red_q[vrow] = q * q;
            red_k[vrow] = k * k;
            float g = (float)raw_gate[((unsigned long long)t * heads + head) * dim + vrow];
            float log_decay = lower_bound /
                (1.0f + expf(-a * (g + dt_bias[(unsigned long long)head * dim + vrow])));
            gate_exp[vrow] = expf(log_decay);
        }
        __syncthreads();

        for (unsigned int stride = 64; stride > 0; stride >>= 1) {
            if (vrow < stride) {
                red_q[vrow] += red_q[vrow + stride];
                red_k[vrow] += red_k[vrow + stride];
            }
            __syncthreads();
        }
        if (vrow == 0) {
            inv_q = rsqrtf(red_q[0] + 1.0e-6f) * scale;
            inv_k = rsqrtf(red_k[0] + 1.0e-6f);
            float b = (float)raw_beta[(unsigned long long)t * heads + head];
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();

        if (vrow < dim) {
            float dot_k = 0.0f;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * gate_exp[k + 0];
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * gate_exp[k + 1];
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * gate_exp[k + 2];
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * gate_exp[k + 3];
                dot_k += h0 * (kv[k + 0] * inv_k) + h1 * (kv[k + 1] * inv_k)
                       + h2 * (kv[k + 2] * inv_k) + h3 * (kv[k + 3] * inv_k);
            }
            const unsigned long long vbase = qbase + (unsigned long long)2 * heads * dim;
            float delta = ((float)qkv[vbase + (unsigned long long)head * dim + vrow]
                           - dot_k) * beta;
            float out = 0.0f;
            float* snapshot = (t + 1u < tokens)
                ? state_inter + (unsigned long long)t * inter_stride
                    + (unsigned long long)head * dim * dim
                : nullptr;
            #pragma unroll 4
            for (unsigned int k = 0; k < dim; k += 4) {
                float h0 = H[(unsigned long long)(k + 0) * dim + vrow] * gate_exp[k + 0]
                         + delta * (kv[k + 0] * inv_k);
                float h1 = H[(unsigned long long)(k + 1) * dim + vrow] * gate_exp[k + 1]
                         + delta * (kv[k + 1] * inv_k);
                float h2 = H[(unsigned long long)(k + 2) * dim + vrow] * gate_exp[k + 2]
                         + delta * (kv[k + 2] * inv_k);
                float h3 = H[(unsigned long long)(k + 3) * dim + vrow] * gate_exp[k + 3]
                         + delta * (kv[k + 3] * inv_k);
                H[(unsigned long long)(k + 0) * dim + vrow] = h0;
                H[(unsigned long long)(k + 1) * dim + vrow] = h1;
                H[(unsigned long long)(k + 2) * dim + vrow] = h2;
                H[(unsigned long long)(k + 3) * dim + vrow] = h3;
                if (snapshot != nullptr) {
                    snapshot[(unsigned long long)(k + 0) * dim + vrow] = h0;
                    snapshot[(unsigned long long)(k + 1) * dim + vrow] = h1;
                    snapshot[(unsigned long long)(k + 2) * dim + vrow] = h2;
                    snapshot[(unsigned long long)(k + 3) * dim + vrow] = h3;
                }
                out += h0 * (qv[k + 0] * inv_q) + h1 * (qv[k + 1] * inv_q)
                     + h2 * (qv[k + 2] * inv_q) + h3 * (qv[k + 3] * inv_q);
            }
            output[((unsigned long long)t * heads + head) * dim + vrow] =
                __float2bfloat16(out);
        }
        __syncthreads();
    }
}

// Owner-batched, register-resident kda_recurrent_bf16_verify_snap: grid
// (heads, owners), owner o advancing its own state (and rollback slab) over
// its `tokens` rows at row o * tokens. Each thread holds its state column in
// registers, so H is loaded and stored once per launch instead of read twice
// and written once per row. The per-row expressions are verify_snap's:
// states and snapshots match it bit for bit, while a few BF16 outputs can
// round differently (FMA contraction of the output dot).
// Up to four owners; dim == 128, block 128.
// Records-mode fold of one state element: decay, then the rank-1 delta-rule
// update, in one explicit rounding order shared by the verify and the commit.
#define KDA_RECORD_FLOATS 384u
__device__ __forceinline__ float kda_fold(float h, float decay, float key, float delta) {
    return __fmaf_rn(delta, key, __fmul_rn(h, decay));
}

template <bool RECORDS>
__device__ __forceinline__ void kda_verify_owners_impl(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    float* state0, float* state1, float* state2, float* state3,
    float* const* inters,
    unsigned long long inter_stride,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int owner = blockIdx.y;
    const unsigned int vrow = threadIdx.x;
    if (head >= heads || owner >= 4 || dim != 128 || blockDim.x != 128) return;
    float* const states[4] = {state0, state1, state2, state3};
    const unsigned long long row0 = (unsigned long long)owner * tokens;
    qkv += row0 * 3 * heads * 128;
    raw_gate += row0 * heads * 128;
    raw_beta += row0 * heads;
    output += row0 * heads * 128;
    float* state_inter = inters[owner];

    __shared__ float qv[128];
    __shared__ float kv[128];
    __shared__ float gate_exp[128];
    __shared__ float red_q[128];
    __shared__ float red_k[128];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float beta;

    float* H = states[owner] + (unsigned long long)head * 128 * 128;
    float h[128];
    #pragma unroll
    for (unsigned int k = 0; k < 128; ++k) h[k] = H[(unsigned long long)k * 128 + vrow];
    const float a = expf(a_log[head]);
    const float scale = rsqrtf((float)dim);

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * 128;
        {
            float q = (float)qkv[qbase + (unsigned long long)head * 128 + vrow];
            float k = (float)qkv[qbase + (unsigned long long)heads * 128
                              + (unsigned long long)head * 128 + vrow];
            qv[vrow] = q;
            kv[vrow] = k;
            red_q[vrow] = q * q;
            red_k[vrow] = k * k;
            float g = (float)raw_gate[((unsigned long long)t * heads + head) * 128 + vrow];
            float log_decay = lower_bound /
                (1.0f + expf(-a * (g + dt_bias[(unsigned long long)head * 128 + vrow])));
            gate_exp[vrow] = expf(log_decay);
        }
        __syncthreads();

        for (unsigned int stride = 64; stride > 0; stride >>= 1) {
            if (vrow < stride) {
                red_q[vrow] += red_q[vrow + stride];
                red_k[vrow] += red_k[vrow + stride];
            }
            __syncthreads();
        }
        if (vrow == 0) {
            inv_q = rsqrtf(red_q[0] + 1.0e-6f) * scale;
            inv_k = rsqrtf(red_k[0] + 1.0e-6f);
            float b = (float)raw_beta[(unsigned long long)t * heads + head];
            beta = 1.0f / (1.0f + expf(-b));
        }
        __syncthreads();

        float dot_k = 0.0f;
        #pragma unroll
        for (unsigned int k = 0; k < 128; k += 4) {
            float h0 = h[k + 0] * gate_exp[k + 0];
            float h1 = h[k + 1] * gate_exp[k + 1];
            float h2 = h[k + 2] * gate_exp[k + 2];
            float h3 = h[k + 3] * gate_exp[k + 3];
            dot_k += h0 * (kv[k + 0] * inv_k) + h1 * (kv[k + 1] * inv_k)
                   + h2 * (kv[k + 2] * inv_k) + h3 * (kv[k + 3] * inv_k);
        }
        const unsigned long long vbase = qbase + (unsigned long long)2 * heads * 128;
        float delta = ((float)qkv[vbase + (unsigned long long)head * 128 + vrow]
                       - dot_k) * beta;
        float out = 0.0f;
        float* snapshot = (!RECORDS && t + 1u < tokens)
            ? state_inter + (unsigned long long)t * inter_stride
                + (unsigned long long)head * 128 * 128
            : nullptr;
        if (RECORDS) {
            // Row t's fold record: decay, normalized key, correction.
            float* rec = state_inter + (unsigned long long)t * inter_stride
                + (unsigned long long)head * KDA_RECORD_FLOATS;
            rec[vrow] = gate_exp[vrow];
            rec[128 + vrow] = kv[vrow] * inv_k;
            rec[256 + vrow] = delta;
        }
        #pragma unroll
        for (unsigned int k = 0; k < 128; k += 4) {
            float h0, h1, h2, h3;
            if (RECORDS) {
                h0 = kda_fold(h[k + 0], gate_exp[k + 0], kv[k + 0] * inv_k, delta);
                h1 = kda_fold(h[k + 1], gate_exp[k + 1], kv[k + 1] * inv_k, delta);
                h2 = kda_fold(h[k + 2], gate_exp[k + 2], kv[k + 2] * inv_k, delta);
                h3 = kda_fold(h[k + 3], gate_exp[k + 3], kv[k + 3] * inv_k, delta);
            } else {
                h0 = h[k + 0] * gate_exp[k + 0] + delta * (kv[k + 0] * inv_k);
                h1 = h[k + 1] * gate_exp[k + 1] + delta * (kv[k + 1] * inv_k);
                h2 = h[k + 2] * gate_exp[k + 2] + delta * (kv[k + 2] * inv_k);
                h3 = h[k + 3] * gate_exp[k + 3] + delta * (kv[k + 3] * inv_k);
            }
            h[k + 0] = h0;
            h[k + 1] = h1;
            h[k + 2] = h2;
            h[k + 3] = h3;
            if (snapshot != nullptr) {
                snapshot[(unsigned long long)(k + 0) * 128 + vrow] = h0;
                snapshot[(unsigned long long)(k + 1) * 128 + vrow] = h1;
                snapshot[(unsigned long long)(k + 2) * 128 + vrow] = h2;
                snapshot[(unsigned long long)(k + 3) * 128 + vrow] = h3;
            }
            out += h0 * (qv[k + 0] * inv_q) + h1 * (qv[k + 1] * inv_q)
                 + h2 * (qv[k + 2] * inv_q) + h3 * (qv[k + 3] * inv_q);
        }
        output[((unsigned long long)t * heads + head) * 128 + vrow] = __float2bfloat16(out);
        __syncthreads();
    }
    if (!RECORDS) {   // records: the live state waits for kda_commit_records
        #pragma unroll
        for (unsigned int k = 0; k < 128; ++k) H[(unsigned long long)k * 128 + vrow] = h[k];
    }
}


#define KDA_VERIFY_OWNERS_ARGS \
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ raw_gate, \
    const __nv_bfloat16* __restrict__ raw_beta, const float* __restrict__ a_log, \
    const float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, \
    float* state0, float* state1, float* state2, float* state3, \
    float* inter0, float* inter1, float* inter2, float* inter3, \
    unsigned long long inter_stride, unsigned int tokens, unsigned int heads, \
    unsigned int dim, float lower_bound

extern "C" __global__ void __launch_bounds__(128) kda_recurrent_bf16_verify_snap_owners(
    KDA_VERIFY_OWNERS_ARGS
) {
    float* const inters[4] = {inter0, inter1, inter2, inter3};
    kda_verify_owners_impl<false>(qkv, raw_gate, raw_beta, a_log, dt_bias, output,
        state0, state1, state2, state3, inters, inter_stride, tokens, heads, dim, lower_bound);
}

// Records variant: `inter*` are each owner's record rows (`inter_stride`
// floats apart, KDA_RECORD_FLOATS per head) and the states are only read.
// Verify-then-commit records are vLLM's RecoverSSM (vllm-project/vllm#51855,
// from ReplaySSM #48018; Apache-2.0) on Atlas's snapshot-verify template; see
// docs/glm-prior-art.md for the credited lineage of this scheme.
extern "C" __global__ void __launch_bounds__(128) kda_recurrent_bf16_verify_rec_owners(
    KDA_VERIFY_OWNERS_ARGS
) {
    atlas_pdl_enter();
    float* const inters[4] = {inter0, inter1, inter2, inter3};
    kda_verify_owners_impl<true>(qkv, raw_gate, raw_beta, a_log, dt_bias, output,
        state0, state1, state2, state3, inters, inter_stride, tokens, heads, dim, lower_bound);
}

// Advance each head's live state over the first `rows` records written by
// kda_recurrent_bf16_verify_rec_owners (the accepted prefix of a verify):
// the same kda_fold per element, so the result is bit-identical to the
// state the verify reached at row `rows - 1`.
//
// The fold is elementwise — h[k][v] takes decay[k], key[k] and delta[v] of
// each row — so any element-to-thread split is exact. One launch commits
// every KDA layer of a sequence: blockIdx.y is the layer, its state and
// records `state_layer_stride` / `records_layer_stride` floats past layer 0
// (the pools' uniform layer stride). Each CTA owns 128 / KDA_COMMIT_KSPLIT
// key rows of one head; thread = (key lane, 4 value columns), float4 I/O.
// Grid: heads * KDA_COMMIT_KSPLIT x layers; block 128.
#define KDA_COMMIT_KSPLIT 4u
#define KDA_COMMIT_KPT (128u / KDA_COMMIT_KSPLIT / 4u)
// One CTA's share: key-row block `bx` (head bx / KSPLIT) of layer `layer`.
__device__ __forceinline__ void kda_commit_records_item(
    float* __restrict__ state,
    unsigned long long state_layer_stride,
    const float* __restrict__ records,
    unsigned long long records_layer_stride,
    unsigned long long record_stride,
    unsigned int rows,
    unsigned int heads,
    unsigned int bx,
    unsigned int layer
) {
    const unsigned int head = bx / KDA_COMMIT_KSPLIT;
    if (head >= heads || blockDim.x != 128 || rows == 0) return;
    const unsigned int col = (threadIdx.x & 31u) * 4u;
    const unsigned int k0 = (bx % KDA_COMMIT_KSPLIT) * (128u / KDA_COMMIT_KSPLIT)
        + (threadIdx.x >> 5);
    float* H = state + (unsigned long long)layer * state_layer_stride
        + (unsigned long long)head * 128 * 128 + col;
    const float* R = records + (unsigned long long)layer * records_layer_stride
        + (unsigned long long)head * KDA_RECORD_FLOATS;
    float4 h[KDA_COMMIT_KPT];
    #pragma unroll
    for (unsigned int i = 0; i < KDA_COMMIT_KPT; ++i)
        h[i] = *reinterpret_cast<const float4*>(H + (unsigned long long)(k0 + 4u * i) * 128);
    for (unsigned int t = 0; t < rows; ++t) {
        const float* rec = R + (unsigned long long)t * record_stride;
        const float4 d = *reinterpret_cast<const float4*>(rec + 256 + col);
        #pragma unroll
        for (unsigned int i = 0; i < KDA_COMMIT_KPT; ++i) {
            const float decay = rec[k0 + 4u * i];
            const float key = rec[128 + k0 + 4u * i];
            h[i].x = kda_fold(h[i].x, decay, key, d.x);
            h[i].y = kda_fold(h[i].y, decay, key, d.y);
            h[i].z = kda_fold(h[i].z, decay, key, d.z);
            h[i].w = kda_fold(h[i].w, decay, key, d.w);
        }
    }
    #pragma unroll
    for (unsigned int i = 0; i < KDA_COMMIT_KPT; ++i)
        *reinterpret_cast<float4*>(H + (unsigned long long)(k0 + 4u * i) * 128) = h[i];
}

extern "C" __global__ void __launch_bounds__(128) kda_commit_records(
    float* __restrict__ state,
    unsigned long long state_layer_stride,
    const float* __restrict__ records,
    unsigned long long records_layer_stride,
    unsigned long long record_stride,
    unsigned int rows,
    unsigned int heads
) {
    atlas_pdl_enter();
    kda_commit_records_item(state, state_layer_stride, records, records_layer_stride,
                            record_stride, rows, heads, blockIdx.x, blockIdx.y);
}

// kda_commit_records over a fixed grid (ATLAS_KDA_COMMIT_CTAS): CTA c takes
// the grid's blocks c, c + gridDim.x, ... of `heads * KSPLIT * layers`, each
// the same elements with the same folds, so the same state words. The grid
// kernel queues every block of every layer at once (4480 at Ling's 35 KDA
// layers), so a kernel or copy enqueued behind it on another stream waits for
// nearly all of them; a grid of a few CTAs a SM leaves room beside the commit.
// Grid: (ctas, 1, 1); block 128.
extern "C" __global__ void __launch_bounds__(128) kda_commit_records_stride(
    float* __restrict__ state,
    unsigned long long state_layer_stride,
    const float* __restrict__ records,
    unsigned long long records_layer_stride,
    unsigned long long record_stride,
    unsigned int rows,
    unsigned int heads,
    unsigned int layers
) {
    atlas_pdl_enter();
    const unsigned int per_layer = heads * KDA_COMMIT_KSPLIT;
    for (unsigned int item = blockIdx.x; item < per_layer * layers; item += gridDim.x)
        kda_commit_records_item(state, state_layer_stride, records, records_layer_stride,
                                record_stride, rows, heads, item % per_layer, item / per_layer);
}

// Precompute normalized Q/K and row decay once per token/head. The recurrence
// fans each token out across 32 column groups, so doing the reductions and
// transcendental operations there would repeat the expensive work 32 times.
// The caller supplies FP32 planes borrowed from otherwise-idle buffers; no
// persistent allocation is added.
extern "C" __global__ void kda_preprocess_regresident(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ raw_beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ q_norm,
    float* __restrict__ k_norm,
    float* __restrict__ decay,
    float* __restrict__ beta,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim,
    float lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int row = threadIdx.x;
    if (head >= heads || t >= tokens || row >= dim || dim != 128) return;

    __shared__ float red_q[128];
    __shared__ float red_k[128];
    __shared__ float inv_q;
    __shared__ float inv_k;
    __shared__ float a;
    const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
    const unsigned long long off = ((unsigned long long)t * heads + head) * dim + row;
    const float q = (float)qkv[qbase + (unsigned long long)head * dim + row];
    const float k = (float)qkv[qbase + (unsigned long long)heads * dim
                               + (unsigned long long)head * dim + row];
    red_q[row] = q * q;
    red_k[row] = k * k;
    __syncthreads();

    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (row < stride) {
            red_q[row] += red_q[row + stride];
            red_k[row] += red_k[row + stride];
        }
        __syncthreads();
    }
    if (row == 0) {
        inv_q = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)dim);
        inv_k = rsqrtf(red_k[0] + 1.0e-6f);
        a = expf(a_log[head]);
        const float b = (float)raw_beta[(unsigned long long)t * heads + head];
        beta[(unsigned long long)t * heads + head] = 1.0f / (1.0f + expf(-b));
    }
    __syncthreads();

    q_norm[off] = q * inv_q;
    k_norm[off] = k * inv_k;
    const float g = (float)raw_gate[off];
    const float bias = dt_bias[(unsigned long long)head * dim + row];
    decay[off] = expf(lower_bound / (1.0f + expf(-a * (g + bias))));
}

// Register-resident prefill recurrence for Ling-3.2-air KDA.
//
// The slow-path kernel assigns one thread to each state column, leaving the
// 128x128 FP32 state in global memory throughout the token loop. That
// streams every state matrix twice per prompt token. Here one warp owns one
// column: its lanes retain four state rows each in registers for the whole
// prompt. State columns are independent under KDA's row-wise decay, so 32
// CTAs per head expose enough parallelism while loading and storing H
// exactly once.
//
// Grid:  (heads, dim / 4, 1)
// Block: (128, 1, 1), four warps and therefore four columns per CTA.
extern "C" __global__ void __launch_bounds__(128, 4)
kda_recurrent_bf16_regresident(
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    unsigned int tokens,
    unsigned int heads,
    unsigned int dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int col = blockIdx.y * 4 + warp;
    if (head >= heads || col >= dim || dim != 128) return;

    const unsigned int r0 = lane;
    const unsigned int r1 = lane + 32;
    const unsigned int r2 = lane + 64;
    const unsigned int r3 = lane + 96;
    float* H = state + (unsigned long long)head * dim * dim;

    float s0 = H[(unsigned long long)r0 * dim + col];
    float s1 = H[(unsigned long long)r1 * dim + col];
    float s2 = H[(unsigned long long)r2 * dim + col];
    float s3 = H[(unsigned long long)r3 * dim + col];

    for (unsigned int t = 0; t < tokens; ++t) {
        const unsigned long long qbase = (unsigned long long)t * 3 * heads * dim;
        const unsigned long long off = ((unsigned long long)t * heads + head) * dim;
        const float q0 = q_norm[off + r0];
        const float q1 = q_norm[off + r1];
        const float q2 = q_norm[off + r2];
        const float q3 = q_norm[off + r3];
        const float k0 = k_norm[off + r0];
        const float k1 = k_norm[off + r1];
        const float k2 = k_norm[off + r2];
        const float k3 = k_norm[off + r3];
        const float d0 = decay[off + r0];
        const float d1 = decay[off + r1];
        const float d2 = decay[off + r2];
        const float d3 = decay[off + r3];

        float dot_k = s0 * d0 * k0 + s1 * d1 * k1
                    + s2 * d2 * k2 + s3 * d3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot_k += __shfl_xor_sync(0xFFFFFFFFu, dot_k, offset);

        const unsigned long long voff = qbase + (unsigned long long)2 * heads * dim
                                         + (unsigned long long)head * dim + col;
        const float delta = ((float)qkv[voff] - dot_k)
                          * beta[(unsigned long long)t * heads + head];

        s0 = s0 * d0 + delta * k0;
        s1 = s1 * d1 + delta * k1;
        s2 = s2 * d2 + delta * k2;
        s3 = s3 * d3 + delta * k3;

        float dot_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot_q += __shfl_xor_sync(0xFFFFFFFFu, dot_q, offset);
        if (lane == 0) {
            output[((unsigned long long)t * heads + head) * dim + col] =
                __float2bfloat16(dot_q);
        }
    }

    H[(unsigned long long)r0 * dim + col] = s0;
    H[(unsigned long long)r1 * dim + col] = s1;
    H[(unsigned long long)r2 * dim + col] = s2;
    H[(unsigned long long)r3 * dim + col] = s3;
}

// Ling o_norm: FusedRMSNormGated over one (token, head) row of head_dim=128,
// gated by g_proj with a sigmoid, weight o_norm [128], eps = rms_norm_eps
// (1e-6). Grid: one block per row.
extern "C" __global__ void kda_sigmoid_gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int heads,
    unsigned int dim,
    float eps
) {
    atlas_pdl_enter();
    const unsigned int row_id = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (dim > 128 || d >= dim) return;
    __shared__ float red[128];
    __shared__ float inv;
    const unsigned long long off = (unsigned long long)row_id * dim + d;
    float x = (float)input[off];
    red[d] = x * x;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (d < stride) red[d] += red[d + stride];
        __syncthreads();
    }
    if (d == 0) inv = rsqrtf(red[0] / (float)dim + eps);
    __syncthreads();
    float g = (float)gate[off];
    float y = x * inv * (float)weight[d] / (1.0f + expf(-g));
    output[off] = __float2bfloat16(y);
}
