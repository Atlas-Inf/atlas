// SPDX-License-Identifier: AGPL-3.0-only

// Ling-3.2-air NEXTN (MTP) draft-path kernels for GB10.
//
// Ling ships exactly one NEXTN layer, `model.layers.42.*`
// (`num_nextn_predict_layers = 1`, sliced off the main stack at
// modeling_bailing_moe_v3_vl.py:1911-1912). `BailingMoeV3MTPLayer`
// (modeling_bailing_moe_v3_vl.py:1442) drafts one position with:
//
//   input_embeds = enorm(input_embeds)                 // :1470
//   hidden_states = hnorm(hidden_states)               // :1471
//   hidden_states = eh_proj(cat([input_embeds,          // :1472
//                                hidden_states], -1))
//   ... one gated-MLA + MoE layer body ...
//   hidden_states = final_layernorm(hidden_states)     // :1498
//   logits = lm_head(hidden_states)                    // shared head
//
// The two entry points below cover the two glue spots of that chain that
// generic kernels do not: (1) the fused enorm/hnorm input pair, and (2) the
// on-device pick of each draft position's candidate token out of the draft
// logits. The eh_proj GEMV, the layer-42 body, and the shared lm_head are
// served by the other kernels of this target (dense_gemv_bf16, the gated-MLA
// and MoE decode kernels, argmax); nothing here overlaps them.
//
// DESIGN CONTRACT FOR VERIFY. The NEXTN layer itself is MLA-style (its
// tensor set is layer 41's: attention.{q_proj, g_proj, kv_a_proj_with_mqa,
// kv_a_layernorm, kv_b_proj, dense} — no A_log/dt_bias/conv1d) and carries
// its own 512-expert NVFP4 MoE; that body's kernels are other agents' files
// in this target. Ling's target stack is 35 KDA (SSM-state) layers + 7
// gated-MLA layers, so draft verification touches recurrent state as well
// as attention KV — transitively, through the rollback/commit semantics of
// the 35 earlier layers, whose device side is the KDA verify-snapshot /
// commit machinery (`kda_recurrent_bf16_verify_snap`, `kda_commit_records`,
// kda-kernels PR #110) with the Rust-side slot buffers of
// `ssm_verify_attach` / `kda_records` (`(mtp_slots + 1) x num_intermediates`
// fold-record rows per slot). This file does NOT duplicate any of that;
// verify-state kernels belong to the kda side.
// Nothing about that rollback needs to know how tokens were chosen — verify
// consumes one candidate token per row. These kernels therefore write every
// pick straight into caller-owned device words, strided so a pick can land
// directly in the `[seq][row]` candidate-token matrix the batched verify
// feed reads. A draft chain then runs launch-to-launch with no logits D2H
// sync and no host-side token round trip, so the draft-and-select tail is
// CUDA-graph capturable.
//
// DRAFT DEPTH IS NOT THIS FILE'S BUSINESS. The pick kernel proposes one
// candidate per draft pass over whatever rows of logits it is handed; depth
// policy (num_drafts, the ladder, adaptive stop) lives in MODEL.toml and the
// proposer. One draft is the measured optimum on the qwen3.8-flash-next
// target with the same 512-expert MoE geometry (its MODEL.toml: the marginal
// verify-row cost is ~0.4 of a decode step through the experts, break-even
// tokens/step > 1 + 0.4*(K-1)); Ling's verify rows additionally walk KDA
// state, so the marginal row is at least as expensive. Baselines to beat:
// accept 2.31 (vLLM NVFP4 + MTP), 2.15-2.65 (SGLang TP2 FP8). Do not
// re-invent a deeper default here; measure on a Spark first.

#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Entry 1: fused [enorm(embed[token]) | hnorm(hidden)] pair.
//
// Port of GLM-5's `glm_mtp_eh_norm` (kernels/gb10/glm-5.3-flash/nvfp4/
// glm_mtp_eh_norm.cu). One two-CTA launch replaces two `rms_norm_vanilla`
// launches and reads the embedding table directly; the output is the
// contiguous [2, hidden] pair `eh_proj` consumes as its [2*hidden] row (the
// modeling reference concatenates the two normed rows, :1472). Each CTA
// preserves rms_norm_vanilla's thread mapping, reduction tree, FP32
// operation order, and BF16 conversion, for byte parity with the unfused
// two-launch path.
//
// DELTA vs the GLM original: the draft token is read from a device word,
// not a kernel argument. GLM's serial chain reads the previous pass's pick
// back to the host and relaunches per token; reading the word from device
// memory lets a captured graph replay with the previous pass's pick
// (`ling_mtp_draft_pick`'s output word) or any host-staged word the caller
// writes before the launch. A word holding 0xFFFFFFFF (the pick kernel's
// never-taken fallback value) reads row 0, so a stale word cannot address
// outside the table.
//
// Grid: (2, 1, 1). Block: (min(hidden_size, 1024), 1, 1).
// hidden_size must be even (Ling: 2560; 1280 u32 words per row).
extern "C" __global__ void ling_mtp_eh_norm(
    const __nv_bfloat16* __restrict__ embed_table,
    const unsigned int* __restrict__ token,
    const __nv_bfloat16* __restrict__ target_hidden,
    const __nv_bfloat16* __restrict__ enorm,
    const __nv_bfloat16* __restrict__ hnorm,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tok = *token;
    const __nv_bfloat16* x = row == 0
        ? embed_table + (size_t)(tok == 0xFFFFFFFFu ? 0u : tok) * hidden_size
        : target_hidden;
    const __nv_bfloat16* weight = row == 0 ? enorm : hnorm;
    __nv_bfloat16* out = output + (size_t)row * hidden_size;

    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int packed = x32[i];
        float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
        float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    float val = sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        float wv = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            wv += __shfl_xor_sync(0xFFFFFFFF, wv, offset);
        }
        if (lane_id == 0) {
            warp_sums[0] = wv;
        }
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        unsigned int px = x32[i];
        unsigned int pw = w32[i];
        float xv0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(px & 0xFFFF)));
        float xv1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(px >> 16)));
        float wv0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pw & 0xFFFF)));
        float wv1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(pw >> 16)));
        unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(xv0 * rms * wv0));
        unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(xv1 * rms * wv1));
        out32[i] = lo | (hi << 16);
    }
    if ((hidden_size & 1) && tid == 0) {
        const float xv = __bfloat162float(x[hidden_size - 1]);
        const float wv = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(xv * rms * wv);
    }
}

// The dflash2 candidate order, from kernels/gb10/common/
// dflash2_candidate_selector.cu: value descending, then vocab index
// ascending — the lower index wins ties, matching the engine's
// first-index-wins argmax contract. A NaN never enters (both comparisons
// are false for NaN); the (-INFINITY, 0xFFFFFFFF) sentinel is the empty
// slot, and a real -inf logit beats it through the index rule.
__device__ __forceinline__ bool ling_mtp_before(
    float va, unsigned int ia, float vb, unsigned int ib
) {
    return va > vb || (va == vb && ia < ib);
}

// Update the online-softmax pair (max, sum) with value `v`, exactly the
// per-element update of `argmax_bf16_batch_lp` (kernels/gb10/common/
// argmax_bf16.cu).
__device__ __forceinline__ void ling_mtp_online_softmax(float& max, float& sum, float v) {
    if (v > max) {
        sum = sum * __expf(max - v) + 1.0f;
        max = v;
    } else {
        sum += __expf(v - max);
    }
}

// Consider (v, c) as a pick candidate, applying this row's end-token ban.
// unused `end*` slots carry 0xFFFFFFFF, which no vocab column ever equals.
__device__ __forceinline__ void ling_mtp_consider(
    float& pick_v, unsigned int& pick_i, float v, unsigned int c,
    bool ban_on, unsigned int end0, unsigned int end1, unsigned int end2, unsigned int end3
) {
    if (ban_on && (c == end0 || c == end1 || c == end2 || c == end3)) return;
    if (ling_mtp_before(v, c, pick_v, pick_i)) {
        pick_v = v;
        pick_i = c;
    }
}

// ---------------------------------------------------------------------------
// Entry 2: per-row draft pick + full-row log-probability, fully on device.
//
// The dflash2 selector discipline (kernels/gb10/common/
// dflash2_candidate_selector.cu) applied to a single-NEXTN draft head: the
// greedy chain reads its logits and writes its per-row candidate token on
// device, so a draft chain needs no logits D2H sync and no host loop. Ling
// has no separate DFlash drafter — no pred/succ codebooks, no selector
// rank — so this is the strict subset of the shared selector's walk that a
// NEXTN head needs: for each logits row, the exact first-index-wins argmax
// and the log softmax probability of that pick, over the whole vocabulary,
// in ONE pass.
//
// What it adds over `argmax_bf16_batch_lp` (the common kernel a NEXTN
// drafter otherwise reuses):
//   * end-token ban per row: below a request's min_tokens floor the verifier
//     may not end the turn, so an end token drafted there would only
//     truncate an otherwise acceptable chain (the dflash2 selector's ban
//     rule; here simplified to a per-row device flag, since one NEXTN layer
//     drafts one position per pass). The ban affects the PICK only; the
//     reported log-probability is the pick's true full-vocabulary log
//     softmax, so cumulative-confidence stop decisions (GLM_DRAFT_TRUNC/
//     D-Cut style: the prefix sum of per-position top-1 log-probs) stay
//     calibrated even when the top-1 was banned.
//   * device-word ban flags and strided outputs, so the launch is
//     CUDA-graph safe and a pick can be written directly into the
//     `[seq][row]` candidate-token matrix the KDA/MLA verify feed consumes:
//     pass `&matrix[slot]` with the matrix row pitch as `out_stride`.
//   * the aligned-body 16-byte scan of the dflash2 selector (fewer issued
//     loads per element), with its element-wise fallback for rows whose
//     stride leaves the row misaligned.
//
// An empty, all-NaN, or fully banned row picks column 0 with log-probability
// 0.0f — `argmax_bf16_batch_lp`'s convention — so a malformed row stays
// deterministic instead of addressing the embedding table wildly.
//
// Grid: (rows, 1, 1), one block per logits row. Block: (1024, 1, 1) — the
// shared arrays size the block here; the tree reduction visits only
// blockDim.x, so smaller power-of-two blocks remain correct.
// `ban` may be NULL (no row bans anything); `out_logprob` may be NULL
// (picks only). `row_stride` and `vocab_size` are in logits elements.
extern "C" __global__ void ling_mtp_draft_pick(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int row_stride,
    unsigned int* __restrict__ out_tokens,
    unsigned int out_stride,
    float* __restrict__ out_logprob,
    const unsigned int* __restrict__ ban,
    unsigned int vocab_size,
    unsigned int end0,
    unsigned int end1,
    unsigned int end2,
    unsigned int end3
) {
    __shared__ float s_max[1024];
    __shared__ float s_sum[1024];
    __shared__ float s_pick_v[1024];
    __shared__ unsigned int s_pick_i[1024];

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* row_logits =
        logits + (unsigned long long)row * (unsigned long long)row_stride;
    const bool ban_on = ban != nullptr && ban[row] != 0u;

    // Aligned-body bounds, the `df2_sel_scan` split with lo = 0: a row whose
    // base is 16-byte aligned scans [0, vocab & ~7) as uint4 words with the
    // odd tail element-wise; a misaligned row scans its whole length
    // element-wise (the body loop is empty; the tail starts past the end).
    const bool aligned16 =
        (reinterpret_cast<unsigned long long>(row_logits) & 15) == 0;
    const unsigned int body_end = aligned16 ? (vocab_size & ~7u) : 0u;
    const unsigned int head_end = aligned16 ? 0u : vocab_size;
    const unsigned int tail_start = aligned16 ? body_end : vocab_size;

    float max = -1e30f;
    float sum = 0.0f;
    float pick_v = -INFINITY;
    unsigned int pick_i = 0xFFFFFFFFu;

    const unsigned short* bits = reinterpret_cast<const unsigned short*>(row_logits);
    for (unsigned int c = tid; c < head_end; c += blockDim.x) {
        float v = __uint_as_float((unsigned int)bits[c] << 16);
        ling_mtp_online_softmax(max, sum, v);
        ling_mtp_consider(pick_v, pick_i, v, c, ban_on, end0, end1, end2, end3);
    }
    const uint4* body = reinterpret_cast<const uint4*>(row_logits);
    for (unsigned int q = tid; q < body_end / 8; q += blockDim.x) {
        const uint4 w = body[q];
        const unsigned int words[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const float v0 = __uint_as_float(words[k] << 16);
            const unsigned int c0 = q * 8 + 2 * k;
            ling_mtp_online_softmax(max, sum, v0);
            ling_mtp_consider(pick_v, pick_i, v0, c0, ban_on, end0, end1, end2, end3);
            const float v1 = __uint_as_float(words[k] & 0xFFFF0000u);
            const unsigned int c1 = c0 + 1;
            ling_mtp_online_softmax(max, sum, v1);
            ling_mtp_consider(pick_v, pick_i, v1, c1, ban_on, end0, end1, end2, end3);
        }
    }
    for (unsigned int c = tail_start + tid; c < vocab_size; c += blockDim.x) {
        float v = __uint_as_float((unsigned int)bits[c] << 16);
        ling_mtp_online_softmax(max, sum, v);
        ling_mtp_consider(pick_v, pick_i, v, c, ban_on, end0, end1, end2, end3);
    }

    s_max[tid] = max;
    s_sum[tid] = sum;
    s_pick_v[tid] = pick_v;
    s_pick_i[tid] = pick_i;
    __syncthreads();

    // Tree reduction over (max, sum) — the byte-identical merge of
    // `argmax_bf16_batch_lp` — plus the pick merge in the dflash2 total
    // order. Threads that scanned nothing carry max = -1e30 / sum = 0 (a
    // no-op online-softmax merge) and the pick sentinel (a no-op in
    // `ling_mtp_before`'s index rule).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float mine = s_max[tid];
            const float other = s_max[tid + s];
            if (other > mine) {
                s_sum[tid] = s_sum[tid] * __expf(mine - other) + s_sum[tid + s];
                s_max[tid] = other;
            } else {
                s_sum[tid] = s_sum[tid] + s_sum[tid + s] * __expf(other - mine);
            }
            if (ling_mtp_before(s_pick_v[tid + s], s_pick_i[tid + s],
                                s_pick_v[tid], s_pick_i[tid])) {
                s_pick_v[tid] = s_pick_v[tid + s];
                s_pick_i[tid] = s_pick_i[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_tokens[(unsigned long long)row * out_stride] =
            s_pick_i[0] == 0xFFFFFFFFu ? 0u : s_pick_i[0];
        if (out_logprob) {
            const float total = s_sum[0];
            // log p(pick) = pick - logsumexp(row); total >= 1 whenever any
            // element contributed (the max contributes exp(0) = 1), so
            // total <= 0 marks a degenerate row: log p = 0.0f.
            out_logprob[(unsigned long long)row * out_stride] =
                (total > 0.0f && s_pick_v[0] > -INFINITY)
                    ? s_pick_v[0] - (s_max[0] + __logf(total))
                    : 0.0f;
        }
    }
}
