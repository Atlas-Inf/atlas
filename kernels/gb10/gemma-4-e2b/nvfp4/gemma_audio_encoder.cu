// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Spark — Gemma-4 E2B Audio Tower CUDA Kernels (module "gemma_audio_encoder")
//
// Wave 4C of the E2B multimodal bring-up: the four gemma-specific entry
// points the Rust `GemmaAudioEncoder` declares via
// `try_kernel(gpu, "gemma_audio_encoder", "<name>")` (crates/spark-model/
// src/layers/gemma_audio_encoder/, built by the parallel 4A task).
//
// MODULE-NAME CONTRACT (must match the Rust): the module is named
// "gemma_audio_encoder" — the FILE STEM, deliberately NOT overridden in
// KERNEL.toml `[modules]`. The serve-load media-encoder fallback
// (crates/spark-server/src/main_modules/serve_load.rs `demote_unsupported_media_towers`)
// keys the audio tower's survival on a module named exactly
// "gemma_audio_encoder" (asserted by serve_load_tests.rs
// `gemma_media_configs_survive_when_target_ships_encoders`); any other name
// silently demotes the tower on the real gemma-4-e2b target. The 4A encoder
// must therefore resolve handles as try_kernel(gpu, "gemma_audio_encoder",
// "gemma_audio_*").
//
// Shared ops are REUSED from this target's existing tree — nothing here
// redefines them: q/k/v/relative_k/FFN/conv-linear GEMMs run on `gemm`
// (dense_gemm_bf16), the many RMSNorms on `norm` (rms_norm), the post-FFN
// SiLU/GLU elementwise ops and output_proj/embed_audio on the shared
// modules. This file only covers the four gemma-specific pieces:
//
//   gemma_audio_softplus     — elementwise softplus (precompute for the
//                              per_dim_scale vector; ~128 elems per layer)
//   gemma_audio_subsample_conv — mel → 2× stride-2 3×3 conv + LayerNorm +
//                              ReLU → flattened [T/4, 1024]
//   gemma_audio_conv1d       — depthwise CAUSAL conv1d, kernel 5 (left pad 4)
//   gemma_audio_chunked_attn — chunked local attention with relative bias,
//                              logit softcap and per-dim softplus scale
//
// Ground truth: transformers main `modeling_gemma4.py` (Gemma4AudioAttention,
// Gemma4AudioSubSampleConvProjection, Gemma4AudioLightConv1d,
// Gemma4AudioModel) + the E2B checkpoint audio_config. Weight shapes verified
// against google/gemma-4-E2B-it model.safetensors on the DGX Spark host:
// subsample convs [128,1,3,3] / [32,128,3,3], LayerNorms [128]/[32],
// input_proj_linear [1024,1024], lconv1d depthwise [1024,1,5],
// per_dim_scale [128], rel_k_proj [1024,1024], FFNs [4096,1024]/[1024,4096],
// output_proj [1536,1024]+bias[1536], 12 layers, hidden 1024, 8 heads × 128.
//
// All ops use BF16 storage; computation uses f32 accumulators. The tower runs
// once per audio prefill (seq ≤ 750, 12 layers), so simplicity wins over peak
// occupancy — same posture as the Wave-3 vision tower.

#include <cuda_bf16.h>

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// ── 1. Softplus: per_dim_scale precompute ───────────────────────────────────
// Elementwise softplus in f32 with BF16 in/out. The audio attention scales q
// by `q_scale * softplus(per_dim_scale[d])` per dim; the Rust runs this once
// per layer on the [head_dim] per_dim_scale parameter and hands the result to
// `gemma_audio_chunked_attn` (its `spd` arg). Matches torch F.softplus
// (beta=1, threshold=20): x > 20 → identity.
//
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void gemma_audio_softplus(
    const __nv_bfloat16* __restrict__ in,  // [n]
    __nv_bfloat16* __restrict__ out,       // [n]
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = bf16_to_f32(in[i]);
    out[i] = f32_to_bf16((x > 20.0f) ? x : logf(1.0f + expf(x)));
}

// ── 2. Subsample conv projection ────────────────────────────────────────────
// Mel features [T, 128] (1 input channel, viewed as a T×128 image) through
// the two stride-2 3×3 Conv2d stages of
// `Gemma4AudioSubSampleConvProjection` (HF modeling_gemma4.py):
//
//   conv1: [1,1,T,128] → [128, T1, 64]   (T1 = (T+1)/2, W 128→64)
//   LayerNorm over the 128 channels + ReLU
//   conv2: [128, T1, 64] → [32, T2, 32]  (T2 = (T1+1)/2, W 64→32)
//   LayerNorm over the 32 channels + ReLU
//   flatten → [T2, 1024] = [T2, 32 freq, 32 ch] row-major
//
// The norms are REAL LayerNorms (mean-subtracting, biased variance, weight
// only — no bias), NOT RMSNorm. The flatten feeds `input_proj_linear`
// ([1024,1024] GEMM, run by the Rust on the shared gemm module).
//
// Optional `mask` [T] u8 (1 = valid): invalid input rows are zeroed at load,
// matching HF's `hidden_states * mask[:, None, :, None]`; the Rust derives
// the post-subsample output mask itself (HF semantics: output_mask[t4] =
// input_mask[4*t4], the [::2]-downsampled mask) and may pass nullptr here.
//
// SPLIT INTO TWO KERNELS (the original fused kernel staged only the block's
// two conv1 rows in smem and read THREE in conv2 — an OOB smem read that
// crashed with ILLEGAL_ADDRESS once the tower actually ran). Each stage now
// reads its halo rows from the global intermediate `out1`, so the conv2
// window (kernel 3, stride 2) spans rows 2*t4-1 .. 2*t4+1 correctly.
//
// Kernel 1 `gemma_audio_subsample_conv1`: mel → conv1 [T1, 64, 128] → LN1 →
// ReLU → `out1`. Grid: (T1, 1, 1)  Block: (1024, 1, 1)
// Args: (input, mask, out1, w1, ln1_w, T, T1, eps)
//
// Kernel 2 `gemma_audio_subsample_conv2`: out1 → conv2 → LN2 → ReLU →
// flattened `[T2, 1024]`. Grid: (T2, 1, 1)  Block: (1024, 1, 1)
// Args: (in1, output, w2, ln2_w, T1, T2, eps)
extern "C" __global__ void gemma_audio_subsample_conv1(
    const __nv_bfloat16* __restrict__ input,  // [T, 128] mel features
    const unsigned char* __restrict__ mask,   // [T] 1=valid, or nullptr
    __nv_bfloat16* __restrict__ out1,         // [T1, 64, 128] (freq-major)
    const __nv_bfloat16* __restrict__ w1,     // [128, 3, 3] conv1 ([128,1,3,3])
    const __nv_bfloat16* __restrict__ ln1_w,  // [128] LayerNorm weight
    unsigned int T, unsigned int T1,
    float eps
) {
    const unsigned int r = blockIdx.x;        // conv1 output row
    const unsigned int tid = threadIdx.x;
    if (r >= T1) return;

    __shared__ __nv_bfloat16 in_rows[3 * 128];  // mel rows 2r-1 .. 2r+1
    __shared__ __nv_bfloat16 c1_out[64 * 128];
    __shared__ float ln1_mv[128];               // [0..64) mean, [64..128) var

    const float inv128 = 1.0f / 128.0f;
    if (tid < 3 * 128) {
        unsigned int rel = tid / 128;
        unsigned int f = tid % 128;
        int g = (int)(2 * r) + (int)rel - 1;
        float v = 0.0f;
        if (g >= 0 && g < (int)T) {
            v = bf16_to_f32(input[(unsigned long long)g * 128 + f]);
            if (mask != nullptr && !mask[g]) v = 0.0f;
        }
        in_rows[tid] = f32_to_bf16(v);
    }
    __syncthreads();

    // Conv1: [64 freq][128 ch] cells, stride 2 over both dims (pad 1).
    for (unsigned int i = tid; i < 64 * 128; i += 1024) {
        unsigned int f = i >> 7;              // i / 128
        unsigned int c1 = i & 127;
        float acc = 0.0f;
        const __nv_bfloat16* w = w1 + c1 * 9;
        for (unsigned int dt = 0; dt < 3; ++dt) {
            for (unsigned int df = 0; df < 3; ++df) {
                int f_in = (int)(2 * f) + (int)df - 1;
                if (f_in < 0 || f_in >= 128) continue;
                float x = bf16_to_f32(in_rows[dt * 128 + f_in]);
                acc += x * bf16_to_f32(w[dt * 3 + df]);
            }
        }
        c1_out[i] = f32_to_bf16(acc);
    }
    __syncthreads();

    // LayerNorm1 stats: mean/var over the 128 channels per freq.
    if (tid < 64) {
        float sum = 0.0f, sumsq = 0.0f;
        for (unsigned int c1 = 0; c1 < 128; ++c1) {
            float v = bf16_to_f32(c1_out[tid * 128 + c1]);
            sum += v;
            sumsq += v * v;
        }
        float mean = sum * inv128;
        float var = sumsq * inv128 - mean * mean;
        ln1_mv[tid] = mean;
        ln1_mv[tid + 64] = (var > 0.0f) ? var : 0.0f;
    }
    __syncthreads();

    // LayerNorm1 apply + ReLU → out1.
    for (unsigned int i = tid; i < 64 * 128; i += 1024) {
        unsigned int f = i >> 7;
        unsigned int c1 = i & 127;
        float v = bf16_to_f32(c1_out[i]);
        float mean = ln1_mv[f];
        float var = ln1_mv[f + 64];
        v = (v - mean) * rsqrtf(var + eps) * bf16_to_f32(ln1_w[c1]);
        out1[(unsigned long long)r * (64 * 128) + i] = f32_to_bf16(fmaxf(v, 0.0f));
    }
}

extern "C" __global__ void gemma_audio_subsample_conv2(
    const __nv_bfloat16* __restrict__ in1,   // [T1, 64, 128] (freq-major)
    __nv_bfloat16* __restrict__ output,      // [T2, 1024] flattened
    const __nv_bfloat16* __restrict__ w2,    // [32, 128, 3, 3] conv2
    const __nv_bfloat16* __restrict__ ln2_w, // [32] LayerNorm weight
    unsigned int T1, unsigned int T2,
    float eps
) {
    const unsigned int t4 = blockIdx.x;       // conv2 output row
    const unsigned int tid = threadIdx.x;
    if (t4 >= T2) return;

    __shared__ float conv2_raw[1024];
    __shared__ float ln2_mv[64];              // [0..32) mean, [32..64) var

    const unsigned int c2 = tid >> 5;         // 0..31 channels
    const unsigned int f2 = tid & 31;         // 0..31 freq
    const float inv32 = 1.0f / 32.0f;

    // Conv2: window over out1 rows 2*t4-1 .. 2*t4+1 (kernel 3, stride 2).
    float acc = 0.0f;
    for (unsigned int dt = 0; dt < 3; ++dt) {
        int r_in = (int)(2 * t4) + (int)dt - 1;
        if (r_in < 0 || r_in >= (int)T1) continue;
        for (unsigned int df = 0; df < 3; ++df) {
            int f_in = (int)(2 * f2) + (int)df - 1;
            if (f_in < 0 || f_in >= 64) continue;
            const __nv_bfloat16* w = w2 + c2 * 1152 + dt * 3 + df;
            const __nv_bfloat16* xp =
                in1 + (unsigned long long)r_in * (64 * 128) + f_in * 128;
            for (unsigned int c1 = 0; c1 < 128; ++c1) {
                acc += bf16_to_f32(xp[c1]) * bf16_to_f32(w[c1 * 9]);
            }
        }
    }
    conv2_raw[tid] = acc;
    __syncthreads();

    // LayerNorm2 stats: mean/var over the 32 channels per freq.
    if (tid < 32) {
        float sum = 0.0f, sumsq = 0.0f;
        for (unsigned int c = 0; c < 32; ++c) {
            float v = conv2_raw[c * 32 + tid];
            sum += v;
            sumsq += v * v;
        }
        float mean = sum * inv32;
        float var = sumsq * inv32 - mean * mean;
        ln2_mv[tid] = mean;
        ln2_mv[tid + 32] = (var > 0.0f) ? var : 0.0f;
    }
    __syncthreads();

    // LayerNorm2 apply + ReLU + flattened write [T2, 32 freq, 32 ch].
    float v = conv2_raw[tid];
    v = (v - ln2_mv[f2]) * rsqrtf(ln2_mv[f2 + 32] + eps) * bf16_to_f32(ln2_w[c2]);
    output[(unsigned long long)t4 * 1024 + f2 * 32 + c2] = f32_to_bf16(fmaxf(v, 0.0f));
}

// ── 3. Light depthwise conv1d (causal, kernel 5) ────────────────────────────
// The `Gemma4AudioLightConv1d` depthwise stage. The HF module is
// `Gemma4AudioCausalConv1d`: nn.functional.pad(x, (kernel-1, 0)) then a
// depthwise Conv1d(kernel 5, groups=hidden, bias=False) — i.e. CAUSAL with a
// left pad of 4, NOT symmetric padding:
//
//   out[t, ch] = Σ_{k=0..4} x[t - 4 + k, ch] * w[ch, k]   (x[i < 0] = 0)
//
// The Rust feeds the GLU'd `linear_start` output ([seq, hidden], bf16); the
// SiLU after conv_norm and the `linear_end` GEMM are shared-ops territory.
//
// Grid: (ceil(hidden/256), 1, 1)  Block: (256, 1, 1)
// Args: (input, weight, output, seq, hidden, ksize, in_stride)
extern "C" __global__ void gemma_audio_conv1d(
    const __nv_bfloat16* __restrict__ input,   // [seq, in_stride] (interleaved GLU output row stride)
    const __nv_bfloat16* __restrict__ weight,  // [hidden, ksize] (from [hidden,1,ksize])
    __nv_bfloat16* __restrict__ output,        // [seq, hidden]
    unsigned int seq, unsigned int hidden, unsigned int ksize, unsigned int in_stride
) {
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= hidden) return;

    const int pad = (int)ksize - 1;
    const __nv_bfloat16* w = weight + ch * ksize;
    for (unsigned int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < ksize; ++k) {
            int idx = (int)t - pad + (int)k;
            if (idx >= 0) {
                acc += bf16_to_f32(input[(unsigned long long)idx * in_stride + ch])
                     * bf16_to_f32(w[k]);
            }
        }
        output[(unsigned long long)t * hidden + ch] = f32_to_bf16(acc);
    }
}

// ── 4. Chunked local attention (Gemma4AudioAttention) ───────────────────────
// Custom blocked attention over 12-token chunks with a 24-wide context window
// (12 past + 12 chunk; ctx_right 0), relative sinusoidal bias, logit softcap
// and a per-dim softplus query scale. NOT flash-attention friendly (masked
// gathers + tiny windows) — one warp per (query-in-chunk), 4 lanes per head
// covering head_dim in 4-stride dims.
//
// Math (HF modeling_gemma4.py Gemma4AudioAttention.forward, all f32):
//   q = q_proj(x) * q_scale * softplus(per_dim_scale[d]), q_scale = hd^-0.5/ln2
//   k = k_proj(x) * k_scale,                              k_scale = ln(1+e)/ln2
//   block b: query qi = b*chunk + q (padded to a multiple of chunk);
//            keys  ki = b*chunk + c - max_past,  c ∈ [0, context)
//            context = chunk + max_past + max_future, max_past = ctx_left - 1
//   score[q, c] = dot(q, k_ki) + bd,  bd = (0 ≤ c-q < rel_len ? dot(q, rel_k[c-q]) : 0)
//            rel_len = context/2 + 1 (13); rel_k = relative_k_proj(pos_emb)
//            [rel_len, heads, head_dim] — HF `_rel_shift` maps raw[j, p] onto
//            context slot c for query j as p = c - j (verified against the HF
//            oracle: shifted[j,c] = raw[j, c-j], 0 when c-j ∉ [0, rel_len)).
//   attn = tanh(score / softcap) * softcap
//   attn = invalid_logits where !valid  (masked AFTER the cap, HF order)
//   valid = host-built BLOCKED mask byte for (block, query, slot) — the single
//           source of truth (HF `_convert_4d_mask_to_blocked_5d`: encodes
//           valid[qi] & valid[ki] & the causal window q-12 ≤ ki ≤ q).
//           mask==nullptr falls back to window math: 0 ≤ qi-ki ≤ max_past
//           (inclusive — HF allows keys q-12..q, i.e. dist ≤ max_past = 12;
//           the general form also allows -max_future ≤ qi-ki < 0).
//   out_qi = softmax(attn) · v_ki  (f32; bf16 store)
//
// Padded rows (query index ≥ seq) are left unwritten: HF slices them off
// before `post` and the output mask strips them downstream, so writing them
// would risk an OOB past a [seq]-sized out buffer (HF's NaN there is never
// consumed). All-invalid VALID rows are likewise unobservable in practice.
//
// Grid: (ceil(seq/chunk), 1, 1)  Block: (chunk*32, 1, 1)  (384 for chunk 12)
// Args: (q, k, v, spd, rel_k, mask, out, seq, heads, head_dim,
//        softcap, invalid_logits, chunk, ctx_left, ctx_right)
extern "C" __global__ void gemma_audio_chunked_attn(
    const __nv_bfloat16* __restrict__ q,      // [seq, heads*head_dim] raw q_proj
    const __nv_bfloat16* __restrict__ k,      // [seq, heads*head_dim] raw k_proj
    const __nv_bfloat16* __restrict__ v,      // [seq, heads*head_dim] raw v_proj
    const __nv_bfloat16* __restrict__ spd,    // [head_dim] = softplus(per_dim_scale)
    const __nv_bfloat16* __restrict__ rel_k,  // [rel_len, heads*head_dim] = rel_k_proj(pos_emb)
    const unsigned char* __restrict__ mask,   // [nblocks, chunk, context] u8 blocked mask (1=attend), or nullptr
    __nv_bfloat16* __restrict__ out,          // [seq, heads*head_dim]
    unsigned int seq, unsigned int heads, unsigned int head_dim,
    float softcap, float invalid_logits,
    unsigned int chunk, unsigned int ctx_left, unsigned int ctx_right
) {
    // Static caps: shipped tower is chunk 12 → context 24, head_dim 128;
    // guard explicitly so a future geometry change fails loudly, not OOB.
    const unsigned int MAX_CTX = 24;
    const unsigned int MAX_HD = 128;
    const unsigned int max_past = ctx_left - 1;
    const unsigned int max_future = ctx_right;
    const unsigned int context = chunk + max_past + max_future;
    const unsigned int rel_len = context / 2 + 1;
    const unsigned int num_blocks = (seq + chunk - 1) / chunk;
    if (context > MAX_CTX || head_dim > MAX_HD) return;

    const unsigned int b = blockIdx.x;
    const unsigned int qidx = threadIdx.x / 32;  // query index within chunk
    const unsigned int lane = threadIdx.x % 32;
    if (b >= num_blocks || qidx >= chunk) return;

    const unsigned int h = lane / 4;           // head (4 lanes per head)
    const unsigned int d0 = lane % 4;          // dim offset within head
    if (h >= heads) return;

    const unsigned int qi = b * chunk + qidx;  // absolute (padded) query row
    const unsigned int stride = heads * head_dim;
    const unsigned long long q_off = (unsigned long long)qi * stride + h * head_dim;

    // HF constants (f64 formula, f32-rounded): q_scale = hd^-0.5 / ln2,
    // k_scale = ln(1+e) / ln2.
    const float q_scale = (float)(pow((double)head_dim, -0.5) / 0.6931471805599453);
    const float k_scale = (float)(1.3132616875182228 / 0.6931471805599453);

    // Padded rows (qi >= seq) are NOT written — HF slices them off before
    // `post` (attn_output[:, :seq_length]), and the output mask strips them
    // downstream, so the Rust's out buffer only needs [seq] rows. (HF's NaN
    // in those rows is never consumed; skipping beats both NaN and an OOB
    // write past a [seq]-sized buffer.)
    if (qi >= seq) return;

    // ── This lane's query slice, pre-scaled: q * q_scale * softplus(scale) ──
    float qv[MAX_HD / 4];
    unsigned int nq = 0;
    for (unsigned int d = d0; d < head_dim; d += 4) {
        qv[nq] = bf16_to_f32(q[q_off + d]) * q_scale * bf16_to_f32(spd[d]);
        ++nq;
    }

    // ── Scores over the 24-slot context window ──
    // Validity: the host-built BLOCKED mask (HF `_convert_4d_mask_to_blocked_5d`
    // semantics — one u8 per (block, query, slot), 1 = attend) is the single
    // source of truth; it encodes valid[q], valid[kv] AND the causal window in
    // one bit, so the kernel does NOT re-derive the window itself (a flat [seq]
    // validity read here was the bug — it aliased the wrong bytes and zeroed
    // q_ok for every query whose flat index fell on a 0 bit). `mask==nullptr`
    // falls back to the raw window math (inclusive `dist <= max_past`, matching
    // HF's `attention_context_left-1` = 12 → keys q-12..q).
    float scores[MAX_CTX];
    const float inv_cap = 1.0f / softcap;
    const unsigned char* mrow = (mask == nullptr)
        ? nullptr
        : mask + (unsigned long long)(b * chunk + qidx) * context;
    for (unsigned int c = 0; c < context; ++c) {
        int ki = (int)(b * chunk) + (int)c - (int)max_past;
        bool key_ok = (ki >= 0 && ki < (int)seq);

        // Absolute term: dot(q, k_ki), k scaled by k_scale (HF applies
        // `key_states * k_scale` before blocking). Keys outside [0, seq) are
        // the padded zero rows — skip the read, the dot is 0 by construction.
        float ac = 0.0f;
        if (key_ok) {
            const __nv_bfloat16* kp = k + (unsigned long long)ki * stride + h * head_dim;
            for (unsigned int j = 0; j < nq; ++j)
                ac += qv[j] * bf16_to_f32(kp[d0 + 4 * j]) * k_scale;
        }
        // Relative term: HF's `_rel_shift` maps raw[j, p] (p = position index
        // into rel_k) onto context slot c for query j as p = c - j (verified
        // against the HF oracle: shifted[j,c] = raw[j, c-j], 0 when c-j ∉
        // [0, rel_len)). So slot c of query qidx reads rel_k[c - qidx], NOT
        // rel_k[c] — the unshifted read put every query's bias on the wrong
        // slot (and since |bd| ≈ 7×|ac|, it dominated the scores).
        int rel_i = (int)c - (int)qidx;
        float bd = 0.0f;
        if (rel_i >= 0 && rel_i < (int)rel_len) {
            const __nv_bfloat16* rp = rel_k + (unsigned long long)rel_i * stride + h * head_dim;
            for (unsigned int j = 0; j < nq; ++j) bd += qv[j] * bf16_to_f32(rp[d0 + 4 * j]);
        }
        // 4-lane group reduce (lanes h*4..h*4+3 of the warp).
        ac += __shfl_xor_sync(0xffffffffu, ac, 1);
        ac += __shfl_xor_sync(0xffffffffu, ac, 2);
        bd += __shfl_xor_sync(0xffffffffu, bd, 1);
        bd += __shfl_xor_sync(0xffffffffu, bd, 2);

        float raw = ac + bd;
        float s = tanhf(raw * inv_cap) * softcap;
        bool valid;
        if (mrow != nullptr) {
            valid = mrow[c] != 0;
        } else {
            int dist = (int)qi - ki;
            bool in_window = (dist >= 0 && dist <= (int)max_past)
                          || (dist < 0 && (unsigned int)(-dist) <= max_future);
            valid = in_window && key_ok;
        }
        scores[c] = valid ? s : invalid_logits;
    }

    // ── Softmax over the context slots (per-lane copies are identical) ──
    float max_s = scores[0];
    for (unsigned int c = 1; c < context; ++c) max_s = fmaxf(max_s, scores[c]);
    float sum = 0.0f;
    for (unsigned int c = 0; c < context; ++c) sum += expf(scores[c] - max_s);

    if (!(sum > 0.0f)) {  // all slots invalid (fully padded row)
        for (unsigned int d = d0; d < head_dim; d += 4) out[q_off + d] = f32_to_bf16(0.0f);
        return;
    }
    float inv_sum = 1.0f / sum;
    for (unsigned int c = 0; c < context; ++c) scores[c] = expf(scores[c] - max_s) * inv_sum;

    // ── Weighted value sum → output row ──
    unsigned int d = d0;
    for (unsigned int j = 0; j < nq; ++j, d += 4) {
        float o = 0.0f;
        for (unsigned int c = 0; c < context; ++c) {
            int ki = (int)(b * chunk) + (int)c - (int)max_past;
            if (scores[c] == 0.0f) continue;  // masked slot: no contribution
            o += scores[c] * bf16_to_f32(v[(unsigned long long)ki * stride + h * head_dim + d]);
        }
        out[q_off + d] = f32_to_bf16(o);
    }
}

// ── gemma_audio_silu ────────────────────────────────────────────────────────
// SiLU activation, in place: x[i] = x[i] · sigmoid(x[i]). Wave-4C contract
// (gemma_audio_encoder/enc_impl/conv1d.rs + ffn.rs): grid (ceil(n/256),1,1),
// block (256,1,1), args (x, n).
// FIX: was v/(1+exp(-v)) then x*v — that squared the input (v²·σ(v)). The
// division already yields v·σ(v); now multiply that by sigmoid(v) once.
extern "C" __global__ void gemma_audio_silu(__nv_bfloat16* x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf16_to_f32(x[i]);
    float sig = 1.0f / (1.0f + __expf(-v));
    x[i] = f32_to_bf16(v * sig);
}

// ── gemma_audio_bias_add ────────────────────────────────────────────────────
// Row-broadcast bias add: out[r·cols + c] += bias[c]. Wave-4C contract
// (gemma_audio_encoder/enc_impl/project.rs): grid (ceil(rows·cols/256),1,1),
// block (256,1,1), args (out, bias, rows, cols).
extern "C" __global__ void gemma_audio_bias_add(
    __nv_bfloat16* out,
    const __nv_bfloat16* bias,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int c = i % cols;
    out[i] = f32_to_bf16(bf16_to_f32(out[i]) + bf16_to_f32(bias[c]));
}
