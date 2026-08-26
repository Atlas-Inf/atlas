// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B audio tower: subsample conv projection + 12-layer conformer +
//! `output_proj` + `embed_audio` projection, GPU-orchestrated from Rust.
//!
//! Wave 4A of the multimodal bring-up: this is the ENCODER STRUCT ONLY — the
//! orchestration contract, buffer layout, weight structs and kernel-handle
//! resolution. CUDA kernels land in Wave 4C; until then every gemma-specific
//! kernel resolves to a documented no-op stub (null handle → launch skipped,
//! debug-logged) exactly as the Gemma-4 vision encoder did in Wave 2, while
//! the shared kernels (dense GEMM / RMSNorm / sigmoid-GLU / scaled-add) are
//! reused as-is.
//!
//! # Pipeline (per clip, mirroring HF `Gemma4AudioModel.forward` +
//! `Gemma4MultimodalEmbedder`; tensor shapes VERIFIED against the
//! google/gemma-4-E2B-it checkpoint on 2026-08-10)
//!
//! 1. **Subsample conv projection** (`model.audio_tower.subsample_conv_projection.*`,
//!    verified `layer0.conv.weight [128,1,3,3]`, `layer0.norm.weight [128]`,
//!    `layer1.conv.weight [32,128,3,3]`, `layer1.norm.weight [32]`,
//!    `input_proj_linear.weight [1024,1024]`): `[T, 128]` mel features →
//!    Conv2d(1→128, 3×3, stride 2, pad 1) → LayerNorm(128) → ReLU →
//!    Conv2d(128→32, 3×3, stride 2, pad 1) → LayerNorm(32) → ReLU →
//!    flatten `[T/4, 32×32=1024]` → Linear(1024→1024). The 4× time reduction
//!    is `(T+3)/4` frames; the mask is applied multiplicatively and
//!    subsampled `[::2]` twice (`subsample_mask`: position `i` ← `mask[4i]`).
//! 2. **12 conformer layers** (`model.audio_tower.layers.{i}.*`, i = 0..11 —
//!    verified 12 layers): `feed_forward1` → [attn block: `norm_pre_attn` →
//!    chunked local attention → `norm_post_attn` → +residual] → `lconv1d`
//!    (light conv) → `feed_forward2` → `norm_out`. Each FFN: clipped
//!    `ffw_layer_1 [4096,1024]` → **SiLU** (config `hidden_act = "silu"`,
//!    verified) → clipped `ffw_layer_2 [1024,4096]` → `post_layer_norm` →
//!    `×residual_weight (0.5)` → +residual. The light conv: clipped
//!    `linear_start [2048,1024]` → `F.glu` (first half × σ(second half)) →
//!    depthwise causal conv1d (kernel 5, left pad 4,
//!    `depthwise_conv1d.weight [1024,1,5]` verified) → `conv_norm` → SiLU →
//!    clipped `linear_end`. The chunked attention: q/k/v + `post` clipped
//!    linears [1024,1024], `relative_k_proj` (plain, unclipped, [1024,1024]),
//!    `per_dim_scale [128]`, q × `q_scale·softplus(per_dim_scale)`, k ×
//!    `k_scale`, relative sinusoidal pos bias over `context_size = chunk +
//!    left−1 + right = 24` (`rel_pos_enc` is pure config math — no checkpoint
//!    tensor), tanh softcap 50.0, invalid logits −1e9. The `gradient_clipping`
//!    clamps (config value 1e10) never fire in BF16 and are documented no-ops.
//! 3. **`output_proj`** (`model.audio_tower.output_proj.weight [1536,1024]`,
//!    **`bias [1536]` verified present**) → **`embed_audio` INSIDE this
//!    encoder** (`model.embed_audio.embedding_projection.weight [1536,1536]`
//!    verified; no norm tensor — RMSNorm `with_scale=False` → ones weight):
//!    `[T/4, 1024]` → `[T/4, 1536]` + bias → RMSNorm(1536, ones) → Linear
//!    (1536→1536). Output lands in [`GemmaAudioEncoder::buf_out`]
//!    `[Σvalid, 1536]` BF16, clip-order, padding tokens DROPPED (HF strips
//!    `audio_features[audio_mask]` before the text splice).
//!
//! # Non-obvious vs the vision tower
//!
//! - **No RoPE.** The conformer uses a relative position BIAS (sinusoidal,
//!   `context/2+1 = 13` positions) — no rotary anywhere.
//! - **Subsample conv ≠ patch embed.** Vision runs a Linear patch embed; audio
//!   runs two 2D convs with LayerNorms (mean-subtracting — NOT RMSNorm) fused
//!   into one Wave-4C kernel, then the flatten linear on the shared GEMM.
//! - **Two FFNs per layer**, each with its own clipped linears and an
//!   internal `0.5·post_norm + residual` blend; `norm_out` closes the layer
//!   WITHOUT a residual add.
//! - **GLU, not GeGLU.** The light conv gates via `a·σ(b)` (first half ×
//!   sigmoid(second half)) — the shared `residual_add::sigmoid_gate_mul`.
//! - **`output_proj` carries a bias** — the first biased linear in the gemma
//!   media stack; a dedicated `gemma_audio_bias_add` stub applies it.
//!
//! # Buffers
//!
//! Scratch is sized to ONE clip's caps: `t_max = token_cap` full rows (the
//! conv output length is bounded by `token_cap` when the mel length is capped
//! at `4×token_cap`), `4×t_max` mel frames, and the packed forward guards
//! Σrows ≤ `t_max` (per-clip fallback loop beyond that), mirroring the vision
//! `forward_batched` contract. `buf_out` holds `t_max` rows of `[·, 1536]`
//! BF16; the guard keeps Σvalid ≤ `t_max`.

use std::sync::atomic::{AtomicUsize, Ordering};

use spark_runtime::gpu::{DevicePtr, KernelHandle};

use crate::layers::gemma_vision_encoder::ClipLinearWeights;
use crate::weight_map::DenseWeight;

/// Output width of the encoder: `output_proj` maps hidden_size (1024) →
/// OUT_HIDDEN_SIZE (1536), the text embedding width. Baked in (not read from
/// config): the downstream splice is a straight copy of `buf_out` rows into
/// the text embedding stream, so a checkpoint whose `output_proj` projects
/// elsewhere must be refused, not stretched. `new()` enforces
/// `cfg.output_proj_dims == OUT_HIDDEN_SIZE`.
pub const OUT_HIDDEN_SIZE: usize = 1536;

/// Tanh softcap of the chunked attention logits (checkpoint
/// `attention_logit_cap = 50.0`). Not carried by the Atlas
/// `GemmaAudioConfig` — baked like `OUT_HIDDEN_SIZE`, enforced nowhere (the
/// Wave-4C kernel takes it as a scalar arg).
pub const ATTENTION_LOGIT_CAP: f32 = 50.0;

/// Logit value written where the blocked attention mask says "no attend"
/// (checkpoint `attention_invalid_logits_value = −1e9`).
pub const ATTENTION_INVALID_LOGITS: f32 = -1e9;

/// Weights of the subsample conv projection
/// (`model.audio_tower.subsample_conv_projection.*`, shapes verified).
#[derive(Debug, Clone)]
pub struct GemmaAudioSubsampleWeights {
    /// `layer0.conv.weight` — [128, 1, 3, 3] Conv2d, no bias.
    pub conv0: DenseWeight,
    /// `layer0.norm.weight` — [128] LayerNorm weight (mean-subtracting, no
    /// bias; consumed inside `gemma_audio_subsample_conv`, Wave 4C).
    pub ln0: DenseWeight,
    /// `layer1.conv.weight` — [32, 128, 3, 3] Conv2d, no bias.
    pub conv1: DenseWeight,
    /// `layer1.norm.weight` — [32] LayerNorm weight.
    pub ln1: DenseWeight,
    /// `input_proj_linear.weight` — [1024, 1024] Linear (flatten → hidden).
    pub input_proj_linear: DenseWeight,
}

/// Weights of one conformer feed-forward
/// (`model.audio_tower.layers.{i}.feed_forward{1,2}.*`).
#[derive(Debug, Clone)]
pub struct GemmaAudioFfnWeights {
    /// `ffw_layer_1` — ClippableLinear [4×hidden, hidden].
    pub ffw_layer_1: ClipLinearWeights,
    /// `ffw_layer_2` — ClippableLinear [hidden, 4×hidden].
    pub ffw_layer_2: ClipLinearWeights,
    /// `pre_layer_norm.weight` — [hidden] RMSNorm.
    pub pre_layer_norm: DenseWeight,
    /// `post_layer_norm.weight` — [hidden] RMSNorm (output × residual_weight).
    pub post_layer_norm: DenseWeight,
}

/// Weights of the light conv1d
/// (`model.audio_tower.layers.{i}.lconv1d.*`).
#[derive(Debug, Clone)]
pub struct GemmaAudioLightConvWeights {
    /// `linear_start` — ClippableLinear [2×hidden, hidden] (GLU inputs).
    pub linear_start: ClipLinearWeights,
    /// `linear_end` — ClippableLinear [hidden, hidden].
    pub linear_end: ClipLinearWeights,
    /// `depthwise_conv1d.weight` — [hidden, 1, 5] depthwise causal conv.
    pub depthwise_conv1d: DenseWeight,
    /// `pre_layer_norm.weight` — [hidden] RMSNorm.
    pub pre_layer_norm: DenseWeight,
    /// `conv_norm.weight` — [hidden] RMSNorm.
    pub conv_norm: DenseWeight,
}

/// Weights of the chunked local attention
/// (`model.audio_tower.layers.{i}.self_attn.*`).
#[derive(Debug, Clone)]
pub struct GemmaAudioAttnWeights {
    /// `q_proj` — ClippableLinear [hidden, hidden].
    pub q_proj: ClipLinearWeights,
    /// `k_proj` — ClippableLinear [hidden, hidden].
    pub k_proj: ClipLinearWeights,
    /// `v_proj` — ClippableLinear [hidden, hidden].
    pub v_proj: ClipLinearWeights,
    /// `post` — ClippableLinear [hidden, hidden].
    pub post: ClipLinearWeights,
    /// `relative_k_proj.weight` — [hidden, hidden] PLAIN linear (not
    /// clipped — verified: no bound scalars in the checkpoint).
    pub relative_k_proj: DenseWeight,
    /// `per_dim_scale` — [head_dim] BF16; `softplus` of it scales q.
    pub per_dim_scale: DenseWeight,
}

/// Per-layer weights of the audio tower (12 layers, i = 0..11, verified).
#[derive(Debug, Clone)]
pub struct GemmaAudioLayerWeights {
    /// `feed_forward1` — FFN before the attention block.
    pub feed_forward1: GemmaAudioFfnWeights,
    /// `feed_forward2` — FFN after the light conv.
    pub feed_forward2: GemmaAudioFfnWeights,
    /// `lconv1d` — the light conv1d.
    pub lconv1d: GemmaAudioLightConvWeights,
    /// `self_attn` — the chunked local attention.
    pub self_attn: GemmaAudioAttnWeights,
    /// `norm_pre_attn.weight` — [hidden] RMSNorm.
    pub norm_pre_attn: DenseWeight,
    /// `norm_post_attn.weight` — [hidden] RMSNorm.
    pub norm_post_attn: DenseWeight,
    /// `norm_out.weight` — [hidden] RMSNorm, layer output (no residual).
    pub norm_out: DenseWeight,
}

/// `output_proj` weights (`model.audio_tower.output_proj.*`, verified).
#[derive(Debug, Clone)]
pub struct GemmaAudioOutputProj {
    /// `weight` — [OUT_HIDDEN_SIZE, hidden].
    pub weight: DenseWeight,
    /// `bias` — [OUT_HIDDEN_SIZE] (verified present in the checkpoint).
    pub bias: DenseWeight,
}

/// All weights of the Gemma-4 E2B audio tower + `embed_audio` projection, as
/// handed to [`GemmaAudioEncoder::new`] by the weight loader (Wave 4B).
#[derive(Debug, Clone)]
pub struct GemmaAudioWeights {
    /// `model.audio_tower.subsample_conv_projection.*`.
    pub subsample: GemmaAudioSubsampleWeights,
    /// One entry per `num_hidden_layers` conformer block.
    pub layers: Vec<GemmaAudioLayerWeights>,
    /// `model.audio_tower.output_proj.*`.
    pub output_proj: GemmaAudioOutputProj,
    /// `model.embed_audio.embedding_projection.weight` — [OUT_HIDDEN_SIZE,
    /// OUT_HIDDEN_SIZE] (the RMSNorm before it has no weight —
    /// `with_scale=False` → ones).
    pub embed_audio_projection: DenseWeight,
}

/// Gemma-4 E2B audio tower: subsample conv → 12×conformer → output_proj →
/// embed_audio projection (the latter INSIDE the encoder so the downstream
/// splice is a straight copy of [`GemmaAudioEncoder::buf_out`]).
pub struct GemmaAudioEncoder {
    // ── Weights (BF16 device pointers) ─────────────────────────────
    /// `layer0.conv.weight` — [128, 1, 3, 3].
    pub subsample_conv0_w: DevicePtr,
    /// `layer0.norm.weight` — [128].
    pub subsample_ln0_w: DevicePtr,
    /// `layer1.conv.weight` — [32, 128, 3, 3].
    pub subsample_conv1_w: DevicePtr,
    /// `layer1.norm.weight` — [32].
    pub subsample_ln1_w: DevicePtr,
    /// `input_proj_linear.weight` — [hidden, flatten].
    pub subsample_proj_w: DevicePtr,
    /// Per-layer conformer weights.
    pub layers: Vec<GemmaAudioLayerWeights>,
    /// `output_proj.weight` — [OUT_HIDDEN_SIZE, hidden].
    pub output_proj_w: DevicePtr,
    /// `output_proj.bias` — [OUT_HIDDEN_SIZE].
    pub output_proj_b: DevicePtr,
    /// `embed_audio.embedding_projection.weight` — [OUT_HIDDEN_SIZE, OUT_HIDDEN_SIZE].
    pub embed_audio_proj_w: DevicePtr,

    // ── Kernel handles ─────────────────────────────────────────────
    /// `gemm::dense_gemm_bf16` — REUSED (common tree, hard).
    k_gemm: KernelHandle,
    /// `norm::rms_norm` — REUSED for every tower RMSNorm (hard).
    k_rms_norm: KernelHandle,
    /// `bf16_add::bf16_add_inplace` — REUSED for the residual adds (soft).
    k_add: KernelHandle,
    /// `residual_add::sigmoid_gate_mul` — REUSED for the light-conv GLU
    /// `a·σ(b)` (common tree, hard).
    k_sigmoid_gate: KernelHandle,
    /// `residual_add::bf16_scaled_add` — REUSED for the FFN
    /// `residual + residual_weight·normed` blend (common tree, hard).
    k_scaled_add: KernelHandle,
    /// `gemma_vision::gemma_vision_clamp` — REUSED from the vision tree
    /// (Wave 3) for every ClippableLinear clamp. STUB until then.
    k_clamp: KernelHandle,
    /// `gemma_audio::gemma_audio_silu` — Wave-4C SiLU elementwise (the
    /// shared tree has only `silu_mul_separate`, which is gated). STUB.
    k_silu: KernelHandle,
    /// `gemma_audio::gemma_audio_subsample_conv1` — mel → conv1 → LN1 → ReLU
    /// → the `[T1, 64, 128]` intermediate (split kernel; the fused version
    /// read conv1 rows its block never staged — OOB smem → ILLEGAL_ADDRESS).
    k_subsample_conv1: KernelHandle,
    /// `gemma_audio::gemma_audio_subsample_conv2` — conv1 intermediate →
    /// conv2 → LN2 → ReLU → flatten `[T2, 1024]`.
    k_subsample_conv2: KernelHandle,
    /// `gemma_audio::gemma_audio_chunked_attn` — Wave-4C chunked local
    /// attention with relative pos bias. STUB.
    k_chunked_attn: KernelHandle,
    /// `gemma_audio::gemma_audio_conv1d` — Wave-4C depthwise causal conv1d.
    /// STUB.
    k_conv1d: KernelHandle,
    /// `gemma_audio::gemma_audio_bias_add` — Wave-4C broadcast bias add for
    /// `output_proj.bias`. STUB.
    k_bias_add: KernelHandle,

    // ── Host-side prep state ───────────────────────────────────────
    /// Per-layer `relative_k = relative_k_proj(pos_emb)` — the [13, hidden]
    /// relative position keys, precomputed at init (pos_emb is pure config
    /// math; the projection weight is downloaded once).
    relative_k: Vec<DevicePtr>,
    /// Per-layer `spd = softplus(per_dim_scale)` — the [head_dim] query
    /// scale vector, precomputed at init (HF applies `softplus` to the
    /// learned `per_dim_scale` once; the chunked-attention kernel reads the
    /// result as its `spd` argument).
    spd_bufs: Vec<DevicePtr>,

    // ── Scratch buffers (one clip's caps; see module docs) ─────────
    /// `[4×t_max × mel_bins]` BF16 — mel upload (f32→BF16 host-side).
    pub buf_mel: DevicePtr,
    /// `[4×t_max]` u8 — per-frame validity masks (packed, clip order).
    pub buf_mask_mel: DevicePtr,
    /// `[t_max × chunk × context]` u8 — blocked attention masks (packed).
    pub buf_mask_attn: DevicePtr,
    /// `[t_max × flatten]` BF16 — subsample conv output (flatten = 32×32).
    pub buf_conv: DevicePtr,
    /// `[2×t_max × 64 × 128]` BF16 — subsample conv1 intermediate
    /// (conv1 rows = ceil(4×t_max/2) = 2×t_max, freq = mel/2 = 64, ch = 128).
    /// Written by `k_subsample_conv1`, consumed by `k_subsample_conv2`.
    pub buf_conv1: DevicePtr,
    /// `[t_max × hidden]` BF16 — hidden states (also packed).
    pub buf_h1: DevicePtr,
    /// `[t_max × hidden]` BF16 — residual save.
    pub buf_h2: DevicePtr,
    /// `[t_max × 3×hidden]` BF16 — q/k/v projection outputs.
    pub buf_qkv: DevicePtr,
    /// `[t_max × hidden]` BF16 — attention-o / light-conv staging.
    pub buf_mlp: DevicePtr,
    /// `[t_max × 2×hidden]` BF16 — light-conv GLU staging (linear_start out).
    pub buf_wide: DevicePtr,
    /// `[t_max × 4×hidden]` BF16 — FFN intermediate.
    pub buf_ffn: DevicePtr,
    /// `[t_max × OUT_HIDDEN_SIZE]` BF16 — output_proj staging (full rows).
    pub buf_proj: DevicePtr,
    /// `[t_max × OUT_HIDDEN_SIZE]` BF16 — final packed output, clip-order,
    /// padding dropped. ← the splice source.
    pub buf_out: DevicePtr,
    /// `[OUT_HIDDEN_SIZE]` BF16 ones — `embed_audio` RMSNorm weight
    /// (`with_scale=False` → pure `x·rms`).
    pub norm_unit_w: DevicePtr,

    // ── Config ─────────────────────────────────────────────────────
    /// Tower hidden dimension (1024).
    pub hidden_size: usize,
    /// Attention heads (8).
    pub num_heads: usize,
    /// Head dimension (128 = hidden/heads).
    pub head_dim: usize,
    /// FFN intermediate = 4×hidden (4096; baked, HF hardcodes ×4).
    pub intermediate_size: usize,
    /// Attention chunk size (12).
    pub chunk_size: usize,
    /// `context_left − 1` — past tokens a query attends to (12).
    pub max_past: usize,
    /// `context_right` — future tokens a query attends to (0).
    pub max_future: usize,
    /// `chunk + max_past + max_future` (24).
    pub context_size: usize,
    /// Subsample conv kernel — the depthwise conv1d kernel size (5).
    pub conv_kernel: usize,
    /// Mel bin count (128).
    pub mel_bins: usize,
    /// Per-time-row flatten width after the convs: `(mel/4) × last_channels`
    /// (32×32 = 1024).
    pub flatten_dim: usize,
    /// Row cap per clip (`token_cap`, 750) — every scratch buffer is sized
    /// to this many rows.
    pub t_max: usize,
    /// RMSNorm epsilon (1e-6).
    pub norm_eps: f32,
    /// FFN residual blend weight (0.5).
    pub residual_weight: f32,
    /// `head_dim^−0.5 / ln 2` — q scale (HF `Gemma4AudioAttention`).
    pub q_scale: f32,
    /// `ln(1+e) / ln 2` — k scale.
    pub k_scale: f32,

    /// Valid tokens produced by the most recent `forward_batched` (sum of the
    /// returned per-clip counts) — the `buf_out` row count the splice reads.
    total_soft_tokens: AtomicUsize,
}

impl GemmaAudioEncoder {
    /// Most recent `forward_batched`'s Σvalid_tokens — the row count the
    /// downstream splice copies from [`Self::buf_out`].
    pub fn total_soft_tokens(&self) -> usize {
        self.total_soft_tokens.load(Ordering::Relaxed)
    }

    /// The packed `[Σvalid, OUT_HIDDEN_SIZE]` BF16 output buffer (clip-order,
    /// padding dropped).
    pub fn buf_out(&self) -> DevicePtr {
        self.buf_out
    }
}

mod enc_impl;
