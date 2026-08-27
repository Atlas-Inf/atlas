// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B vision tower: patch embedder + 16-layer ViT + pooler +
//! `embed_vision` projection, GPU-orchestrated from Rust.
//!
//! Wave 2 of the multimodal bring-up: this is the ENCODER STRUCT ONLY — the
//! orchestration contract, buffer layout, weight structs and kernel-handle
//! resolution. CUDA kernels land in Wave 3; until then every gemma-specific
//! kernel resolves to a documented no-op stub (null handle → launch skipped,
//! debug-logged) exactly as the Qwen3-VL encoder did pre-kernel, while the
//! shared kernels (dense GEMM / RMSNorm / GeGLU) are reused as-is.
//!
//! # Pipeline (per image, see `crate::media::gemma_vision::GemmaImageInput`)
//!
//! 1. **Patch embedder** — in-model scale `2×(x−0.5)` on the `[P, 768]` patch
//!    vectors (host-side in Wave 2, fused kernel in Wave 3), then
//!    `input_proj: Linear(768→768, bias=False)`, then add the learned 2D
//!    position table `[2, 10240, 768]` — `x_emb + y_emb` per patch, padding
//!    patches get zero.
//! 2. **16-layer ViT** — per layer, the Gemma-4 **four-norm** sandwich
//!    (norm → attention → norm → residual; norm → GeGLU → norm → residual):
//!    `input_layernorm` (RMSNorm 768) → MHA (12 heads × head_dim 64, **no
//!    GQA**) with **QK-Norm** (RMSNorm over head_dim 64 on q and k, applied
//!    BEFORE the rotary + attention) → `o_proj` →
//!    `post_attention_layernorm` → `pre_feedforward_layernorm` → GeGLU MLP
//!    (`gate_proj`/`up_proj` [3072,768], `down_proj` [768,3072],
//!    `gelu_pytorch_tanh`) → `post_feedforward_layernorm`. RoPE: θ=100,
//!    head_dim 64, per-head on q/k after QK-Norm; cos/sin precomputed on the
//!    host per image from `pos_ids`.
//! 3. **Pooler** — average-pool `pooling_kernel_size²` patch groups → soft
//!    tokens, scaled by `√hidden_size` in f32, padding stripped.
//! 4. **`embed_vision` projection (INSIDE this encoder)** — RMSNorm(768,
//!    `with_scale=False` → ones weight) then `Linear(768→1536, bias=False)`.
//!    Output lands in [`GemmaVisionEncoder::buf_out`] `[Σsoft, 1536]` BF16,
//!    image-order packed, so the downstream text splice is a straight copy.
//!
//! # Non-obvious vs Qwen3-VL
//!
//! - **Clipped linears.** Every attention/MLP linear is a `ClippableLinear`:
//!   activation clamped to `[input_min, input_max]` BEFORE the GEMM, output
//!   clamped to `[output_min, output_max]` AFTER (4 scalar bounds loaded with
//!   the weights). Qwen3-VL has no clip stage; the bounds ride as kernel
//!   scalar args in Wave 2 and become `gemma_vision_clamp` launches in Wave 3.
//! - **QK-Norm.** Qwen3-VL's ViT skips it; Gemma normalizes q and k per head
//!   (RMSNorm over head_dim) after the Q/K projections and before RoPE.
//! - **Per-image geometry is host-side.** Like Qwen3-VL, per-image pos/rope
//!   tables are precomputed on the host and uploaded; the position table is
//!   downloaded once at init (31 MB BF16) and gathered per image.
//! - **Pooling replaces the merger.** Qwen3-VL merges 2×2 with MLPs; Gemma
//!   average-pools 3×3 groups into soft tokens.
//!
//! # Buffers
//!
//! Scratch is sized to ONE image's caps (`max_patches` patch rows,
//! `max_soft_tokens` soft-token rows) and the batched forward guards
//! Σpatches ≤ `max_patches` (per-image fallback loop beyond that), mirroring
//! the Qwen3-VL `forward_batched` contract. `buf_out` holds `max_soft_tokens`
//! rows of `[·, 1536]` BF16; the batched guard keeps Σsoft ≤ `max_soft_tokens`
//! in the packed path and the fallback refuses to overrun.

use std::sync::atomic::{AtomicUsize, Ordering};

use spark_runtime::gpu::{DevicePtr, KernelHandle};

use crate::weight_map::DenseWeight;

/// Output width of the encoder: the `embed_vision` projection maps
/// hidden_size (768) → OUT_HIDDEN_SIZE (1536), the text embedding width.
/// Baked in (not read from config): the downstream splice is a straight
/// copy of `buf_out` rows into the text embedding stream, so a checkpoint
/// whose `embed_vision` projects elsewhere must be refused, not stretched.
pub const OUT_HIDDEN_SIZE: usize = 1536;

/// A Gemma-4 E2B `ClippableLinear`: a biasless BF16 `Linear` with four
/// scalar clip bounds loaded alongside the weights.
///
/// Semantics (HF `ClippableLinear`): the activation is clamped to
/// `[input_min, input_max]` BEFORE the GEMM and the output is clamped to
/// `[output_min, output_max]` AFTER it. The bounds are passed as kernel
/// scalar args at launch; Wave 3's `gemma_vision_clamp` kernel consumes
/// them.
#[derive(Debug, Clone, Copy)]
pub struct ClipLinearWeights {
    /// BF16 weight, row-major `[out_features, in_features]` (no bias).
    pub weight: DenseWeight,
    /// Activation clamp, lower bound (applied pre-GEMM).
    pub input_min: f32,
    /// Activation clamp, upper bound (applied pre-GEMM).
    pub input_max: f32,
    /// Output clamp, lower bound (applied post-GEMM).
    pub output_min: f32,
    /// Output clamp, upper bound (applied post-GEMM).
    pub output_max: f32,
}

/// Per-layer weights of the Gemma-4 E2B vision tower.
///
/// Checkpoint naming (verified): `model.vision_tower.encoder.layers.{i}.`
/// `{input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
/// post_feedforward_layernorm}.weight`, `.{self_attn.q_norm,k_norm}.weight`
/// [head_dim], `.{self_attn.q_proj,k_proj,v_proj,o_proj}.linear.weight`
/// (each a `ClippableLinear` with its 4 scalar bounds), and
/// `.{mlp.gate_proj,up_proj,down_proj}.linear.weight`.
#[derive(Debug, Clone)]
pub struct GemmaVisionLayerWeights {
    /// `input_layernorm.weight` — [hidden_size] RMSNorm.
    pub input_layernorm: DenseWeight,
    /// `self_attn.q_norm.weight` — [head_dim] per-head QK-Norm.
    pub q_norm: DenseWeight,
    /// `self_attn.k_norm.weight` — [head_dim] per-head QK-Norm.
    pub k_norm: DenseWeight,
    /// `self_attn.q_proj` — ClippableLinear [hidden, hidden].
    pub q_proj: ClipLinearWeights,
    /// `self_attn.k_proj` — ClippableLinear [hidden, hidden].
    pub k_proj: ClipLinearWeights,
    /// `self_attn.v_proj` — ClippableLinear [hidden, hidden].
    pub v_proj: ClipLinearWeights,
    /// `self_attn.o_proj` — ClippableLinear [hidden, hidden].
    pub o_proj: ClipLinearWeights,
    /// `post_attention_layernorm.weight` — [hidden_size] RMSNorm.
    pub post_attention_layernorm: DenseWeight,
    /// `pre_feedforward_layernorm.weight` — [hidden_size] RMSNorm.
    pub pre_feedforward_layernorm: DenseWeight,
    /// `mlp.gate_proj` — ClippableLinear [intermediate, hidden].
    pub gate_proj: ClipLinearWeights,
    /// `mlp.up_proj` — ClippableLinear [intermediate, hidden].
    pub up_proj: ClipLinearWeights,
    /// `mlp.down_proj` — ClippableLinear [hidden, intermediate].
    pub down_proj: ClipLinearWeights,
    /// `post_feedforward_layernorm.weight` — [hidden_size] RMSNorm.
    pub post_feedforward_layernorm: DenseWeight,
}

/// All weights of the Gemma-4 E2B vision tower + `embed_vision` projection,
/// as handed to [`GemmaVisionEncoder::new`] by the weight loader
/// (`weight_loader/gemma4/loader_d.rs`, Wave 3).
#[derive(Debug, Clone)]
pub struct GemmaVisionWeights {
    /// `patch_embedder.input_proj.weight` — [hidden, hidden] Linear, no bias.
    pub input_proj: DenseWeight,
    /// `patch_embedder.position_embedding_table` — [2, position_embedding_size, hidden].
    /// First dim is the image/frame index slot (static images use slot 0);
    /// per patch the embedding is `table[slot][x] + table[slot][y]`.
    pub position_table: DenseWeight,
    /// One entry per `num_hidden_layers` ViT block.
    pub layers: Vec<GemmaVisionLayerWeights>,
    /// `embed_vision.embedding_projection.weight` — [OUT_HIDDEN_SIZE, hidden].
    pub embed_vision_projection: DenseWeight,
}

/// Gemma-4 E2B vision tower: patch embedder → 16×ViT → pooler →
/// embed_vision projection (the latter INSIDE the encoder so the downstream
/// splice is a straight copy of [`GemmaVisionEncoder::buf_out`]).
pub struct GemmaVisionEncoder {
    // ── Weights (BF16 device pointers) ─────────────────────────────
    /// `input_proj` weight — [hidden, hidden].
    pub input_proj_w: DevicePtr,
    /// Position table base — [2, position_embedding_size, hidden].
    pub position_table: DevicePtr,
    /// Per-layer weights (norms, QK-norm, 7 clipped linears).
    pub layers: Vec<GemmaVisionLayerWeights>,
    /// `embed_vision` projection weight — [OUT_HIDDEN_SIZE, hidden].
    pub embed_vision_proj_w: DevicePtr,

    // ── Kernel handles ─────────────────────────────────────────────
    /// `gemm::dense_gemm_bf16` — REUSED (common tree, hard).
    k_gemm: KernelHandle,
    /// `norm::rms_norm` — REUSED for the four per-layer norms + embed_vision
    /// RMSNorm (gemma-4-e2b tree, hard).
    k_rms_norm: KernelHandle,
    /// `gelu::gelu_mul` — REUSED for the GeGLU activation
    /// (`output = gelu_tanh(gate) × up`; gemma-4-e2b tree, soft).
    k_gelu_mul: KernelHandle,
    /// `bf16_add::bf16_add_inplace` — REUSED for the pos-add and both
    /// residual adds (common tree, soft).
    k_add: KernelHandle,
    /// QK-Norm — per-head RMSNorm over head_dim 64. Wave 2 resolves this to
    /// the generic `norm::rms_norm` (shape-compatible per head); Wave 3
    /// points it at a dedicated `gemma_vision_qk_norm` kernel.
    k_qk_norm: KernelHandle,
    /// `gemma_vision::gemma_vision_rope_rotate` — Wave 3 kernel. STUB today:
    /// null → launch skipped (no-op, debug-logged).
    k_rope_rotate: KernelHandle,
    /// `gemma_vision::gemma_vision_attention` — Wave 3 MHA kernel. STUB today.
    k_attn: KernelHandle,
    /// `gemma_vision::gemma_vision_clamp` — Wave 3 ClippableLinear clamp
    /// (pre-GEMM input / post-GEMM output). STUB today.
    k_clamp: KernelHandle,
    /// `gemma_vision::gemma_vision_pool` — Wave 3 average-pool kernel. STUB
    /// today.
    k_pool: KernelHandle,

    // ── Host-side prep state ───────────────────────────────────────
    /// Position table downloaded at init, BF16 bits `[2 × pos_size × hidden]`
    /// row-major, for the per-image `x_emb + y_emb` gather.
    position_table_host: Vec<u16>,
    /// RoPE inverse frequencies: `inv_freq[k] = θ^(−2k/head_dim)`, k in
    /// `[0, head_dim/4)`, θ = `cfg.rope_theta` (100.0 for the vision tower).
    rope_inv_freq: Vec<f32>,

    // ── Scratch buffers (one image's caps; see module docs) ────────
    /// `[max_patches, hidden]` BF16 — active hidden (pixels in, hidden out).
    pub buf_h1: DevicePtr,
    /// `[max_patches, hidden]` BF16 — residual save.
    pub buf_h2: DevicePtr,
    /// `[max_patches, hidden]` BF16 — per-image position embedding gather.
    pub buf_pos: DevicePtr,
    /// `[max_patches, 3×hidden]` BF16 — q/k/v projection outputs.
    pub buf_qkv: DevicePtr,
    /// `[2 × max_patches, intermediate]` BF16 — gate plane + up plane.
    pub buf_wide: DevicePtr,
    /// `[max_patches, hidden]` BF16 — attention-o staging.
    pub buf_mlp: DevicePtr,
    /// `[max_soft_tokens, hidden]` BF16 — pooled soft tokens (packed batch).
    pub buf_pool: DevicePtr,
    /// `[max_patches, head_dim]` BF16 — per-image rotary cos.
    pub buf_rope_cos: DevicePtr,
    /// `[max_patches, head_dim]` BF16 — per-image rotary sin.
    pub buf_rope_sin: DevicePtr,
    /// `[max_soft_tokens, OUT_HIDDEN_SIZE]` BF16 — final packed output,
    /// image-order. ← the splice source.
    pub buf_out: DevicePtr,
    /// `[hidden]` BF16 ones — `embed_vision` RMSNorm weight
    /// (`with_scale=False` → pure `x·rms`).
    pub norm_unit_w: DevicePtr,
    /// Ones over head_dim — V-norm weight (HF v_norm, with_scale=False).
    pub head_norm_unit_w: DevicePtr,

    // ── Config ─────────────────────────────────────────────────────
    /// ViT hidden dimension (768).
    pub hidden_size: usize,
    /// Number of attention heads (12).
    pub num_heads: usize,
    /// Attention head dimension (64).
    pub head_dim: usize,
    /// MLP intermediate size (3072).
    pub intermediate_size: usize,
    /// Patch pooling kernel size (3).
    pub pooling_kernel_size: usize,
    /// Patch rows per image cap (2520).
    pub p_max: usize,
    /// Soft-token rows per image cap (280) — also `buf_out` row capacity.
    pub s_max: usize,
    /// Position table slot count = `position_embedding_size` (10240).
    pub position_embedding_size: usize,
    /// RMSNorm epsilon (1e-6).
    pub norm_eps: f32,
    /// RoPE θ (100.0).
    pub rope_theta: f32,
    /// Per-image patch vector dim `3 × patch_size²` (768).
    pub patch_dim: usize,

    /// Soft tokens produced by the most recent `forward_batched` (sum of the
    /// returned per-image counts) — the `buf_out` row count the splice reads.
    total_soft_tokens: AtomicUsize,
}

impl GemmaVisionEncoder {
    /// Most recent `forward_batched`'s Σsoft_tokens — the row count the
    /// downstream splice copies from [`Self::buf_out`].
    pub fn total_soft_tokens(&self) -> usize {
        self.total_soft_tokens.load(Ordering::Relaxed)
    }

    /// The packed `[Σsoft, OUT_HIDDEN_SIZE]` BF16 output buffer (image-order).
    pub fn buf_out(&self) -> DevicePtr {
        self.buf_out
    }
}

mod enc_impl;
