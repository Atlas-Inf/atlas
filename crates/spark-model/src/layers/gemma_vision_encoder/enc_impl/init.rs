// SPDX-License-Identifier: AGPL-3.0-only

//! `GemmaVisionEncoder::new` constructor: geometry validation (fail fast),
//! one-image-cap scratch allocation, kernel-handle resolution (reused vs
//! Wave-3 stubs) and the host-side position-table / RoPE precompute.

use anyhow::{Result, ensure};
use atlas_core::config::GemmaVisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{GemmaVisionEncoder, GemmaVisionWeights, OUT_HIDDEN_SIZE};

impl GemmaVisionEncoder {
    /// Build the vision tower from loaded weights + config.
    ///
    /// Weights come as already-loaded BF16 device pointers
    /// ([`GemmaVisionWeights`]); the loader (`weight_loader/gemma4/loader_d.rs`)
    /// slices the checkpoint's `model.vision_tower.*` / `model.embed_vision.*`
    /// tensors into them. The constructor validates the geometry against
    /// [`GemmaVisionConfig`] (heads×head_dim == hidden, layer count, position
    /// table dims), allocates the one-image scratch set, resolves kernel
    /// handles (shared kernels hard/soft, gemma-specific ones as documented
    /// stubs), and downloads the position table for the host-side per-image
    /// `x_emb + y_emb` gather.
    pub fn new(
        w: &GemmaVisionWeights,
        cfg: &GemmaVisionConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let p_max = cfg.max_patches;
        let s_max = cfg.max_soft_tokens;
        let pks = cfg.pooling_kernel_size;

        // ── Geometry validation (PCND: refuse, don't silently stretch) ──
        ensure!(
            heads * head_dim == hidden,
            "gemma vision: {heads} heads × head_dim {head_dim} != hidden_size {hidden}; \
             MHA needs heads×head_dim == hidden"
        );
        ensure!(
            w.layers.len() == cfg.num_hidden_layers,
            "gemma vision: {num} layer weight sets for {cfg} layers",
            num = w.layers.len(),
            cfg = cfg.num_hidden_layers
        );
        ensure!(
            cfg.position_table_shape.0 == 2
                && cfg.position_table_shape.1 == cfg.position_embedding_size
                && cfg.position_table_shape.2 == hidden,
            "gemma vision: position table {:?} != expected (2, {}, {hidden})",
            cfg.position_table_shape,
            cfg.position_embedding_size
        );
        ensure!(
            p_max > 0 && s_max > 0,
            "gemma vision: max_patches/max_soft_tokens must be > 0"
        );
        ensure!(
            p_max == s_max * pks * pks,
            "gemma vision: max_patches {p_max} != max_soft_tokens {s_max} × pks² {pks}²; \
             the preprocessor ties the two budgets together"
        );
        ensure!(pks >= 1, "gemma vision: pooling_kernel_size must be >= 1");
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        ensure!(cfg.patch_size >= 1, "gemma vision: patch_size must be >= 1");
        ensure!(cfg.rope_theta > 0.0, "gemma vision: rope_theta must be > 0");

        // ── Scratch: one image's caps (see module docs) ──
        // buf_h1/buf_h2 stage the pixel upload: they must hold the wider of
        // `[p, patch_dim]` (pixels in) and `[p, hidden]` (hidden out) — equal
        // on the shipped tower (both 768), patch_dim wins on small tests.
        let bf16 = |n: usize| -> Result<DevicePtr> { gpu.alloc(n * 2) };
        let stage_w = patch_dim.max(hidden);
        let buf_h1 = bf16(p_max * stage_w)?;
        let buf_h2 = bf16(p_max * stage_w)?;
        let buf_pos = bf16(p_max * hidden)?;
        let buf_qkv = bf16(p_max * (3 * hidden))?;
        let buf_wide = bf16(p_max * 2 * inter)?; // gate plane + up plane
        let buf_mlp = bf16(p_max * hidden)?;
        let buf_pool = bf16(s_max * hidden)?;
        let buf_rope_cos = bf16(p_max * head_dim)?;
        let buf_rope_sin = bf16(p_max * head_dim)?;
        let buf_out = bf16(s_max * OUT_HIDDEN_SIZE)?;
        let norm_unit_w = bf16(hidden)?;
        // V-norm (HF Gemma4VisionAttention.v_norm, with_scale=False) — ones
        // over head_dim, applied to v after the v_proj GEMM. The checkpoint
        // ships no v_norm weight (pure `x·rms`, same convention as the
        // embed_vision norm).
        let head_norm_unit_w = bf16(head_dim)?;
        {
            let ones: Vec<u16> =
                std::iter::repeat_n(super::f32_to_bf16_bits(1.0), head_dim).collect();
            gpu.copy_h2d(
                // SAFETY: `ones` is a live `vec![u16; head_dim]`; byte length
                // derived from that same Vec; u16 has no invalid bit patterns.
                unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 2) },
                head_norm_unit_w,
            )?;
        }

        // ── Kernel handles ──
        // Reused (shape-compatible) kernels: hard where the gemma-4-e2b text
        // stack already hard-requires them (boot audit), soft otherwise.
        let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let k_rms_norm = gpu.kernel("norm", "rms_norm")?;
        let k_gelu_mul = crate::layers::try_kernel(gpu, "gelu", "gelu_mul");
        let k_add = crate::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");
        // QK-Norm: generic rms_norm is RMSNorm over dim 64 — shape-compatible
        // per head; Wave 3 points this handle at a dedicated per-head kernel.
        let k_qk_norm = crate::layers::try_kernel(gpu, "norm", "rms_norm");
        // Gemma-specific kernels — Wave 3 adds them to the gemma-4-e2b tree
        // under the `gemma_vision` module. Null today → launch_optional no-ops.
        let k_rope_rotate =
            crate::layers::try_kernel(gpu, "gemma_vision", "gemma_vision_rope_rotate");
        let k_attn = crate::layers::try_kernel(gpu, "gemma_vision", "gemma_vision_attention");
        let k_clamp = crate::layers::try_kernel(gpu, "gemma_vision", "gemma_vision_clamp");
        let k_pool = crate::layers::try_kernel(gpu, "gemma_vision", "gemma_vision_pool");

        // ── Host-side prep state ──
        // Position table [2, pos_size, hidden] BF16 → host, for the per-image
        // x_emb + y_emb gather (same host-side idiom as Qwen3-VL's
        // interpolated pos_embed; ~31 MB on the shipped tower, one-time).
        let pos_n = 2 * cfg.position_embedding_size * hidden;
        let mut table_bytes = vec![0u8; pos_n * 2];
        gpu.copy_d2h(w.position_table.weight, &mut table_bytes)?;
        let position_table_host: Vec<u16> = table_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        // RoPE inverse frequencies (Qwen3-VL host-precompute idiom):
        // RoPE inverse frequencies — HF Gemma4VisionRotaryEmbedding:
        // spatial_dim = head_dim/2 (=32); inv_freq = 1/θ^(arrange(0,
        // spatial_dim, 2)/spatial_dim) — ONE frequency per 2 channels, i.e.
        // head_dim/4 (=16) distinct frequencies (not head_dim/2 — the
        // original formula here used all 32 and was off by the even-step).
        let spatial_dim = head_dim / 2;
        let theta = cfg.rope_theta;
        let rope_inv_freq: Vec<f32> = (0..spatial_dim)
            .step_by(2)
            .map(|k| 1.0 / theta.powf(k as f32 / spatial_dim as f32))
            .collect();

        // embed_vision RMSNorm(with_scale=False): ones weight (the checkpoint
        // ships no norm weight for it — pure `x·rms`, Gemma-4 convention).
        let ones: Vec<u16> = std::iter::repeat_n(super::f32_to_bf16_bits(1.0), hidden).collect();
        gpu.copy_h2d(
            // SAFETY: `ones` is a live `vec![u16; hidden]`; the byte length is
            // derived from that same Vec, so the view stays in-bounds. u16 has
            // no invalid bit patterns and u8 has alignment 1.
            unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 2) },
            norm_unit_w,
        )?;

        Ok(GemmaVisionEncoder {
            input_proj_w: w.input_proj.weight,
            position_table: w.position_table.weight,
            layers: w.layers.clone(),
            embed_vision_proj_w: w.embed_vision_projection.weight,
            k_gemm,
            k_rms_norm,
            k_gelu_mul,
            k_add,
            k_qk_norm,
            k_rope_rotate,
            k_attn,
            k_clamp,
            k_pool,
            position_table_host,
            rope_inv_freq,
            buf_h1,
            buf_h2,
            buf_pos,
            buf_qkv,
            buf_wide,
            buf_mlp,
            buf_pool,
            buf_rope_cos,
            buf_rope_sin,
            buf_out,
            norm_unit_w,
            head_norm_unit_w,
            hidden_size: hidden,
            num_heads: heads,
            head_dim,
            intermediate_size: inter,
            pooling_kernel_size: pks,
            p_max,
            s_max,
            position_embedding_size: cfg.position_embedding_size,
            norm_eps: cfg.norm_eps,
            rope_theta: theta,
            patch_dim,
            total_soft_tokens: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}
