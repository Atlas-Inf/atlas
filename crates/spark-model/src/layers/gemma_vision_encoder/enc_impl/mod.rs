// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GemmaVisionEncoder` body, split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `GemmaVisionEncoder` inherent
//! impl.
//!
//! - `init`        — `new()` constructor (buffers, kernel handles, host tables)
//! - `pos`         — `build_rope_cossin_into`, `gather_pos_emb_into`
//! - `patch_embed` — `patch_embed_batched` (pixel upload + input_proj + pos add)
//! - `qk_norm`     — `qk_norm_inplace` (QK-Norm on q/k slices)
//! - `attention`   — `attention_stage` (Wave-3 MHA stub dispatch)
//! - `mlp`         — `mlp_stage` (GeGLU with clipped linears)
//! - `pool`        — `pool_stage` (average-pool → soft tokens)
//! - `project`     — `embed_vision_project` (RMSNorm + 768→1536 projection)
//! - `forward`     — top-level `forward_batched` + oversized fallback

mod attention;
mod forward;
mod init;
mod mlp;
mod patch_embed;
mod pool;
mod pos;
mod project;
mod qk_norm;

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Convert an f32 to BF16 bits using round-to-nearest-even. Same helper the
/// Qwen3-VL encoder uses for its host-side pos/rope tables.
#[inline]
pub(super) fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN → canonical quiet NaN in BF16.
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// Launch a Wave-3 stub/optional kernel under the documented soft-fallback
/// contract: a NULL handle (kernel not yet in the target's tree) skips the
/// launch as a debug-logged no-op, so Wave-2 orchestration runs shape- and
/// order-correct on every target while the real PTX lands in Wave 3. The
/// MockGpuBackend returns a non-null handle for every lookup, so tests still
/// observe every launch.
pub(super) fn launch_optional(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    name: &str,
    stream: u64,
    build: impl FnOnce(KernelLaunch<'_>) -> KernelLaunch<'_>,
) -> Result<()> {
    if kernel.0 == 0 {
        tracing::debug!("gemma vision: {name} kernel absent — no-op stub (Wave 3 fills)");
        return Ok(());
    }
    build(KernelLaunch::new(gpu, kernel)).launch(stream)
}

#[cfg(test)]
mod tests {
    use super::super::{
        ClipLinearWeights, GemmaVisionEncoder, GemmaVisionLayerWeights, GemmaVisionWeights,
        OUT_HIDDEN_SIZE,
    };
    use atlas_core::config::GemmaVisionConfig;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use crate::media::gemma_vision::GemmaImageInput;
    use crate::weight_map::DenseWeight;

    /// Synthetic vision geometry mirroring the real tower's SHAPE relations
    /// (heads×head_dim == hidden; max_patches == max_soft_tokens × pks²) at a
    /// size that keeps test allocations trivial: hidden 12, 3 heads × 4,
    /// 2 layers, pks 3 (so a 6×6 grid → 4 soft tokens, 3×3 → 1).
    fn test_cfg() -> GemmaVisionConfig {
        GemmaVisionConfig {
            hidden_size: 12,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 3,
            head_dim: 4,
            patch_size: 4,
            pooling_kernel_size: 3,
            position_embedding_size: 64,
            use_clipped_linears: true,
            image_token_id: 261_022,
            rope_theta: 100.0,
            max_patches: 252, // 28 soft tokens × 9
            max_soft_tokens: 28,
            position_table_shape: (2, 64, 12),
            norm_eps: 1e-6,
            video_frames: 16,
            video_soft_tokens_per_frame: 128,
            video_token_id: 261_023,
            boi_token_id: 261_024,
            eoi_token_id: 261_025,
        }
    }

    /// Allocate a weight buffer on the mock and return its pointer (zeroed).
    /// The position table is READ BACK at init, so it (like every weight)
    /// must be a real allocation.
    fn alloc_weight(gpu: &MockGpuBackend, elems: usize) -> DenseWeight {
        let ptr = gpu.alloc(elems * 2).unwrap();
        gpu.copy_h2d(&vec![0u8; elems * 2], ptr).unwrap();
        DenseWeight { weight: ptr }
    }

    fn clip(gpu: &MockGpuBackend, out: usize, inp: usize) -> ClipLinearWeights {
        ClipLinearWeights {
            weight: alloc_weight(gpu, out * inp),
            input_min: -10.0,
            input_max: 10.0,
            output_min: -30.0,
            output_max: 30.0,
        }
    }

    fn test_weights(gpu: &MockGpuBackend, cfg: &GemmaVisionConfig) -> GemmaVisionWeights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let layer = |gpu: &MockGpuBackend| GemmaVisionLayerWeights {
            input_layernorm: alloc_weight(gpu, h),
            q_norm: alloc_weight(gpu, cfg.head_dim),
            k_norm: alloc_weight(gpu, cfg.head_dim),
            q_proj: clip(gpu, h, h),
            k_proj: clip(gpu, h, h),
            v_proj: clip(gpu, h, h),
            o_proj: clip(gpu, h, h),
            post_attention_layernorm: alloc_weight(gpu, h),
            pre_feedforward_layernorm: alloc_weight(gpu, h),
            gate_proj: clip(gpu, i, h),
            up_proj: clip(gpu, i, h),
            down_proj: clip(gpu, h, i),
            post_feedforward_layernorm: alloc_weight(gpu, h),
        };
        GemmaVisionWeights {
            input_proj: alloc_weight(gpu, h * h),
            position_table: alloc_weight(gpu, 2 * cfg.position_embedding_size * h),
            layers: (0..cfg.num_hidden_layers).map(|_| layer(gpu)).collect(),
            embed_vision_projection: alloc_weight(gpu, OUT_HIDDEN_SIZE * h),
        }
    }

    fn image(grid_h: usize, grid_w: usize, cfg: &GemmaVisionConfig) -> GemmaImageInput {
        let p = grid_h * grid_w;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let mut pos_ids = Vec::with_capacity(p);
        for y in 0..grid_h {
            for x in 0..grid_w {
                pos_ids.push((x as i32, y as i32));
            }
        }
        GemmaImageInput {
            pixels: vec![0.0f32; p * patch_dim],
            grid_h,
            grid_w,
            pos_ids,
            soft_token_count: p / (cfg.pooling_kernel_size * cfg.pooling_kernel_size),
        }
    }

    fn build(gpu: &MockGpuBackend) -> (GemmaVisionEncoder, GemmaVisionConfig) {
        let cfg = test_cfg();
        let w = test_weights(gpu, &cfg);
        let enc = GemmaVisionEncoder::new(&w, &cfg, gpu).unwrap();
        (enc, cfg)
    }

    /// ORCHESTRATION CONTRACT: `forward_batched` over 2 synthetic images
    /// (6×6 grid → 36 patches → 4 soft tokens; 3×3 grid → 9 patches → 1 soft
    /// token) returns `[4, 1]` in image order, packs the final 1536-wide
    /// projections into `buf_out` with capacity ≥ 5×1536 elements, and runs
    /// the whole kernel sequence without panicking under mock kernels (the
    /// Wave-3 stubs no-op).
    #[test]
    fn forward_batched_returns_soft_counts_and_packs_buf_out() {
        let gpu = MockGpuBackend::new();
        let (enc, cfg) = build(&gpu);
        let imgs = [image(6, 6, &cfg), image(3, 3, &cfg)];
        let counts = enc.forward_batched(&imgs, &gpu, 0).unwrap();
        assert_eq!(counts, vec![4, 1]);
        assert_eq!(enc.total_soft_tokens(), 5);
        // buf_out holds [Σsoft, OUT_HIDDEN_SIZE] BF16, image-order packed:
        // capacity must cover all 5 soft-token rows.
        let out_bytes = gpu.read_alloc(enc.buf_out()).unwrap().len();
        assert!(
            out_bytes >= 5 * OUT_HIDDEN_SIZE * 2,
            "buf_out capacity {out_bytes} bytes < 5×{OUT_HIDDEN_SIZE}×2"
        );
        // The orchestration actually issued kernels (stub launches recorded).
        assert!(gpu.launch_count() > 0, "forward_batched launched nothing");
    }

    /// ACCOUNTING CONTRACT: `total_soft_tokens()` equals the sum of the
    /// per-image counts, is 0 before any forward, and tracks a re-run.
    #[test]
    fn total_soft_tokens_accounts_the_batch() {
        let gpu = MockGpuBackend::new();
        let (enc, cfg) = build(&gpu);
        assert_eq!(enc.total_soft_tokens(), 0);
        enc.forward_batched(&[image(6, 6, &cfg)], &gpu, 0).unwrap();
        assert_eq!(enc.total_soft_tokens(), 4);
        enc.forward_batched(&[image(6, 6, &cfg), image(6, 6, &cfg)], &gpu, 0)
            .unwrap();
        assert_eq!(enc.total_soft_tokens(), 8);
    }

    /// CONFIG-GEOMETRY CONTRACT: the encoder is built from a
    /// `GemmaVisionConfig`; the constructor enforces heads×head_dim ==
    /// hidden_size, layer-count and position-table consistency, and sizes the
    /// scratch buffers to `max_patches` / `max_soft_tokens`.
    #[test]
    fn config_geometry_is_enforced_and_sizes_buffers() {
        let gpu = MockGpuBackend::new();
        let cfg = test_cfg();
        assert_eq!(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);
        let w = test_weights(&gpu, &cfg);
        let enc = GemmaVisionEncoder::new(&w, &cfg, &gpu).unwrap();
        assert_eq!(enc.num_heads * enc.head_dim, enc.hidden_size);
        assert_eq!(enc.layers.len(), cfg.num_hidden_layers);
        assert_eq!(enc.position_embedding_size, cfg.position_embedding_size);
        // Position table dims: (2, position_embedding_size, hidden).
        assert_eq!(cfg.position_table_shape.1, cfg.position_embedding_size);
        assert_eq!(cfg.position_table_shape.2, cfg.hidden_size);
        // Scratch sized to one image's caps. buf_h1/buf_h2 stage the pixel
        // upload too, so they hold the wider of patch_dim (pixels in) and
        // hidden (hidden out) — equal on the shipped tower (768).
        let stage_w = (3 * cfg.patch_size * cfg.patch_size).max(cfg.hidden_size);
        assert_eq!(
            gpu.read_alloc(enc.buf_h1).unwrap().len(),
            cfg.max_patches * stage_w * 2
        );
        assert_eq!(
            gpu.read_alloc(enc.buf_out).unwrap().len(),
            cfg.max_soft_tokens * OUT_HIDDEN_SIZE * 2
        );

        // A config where heads×head_dim != hidden_size must be REFUSED.
        let mut bad = test_cfg();
        bad.head_dim = 5;
        assert!(GemmaVisionEncoder::new(&w, &bad, &gpu).is_err());
        // A layer-count mismatch must be REFUSED.
        let mut bad = test_cfg();
        bad.num_hidden_layers = 3;
        assert!(GemmaVisionEncoder::new(&w, &bad, &gpu).is_err());
    }

    /// CLIP-BOUND CONTRACT: the four scalar bounds ride with each
    /// ClippableLinear weight and survive the trip into the encoder.
    #[test]
    fn clip_bounds_ride_with_the_linear_weights() {
        let gpu = MockGpuBackend::new();
        let (enc, _cfg) = build(&gpu);
        let l = &enc.layers[0];
        assert_eq!(l.q_proj.input_min, -10.0);
        assert_eq!(l.q_proj.input_max, 10.0);
        assert_eq!(l.q_proj.output_min, -30.0);
        assert_eq!(l.q_proj.output_max, 30.0);
        assert!(!l.q_proj.weight.weight.is_null());
        assert_eq!(enc.layers[1].gate_proj.input_min, -10.0);
    }
}
