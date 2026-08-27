// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GemmaAudioEncoder` body, split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `GemmaAudioEncoder` inherent
//! impl.
//!
//! - `init`        — `new()` constructor (buffers, kernel handles, host prep)
//! - `subsample`   — `subsample_stage` (mel upload + conv projection)
//! - `ffn`         — `ffn_sub_block` + `clipped_gemm` (SiLU FFN with clipped
//!   linears and the residual_weight blend)
//! - `chunked_attn`— attention block (norm → qkv → scale → chunked attn →
//!   post → norm → residual)
//! - `conv1d`      — `light_conv_sub_block` (GLU + depthwise causal conv)
//! - `conformer`   — `conformer_layer_batched` (the 12-layer driver)
//! - `project`     — `embed_audio_project` (output_proj + bias + RMSNorm +
//!   1536→1536)
//! - `forward`     — top-level `forward_batched` + oversized fallback
//!
//! # Wave-4C kernel contracts
//!
//! The gemma-specific kernels below are declared as soft handles
//! (`try_kernel` → null → `launch_optional` no-ops, debug-logged). The
//! argument ORDER at each launch site IS the contract (verified against
//! the PTX in the gemma-4-e2b tree's `gemma_audio` module):
//!
//! - `gemma_audio_subsample_conv1(features, mask, out1, conv0_w, ln0_w,
//!   t_mel, t1, eps)` — features `[t_mel × mel]` BF16 row-major; mask
//!   `[t_mel]` u8 (1 = valid, zeroed at load); conv weight `[c0, 1, 3, 3]`
//!   (biasless, stride 2, pad 1); LayerNorm weight `[c0]`; out1
//!   `[t1 × mel/2 × c0]` BF16 (freq-major row). Math: `x ×= mask` →
//!   conv0 → LN0 → ReLU. Grid `(t1,1,1)`, block `(1024,1,1)`.
//! - `gemma_audio_subsample_conv2(in1, out, conv1_w, ln1_w, t1, t2, eps)`
//!   — reads the `[t1 × mel/2 × c0]` intermediate with the exact conv2
//!   window (rows `2·t4−1..2·t4+1` from the global buffer — the fused
//!   single kernel staged only 2 of the 3 slices it read, an OOB smem
//!   fault), → conv1 → LN1 → ReLU → flattened `[t2 × flatten]` BF16,
//!   row `[freq × channel]`. Grid `(t2,1,1)`, block `(1024,1,1)`.
//! - `gemma_audio_chunked_attn(q, k, v, spd, rel_k, mask, out, seq, heads,
//!   head_dim, softcap, invalid_logits, chunk, ctx_left, ctx_right)` —
//!   q/k/v `[seq × heads×head_dim]` RAW (the kernel applies q_scale and
//!   k_scale internally), `spd [head_dim]` = softplus(per_dim_scale)
//!   (host-precomputed per layer at init), `rel_k [context/2+1 ×
//!   heads×head_dim]`, blocked mask `[nblocks × chunk × context]` u8
//!   (1 = attend, 0 → `invalid_logits`); tanh softcap `softcap`,
//!   per-block windows per HF `Gemma4AudioAttention` (relative-key path
//!   with the transformer-xl `_rel_shift`). Grid `(nblocks,1,1)`, block
//!   `(chunk×32,1,1)`.
//! - `gemma_audio_conv1d(x, dw, out, seq, hidden, kernel, in_stride)` —
//!   depthwise CAUSAL conv, left pad `kernel−1`: `out[t][c] = Σ_k dw[c][k] ×
//!   x[t−(kernel−1)+k][c]` for `0 ≤ t−(kernel−1)+k ≤ t`. The input rows are
//!   the row-interleaved GLU output inside `buf_wide` (stride `2×hidden`).
//!   Grid `(ceil(hidden/256),1,1)`.
//! - `gemma_audio_silu(x, n)` — `x[i] = x[i]·σ(x[i])` in place.
//! - `gemma_audio_bias_add(out, bias, rows, cols)` — `out[r·cols+c] +=
//!   bias[c]`; grid `(ceil(rows·cols/256),1,1)`.
//!
//! The ClippableLinear clamps reuse the VISION handle
//! (`gemma_vision::gemma_vision_clamp`, Wave 3) with the identical
//! `(buf, lo, hi, n)` contract.

mod chunked_attn;
mod conformer;
mod conv1d;
mod ffn;
mod forward;
mod init;
mod project;
mod subsample;

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Convert an f32 to BF16 bits using round-to-nearest-even. Same helper the
/// Gemma vision encoder uses for its host-side uploads.
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

/// Launch a Wave-4C stub/optional kernel under the documented soft-fallback
/// contract: a NULL handle (kernel not yet in the target's tree) skips the
/// launch as a debug-logged no-op, so Wave-4A orchestration runs shape- and
/// order-correct on every target while the real PTX lands in Wave 4C. The
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
        tracing::debug!("gemma audio: {name} kernel absent — no-op stub (Wave 4C fills)");
        return Ok(());
    }
    build(KernelLaunch::new(gpu, kernel)).launch(stream)
}

#[cfg(test)]
mod tests {
    use super::super::{
        GemmaAudioEncoder, GemmaAudioLayerWeights, GemmaAudioLightConvWeights,
        GemmaAudioSubsampleWeights, GemmaAudioWeights, OUT_HIDDEN_SIZE,
    };
    use crate::layers::gemma_vision_encoder::ClipLinearWeights;
    use crate::media::gemma_audio::{GemmaAudioInput, subsample_conv_len};
    use crate::weight_map::DenseWeight;
    use atlas_core::config::GemmaAudioConfig;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// Synthetic audio geometry mirroring the real tower's SHAPE relations
    /// (heads×head_dim == hidden; flatten == hidden; mel_bins/4 × last_conv
    /// channels == flatten) at sizes that keep test allocations trivial:
    /// hidden 16, 4 heads × 4, 2 layers, mel 16 → flatten 4×4 = 16.
    fn test_cfg() -> GemmaAudioConfig {
        GemmaAudioConfig {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            subsampling_conv_channels: vec![8, 4],
            conv_kernel_size: 5,
            attention_chunk_size: 4,
            attention_context_left: 3,
            attention_context_right: 0,
            output_proj_dims: OUT_HIDDEN_SIZE,
            residual_weight: 0.5,
            use_clipped_linears: true,
            audio_token_id: 258_881,
            mel_bins: 16,
            frame_length: 320,
            hop_length: 160,
            fft_size: 512,
            mel_floor: 1e-3,
            mel_scale: "htk".to_string(),
            token_cap: 8,
            norm_eps: 1e-6,
            activation: "silu".to_string(),
            boa_token_id: 256_000,
            eoa_token_id: 258_883,
        }
    }

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

    fn ffn(
        gpu: &MockGpuBackend,
        h: usize,
    ) -> crate::layers::gemma_audio_encoder::GemmaAudioFfnWeights {
        crate::layers::gemma_audio_encoder::GemmaAudioFfnWeights {
            ffw_layer_1: clip(gpu, 4 * h, h),
            ffw_layer_2: clip(gpu, h, 4 * h),
            pre_layer_norm: alloc_weight(gpu, h),
            post_layer_norm: alloc_weight(gpu, h),
        }
    }

    fn lconv(gpu: &MockGpuBackend, h: usize) -> GemmaAudioLightConvWeights {
        GemmaAudioLightConvWeights {
            linear_start: clip(gpu, 2 * h, h),
            linear_end: clip(gpu, h, h),
            depthwise_conv1d: alloc_weight(gpu, h * 5),
            pre_layer_norm: alloc_weight(gpu, h),
            conv_norm: alloc_weight(gpu, h),
        }
    }

    fn layer(gpu: &MockGpuBackend, h: usize, hd: usize) -> GemmaAudioLayerWeights {
        GemmaAudioLayerWeights {
            feed_forward1: ffn(gpu, h),
            feed_forward2: ffn(gpu, h),
            lconv1d: lconv(gpu, h),
            self_attn: crate::layers::gemma_audio_encoder::GemmaAudioAttnWeights {
                q_proj: clip(gpu, h, h),
                k_proj: clip(gpu, h, h),
                v_proj: clip(gpu, h, h),
                post: clip(gpu, h, h),
                relative_k_proj: alloc_weight(gpu, h * h),
                per_dim_scale: alloc_weight(gpu, hd),
            },
            norm_pre_attn: alloc_weight(gpu, h),
            norm_post_attn: alloc_weight(gpu, h),
            norm_out: alloc_weight(gpu, h),
        }
    }

    fn test_weights(gpu: &MockGpuBackend, cfg: &GemmaAudioConfig) -> GemmaAudioWeights {
        let h = cfg.hidden_size;
        let [c0, c1] = [
            cfg.subsampling_conv_channels[0],
            cfg.subsampling_conv_channels[1],
        ];
        GemmaAudioWeights {
            subsample: GemmaAudioSubsampleWeights {
                conv0: alloc_weight(gpu, c0 * 3 * 3),
                ln0: alloc_weight(gpu, c0),
                conv1: alloc_weight(gpu, c1 * c0 * 3 * 3),
                ln1: alloc_weight(gpu, c1),
                input_proj_linear: alloc_weight(gpu, h * (cfg.mel_bins / 4 * c1)),
            },
            layers: (0..cfg.num_hidden_layers)
                .map(|_| layer(gpu, h, h / cfg.num_attention_heads))
                .collect(),
            output_proj: crate::layers::gemma_audio_encoder::GemmaAudioOutputProj {
                weight: alloc_weight(gpu, OUT_HIDDEN_SIZE * h),
                bias: alloc_weight(gpu, OUT_HIDDEN_SIZE),
            },
            embed_audio_projection: alloc_weight(gpu, OUT_HIDDEN_SIZE * OUT_HIDDEN_SIZE),
        }
    }

    /// A fully-valid synthetic clip: `t` mel frames → `subsample_conv_len(t)`
    /// tokens after the 4× subsampling.
    fn audio_clip(t: usize, cfg: &GemmaAudioConfig) -> GemmaAudioInput {
        GemmaAudioInput {
            features: vec![0.0f32; t * cfg.mel_bins],
            n_frames: t,
            n_mels: cfg.mel_bins,
            mask: vec![1u8; t],
        }
    }

    fn build(gpu: &MockGpuBackend) -> (GemmaAudioEncoder, GemmaAudioConfig) {
        let cfg = test_cfg();
        let w = test_weights(gpu, &cfg);
        let enc = GemmaAudioEncoder::new(&w, &cfg, gpu).unwrap();
        (enc, cfg)
    }

    /// ORCHESTRATION CONTRACT: `forward_batched` over 2 synthetic clips
    /// (8 mel frames → 2 tokens; 4 mel frames → 1 token after the 4×
    /// subsampling) returns `[2, 1]` in clip order, packs the final 1536-wide
    /// projections into `buf_out` with capacity ≥ 3×1536 elements, and runs
    /// the whole kernel sequence without panicking under mock kernels (the
    /// Wave-4C stubs no-op).
    #[test]
    fn forward_batched_returns_token_counts_and_packs_buf_out() {
        let gpu = MockGpuBackend::new();
        let (enc, cfg) = build(&gpu);
        assert_eq!(subsample_conv_len(8), 2);
        assert_eq!(subsample_conv_len(4), 1);
        let clips = [audio_clip(8, &cfg), audio_clip(4, &cfg)];
        let counts = enc.forward_batched(&clips, &gpu, 0).unwrap();
        assert_eq!(counts, vec![2, 1]);
        assert_eq!(enc.total_soft_tokens(), 3);
        // buf_out holds [Σvalid, OUT_HIDDEN_SIZE] BF16, clip-order packed:
        // capacity must cover all 3 token rows.
        let out_bytes = gpu.read_alloc(enc.buf_out()).unwrap().len();
        assert!(
            out_bytes >= 3 * OUT_HIDDEN_SIZE * 2,
            "buf_out capacity {out_bytes} bytes < 3×{OUT_HIDDEN_SIZE}×2"
        );
        // The orchestration actually issued kernels (stub launches recorded).
        assert!(gpu.launch_count() > 0, "forward_batched launched nothing");
    }

    /// ACCOUNTING CONTRACT: `total_soft_tokens()` equals the sum of the
    /// per-clip counts, is 0 before any forward, and tracks a re-run.
    #[test]
    fn total_soft_tokens_accounts_the_batch() {
        let gpu = MockGpuBackend::new();
        let (enc, cfg) = build(&gpu);
        assert_eq!(enc.total_soft_tokens(), 0);
        enc.forward_batched(&[audio_clip(8, &cfg)], &gpu, 0)
            .unwrap();
        assert_eq!(enc.total_soft_tokens(), 2);
        enc.forward_batched(&[audio_clip(8, &cfg), audio_clip(4, &cfg)], &gpu, 0)
            .unwrap();
        assert_eq!(enc.total_soft_tokens(), 3);
    }

    /// MASK CONTRACT: padding frames are dropped from the token count —
    /// the valid count is the popcount of the subsampled mask, and
    /// `buf_out` packs only those rows.
    #[test]
    fn mask_drops_padding_tokens_from_the_count() {
        let gpu = MockGpuBackend::new();
        let (enc, cfg) = build(&gpu);
        // 8 frames; mask[4i] = {1, 0} → 1 valid token.
        let mut c = audio_clip(8, &cfg);
        c.mask = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let counts = enc.forward_batched(&[c], &gpu, 0).unwrap();
        assert_eq!(counts, vec![1]);
        assert_eq!(enc.total_soft_tokens(), 1);
        let out_bytes = gpu.read_alloc(enc.buf_out()).unwrap().len();
        assert!(out_bytes >= OUT_HIDDEN_SIZE * 2);
    }

    /// CONFIG-GEOMETRY CONTRACT: the constructor enforces heads×head_dim ==
    /// hidden, layer count, the two conv stages, `mel_bins % 4 == 0` and
    /// `output_proj_dims == OUT_HIDDEN_SIZE`, and sizes the scratch buffers
    /// to `token_cap` rows.
    #[test]
    fn config_geometry_is_enforced_and_sizes_buffers() {
        let gpu = MockGpuBackend::new();
        let cfg = test_cfg();
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(cfg.num_attention_heads * hd, cfg.hidden_size);
        let w = test_weights(&gpu, &cfg);
        let enc = GemmaAudioEncoder::new(&w, &cfg, &gpu).unwrap();
        assert_eq!(enc.num_heads * enc.head_dim, enc.hidden_size);
        assert_eq!(enc.layers.len(), cfg.num_hidden_layers);
        assert_eq!(enc.t_max, cfg.token_cap);
        assert_eq!(
            enc.flatten_dim,
            (cfg.mel_bins / 4) * cfg.subsampling_conv_channels[1]
        );
        assert_eq!(enc.context_size, 4 + 2); // chunk + (left−1), right = 0
        // Scratch sized to one clip's caps: buf_out rows × 1536 × 2.
        assert_eq!(
            gpu.read_alloc(enc.buf_out).unwrap().len(),
            cfg.token_cap * OUT_HIDDEN_SIZE * 2
        );
        // buf_mel holds the 4× mel budget.
        assert_eq!(
            gpu.read_alloc(enc.buf_mel).unwrap().len(),
            4 * cfg.token_cap * cfg.mel_bins * 2
        );

        // A config where heads×head_dim != hidden must be REFUSED.
        let mut bad = test_cfg();
        bad.num_attention_heads = 5;
        assert!(GemmaAudioEncoder::new(&w, &bad, &gpu).is_err());
        // A layer-count mismatch must be REFUSED.
        let mut bad = test_cfg();
        bad.num_hidden_layers = 3;
        assert!(GemmaAudioEncoder::new(&w, &bad, &gpu).is_err());
        // output_proj_dims other than OUT_HIDDEN_SIZE must be REFUSED.
        let mut bad = test_cfg();
        bad.output_proj_dims = 1024;
        assert!(GemmaAudioEncoder::new(&w, &bad, &gpu).is_err());
        // mel_bins not divisible by 4 must be REFUSED.
        let mut bad = test_cfg();
        bad.mel_bins = 18;
        assert!(GemmaAudioEncoder::new(&w, &bad, &gpu).is_err());
    }

    /// CLIP-BOUND CONTRACT: the four scalar bounds ride with each
    /// ClippableLinear weight and survive the trip into the encoder.
    #[test]
    fn clip_bounds_ride_with_the_linear_weights() {
        let gpu = MockGpuBackend::new();
        let (enc, _cfg) = build(&gpu);
        let l = &enc.layers[0];
        assert_eq!(l.self_attn.q_proj.input_min, -10.0);
        assert_eq!(l.self_attn.q_proj.input_max, 10.0);
        assert_eq!(l.self_attn.q_proj.output_min, -30.0);
        assert_eq!(l.self_attn.q_proj.output_max, 30.0);
        assert_eq!(enc.layers[1].lconv1d.linear_end.input_min, -10.0);
        assert_eq!(enc.layers[0].feed_forward1.ffw_layer_2.output_max, 30.0);
        assert!(!l.self_attn.relative_k_proj.weight.is_null());
    }
}
