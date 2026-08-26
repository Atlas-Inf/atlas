// SPDX-License-Identifier: AGPL-3.0-only

//! Shared fixtures for the gemma media tests (`gemma_media_tests.rs` /
//! `gemma_audio_tests.rs`): synthetic gemma vision/audio configs, encoders,
//! weights and a `TransformerModel` on the `MockGpuBackend`. Split out so
//! every test file stays under the ≤500-LoC cap.

use crate::layers::gemma_audio_encoder::{
    GemmaAudioAttnWeights, GemmaAudioEncoder, GemmaAudioFfnWeights, GemmaAudioLayerWeights,
    GemmaAudioLightConvWeights, GemmaAudioOutputProj, GemmaAudioSubsampleWeights,
    GemmaAudioWeights,
};
use crate::layers::gemma_vision_encoder::{
    ClipLinearWeights, GemmaVisionEncoder, GemmaVisionLayerWeights, GemmaVisionWeights,
    OUT_HIDDEN_SIZE,
};
use crate::media::gemma_audio::GemmaAudioInput;
use crate::media::gemma_vision::GemmaImageInput;
use crate::model::types::TransformerModel;
use crate::weight_map::DenseWeight;
use atlas_core::config::{GemmaAudioConfig, GemmaVisionConfig, ModelConfig};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;

// Real Gemma-4 E2B slot ids (per the checkpoint's config.json).
pub(crate) const IMAGE_SLOT: u32 = 258_880;
pub(crate) const AUDIO_SLOT: u32 = 258_881;
pub(crate) const VIDEO_SLOT: u32 = 258_884;
// Qwen3-VL image-pad id — must NOT be matched by the gemma splice.
pub(crate) const QWEN_PAD: u32 = 151_655;

// ── test fixtures ──────────────────────────────────────────────────────

/// Synthetic gemma vision geometry mirroring the encoder's shape relations
/// (heads×head_dim == hidden; max_patches == max_soft_tokens × pks²) at tiny
/// sizes, with the REAL gemma-4 E2B slot ids so the splice matching runs
/// against production token ids.
pub(crate) fn gemma_vision_cfg() -> GemmaVisionConfig {
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
        image_token_id: IMAGE_SLOT,
        rope_theta: 100.0,
        max_patches: 252, // 28 soft tokens × 9
        max_soft_tokens: 28,
        position_table_shape: (2, 64, 12),
        norm_eps: 1e-6,
        video_frames: 16,
        video_soft_tokens_per_frame: 128,
        video_token_id: VIDEO_SLOT,
        boi_token_id: 255_999,
        eoi_token_id: 258_882,
    }
}

pub(crate) fn alloc_weight(gpu: &MockGpuBackend, elems: usize) -> DenseWeight {
    let ptr = gpu.alloc(elems * 2).unwrap();
    DenseWeight { weight: ptr }
}

pub(crate) fn clip(gpu: &MockGpuBackend, out: usize, inp: usize) -> ClipLinearWeights {
    ClipLinearWeights {
        weight: alloc_weight(gpu, out * inp),
        input_min: -10.0,
        input_max: 10.0,
        output_min: -30.0,
        output_max: 30.0,
    }
}

pub(crate) fn vision_weights(gpu: &MockGpuBackend, cfg: &GemmaVisionConfig) -> GemmaVisionWeights {
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

pub(crate) fn build_encoder(gpu: &MockGpuBackend, cfg: &GemmaVisionConfig) -> GemmaVisionEncoder {
    let w = vision_weights(gpu, cfg);
    GemmaVisionEncoder::new(&w, cfg, gpu).unwrap()
}

/// Synthetic audio-tower weights at the test geometry. `relative_k_proj`
/// must be a REAL allocation — `GemmaAudioEncoder::new` downloads it to
/// precompute the relative position keys.
pub(crate) fn audio_weights(gpu: &MockGpuBackend, cfg: &GemmaAudioConfig) -> GemmaAudioWeights {
    let h = cfg.hidden_size;
    let head_dim = h / cfg.num_attention_heads;
    let ffn = |gpu: &MockGpuBackend| GemmaAudioFfnWeights {
        ffw_layer_1: clip(gpu, 4 * h, h),
        ffw_layer_2: clip(gpu, h, 4 * h),
        pre_layer_norm: alloc_weight(gpu, h),
        post_layer_norm: alloc_weight(gpu, h),
    };
    let layer = |gpu: &MockGpuBackend| GemmaAudioLayerWeights {
        feed_forward1: ffn(gpu),
        feed_forward2: ffn(gpu),
        lconv1d: GemmaAudioLightConvWeights {
            linear_start: clip(gpu, 2 * h, h),
            linear_end: clip(gpu, h, h),
            depthwise_conv1d: alloc_weight(gpu, h * 5),
            pre_layer_norm: alloc_weight(gpu, h),
            conv_norm: alloc_weight(gpu, h),
        },
        self_attn: GemmaAudioAttnWeights {
            q_proj: clip(gpu, h, h),
            k_proj: clip(gpu, h, h),
            v_proj: clip(gpu, h, h),
            post: clip(gpu, h, h),
            relative_k_proj: DenseWeight {
                weight: gpu.alloc(h * h * 2).unwrap(),
            },
            per_dim_scale: alloc_weight(gpu, head_dim),
        },
        norm_pre_attn: alloc_weight(gpu, h),
        norm_post_attn: alloc_weight(gpu, h),
        norm_out: alloc_weight(gpu, h),
    };
    let c = &cfg.subsampling_conv_channels;
    GemmaAudioWeights {
        subsample: GemmaAudioSubsampleWeights {
            conv0: alloc_weight(gpu, c[0] * 9),
            ln0: alloc_weight(gpu, c[0]),
            conv1: alloc_weight(gpu, c[1] * c[0] * 9),
            ln1: alloc_weight(gpu, c[1]),
            input_proj_linear: alloc_weight(gpu, h * h),
        },
        layers: (0..cfg.num_hidden_layers).map(|_| layer(gpu)).collect(),
        output_proj: GemmaAudioOutputProj {
            weight: alloc_weight(gpu, 1536 * h),
            bias: alloc_weight(gpu, 1536),
        },
        embed_audio_projection: alloc_weight(gpu, 1536 * 1536),
    }
}

pub(crate) fn build_audio_encoder(
    gpu: &MockGpuBackend,
    cfg: &GemmaAudioConfig,
) -> GemmaAudioEncoder {
    let w = audio_weights(gpu, cfg);
    GemmaAudioEncoder::new(&w, cfg, gpu).unwrap()
}

/// A synthetic mel-frontended clip: `n_frames` valid all-ones frames.
pub(crate) fn audio_clip(cfg: &GemmaAudioConfig, n_frames: usize) -> GemmaAudioInput {
    GemmaAudioInput {
        features: vec![0.0f32; n_frames * cfg.mel_bins],
        n_frames,
        n_mels: cfg.mel_bins,
        mask: vec![1u8; n_frames],
    }
}

/// Same as `write_buf_out_rows` for the AUDIO encoder's buf_out (row width
/// is also OUT_HIDDEN_SIZE × 2 — the audio tower projects into the text
/// embedding width).
pub(crate) fn write_audio_buf_out_rows(
    gpu: &dyn GpuBackend,
    enc: &GemmaAudioEncoder,
    nrows: usize,
) {
    let row_bytes = crate::layers::gemma_audio_encoder::OUT_HIDDEN_SIZE * 2;
    for r in 0..nrows {
        let row = vec![0x50u8 + r as u8; row_bytes];
        gpu.copy_h2d(&row, enc.buf_out().offset(r * row_bytes))
            .unwrap();
    }
}

pub(crate) fn gemma_audio_cfg() -> GemmaAudioConfig {
    GemmaAudioConfig {
        hidden_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        // Real E2B geometry relations (two conv stages, mel % 4 == 0, heads
        // divide hidden) — `GemmaAudioEncoder::new` validates these.
        subsampling_conv_channels: vec![8, 4],
        conv_kernel_size: 3,
        attention_chunk_size: 32,
        attention_context_left: 4,
        attention_context_right: 4,
        // Baked by the encoder: output_proj must land on OUT_HIDDEN_SIZE.
        output_proj_dims: 1536,
        residual_weight: 0.1,
        use_clipped_linears: true,
        audio_token_id: AUDIO_SLOT,
        mel_bins: 8,
        frame_length: 400,
        hop_length: 160,
        fft_size: 512,
        mel_floor: 1.0,
        mel_scale: "htk".to_string(),
        token_cap: 256,
        norm_eps: 1e-6,
        activation: "silu".to_string(),
        boa_token_id: 256_000,
        eoa_token_id: 258_883,
    }
}

/// A synthetic preprocessed gemma image: `grid_h × grid_w` patches (zero
/// pixels — the mock GEMMs no-op anyway). pks=3 so a 6×6 grid → 4 soft
/// tokens and a 3×3 grid → 1.
pub(crate) fn image(grid_h: usize, grid_w: usize, cfg: &GemmaVisionConfig) -> GemmaImageInput {
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

/// Build a `TransformerModel` on the mock backend with the gemma configs +
/// optional gemma vision/audio encoders installed and the Qwen vision
/// encoder absent (a gemma serve). `gpu` is moved into the model; reads
/// afterwards go through `model.gpu_backend()`.
pub(crate) fn build_model(
    gpu: MockGpuBackend,
    gemma_enc: Option<GemmaVisionEncoder>,
    audio_enc: Option<GemmaAudioEncoder>,
) -> TransformerModel {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.gemma_vision = Some(gemma_vision_cfg());
    config.gemma_audio = Some(gemma_audio_cfg());
    let buffers = spark_runtime::buffers::BufferArena::new(&config, 32, 256, 16, 1, &gpu).unwrap();
    let kv_config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_attention_layers(),
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        layer_to_pool: vec![],
        cache_blocks_per_seq: None,
    };
    let kv_cache = PagedKvCache::new(kv_config, 4, &gpu).unwrap();
    let prefix_cache: Box<dyn PrefixCache> = Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
    let embed = DenseWeight {
        weight: gpu.alloc(8).unwrap(),
    };
    let final_norm = DenseWeight {
        weight: gpu.alloc(8).unwrap(),
    };
    let lm_head = DenseWeight {
        weight: gpu.alloc(8).unwrap(),
    };
    TransformerModel::new(
        config,
        embed,
        final_norm,
        lm_head,
        None,   // lm_head_nvfp4
        None,   // lm_head_fp8
        None,   // mtp_lm_head_nvfp4
        vec![], // layers (unused on the mock embed path)
        buffers,
        kv_cache,
        vec![], // mtp_weights
        Box::new(gpu),
        256, // max_seq_len
        1,   // max_batch_size
        crate::layers::mtp_head::MtpQuantization::Bf16,
        false, // use_speculative
        prefix_cache,
        0,         // mtp_vocab_size
        None,      // comm
        false,     // self_speculative
        1,         // num_drafts
        None,      // vision_encoder (Qwen)
        gemma_enc, // gemma_vision_encoder
        audio_enc, // gemma_audio_encoder
        0,         // ssm_cache_slots
        0,         // ssm_checkpoint_interval
        None,      // ple_tables
    )
    .unwrap()
}

/// Write `nrows` distinct one-byte-fill rows into the encoder's buf_out
/// (row `r` = byte `0x10 + r`). Row width = OUT_HIDDEN_SIZE × 2 BF16 bytes.
pub(crate) fn write_buf_out_rows(gpu: &dyn GpuBackend, enc: &GemmaVisionEncoder, nrows: usize) {
    let row_bytes = OUT_HIDDEN_SIZE * 2;
    for r in 0..nrows {
        let row = vec![0x10u8 + r as u8; row_bytes];
        gpu.copy_h2d(&row, enc.buf_out().offset(r * row_bytes))
            .unwrap();
    }
}

/// Read the first `OUT_HIDDEN_SIZE × 2` bytes of hidden row `i` (row stride
/// = config.hidden_size × 2).
pub(crate) fn read_hidden_row(model: &TransformerModel, hidden: DevicePtr, i: usize) -> Vec<u8> {
    let mut buf = vec![0u8; OUT_HIDDEN_SIZE * 2];
    let stride = model.config.hidden_size * 2;
    model
        .gpu_backend()
        .copy_d2h(hidden.offset(i * stride), &mut buf)
        .unwrap();
    buf
}
