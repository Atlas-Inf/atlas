// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A swap must CARRY the process-scoped stores, never rebuild them.
///
/// Asserted by POINTER identity, not equality: two freshly-built empty stores
/// compare equal, so an equality check would pass on exactly the bug this
/// guards — a swap that silently drops every stored conversation and resets
/// every rate-limit bucket while looking fine.
#[test]
fn carried_state_is_the_same_allocation_not_an_equal_one() {
    let first = Carried::from_env();
    let cloned = first.clone();

    assert!(
        std::sync::Arc::ptr_eq(&first.response_store, &cloned.response_store),
        "responses must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.rate_limiter, &cloned.rate_limiter),
        "rate-limit buckets must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.conversation_store, &cloned.conversation_store),
        "stored conversations must survive a swap"
    );
}

/// Two independent `from_env()` calls are DIFFERENT allocations — which is why
/// `load_model` takes `Carried` rather than building its own.
#[test]
fn building_from_env_twice_would_lose_the_stores() {
    let first = Carried::from_env();
    let second = Carried::from_env();
    assert!(
        !std::sync::Arc::ptr_eq(&first.conversation_store, &second.conversation_store),
        "if this ever passes, from_env has become a singleton and the carried \
         parameter is no longer what protects the stores — re-check the swap"
    );
}

// ── Gemma-4 E2B media-config fallback (Wave 0) ────────────────────────
//
// `demote_unsupported_media_towers` must mirror the Qwen3-VL
// `vision_encoder` text-only fallback for gemma's own towers: a
// checkpoint that declares `gemma_vision`/`gemma_audio` but loads a
// kernel target without the `gemma_vision`/`gemma_audio`
// PTX modules serves TEXT-ONLY (config dropped) instead of failing at
// the first encoder lookup.

fn gemma_vision_config() -> atlas_core::config::GemmaVisionConfig {
    atlas_core::config::GemmaVisionConfig {
        hidden_size: 1152,
        intermediate_size: 4304,
        num_hidden_layers: 27,
        num_attention_heads: 16,
        head_dim: 72,
        patch_size: 16,
        pooling_kernel_size: 2,
        position_embedding_size: 1024,
        use_clipped_linears: false,
        image_token_id: 262144,
        rope_theta: 100.0,
        max_patches: 1120,
        max_soft_tokens: 280,
        position_table_shape: (2, 1024, 1152),
        norm_eps: 1e-6,
        video_frames: 32,
        video_soft_tokens_per_frame: 70,
        video_token_id: 262146,
        boi_token_id: 262147,
        eoi_token_id: 262148,
    }
}

fn gemma_audio_config() -> atlas_core::config::GemmaAudioConfig {
    atlas_core::config::GemmaAudioConfig {
        hidden_size: 1024,
        num_hidden_layers: 12,
        num_attention_heads: 16,
        subsampling_conv_channels: vec![256, 128],
        conv_kernel_size: 7,
        attention_chunk_size: 32,
        attention_context_left: 16,
        attention_context_right: 16,
        output_proj_dims: 2048,
        residual_weight: 0.1,
        use_clipped_linears: false,
        audio_token_id: 262145,
        mel_bins: 128,
        frame_length: 320,
        hop_length: 160,
        fft_size: 512,
        mel_floor: 1e-3,
        mel_scale: "htk".to_string(),
        token_cap: 750,
        norm_eps: 1e-6,
        activation: "silu".to_string(),
        boa_token_id: 262149,
        eoa_token_id: 262150,
    }
}

#[test]
fn gemma_media_configs_survive_when_target_ships_encoders() {
    let mut vision: Option<atlas_core::config::VisionConfig> = None;
    let mut gemma_vision = Some(gemma_vision_config());
    let mut gemma_audio = Some(gemma_audio_config());
    // Full gemma-4-e2b multimodal target: both gemma encoder modules ship.
    let modules: &[(&'static str, &'static [u8])] = &[
        ("vision_encoder", &b"ptx"[..]),
        ("gemma_vision", &b"ptx"[..]),
        ("gemma_audio", &b"ptx"[..]),
    ];
    demote_unsupported_media_towers(
        &mut vision,
        &mut gemma_vision,
        &mut gemma_audio,
        modules,
        "gemma-4-e2b",
    );
    assert!(
        gemma_vision.is_some(),
        "gemma vision tower survives when its module ships"
    );
    assert!(
        gemma_audio.is_some(),
        "gemma audio tower survives when its module ships"
    );
}

#[test]
fn gemma_media_configs_dropped_when_target_ships_no_encoders() {
    let mut vision: Option<atlas_core::config::VisionConfig> = None;
    let mut gemma_vision = Some(gemma_vision_config());
    let mut gemma_audio = Some(gemma_audio_config());
    // Text-only gemma-4-e2b target: no media encoder modules at all.
    let modules: &[(&'static str, &'static [u8])] = &[("gemma_encoder", &b"ptx"[..])];
    demote_unsupported_media_towers(
        &mut vision,
        &mut gemma_vision,
        &mut gemma_audio,
        modules,
        "gemma-4-e2b",
    );
    assert!(
        gemma_vision.is_none(),
        "gemma vision tower dropped on a text-only target"
    );
    assert!(
        gemma_audio.is_none(),
        "gemma audio tower dropped on a text-only target"
    );
}

#[test]
fn qwen_vision_unaffected_by_gemma_fallback() {
    let mut vision = Some(atlas_core::config::VisionConfig {
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4304,
        out_hidden_size: 2048,
        deepstack_visual_indexes: vec![8, 16, 24],
        image_pad_token_id: 151655,
        max_pixels: None,
        video_pad_token_id: 151656,
    });
    let mut gemma_vision = Some(gemma_vision_config());
    let mut gemma_audio = Some(gemma_audio_config());
    // Qwen3-VL-style target ships only vision_encoder — gemma towers drop,
    // the Qwen tower must survive untouched.
    let modules: &[(&'static str, &'static [u8])] = &[("vision_encoder", &b"ptx"[..])];
    demote_unsupported_media_towers(
        &mut vision,
        &mut gemma_vision,
        &mut gemma_audio,
        modules,
        "qwen3-vl",
    );
    assert!(
        vision.is_some(),
        "qwen vision tower untouched by the gemma fallback"
    );
    assert!(
        gemma_vision.is_none(),
        "gemma vision dropped on a qwen-style target"
    );
    assert!(
        gemma_audio.is_none(),
        "gemma audio dropped on a qwen-style target"
    );
}

#[test]
fn carried_uses_the_process_limiter_rather_than_minting_its_own() {
    // Handlers refund through `AppState.rate_limiter` and the middleware debits
    // through the host's. If those are two instances, refunds credit buckets
    // the middleware never debited and the accounting silently drifts.
    let host = crate::main_modules::model_host::ModelHost::empty();
    let carried = Carried::from_env();
    let process = carried.rate_limiter.clone();
    host.set_process(carried);
    assert!(
        std::sync::Arc::ptr_eq(&host.rate_limiter().expect("installed"), &process),
        "the host's limiter IS the one the model's AppState will hold"
    );
}
