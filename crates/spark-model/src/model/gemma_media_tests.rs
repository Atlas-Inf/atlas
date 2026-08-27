// SPDX-License-Identifier: AGPL-3.0-only

//! Wave 2C tests — Gemma-4 E2B vision encoder wiring in the model forward
//! path: `prepare_gemma_media_embed` dispatch, the embed-splice row-consume
//! order, the audio-slot guard, the pending-patch gate, prefix-cache
//! suppression, and Qwen-path non-interference. All on a `MockGpuBackend`
//! (no CUDA on the host). Fixtures live in `gemma_media_fixtures`.

use super::gemma_media_fixtures::{
    AUDIO_SLOT, IMAGE_SLOT, QWEN_PAD, VIDEO_SLOT, build_encoder, build_model, gemma_vision_cfg,
    image, read_hidden_row, write_buf_out_rows,
};
use crate::layers::gemma_vision_encoder::OUT_HIDDEN_SIZE;
use crate::traits::Model;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

// ── tests ──────────────────────────────────────────────────────────────

/// PREPARE DISPATCH: `prepare_gemma_media_embed` with 2 images (6×6 → 4 soft
/// tokens, 3×3 → 1) stages Σpatches = 5 and per-image counts [4, 1], and
/// no-ops (stages nothing) when no gemma encoder is installed.
#[test]
fn prepare_gemma_media_embed_stages_patch_counts() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let enc = build_encoder(&gpu, &cfg);
    let model = build_model(gpu, Some(enc), None);

    let imgs = [image(6, 6, &cfg), image(3, 3, &cfg)];
    model.prepare_gemma_media_embed(&imgs).unwrap();

    assert_eq!(*model.gemma_vision_embed_patches.lock(), 5);
    assert_eq!(*model.gemma_vision_soft_counts.lock(), vec![4, 1]);
}

#[test]
fn prepare_gemma_media_embed_noops_without_encoder() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let model = build_model(gpu, None, None);

    let imgs = [image(6, 6, &cfg), image(3, 3, &cfg)];
    model.prepare_gemma_media_embed(&imgs).unwrap();

    assert_eq!(*model.gemma_vision_embed_patches.lock(), 0);
    assert!(model.gemma_vision_soft_counts.lock().is_empty());
}

/// SPLICE ORDER: a stream `[7, IMAGE, IMAGE, VIDEO, 9, IMAGE, 11]` consumes
/// buf_out rows 0,1,2,3 in encounter order — image AND video slots draw from
/// the SAME vision buf_out — and leaves text rows untouched.
#[test]
fn gemma_splice_consumes_rows_in_encounter_order() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let enc = build_encoder(&gpu, &cfg);
    write_buf_out_rows(&gpu, &enc, 5);
    let model = build_model(gpu, Some(enc), None);

    // Arm the splice without running forward_batched (which would overwrite
    // buf_out with mock-zeroed output).
    *model.gemma_vision_embed_patches.lock() = 5;

    let tokens: Vec<u32> = vec![7, IMAGE_SLOT, IMAGE_SLOT, VIDEO_SLOT, 9, IMAGE_SLOT, 11];
    let hidden = model.buffers.hidden_states();
    model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap();

    // Slot positions 1, 2, 3, 5 carry buf_out rows 0, 1, 2, 3.
    for (pos, row) in [(1usize, 0usize), (2, 1), (3, 2), (5, 3)] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0x10u8 + row as u8; OUT_HIDDEN_SIZE * 2],
            "hidden row {pos} should carry buf_out row {row}"
        );
    }
    // Text rows (0, 4, 6) are untouched (embed kernel no-ops on the mock →
    // all zeros).
    for pos in [0usize, 4, 6] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0u8; OUT_HIDDEN_SIZE * 2],
            "text row {pos} must not be spliced"
        );
    }
}

/// AUDIO GUARD: an audio slot token (258881) in the stream fails loudly —
/// no audio encoder exists until Wave 4 — rather than silently splicing a
/// zero row. A preceding image slot consumes row 0 before the error.
#[test]
fn gemma_audio_slot_errors_without_audio_encoder() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let enc = build_encoder(&gpu, &cfg);
    write_buf_out_rows(&gpu, &enc, 5);
    let model = build_model(gpu, Some(enc), None);
    *model.gemma_vision_embed_patches.lock() = 5;

    let tokens: Vec<u32> = vec![7, IMAGE_SLOT, AUDIO_SLOT, 9];
    let hidden = model.buffers.hidden_states();
    let err = model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("audio") && msg.contains(&AUDIO_SLOT.to_string()),
        "audio-slot error should name the token and the missing encoder, got: {msg}"
    );
}

/// GATE: with zero pending patches (no prepare call) the splice is a no-op
/// even when slot tokens are present — gemma image slots fall back to their
/// vocab embeddings rather than reading stale buf_out rows.
#[test]
fn gemma_splice_gated_on_pending_patches() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let enc = build_encoder(&gpu, &cfg);
    write_buf_out_rows(&gpu, &enc, 5);
    let model = build_model(gpu, Some(enc), None);
    // NOTE: gemma_vision_embed_patches deliberately left at 0.

    let tokens: Vec<u32> = vec![7, IMAGE_SLOT, VIDEO_SLOT];
    let hidden = model.buffers.hidden_states();
    model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap();

    for pos in 0..tokens.len() {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0u8; OUT_HIDDEN_SIZE * 2],
            "row {pos} must stay untouched when no gemma patches are pending"
        );
    }
}

/// QWEN NON-INTERFERENCE: the gemma splice matches ONLY gemma slot ids —
/// a Qwen3-VL image-pad token (151655) in the same stream is left to the
/// (absent here) Qwen vision encoder.
#[test]
fn gemma_splice_ignores_qwen_pad_token() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_vision_cfg();
    let enc = build_encoder(&gpu, &cfg);
    write_buf_out_rows(&gpu, &enc, 5);
    let model = build_model(gpu, Some(enc), None);
    *model.gemma_vision_embed_patches.lock() = 5;

    let tokens: Vec<u32> = vec![QWEN_PAD, IMAGE_SLOT];
    let hidden = model.buffers.hidden_states();
    model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap();

    // The Qwen pad position is untouched; the gemma image slot consumes row 0.
    assert_eq!(
        read_hidden_row(&model, hidden, 0),
        vec![0u8; OUT_HIDDEN_SIZE * 2],
        "Qwen pad token must not be spliced by the gemma path"
    );
    assert_eq!(
        read_hidden_row(&model, hidden, 1),
        vec![0x10u8; OUT_HIDDEN_SIZE * 2],
        "gemma image slot should consume buf_out row 0"
    );
}

/// PREFIX CACHE: `tokens_have_vision_pad` is true for every gemma media slot
/// id (image, video, audio) and false for plain text — media prompts keep
/// the prefix cache disabled.
#[test]
fn tokens_have_vision_pad_covers_gemma_slot_ids() {
    let gpu = MockGpuBackend::new();
    let model = build_model(gpu, None, None);

    assert!(model.tokens_have_vision_pad(&[IMAGE_SLOT]));
    assert!(model.tokens_have_vision_pad(&[VIDEO_SLOT]));
    assert!(model.tokens_have_vision_pad(&[AUDIO_SLOT]));
    assert!(model.tokens_have_vision_pad(&[7, IMAGE_SLOT, 9]));
    assert!(!model.tokens_have_vision_pad(&[7, 8, 9]));
    assert!(model.tokens_have_vision_pad(&[QWEN_PAD])); // Qwen path preserved
}
