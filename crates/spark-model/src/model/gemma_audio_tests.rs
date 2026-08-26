// SPDX-License-Identifier: AGPL-3.0-only

//! Wave 4B tests — Gemma-4 E2B audio encoder wiring in the model forward
//! path: `prepare_gemma_audio_embed` dispatch and the audio-slot embed
//! splice (own buf_out, encounter-order consumption, right-buffer selection
//! against image/video slots). All on a `MockGpuBackend` (no CUDA on the
//! host). Split from `gemma_media_tests` for the ≤500-LoC cap; shared
//! fixtures live there (`gemma_media_tests::*`).

use super::gemma_media_fixtures::{
    AUDIO_SLOT, IMAGE_SLOT, VIDEO_SLOT, audio_clip, build_audio_encoder, build_encoder,
    build_model, gemma_audio_cfg, gemma_vision_cfg, read_hidden_row, write_audio_buf_out_rows,
    write_buf_out_rows,
};
use crate::layers::gemma_vision_encoder::OUT_HIDDEN_SIZE;
use crate::traits::Model;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

/// AUDIO PREPARE: `prepare_gemma_audio_embed` with 2 clips (8 frames → 2
/// valid tokens, 4 frames → 1) stages Σvalid = 3 and per-clip counts
/// [2, 1]; no-ops (stages nothing) when no audio encoder is installed.
#[test]
fn prepare_gemma_audio_embed_stages_counts() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_audio_cfg();
    let enc = build_audio_encoder(&gpu, &cfg);
    let model = build_model(gpu, None, Some(enc));

    let clips = [audio_clip(&cfg, 8), audio_clip(&cfg, 4)];
    model.prepare_gemma_audio_embed(&clips).unwrap();

    assert_eq!(*model.gemma_audio_embed_patches.lock(), 3);
    assert_eq!(*model.gemma_audio_soft_counts.lock(), vec![2, 1]);
}

#[test]
fn prepare_gemma_audio_embed_noops_without_encoder() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_audio_cfg();
    let model = build_model(gpu, None, None);

    let clips = [audio_clip(&cfg, 8)];
    model.prepare_gemma_audio_embed(&clips).unwrap();

    assert_eq!(*model.gemma_audio_embed_patches.lock(), 0);
    assert!(model.gemma_audio_soft_counts.lock().is_empty());
}

/// AUDIO SPLICE: a stream `[7, AUDIO, AUDIO, 9, AUDIO, 11]` consumes the
/// audio encoder's buf_out rows 0, 1, 2 in encounter order — the audio
/// tower has its OWN buf_out (unlike image/video, which share the vision
/// one) — and leaves text rows untouched.
#[test]
fn gemma_audio_splice_consumes_rows_in_encounter_order() {
    let gpu = MockGpuBackend::new();
    let cfg = gemma_audio_cfg();
    let enc = build_audio_encoder(&gpu, &cfg);
    write_audio_buf_out_rows(&gpu, &enc, 4);
    let model = build_model(gpu, None, Some(enc));

    // Arm the audio splice without running forward_batched (which would
    // overwrite buf_out with mock-zeroed output).
    *model.gemma_audio_embed_patches.lock() = 3;

    let tokens: Vec<u32> = vec![7, AUDIO_SLOT, AUDIO_SLOT, 9, AUDIO_SLOT, 11];
    let hidden = model.buffers.hidden_states();
    model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap();

    for (pos, row) in [(1usize, 0usize), (2, 1), (4, 2)] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0x50u8 + row as u8; OUT_HIDDEN_SIZE * 2],
            "hidden row {pos} should carry audio buf_out row {row}"
        );
    }
    for pos in [0usize, 3, 5] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0u8; OUT_HIDDEN_SIZE * 2],
            "text row {pos} must not be spliced"
        );
    }
}

/// MIXED MEDIA: one stream with image + audio + video slots consumes from
/// the RIGHT buffers — image/video draw the vision buf_out (rows 0,1,2 at
/// positions 0,2,4), audio draws the audio buf_out (rows 0,1 at positions
/// 1,3) — each with its own independent row counter.
#[test]
fn gemma_mixed_media_splice_uses_right_buffers() {
    let gpu = MockGpuBackend::new();
    let vcfg = gemma_vision_cfg();
    let venc = build_encoder(&gpu, &vcfg);
    write_buf_out_rows(&gpu, &venc, 5);
    let acfg = gemma_audio_cfg();
    let aenc = build_audio_encoder(&gpu, &acfg);
    write_audio_buf_out_rows(&gpu, &aenc, 4);
    let model = build_model(gpu, Some(venc), Some(aenc));

    *model.gemma_vision_embed_patches.lock() = 5;
    *model.gemma_audio_embed_patches.lock() = 3;

    let tokens: Vec<u32> = vec![IMAGE_SLOT, AUDIO_SLOT, VIDEO_SLOT, AUDIO_SLOT, IMAGE_SLOT];
    let hidden = model.buffers.hidden_states();
    model
        .prefill_b_embed_chunk_at(&tokens, 0, tokens.len(), hidden, 0)
        .unwrap();

    // Vision rows (fill 0x10+r) land on image/video slots...
    for (pos, row) in [(0usize, 0usize), (2, 1), (4, 2)] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0x10u8 + row as u8; OUT_HIDDEN_SIZE * 2],
            "image/video slot {pos} should carry vision buf_out row {row}"
        );
    }
    // ...audio rows (fill 0x50+r) land on audio slots, independent counter.
    for (pos, row) in [(1usize, 0usize), (3, 1)] {
        assert_eq!(
            read_hidden_row(&model, hidden, pos),
            vec![0x50u8 + row as u8; OUT_HIDDEN_SIZE * 2],
            "audio slot {pos} should carry audio buf_out row {row}"
        );
    }
}
