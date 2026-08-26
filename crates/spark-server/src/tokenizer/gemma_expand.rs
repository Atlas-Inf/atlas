// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B multimodal placeholder expansion (Wave 1).
//!
//! The Gemma-4 E2B chat template renders exactly ONE placeholder token per
//! media part: `<|image|>` (258880), `<|audio|>` (258881), `<|video|>`
//! (258884). The processor expands each placeholder into its full boundary-
//! token sequence BEFORE the model sees the stream — this module is that
//! expansion, as a pure function.
//!
//! Expansion rules (ground truth, verified against the E2B processor):
//! * IMAGE: `<|image|>` → `<|image>` (255999) + `<|image|>` × N + `<image|>`
//!   (258882), N = soft-token count for that image (e.g. 256 for 768×768).
//! * AUDIO: `<|audio|>` → `<|audio>` (256000) + `<|audio|>` × N + `<audio|>`
//!   (258883), N = audio token count (from mel, cap 750).
//! * VIDEO: `<|video|>` → per frame: tokenize("MM:SS") + `<|image>` (255999)
//!   + `<|video|>` × N + `<image|>` (258882), N = soft tokens per frame (70),
//!   frame i timestamp "00:00" … "00:31" (zero-padded MM:SS).
//!
//! The placeholder ids mirror `atlas_core::config`: `GemmaVisionConfig`
//! (image_token_id / video_token_id / boi_token_id / eoi_token_id) and
//! `GemmaAudioConfig` (audio_token_id / boa_token_id / eoa_token_id).

/// `<|image|>` — one per image in the rendered stream.
pub const IMAGE_PLACEHOLDER_TOKEN_ID: u32 = 258_880;
/// `<|audio|>` — one per audio clip in the rendered stream.
pub const AUDIO_PLACEHOLDER_TOKEN_ID: u32 = 258_881;
/// `<|video|>` — one per video clip in the rendered stream.
pub const VIDEO_PLACEHOLDER_TOKEN_ID: u32 = 258_884;
/// `<|image>` — begin-of-image boundary token.
pub const BOI_TOKEN_ID: u32 = 255_999;
/// `<image|>` — end-of-image boundary token.
pub const EOI_TOKEN_ID: u32 = 258_882;
/// `<|audio>` — begin-of-audio boundary token.
pub const BOA_TOKEN_ID: u32 = 256_000;
/// `<audio|>` — end-of-audio boundary token.
pub const EOA_TOKEN_ID: u32 = 258_883;

/// Per-media soft-token counts driving the placeholder expansion.
///
/// `image_counts` / `audio_counts` are consumed in encounter order: the i-th
/// `<|image|>` placeholder in the stream expands with `image_counts[i]`, the
/// i-th `<|audio|>` with `audio_counts[i]`. A placeholder with no remaining
/// count expands to its bare boundary pair (0 soft tokens); counts beyond the
/// number of placeholders are ignored — never an error.
///
/// Video parameters are model-level processor constants shared by every
/// `<|video|>` placeholder in the stream (32 frames × 70 soft tokens for
/// Gemma-4 E2B); `video_timestamps` toggles the per-frame "MM:SS" splice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaMediaCounts {
    /// Soft tokens per image, in encounter order (from preprocessing).
    pub image_counts: Vec<usize>,
    /// Token counts per audio clip, in encounter order (from mel).
    pub audio_counts: Vec<usize>,
    /// Frames per video clip (32 for Gemma-4 E2B).
    pub video_frames: usize,
    /// Soft tokens per video frame (70 for Gemma-4 E2B).
    pub video_soft_tokens_per_frame: usize,
    /// Splice a tokenized "MM:SS" frame timestamp before each video frame.
    pub video_timestamps: bool,
}

/// Format a video frame index as the zero-padded "MM:SS" timestamp the E2B
/// processor splices before each frame's boundary tokens. Minutes roll over
/// at 60 frames: frame 31 → "00:31", frame 61 → "01:01".
fn format_video_timestamp(frame: usize) -> String {
    format!("{:02}:{:02}", frame / 60, frame % 60)
}

/// Expand Gemma-4 E2B multimodal placeholders into their full boundary-token
/// sequences. Pure: returns a fresh token stream, consuming counts in
/// encounter order. `tokenize_ts` converts a "MM:SS" timestamp string into
/// token ids (typically `|s| tokenizer.encode(s)`); it is only invoked when
/// `counts.video_timestamps` is set.
///
/// Behavior on missing/extra counts (never errors):
/// * A placeholder with no remaining count expands to its bare boundary pair.
/// * Counts beyond the placeholders in the stream are ignored.
/// * `video_frames == 0` drops the placeholder entirely.
pub fn expand_gemma_multimodal(
    tokens: &[u32],
    counts: &GemmaMediaCounts,
    tokenize_ts: impl Fn(&str) -> Vec<u32>,
) -> Vec<u32> {
    // Count placeholders per modality for the preallocation budget.
    let (n_img, n_aud, n_vid) =
        tokens
            .iter()
            .fold((0usize, 0usize, 0usize), |(i, a, v), &t| match t {
                IMAGE_PLACEHOLDER_TOKEN_ID => (i + 1, a, v),
                AUDIO_PLACEHOLDER_TOKEN_ID => (i, a + 1, v),
                VIDEO_PLACEHOLDER_TOKEN_ID => (i, a, v + 1),
                _ => (i, a, v),
            });
    if n_img + n_aud + n_vid == 0 {
        return tokens.to_vec();
    }
    // +8 per video frame: generous upper bound for a 5-char "MM:SS" encode
    // (an underestimate only costs a reallocation, never correctness).
    let ts_budget = if counts.video_timestamps { 8 } else { 0 };
    let cap = tokens.len()
        + n_img * 2
        + counts.image_counts.iter().sum::<usize>()
        + n_aud * 2
        + counts.audio_counts.iter().sum::<usize>()
        + n_vid * counts.video_frames * (2 + counts.video_soft_tokens_per_frame + ts_budget);
    let mut out = Vec::with_capacity(cap);
    let mut img_idx = 0usize;
    let mut aud_idx = 0usize;
    for &t in tokens {
        match t {
            IMAGE_PLACEHOLDER_TOKEN_ID => {
                out.push(BOI_TOKEN_ID);
                let n = counts.image_counts.get(img_idx).copied().unwrap_or(0);
                out.extend(std::iter::repeat_n(IMAGE_PLACEHOLDER_TOKEN_ID, n));
                out.push(EOI_TOKEN_ID);
                img_idx += 1;
            }
            AUDIO_PLACEHOLDER_TOKEN_ID => {
                out.push(BOA_TOKEN_ID);
                let n = counts.audio_counts.get(aud_idx).copied().unwrap_or(0);
                out.extend(std::iter::repeat_n(AUDIO_PLACEHOLDER_TOKEN_ID, n));
                out.push(EOA_TOKEN_ID);
                aud_idx += 1;
            }
            VIDEO_PLACEHOLDER_TOKEN_ID => {
                for frame in 0..counts.video_frames {
                    if counts.video_timestamps {
                        out.extend(tokenize_ts(&format_video_timestamp(frame)));
                    }
                    out.push(BOI_TOKEN_ID);
                    out.extend(std::iter::repeat_n(
                        VIDEO_PLACEHOLDER_TOKEN_ID,
                        counts.video_soft_tokens_per_frame,
                    ));
                    out.push(EOI_TOKEN_ID);
                }
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stand-in for the model tokenizer: each "MM:SS"
    /// timestamp maps to one id per byte ("00:00" → [48, 48, 58, 48, 48]),
    /// so tests can assert the exact timestamp STRING that reached the
    /// expansion and the order the frames were spliced in.
    fn byte_ts(ts: &str) -> Vec<u32> {
        ts.bytes().map(u32::from).collect()
    }

    fn counts(
        images: &[usize],
        audios: &[usize],
        frames: usize,
        soft: usize,
        ts: bool,
    ) -> GemmaMediaCounts {
        GemmaMediaCounts {
            image_counts: images.to_vec(),
            audio_counts: audios.to_vec(),
            video_frames: frames,
            video_soft_tokens_per_frame: soft,
            video_timestamps: ts,
        }
    }

    #[test]
    fn image_only_expands_to_boi_plus_n_soft_tokens_plus_eoi() {
        let tokens = vec![7, IMAGE_PLACEHOLDER_TOKEN_ID, 8];
        let out = expand_gemma_multimodal(&tokens, &counts(&[256], &[], 0, 0, false), byte_ts);
        // 3 input tokens − 1 placeholder + boi/eoi + 256 soft = 260
        assert_eq!(out.len(), 4 + 256);
        assert_eq!(out[0], 7);
        assert_eq!(out[1], BOI_TOKEN_ID);
        assert!(
            out[2..2 + 256]
                .iter()
                .all(|&t| t == IMAGE_PLACEHOLDER_TOKEN_ID),
            "expected 256 soft <|image|> tokens"
        );
        assert_eq!(out[2 + 256], EOI_TOKEN_ID);
        assert_eq!(out[3 + 256], 8);
    }

    #[test]
    fn image_missing_count_expands_to_bare_boundaries() {
        let out = expand_gemma_multimodal(
            &[IMAGE_PLACEHOLDER_TOKEN_ID],
            &counts(&[], &[], 0, 0, false),
            byte_ts,
        );
        assert_eq!(out, vec![BOI_TOKEN_ID, EOI_TOKEN_ID]);
    }

    #[test]
    fn extra_counts_beyond_placeholders_are_ignored() {
        let tokens = vec![IMAGE_PLACEHOLDER_TOKEN_ID];
        let out = expand_gemma_multimodal(&tokens, &counts(&[4, 99], &[], 0, 0, false), byte_ts);
        let expected = vec![
            BOI_TOKEN_ID,
            IMAGE_PLACEHOLDER_TOKEN_ID,
            IMAGE_PLACEHOLDER_TOKEN_ID,
            IMAGE_PLACEHOLDER_TOKEN_ID,
            IMAGE_PLACEHOLDER_TOKEN_ID,
            EOI_TOKEN_ID,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn audio_only_expands_to_boa_plus_n_soft_tokens_plus_eoa() {
        let tokens = vec![AUDIO_PLACEHOLDER_TOKEN_ID];
        let out = expand_gemma_multimodal(&tokens, &counts(&[], &[750], 0, 0, false), byte_ts);
        assert_eq!(out.len(), 750 + 2);
        assert_eq!(out[0], BOA_TOKEN_ID);
        assert!(
            out[1..1 + 750]
                .iter()
                .all(|&t| t == AUDIO_PLACEHOLDER_TOKEN_ID),
            "expected 750 soft <|audio|> tokens"
        );
        assert_eq!(out[1 + 750], EOA_TOKEN_ID);
    }

    #[test]
    fn video_only_expands_32_frames_with_ordered_timestamps() {
        let tokens = vec![9, VIDEO_PLACEHOLDER_TOKEN_ID, 10];
        let out = expand_gemma_multimodal(&tokens, &counts(&[], &[], 32, 70, true), byte_ts);
        // Per frame: 5 timestamp ids ("MM:SS" = 5 bytes) + boi + 70 soft + eoi.
        let frame_len = 5 + 1 + 70 + 1;
        assert_eq!(out.len(), 2 + 32 * frame_len);
        assert_eq!(out[0], 9);
        assert_eq!(out[out.len() - 1], 10);
        let mut pos = 1;
        for frame in 0..32 {
            let expected_ts = format!("{:02}:{:02}", frame / 60, frame % 60);
            assert_eq!(
                &out[pos..pos + 5],
                byte_ts(&expected_ts).as_slice(),
                "frame {frame} timestamp must precede its boundary tokens"
            );
            assert_eq!(out[pos + 5], BOI_TOKEN_ID, "frame {frame} boi");
            assert!(
                out[pos + 6..pos + 6 + 70]
                    .iter()
                    .all(|&t| t == VIDEO_PLACEHOLDER_TOKEN_ID),
                "frame {frame} soft tokens"
            );
            assert_eq!(out[pos + 6 + 70], EOI_TOKEN_ID, "frame {frame} eoi");
            pos += frame_len;
        }
        // Frame 0 is "00:00", frame 31 is "00:31".
        assert_eq!(&out[1..1 + 5], byte_ts("00:00").as_slice());
        assert_eq!(
            &out[1 + 31 * frame_len..1 + 31 * frame_len + 5],
            byte_ts("00:31").as_slice()
        );
    }

    #[test]
    fn video_timestamps_disabled_omits_timestamp_ids() {
        let tokens = vec![VIDEO_PLACEHOLDER_TOKEN_ID];
        let out = expand_gemma_multimodal(&tokens, &counts(&[], &[], 2, 3, false), byte_ts);
        // 2 frames × (boi + 3 soft + eoi), no timestamp ids.
        assert_eq!(out.len(), 2 * (1 + 3 + 1));
        assert_eq!(out[0], BOI_TOKEN_ID);
        assert_eq!(out[1 + 3], EOI_TOKEN_ID);
        assert_eq!(out[1 + 3 + 1], BOI_TOKEN_ID, "second frame starts at eoi+1");
        assert!(
            !out.contains(&48),
            "no '0' timestamp byte ids when video_timestamps is off"
        );
    }

    #[test]
    fn video_zero_frames_drops_placeholder() {
        let tokens = vec![VIDEO_PLACEHOLDER_TOKEN_ID];
        let out = expand_gemma_multimodal(&tokens, &counts(&[], &[], 0, 70, true), byte_ts);
        assert!(out.is_empty());
    }

    #[test]
    fn mixed_modalities_consume_counts_in_encounter_order() {
        // image #1 (10) → audio (5) → image #2 (20) → video (2 frames × 3).
        let tokens = vec![
            IMAGE_PLACEHOLDER_TOKEN_ID,
            AUDIO_PLACEHOLDER_TOKEN_ID,
            IMAGE_PLACEHOLDER_TOKEN_ID,
            VIDEO_PLACEHOLDER_TOKEN_ID,
        ];
        let out = expand_gemma_multimodal(&tokens, &counts(&[10, 20], &[5], 2, 3, true), byte_ts);
        // 12 + 7 + 22 + 2 × (5 ts + 1 boi + 3 soft + 1 eoi) = 61.
        assert_eq!(out.len(), 61);
        let mut i = 0;
        // Image #1 consumes image_counts[0] = 10.
        assert_eq!(out[i], BOI_TOKEN_ID);
        i += 1;
        assert!(
            out[i..i + 10]
                .iter()
                .all(|&t| t == IMAGE_PLACEHOLDER_TOKEN_ID)
        );
        i += 10;
        assert_eq!(out[i], EOI_TOKEN_ID);
        i += 1;
        // Audio consumes audio_counts[0] = 5.
        assert_eq!(out[i], BOA_TOKEN_ID);
        i += 1;
        assert!(
            out[i..i + 5]
                .iter()
                .all(|&t| t == AUDIO_PLACEHOLDER_TOKEN_ID)
        );
        i += 5;
        assert_eq!(out[i], EOA_TOKEN_ID);
        i += 1;
        // Image #2 consumes image_counts[1] = 20 — the i-th placeholder
        // uses the i-th count.
        assert_eq!(out[i], BOI_TOKEN_ID);
        i += 1;
        assert!(
            out[i..i + 20]
                .iter()
                .all(|&t| t == IMAGE_PLACEHOLDER_TOKEN_ID)
        );
        i += 20;
        assert_eq!(out[i], EOI_TOKEN_ID);
        i += 1;
        // Video: 2 frames, timestamps "00:00" then "00:01".
        for ts in ["00:00", "00:01"] {
            assert_eq!(&out[i..i + 5], byte_ts(ts).as_slice());
            i += 5;
            assert_eq!(out[i], BOI_TOKEN_ID);
            i += 1;
            assert!(
                out[i..i + 3]
                    .iter()
                    .all(|&t| t == VIDEO_PLACEHOLDER_TOKEN_ID)
            );
            i += 3;
            assert_eq!(out[i], EOI_TOKEN_ID);
            i += 1;
        }
        assert_eq!(i, out.len());
    }

    #[test]
    fn no_placeholders_returns_tokens_unchanged() {
        let tokens = vec![1, 2, 3];
        let out = expand_gemma_multimodal(&tokens, &counts(&[], &[], 0, 0, false), byte_ts);
        assert_eq!(out, tokens);
    }

    #[test]
    fn empty_stream_stays_empty() {
        let out = expand_gemma_multimodal(&[], &counts(&[], &[], 0, 0, false), byte_ts);
        assert!(out.is_empty());
    }

    #[test]
    fn format_video_timestamp_is_zero_padded_mm_ss() {
        assert_eq!(format_video_timestamp(0), "00:00");
        assert_eq!(format_video_timestamp(31), "00:31");
        assert_eq!(format_video_timestamp(59), "00:59");
        assert_eq!(format_video_timestamp(60), "01:00");
        assert_eq!(format_video_timestamp(61), "01:01");
    }
}
