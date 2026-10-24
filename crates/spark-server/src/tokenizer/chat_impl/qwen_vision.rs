// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8 image/video placeholder expansion for rendered chat prompts.

use anyhow::{Result, bail};

use super::super::{ChatTokenizer, QwenVisionKind, QwenVisionPlaceholder};

impl ChatTokenizer {
    /// Expand the compact Qwen marker emitted by the shipped template. The
    /// processor used by Qwen3.8 writes one timestamped vision range for each
    /// temporal group of a video, while still images use one range.
    pub(crate) fn expand_qwen_vision_placeholders(
        &self,
        tokens: Vec<u32>,
        placeholders: &[QwenVisionPlaceholder],
    ) -> Result<Vec<u32>> {
        if placeholders.is_empty() {
            return Ok(tokens);
        }
        let vision_start = self.qwen_single_token("<|vision_start|>")?;
        let vision_end = self.qwen_single_token("<|vision_end|>")?;
        let image_pad = self.qwen_single_token("<|image_pad|>")?;
        let video_pad = self.qwen_single_token("<|video_pad|>")?;
        expand_qwen_tokens(
            &tokens,
            placeholders,
            QwenTokenIds {
                vision_start,
                vision_end,
                image_pad,
                video_pad,
            },
            |text| self.encode(text),
        )
    }

    fn qwen_single_token(&self, text: &str) -> Result<u32> {
        let ids = self.encode(text)?;
        if ids.len() != 1 {
            bail!("Qwen vision marker {text:?} is not a single tokenizer token")
        }
        Ok(ids[0])
    }
}

#[derive(Debug, Clone, Copy)]
struct QwenTokenIds {
    vision_start: u32,
    vision_end: u32,
    image_pad: u32,
    video_pad: u32,
}

fn find_subslice(haystack: &[u32], needle: &[u32], from: usize) -> Option<usize> {
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

/// Pure Qwen expansion, kept separate from the tokenizer wrapper so the
/// marker/count contract has CPU tests without loading a checkpoint.
fn expand_qwen_tokens<F>(
    tokens: &[u32],
    placeholders: &[QwenVisionPlaceholder],
    ids: QwenTokenIds,
    mut encode_timestamp: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    let mut out = Vec::with_capacity(tokens.len());
    let mut search_from = 0usize;
    for (placeholder_idx, placeholder) in placeholders.iter().enumerate() {
        let target_pad = match placeholder.kind {
            QwenVisionKind::Image => ids.image_pad,
            QwenVisionKind::Video => ids.video_pad,
        };
        let target = [ids.vision_start, target_pad, ids.vision_end];
        let replacement = match placeholder.kind {
            QwenVisionKind::Image => {
                let mut replacement = Vec::with_capacity(placeholder.pad_count + 2);
                replacement.push(ids.vision_start);
                replacement.extend(std::iter::repeat_n(
                    ids.image_pad,
                    placeholder.pad_count.max(1),
                ));
                replacement.push(ids.vision_end);
                replacement
            }
            QwenVisionKind::Video => {
                let frame_count = placeholder.timestamps.len();
                let per_frame = placeholder.per_frame_pad_count;
                let expected_pad_count = per_frame.checked_mul(frame_count);
                if frame_count == 0
                    || per_frame == 0
                    || expected_pad_count != Some(placeholder.pad_count)
                {
                    bail!(
                        "Qwen video placeholder {} has pad_count={} frame_count={} per_frame={}",
                        placeholder_idx,
                        placeholder.pad_count,
                        frame_count,
                        per_frame
                    );
                }
                let mut replacement = Vec::with_capacity(2 + frame_count * (per_frame + 4));
                for timestamp in placeholder.timestamps.iter().copied() {
                    // This is the exact Qwen3-VL/Qwen3.8 processor spelling.
                    replacement.extend(encode_timestamp(&format!("<{timestamp:.1} seconds>"))?);
                    replacement.push(ids.vision_start);
                    replacement.extend(std::iter::repeat_n(ids.video_pad, per_frame));
                    replacement.push(ids.vision_end);
                }
                replacement
            }
        };
        let Some(pos) = find_subslice(tokens, &target, search_from) else {
            bail!(
                "Qwen vision placeholder {} was not found in rendered prompt",
                placeholder_idx
            );
        };
        out.extend_from_slice(&tokens[search_from..pos]);
        out.extend(replacement);
        search_from = pos + target.len();
    }
    out.extend_from_slice(&tokens[search_from..]);
    Ok(out)
}

#[cfg(test)]
mod qwen_vision_tests {
    use super::*;

    const IDS: QwenTokenIds = QwenTokenIds {
        vision_start: 10,
        vision_end: 11,
        image_pad: 14,
        video_pad: 15,
    };

    fn expand(tokens: &[u32], placeholders: &[QwenVisionPlaceholder]) -> Vec<u32> {
        expand_qwen_tokens(tokens, placeholders, IDS, |text| {
            Ok(match text {
                "<0.0 seconds>" => vec![90, 91],
                "<1.5 seconds>" => vec![92, 93],
                _ => vec![99],
            })
        })
        .expect("synthetic Qwen markers expand")
    }

    #[test]
    fn image_expands_without_changing_marker_order() {
        let input = [1, 10, 14, 11, 2, 10, 14, 11, 3];
        let placeholders = [
            QwenVisionPlaceholder {
                kind: QwenVisionKind::Image,
                pad_count: 4,
                per_frame_pad_count: 4,
                timestamps: vec![],
            },
            QwenVisionPlaceholder {
                kind: QwenVisionKind::Image,
                pad_count: 2,
                per_frame_pad_count: 2,
                timestamps: vec![],
            },
        ];
        assert_eq!(
            expand(&input, &placeholders),
            vec![1, 10, 14, 14, 14, 14, 11, 2, 10, 14, 14, 11, 3]
        );
    }

    #[test]
    fn video_expands_to_timestamped_per_frame_ranges() {
        let input = [5, 10, 15, 11, 6];
        let p = QwenVisionPlaceholder {
            kind: QwenVisionKind::Video,
            pad_count: 4,
            per_frame_pad_count: 2,
            timestamps: vec![0.0, 1.5],
        };
        assert_eq!(
            expand(&input, &[p]),
            vec![5, 90, 91, 10, 15, 15, 11, 92, 93, 10, 15, 15, 11, 6]
        );
    }

    #[test]
    fn mixed_image_video_image_follows_content_order() {
        let input = [10, 14, 11, 20, 10, 15, 11, 21, 10, 14, 11];
        let placeholders = [
            QwenVisionPlaceholder {
                kind: QwenVisionKind::Image,
                pad_count: 1,
                per_frame_pad_count: 1,
                timestamps: vec![],
            },
            QwenVisionPlaceholder {
                kind: QwenVisionKind::Video,
                pad_count: 2,
                per_frame_pad_count: 1,
                timestamps: vec![0.0, 1.5],
            },
            QwenVisionPlaceholder {
                kind: QwenVisionKind::Image,
                pad_count: 1,
                per_frame_pad_count: 1,
                timestamps: vec![],
            },
        ];
        assert_eq!(
            expand(&input, &placeholders),
            vec![
                10, 14, 11, 20, 90, 91, 10, 15, 11, 92, 93, 10, 15, 11, 21, 10, 14, 11
            ]
        );
    }

    #[test]
    fn malformed_video_metadata_is_rejected() {
        let p = QwenVisionPlaceholder {
            kind: QwenVisionKind::Video,
            pad_count: 3,
            per_frame_pad_count: 2,
            timestamps: vec![0.0, 1.0],
        };
        let err = expand_qwen_tokens(&[10, 15, 11], &[p], IDS, |_| Ok(vec![99]))
            .expect_err("mismatched metadata");
        assert!(err.to_string().contains("pad_count=3"));
    }
}
