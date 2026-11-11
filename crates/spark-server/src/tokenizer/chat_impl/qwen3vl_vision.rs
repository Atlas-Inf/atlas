// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-VL video placeholder expansion for rendered chat prompts.
//!
//! HF's `Qwen3VLProcessor.replace_video_token` rewrites the compact
//! `vision_start / video_pad / vision_end` triple the chat template emits for
//! a clip into a nested per-temporal-group form: the outer start/end are
//! kept, and inside them each group carries its own `<t.s seconds>`
//! timestamp text followed by `vision_start` + the group's pads +
//! `vision_end`. Images keep the flat `start pad*N end` expansion — only the
//! video shape changed upstream of Qwen2-VL.

use anyhow::{Result, bail};

use super::super::{ChatTokenizer, Glm5VisionKind, Glm5VisionPlaceholder};

impl ChatTokenizer {
    /// Expand Qwen3-VL's compact video triples to the nested per-group form
    /// HF's processor emits. Images expand to the flat pad run as before.
    pub(crate) fn expand_qwen3vl_vision_placeholders(
        &self,
        tokens: Vec<u32>,
        placeholders: &[Glm5VisionPlaceholder],
    ) -> Result<Vec<u32>> {
        if placeholders.is_empty() {
            return Ok(tokens);
        }
        let vision_start = self.qwen3vl_marker("<|vision_start|>")?;
        let vision_end = self.qwen3vl_marker("<|vision_end|>")?;
        let image_pad = self.image_pad_token_id().unwrap_or(u32::MAX);
        let video_pad = self.video_pad_token_id().unwrap_or(u32::MAX);
        expand_qwen3vl_tokens(
            &tokens,
            placeholders,
            Qwen3vlTokenIds {
                vision_start,
                vision_end,
                image_pad,
                video_pad,
            },
            |text| self.encode(text),
        )
    }

    fn qwen3vl_marker(&self, text: &str) -> Result<u32> {
        let ids = self.encode(text)?;
        if ids.len() != 1 {
            bail!("Qwen3-VL vision marker {text:?} is not a single tokenizer token")
        }
        Ok(ids[0])
    }
}

#[derive(Debug, Clone, Copy)]
struct Qwen3vlTokenIds {
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

/// Pure expansion, kept separate from the tokenizer wrapper so the
/// marker/count contract has CPU tests without loading a checkpoint.
fn expand_qwen3vl_tokens<F>(
    tokens: &[u32],
    placeholders: &[Glm5VisionPlaceholder],
    ids: Qwen3vlTokenIds,
    mut encode_timestamp: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    let mut out = Vec::with_capacity(tokens.len());
    let mut search_from = 0usize;
    for (placeholder_idx, placeholder) in placeholders.iter().enumerate() {
        let (target, replacement) = match placeholder.kind {
            Glm5VisionKind::Image => {
                let target = [ids.vision_start, ids.image_pad, ids.vision_end];
                let mut replacement = Vec::with_capacity(placeholder.pad_count + 2);
                replacement.push(ids.vision_start);
                replacement.extend(std::iter::repeat_n(
                    ids.image_pad,
                    placeholder.pad_count.max(1),
                ));
                replacement.push(ids.vision_end);
                (target.to_vec(), replacement)
            }
            Glm5VisionKind::Video => {
                let group_count = placeholder.timestamps.len();
                let per_group = placeholder.per_frame_pad_count;
                if group_count == 0
                    || per_group == 0
                    || per_group.checked_mul(group_count) != Some(placeholder.pad_count)
                {
                    bail!(
                        "Qwen3-VL video placeholder {} has pad_count={} groups={} per_group={}",
                        placeholder_idx,
                        placeholder.pad_count,
                        group_count,
                        per_group
                    );
                }
                // HF: the template's outer start/end wrap the whole item; per
                // group the timestamp text precedes its own inner start, pad
                // run and end.
                let target = [ids.vision_start, ids.video_pad, ids.vision_end];
                let mut replacement =
                    Vec::with_capacity(2 + group_count * (per_group + 2) + group_count * 4);
                replacement.push(ids.vision_start);
                for timestamp in placeholder.timestamps.iter().copied().take(group_count) {
                    let text = format!("<{timestamp:.1} seconds>");
                    replacement.extend(encode_timestamp(&text)?);
                    replacement.push(ids.vision_start);
                    replacement.extend(std::iter::repeat_n(ids.video_pad, per_group));
                    replacement.push(ids.vision_end);
                }
                replacement.push(ids.vision_end);
                (target.to_vec(), replacement)
            }
        };
        let Some(pos) = find_subslice(tokens, &target, search_from) else {
            bail!(
                "Qwen3-VL vision placeholder {} was not found in rendered prompt",
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
mod qwen3vl_vision_tests {
    use super::*;

    const IDS: Qwen3vlTokenIds = Qwen3vlTokenIds {
        vision_start: 10,
        vision_end: 11,
        image_pad: 14,
        video_pad: 15,
    };

    fn expand(tokens: &[u32], placeholders: &[Glm5VisionPlaceholder]) -> Vec<u32> {
        expand_qwen3vl_tokens(tokens, placeholders, IDS, |text| {
            Ok(match text {
                "<0.2 seconds>" => vec![90, 91],
                "<1.4 seconds>" => vec![92, 93],
                "<2.5 seconds>" => vec![94, 95],
                _ => vec![99],
            })
        })
        .expect("synthetic Qwen3-VL markers expand")
    }

    #[test]
    fn two_group_video_nests_timestamps_inside_the_outer_pair() {
        let input = [1, 10, 15, 11, 2];
        let p = Glm5VisionPlaceholder {
            kind: Glm5VisionKind::Video,
            pad_count: 8,
            per_frame_pad_count: 4,
            timestamps: vec![0.2, 1.4],
            qwen3vl: true,
        };
        assert_eq!(
            expand(&input, &[p]),
            vec![
                1, 10, // outer start kept
                90, 91, 10, 15, 15, 15, 15, 11, // <0.2s> vs pads ve
                92, 93, 10, 15, 15, 15, 15, 11, // <1.4s> vs pads ve
                11, 2 // outer end + tail
            ]
        );
    }

    #[test]
    fn image_then_video_orders_match_the_content_array() {
        let input = [5, 10, 14, 11, 10, 15, 11, 6];
        let placeholders = [
            Glm5VisionPlaceholder {
                kind: Glm5VisionKind::Image,
                pad_count: 4,
                per_frame_pad_count: 4,
                timestamps: vec![],
                qwen3vl: false,
            },
            Glm5VisionPlaceholder {
                kind: Glm5VisionKind::Video,
                pad_count: 4,
                per_frame_pad_count: 2,
                timestamps: vec![0.2, 2.5],
                qwen3vl: true,
            },
        ];
        assert_eq!(
            expand(&input, &placeholders),
            vec![
                5, 10, 14, 14, 14, 14, 11, 10, 90, 91, 10, 15, 15, 11, 94, 95, 10, 15, 15, 11, 11,
                6
            ]
        );
    }

    #[test]
    fn a_missing_triple_is_an_error_not_a_silent_truncate() {
        let input = [1, 10, 14, 11, 2];
        let p = Glm5VisionPlaceholder {
            kind: Glm5VisionKind::Video,
            pad_count: 4,
            per_frame_pad_count: 4,
            timestamps: vec![0.2],
            qwen3vl: true,
        };
        assert!(expand_qwen3vl_tokens(&input, &[p], IDS, |_| Ok(vec![99])).is_err());
    }
}
