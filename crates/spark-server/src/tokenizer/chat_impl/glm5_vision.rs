// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 image/video placeholder expansion for rendered chat prompts.

use anyhow::{Result, bail};

use super::super::{ChatTokenizer, Glm5VisionKind, Glm5VisionPlaceholder};

impl ChatTokenizer {
    /// Expand the compact GLM-5 image/video triples emitted by the shipped
    /// template. GLM's checkpoint processor keeps the template text unchanged
    /// and vLLM performs this update afterwards: images become one image token
    /// per merged patch, while videos become one image-marked frame per
    /// temporal group with timestamp text between frames.
    pub(crate) fn expand_glm5_vision_placeholders(
        &self,
        tokens: Vec<u32>,
        placeholders: &[Glm5VisionPlaceholder],
    ) -> Result<Vec<u32>> {
        if placeholders.is_empty() {
            return Ok(tokens);
        }
        let image_start = self.single_token("<|begin_of_image|>")?;
        let image_end = self.single_token("<|end_of_image|>")?;
        let video_start = self.single_token("<|begin_of_video|>")?;
        let video_end = self.single_token("<|end_of_video|>")?;
        let image_pad = self.single_token("<|image|>")?;
        let video_pad = self.single_token("<|video|>")?;
        expand_glm5_tokens(
            &tokens,
            placeholders,
            Glm5TokenIds {
                image_start,
                image_end,
                video_start,
                video_end,
                image_pad,
                video_pad,
            },
            |text| self.encode(text),
        )
    }

    fn single_token(&self, text: &str) -> Result<u32> {
        let ids = self.encode(text)?;
        if ids.len() != 1 {
            bail!("GLM-5 vision marker {text:?} is not a single tokenizer token")
        }
        Ok(ids[0])
    }
}

#[derive(Debug, Clone, Copy)]
struct Glm5TokenIds {
    image_start: u32,
    image_end: u32,
    video_start: u32,
    video_end: u32,
    image_pad: u32,
    video_pad: u32,
}

fn find_subslice(haystack: &[u32], needle: &[u32], from: usize) -> Option<usize> {
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

/// Pure GLM expansion, kept separate from the tokenizer wrapper so the
/// marker/count contract has CPU tests without loading a checkpoint.
fn expand_glm5_tokens<F>(
    tokens: &[u32],
    placeholders: &[Glm5VisionPlaceholder],
    ids: Glm5TokenIds,
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
                let target = [ids.image_start, ids.image_pad, ids.image_end];
                let mut replacement = Vec::with_capacity(placeholder.pad_count + 2);
                replacement.push(ids.image_start);
                replacement.extend(std::iter::repeat_n(
                    ids.image_pad,
                    placeholder.pad_count.max(1),
                ));
                replacement.push(ids.image_end);
                (target.to_vec(), replacement)
            }
            Glm5VisionKind::Video => {
                let frame_count = placeholder.timestamps.len();
                let per_frame = placeholder.per_frame_pad_count;
                let expected_pad_count = per_frame.checked_mul(frame_count);
                if frame_count == 0
                    || per_frame == 0
                    || expected_pad_count != Some(placeholder.pad_count)
                {
                    bail!(
                        "GLM-5 video placeholder {} has pad_count={} frame_count={} per_frame={}",
                        placeholder_idx,
                        placeholder.pad_count,
                        frame_count,
                        per_frame
                    );
                }
                let target = [ids.video_start, ids.video_pad, ids.video_end];
                let mut replacement =
                    Vec::with_capacity(2 + frame_count * (per_frame + 2) + frame_count * 3);
                replacement.push(ids.video_start);
                for timestamp in placeholder.timestamps.iter().copied().take(frame_count) {
                    // GLM-5's pinned processor formats the old GLM4.1V
                    // timestamp indices as one-decimal seconds.
                    replacement.push(ids.image_start);
                    replacement.extend(std::iter::repeat_n(ids.image_pad, per_frame));
                    replacement.push(ids.image_end);
                    let text = format!("{timestamp:.1} seconds");
                    replacement.extend(encode_timestamp(&text)?);
                }
                replacement.push(ids.video_end);
                (target.to_vec(), replacement)
            }
        };
        let Some(pos) = find_subslice(tokens, &target, search_from) else {
            bail!(
                "GLM-5 vision placeholder {} was not found in rendered prompt",
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
mod glm5_vision_tests {
    use super::*;

    const IDS: Glm5TokenIds = Glm5TokenIds {
        image_start: 10,
        image_end: 11,
        video_start: 12,
        video_end: 13,
        image_pad: 14,
        video_pad: 15,
    };

    fn expand(tokens: &[u32], placeholders: &[Glm5VisionPlaceholder]) -> Vec<u32> {
        expand_glm5_tokens(tokens, placeholders, IDS, |text| {
            Ok(match text {
                "0.0 seconds" => vec![90, 91],
                "1.0 seconds" => vec![92, 93],
                _ => vec![99],
            })
        })
        .expect("synthetic GLM markers expand")
    }

    #[test]
    fn two_images_expand_in_client_order() {
        let input = [1, 10, 14, 11, 2, 10, 14, 11, 3];
        let placeholders = [
            Glm5VisionPlaceholder {
                kind: Glm5VisionKind::Image,
                pad_count: 4,
                per_frame_pad_count: 4,
                timestamps: vec![],
            },
            Glm5VisionPlaceholder {
                kind: Glm5VisionKind::Image,
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
    fn short_video_uses_image_frames_and_timestamp_tokens() {
        let input = [5, 12, 15, 13, 6];
        let p = Glm5VisionPlaceholder {
            kind: Glm5VisionKind::Video,
            pad_count: 8,
            per_frame_pad_count: 4,
            timestamps: vec![0.0, 1.0],
        };
        assert_eq!(
            expand(&input, &[p]),
            vec![
                5, 12, 10, 14, 14, 14, 14, 11, 90, 91, 10, 14, 14, 14, 14, 11, 92, 93, 13, 6
            ]
        );
    }
}
