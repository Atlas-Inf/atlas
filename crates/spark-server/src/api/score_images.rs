// SPDX-License-Identifier: AGPL-3.0-only

use base64::Engine;
use serde::Deserialize;
use spark_model::VisionItem;

use crate::tokenizer::ChatTokenizer;

#[derive(Deserialize)]
pub(super) struct ScoreImageInput {
    pub data_base64: String,
    pub format: String,
}

/// Image decodes in flight across all score requests.
pub(super) static DECODE_PERMITS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);

#[derive(Clone, Copy)]
pub(super) struct ImageTokenIds {
    pub vision_start: u32,
    pub image_pad: u32,
    pub vision_end: u32,
    /// The video pad, when the tokenizer has one: refused in image requests
    /// (the splice and M-RoPE would count it as a vision row).
    pub video_pad: Option<u32>,
}

pub(super) fn images_with_pack(images: Option<&[ScoreImageInput]>, has_pack: bool) -> bool {
    images.is_some() && has_pack
}

pub(super) fn vision_tower_missing(image_count: usize, has_tower: bool) -> bool {
    image_count > 0 && !has_tower
}

pub(super) fn image_token_ids(
    tokenizer: &ChatTokenizer,
    vision: Option<&atlas_core::config::VisionConfig>,
) -> Result<Option<ImageTokenIds>, String> {
    let token_id = |token: &str| {
        tokenizer
            .encode(token)
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    };
    let (Some(vision_start), Some(image_pad), Some(vision_end)) = (
        token_id("<|vision_start|>"),
        token_id("<|image_pad|>"),
        token_id("<|vision_end|>"),
    ) else {
        return Ok(None);
    };
    if let Some(vision) = vision
        && vision.image_pad_token_id != 0
        && vision.image_pad_token_id != image_pad
    {
        return Err(format!(
            "score: tokenizer image-pad id {image_pad} disagrees with the vision config id {}",
            vision.image_pad_token_id
        ));
    }
    Ok(Some(ImageTokenIds {
        vision_start,
        image_pad,
        vision_end,
        video_pad: token_id("<|video_pad|>"),
    }))
}

pub(super) fn validate_image_tokens(
    prefix: &[u32],
    branches: &[Vec<u32>],
    image_count: usize,
    ids: Option<ImageTokenIds>,
) -> Result<Vec<usize>, String> {
    let Some(ids) = ids else {
        return if image_count == 0 {
            Ok(Vec::new())
        } else {
            Err("score: tokenizer does not provide the model's image markers".into())
        };
    };
    if branches
        .iter()
        .any(|branch| branch.contains(&ids.image_pad))
    {
        return Err("score: image pads in branches are not supported".into());
    }
    if let Some(video_pad) = ids.video_pad
        && (prefix.contains(&video_pad) || branches.iter().any(|b| b.contains(&video_pad)))
    {
        return Err("score: video pads are not supported in image requests".into());
    }
    let counts = image_run_pad_counts(prefix, ids)?;
    if counts.len() != image_count {
        return Err(format!(
            "score: prefix has {} image runs but images contains {image_count}",
            counts.len()
        ));
    }
    Ok(counts)
}

fn image_run_pad_counts(tokens: &[u32], ids: ImageTokenIds) -> Result<Vec<usize>, String> {
    let mut runs = Vec::new();
    let (mut in_run, mut pads) = (false, 0usize);
    for (position, &token) in tokens.iter().enumerate() {
        if token == ids.vision_start {
            if in_run {
                return Err(format!(
                    "score: nested vision_start at prefix position {position}"
                ));
            }
            in_run = true;
            pads = 0;
        } else if token == ids.image_pad {
            if !in_run {
                return Err(format!(
                    "score: image_pad outside an image run at prefix position {position}"
                ));
            }
            pads += 1;
        } else if token == ids.vision_end {
            if !in_run || pads == 0 {
                return Err(format!(
                    "score: vision_end outside a non-empty image run at prefix position {position}"
                ));
            }
            runs.push(pads);
            in_run = false;
        } else if in_run {
            return Err(format!(
                "score: token outside an image run at prefix position {position}"
            ));
        }
    }
    if in_run {
        return Err("score: unterminated image run in prefix".into());
    }
    Ok(runs)
}

pub(super) fn preprocess_images(
    images: Vec<ScoreImageInput>,
    pad_counts: &[usize],
    vision: &atlas_core::config::VisionConfig,
    max_pixels: Option<usize>,
) -> Result<Vec<VisionItem>, String> {
    if images.len() != pad_counts.len() {
        return Err(format!(
            "score: {} images do not match {} image runs",
            images.len(),
            pad_counts.len()
        ));
    }
    images
        .into_iter()
        .zip(pad_counts)
        .enumerate()
        .map(|(index, (image, &run_pads))| {
            let bytes = decode_base64(&image.data_base64)?;
            let (pixels, grid_h, grid_w) = spark_model::vision_preprocess::preprocess_score_image(
                &bytes,
                &image.format,
                vision,
                max_pixels,
            )
            .map_err(|e| format!("score: image {index} could not be decoded: {e:#}"))?;
            let item = VisionItem::image(pixels, grid_h, grid_w);
            validate_pad_count(index, run_pads, item.pad_count(vision.spatial_merge_size))?;
            Ok(item)
        })
        .collect()
}

fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    use spark_model::vision_preprocess::MAX_SCORE_IMAGE_BYTES;
    let max_encoded = MAX_SCORE_IMAGE_BYTES.div_ceil(3) * 4;
    if encoded.len() > max_encoded {
        return Err(format!(
            "score: encoded image exceeds the {MAX_SCORE_IMAGE_BYTES} byte limit"
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("score: image data_base64 is invalid: {e}"))?;
    if bytes.len() > MAX_SCORE_IMAGE_BYTES {
        return Err(format!(
            "score: image decodes to {} bytes, over the {MAX_SCORE_IMAGE_BYTES} byte limit",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn validate_pad_count(index: usize, run_pads: usize, image_pads: usize) -> Result<(), String> {
    if run_pads != image_pads {
        return Err(format!(
            "score: image {index} has {run_pads} image pads in prefix, but Atlas computes {image_pads}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: ImageTokenIds = ImageTokenIds {
        vision_start: 10,
        image_pad: 11,
        vision_end: 12,
        video_pad: Some(13),
    };

    #[test]
    fn image_runs_must_be_well_formed_and_match_image_count() {
        assert_eq!(
            validate_image_tokens(&[1, 10, 11, 11, 12, 2], &[], 1, Some(IDS)).unwrap(),
            [2]
        );
        assert!(
            validate_image_tokens(&[1, 10, 11, 12], &[], 0, Some(IDS))
                .unwrap_err()
                .contains("1 image runs but images contains 0")
        );
        for tokens in [
            &[11][..],
            &[10, 12],
            &[10, 11],
            &[12],
            &[10, 10, 11, 12],
            &[10, 11, 1, 12],
        ] {
            assert!(image_run_pad_counts(tokens, IDS).is_err(), "{tokens:?}");
        }
    }

    #[test]
    fn video_pads_are_refused_in_image_requests() {
        assert!(
            validate_image_tokens(&[10, 11, 12, 13], &[], 1, Some(IDS))
                .unwrap_err()
                .contains("video pads")
        );
        assert!(
            validate_image_tokens(&[10, 11, 12], &[vec![13]], 1, Some(IDS))
                .unwrap_err()
                .contains("video pads")
        );
    }

    #[test]
    fn image_pads_in_branches_are_refused() {
        assert!(
            validate_image_tokens(&[], &[vec![1, 11, 2]], 0, Some(IDS))
                .unwrap_err()
                .contains("branches")
        );
    }

    #[test]
    fn base64_is_strict_and_bounded() {
        assert!(decode_base64("%%%").unwrap_err().contains("invalid"));
        let too_long =
            "A".repeat(spark_model::vision_preprocess::MAX_SCORE_IMAGE_BYTES.div_ceil(3) * 4 + 4);
        assert!(decode_base64(&too_long).unwrap_err().contains("limit"));
    }

    #[test]
    fn pad_count_refusal_names_both_counts() {
        let error = validate_pad_count(2, 17, 16).unwrap_err();
        assert!(error.contains("17 image pads"));
        assert!(error.contains("Atlas computes 16"));
    }

    #[test]
    fn images_with_packs_are_refused() {
        let image = ScoreImageInput {
            data_base64: String::new(),
            format: "png".into(),
        };
        assert!(images_with_pack(Some(&[image]), true));
        assert!(images_with_pack(Some(&[]), true));
        assert!(!images_with_pack(None, true));
        assert!(vision_tower_missing(1, false));
        assert!(!vision_tower_missing(0, false));
        assert!(!vision_tower_missing(1, true));
    }
}
