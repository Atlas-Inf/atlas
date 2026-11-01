// SPDX-License-Identifier: AGPL-3.0-only

//! The pixel bounds a checkpoint's processor config declares, per modality.
//! Split out of `serve.rs` (allow-listed past the file-size cap) when the
//! image `min_pixels` and the video bounds joined the image `max_pixels`.

/// Which processor's bounds to read.
#[derive(Clone, Copy, Debug)]
pub(super) enum Modality {
    Image,
    Video,
}

/// One processor's `size = {shortest_edge, longest_edge}` (or
/// `min_pixels` / `max_pixels`): pixel COUNTS, not edge lengths, despite the
/// names (16777216 = 4096², 65536 = 256²).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DeclaredPixels {
    pub min_pixels: Option<usize>,
    pub max_pixels: usize,
}

/// The bounds `modality`'s processor declares, if the checkpoint ships one.
/// `None` on any absence or malformation — a model without the file, or
/// with one we cannot read, must keep the historical behaviour rather than
/// fail to serve.
///
/// SEVERAL FILENAMES, because the ecosystem uses all of them. HF's
/// `save_pretrained` writes `preprocessor_config.json` (and, for video,
/// `video_preprocessor_config.json`) with the fields at the top level
/// (`Qwen/Qwen3.6-35B-A3B-FP8`), while a combined processor writes
/// `processor_config.json` with each modality under its own key
/// (`unsloth/Qwen3.6-27B-NVFP4` → `image_processor.size.longest_edge`).
/// Reading only the first is why the unsloth checkpoints still ran the
/// 1280px fallback after the area bound was supposedly honoured.
///
/// The nested form is also why this cannot be a search for the first
/// `longest_edge` in the document: `processor_config.json` carries the video
/// bound (25165824, a `t x h x w` volume) beside the image one (16777216);
/// picking it would over-admit every still image by half again its
/// permitted area. Each modality is addressed explicitly, and both bounds
/// come from the one source that declares the max.
pub(super) fn read_declared_pixels(
    model_dir: &std::path::Path,
    modality: Modality,
) -> Option<DeclaredPixels> {
    // Ordered by precedence: the dedicated file wins over a combined one.
    let sources: &[(&str, Option<&str>)] = match modality {
        Modality::Image => &[
            ("preprocessor_config.json", None),
            ("preprocessor_config.json", Some("image_processor")),
            ("processor_config.json", Some("image_processor")),
        ],
        Modality::Video => &[
            ("video_preprocessor_config.json", None),
            ("processor_config.json", Some("video_processor")),
        ],
    };
    for &(file, nest) in sources {
        let path = model_dir.join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(scope) = nest.map_or(Some(&json), |key| json.get(key)) else {
            continue;
        };
        let count = |size_key: &str, direct_key: &str| {
            scope
                .get("size")
                .and_then(|s| s.get(size_key))
                .or_else(|| scope.get(direct_key))
                .and_then(serde_json::Value::as_u64)
                .filter(|&p| p > 0)
                .map(|p| p as usize)
        };
        let Some(max_pixels) = count("longest_edge", "max_pixels") else {
            continue;
        };
        let declared = DeclaredPixels {
            min_pixels: count("shortest_edge", "min_pixels"),
            max_pixels,
        };
        tracing::info!(
            "Vision {modality:?} bounds {declared:?} from {}{}",
            path.display(),
            nest.map(|k| format!(" [{k}]")).unwrap_or_default(),
        );
        return Some(declared);
    }
    None
}

#[cfg(test)]
#[path = "processor_pixels_tests.rs"]
mod tests;
