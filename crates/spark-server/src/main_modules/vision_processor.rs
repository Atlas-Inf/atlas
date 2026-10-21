// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint video-processor detection for the serve-time vision config.
//! Kept out of `serve.rs` (an allow-listed-size file) — see AGENTS.md's
//! 500-LoC cap.

/// Whether the checkpoint's VIDEO processor is HF's Qwen3VL — the nested
/// per-temporal-group prompt shape whose MRoPE gives every group a t=1
/// plane. Read from the same preprocessor files the image bound comes from:
/// `video_preprocessor_config.json` is the dedicated file HF writes for
/// combined processors, with the class under `video_processor_type` or
/// `processor_class`; `processor_config.json` carries the same fields under
/// `video_processor`. `None` on any absence — a checkpoint without the file
/// keeps the legacy flat expansion.
pub(super) fn read_is_qwen3vl_video(model_dir: &std::path::Path) -> Option<bool> {
    const SOURCES: [(&str, Option<&str>); 4] = [
        ("video_preprocessor_config.json", None),
        ("preprocessor_config.json", Some("video_processor")),
        ("preprocessor_config.json", None),
        ("processor_config.json", Some("video_processor")),
    ];
    for (file, nest) in SOURCES {
        let path = model_dir.join(file);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let scope = match nest {
            Some(key) => match json.get(key) {
                Some(v) => v,
                None => continue,
            },
            None => &json,
        };
        let is_qwen3vl = scope
            .get("video_processor_type")
            .or_else(|| scope.get("processor_class"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| s == "Qwen3VLVideoProcessor" || s == "Qwen3VLProcessor");
        if is_qwen3vl {
            tracing::info!("Qwen3-VL video processor detected in {}", path.display());
            return Some(true);
        }
    }
    None
}
