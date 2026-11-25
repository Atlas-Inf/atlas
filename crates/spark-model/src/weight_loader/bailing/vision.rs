// SPDX-License-Identifier: AGPL-3.0-only

//! Vision load-skip seam (text-only phase).
//!
//! `bailing_moe_v3_vl` ships a 27-block `qwen3_moe_vit` tower under
//! `model.visual.*` (329 tensors) plus a `linear_proj.{0,2}.{weight,bias}`
//! projector (GELU MLP mapping the tower output into the text hidden size).
//! The campaign is text-only first, so both namespaces are declared dead —
//! skipped at upload instead of holding GPU memory as dead weight — and
//! `load_vision_encoder` returns `None`.
//!
//! Extension seam (mirrors `weight_loader/glm5/vision.rs`): when the vision
//! phase lands, replace [`dead_namespaces`] with an empty Vec here, load the
//! tower and projector in an override of
//! [`crate::weight_loader::ModelWeightLoader::load_vision_encoder`], and
//! keep the projector's `linear_proj.*` prefix (held by
//! `BailingMoeV3VLForConditionalGeneration`, NOT under `model.`) in mind:
//! `linear_proj.0` maps `[vision_out_dim=hidden_size*spatial_merge²,
//! hidden]` (checkpoint `disable_merger_proj = true` → out_dim 4608) and
//! `linear_proj.2` maps `[hidden, hidden]`.

/// Namespace prefixes this loader never reads.
pub(super) fn dead_namespaces() -> Vec<String> {
    [super::names::SKIP_TOWER, super::names::SKIP_PROJECTOR]
        .map(String::from)
        .to_vec()
}
