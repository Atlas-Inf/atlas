// SPDX-License-Identifier: AGPL-3.0-only

//! Which PTX module a checkpoint's vision tower needs from the kernel target.

/// Return the PTX module that implements the checkpoint's vision tower.
///
/// Qwen-shaped towers use the historical `vision_encoder` module name. GLM-
/// 5.3 has intentionally separate kernels and therefore ships
/// `glm_vision_encoder`; accepting that name only for the corresponding
/// parsed config keeps the startup guard fail-closed for every other model.
pub(super) fn required_vision_module(is_glm5_next: bool) -> &'static str {
    if is_glm5_next {
        "glm_vision_encoder"
    } else {
        "vision_encoder"
    }
}

pub(super) fn target_has_required_vision_module(
    is_glm5_next: bool,
    modules: &[(&str, &[u8])],
) -> bool {
    let required = required_vision_module(is_glm5_next);
    modules.iter().any(|(name, _)| *name == required)
}
