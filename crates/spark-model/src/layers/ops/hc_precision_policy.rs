// SPDX-License-Identifier: AGPL-3.0-only

//! Select the GLM highway format without coupling native MTP to DFlash.

/// `lane` is asked last: the DFlash and native MTP switches are environment
/// reads, which a non-GLM model or an unrequested highway never pays.
pub(super) fn bf16_highway(model_type: &str, requested: bool, lane: impl FnOnce() -> bool) -> bool {
    model_type == "glm5_next" && requested && lane()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_mtp_keeps_the_requested_bf16_target_without_dflash() {
        for (model, requested, lane, expected) in [
            ("glm5_next", true, true, true),
            ("glm5_next", true, false, false),
            ("glm5_next", false, true, false),
            ("qwen3_next", true, true, false),
        ] {
            assert_eq!(bf16_highway(model, requested, || lane), expected);
        }
    }

    #[test]
    fn the_lane_is_not_read_for_other_models() {
        let unread = || -> bool { panic!("lane read") };
        assert!(!bf16_highway("qwen3_next", true, unread));
        assert!(!bf16_highway("glm5_next", false, unread));
    }
}
