// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in shared by speculative target attention and drafter cache population.
pub(super) fn enabled(model: &str) -> bool {
    model == "glm5_next" && std::env::var("ATLAS_GLM_MTP_LONG_CONTEXT").as_deref() == Ok("1")
}
