// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM reasoning-to-tool boundary, resolved from actual model metadata.
//! This is not text salvage: callers must match the tokenizer's native opener.
pub(crate) fn native_opener(
    model_type: &str,
    tools_active: bool,
    native: Option<u32>,
) -> Option<u32> {
    if model_type == "glm5_next" && tools_active {
        native
    } else {
        None
    }
}

/// Native GLM end-of-turn while reasoning, not an implicit `</think>`.
/// This identifies the token only; callers retain all other stop guards.
pub(crate) fn native_eos_while_thinking(
    native_boundary: Option<u32>,
    inside_thinking: bool,
    token: u32,
    eos_tokens: &[u32],
) -> bool {
    native_boundary.is_some() && inside_thinking && eos_tokens.contains(&token)
}
