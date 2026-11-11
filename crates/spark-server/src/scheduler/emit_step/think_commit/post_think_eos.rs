// SPDX-License-Identifier: AGPL-3.0-only

//! Post-thinking EOS guard: hold a short post-`</think>` EOS only for a real tool obligation.

use super::*;

const POST_THINK_MIN_CONTENT: u32 = 16;

/// Whether a short post-thinking EOS must be held for a real tool obligation.
///
/// `ActiveSeq::tool_request` is sticky tool availability, not evidence that
/// this assistant turn currently has to emit a call. In particular, an
/// auto-tools request answering a preceding tool result must be able to end
/// with a short plain-text answer. Required calls and actually opened,
/// incomplete calls still keep the post-thinking guard armed.
pub(in crate::scheduler) fn should_suppress_post_think_eos(
    a: &ActiveSeq,
    post_think_content_tokens: u32,
) -> bool {
    let tools_armed =
        a.require_tool_call || a.inside_tool_body || (a.tool_call_opened && !a.tool_call_completed);
    tools_armed && a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT
}
