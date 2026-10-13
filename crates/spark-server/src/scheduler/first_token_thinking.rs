// SPDX-License-Identifier: AGPL-3.0-only

/// Thinking state after the prefill's first sampled token. Token accounting,
/// spontaneous-start suppression and budget selection stay with the caller.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FirstTokenThinking {
    pub inside_thinking: bool,
    pub think_ended: bool,
    pub think_just_ended: bool,
}

impl FirstTokenThinking {
    pub fn resolve_with_tool_boundary(
        enabled: bool,
        first: u32,
        start: Option<u32>,
        end: Option<u32>,
        native_tool_boundary: Option<u32>,
    ) -> Self {
        if enabled && native_tool_boundary == Some(first) {
            tracing::info!(
                native_opener = first,
                "GLM first-token implicit thinking boundary"
            );
            return Self {
                inside_thinking: false,
                think_ended: true,
                // The native opener is already emitted; never force a second one.
                think_just_ended: false,
            };
        }
        Self::resolve(enabled, first, start, end)
    }

    pub fn resolve(enabled: bool, first: u32, start: Option<u32>, end: Option<u32>) -> Self {
        let spontaneous = !enabled && start == Some(first);
        // Prefill already emitted this token. Mirror the decode-time close
        // instead of making the scheduler wait for a second </think>.
        let closed = enabled && end == Some(first);
        Self {
            inside_thinking: !closed && (spontaneous || (enabled && end.is_some())),
            think_ended: closed || (!spontaneous && !enabled && end.is_some()),
            think_just_ended: closed,
        }
    }
}

/// Consume only a configured native GLM delimiter, never text within reasoning.
/// Call before grammar, tool bookkeeping and token accounting on every emit path.
pub(super) fn apply_native_tool_boundary(
    a: &mut super::ActiveSeq,
    token: u32,
    native: Option<u32>,
) {
    if !a.inside_thinking
        || !a.tools_present
        || native != Some(token)
        || a.tool_call_start_token != Some(token)
    {
        return;
    }
    a.inside_thinking = false;
    a.think_ended = true;
    a.think_just_ended = false;
    a.force_end_thinking = false;
    a.think_force_closed = false;
    a.sentence_defer_count = 0;
    a.consecutive_confident = 0;
    a.in_code_fence = false;
    a.think_skip_count = 0;
    a.post_think_emitted = 0;
    a.require_tool_call = false;
    a.tool_call_opened = true;
    tracing::info!(
        native_opener = token,
        session_hash = a.session_hash,
        slot = a.seq.slot_idx,
        "GLM native tool opener ended thinking"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_token_thinking_requested_close_matches_decode_transition() {
        // The configured close is authoritative; no inference from the start
        // ID, even when absent or equal, is needed to consume this close.
        for start in [Some(10), None, Some(20)] {
            assert_eq!(
                FirstTokenThinking::resolve(true, 20, start, Some(20)),
                FirstTokenThinking {
                    inside_thinking: false,
                    think_ended: true,
                    think_just_ended: true,
                }
            );
        }
    }

    #[test]
    fn first_token_thinking_preserves_other_initial_states() {
        for (enabled, first, start, end, expected) in [
            (true, 30, Some(10), Some(20), (true, false, false)),
            (true, 10, Some(10), Some(20), (true, false, false)),
            (false, 10, Some(10), Some(20), (true, false, false)),
            (false, 30, Some(10), Some(20), (false, true, false)),
            (false, 20, Some(10), Some(20), (false, true, false)),
            (true, 30, None, None, (false, false, false)),
            (true, 20, Some(10), None, (false, false, false)),
            (false, 10, Some(10), None, (true, false, false)),
            (false, 30, None, None, (false, false, false)),
        ] {
            let state = FirstTokenThinking::resolve(enabled, first, start, end);
            assert_eq!(
                (
                    state.inside_thinking,
                    state.think_ended,
                    state.think_just_ended
                ),
                expected,
                "enabled={enabled}, first={first}, start={start:?}, end={end:?}"
            );
        }
    }
}
