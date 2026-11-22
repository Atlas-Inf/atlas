// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Promote a prefill of thinking state `enabled` (request budget 128,
/// server ceiling 64) whose first sampled token is `first`, under `limits`.
fn promote(
    enabled: bool,
    first: u32,
    max_tokens: usize,
    limits: super::super::limits::SchedLimits,
) -> ActiveSeq {
    let (a, _rx) = super::super::test_support::test_seq(vec![], 256, None, 28);
    let now = Instant::now();
    let p = super::super::prefill_a_step_params::build_prefill_in_progress(
        std::sync::Arc::new(vec![7; 28]),
        0,
        a.seq,
        28,
        max_tokens,
        0,
        super::super::test_support::EOS.to_vec(),
        a.sink,
        None,
        now,
        0.0,
        0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        vec![],
        enabled,
        Some(128),
        None,
        64,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
    );
    let spontaneous = super::first_token_thinking::spontaneous_start(
        enabled,
        first,
        Some(10),
        limits.message_open,
    );
    let immediate = !spontaneous && max_tokens <= 1;
    build_active_seq_from_prefill(
        p,
        first,
        spontaneous,
        false,
        0,
        immediate,
        now,
        Some(20),
        Some(10),
        None,
        None,
        0,
        limits,
    )
}

#[test]
fn first_token_thinking_promotion_preserves_accounting_and_handles_close() {
    for (enabled, first, max_tokens, expected) in [
        (true, 20, 256, (false, true, true)),
        (true, 20, 1, (false, true, true)),
        (true, 30, 256, (true, false, false)),
        (false, 10, 256, (true, false, false)),
        (false, 30, 256, (false, true, false)),
        (false, 20, 256, (false, true, false)),
    ] {
        let a = promote(
            enabled,
            first,
            max_tokens,
            super::super::limits::SchedLimits::NONE,
        );
        let spontaneous = !enabled && first == 10;
        let immediate = !spontaneous && max_tokens <= 1;
        assert_eq!(
            (a.inside_thinking, a.think_ended, a.think_just_ended),
            expected,
            "enabled={enabled}, first={first}, max_tokens={max_tokens}"
        );
        assert_eq!(
            a.thinking_tokens, 0,
            "preserve existing first-token accounting"
        );
        assert_eq!(a.remaining, max_tokens - 1);
        assert_eq!(
            a.output_tokens,
            if spontaneous { vec![] } else { vec![first] }
        );
        assert_eq!(a.thinking_budget, Some(if spontaneous { 64 } else { 128 }));
        assert_eq!(a.finished, immediate);
        assert!(!a.force_end_thinking && !a.think_force_closed);
    }
}

/// Inkling 2 (`message_open`): the turn's first token is a message header and
/// its first block may be the answer, so promotion starts outside reasoning,
/// with nothing closed, the token recorded and the request's own budget —
/// whatever the first token and thinking state (the think start, 10, is
/// never a spontaneous `<think>` there: the chat paths need it recorded).
#[test]
fn an_inkling_turn_is_promoted_between_blocks() {
    const MESSAGE: u32 = 40;
    let limits = super::super::limits::SchedLimits {
        message_open: Some(MESSAGE),
        ..super::super::limits::SchedLimits::NONE
    };
    for enabled in [true, false] {
        for first in [MESSAGE, 10, 20, 30] {
            let a = promote(enabled, first, 256, limits);
            assert_eq!(
                (a.inside_thinking, a.think_ended, a.think_just_ended),
                (false, false, false),
                "enabled={enabled}, first={first}"
            );
            assert_eq!(a.output_tokens, [first]);
            assert_eq!(a.thinking_budget, Some(128), "the request's budget");
            assert!(!a.finished);
        }
    }
}
