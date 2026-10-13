// SPDX-License-Identifier: AGPL-3.0-only

use super::emit_token;
use crate::api::StreamEvent;
use crate::scheduler::{sched_ctx::SchedCtx, test_support::test_seq, types::ResponseSink};

const EOS: u32 = 900;
const CLOSE: u32 = 901;

#[test]
fn mtp_thinking_eos_waits_for_visible_answer_boundary() {
    for requested in [true, false] {
        let (mut a, _) = test_seq(vec![10], 20, None, 12);
        a.finished = false;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        a.sink = ResponseSink::Streaming(tx);
        a.min_tokens = 0;
        a.eos_tokens = vec![EOS];
        a.inside_thinking = true;
        a.enable_thinking = requested;
        a.think_ended = false;
        a.think_end_token = Some(CLOSE);
        a.thinking_budget = Some(32);
        let sched = SchedCtx::for_test();
        emit_token(&mut a, EOS, None, &sched);
        assert!(
            !a.finished,
            "thinking EOS must not end a plain MTP response"
        );
        assert_eq!(a.remaining, 19);
        // Discarded as serial decode discards it (`min_tokens_eos_tests`).
        assert_eq!(a.output_tokens, vec![10]);
        assert!(rx.try_recv().is_err(), "suppressed EOS must not stream");
        emit_token(&mut a, CLOSE, None, &sched);
        assert!(!a.inside_thinking && a.think_ended && !a.finished);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(CLOSE))));
        emit_token(&mut a, 42, None, &sched);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(42))));
        emit_token(&mut a, EOS, None, &sched);
        assert!(a.finished);
        assert_eq!(a.remaining, 16);
        assert!(rx.try_recv().is_err());
    }
}

#[test]
fn mtp_eos_cannot_bypass_output_or_context_ceiling() {
    for thinking in [true, false] {
        for output_ceiling in [true, false] {
            let remaining = if output_ceiling { 1 } else { 20 };
            let (mut a, _) = test_seq(vec![10], remaining, None, 12);
            a.finished = false;
            a.eos_tokens = vec![EOS];
            a.min_tokens = 100;
            a.inside_thinking = thinking;
            a.think_ended = !thinking;
            let mut sched = SchedCtx::for_test();
            sched.limits.max_seq_len = if output_ceiling { 100 } else { 12 };
            emit_token(&mut a, EOS, None, &sched);
            assert!(a.finished, "EOS suppression must not bypass a hard ceiling");
            assert_eq!(a.remaining, remaining - 1);
            assert_eq!(a.output_tokens, vec![10, EOS]);
        }
    }
}
