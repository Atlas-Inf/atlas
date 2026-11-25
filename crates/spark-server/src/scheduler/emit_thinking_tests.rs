// SPDX-License-Identifier: AGPL-3.0-only

use super::emit_token;
use crate::api::StreamEvent;
use crate::scheduler::{sched_ctx::SchedCtx, test_support::test_seq, types::ResponseSink};

const EOS: u32 = 900;
const CLOSE: u32 = 901;

#[test]
fn mtp_thinking_eos_waits_for_visible_answer_boundary() {
    for requested in [true, false] {
        let (mut a, _response_rx) = test_seq(vec![10], 20, None, 12);
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
            // Held: a dropped receiver reads as a client disconnect.
            let (mut a, _rx) = test_seq(vec![10], remaining, None, 12);
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

/// Inkling 2 with thinking off: the reasoning block's body (its think start
/// included) is not streamed, as for a spontaneous `<think>`; its close and
/// the answer are. The stream's Inkling rewrite drops a thinking block when
/// thinking is off, so it reads the same answer either way
/// (`inkling_turn_tests::a_suppressed_thinking_body_streams_the_same_answer`).
#[test]
fn inkling_thinking_off_streams_the_close_and_the_answer_only() {
    use crate::scheduler::test_support::inkling::{self, END, MESSAGE, TEXT, THINKING};
    for requested in [true, false] {
        let (mut a, _response_rx) = inkling::seq(requested, None);
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        a.sink = ResponseSink::Streaming(tx);
        let sched = inkling::sched();
        let toks = [THINKING, 5, END, MESSAGE, TEXT, 6];
        for &t in &toks {
            emit_token(&mut a, t, None, &sched);
        }
        let mut streamed = Vec::new();
        while let Ok(StreamEvent::Token(t)) = rx.try_recv() {
            streamed.push(t);
        }
        let want: &[u32] = if requested {
            &toks
        } else {
            &[END, MESSAGE, TEXT, 6]
        };
        assert_eq!(streamed, want, "thinking requested={requested}");
        assert_eq!(a.output_tokens.len(), 1 + toks.len(), "all recorded");
    }
}
