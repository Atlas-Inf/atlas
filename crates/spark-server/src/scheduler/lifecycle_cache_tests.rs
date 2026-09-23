// SPDX-License-Identifier: AGPL-3.0-only

//! `finish_sequence` vs the prefix cache: a sequence that died of an ENGINE
//! ERROR must not leave a finish leaf behind.
//!
//! A finish leaf pairs `seq.tokens` with a snapshot of the sequence's LIVE
//! recurrent state. After a mid-forward error the tokens are un-advanced but
//! the layers before the failing one have already consumed the step, so the
//! pair is inconsistent — and the client's retry of the same conversation
//! would restore it. Split from `lifecycle_tests.rs` for the 500-line cap;
//! the stub model and the sequence builder live there.

use super::lifecycle::finish_sequence;
use super::lifecycle_tests::{CACHE_CALLS, FREE_CALLS, MAX_SEQ_LEN, StubModel, test_seq};
use super::types::ResponseSink;

fn calls() -> (usize, usize) {
    (CACHE_CALLS.with(|c| c.get()), FREE_CALLS.with(|c| c.get()))
}

/// An engine abort must never read as a model stop: the blocking client gets
/// an error payload (→ HTTP 500), not a completion with finish_reason "stop".
#[test]
fn engine_error_is_never_reported_as_stop() {
    let (mut a, mut rx) = test_seq(vec![5, 6, 42], 5, None, 10);
    a.engine_error = Some("boom".into());
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    let err = match rx
        .try_recv()
        .expect("finish_sequence must send the blocking response")
    {
        Err(e) => e,
        Ok(r) => panic!(
            "an engine error must surface as Err, got finish_reason={}",
            r.finish_reason
        ),
    };
    assert!(format!("{err:#}").contains("boom"), "{err:#}");
}

#[test]
fn a_clean_finish_caches_the_sequence_then_frees_it() {
    let (mut a, _rx) = test_seq(vec![5, 6, 42], 5, None, 10);
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    assert_eq!(calls(), (1, 1));
}

/// The case a check placed AT the caching line gets wrong: the blocking branch
/// `take()`s the engine error to build the response, so by then the field is
/// already `None`. The decision has to be read before the sink is served.
#[test]
fn an_engine_error_on_a_blocking_sink_skips_the_cache_and_still_frees() {
    let (mut a, _rx) = test_seq(vec![5, 6, 42], 5, None, 10);
    a.engine_error = Some("verify failed mid-forward".into());
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    assert_eq!(
        calls(),
        (0, 1),
        "a dead sequence left a finish leaf in the prefix cache"
    );
}

#[test]
fn an_engine_error_on_a_streaming_sink_skips_the_cache_and_still_frees() {
    let (mut a, _rx) = test_seq(vec![5, 6, 42], 5, None, 10);
    let (tx, _events) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    a.engine_error = Some("verify failed mid-forward".into());
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    assert_eq!(calls(), (0, 1));
}
