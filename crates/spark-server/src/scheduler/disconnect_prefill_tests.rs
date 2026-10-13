// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::cancel_test_model::TestModel;
use crate::scheduler::prefill_a_step_params::build_prefill_in_progress;
use crate::scheduler::types::ResponseSink;
use spark_model::traits::SequenceState;
use std::time::Instant;

fn prefill(sink: ResponseSink) -> PrefillInProgress {
    build_prefill_in_progress(
        std::sync::Arc::new(vec![1, 2, 3]),
        0,
        SequenceState::host_only(0),
        0,
        16,
        0,
        vec![2],
        sink,
        None,
        Instant::now(),
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
        false,
        None,
        None,
        0,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
    )
}

#[test]
fn dead_blocking_prefill_retires_and_live_one_survives() {
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    let (dead_tx, dead_rx) = tokio::sync::oneshot::channel();
    drop(dead_rx);
    let (live_tx, mut live_rx) = tokio::sync::oneshot::channel();
    let mut prefilling = vec![
        prefill(ResponseSink::Blocking(Some(dead_tx))),
        prefill(ResponseSink::Blocking(Some(live_tx))),
    ];
    assert_eq!(retire_disconnected_prefills(&model, &mut prefilling), 1);
    assert_eq!(prefilling.len(), 1, "only the dead prefill is dropped");
    assert!(!prefilling[0].sink.receiver_closed());
    assert!(live_rx.try_recv().is_err(), "live receiver untouched");
}

#[test]
fn streaming_prefill_is_never_retired_by_disconnect_sweep() {
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let mut prefilling = vec![prefill(ResponseSink::Streaming(tx))];
    assert_eq!(retire_disconnected_prefills(&model, &mut prefilling), 0);
    assert_eq!(prefilling.len(), 1);
}
