// SPDX-License-Identifier: AGPL-3.0-only

//! `requeue_kv_refusals`: a speculative step refused for KV on every rank
//! preempts a victim instead of failing its sequences, and fails a sequence
//! only when nothing else can give up blocks.

use super::super::preempt::requeue_kv_refusals;
use super::*;

/// The engine error a step records for an agreed, retryable decode refusal.
fn refusal_text() -> String {
    let e: anyhow::Error = KvAdmissionRefused {
        by_peer: true,
        retryable: true,
        decode: true,
    }
    .into();
    format!("{:#}", e.context("decode_verify_batched"))
}

fn refused(a: &mut ActiveSeq) {
    a.engine_error = Some(refusal_text());
    a.finished = true;
}

#[test]
fn a_refused_batch_preempts_the_least_progress_sequence_and_retries_the_rest() {
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 9);
    let (mut a1, mut r1) = streaming_seq(1, 2); // least progress
    let (mut a2, _r2) = active_seq(2, 5);
    for a in [&mut a0, &mut a1, &mut a2] {
        refused(a);
    }
    let mut active = vec![a0, a1, a2];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());

    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);

    assert_eq!(preempted.len(), 1);
    assert_eq!(preempted[0].a.seq.slot_idx, 1);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    // Release reached the worker.
    assert_eq!(*model.wire.lock().unwrap(), vec![0xFFFF_FFF1]);
    // The rest retry next tick: running, no error recorded.
    assert_eq!(active.len(), 2);
    assert!(
        active
            .iter()
            .all(|a| !a.finished && a.engine_error.is_none())
    );
    // The victim's stream stays open and silent (resume, not kill).
    assert!(matches!(
        r1.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn a_running_sequence_can_be_the_victim_of_another_ones_refusal() {
    let model = PreemptStubModel::default();
    let (mut big, _r0) = active_seq(0, 40);
    refused(&mut big);
    let (small, _r1) = active_seq(1, 3);
    let mut active = vec![big, small];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);
    assert_eq!(preempted.len(), 1);
    assert_eq!(preempted[0].a.seq.slot_idx, 1);
    assert_eq!(active.len(), 1);
    assert!(!active[0].finished && active[0].engine_error.is_none());
}

#[test]
fn a_lone_sequence_that_outgrew_the_pool_finishes_with_the_error() {
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 4);
    refused(&mut a0);
    let mut active = vec![a0];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(preempted.is_empty() && swapped.is_empty());
    assert!(active[0].finished);
    let e = active[0].engine_error.as_deref().unwrap();
    assert!(e.contains("KV cache exhausted"), "{e}");
    assert!(model.freed_slots.lock().unwrap().is_empty());
}

#[test]
fn other_engine_errors_and_finished_sequences_are_left_alone() {
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 4);
    a0.engine_error = Some("CUDA error 700: illegal memory access".into());
    a0.finished = true;
    // An unagreed exhaustion (raised mid-forward on one rank) is not one.
    let (mut a1, _r1) = active_seq(1, 2);
    a1.engine_error = Some("KV cache exhausted: no free blocks".into());
    a1.finished = true;
    let (a2, _r2) = active_seq(2, 1);
    let mut active = vec![a0, a1, a2];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(preempted.is_empty());
    assert_eq!(active.len(), 3);
    assert!(active[0].finished && active[1].finished && !active[2].finished);
}

#[test]
fn a_refusal_whose_only_other_sequence_is_finished_fails_alone() {
    // A finished sequence holds blocks only until retirement; it is not a
    // victim, so the refused one is the lone running sequence.
    let model = PreemptStubModel::default();
    let (mut done, _r0) = active_seq(0, 1);
    done.finished = true;
    let (mut a1, _r1) = active_seq(1, 6);
    refused(&mut a1);
    let mut active = vec![done, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(preempted.is_empty());
    assert!(active[1].finished && active[1].engine_error.is_some());
}

#[test]
fn a_bootstrap_refusal_after_part_of_the_batch_ran_is_not_requeued() {
    // `decode_batch` under the batched MTP bootstrap: a refusal raised after
    // some sequences already ran is final for the sequences it covered.
    let model = PreemptStubModel::default();
    let e: anyhow::Error = KvAdmissionRefused {
        by_peer: true,
        retryable: true,
        decode: true,
    }
    .into();
    let text = format!(
        "{:#}",
        spark_model::model::kv_admission::after_partial_step(e, 1, 3)
            .context("decode_mtp_bootstrap_batched")
    );
    let mut active = Vec::new();
    let mut rxs = Vec::new();
    for slot in 0..3 {
        let (mut a, rx) = active_seq(slot, 2 + slot);
        a.engine_error = Some(text.clone());
        a.finished = true;
        active.push(a);
        rxs.push(rx);
    }
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    requeue_kv_refusals(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(preempted.is_empty() && swapped.is_empty());
    assert!(
        active
            .iter()
            .all(|a| a.finished && a.engine_error.is_some())
    );
    assert!(model.freed_slots.lock().unwrap().is_empty());
}
