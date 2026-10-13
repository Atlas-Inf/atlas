// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_with_preemption`: the head's side of the agreed KV-refusal
//! retry, driven against the scripted stub, asserted on the EP wire.

use super::super::prefill_preempt::{prefill_chunk_with_preemption, resume_point};
use super::*;
use spark_model::model::kv_admission::kv_admission_refusal;

const CHUNK: u32 = 0xFFFFFFF0;
const RELEASE: u32 = 0xFFFFFFF1;

/// A 40-token prompt whose first 16 tokens are already prefilled.
fn prefilling() -> (Vec<u32>, SequenceState) {
    let prompt: Vec<u32> = (1..=40).collect();
    let mut seq = SequenceState::host_only(3);
    seq.tokens = prompt[..16].to_vec();
    seq.seq_len = 16;
    (prompt, seq)
}

fn holding(slot: usize, blocks: usize) -> ActiveSeq {
    let (mut a, _rx) = active_seq(slot, 4);
    a.seq.block_table = vec![0; blocks];
    a
}

fn run(
    model: &PreemptStubModel,
    seq: &mut SequenceState,
    active: &mut Vec<ActiveSeq>,
) -> Result<DevicePtr> {
    let (prompt, _) = prefilling();
    prefill_chunk_with_preemption(model, &prompt, seq, false, 16, 40, true, 0, active)
}

#[test]
fn resume_point_accepts_only_progress_inside_the_chunk() {
    assert_eq!(resume_point(0, 8192, 0).unwrap(), 0);
    assert_eq!(resume_point(4096, 8192, 7936).unwrap(), 7936);
    assert!(resume_point(4096, 8192, 4095).is_err());
    assert!(resume_point(4096, 8192, 8192).is_err());
}

#[test]
fn a_refused_split_half_resumes_at_the_cut_after_one_preemption() {
    let model = PreemptStubModel {
        refuse_chunks: AtomicUsize::new(1),
        split_cut: 32,
        ..Default::default()
    };
    let (prompt, mut seq) = prefilling();
    let mut active = vec![holding(0, 2), holding(1, 5)];
    run(&model, &mut seq, &mut active).unwrap();

    // The first half [16,32) ran once; the retry ran only [32,40).
    assert_eq!(*model.chunk_calls.lock().unwrap(), vec![(16, 24), (32, 8)]);
    assert_eq!(seq.tokens, prompt, "no token appended twice");
    assert_eq!(seq.seq_len, 40);
    // The largest holder was killed and its release mirrored before the
    // re-sent chunk, which starts at the cut on the worker too.
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert_eq!(active.len(), 1);
    assert_eq!(
        *model.wire.lock().unwrap(),
        vec![CHUNK, 24, 16, 40, RELEASE, CHUNK, 8, 32, 40]
    );
}

#[test]
fn a_same_worded_error_that_is_not_the_agreed_refusal_is_not_retried() {
    let model = PreemptStubModel {
        hard_error: Some("KV cache exhausted: no free blocks"),
        ..Default::default()
    };
    let (_, mut seq) = prefilling();
    let mut active = vec![holding(0, 2)];
    let e = run(&model, &mut seq, &mut active).unwrap_err();
    assert!(kv_admission_refusal(&e).is_none());
    assert_eq!(model.chunk_calls.lock().unwrap().len(), 1);
    assert!(model.freed_slots.lock().unwrap().is_empty());
    assert_eq!(active.len(), 1, "no victim for a non-retryable error");
}

#[test]
fn a_refusal_with_nothing_to_evict_fails_after_one_attempt() {
    let model = PreemptStubModel {
        refuse_chunks: AtomicUsize::new(9),
        ..Default::default()
    };
    let (_, mut seq) = prefilling();
    let e = run(&model, &mut seq, &mut Vec::new()).unwrap_err();
    assert!(kv_admission_refusal(&e).is_some_and(|r| r.retryable));
    assert_eq!(*model.wire.lock().unwrap(), vec![CHUNK, 24, 16, 40]);
    assert_eq!((seq.seq_len, seq.tokens.len()), (16, 16));
}

/// An agreed refusal for a reservation error other than exhaustion: every
/// rank rolled back, but no victim can cure it, so only this request fails.
#[test]
fn a_final_agreed_refusal_fails_this_request_without_a_victim() {
    let model = PreemptStubModel {
        refuse_chunks: AtomicUsize::new(1),
        refusal_is_final: true,
        ..Default::default()
    };
    let (_, mut seq) = prefilling();
    let mut active = vec![holding(0, 2), holding(1, 5)];
    let e = run(&model, &mut seq, &mut active).unwrap_err();
    assert!(kv_admission_refusal(&e).is_some_and(|r| !r.retryable));
    assert_eq!(*model.wire.lock().unwrap(), vec![CHUNK, 24, 16, 40]);
    assert!(model.freed_slots.lock().unwrap().is_empty());
    assert_eq!(active.len(), 2, "no victim for a final refusal");
}

#[test]
fn progress_outside_the_chunk_sends_nothing() {
    let model = PreemptStubModel::default();
    let (_, mut seq) = prefilling();
    seq.seq_len = 8;
    assert!(run(&model, &mut seq, &mut Vec::new()).is_err());
    assert!(model.wire.lock().unwrap().is_empty());
    assert!(model.chunk_calls.lock().unwrap().is_empty());
}
