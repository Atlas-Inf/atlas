// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-prefix admission against the scripted model: what the head puts on
//! the worker's command stream, and what goes back to the pending queue.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};
use spark_model::traits::EP_CMD_PC_PLANT;

use super::super::prefill_a_step_params::build_prefill_in_progress;
use super::super::sched_ctx::SchedCtx;
use super::super::shared_prefix::{SharedPrefix, admit};
use super::super::test_support::{EOS, test_request};
use super::super::types::{PendingQueue, PrefillInProgress};
use super::super::{StartPrefillResult, start_chunked_prefill};
use super::*;
use crate::api::InferenceRequest;

const PREFILL: u32 = 0xFFFF_FFF0;
const SHARED: usize = 4_096;

/// A conversation: `SHARED` common tokens, then `suffix` tokens unique to `id`.
fn convo(id: u32, suffix: usize) -> Vec<u32> {
    let mut t: Vec<u32> = (0..SHARED as u32).map(|i| i + 7).collect();
    t.extend((0..suffix as u32).map(|i| 1_000_000 * (id + 1) + i));
    t
}

fn request(tokens: Vec<u32>) -> InferenceRequest {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    std::mem::forget(rx); // keep the caller "connected"
    let mut req = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking { prompt_tokens, .. } = &mut req {
        *prompt_tokens = Arc::new(tokens);
    }
    req
}

fn in_flight(tokens: Vec<u32>, done: usize) -> PrefillInProgress {
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::mem::forget(rx);
    let mut seq = SequenceState::host_only(3);
    seq.seq_len = done;
    build_prefill_in_progress(
        Arc::new(tokens),
        0,
        seq,
        done,
        16,
        0,
        vec![2],
        ResponseSink::Blocking(Some(tx)),
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

fn queue() -> Arc<(Mutex<PendingQueue>, Condvar)> {
    let q = PendingQueue {
        requests: Vec::new(),
        closed: false,
        rotations: Vec::new(),
    };
    Arc::new((Mutex::new(q), Condvar::new()))
}

fn tokens_of(req: &InferenceRequest) -> Vec<u32> {
    req.prompt_tokens_arc().as_ref().clone()
}

fn model() -> PreemptStubModel {
    PreemptStubModel {
        inflight_min: Some(2048),
        ..Default::default()
    }
}

/// A new leader's plant reaches the worker before its chunk-0 command, so
/// the worker's prefill of that chunk already knows it.
#[test]
fn a_new_leader_sends_its_plant_before_the_first_chunk() {
    let model = model();
    let sched = SchedCtx::for_test();
    let req = request(convo(0, 200));
    let res = start_chunked_prefill(
        &sched,
        None,
        None,
        None,
        None,
        &model,
        req,
        EOS,
        1_024,
        0,
        0,
        &mut None,
        0,
        false,
        None,
        None,
        Some(SHARED),
    )
    .expect("prefill starts");
    let StartPrefillResult::InProgress(p) = res else {
        panic!("a 4,296-token prompt at 1,024-token chunks is in progress");
    };
    assert_eq!(p.seq.pc_plant_at, Some(SHARED));
    let wire = model.wire.lock().unwrap().clone();
    let plant = wire.iter().position(|&w| w == EP_CMD_PC_PLANT).unwrap();
    let chunk = wire.iter().position(|&w| w == PREFILL).unwrap();
    assert!(plant < chunk, "plant after the chunk command: {wire:?}");
    assert_eq!(wire[plant + 1], SHARED as u32);
}

/// Followers of a prefill in flight go back to the front of the queue in
/// arrival order; the leader is planted once, over the command stream; an
/// unrelated request starts at once.
#[test]
fn followers_wait_and_the_leader_in_flight_is_planted() {
    let model = model();
    let pending = queue();
    let mut prefilling = vec![in_flight(convo(0, 200), 1_024)];
    let other: Vec<u32> = (0..5_000).map(|i| 900_000 + i).collect();
    let new = vec![
        request(convo(1, 50)),
        request(other.clone()),
        request(convo(2, 80)),
    ];
    let mut state = SharedPrefix::default();
    let (now, plants) = admit(&model, &pending, new, &mut prefilling, true, &mut state);
    assert_eq!(now.iter().map(tokens_of).collect::<Vec<_>>(), vec![other]);
    assert_eq!(plants, vec![None]);
    let held: Vec<_> = pending.0.lock().requests.iter().map(tokens_of).collect();
    assert_eq!(held, vec![convo(1, 50), convo(2, 80)]);
    assert_eq!(prefilling[0].seq.pc_plant_at, Some(SHARED));
    assert_eq!(
        *model.wire.lock().unwrap(),
        vec![EP_CMD_PC_PLANT, SHARED as u32]
    );

    // Next tick, still short of the checkpoint: held again, nothing resent.
    let again: Vec<_> = std::mem::take(&mut pending.0.lock().requests);
    prefilling[0].chunk_offset = 2_048;
    let (now, _) = admit(&model, &pending, again, &mut prefilling, true, &mut state);
    assert!(now.is_empty());
    assert_eq!(pending.0.lock().requests.len(), 2);
    assert_eq!(model.wire.lock().unwrap().len(), 2);

    // Past it: both start, and neither waits on the other.
    let again: Vec<_> = std::mem::take(&mut pending.0.lock().requests);
    prefilling[0].chunk_offset = SHARED + 64;
    let (now, plants) = admit(&model, &pending, again, &mut prefilling, true, &mut state);
    assert_eq!(now.len(), 2);
    assert_eq!(plants, vec![None, None]);
    assert!(pending.0.lock().requests.is_empty());
}

/// Off (the default), or on a single-rank world, nothing is held or sent.
#[test]
fn off_by_default_nothing_is_held() {
    let model = PreemptStubModel::default();
    let pending = queue();
    let mut prefilling = vec![in_flight(convo(0, 200), 1_024)];
    let new = vec![request(convo(1, 50)), request(convo(2, 80))];
    let mut state = SharedPrefix::default();
    let (now, plants) = admit(&model, &pending, new, &mut prefilling, true, &mut state);
    assert_eq!(now.len(), 2);
    assert_eq!(plants, vec![None, None]);
    assert!(pending.0.lock().requests.is_empty());
    assert!(model.wire.lock().unwrap().is_empty());
    // Unchunked prefill never plants either.
    let model = self::model();
    let new = vec![request(convo(1, 50)), request(convo(2, 80))];
    let (now, _) = admit(&model, &pending, new, &mut prefilling, false, &mut state);
    assert_eq!(now.len(), 2);
}
