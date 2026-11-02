// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`phase_start_prefills::packed`) against the
//! scripted model: which prompts of a tick share a packed pass, that the
//! rest take the ordinary chunk path, and that nothing eligible is left to
//! the ordinary path when one pass is full.

use std::sync::Arc;

use super::super::phase_start_prefills::start_new_requests;
use super::super::sched_ctx::SchedCtx;
use super::super::test_support::{EOS, test_request};
use super::super::types::{ActiveSeq, PrefillInProgress};
use super::*;
use crate::api::InferenceRequest;

const PREFILL: u32 = 0xFFFF_FFF0;

fn request(tokens: Vec<u32>) -> InferenceRequest {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    std::mem::forget(rx); // keep the caller "connected"
    let mut req = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking { prompt_tokens, .. } = &mut req {
        *prompt_tokens = Arc::new(tokens);
    }
    req
}

fn prompt(id: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 * (id + 1) + i).collect()
}

/// One tick of `start_new_requests` over `prompts`; returns what started.
fn tick(
    model: &PreemptStubModel,
    prompts: Vec<Vec<u32>>,
) -> (Vec<ActiveSeq>, Vec<PrefillInProgress>) {
    let sched = SchedCtx::for_test();
    let n = prompts.len();
    let (mut active, mut prefilling) = (Vec::new(), Vec::new());
    start_new_requests(
        model,
        &sched,
        prompts.into_iter().map(request).collect(),
        vec![None; n],
        true,
        false,
        64,
        64,
        EOS,
        0,
        0,
        &mut None,
        0,
        None,
        None,
        None,
        None,
        &mut active,
        &mut prefilling,
    );
    (active, prefilling)
}

#[test]
fn eligible_prompts_share_one_packed_pass_and_the_rest_prefill_as_before() {
    let model = PreemptStubModel {
        packed: Some((8192, 16)),
        ..Default::default()
    };
    let mut hit = prompt(2, 6);
    hit.push(PACKED_HIT);
    let prompts = vec![prompt(0, 5), prompt(1, 9), hit.clone(), prompt(3, 4)];
    let (active, prefilling) = tick(&model, prompts.clone());
    // The three admitted prompts, in arrival order, in one pass.
    assert_eq!(
        *model.packed_calls.lock().unwrap(),
        vec![vec![
            prompts[0].clone(),
            prompts[1].clone(),
            prompts[3].clone()
        ]]
    );
    // The cache hit took the ordinary chunk path, and only it.
    assert_eq!(*model.chunk_calls.lock().unwrap(), vec![(0, hit.len())]);
    let wire = model.wire.lock().unwrap();
    assert_eq!(wire.iter().filter(|&&w| w == PREFILL).count(), 1);
    // Every prompt prefilled in this tick and decodes next.
    assert!(prefilling.is_empty());
    assert_eq!(active.len(), 4);
    for a in &active {
        assert!(a.seq.seq_len >= 4 && a.seq.seq_len == a.seq.prompt_len);
    }
}

#[test]
fn a_lone_prompt_is_packed_too() {
    let model = PreemptStubModel {
        packed: Some((8192, 16)),
        ..Default::default()
    };
    let (active, _) = tick(&model, vec![prompt(0, 7)]);
    assert_eq!(model.packed_calls.lock().unwrap().len(), 1);
    assert!(model.chunk_calls.lock().unwrap().is_empty());
    assert_eq!(active.len(), 1);
}

#[test]
fn a_full_pass_spills_into_another_packed_pass() {
    // At most 10 rows and two prompts per pass.
    let model = PreemptStubModel {
        packed: Some((10, 2)),
        ..Default::default()
    };
    let prompts = vec![prompt(0, 4), prompt(1, 4), prompt(2, 4), prompt(3, 6)];
    let (active, prefilling) = tick(&model, prompts.clone());
    assert_eq!(
        *model.packed_calls.lock().unwrap(),
        vec![
            vec![prompts[0].clone(), prompts[1].clone()],
            vec![prompts[2].clone(), prompts[3].clone()],
        ]
    );
    assert!(model.chunk_calls.lock().unwrap().is_empty());
    assert!(prefilling.is_empty());
    assert_eq!(active.len(), 4);
}

#[test]
fn off_nothing_is_packed() {
    let model = PreemptStubModel::default();
    let (active, _) = tick(&model, vec![prompt(0, 5), prompt(1, 6)]);
    assert!(model.packed_calls.lock().unwrap().is_empty());
    assert_eq!(model.chunk_calls.lock().unwrap().len(), 2);
    assert_eq!(active.len(), 2);
}
