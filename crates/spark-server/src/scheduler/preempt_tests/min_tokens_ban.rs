// SPDX-License-Identifier: AGPL-3.0-only

//! The min_tokens end-token ban must be armed on every way into decode, not
//! only on the multi-chunk promotion: a prompt that prefills in one chunk,
//! the single-shot prefill of an unchunked server, a preempted sequence's
//! re-prefill and a swapped-out sequence's restore decode the same request.

use super::super::sched_ctx::SchedCtx;
use super::super::test_support::{EOS, test_request};
use super::super::{StartPrefillResult, prefill_request, start_chunked_prefill};
use super::*;
use crate::api::InferenceRequest;
use spark_model::speculative::ProposerState;
use spark_model::traits::EosBan;

const MIN_TOKENS: usize = 8;

/// A drafter state that records the floor it was handed.
struct FloorProbe(usize);

impl ProposerState for FloorProbe {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_end_floor(&mut self, floor: usize) {
        self.0 = floor;
    }
}

fn drafter_floor(seq: &SequenceState) -> usize {
    let state = seq.proposer_state.as_ref().expect("drafter state");
    state.as_any().downcast_ref::<FloorProbe>().unwrap().0
}

/// The 3-token test prompt with `min_tokens` set.
fn request(min_tokens: usize) -> InferenceRequest {
    let (response_tx, _rx) = tokio::sync::oneshot::channel();
    let mut req = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking { min_tokens: m, .. } = &mut req {
        *m = min_tokens;
    }
    req
}

/// Start the test prompt with a chunk budget of `budget` tokens.
fn start(min_tokens: usize, budget: usize) -> StartPrefillResult {
    let req = request(min_tokens);
    let model = PreemptStubModel::default();
    let sched = SchedCtx::for_test();
    start_chunked_prefill(
        &sched, None, None, None, None, &model, req, EOS, budget, 0, 0, &mut None, 0, false, None,
        None,
    )
    .expect("prefill starts")
}

#[test]
fn a_single_chunk_prompt_arms_the_min_tokens_ban() {
    let StartPrefillResult::Active(a) = start(MIN_TOKENS, 64) else {
        panic!("a 3-token prompt within the chunk budget decodes at once");
    };
    assert_eq!(a.min_tokens, MIN_TOKENS);
    assert_eq!(a.seq.prompt_len, 3);
    assert_eq!(a.seq.eos_ban, EosBan::new(3, MIN_TOKENS, EOS));
    // The first token sits at position 3, so the floor is 11: a first 8-row
    // verify predicts 4..=11 and only its last row may end the turn.
    assert_eq!(a.seq.eos_ban.row_mask(a.seq.seq_len, 8), 0x7F);
}

#[test]
fn a_single_chunk_prompt_without_min_tokens_masks_no_row() {
    let StartPrefillResult::Active(a) = start(0, 64) else {
        panic!("a 3-token prompt within the chunk budget decodes at once");
    };
    assert_eq!(a.seq.eos_ban.floor, 0);
    assert_eq!(a.seq.eos_ban.row_mask(a.seq.seq_len, 8), 0);
}

#[test]
fn single_and_multi_chunk_prompts_get_the_same_ban() {
    let StartPrefillResult::Active(single) = start(MIN_TOKENS, 64) else {
        panic!("single chunk");
    };
    // A 2-token budget leaves the prompt in progress; it is promoted by the
    // same constructor once its last chunk lands.
    let StartPrefillResult::InProgress(mut p) = start(MIN_TOKENS, 2) else {
        panic!("multi chunk");
    };
    p.seq.proposer_state = Some(Box::new(FloorProbe(0)));
    let multi = super::super::phase_promote_prefills::build_active_seq_from_prefill(
        p,
        0,
        false,
        false,
        0,
        false,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        0,
        None,
    );
    assert_eq!(multi.seq.eos_ban, single.seq.eos_ban);
    // ... and the drafter skips end tokens below the same floor.
    assert_eq!(drafter_floor(&multi.seq), 3 + MIN_TOKENS);
}

#[test]
fn the_single_shot_prefill_arms_the_min_tokens_ban() {
    let model = PreemptStubModel::default();
    let sched = SchedCtx::for_test();
    let req = request(MIN_TOKENS);
    let a = prefill_request(
        &sched, None, None, None, None, &model, req, EOS, &mut None, 0, None,
    )
    .expect("prefill runs")
    .expect("a 3-token prompt decodes");
    assert_eq!(a.seq.eos_ban, EosBan::new(3, MIN_TOKENS, EOS));
}

#[test]
fn a_preempt_resume_keeps_the_min_tokens_ban() {
    let model = PreemptStubModel::default();
    let (mut a, _rx) = active_seq(3, 6);
    let ban = EosBan::new(4, 64, EOS);
    a.seq.eos_ban = ban;
    let resumed = resume_preempted_seq(&model, preempt_requeue(&model, a)).expect("resumes");
    // The re-prefill restamps prompt_len to the whole history; the floor is
    // an absolute position and must survive it.
    assert_eq!(resumed.seq.prompt_len, resumed.seq.tokens.len());
    assert_eq!(resumed.seq.eos_ban, ban);
}

#[test]
fn a_swap_resume_keeps_the_min_tokens_ban() {
    let model = PreemptStubModel::default();
    let (mut a, _rx) = active_seq(3, 6);
    let ban = EosBan::new(4, 64, EOS);
    a.seq.eos_ban = ban;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("atlas_min_tokens_swap_test_{nanos}"));
    let mut spill = KvSpillManager::new(dir, 1024 * 1024).unwrap();
    let swapped = match spill_out_sequence(&model, a, &mut spill) {
        Ok(swapped) => swapped,
        Err((_active, error)) => panic!("swap out failed: {error:#}"),
    };
    // The restore allocates a fresh sequence: the ban travels with the image.
    let resumed = resume_swapped_seq(None, None, &model, swapped, &mut spill).unwrap();
    assert_eq!(resumed.seq.eos_ban, ban);
}
