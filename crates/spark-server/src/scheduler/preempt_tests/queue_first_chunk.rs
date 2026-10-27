// SPDX-License-Identifier: AGPL-3.0-only
// Ported from Enntity/atlas dc896cc, f1cc7cb, 7fd2412 (AGPL-3.0-only, sparkqwen queued-first-chunk series).

//! Queued first chunks (`ATLAS_PREFILL_QUEUE_FIRST_CHUNK`) against the
//! scripted model: the real start and continue phases, asserted on the order
//! the head runs chunks in and on the EP worker's command stream.

use std::sync::Arc;

use super::super::levers::SchedLevers;
use super::super::limits::SchedLimits;
use super::super::phase_continue_prefills::{SpecStep, continue_in_progress_prefills};
use super::super::phase_start_prefills::start_new_requests;
use super::super::sched_ctx::SchedCtx;
use super::super::test_support::{EOS, RespRx, test_request};
use super::super::types::PrefillInProgress;
use super::*;
use crate::api::InferenceRequest;
use crate::scheduling_policy::FifoPolicy;

const CHUNK: u32 = 0xFFFF_FFF0;
const RELEASE: u32 = 0xFFFF_FFF1;
/// The chunk budget: a longer prompt takes more than one chunk.
const BUDGET: usize = 16;

fn multi_rank() -> PreemptStubModel {
    PreemptStubModel {
        ep: true,
        ..Default::default()
    }
}

fn sched(queue: bool, srpt: bool) -> SchedCtx {
    let mut levers = SchedLevers::defaults();
    levers.prefill_queue_first_chunk = queue;
    levers.prefill_srpt = srpt;
    SchedCtx::new(
        Default::default(),
        Arc::new(levers),
        Default::default(),
        SchedLimits::NONE,
        Default::default(),
    )
}

/// A buffered request for a `len`-token prompt. `min_tokens` bans the end
/// token from the first sample, which the stub then draws on the host.
fn request(len: usize) -> (InferenceRequest, RespRx) {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    let mut req = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking {
        prompt_tokens,
        min_tokens,
        ..
    } = &mut req
    {
        *prompt_tokens = Arc::new((10..10 + len as u32).collect());
        *min_tokens = 8;
    }
    (req, rx)
}

/// The same prompt, streamed; returns the token receiver.
fn streaming(
    len: usize,
) -> (
    InferenceRequest,
    tokio::sync::mpsc::Receiver<crate::api::StreamEvent>,
) {
    let (token_tx, token_rx) = tokio::sync::mpsc::channel(16);
    let mut req = test_request!(Streaming, token_tx,
        cancel_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),);
    if let InferenceRequest::Streaming {
        prompt_tokens,
        min_tokens,
        ..
    } = &mut req
    {
        *prompt_tokens = Arc::new((10..10 + len as u32).collect());
        *min_tokens = 8;
    }
    (req, token_rx)
}

/// The scheduler's prefill state across ticks.
struct Tick<'a> {
    model: &'a PreemptStubModel,
    sched: SchedCtx,
    active: Vec<ActiveSeq>,
    prefilling: Vec<PrefillInProgress>,
}

impl<'a> Tick<'a> {
    fn new(model: &'a PreemptStubModel, queue: bool, srpt: bool) -> Self {
        Self {
            model,
            sched: sched(queue, srpt),
            active: Vec::new(),
            prefilling: Vec::new(),
        }
    }

    /// One tick's start phase.
    fn admit(&mut self, reqs: Vec<InferenceRequest>) {
        let plants = vec![None; reqs.len()];
        start_new_requests(
            self.model,
            &self.sched,
            reqs,
            plants,
            true,
            false,
            BUDGET,
            BUDGET,
            EOS,
            0,
            0,
            &mut None,
            0,
            None,
            None,
            None,
            None,
            &mut self.active,
            &mut self.prefilling,
        );
    }

    /// One tick's continue phase.
    fn step(&mut self) {
        let spec = SpecStep {
            num_drafts: 0,
            dflash_verify_raw_argmax: false,
        };
        let mut rode = Vec::new();
        continue_in_progress_prefills(
            self.model,
            &FifoPolicy,
            &mut self.active,
            &mut self.prefilling,
            BUDGET,
            BUDGET,
            false,
            0,
            0,
            false,
            false,
            false,
            None,
            None,
            None,
            None,
            false,
            &self.sched,
            &spec,
            &mut rode,
        );
    }

    /// `(prompt tokens, tokens done, queued first chunk)` per prefill, in
    /// service order.
    fn queue(&self) -> Vec<(usize, usize, Option<usize>)> {
        self.prefilling
            .iter()
            .map(|p| (p.prompt_tokens.len(), p.chunk_offset, p.queued_first_chunk))
            .collect()
    }
}

fn wire(model: &PreemptStubModel) -> Vec<u32> {
    model.wire.lock().unwrap().clone()
}

fn chunks(model: &PreemptStubModel) -> Vec<(usize, usize)> {
    model.chunk_calls.lock().unwrap().clone()
}

/// The 40-token prompt in flight at 16 of 40 and a 36-token newcomer: the
/// newcomer's first chunk waits for the older prompt's tail, and its
/// command reaches the worker only when the head runs that chunk.
#[test]
fn a_newcomer_waits_for_the_tail_of_the_prefill_in_flight() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, false);
    let (a, _ra) = request(40);
    let (b, _rb) = request(36);
    t.admit(vec![a]);
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40]);
    t.admit(vec![b]);
    assert_eq!(t.queue(), vec![(40, 16, None), (36, 0, Some(16))]);
    assert_eq!(
        wire(&model),
        vec![CHUNK, 16, 0, 40],
        "no command at admission"
    );
    for _ in 0..5 {
        t.step();
    }
    // The tail first, then the newcomer's chunks from 0 — never its whole
    // prompt at once (the two-phase prefill sends the worker nothing).
    assert_eq!(
        chunks(&model),
        vec![(0, 16), (16, 16), (32, 8), (0, 16), (16, 16), (32, 4)]
    );
    assert_eq!(
        wire(&model),
        vec![
            CHUNK, 16, 0, 40, CHUNK, 16, 16, 40, CHUNK, 8, 32, 40, CHUNK, 16, 0, 36, CHUNK, 16, 16,
            36, CHUNK, 4, 32, 36,
        ]
    );
    assert!(t.prefilling.is_empty());
    assert_eq!(t.active.len(), 2, "both decode");
}

/// Off (the default), the newcomer's first chunk runs at admission, ahead of
/// the older prompt's tail, exactly as before.
#[test]
fn off_by_default_the_first_chunk_runs_at_admission() {
    let model = multi_rank();
    let mut t = Tick::new(&model, false, false);
    let (a, _ra) = request(40);
    let (b, _rb) = request(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    assert_eq!(t.queue(), vec![(40, 16, None), (36, 16, None)]);
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40, CHUNK, 16, 0, 36]);
    t.step();
    assert_eq!(chunks(&model), vec![(0, 16), (0, 16), (16, 16)]);
}

/// One GPU keeps the inline chunk with the lever on: its batched prefill
/// steps advance every prompt together, and nothing waits on a worker.
#[test]
fn one_gpu_keeps_the_inline_first_chunk() {
    let model = PreemptStubModel::default();
    let mut t = Tick::new(&model, true, false);
    let (a, _ra) = request(40);
    let (b, _rb) = request(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    assert_eq!(t.queue(), vec![(40, 16, None), (36, 16, None)]);
    assert_eq!(chunks(&model), vec![(0, 16), (0, 16)]);
}

/// A burst admitted in one tick with nothing in flight: the first prompt
/// runs its first chunk at admission, the rest queue behind it in arrival
/// order (on the cold pair all four first chunks used to run back to back).
#[test]
fn a_burst_runs_its_first_chunks_in_arrival_order() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, false);
    let (a, _ra) = request(40);
    let (b, _rb) = request(36);
    let (c, _rc) = request(20);
    t.admit(vec![a, b, c]);
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40]);
    assert_eq!(
        t.queue(),
        vec![(40, 16, None), (36, 0, Some(16)), (20, 0, Some(16))]
    );
}

/// With `ATLAS_PREFILL_SRPT=1` a short tail goes ahead of a new long
/// prompt, and a short newcomer ahead of a long remainder.
#[test]
fn srpt_orders_queued_first_chunks_by_tokens_left() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, true);
    let (a, _ra) = request(20);
    let (b, _rb) = request(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    assert_eq!(t.queue(), vec![(20, 16, None), (36, 0, Some(16))]);
    t.step();
    assert_eq!(
        chunks(&model),
        vec![(0, 16), (16, 4)],
        "the 4-token tail first"
    );

    let model = multi_rank();
    let mut t = Tick::new(&model, true, true);
    let (a, _ra) = request(40);
    let (c, _rc) = request(3);
    t.admit(vec![a]);
    t.admit(vec![c]);
    assert_eq!(t.queue(), vec![(3, 0, Some(3)), (40, 16, None)]);
    t.step();
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40, CHUNK, 3, 0, 3]);
    assert_eq!(t.active.len(), 1, "the 3-token prompt decodes");
    assert_eq!(t.queue(), vec![(40, 16, None)]);
}

/// A queued request whose caller left frees its slot on every rank, and the
/// worker never sees a prefill command for it.
#[test]
fn a_queued_request_whose_caller_left_is_freed_without_a_chunk() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, false);
    let (a, _ra) = request(40);
    let (b, rb) = request(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    drop(rb);
    t.step();
    assert_eq!(t.queue(), vec![(40, 16, None)]);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![0]);
    t.step();
    t.step();
    assert_eq!(
        wire(&model),
        vec![
            CHUNK, 16, 0, 40, RELEASE, CHUNK, 16, 16, 40, CHUNK, 8, 32, 40
        ]
    );
}

/// A queued streaming request whose client went away leaves the queue too;
/// one already prefilling keeps the streaming contract (`cancel_flag`).
#[test]
fn a_queued_stream_whose_client_left_is_freed_without_a_chunk() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, false);
    let (a, ra) = streaming(40);
    let (b, rb) = streaming(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    drop((ra, rb));
    t.step();
    assert_eq!(t.queue(), vec![(40, 16, None)]);
    t.step();
    assert_eq!(
        wire(&model),
        vec![CHUNK, 16, 0, 40, RELEASE, CHUNK, 16, 16, 40]
    );
}

/// A refused queued first chunk fails only its request, as an inline one
/// does: no victim, and the release follows the refused command.
#[test]
fn a_refused_queued_first_chunk_fails_only_its_request() {
    let model = multi_rank();
    let mut t = Tick::new(&model, true, false);
    let (a, _ra) = request(20);
    let (b, mut rb) = request(36);
    t.admit(vec![a]);
    t.admit(vec![b]);
    t.step();
    assert_eq!(t.active.len(), 1);
    model.refuse_chunks.store(1, Ordering::SeqCst);
    t.step();
    assert!(rb.try_recv().expect("response sent").is_err());
    assert!(t.prefilling.is_empty());
    assert_eq!(t.active.len(), 1, "the decoding request is no victim");
    assert_eq!(
        wire(&model),
        vec![
            CHUNK, 16, 0, 20, CHUNK, 4, 16, 20, CHUNK, 16, 0, 36, RELEASE
        ]
    );
}
