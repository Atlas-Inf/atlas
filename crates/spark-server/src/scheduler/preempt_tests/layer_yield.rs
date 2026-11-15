// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill passes that yield to decoders (`ATLAS_GLM_PREFILL_YIELD_MS`,
//! `phase_continue_prefills::layer_yield`) against the scripted model: the
//! real start and continue phases, asserted on the chunk calls the head
//! makes and on the EP worker's command stream.

use std::sync::Arc;
use std::time::Duration;

use spark_model::traits::EP_CMD_PREFILL_SLICE as SLICE;

use super::super::levers::SchedLevers;
use super::super::limits::SchedLimits;
use super::super::phase_continue_prefills::layer_yield::Paused;
use super::super::sched_ctx::SchedCtx;
use super::queue_first_chunk::{CHUNK, Tick, chunks, request, wire};
use super::*;

/// Layers of every pass of the scripted model.
const LAYERS: usize = 6;

/// A pair whose passes can stop between layers.
fn pair() -> PreemptStubModel {
    PreemptStubModel {
        ep: true,
        pass_layers: LAYERS,
        ..Default::default()
    }
}

/// A run with the yield on (`srpt`: `ATLAS_PREFILL_SRPT`).
fn yielding<'a>(model: &'a PreemptStubModel, srpt: bool) -> Tick<'a> {
    let mut levers = SchedLevers::defaults();
    levers.prefill_yield = Some(Duration::from_millis(300));
    levers.prefill_srpt = srpt;
    let mut t = Tick::new(model, false, srpt);
    t.sched = SchedCtx::new(
        Default::default(),
        Arc::new(levers),
        Default::default(),
        SchedLimits::NONE,
        Default::default(),
    );
    t
}

/// The scripted `prefill_chunk_slice`: a pass of `pass_layers` layers that
/// stops when the call's `layers` are spent and ends as `prefill_chunk` does.
pub(super) fn slice(
    model: &PreemptStubModel,
    tokens: &[u32],
    seq: &mut SequenceState,
    (start, len): (usize, usize),
    last: bool,
    stream: u64,
    layers: Option<usize>,
) -> Result<spark_model::traits::ChunkStep> {
    use spark_model::traits::{ChunkStep, PausedPass};
    model.slice_calls.lock().unwrap().push((start, len, layers));
    // A refusal comes before any layer runs.
    if seq.paused_pass.is_none() && model.refuse_chunks.load(Ordering::SeqCst) > 0 {
        return model
            .prefill_chunk(tokens, seq, start, len, last, stream)
            .map(ChunkStep::Done);
    }
    let at = match seq.paused_pass.take() {
        Some(p) => {
            anyhow::ensure!(
                p.span == (start, len),
                "another chunk while {p:?} is paused"
            );
            p.next_layer
        }
        None => 0,
    };
    let ran = layers
        .unwrap_or(usize::MAX)
        .min(model.pass_layers.max(1) - at);
    if model.pass_layers > 0 && at + ran < model.pass_layers {
        seq.paused_pass = Some(Box::new(PausedPass {
            span: (start, len),
            next_layer: at + ran,
            proc: (start, len),
            effective_seq_len_start: start,
            kv_write_start: 0,
            marconi_skip: false,
            highway: Vec::new(),
            token_ids: Vec::new(),
        }));
        return Ok(ChunkStep::Paused { ran });
    }
    model
        .prefill_chunk(tokens, seq, start, len, last, stream)
        .map(ChunkStep::Done)
}

fn slices(model: &PreemptStubModel) -> Vec<(usize, usize, Option<usize>)> {
    model.slice_calls.lock().unwrap().clone()
}

/// One sequence decoding in slot 7.
fn decoding(t: &mut Tick<'_>) -> super::super::test_support::RespRx {
    let (a, rx) = active_seq(7, 3);
    t.active.push(a);
    rx
}

/// Beside a decoder a new prompt's chunk is the continue phase's to run, in
/// slices: a probe, then the rest at the measured pace, each preceded on the
/// wire by its layer count, and the prompt advances only when its pass ends.
#[test]
fn a_chunk_beside_a_decoder_runs_in_slices() {
    let model = pair();
    let mut t = yielding(&model, false);
    let _rx = decoding(&mut t);
    let (a, _ra) = request(12);
    t.admit(vec![a]);
    assert_eq!(t.queue(), vec![(12, 0, Some(12))], "queued, not run");
    assert!(wire(&model).is_empty(), "no command at admission");

    t.step();
    assert_eq!(slices(&model), vec![(0, 12, Some(2))], "the probe");
    assert_eq!(wire(&model), vec![SLICE, 2, CHUNK, 12, 0, 12]);
    assert_eq!(t.queue(), vec![(12, 0, Some(12))], "still its first chunk");
    assert_eq!(t.prefilling[0].paused.map(|p| p.ran), Some(2));
    assert!(chunks(&model).is_empty(), "the pass has not ended");

    // The probe's two layers took 100 ms: six fit the 300 ms target.
    t.prefilling[0].paused = Some(Paused {
        ran: 2,
        took: Duration::from_millis(100),
    });
    t.step();
    assert_eq!(slices(&model), vec![(0, 12, Some(2)), (0, 12, Some(6))]);
    assert_eq!(
        wire(&model),
        vec![SLICE, 2, CHUNK, 12, 0, 12, SLICE, 6, CHUNK, 12, 0, 12],
        "the same chunk command again",
    );
    assert_eq!(chunks(&model), vec![(0, 12)], "one pass, run once");
    assert!(t.prefilling.is_empty());
    assert_eq!(t.active.len(), 2, "the prompt decodes beside the decoder");
}

/// With nothing decoding there is nobody to yield to: the first chunk runs
/// at admission and later ones whole, with no slice word.
#[test]
fn with_nothing_decoding_chunks_run_whole() {
    let model = pair();
    let mut t = yielding(&model, false);
    let (a, _ra) = request(40);
    t.admit(vec![a]);
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40]);
    t.step();
    assert_eq!(slices(&model), vec![(16, 16, None)]);
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40, CHUNK, 16, 16, 40]);
    assert_eq!(t.queue(), vec![(40, 32, None)]);
}

/// The decoders finished while the pass was paused: the next call runs the
/// rest of it at once, and tells the worker so by sending no slice word.
#[test]
fn a_paused_pass_runs_out_when_its_decoders_are_gone() {
    let model = pair();
    let mut t = yielding(&model, false);
    let _rx = decoding(&mut t);
    let (a, _ra) = request(12);
    t.admit(vec![a]);
    t.step();
    assert!(t.prefilling[0].paused.is_some());
    t.active.clear();
    t.step();
    assert_eq!(slices(&model), vec![(0, 12, Some(2)), (0, 12, None)]);
    assert_eq!(
        wire(&model),
        vec![SLICE, 2, CHUNK, 12, 0, 12, CHUNK, 12, 0, 12]
    );
    assert_eq!(chunks(&model), vec![(0, 12)]);
    assert!(t.prefilling.is_empty());
}

/// A paused pass holds the arena, so its prompt runs next whatever stands
/// ahead of it in the queue.
#[test]
fn the_paused_prompt_runs_before_one_queued_ahead_of_it() {
    let model = pair();
    let mut t = yielding(&model, true);
    let _rx = decoding(&mut t);
    let (long, _rl) = request(40);
    t.admit(vec![long]);
    t.step();
    assert!(t.prefilling[0].paused.is_some());
    // The scheduler admits nothing while a pass is paused; a prompt that got
    // ahead of it all the same (shortest first) must still wait.
    let (short, _rs) = request(10);
    t.admit(vec![short]);
    assert_eq!(t.queue(), vec![(10, 0, Some(10)), (40, 0, Some(16))]);
    t.step();
    assert_eq!(chunks(&model), vec![(0, 16)], "the paused pass, to its end");
    assert_eq!(t.queue(), vec![(10, 0, Some(10)), (40, 16, None)]);
    assert!(t.prefilling.iter().all(|p| p.paused.is_none()));
}

/// A model that cannot stop a pass, or a run with the switch off, is as
/// before: the chunk at admission, whole.
#[test]
fn off_or_unsupported_nothing_changes() {
    let whole = PreemptStubModel {
        ep: true,
        ..Default::default()
    };
    let mut t = yielding(&whole, false);
    let _rx = decoding(&mut t);
    let (a, _ra) = request(12);
    t.admit(vec![a]);
    assert_eq!(wire(&whole), vec![CHUNK, 12, 0, 12]);
    assert!(t.prefilling.is_empty());

    let model = pair();
    let mut t = Tick::new(&model, false, false);
    let _rx = decoding(&mut t);
    let (a, _ra) = request(40);
    t.admit(vec![a]);
    t.step();
    assert_eq!(wire(&model), vec![CHUNK, 16, 0, 40, CHUNK, 16, 16, 40]);
    assert_eq!(slices(&model), vec![(16, 16, None)]);
}

/// A slice budget survives the KV-refusal retry: the worker gets the layer
/// count again with the re-sent chunk. The refusal killed the only decoder,
/// so the pass, paused with nobody left to yield to, runs out next tick.
#[test]
fn a_refused_slice_is_sent_again_with_its_layer_count() {
    const RELEASE: u32 = 0xFFFF_FFF1;
    let model = PreemptStubModel {
        refuse_chunks: AtomicUsize::new(1),
        ..pair()
    };
    let mut t = yielding(&model, false);
    let (mut holder, _rx) = active_seq(7, 3);
    holder.seq.block_table = vec![0; 4];
    t.active.push(holder);
    let (a, _ra) = request(12);
    t.admit(vec![a]);
    t.step();
    assert_eq!(
        wire(&model),
        vec![
            SLICE, 2, CHUNK, 12, 0, 12, RELEASE, SLICE, 2, CHUNK, 12, 0, 12
        ]
    );
    assert!(t.active.is_empty() && t.prefilling[0].paused.is_some());
    t.step();
    assert_eq!(
        slices(&model),
        vec![(0, 12, Some(2)), (0, 12, Some(2)), (0, 12, None)]
    );
    assert!(t.prefilling.is_empty());
    assert_eq!(t.active.len(), 1, "the prompt decodes");
}

/// A paused prompt whose caller hung up is freed like any other, on both
/// ranks, and the next prompt starts a pass of its own.
#[test]
fn a_paused_prompt_whose_caller_left_is_freed_and_the_next_one_runs() {
    const RELEASE: u32 = 0xFFFF_FFF1;
    let model = pair();
    let mut t = yielding(&model, false);
    let _rx = decoding(&mut t);
    let (a, ra) = request(12);
    t.admit(vec![a]);
    t.step();
    assert!(t.prefilling[0].paused.is_some());
    drop(ra);
    t.step();
    assert!(t.prefilling.is_empty(), "retired mid-pass");
    assert_eq!(wire(&model).last(), Some(&RELEASE));
    assert!(chunks(&model).is_empty(), "its pass never ended");

    let (b, _rb) = request(10);
    t.admit(vec![b]);
    t.step();
    assert_eq!(slices(&model).last(), Some(&(0, 10, Some(2))), "a probe");
    assert_eq!(t.prefilling[0].paused.map(|p| p.ran), Some(2));
}
