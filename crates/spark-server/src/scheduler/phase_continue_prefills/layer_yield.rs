// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_PREFILL_YIELD_MS=T` (default 0 = off): while sequences are
//! decoding, a prefill pass runs as slices of whole layers about `T` ms long
//! and the tick's ordinary decode step runs between them.
//!
//! The resting path runs a whole chunk per tick, so a decoder advances one
//! step per chunk: 2.9 s apart behind an 8K-token chunk on the GLM pair, 13 s
//! of near silence behind a cold 36K prompt. Chunk size cannot shrink without
//! changing the pass shape and so the bits (`prefill_b::pc_policy`,
//! "Accumulation order"). A slice changes neither: the pass runs the same
//! layers over the same rows in the same order, and only the tick it runs
//! them in differs (`Model::prefill_chunk_slice`).
//!
//! This is the head's pacing only. The model keeps the paused pass and
//! restores what a decode step overwrote; the worker follows the layer count
//! the head sends with each slice, so the ranks need no clock in common.
//!
//! A pass starts with a short probe slice, since its cost per layer depends
//! on how many of its rows the prefix cache left to compute. Later slices
//! take `T / (measured time per layer)` layers. The prompt's first token is
//! later by the decode steps that ran in between, and by nothing else.

use std::time::{Duration, Instant};

use anyhow::Result;
use spark_model::traits::{ChunkStep, Model};
use spark_runtime::gpu::DevicePtr;

use crate::scheduler::levers::SchedLevers;

/// Layers of a pass's first slice, before its time per layer is known.
const PROBE_LAYERS: usize = 2;

/// The most layers a slice asks for: more than any model has, and a count
/// that fits the worker's command word.
const MAX_LAYERS: usize = u16::MAX as usize;

/// The target time between decode steps while a pass runs
/// (`SchedLevers::prefill_yield`), from the switch's value.
pub(crate) fn parse(v: Option<&str>) -> Option<Duration> {
    v.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map(Duration::from_millis)
}

/// What the head knows of a pass it paused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Paused {
    /// Layers of the pass that ran in its last slice.
    pub ran: usize,
    /// How long that slice took, submission to drained stream.
    pub took: Duration,
}

/// Layers the next slice of a pass may run; `None` runs the pass to its end.
///
/// `decoding`: a sequence would step between slices. `last`: the pass's
/// previous slice, `None` before its first.
pub(crate) fn budget(
    target: Option<Duration>,
    decoding: bool,
    last: Option<Paused>,
) -> Option<usize> {
    let target = target?;
    if !decoding {
        return None;
    }
    let Some(last) = last.filter(|l| l.ran > 0) else {
        return Some(PROBE_LAYERS);
    };
    let per_layer = last.took.as_nanos() / last.ran as u128;
    if per_layer == 0 {
        return None;
    }
    Some((target.as_nanos() / per_layer).clamp(1, MAX_LAYERS as u128) as usize)
}

/// The prefill the continue phase serves: the queue's head, unless a pass is
/// paused. That pass holds the arena, so its prompt runs whatever the queue
/// order has become since.
pub(crate) fn next(prefilling: &[crate::scheduler::types::PrefillInProgress]) -> usize {
    prefilling
        .iter()
        .position(|p| p.paused.is_some())
        .unwrap_or(0)
}

/// Whether `p` waits for a first chunk that queued because passes yield
/// (`phase_start_prefills::queue_first`): nothing of it ran at admission.
pub(crate) fn queued(
    levers: &SchedLevers,
    model: &dyn Model,
    p: &crate::scheduler::types::PrefillInProgress,
) -> bool {
    on(levers, model) && p.queued_first_chunk.is_some()
}

/// Whether passes are sliced on this run at all.
pub(crate) fn on(levers: &SchedLevers, model: &dyn Model) -> bool {
    levers.prefill_yield.is_some() && model.prefill_slices()
}

/// The layer budget of the next chunk call of a prefill last paused at
/// `last`, with `decoding` sequences to step between slices.
pub(crate) fn slice(
    levers: &SchedLevers,
    model: &dyn Model,
    decoding: bool,
    last: Option<Paused>,
) -> Option<usize> {
    budget(
        levers.prefill_yield.filter(|_| model.prefill_slices()),
        decoding,
        last,
    )
}

/// Record what a chunk call that began at `began` left: `None` when its pass
/// paused (the caller ends the tick's prefill work and calls again next
/// tick), else the finished chunk's result.
pub(crate) fn settle(
    paused: &mut Option<Paused>,
    began: Instant,
    step: Result<ChunkStep>,
) -> Option<Result<DevicePtr>> {
    match step {
        Ok(ChunkStep::Paused { ran }) => {
            let took = began.elapsed();
            tracing::debug!("Prefill pass paused after {ran} more layers ({took:?})");
            *paused = Some(Paused { ran, took });
            None
        }
        Ok(ChunkStep::Done(logits)) => {
            *paused = None;
            Some(Ok(logits))
        }
        Err(e) => {
            *paused = None;
            Some(Err(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: Option<Duration> = Some(Duration::from_millis(300));

    fn paused(ran: usize, ms: u64) -> Option<Paused> {
        Some(Paused {
            ran,
            took: Duration::from_millis(ms),
        })
    }

    #[test]
    fn off_by_default_and_for_zero_or_garbage() {
        assert_eq!(parse(None), None);
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some("fast")), None);
        assert_eq!(parse(Some(" 250 ")), Some(Duration::from_millis(250)));
    }

    #[test]
    fn off_never_slices() {
        assert_eq!(budget(None, true, None), None);
        assert_eq!(budget(None, true, paused(2, 130)), None);
    }

    #[test]
    fn with_nothing_decoding_the_pass_runs_whole() {
        assert_eq!(budget(T, false, None), None);
        // Its decoders finished while it was paused: the rest runs at once.
        assert_eq!(budget(T, false, paused(4, 260)), None);
    }

    #[test]
    fn a_pass_starts_with_a_probe() {
        assert_eq!(budget(T, true, None), Some(PROBE_LAYERS));
    }

    #[test]
    fn later_slices_fill_the_target_at_the_measured_pace() {
        // An 8K-row pass on the pair: about 65 ms a layer.
        assert_eq!(budget(T, true, paused(2, 130)), Some(4));
        // A 1K-row pass: the rest of its 45 layers fit one slice.
        assert_eq!(budget(T, true, paused(2, 16)), Some(37));
    }

    #[test]
    fn a_slice_never_asks_for_more_than_the_command_word_holds() {
        let fast = Some(Paused {
            ran: 2,
            took: Duration::from_nanos(2),
        });
        assert_eq!(budget(T, true, fast), Some(MAX_LAYERS));
    }

    #[test]
    fn a_layer_slower_than_the_target_still_advances() {
        assert_eq!(budget(T, true, paused(1, 900)), Some(1));
    }

    #[test]
    fn a_paused_pass_is_recorded_and_a_finished_or_failed_one_cleared() {
        let mut state = None;
        let began = Instant::now();
        assert!(settle(&mut state, began, Ok(ChunkStep::Paused { ran: 3 })).is_none());
        assert_eq!(state.map(|p| p.ran), Some(3));
        let done = settle(&mut state, began, Ok(ChunkStep::Done(DevicePtr::NULL)));
        assert!(matches!(done, Some(Ok(_))) && state.is_none());
        state = paused(3, 10);
        let failed = settle(&mut state, began, Err(anyhow::anyhow!("boom")));
        assert!(matches!(failed, Some(Err(_))) && state.is_none());
    }

    #[test]
    fn an_unmeasurable_slice_ends_the_slicing() {
        assert_eq!(budget(T, true, paused(3, 0)), None);
    }
}
