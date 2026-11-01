// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_WARM_CHUNK_RUN=1` (default off): a prefill chunk that computed
//! nothing does not end the scheduler's tick.
//!
//! While another sequence decodes, the scheduler runs one prefill chunk per
//! tick and then a decode step, so that a long prompt does not hold the
//! decoders for more than a chunk (`run_standard`; under EP the first chunk
//! ends its tick too, `scheduler::run`). A chunk under the restore depth of
//! a prefix-cached turn computes nothing
//! (`SequenceState::prefill_chunk_cached`), yet it took a tick like any
//! other: a 45K-token warm turn at 8K-token chunks waited five decode or
//! verify steps of the other sequences before its first row ran, a
//! 512K-token one over sixty.
//!
//! With the switch the chunk after such a chunk runs in the same tick, until
//! one computes or the run has taken [`BUDGET`]. The decoders then wait one
//! run of cached chunks (a few milliseconds each with
//! `ATLAS_GLM_WARM_SKIP_CACHED`, an arena zero and an embed each without)
//! plus the one chunk that computes, where they waited that chunk alone.
//!
//! Exact per request: the same chunk commands in the same order for the
//! prompt, and the same steps for every other sequence; only their
//! interleaving changes, so what a decode step finds left in the scratch
//! arena can differ as it does between any two request histories. Only the
//! head schedules (the worker follows its commands), so the switch needs no
//! rank agreement.

use std::time::{Duration, Instant};

use super::PrefillInProgress;

/// How long a run of cached chunks may hold the tick: about one verify step.
const BUDGET: Duration = Duration::from_millis(40);

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_WARM_CHUNK_RUN").as_deref() == Ok("1"))
}

/// Whether the chunk `p` just ran computed nothing and another follows.
pub(super) fn chunk_was_cached(p: &PrefillInProgress) -> bool {
    p.chunk_offset < p.prompt_tokens.len() && p.seq.prefill_chunk_cached(p.chunk_offset, false)
}

/// Whether the tick goes on to the second chunk of the prompt whose first
/// chunk it just ran inline (`start_new_requests`).
pub(in crate::scheduler) fn follows_first_chunk(p: &PrefillInProgress) -> bool {
    enabled() && chunk_was_cached(p)
}

/// Run one chunk; with the switch, more while `chunk` says the one it ran
/// was cached with nothing else done in it, and the run is within budget.
pub(super) fn run(chunk: impl FnMut() -> bool) -> usize {
    run_with(enabled(), BUDGET, chunk)
}

fn run_with(on: bool, budget: Duration, mut chunk: impl FnMut() -> bool) -> usize {
    let began = Instant::now();
    let mut chunks = 1;
    while chunk() && on && began.elapsed() < budget {
        chunks += 1;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prompt of `total` tokens whose chunks each end one `step` further,
    /// cached below `restored`; returns the chunk ends one tick runs.
    fn tick(on: bool, budget: Duration, restored: usize, total: usize, step: usize) -> Vec<usize> {
        let mut seq = spark_model::traits::SequenceState::host_only(0);
        seq.marconi_skip_to = restored;
        let (mut at, mut ends) = (0, Vec::new());
        let n = run_with(on, budget, || {
            at = total.min(at + step);
            ends.push(at);
            at < total && seq.prefill_chunk_cached(at, false)
        });
        assert_eq!(n, ends.len());
        ends
    }

    const LONG: Duration = Duration::from_secs(60);

    /// A 45K-token warm turn restored at 45,520: five cached chunks.
    #[test]
    fn a_tick_runs_the_cached_chunks_and_the_first_one_that_computes() {
        let ends = tick(true, LONG, 45_520, 45_556, 8_192);
        assert_eq!(ends, [8_192, 16_384, 24_576, 32_768, 40_960, 45_556]);
        // A restore inside the second chunk: that chunk computes, and ends
        // the tick.
        assert_eq!(tick(true, LONG, 10_000, 45_556, 8_192), [8_192, 16_384]);
        // A cold prompt: one chunk a tick, as without the switch.
        assert_eq!(tick(true, LONG, 0, 45_556, 8_192), [8_192]);
        // A prompt cached to its last chunk boundary still runs that chunk.
        assert_eq!(
            tick(true, LONG, 16_384, 16_400, 8_192),
            [8_192, 16_384, 16_400]
        );
    }

    #[test]
    fn without_the_switch_a_tick_is_one_chunk() {
        assert_eq!(tick(false, LONG, 45_520, 45_556, 8_192), [8_192]);
        assert_eq!(tick(false, LONG, 0, 45_556, 8_192), [8_192]);
    }

    /// The run ends once it has used its budget, cached chunks left or not.
    #[test]
    fn a_run_past_its_budget_ends_the_tick() {
        assert_eq!(tick(true, Duration::ZERO, 45_520, 45_556, 8_192), [8_192]);
        let mut chunks = 0;
        let n = run_with(true, Duration::from_millis(30), || {
            chunks += 1;
            std::thread::sleep(Duration::from_millis(20));
            true
        });
        assert_eq!((n, chunks), (2, 2));
    }

    #[test]
    fn the_switch_is_off_in_this_process() {
        assert!(!enabled());
        let mut chunks = 0;
        assert_eq!(
            run(|| {
                chunks += 1;
                true
            }),
            1
        );
        assert_eq!(chunks, 1);
    }
}
