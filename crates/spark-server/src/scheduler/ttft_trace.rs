// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request time-to-first-token trace (`ATLAS_TTFT_TRACE=1`, default off).
//!
//! One line per request, logged on the scheduler tick in which the client's
//! first token left (the first non-empty `output_tokens`: a spontaneous
//! `<think>` from the prefill is not streamed, so for GLM that is usually the
//! first decode step):
//!
//! `TTFT_TRACE slot=<s> prompt=<n> cohort=<k> queue=<ms> setup=<ms>
//! pf_submit=<ms> pf_sync=<ms> to_step=<ms> steps=<n> step=<ms> ttft=<ms>
//! first=<think|token|none>`
//!
//! * `queue`: scheduler queue arrival (the receiver thread's push, right after
//!   the HTTP handler's tokenize + template) to `alloc_sequence` in
//!   `start_chunked_prefill`. Includes every earlier prompt's prefill in the
//!   same tick and any decode steps run before this prompt was drained.
//! * `cohort`: prompts `start_new_requests` admitted in the same tick.
//! * `setup`: admission to the `prefill_chunk` call (grammar, EP broadcast).
//! * `pf_submit`: the `prefill_chunk` call (host submission; it returns before
//!   the GPU finishes unless the model syncs inside).
//! * `pf_sync`: `prefill_chunk` return to the first token sampled (the sampler
//!   reads the token back, so this ends when the prefill's GPU work does).
//! * `to_step`: first token sampled to the start of the decode dispatch that
//!   produced the first visible token (other prompts' prefills in between).
//! * `steps`, `step`: decode dispatches this request was in until then, and
//!   the wall time of the last one.
//! * `ttft`: queue arrival to the end of that decode dispatch.
//!
//! A multi-chunk prompt reports `pf_submit`/`pf_sync` for chunk 0 only.
//! The `ATLAS_MTP_STEP_TRACE` line of the same tick has the step's shape.
//! Disarmed, every hook costs one `OnceLock` load.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::ActiveSeq;

/// `ATLAS_TTFT_TRACE=1` arms the trace. Read once per process.
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_TTFT_TRACE").as_deref() == Ok("1"))
}

/// Queue arrivals, keyed by the address of the request's prompt `Arc`
/// (stable and unique while the request lives). Written by the receiver
/// thread, read by the scheduler thread.
static ARRIVALS: Mutex<Option<HashMap<usize, Instant>>> = Mutex::new(None);

/// Arrivals older than this are dropped (requests that never reached
/// admission: cancelled or rejected).
const ARRIVAL_TTL: Duration = Duration::from_secs(600);

thread_local! {
    /// Admitted requests by slot (scheduler thread only).
    static SEQS: RefCell<HashMap<usize, Rec>> = RefCell::new(HashMap::new());
    /// Prompts admitted so far in the current tick.
    static COHORT: RefCell<usize> = const { RefCell::new(0) };
}

fn key(prompt: &std::sync::Arc<Vec<u32>>) -> usize {
    std::sync::Arc::as_ptr(prompt) as usize
}

/// The receiver thread queued `prompt`'s request.
pub(super) fn arrive(prompt: &std::sync::Arc<Vec<u32>>) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut g = ARRIVALS.lock().unwrap_or_else(|e| e.into_inner());
    let map = g.get_or_insert_with(HashMap::new);
    map.retain(|_, t| now.saturating_duration_since(*t) < ARRIVAL_TTL);
    map.insert(key(prompt), now);
}

/// A tick's admissions begin (before `start_new_requests`).
pub(super) fn begin_tick() {
    if enabled() {
        COHORT.with(|c| *c.borrow_mut() = 0);
    }
}

/// `start_chunked_prefill` allocated `slot` for `prompt`.
pub(super) fn admit(prompt: &std::sync::Arc<Vec<u32>>, slot: usize) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let arrival = ARRIVALS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|m| m.remove(&key(prompt)));
    let cohort = COHORT.with(|c| {
        *c.borrow_mut() += 1;
        *c.borrow()
    });
    SEQS.with(|s| {
        s.borrow_mut().insert(
            slot,
            Rec::new(arrival.unwrap_or(now), now, prompt.len(), cohort),
        )
    });
}

/// Stamp one lifecycle point of `slot`'s request.
pub(super) fn mark(slot: usize, what: Mark) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    SEQS.with(|s| {
        if let Some(r) = s.borrow_mut().get_mut(&slot) {
            r.mark(what, now);
        }
    });
}

/// Start of a decode dispatch; `None` when disarmed.
pub(super) fn step_begin() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// End of the decode dispatch begun at `t0`: log every request in `active`
/// whose first token left during it.
pub(super) fn step_end(t0: Option<Instant>, active: &[ActiveSeq]) {
    let Some(t0) = t0 else { return };
    let now = Instant::now();
    SEQS.with(|s| {
        let mut s = s.borrow_mut();
        for a in active {
            let slot = a.seq.slot_idx;
            let Some(r) = s.get_mut(&slot) else { continue };
            r.steps += 1;
            if a.output_tokens.is_empty() {
                r.first_step_start.get_or_insert(t0);
                continue;
            }
            r.last_step = (t0, now);
            let line = r.line(slot);
            s.remove(&slot);
            tracing::info!("{line}");
        }
    });
}

/// A request that produced its first visible token outside a decode step
/// (the prefill's own first token was streamed): log it now.
pub(super) fn visible_at_prefill(slot: usize) {
    if !enabled() {
        return;
    }
    SEQS.with(|s| {
        if let Some(r) = s.borrow_mut().remove(&slot) {
            tracing::info!("{}", r.line(slot));
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mark {
    PrefillStart,
    PrefillSubmitted,
    /// First token sampled; `true` when it is streamed to the client.
    FirstSampled(bool),
}

#[derive(Clone, Debug)]
struct Rec {
    arrival: Instant,
    admit: Instant,
    prompt: usize,
    cohort: usize,
    pf_start: Option<Instant>,
    pf_submitted: Option<Instant>,
    first_sampled: Option<(Instant, bool)>,
    first_step_start: Option<Instant>,
    steps: u32,
    last_step: (Instant, Instant),
}

fn ms(a: Option<Instant>, b: Option<Instant>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => format!("{:.1}", b.saturating_duration_since(a).as_secs_f64() * 1e3),
        _ => "-".to_string(),
    }
}

impl Rec {
    fn new(arrival: Instant, admit: Instant, prompt: usize, cohort: usize) -> Self {
        Self {
            arrival,
            admit,
            prompt,
            cohort,
            pf_start: None,
            pf_submitted: None,
            first_sampled: None,
            first_step_start: None,
            steps: 0,
            last_step: (admit, admit),
        }
    }

    fn mark(&mut self, what: Mark, now: Instant) {
        match what {
            Mark::PrefillStart => {
                self.pf_start.get_or_insert(now);
            }
            Mark::PrefillSubmitted => {
                self.pf_submitted.get_or_insert(now);
            }
            Mark::FirstSampled(visible) => {
                self.first_sampled.get_or_insert((now, visible));
            }
        }
    }

    fn line(&self, slot: usize) -> String {
        let sampled = self.first_sampled.map(|(t, _)| t);
        let (step0, step1) = self.last_step;
        let (to_step, step, end) = if self.steps == 0 {
            ("-".to_string(), "-".to_string(), sampled.unwrap_or(step1))
        } else {
            (
                ms(sampled, Some(step0)),
                ms(Some(step0), Some(step1)),
                step1,
            )
        };
        let first = match self.first_sampled {
            Some((_, true)) => "token",
            Some((_, false)) => "think",
            None => "none",
        };
        format!(
            "TTFT_TRACE slot={slot} prompt={} cohort={} queue={} setup={} pf_submit={} \
             pf_sync={} to_step={to_step} steps={} step={step} ttft={} first={first}",
            self.prompt,
            self.cohort,
            ms(Some(self.arrival), Some(self.admit)),
            ms(Some(self.admit), self.pf_start),
            ms(self.pf_start, self.pf_submitted),
            ms(self.pf_submitted, sampled),
            self.steps,
            ms(Some(self.arrival), Some(end)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(t0: Instant, ms: u64) -> Instant {
        t0 + Duration::from_millis(ms)
    }

    #[test]
    fn line_attributes_each_phase() {
        let t0 = Instant::now();
        let mut r = Rec::new(t0, at(t0, 100), 87, 3);
        r.mark(Mark::PrefillStart, at(t0, 102));
        r.mark(Mark::PrefillSubmitted, at(t0, 170));
        r.mark(Mark::FirstSampled(false), at(t0, 330));
        r.mark(Mark::PrefillStart, at(t0, 999)); // first stamp wins
        r.steps = 1;
        r.last_step = (at(t0, 900), at(t0, 1000));
        assert_eq!(
            r.line(5),
            "TTFT_TRACE slot=5 prompt=87 cohort=3 queue=100.0 setup=2.0 pf_submit=68.0 \
             pf_sync=160.0 to_step=570.0 steps=1 step=100.0 ttft=1000.0 first=think"
        );
    }

    #[test]
    fn visible_prefill_token_ends_at_the_sample() {
        let t0 = Instant::now();
        let mut r = Rec::new(t0, t0, 10, 1);
        r.mark(Mark::PrefillStart, at(t0, 1));
        r.mark(Mark::PrefillSubmitted, at(t0, 2));
        r.mark(Mark::FirstSampled(true), at(t0, 50));
        let line = r.line(0);
        assert!(
            line.contains("to_step=- steps=0 step=- ttft=50.0 first=token"),
            "{line}"
        );
    }

    #[test]
    fn unstamped_phases_print_dashes() {
        let t0 = Instant::now();
        let r = Rec::new(t0, t0, 10, 1);
        let line = r.line(0);
        assert!(line.contains("setup=- pf_submit=- pf_sync=-"), "{line}");
        assert!(line.ends_with("first=none"), "{line}");
    }
}
