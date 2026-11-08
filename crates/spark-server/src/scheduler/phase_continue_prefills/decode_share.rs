// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DECODE_STEPS_PER_CHUNK=k` (default 0 = off): while sequences are
//! decoding, run at most one continued prefill chunk per `k` decode steps.
//!
//! The resting path runs one chunk per scheduler tick, so each active
//! decode advances one step per chunk. On qwen4_exp a chunk costs 1.9 s
//! (2K tokens) to 5.5 s (8K), and a 16-token reply that arrived behind a
//! 20K prompt waited out the whole prompt (~18 s, sparky 2026-10-23).
//! Holding chunks back for `k` decode steps lets short replies finish
//! while the long prompt's TTFT grows by about `k` decode steps per chunk.

//!
//! `ATLAS_DECODE_TIME_SHARE=f` (0 < f < 1, default off): the same hold, by
//! wall time. After a continued chunk that took `d`, decodes step alone for
//! `d * f / (1 - f)` before the next chunk, so live sequences keep at least
//! the share `f` of the time while a long prompt prefills. A chunk deep in a
//! 254K prompt takes about 40 s on the 27B; with one chunk a tick, the
//! eight requests decoding beside it got a token a chunk (0.2 tok/s) and ran
//! into the request timeout.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static STEPS_SINCE_CHUNK: AtomicUsize = AtomicUsize::new(0);

/// The last continued chunk: when it started, and once a decode step
/// followed it, when it ended.
static LAST_CHUNK: Mutex<Option<(Instant, Option<Instant>)>> = Mutex::new(None);

fn time_share() -> Option<f64> {
    static F: OnceLock<Option<f64>> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("ATLAS_DECODE_TIME_SHARE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
    })
}

/// How long decodes run alone after a chunk of `chunk` at share `f`.
fn decode_window(chunk: Duration, f: f64) -> Duration {
    chunk.mul_f64(f / (1.0 - f))
}

fn steps_per_chunk() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("ATLAS_DECODE_STEPS_PER_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Whether the knob is on. When it is, a new prompt's first chunk also
/// ends the tick (as under EP), so requests that arrived during that
/// chunk are admitted before the second one runs.
pub(crate) fn enabled() -> bool {
    steps_per_chunk() > 0 || time_share().is_some()
}

/// A standalone decode step (not fused into a chunk) just ran.
pub(crate) fn note_decode_step() {
    if let Ok(mut c) = LAST_CHUNK.lock()
        && let Some((_, end @ None)) = c.as_mut()
    {
        *end = Some(Instant::now());
    }
    let _ = STEPS_SINCE_CHUNK.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_add(1))
    });
}

/// A tick ended with nothing decoding. The window restarts, so steps from
/// earlier requests can't release the first chunk after new ones join
/// (the long prompt's inline first chunk never passes through
/// `hold_chunk`).
pub(crate) fn note_idle() {
    STEPS_SINCE_CHUNK.store(0, Ordering::Relaxed);
    if let Ok(mut c) = LAST_CHUNK.lock() {
        *c = None;
    }
}

/// Whether to skip this tick's continued chunk so decodes can step.
/// Records the chunk when it is allowed to run.
pub(super) fn hold_chunk(decoding: bool) -> bool {
    if let Some(f) = time_share() {
        let now = Instant::now();
        let Ok(mut c) = LAST_CHUNK.lock() else {
            return false;
        };
        // A chunk fused with the decode rows (the mixed forward) is followed
        // by no standalone decode step: it ended by this tick.
        if let Some((_, end @ None)) = c.as_mut() {
            *end = Some(now);
        }
        if decoding
            && let Some((start, Some(end))) = *c
            && now.duration_since(end) < decode_window(end.duration_since(start), f)
        {
            return true;
        }
        *c = Some((now, None));
        return false;
    }
    let k = steps_per_chunk();
    if k == 0 {
        return false;
    }
    // Count from the last chunk of any kind: a chunk that ran with no
    // decoders (the long prompt before the short ones joined) starts the
    // window too, or the first chunk after they join would never wait.
    if decoding && STEPS_SINCE_CHUNK.load(Ordering::Relaxed) < k {
        return true;
    }
    STEPS_SINCE_CHUNK.store(0, Ordering::Relaxed);
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_window_keeps_the_share() {
        let w = decode_window(Duration::from_secs(40), 0.3);
        assert!((w.as_secs_f64() - 40.0 * 0.3 / 0.7).abs() < 1e-6);
        assert_eq!(decode_window(Duration::from_secs(40), 0.5), Duration::from_secs(40));
    }

    #[test]
    fn off_by_default_never_holds() {
        assert!(!hold_chunk(true));
        assert!(!hold_chunk(false));
    }
}
