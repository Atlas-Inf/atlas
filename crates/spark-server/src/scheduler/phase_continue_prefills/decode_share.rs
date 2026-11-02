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

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static STEPS_SINCE_CHUNK: AtomicUsize = AtomicUsize::new(0);

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
    steps_per_chunk() > 0
}

/// A standalone decode step (not fused into a chunk) just ran.
pub(crate) fn note_decode_step() {
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
}

/// Whether to skip this tick's continued chunk so decodes can step.
/// Records the chunk when it is allowed to run.
pub(super) fn hold_chunk(decoding: bool) -> bool {
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
    fn off_by_default_never_holds() {
        assert!(!hold_chunk(true));
        assert!(!hold_chunk(false));
    }
}
