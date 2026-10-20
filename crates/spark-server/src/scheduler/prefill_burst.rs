// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_PREFILL_BURST=1` (default off): drain a short-prompt burst before
//! the next decode step.
//!
//! A tick drains whatever is pending at its start, prefills it, then decodes.
//! When a burst lands while the first prefills run (a C8 wave arrives within
//! ~1 ms and the idle scheduler wakes on its first request), the rest wait a
//! decode step and the streams start out of step. With this policy on, a tick
//! that admitted requests and finds more short prompts already waiting skips
//! its decode and goes straight back to drain, so the burst prefills back to
//! back and every stream decodes the whole wave together.
//!
//! Exactness: each prompt still prefills alone through the same entry with
//! the same chunking (nothing is batched), and decode is per-sequence exact
//! whatever the batch composition, so every sequence's tokens are unchanged.
//! Under EP the worker needs no change: it follows rank 0's commands, and a
//! prefill after a prefill is the sequence a multi-request tick already sends.
//!
//! Bounds: a hold needs every prompt admitted so far to have finished in its
//! first chunk (`prefilling` empty — a multi-chunk prompt keeps the normal
//! interleave), at most `ATLAS_PREFILL_BURST_MAX_TOKENS` (default 16384)
//! prompt tokens pending, a free batch slot, fewer than `max_batch` prompts
//! drained in this hold, and less than `ATLAS_PREFILL_BURST_MAX_MS` (default
//! 1500) since the hold began. Decode runs as soon as any of these fails.
//! Off, [`PrefillBurst::hold`] returns false before reading anything, and
//! the caller does not touch the pending queue.

use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::types::PendingQueue;

pub(super) const DEFAULT_MAX_MS: u64 = 1500;
pub(super) const DEFAULT_MAX_TOKENS: usize = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BurstConfig {
    /// Longest a burst may hold decode back, from its first hold.
    pub max_hold: Duration,
    /// Most prompt tokens pending for which a hold is taken.
    pub max_tokens: usize,
}

impl BurstConfig {
    /// `None` unless `ATLAS_PREFILL_BURST=1`. Read once at scheduler start.
    pub(super) fn from_env() -> Option<Self> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    fn from_vars(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        if get("ATLAS_PREFILL_BURST").as_deref() != Some("1") {
            return None;
        }
        let max_ms = get("ATLAS_PREFILL_BURST_MAX_MS")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_MS);
        let max_tokens = get("ATLAS_PREFILL_BURST_MAX_TOKENS")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS);
        Some(Self {
            max_hold: Duration::from_millis(max_ms),
            max_tokens,
        })
    }
}

/// The tick's state after `start_new_requests`.
#[derive(Clone, Copy, Debug)]
pub(super) struct TickView {
    /// Sequences this tick's `start_new_requests` started (the growth of
    /// `in_flight`): a request it dropped or failed drains nothing.
    pub admitted: usize,
    /// Requests still in the pending queue.
    pub pending: usize,
    /// Their prompt tokens.
    pub pending_tokens: usize,
    /// Prefills in progress (a prompt that did not finish in one chunk).
    pub prefilling: usize,
    /// Active plus prefilling sequences.
    pub in_flight: usize,
    pub max_batch: usize,
}

#[derive(Debug, Default)]
pub(super) struct PrefillBurst {
    cfg: Option<BurstConfig>,
    held_since: Option<Instant>,
    drained: usize,
}

impl PrefillBurst {
    pub(super) fn new(cfg: Option<BurstConfig>) -> Self {
        if let Some(c) = cfg {
            tracing::info!(
                "ATLAS_PREFILL_BURST=1: short-prompt bursts prefill back to back before \
                 the next decode (hold <= {} ms, <= {} pending prompt tokens)",
                c.max_hold.as_millis(),
                c.max_tokens,
            );
        }
        Self {
            cfg,
            held_since: None,
            drained: 0,
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.cfg.is_some()
    }

    /// Whether this tick skips its decode and drains again. Called once per
    /// tick after `start_new_requests`; a `false` ends any hold in progress.
    pub(super) fn hold(&mut self, now: Instant, v: &TickView) -> bool {
        let Some(cfg) = self.cfg else {
            return false;
        };
        let drained = self.drained + v.admitted;
        let held = self
            .held_since
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t));
        let take = v.admitted > 0
            && v.pending > 0
            && v.prefilling == 0
            && v.in_flight < v.max_batch
            && drained < v.max_batch
            && v.pending_tokens <= cfg.max_tokens
            && held < cfg.max_hold;
        if take {
            self.held_since.get_or_insert(now);
            self.drained = drained;
            return true;
        }
        if self.held_since.take().is_some() {
            tracing::info!(
                "prefill burst: {drained} prompt(s) prefilled back to back, decode held {:.1} ms",
                held.as_secs_f64() * 1e3,
            );
        }
        self.drained = 0;
        false
    }
}

/// Pending requests and their prompt tokens (one short lock).
pub(super) fn pending_load(pending: &(Mutex<PendingQueue>, Condvar)) -> (usize, usize) {
    let g = pending.0.lock();
    let tokens = g.requests.iter().map(|r| r.prompt_len()).sum();
    (g.requests.len(), tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> Option<BurstConfig> {
        Some(BurstConfig {
            max_hold: Duration::from_millis(DEFAULT_MAX_MS),
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    /// A C8 wave of 54-token prompts after `admitted` of them started.
    fn wave(admitted: usize, started_before: usize) -> TickView {
        let in_flight = started_before + admitted;
        let pending = 8 - in_flight;
        TickView {
            admitted,
            pending,
            pending_tokens: pending * 54,
            prefilling: 0,
            in_flight,
            max_batch: 8,
        }
    }

    #[test]
    fn burst_drains_before_decode() {
        let mut b = PrefillBurst::new(on());
        let t0 = Instant::now();
        // Tick 1 woke on the first arrival and took the 3 already queued.
        assert!(b.hold(t0, &wave(3, 0)));
        // Tick 2 takes the other 5; nothing left, so it decodes.
        let t1 = t0 + Duration::from_millis(430);
        assert!(!b.hold(t1, &wave(5, 3)));
        assert_eq!(b.held_since, None);
        assert_eq!(b.drained, 0);
    }

    #[test]
    fn trickle_keeps_holding_until_drained() {
        let mut b = PrefillBurst::new(on());
        let t0 = Instant::now();
        for k in 0..7 {
            let t = t0 + Duration::from_millis(140 * k as u64);
            assert!(b.hold(t, &wave(1, k)), "tick {k}");
        }
        assert!(!b.hold(t0 + Duration::from_millis(980), &wave(1, 7)));
    }

    #[test]
    fn hold_is_capped_in_time() {
        let mut b = PrefillBurst::new(on());
        let t0 = Instant::now();
        assert!(b.hold(t0, &wave(1, 0)));
        assert!(b.hold(t0 + Duration::from_millis(1499), &wave(1, 1)));
        assert!(!b.hold(t0 + Duration::from_millis(1500), &wave(1, 2)));
        // The cap ended that burst; the next one starts a fresh clock.
        assert!(b.hold(t0 + Duration::from_millis(1600), &wave(1, 3)));
    }

    #[test]
    fn hold_is_capped_in_prompts() {
        let mut b = PrefillBurst::new(on());
        let t0 = Instant::now();
        // Slots free up as streams retire, but one hold drains < max_batch.
        let mut v = wave(4, 0);
        v.pending = 10;
        assert!(b.hold(t0, &v));
        v.admitted = 3;
        v.in_flight = 1;
        assert!(b.hold(t0, &v));
        v.admitted = 1;
        assert!(!b.hold(t0, &v), "8th prompt of the hold ends it");
    }

    #[test]
    fn no_hold_without_a_free_slot() {
        let mut b = PrefillBurst::new(on());
        let mut v = wave(2, 6);
        v.pending = 3;
        assert!(!b.hold(Instant::now(), &v));
    }

    #[test]
    fn long_prompts_keep_interleaving() {
        let mut b = PrefillBurst::new(on());
        let now = Instant::now();
        // Pending prompts over the token threshold.
        let mut v = wave(1, 0);
        v.pending_tokens = DEFAULT_MAX_TOKENS + 1;
        assert!(!b.hold(now, &v));
        // A prompt that did not finish in one chunk is prefilling.
        let mut v = wave(1, 0);
        v.prefilling = 1;
        assert!(!b.hold(now, &v));
        // At the threshold still holds.
        let mut v = wave(1, 0);
        v.pending_tokens = DEFAULT_MAX_TOKENS;
        assert!(b.hold(now, &v));
    }

    #[test]
    fn no_hold_when_nothing_was_admitted_or_waits() {
        let mut b = PrefillBurst::new(on());
        let now = Instant::now();
        // Nothing admitted (policy or KV gate refused): decode, never spin.
        assert!(!b.hold(now, &wave(0, 2)));
        // Nothing waiting.
        assert!(!b.hold(now, &wave(8, 0)));
    }

    #[test]
    fn off_never_holds_and_keeps_no_state() {
        let mut b = PrefillBurst::new(None);
        assert!(!b.enabled());
        let t0 = Instant::now();
        for admitted in 0..=8 {
            for before in 0..=(8 - admitted) {
                assert!(!b.hold(t0, &wave(admitted, before)));
            }
        }
        assert_eq!(b.held_since, None);
        assert_eq!(b.drained, 0);
    }

    #[test]
    fn config_from_vars() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(BurstConfig::from_vars(vars(&[])), None);
        assert_eq!(
            BurstConfig::from_vars(vars(&[("ATLAS_PREFILL_BURST", "0")])),
            None
        );
        assert_eq!(
            BurstConfig::from_vars(vars(&[("ATLAS_PREFILL_BURST", "1")])),
            on()
        );
        assert_eq!(
            BurstConfig::from_vars(vars(&[
                ("ATLAS_PREFILL_BURST", "1"),
                ("ATLAS_PREFILL_BURST_MAX_MS", "500"),
                ("ATLAS_PREFILL_BURST_MAX_TOKENS", "4096"),
            ])),
            Some(BurstConfig {
                max_hold: Duration::from_millis(500),
                max_tokens: 4096,
            })
        );
    }
}
