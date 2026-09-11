// SPDX-License-Identifier: AGPL-3.0-only

use super::{GraphFallbackReason, GraphPhase, SpeculativeAlgorithm};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

const FALLBACK_COUNT: usize = GraphFallbackReason::ALL.len();
const PHASE_COUNT: usize = GraphPhase::COUNT;

/// Global CUDA graph counters plus per-phase breakdowns.
///
/// The global totals answer "is graph capture engaged at all"; the per-phase
/// maps answer "which phase is falling back and why", which is what an
/// operator needs when prefill stays eager while decode captures. The active
/// speculative algorithm is carried as a single label because a serve process
/// runs exactly one.
#[derive(Debug)]
pub struct GraphMetrics {
    captures: AtomicU64,
    recaptures: AtomicU64,
    replays: AtomicU64,
    capture_failures: AtomicU64,
    replay_failures: AtomicU64,
    eager_fallbacks: AtomicU64,
    evictions: AtomicU64,
    launches: AtomicU64,
    fallback_reasons: [AtomicU64; FALLBACK_COUNT],
    phase_captures: [AtomicU64; PHASE_COUNT],
    phase_replays: [AtomicU64; PHASE_COUNT],
    phase_launches: [AtomicU64; PHASE_COUNT],
    phase_eager_fallbacks: [AtomicU64; PHASE_COUNT],
    phase_graph_bytes: [AtomicU64; PHASE_COUNT],
    phase_evictions: [AtomicU64; PHASE_COUNT],
    algorithm: AtomicU8,
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self {
            captures: AtomicU64::new(0),
            recaptures: AtomicU64::new(0),
            replays: AtomicU64::new(0),
            capture_failures: AtomicU64::new(0),
            replay_failures: AtomicU64::new(0),
            eager_fallbacks: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            launches: AtomicU64::new(0),
            fallback_reasons: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_captures: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_replays: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_launches: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_eager_fallbacks: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_graph_bytes: std::array::from_fn(|_| AtomicU64::new(0)),
            phase_evictions: std::array::from_fn(|_| AtomicU64::new(0)),
            algorithm: AtomicU8::new(SpeculativeAlgorithm::None as u8),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GraphMetricsSnapshot {
    pub captures: u64,
    pub recaptures: u64,
    pub replays: u64,
    pub capture_failures: u64,
    pub replay_failures: u64,
    pub eager_fallbacks: u64,
    pub evictions: u64,
    pub launches: u64,
    pub graph_bytes: u64,
    pub algorithm: String,
    pub fallback_reasons: BTreeMap<String, u64>,
    pub phase_captures: BTreeMap<String, u64>,
    pub phase_replays: BTreeMap<String, u64>,
    pub phase_launches: BTreeMap<String, u64>,
    pub phase_eager_fallbacks: BTreeMap<String, u64>,
    pub phase_graph_bytes: BTreeMap<String, u64>,
    pub phase_evictions: BTreeMap<String, u64>,
}

impl GraphMetrics {
    pub fn snapshot(&self) -> GraphMetricsSnapshot {
        let fallback_reasons = GraphFallbackReason::ALL
            .into_iter()
            .filter_map(|reason| {
                let count = self.fallback_reasons[reason.index()].load(Ordering::Relaxed);
                (count > 0).then(|| (reason.label().to_string(), count))
            })
            .collect();
        GraphMetricsSnapshot {
            captures: self.captures.load(Ordering::Relaxed),
            recaptures: self.recaptures.load(Ordering::Relaxed),
            replays: self.replays.load(Ordering::Relaxed),
            capture_failures: self.capture_failures.load(Ordering::Relaxed),
            replay_failures: self.replay_failures.load(Ordering::Relaxed),
            eager_fallbacks: self.eager_fallbacks.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            launches: self.launches.load(Ordering::Relaxed),
            graph_bytes: self
                .phase_graph_bytes
                .iter()
                .map(|counter| counter.load(Ordering::Relaxed))
                .sum(),
            algorithm: self.algorithm().to_string(),
            fallback_reasons,
            phase_captures: phase_map(&self.phase_captures),
            phase_replays: phase_map(&self.phase_replays),
            phase_launches: phase_map(&self.phase_launches),
            phase_eager_fallbacks: phase_map(&self.phase_eager_fallbacks),
            phase_graph_bytes: phase_map(&self.phase_graph_bytes),
            phase_evictions: phase_map(&self.phase_evictions),
        }
    }

    pub(crate) fn set_algorithm(&self, algorithm: SpeculativeAlgorithm) {
        self.algorithm.store(algorithm as u8, Ordering::Relaxed);
    }

    fn algorithm(&self) -> SpeculativeAlgorithm {
        SpeculativeAlgorithm::from_code(self.algorithm.load(Ordering::Relaxed))
    }

    pub(crate) fn record_capture(&self, phase: GraphPhase, recapture: bool) {
        let counter = if recapture {
            &self.recaptures
        } else {
            &self.captures
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.phase_captures[phase.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_replay(&self, phase: GraphPhase) {
        self.replays.fetch_add(1, Ordering::Relaxed);
        self.phase_replays[phase.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_launch(&self, phase: GraphPhase) {
        self.launches.fetch_add(1, Ordering::Relaxed);
        self.phase_launches[phase.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_capture_failure(&self) {
        self.capture_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_replay_failure(&self) {
        self.replay_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_eviction(&self, phase: GraphPhase) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.phase_evictions[phase.index()].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_fallback(&self, phase: GraphPhase, reason: GraphFallbackReason) {
        self.eager_fallbacks.fetch_add(1, Ordering::Relaxed);
        self.fallback_reasons[reason.index()].fetch_add(1, Ordering::Relaxed);
        self.phase_eager_fallbacks[phase.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The byte gauge is maintained only per phase; the global total is the
    /// sum at snapshot time. Two independent atomics could interleave and
    /// leave the global permanently diverged from the phases under concurrent
    /// add/remove, so there is deliberately no separate global counter.
    pub(crate) fn add_bytes(&self, phase: GraphPhase, bytes: u64) {
        self.phase_graph_bytes[phase.index()].fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn remove_bytes(&self, phase: GraphPhase, bytes: u64) {
        let _ = self.phase_graph_bytes[phase.index()].fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| Some(current.saturating_sub(bytes)),
        );
    }
}

fn phase_map(counters: &[AtomicU64; PHASE_COUNT]) -> BTreeMap<String, u64> {
    GraphPhase::ALL
        .into_iter()
        .filter_map(|phase| {
            let count = counters[phase.index()].load(Ordering::Relaxed);
            (count > 0).then(|| (phase.to_string(), count))
        })
        .collect()
}
