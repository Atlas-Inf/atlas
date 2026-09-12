// SPDX-License-Identifier: AGPL-3.0-only

//! Metrics, cache-churn, and key-identity tests. Split out of `tests.rs` for
//! the 500-LoC cap; a submodule of the same `tests` module, so `use super::*`
//! reaches the shared helpers.

use super::*;

#[test]
fn metrics_carry_the_active_speculative_algorithm() {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime_with_algorithm(backend, 2, 100, 10, 1_000, SpeculativeAlgorithm::Dspark);
    assert_eq!(runtime.metrics().algorithm, "dspark");
}

#[test]
fn replay_metrics_are_phase_labeled() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 2, 100);
    let lease = runtime.capture(
        identity(GraphPhase::Decode, 1),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    runtime.launch(&lease, 7)?;
    let snapshot = runtime.metrics();
    assert_eq!(snapshot.phase_captures.get("decode"), Some(&1));
    assert_eq!(snapshot.phase_launches.get("decode"), Some(&1));
    assert_eq!(snapshot.phase_replays.get("decode"), Some(&1));
    assert!(!snapshot.phase_captures.contains_key("prefill"));
    Ok(())
}

#[test]
fn churn_attributes_evictions_to_the_evicted_phase() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    // Two resident graphs process-wide, roomy per-phase quotas, so the third
    // capture must evict the global LRU — the prefill graph.
    let runtime = runtime_with_global(backend, 4, 1_000, 2, 1_000);
    for phase in [GraphPhase::Prefill, GraphPhase::Decode, GraphPhase::Verify] {
        runtime.capture(
            identity(phase, 1),
            7,
            cost(60),
            vec![],
            CaptureFailurePolicy::Retry,
            |_| Ok(()),
        )?;
    }
    let snapshot = runtime.metrics();
    assert_eq!(snapshot.captures, 3);
    assert_eq!(snapshot.evictions, 1);
    assert_eq!(snapshot.phase_captures.get("prefill"), Some(&1));
    assert_eq!(snapshot.phase_captures.get("verify"), Some(&1));
    assert_eq!(snapshot.phase_evictions.get("prefill"), Some(&1));
    assert!(!snapshot.phase_evictions.contains_key("verify"));
    Ok(())
}

#[test]
fn propose_key_words_participate_in_the_graph_key() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 4, 1_000);
    let with_words = |words: Vec<u64>| {
        let mut id = identity(GraphPhase::Propose, 4);
        if let GraphPayload::Propose { key_words, .. } = &mut id.key.payload {
            *key_words = words;
        }
        id
    };
    let captured = with_words(vec![1, 2, 3]);
    let _lease = runtime.capture(
        captured.clone(),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    // Same identity words hit…
    assert!(runtime.lookup(&with_words(vec![1, 2, 3]))?.is_some());
    // …a different lane/owner (different words) must miss, or a graph would
    // replay against state it was not captured for.
    assert!(runtime.lookup(&with_words(vec![1, 2, 4]))?.is_none());
    Ok(())
}

#[test]
fn stale_key_fallback_is_labeled_by_phase() {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 2, 100);
    let mut key = identity(GraphPhase::Verify, 3).key;
    key.mode = GraphMode::Breakable;
    let error = runtime.lookup_key(&key).err().unwrap();
    assert_eq!(error.reason, GraphFallbackReason::StaleKey);
    let snapshot = runtime.metrics();
    assert_eq!(snapshot.phase_eager_fallbacks.get("verify"), Some(&1));
    assert_eq!(snapshot.fallback_reasons.get("stale_key"), Some(&1));
}
