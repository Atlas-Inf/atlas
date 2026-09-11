// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::{DevicePtr, KernelHandle};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "test_backend.rs"]
mod test_backend;
use test_backend::TestBackend;

fn policy(entries: usize, bytes: u64) -> PhasePolicy {
    PhasePolicy {
        max_entries: entries,
        max_estimated_bytes: bytes,
        capture_enabled: true,
        replay_enabled: true,
        prewarm_enabled: true,
    }
}

fn policies(entries: usize, bytes: u64) -> GraphPolicies {
    GraphPolicies::new(
        policy(entries, bytes),
        policy(entries, bytes),
        policy(entries, bytes),
        policy(entries, bytes),
        policy(entries, bytes),
    )
    .unwrap()
}

fn runtime(backend: Arc<TestBackend>, entries: usize, bytes: u64) -> GraphRuntime {
    runtime_with_global(
        backend,
        entries,
        bytes,
        entries * GraphPhase::COUNT,
        bytes * GraphPhase::COUNT as u64,
    )
}

fn runtime_with_global(
    backend: Arc<TestBackend>,
    entries: usize,
    bytes: u64,
    global_entries: usize,
    global_bytes: u64,
) -> GraphRuntime {
    runtime_with_algorithm(
        backend,
        entries,
        bytes,
        global_entries,
        global_bytes,
        SpeculativeAlgorithm::None,
    )
}

fn runtime_with_algorithm(
    backend: Arc<TestBackend>,
    entries: usize,
    bytes: u64,
    global_entries: usize,
    global_bytes: u64,
    algorithm: SpeculativeAlgorithm,
) -> GraphRuntime {
    GraphRuntime::new(
        backend,
        GraphCapabilities {
            basic_graphs: true,
            debug_dot: true,
            graph_upload: false,
            conditional_nodes: false,
            while_nodes: false,
            native_serialization: false,
        },
        GraphRuntimeConfig {
            mode: GraphMode::Full,
            speculative_algorithm: algorithm,
            shape_buckets: vec![ShapeBucket {
                token_limit: 128,
                request_limit: 8,
            }],
            policies: policies(entries, bytes),
            max_cache_entries: global_entries,
            max_cache_bytes: global_bytes,
            fingerprint: fingerprint(),
            resource_generation: 1,
            compatibility_rules: Vec::new(),
            export_dir: None,
            prewarm_profile: None,
        },
        Arc::new(GraphMetrics::default()),
    )
    .unwrap()
}

fn fingerprint() -> GraphFingerprint {
    GraphFingerprint {
        runtime: "test-runtime".into(),
        model: "test-model".into(),
        kernel_build: "test-kernels".into(),
        device: "test-device".into(),
        cuda: "13.0".into(),
        driver: "test-driver".into(),
        memory_layout: "layout-1".into(),
    }
}

fn identity(phase: GraphPhase, value: u32) -> GraphIdentity {
    let payload = match phase {
        GraphPhase::Prefill => GraphPayload::Prefill {
            request_count: 1,
            token_count: value,
            sequence_lengths: vec![value],
            slots: vec![0],
            block_counts: vec![1],
            chunk_start: 0,
            last_chunk: true,
            paged: false,
            mrope: false,
        },
        GraphPhase::Decode => GraphPayload::Decode {
            request_count: 1,
            padded_request_count: 1,
            slots: vec![value],
        },
        GraphPhase::Verify => GraphPayload::Verify {
            request_count: 1,
            row_count: value,
            slots: vec![0],
            depths: vec![value],
            layout: Vec::new(),
        },
        GraphPhase::Propose => GraphPayload::Propose {
            request_count: 1,
            draft_depth: value,
            segment_index: 0,
        },
        GraphPhase::Fused => GraphPayload::Fused {
            request_count: 1,
            row_count: value,
            draft_depth: value.saturating_sub(1),
            slots: vec![0],
        },
    };
    GraphIdentity {
        bucket: ShapeBucket {
            token_limit: 128,
            request_limit: 8,
        },
        key: GraphKey {
            schema_version: GRAPH_KEY_SCHEMA_VERSION,
            phase,
            mode: GraphMode::Full,
            segment: GraphSegment::Whole,
            speculative_algorithm: SpeculativeAlgorithm::None,
            fingerprint: fingerprint(),
            resource_generation: 1,
            payload,
        },
    }
}

fn cost(bytes: u64) -> GraphCost {
    GraphCost {
        estimated_bytes: bytes,
        node_count: 1,
        child_count: 0,
        staging_bytes: 0,
    }
}

#[test]
fn bucket_classification_is_separate_from_the_concrete_key() {
    let buckets = [
        ShapeBucket {
            token_limit: 16,
            request_limit: 1,
        },
        ShapeBucket {
            token_limit: 64,
            request_limit: 4,
        },
    ];
    assert_eq!(classify_shape_bucket(&buckets, 12, 1), Some(buckets[0]));
    assert_eq!(classify_shape_bucket(&buckets, 17, 1), Some(buckets[1]));
    assert_eq!(classify_shape_bucket(&buckets, 65, 1), None);
}

#[test]
fn lru_and_byte_quotas_are_phase_local() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 2, 100);
    let first = runtime.capture(
        identity(GraphPhase::Prefill, 1),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    let _decode = runtime.capture(
        identity(GraphPhase::Decode, 1),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    let _second = runtime.capture(
        identity(GraphPhase::Prefill, 2),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    assert_eq!(runtime.resident_counts(), [1, 1, 0, 0, 0]);
    assert_eq!(runtime.metrics().evictions, 1);
    drop(first);
    Ok(())
}

#[test]
fn global_lru_caps_entries_and_bytes_across_phases() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime_with_global(backend, 10, 1_000, 2, 120);
    let prefill = identity(GraphPhase::Prefill, 1);
    let decode = identity(GraphPhase::Decode, 1);
    let verify = identity(GraphPhase::Verify, 1);
    runtime.capture(
        prefill.clone(),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    runtime.capture(
        decode.clone(),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    assert!(runtime.lookup(&prefill)?.is_some());
    runtime.capture(
        verify.clone(),
        7,
        cost(60),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    assert!(runtime.lookup(&prefill)?.is_some());
    assert!(runtime.lookup(&decode)?.is_none());
    assert!(runtime.lookup(&verify)?.is_some());
    assert_eq!(runtime.resident_counts(), [1, 0, 1, 0, 0]);
    Ok(())
}

#[test]
fn deterministic_capture_failure_is_negative_cached() {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend.clone(), 2, 100);
    let identity = identity(GraphPhase::Prefill, 1);
    backend.set_fail_begin(true);
    let first = runtime.capture(
        identity.clone(),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::NegativeCache,
        |_| Ok(()),
    );
    assert_eq!(
        first.err().unwrap().reason,
        GraphFallbackReason::CaptureFailed
    );
    backend.set_fail_begin(false);
    let second = runtime.capture(
        identity,
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    );
    assert_eq!(
        second.err().unwrap().reason,
        GraphFallbackReason::NegativeCached
    );
    assert_eq!(backend.graph.lock().begin_calls, 1);
}

#[test]
fn capture_error_aborts_and_replay_error_never_retries_eager() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend.clone(), 2, 100);
    let attempts = AtomicUsize::new(0);
    let failed = runtime.capture(
        identity(GraphPhase::Prefill, 1),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| {
            attempts.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("injected body failure")
        },
    );
    assert_eq!(
        failed.err().unwrap().reason,
        GraphFallbackReason::CaptureBodyFailed
    );
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    assert_eq!(backend.graph.lock().abort_calls, 1);

    let lease = runtime.capture(
        identity(GraphPhase::Decode, 1),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    backend.set_fail_launch(true);
    assert_eq!(
        runtime.launch(&lease, 7).unwrap_err().reason,
        GraphFallbackReason::ReplayLaunchFailed
    );
    assert_eq!(runtime.metrics().replay_failures, 1);
    Ok(())
}

#[test]
fn conditional_capture_falls_back_before_running_either_branch() {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 2, 100);
    let branches = AtomicUsize::new(0);
    let error = runtime
        .capture_if_else(
            identity(GraphPhase::Verify, 2),
            7,
            cost(1),
            DevicePtr(1),
            KernelHandle(1),
            CaptureFailurePolicy::Retry,
            |_| {
                branches.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
            |_| {
                branches.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .err()
        .unwrap();
    assert_eq!(error.reason, GraphFallbackReason::DriverUnsupported);
    assert_eq!(branches.load(Ordering::Relaxed), 0);
}

#[test]
fn bounded_loop_validates_hard_cap_before_capture() {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend, 2, 100);
    let state = BoundedLoopState {
        continuation: DevicePtr(1),
        iteration: DevicePtr(2),
        cancellation: DevicePtr(3),
        exit_reason: DevicePtr(4),
        output: DevicePtr(5),
        output_capacity: 3,
        max_iterations: 4,
    };
    let mut identity = identity(GraphPhase::Decode, 1);
    identity.key.segment = GraphSegment::LoopBody;
    identity.key.payload = GraphPayload::DecodeLoop {
        request_count: 1,
        max_iterations: 4,
        output_capacity: 3,
        slots: vec![0],
    };
    let error = runtime
        .capture_bounded_while(
            identity,
            7,
            cost(1),
            state,
            KernelHandle(1),
            KernelHandle(2),
            CaptureFailurePolicy::Retry,
            |_, _| Ok(()),
        )
        .err()
        .unwrap();
    assert_eq!(error.reason, GraphFallbackReason::ShapeUnsupported);
}

#[test]
fn eviction_waits_for_leases_and_completion_event() -> anyhow::Result<()> {
    let backend = Arc::new(TestBackend::new());
    let runtime = runtime(backend.clone(), 1, 100);
    let first = runtime.capture(
        identity(GraphPhase::Prefill, 1),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    let _second = runtime.capture(
        identity(GraphPhase::Prefill, 2),
        7,
        cost(1),
        vec![],
        CaptureFailurePolicy::Retry,
        |_| Ok(()),
    )?;
    runtime.poll_retirements();
    assert!(backend.destroyed_graphs().is_empty());
    drop(first);
    runtime.poll_retirements();
    assert!(backend.destroyed_graphs().is_empty());
    backend.complete_events();
    runtime.poll_retirements();
    assert_eq!(backend.destroyed_graphs(), vec![1]);
    Ok(())
}

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
