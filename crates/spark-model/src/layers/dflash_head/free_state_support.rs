// SPDX-License-Identifier: AGPL-3.0-only

//! Test support for `free_state_tests`: a disabled-but-valid graph runtime for
//! the zeroed head. Split out for the 500-LoC cap.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::graph_runtime::{
    GraphCapabilities, GraphFingerprint, GraphMode, GraphPolicies, GraphRuntime,
    GraphRuntimeConfig, PhasePolicy, ShapeBucket, SpeculativeAlgorithm,
};

/// A disabled-but-valid graph runtime for the zero head. `free_state` only
/// reaches it to invalidate retired keys, and this head never captures any.
pub(super) fn zero_graph_runtime() -> std::sync::Arc<spark_runtime::graph_runtime::GraphRuntime> {
    let policy = PhasePolicy {
        max_entries: 1,
        max_estimated_bytes: 1 << 20,
        capture_enabled: true,
        replay_enabled: true,
        prewarm_enabled: false,
    };
    let backend: std::sync::Arc<dyn GpuBackend> =
        std::sync::Arc::new(spark_runtime::gpu::mock::MockGpuBackend::new());
    std::sync::Arc::new(
        GraphRuntime::new(
            backend,
            GraphCapabilities {
                basic_graphs: true,
                debug_dot: false,
                graph_upload: false,
                conditional_nodes: false,
                while_nodes: false,
                native_serialization: false,
            },
            GraphRuntimeConfig {
                mode: GraphMode::Full,
                speculative_algorithm: SpeculativeAlgorithm::None,
                shape_buckets: vec![ShapeBucket {
                    token_limit: 128,
                    request_limit: 8,
                }],
                policies: GraphPolicies::new(policy, policy, policy, policy, policy).unwrap(),
                max_cache_entries: 8,
                max_cache_bytes: 8 << 20,
                fingerprint: GraphFingerprint {
                    runtime: "test".into(),
                    model: "test".into(),
                    kernel_build: "test".into(),
                    device: "test".into(),
                    cuda: "0".into(),
                    driver: "0".into(),
                    memory_layout: "test".into(),
                },
                resource_generation: 1,
                compatibility_rules: Vec::new(),
                export_dir: None,
                prewarm_profile: None,
            },
            std::sync::Arc::new(spark_runtime::graph_runtime::GraphMetrics::default()),
        )
        .expect("graph runtime"),
    )
}
