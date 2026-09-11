// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::graph_runtime::{
    GraphPayload, GraphPhase, GraphSegment, ShapeBucket, SpeculativeAlgorithm,
};

fn fingerprint() -> GraphFingerprint {
    GraphFingerprint {
        runtime: "runtime-1".into(),
        model: "model-checksum".into(),
        kernel_build: "kernel-build".into(),
        device: "sm_121".into(),
        cuda: "13000".into(),
        driver: "driver-1".into(),
        memory_layout: "layout-1".into(),
    }
}

fn identity() -> GraphIdentity {
    GraphIdentity {
        bucket: ShapeBucket {
            token_limit: 32,
            request_limit: 1,
        },
        key: GraphKey {
            schema_version: GRAPH_KEY_SCHEMA_VERSION,
            phase: GraphPhase::Prefill,
            mode: GraphMode::Piecewise,
            segment: GraphSegment::PrefillCompute,
            speculative_algorithm: SpeculativeAlgorithm::None,
            fingerprint: fingerprint(),
            resource_generation: 1,
            payload: GraphPayload::Prefill {
                request_count: 1,
                token_count: 32,
                sequence_lengths: vec![32],
                slots: vec![0],
                block_counts: vec![2],
                chunk_start: 0,
                last_chunk: true,
                paged: false,
                mrope: false,
            },
        },
    }
}

fn capabilities() -> GraphCapabilities {
    GraphCapabilities {
        basic_graphs: true,
        debug_dot: true,
        graph_upload: false,
        conditional_nodes: false,
        while_nodes: false,
        native_serialization: false,
    }
}

fn manifest() -> GraphManifest {
    let identity = identity();
    GraphManifest {
        schema_version: GRAPH_MANIFEST_SCHEMA_VERSION,
        key_schema_version: GRAPH_KEY_SCHEMA_VERSION,
        fingerprint: fingerprint(),
        mode: GraphMode::Piecewise,
        capabilities: capabilities(),
        native_serialization: NativeGraphSerialization::deliberate_no_go(),
        entries: vec![GraphManifestEntry {
            key_hash: graph_key_hash(&identity.key),
            identity,
            cost: GraphCost {
                estimated_bytes: 1024,
                node_count: 4,
                child_count: 0,
                staging_bytes: 128,
            },
            topology_dot: Some("prefill.dot".into()),
            prewarm_eligible: true,
            fallback_reason: None,
        }],
    }
}

#[test]
fn manifest_round_trip_has_no_process_local_artifacts() {
    let manifest = manifest();
    let bytes = manifest.to_json().unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(!text.contains("cudaGraphExec_t"));
    assert!(!text.contains("0x"));
    assert_eq!(GraphManifest::from_json(&bytes).unwrap(), manifest);
    manifest
        .validate_for(&fingerprint(), GraphMode::Piecewise, capabilities())
        .unwrap();
}

#[test]
fn every_fingerprint_dimension_invalidates_a_manifest() {
    let manifest = manifest();
    let mut variants = Vec::new();
    let mut value = fingerprint();
    value.runtime.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.model.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.kernel_build.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.device.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.cuda.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.driver.push('x');
    variants.push(value);
    let mut value = fingerprint();
    value.memory_layout.push('x');
    variants.push(value);
    for incompatible in variants {
        assert!(
            manifest
                .validate_for(&incompatible, GraphMode::Piecewise, capabilities())
                .is_err()
        );
    }
}

#[test]
fn stale_key_hash_and_conditional_capability_are_rejected() {
    let mut stale = manifest();
    stale.entries[0].key_hash = "stale".into();
    assert!(
        stale
            .validate_for(&fingerprint(), GraphMode::Piecewise, capabilities())
            .is_err()
    );
    let mut conditional = manifest();
    conditional.capabilities.conditional_nodes = true;
    assert!(
        conditional
            .validate_for(&fingerprint(), GraphMode::Piecewise, capabilities())
            .is_err()
    );
}

#[test]
fn prewarm_profile_round_trip_rejects_stale_runtime() {
    let profile = GraphPrewarmProfile {
        schema_version: GRAPH_PREWARM_SCHEMA_VERSION,
        key_schema_version: GRAPH_KEY_SCHEMA_VERSION,
        fingerprint: fingerprint(),
        mode: GraphMode::Piecewise,
        identities: vec![identity()],
    };
    let bytes = profile.to_json().unwrap();
    let decoded = GraphPrewarmProfile::from_json(&bytes).unwrap();
    decoded
        .validate_for(&fingerprint(), GraphMode::Piecewise)
        .unwrap();
    assert!(
        decoded
            .validate_for(&fingerprint(), GraphMode::Full)
            .is_err()
    );
}

#[test]
fn filesystem_router_writes_complete_artifacts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    let io = FsGraphArtifactIo;
    io.write_atomic(&path, b"complete").unwrap();
    assert_eq!(io.read(&path).unwrap(), b"complete");
}
