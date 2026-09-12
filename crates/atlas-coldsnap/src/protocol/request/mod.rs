// SPDX-License-Identifier: AGPL-3.0-only

//! The operation request: `format 4`, `coldsnap-operation-request`.
//!
//! This is the typed form. `sleep` / `wake` / `status` are *not* built by
//! round-tripping through it — see [`crate::template`] — because a typed
//! round trip can drop a field the caller sent, reorder nested objects, or
//! turn an absent field into a present one. Those three operations must
//! preserve the restore request's shape exactly.

pub mod execution;
pub mod policy;

use crate::error::{ColdsnapError, ViolationKind};
use crate::id::OperationId;
use crate::protocol::constants::{REQUEST_FORMAT, REQUEST_KIND};
use crate::protocol::driver::{SnapshotDriver, SnapshotDriverSelection};
use crate::protocol::engine::ColdsnapEngine;
use crate::protocol::operation::Operation;

pub use execution::{
    AdapterTopology, ExecutionGraph, LaunchUnit, Mount, ProcessGroup, ServiceDomain, Worker,
};
pub use policy::{
    ArtifactScope, CachePolicy, CapsulePolicy, CompatibilityPolicy, GraphPolicy,
    KernelCompatibility, NativeMaterialization, NativePolicy, PreparedCache, ProcessPolicy,
    RecoveryPolicy, SnapshotPolicy, WeightPolicy,
};

/// The model a workload serves.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSource {
    /// Model id, e.g. a Hugging Face repo.
    pub id: String,
    /// An immutable revision. A branch such as `main` is not reproducible.
    pub revision: String,
    /// Source kind, e.g. `huggingface`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// The launch specification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchSpec {
    /// The engine. Closed — see [`ColdsnapEngine`].
    pub engine: ColdsnapEngine,
    /// The model being served.
    pub model: ModelSource,
    /// Container/process boundaries.
    pub units: Vec<LaunchUnit>,
    /// The process topology inside those units.
    pub execution: ExecutionGraph,
}

/// The portable acceptance check.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ValidationPolicy {
    /// Health endpoint path.
    pub health_path: String,
    /// Prompt sent to the restored service.
    pub prompt: String,
    /// Exact expected response.
    pub expected: String,
}

/// The default acceptance check: model-independent and portable.
pub fn default_validation() -> ValidationPolicy {
    ValidationPolicy {
        health_path: "/health".to_owned(),
        prompt: "Reply with exactly: coldsnap-cuda-snapshot-ok".to_owned(),
        expected: "coldsnap-cuda-snapshot-ok".to_owned(),
    }
}

/// Lets an orchestrator make restored containers first-class members of its
/// own lifecycle. Optional for direct usage.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkloadIdentity {
    /// The orchestrator's cluster identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<String>,
    /// The orchestrator's intent identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent_id: Option<String>,
    /// Recipe name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Runtime name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    /// Model name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The name the service is advertised under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model_name: Option<String>,
    /// Where the orchestrator collects logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
}

/// Selects the state a restore activation returns in.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct LifecyclePolicy {
    /// `running` or `warm`; absent means `running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_state: Option<ActivationState>,
}

/// The activation state a restore returns in.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationState {
    /// Fully hydrated and serving.
    Running,
    /// Hydrated only to the hydration boundary.
    Warm,
}

/// A complete operation request.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OperationRequest {
    /// Always [`REQUEST_FORMAT`].
    pub format: u32,
    /// Always [`REQUEST_KIND`].
    pub kind: String,
    /// What to do.
    pub operation: Operation,
    /// A fresh id per operation.
    pub id: OperationId,
    /// The selected snapshot driver.
    pub snapshot_driver: SnapshotDriverSelection,
    /// Input artifact. Required for restore/sleep/wake/status/publish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    /// Output artifact. Required for capture/publish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// The workload to capture or restore.
    pub launch: LaunchSpec,
    /// The snapshot policy.
    pub policy: SnapshotPolicy,
    /// The acceptance check.
    pub validation: ValidationPolicy,
    /// Orchestrator lifecycle identity. Always emitted, matching the
    /// controller's encoder.
    pub workload: WorkloadIdentity,
    /// Activation-state selection. Always emitted, matching the controller.
    pub lifecycle: LifecyclePolicy,
}

impl OperationRequest {
    /// Build a request with the literal envelope already set.
    ///
    /// Everything else is required: there is no constructor that guesses a
    /// driver, a policy, or a validation prompt.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation: Operation,
        id: OperationId,
        driver: SnapshotDriver,
        artifact: Option<String>,
        output: Option<String>,
        launch: LaunchSpec,
        policy: SnapshotPolicy,
        validation: ValidationPolicy,
        workload: WorkloadIdentity,
        lifecycle: LifecyclePolicy,
    ) -> Self {
        Self {
            format: REQUEST_FORMAT,
            kind: REQUEST_KIND.to_owned(),
            operation,
            id,
            snapshot_driver: driver.selection(),
            artifact,
            output,
            launch,
            policy,
            validation,
            workload,
            lifecycle,
        }
    }

    /// The artifact path, treating an empty string as absent.
    pub fn artifact_path(&self) -> Option<&str> {
        self.artifact.as_deref().filter(|path| !path.is_empty())
    }

    /// The output path, treating an empty string as absent.
    pub fn output_path(&self) -> Option<&str> {
        self.output.as_deref().filter(|path| !path.is_empty())
    }

    /// Check the invariants the controller's `Validate` enforces.
    ///
    /// The newtypes already guarantee id, operation, and driver validity;
    /// what remains are the envelope and the field-presence rules.
    pub fn validate(&self) -> Result<(), ColdsnapError> {
        if self.format != REQUEST_FORMAT || self.kind != REQUEST_KIND {
            return Err(ColdsnapError::Protocol(
                crate::error::ProtocolViolation::new(
                    ViolationKind::WrongEnvelope,
                    format!(
                        "request envelope is format {} kind {:?}, expected format {} kind {:?}",
                        self.format, self.kind, REQUEST_FORMAT, REQUEST_KIND
                    ),
                ),
            ));
        }
        if self.operation.requires_output() && self.output_path().is_none() {
            return Err(ColdsnapError::unsupported(
                format!("{} without output", self.operation),
                "this operation requires an output path",
            ));
        }
        if self.operation.requires_artifact() && self.artifact_path().is_none() {
            return Err(ColdsnapError::unsupported(
                format!("{} without artifact", self.operation),
                "this operation requires an artifact path",
            ));
        }
        if let Some(state) = self.lifecycle.activation_state {
            // The enum makes an out-of-set value unrepresentable, so this
            // arm exists only to keep the rule visible next to its source.
            debug_assert!(matches!(
                state,
                ActivationState::Running | ActivationState::Warm
            ));
        }
        Ok(())
    }

    /// The engine this request names.
    pub fn engine(&self) -> ColdsnapEngine {
        self.launch.engine
    }
}

/// Shared test fixtures. `pub(crate)` so the template tests can seed a
/// template from a genuinely typed request rather than hand-written JSON.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::protocol::request::execution::AdapterTopology;

    pub(crate) fn launch() -> LaunchSpec {
        LaunchSpec {
            engine: ColdsnapEngine::Vllm,
            model: ModelSource {
                id: "Qwen/Qwen3.5-0.8B".to_owned(),
                revision: "0".repeat(40),
                source: Some("huggingface".to_owned()),
            },
            units: vec![LaunchUnit {
                id: "unit-0".to_owned(),
                index: 0,
                host: "gpu-a.example".to_owned(),
                devices: vec!["0".to_owned()],
                image: "registry.example/vllm@sha256:0".to_owned(),
                image_digest: "sha256:0".to_owned(),
                command: vec!["vllm".to_owned(), "serve".to_owned()],
                environment: Default::default(),
                mounts: Vec::new(),
            }],
            execution: ExecutionGraph {
                workers: vec![Worker {
                    id: "worker-0".to_owned(),
                    unit: "unit-0".to_owned(),
                    service: "model".to_owned(),
                    process_slot: 0,
                    device_slots: vec![0],
                }],
                groups: vec![ProcessGroup {
                    id: "world".to_owned(),
                    kind: "vllm:world".to_owned(),
                    service: "model".to_owned(),
                    members: vec!["worker-0".to_owned()],
                }],
                services: vec![ServiceDomain {
                    id: "model".to_owned(),
                    role: "vllm:serve".to_owned(),
                    workers: vec!["worker-0".to_owned()],
                }],
                adapter: AdapterTopology {
                    schema: "vllm:direct-v1".to_owned(),
                    digest: format!("sha256:{}", "1".repeat(64)),
                    payload: serde_json::json!({"runtime": "vllm-direct"}),
                },
            },
        }
    }

    pub(crate) fn request(operation: Operation, artifact: Option<&str>) -> OperationRequest {
        OperationRequest::new(
            operation,
            OperationId::parse("op-v1").unwrap(),
            SnapshotDriver::N610,
            artifact.map(str::to_owned),
            None,
            launch(),
            SnapshotPolicy::canonical_for(SnapshotDriver::N610),
            default_validation(),
            WorkloadIdentity::default(),
            LifecyclePolicy::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::request::fixtures::request;

    #[test]
    fn a_sleep_request_without_an_artifact_is_refused() {
        let err = request(Operation::Sleep, None).validate().unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
    }

    #[test]
    fn a_sleep_request_with_an_artifact_validates() {
        request(Operation::Sleep, Some("./artifact.json"))
            .validate()
            .unwrap();
    }

    #[test]
    fn capture_requires_output_not_artifact() {
        let err = request(Operation::Capture, None).validate().unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
        let mut capture = request(Operation::Capture, None);
        capture.output = Some("./out.json".to_owned());
        capture.validate().unwrap();
    }

    #[test]
    fn an_empty_artifact_string_counts_as_absent() {
        let mut req = request(Operation::Sleep, None);
        req.artifact = Some(String::new());
        assert!(req.artifact_path().is_none());
        assert!(req.validate().is_err());
    }

    #[test]
    fn the_envelope_is_emitted_and_round_trips() {
        let req = request(Operation::Sleep, Some("./a.json"));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""format":4"#));
        assert!(json.contains(r#""kind":"coldsnap-operation-request""#));
        assert!(json.contains(r#""operation":"sleep""#));
        let back: OperationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn workload_and_lifecycle_are_always_emitted_like_the_controller() {
        // Go's encoder emits these struct fields even when empty, and the
        // request digest is computed over the bytes as sent.
        let json = serde_json::to_string(&request(Operation::Sleep, Some("./a.json"))).unwrap();
        assert!(json.contains(r#""workload":{}"#), "{json}");
        assert!(json.contains(r#""lifecycle":{}"#), "{json}");
    }

    #[test]
    fn an_out_of_set_envelope_is_a_protocol_violation() {
        let mut req = request(Operation::Sleep, Some("./a.json"));
        req.format = 3;
        let err = req.validate().unwrap_err();
        assert!(matches!(
            err,
            ColdsnapError::Protocol(v) if v.kind == ViolationKind::WrongEnvelope
        ));
    }
}
