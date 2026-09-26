// SPDX-License-Identifier: AGPL-3.0-only

//! The capabilities document: `format 1`,
//! `coldsnap-controller-capabilities`.
//!
//! Unlike a receipt, this is a **discovery** document, so parsing is tolerant
//! of unknown fields: a newer controller adding a key must not make Atlas
//! unable to ask what it supports. The fields this crate depends on are still
//! required and still checked — tolerance of the unknown is not indifference
//! to the known.
//!
//! [`Capabilities::admit`] is the gate to run before any operation: it is
//! cheaper to learn that a controller lacks `wake` from a static document than
//! from a spawn and a manager-side round trip.

use crate::error::ColdsnapError;
use crate::protocol::constants::{
    CAPABILITIES_FORMAT, CAPABILITIES_KIND, RECEIPT_FORMAT, REQUEST_FORMAT,
};
use crate::protocol::driver::SnapshotDriver;
use crate::protocol::engine::ColdsnapEngine;
use crate::protocol::operation::Operation;

/// The controller build identity.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct ControllerInfo {
    /// Release version.
    pub version: String,
    /// Commit, when the build recorded one.
    #[serde(default)]
    pub commit: Option<String>,
}

/// Protocol format versions the controller speaks.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Protocols {
    /// The request format this build emits.
    pub request_format: u32,
    /// Request formats this build accepts.
    pub accepted_request_formats: Vec<u32>,
    /// The artifact format this build emits.
    pub artifact_format: u32,
    /// Artifact formats this build accepts.
    pub accepted_artifact_formats: Vec<u32>,
    /// The receipt format this build emits.
    pub receipt_format: u32,
    /// The timing-event format.
    pub timing_event_format: u32,
    /// The host-provider protocol format.
    pub host_provider_format: u32,
    /// The payload-validation RPC format.
    pub payload_validation_rpc_format: u32,
}

/// One operation and its replay semantics.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct OperationCapability {
    /// The operation name.
    pub operation: String,
    /// Its replay semantics.
    pub replay_semantics: String,
    /// Whether a `--prepare-only` variant exists.
    #[serde(default)]
    pub prepare_only: bool,
}

/// One registered engine adapter.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct EngineAdapter {
    /// Engine name.
    pub engine: String,
    /// Adapter executable name.
    pub executable: String,
    /// Environment variable that overrides the executable path.
    pub environment: String,
    /// Operations the adapter supports.
    pub operations: Vec<String>,
    /// Whether the adapter supports prepare-only restore.
    pub prepare_restore: bool,
    /// The adapter's controller model.
    pub controller_model: String,
    /// Supported native-materialization modes.
    pub native_materialization: Vec<String>,
    /// The default native-materialization mode.
    pub default_native_materialization: String,
}

/// One snapshot driver contract.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SnapshotDriverContract {
    /// Driver id.
    pub id: String,
    /// Driver ABI.
    pub abi: u32,
    /// The lifecycle boundary this driver captures at.
    pub capture_boundary: String,
    /// Minimum NVIDIA host driver major version.
    pub minimum_nvidia_driver_major: u32,
    /// Host feature requirements. Retained verbatim; this crate does not
    /// interpret the feature inventory.
    #[serde(default)]
    pub base_requirements: serde_json::Value,
}

/// One host transport.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct HostTransport {
    /// Transport name.
    pub name: String,
    /// Whether engine adapters issue requests through it.
    pub engine_adapter_requests: bool,
    /// The transport's security model.
    pub security: String,
    /// Whether the transport is required or optional.
    pub status: String,
}

/// The parsed capabilities document.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Capabilities {
    /// Always [`CAPABILITIES_FORMAT`].
    pub format: u32,
    /// Always [`CAPABILITIES_KIND`].
    pub kind: String,
    /// Controller build identity.
    pub controller: ControllerInfo,
    /// Protocol formats.
    pub protocols: Protocols,
    /// Advertised operations.
    pub operations: Vec<OperationCapability>,
    /// Registered engine adapters.
    pub engine_adapters: Vec<EngineAdapter>,
    /// Registered snapshot drivers.
    pub snapshot_drivers: Vec<SnapshotDriverContract>,
    /// Host transports.
    pub host_transports: Vec<HostTransport>,
    /// Named features.
    pub features: Vec<String>,
}

impl Capabilities {
    /// Parse a capabilities document.
    pub fn parse(bytes: &[u8]) -> Result<Self, ColdsnapError> {
        let parsed: Self = serde_json::from_slice(bytes).map_err(|error| {
            ColdsnapError::unsupported(
                "the controller's capabilities document",
                format!("it could not be parsed: {error}"),
            )
        })?;
        parsed.check_envelope()?;
        Ok(parsed)
    }

    fn check_envelope(&self) -> Result<(), ColdsnapError> {
        if self.format != CAPABILITIES_FORMAT || self.kind != CAPABILITIES_KIND {
            return Err(ColdsnapError::unsupported(
                "the controller's capabilities document",
                format!(
                    "envelope is format {} kind {:?}, expected format {} kind {:?}",
                    self.format, self.kind, CAPABILITIES_FORMAT, CAPABILITIES_KIND
                ),
            ));
        }
        if !self
            .protocols
            .accepted_request_formats
            .contains(&REQUEST_FORMAT)
        {
            return Err(ColdsnapError::unsupported(
                format!("request format {REQUEST_FORMAT}"),
                format!(
                    "the controller accepts {:?}",
                    self.protocols.accepted_request_formats
                ),
            ));
        }
        if self.protocols.receipt_format != RECEIPT_FORMAT {
            return Err(ColdsnapError::unsupported(
                format!("receipt format {RECEIPT_FORMAT}"),
                format!("the controller emits {}", self.protocols.receipt_format),
            ));
        }
        Ok(())
    }

    /// Whether the controller advertises `operation`.
    pub fn supports_operation(&self, operation: Operation) -> bool {
        self.operations
            .iter()
            .any(|capability| capability.operation == operation.as_wire_str())
    }

    /// Whether the controller advertises an adapter for `engine`.
    pub fn supports_engine(&self, engine: ColdsnapEngine) -> bool {
        self.engine_adapters
            .iter()
            .any(|adapter| adapter.engine == engine.as_wire_str())
    }

    /// Whether the controller advertises `driver`.
    pub fn supports_driver(&self, driver: SnapshotDriver) -> bool {
        self.snapshot_drivers
            .iter()
            .any(|contract| contract.id == driver.as_wire_str())
    }

    /// Whether the controller advertises a named feature.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|name| name == feature)
    }

    /// Refuse to run `operation` on `engine`/`driver` unless this controller
    /// advertises all three, plus every `required_features` entry.
    ///
    /// Admission is a static check; it is not a claim that the target hosts
    /// are ready. ColdSnap makes that distinction itself, and so does this.
    pub fn admit(
        &self,
        operation: Operation,
        engine: ColdsnapEngine,
        driver: SnapshotDriver,
        required_features: &[&str],
    ) -> Result<(), ColdsnapError> {
        if !self.supports_operation(operation) {
            return Err(ColdsnapError::unsupported(
                format!("the {operation} operation"),
                format!(
                    "controller {} advertises {:?}",
                    self.controller.version,
                    self.operations
                        .iter()
                        .map(|capability| capability.operation.as_str())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        if !self.supports_engine(engine) {
            return Err(ColdsnapError::unsupported(
                format!("an engine adapter for {engine}"),
                format!(
                    "controller {} advertises {:?}",
                    self.controller.version,
                    self.engine_adapters
                        .iter()
                        .map(|adapter| adapter.engine.as_str())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        if !self.supports_driver(driver) {
            return Err(ColdsnapError::unsupported(
                format!("the {driver} snapshot driver"),
                format!(
                    "controller {} advertises {:?}",
                    self.controller.version,
                    self.snapshot_drivers
                        .iter()
                        .map(|contract| contract.id.as_str())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        for feature in required_features {
            if !self.has_feature(feature) {
                return Err(ColdsnapError::unsupported(
                    format!("the {feature:?} feature"),
                    format!(
                        "controller {} does not advertise it",
                        self.controller.version
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped from `internal/cli/capabilities.go`. Unknown fields are added by
    /// one test to prove tolerance.
    fn document() -> serde_json::Value {
        serde_json::json!({
            "format": 1,
            "kind": "coldsnap-controller-capabilities",
            "controller": {"version": "0.3.23", "commit": "deadbeef"},
            "protocols": {
                "request_format": 4,
                "accepted_request_formats": [4],
                "artifact_format": 9,
                "accepted_artifact_formats": [9],
                "receipt_format": 2,
                "timing_event_format": 1,
                "host_provider_format": 1,
                "payload_validation_rpc_format": 1
            },
            "operations": [
                {"operation": "capture", "replay_semantics": "conflict-on-existing-output"},
                {"operation": "restore", "replay_semantics": "reconcile-replace"},
                {"operation": "restore", "replay_semantics": "safe-repeat", "prepare_only": true},
                {"operation": "sleep", "replay_semantics": "convergent"},
                {"operation": "status", "replay_semantics": "convergent"},
                {"operation": "wake", "replay_semantics": "convergent"}
            ],
            "engine_adapters": [
                {
                    "engine": "vllm",
                    "executable": "coldsnap-vllm-adapter",
                    "environment": "COLDSNAP_VLLM_ADAPTER",
                    "operations": ["capture", "restore", "sleep", "status", "wake"],
                    "prepare_restore": true,
                    "controller_model": "external-process",
                    "native_materialization": ["off", "async", "required"],
                    "default_native_materialization": "off"
                }
            ],
            "snapshot_drivers": [
                {
                    "id": "n580",
                    "abi": 1,
                    "capture_boundary": "pre-cuda-process-template",
                    "minimum_nvidia_driver_major": 580,
                    "base_requirements": {"features": ["criu-process-template"]}
                },
                {
                    "id": "n610",
                    "abi": 1,
                    "capture_boundary": "initialized-cuda-process-state",
                    "minimum_nvidia_driver_major": 610,
                    "base_requirements": {"features": ["cuda-checkpoint-api"]}
                }
            ],
            "host_transports": [
                {
                    "name": "manager-provider",
                    "engine_adapter_requests": true,
                    "security": "operation-scoped-unix-token",
                    "status": "required"
                }
            ],
            "features": ["operation-receipts", "operation-timing-spans", "prepare-only-restore"]
        })
    }

    fn parsed() -> Capabilities {
        Capabilities::parse(document().to_string().as_bytes()).unwrap()
    }

    #[test]
    fn parses_a_real_shaped_document() {
        let capabilities = parsed();
        assert_eq!(capabilities.controller.version, "0.3.23");
        assert_eq!(capabilities.protocols.request_format, 4);
        assert_eq!(capabilities.snapshot_drivers.len(), 2);
    }

    #[test]
    fn tolerates_unknown_fields_because_it_is_a_discovery_document() {
        let mut value = document();
        value["future_section"] = serde_json::json!({"anything": true});
        value["controller"]["extra"] = serde_json::json!(1);
        let capabilities = Capabilities::parse(value.to_string().as_bytes()).unwrap();
        assert_eq!(capabilities.controller.version, "0.3.23");
    }

    #[test]
    fn rejects_a_wrong_envelope() {
        let mut value = document();
        value["format"] = serde_json::json!(2);
        assert!(Capabilities::parse(value.to_string().as_bytes()).is_err());
    }

    #[test]
    fn rejects_a_controller_that_does_not_accept_our_request_format() {
        let mut value = document();
        value["protocols"]["accepted_request_formats"] = serde_json::json!([3]);
        let err = Capabilities::parse(value.to_string().as_bytes()).unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
    }

    #[test]
    fn rejects_a_controller_that_emits_a_different_receipt_format() {
        let mut value = document();
        value["protocols"]["receipt_format"] = serde_json::json!(1);
        assert!(Capabilities::parse(value.to_string().as_bytes()).is_err());
    }

    #[test]
    fn admission_accepts_a_fully_supported_combination() {
        parsed()
            .admit(
                Operation::Sleep,
                ColdsnapEngine::Vllm,
                SnapshotDriver::N610,
                &["operation-receipts"],
            )
            .unwrap();
    }

    #[test]
    fn admission_refuses_a_missing_engine_adapter() {
        let err = parsed()
            .admit(
                Operation::Sleep,
                ColdsnapEngine::Sglang,
                SnapshotDriver::N610,
                &[],
            )
            .unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
        assert!(err.to_string().contains("sglang"));
    }

    #[test]
    fn admission_refuses_a_missing_feature() {
        let err = parsed()
            .admit(
                Operation::Sleep,
                ColdsnapEngine::Vllm,
                SnapshotDriver::N610,
                &["operation-timing-events-ndjson-v1"],
            )
            .unwrap_err();
        assert!(err.to_string().contains("ndjson"));
    }

    #[test]
    fn admission_refuses_an_unadvertised_driver() {
        let mut value = document();
        value["snapshot_drivers"] = serde_json::json!([]);
        let capabilities = Capabilities::parse(value.to_string().as_bytes()).unwrap();
        assert!(
            capabilities
                .admit(
                    Operation::Sleep,
                    ColdsnapEngine::Vllm,
                    SnapshotDriver::N610,
                    &[]
                )
                .is_err()
        );
    }
}
