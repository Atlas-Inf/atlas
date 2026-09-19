// SPDX-License-Identifier: AGPL-3.0-only

//! The snapshot policy, and the canonical values the controller would apply
//! if a field were omitted.
//!
//! [`SnapshotPolicy::canonical_for`] mirrors `snapshot.DefaultSnapshotPolicyForDriver`.
//! It exists so a caller can be *explicit* without being wrong: ColdSnap will
//! substitute these values anyway, and writing them out makes the request
//! self-describing and the committed digest stable across controller versions.
//!
//! Fields whose value set this crate could not verify from the controller's
//! source stay `String` rather than becoming an enum that guesses.

use crate::protocol::driver::SnapshotDriver;

/// The graph-preservation policy. `preserve-exec` currently downgrades to
/// `recreate-from-plan` for both engines.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphPolicy {
    /// Preserve a graph executable only when all dependencies are proven stable.
    PreserveExec,
    /// Rebuild graph executables from the engine plan. The n580 default.
    RecreateFromPlan,
    /// Keep graph-visible communicator resources, reconstruct transport in
    /// place. The n610 distributed default.
    PreserveNcclExec,
}

/// Kernel compatibility admission.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelCompatibility {
    /// Run the capsule-pinned `criu check` on the destination.
    Capability,
    /// Require an exact `uname` release match.
    Exact,
}

/// Whether a request produces a portable or target-local artifact.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactScope {
    /// Placement-independent.
    Portable,
    /// Only restorable on the capture hosts.
    TargetLocal,
}

/// Native read-through-cache materialization.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NativeMaterialization {
    /// Recovery without populating the native cache. The default everywhere.
    Off,
    /// Return after validation; write native payloads in the background.
    /// vLLM only.
    Async,
    /// Do not report success until every worker's payload is written. vLLM only.
    Required,
}

/// The complete snapshot policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SnapshotPolicy {
    /// Process and memory lifecycle.
    pub process: ProcessPolicy,
    /// Weight-provider selection.
    pub weights: WeightPolicy,
    /// Derived-cache capture.
    pub cache: CachePolicy,
    /// OCI capsule publication.
    pub capsule: CapsulePolicy,
    /// Restore admission checks.
    pub compatibility: CompatibilityPolicy,
}

/// Process and memory lifecycle policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProcessPolicy {
    /// Checkpoint backend, e.g. `cuda-criu`.
    pub backend: String,
    /// Discard KV payloads semantically rather than storing them.
    pub kv_discard: bool,
    /// Allow asynchronous CUDA graph capture.
    pub async_graphs: bool,
    /// Graph-preservation policy. Omitted means the driver's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_policy: Option<GraphPolicy>,
    /// Shape calibration ownership, e.g. `auto`.
    pub shape_calibration: String,
    /// Portable or target-local.
    pub artifact_scope: ArtifactScope,
}

/// Weight-provider policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightPolicy {
    /// Selection mode, e.g. `auto` (native then recovery).
    pub mode: String,
    /// Native payload provider.
    pub native: NativePolicy,
    /// Safetensors recovery provider.
    pub recovery: RecoveryPolicy,
}

/// Native model payload provider.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct NativePolicy {
    /// Repository the payload is published to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Payload revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Worker id → payload file.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub files_by_worker: std::collections::BTreeMap<String, String>,
    /// Staged payload descriptors. Adapter-owned shape; left opaque rather
    /// than guessed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub staged: Vec<serde_json::Value>,
    /// Read-through-cache materialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub materialize: Option<NativeMaterialization>,
}

/// Safetensors recovery provider.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RecoveryPolicy {
    /// Whether recovery is permitted.
    pub enabled: bool,
    /// Recovery source, e.g. `huggingface-safetensors`.
    pub source: String,
    /// Loader backend, e.g. `auto`.
    pub loader_backend: String,
}

/// Derived-cache capture policy.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CachePolicy {
    /// Whether compiler/JIT caches are captured into the capsule.
    pub seed: bool,
    /// Cache roots to seed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<String>,
    /// Staged cache descriptors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub staged: Vec<PreparedCache>,
}

/// A staged per-unit cache copy.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PreparedCache {
    /// Owning unit id.
    pub unit: String,
    /// Cache path.
    pub path: String,
}

/// Capsule publication policy.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct CapsulePolicy {
    /// Repository capsules are pushed to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
}

/// Restore admission checks stricter than the driver's own contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompatibilityPolicy {
    /// Require the destination driver to meet the captured floor.
    pub enforce_captured_driver_floor: bool,
    /// Kernel admission mode.
    pub kernel: KernelCompatibility,
}

/// `snapshot.CanonicalRuntimeCachePath`.
pub const CANONICAL_RUNTIME_CACHE_PATH: &str = "/var/cache/coldsnap/runtime";

impl SnapshotPolicy {
    /// The policy ColdSnap would resolve for `driver` if the caller omitted
    /// one — written out explicitly.
    ///
    /// The only difference between the two drivers is `graph_policy`, which
    /// is exactly why this takes the driver rather than defaulting.
    pub fn canonical_for(driver: SnapshotDriver) -> Self {
        Self {
            process: ProcessPolicy {
                backend: "cuda-criu".to_owned(),
                kv_discard: true,
                async_graphs: true,
                graph_policy: Some(match driver {
                    SnapshotDriver::N580 => GraphPolicy::RecreateFromPlan,
                    SnapshotDriver::N610 => GraphPolicy::PreserveNcclExec,
                }),
                shape_calibration: "auto".to_owned(),
                artifact_scope: ArtifactScope::Portable,
            },
            weights: WeightPolicy {
                mode: "auto".to_owned(),
                native: NativePolicy::default(),
                recovery: RecoveryPolicy {
                    enabled: true,
                    source: "huggingface-safetensors".to_owned(),
                    loader_backend: "auto".to_owned(),
                },
            },
            cache: CachePolicy {
                seed: true,
                paths: vec![CANONICAL_RUNTIME_CACHE_PATH.to_owned()],
                staged: Vec::new(),
            },
            capsule: CapsulePolicy::default(),
            compatibility: CompatibilityPolicy {
                enforce_captured_driver_floor: true,
                kernel: KernelCompatibility::Capability,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_only_driver_difference_is_graph_policy() {
        let n580 = SnapshotPolicy::canonical_for(SnapshotDriver::N580);
        let n610 = SnapshotPolicy::canonical_for(SnapshotDriver::N610);
        assert_eq!(
            n580.process.graph_policy,
            Some(GraphPolicy::RecreateFromPlan)
        );
        assert_eq!(
            n610.process.graph_policy,
            Some(GraphPolicy::PreserveNcclExec)
        );

        let mut n580_normalised = n580.clone();
        n580_normalised.process.graph_policy = None;
        let mut n610_normalised = n610.clone();
        n610_normalised.process.graph_policy = None;
        assert_eq!(n580_normalised, n610_normalised);
    }

    #[test]
    fn canonical_policy_matches_the_documented_defaults() {
        let policy = SnapshotPolicy::canonical_for(SnapshotDriver::N610);
        assert_eq!(policy.process.backend, "cuda-criu");
        assert!(policy.process.kv_discard);
        assert!(policy.process.async_graphs);
        assert_eq!(policy.weights.mode, "auto");
        assert!(policy.weights.recovery.enabled);
        assert_eq!(policy.weights.recovery.source, "huggingface-safetensors");
        assert!(policy.cache.seed);
        assert_eq!(policy.cache.paths, [CANONICAL_RUNTIME_CACHE_PATH]);
        assert!(policy.compatibility.enforce_captured_driver_floor);
        assert_eq!(policy.compatibility.kernel, KernelCompatibility::Capability);
        // Both drivers default native materialization to `off`.
        assert_eq!(policy.weights.native.materialize, None);
    }

    #[test]
    fn graph_policy_serializes_in_kebab_case() {
        let json = serde_json::to_string(&GraphPolicy::PreserveNcclExec).unwrap();
        assert_eq!(json, "\"preserve-nccl-exec\"");
        let json = serde_json::to_string(&KernelCompatibility::Capability).unwrap();
        assert_eq!(json, "\"capability\"");
        let json = serde_json::to_string(&ArtifactScope::TargetLocal).unwrap();
        assert_eq!(json, "\"target-local\"");
    }
}
