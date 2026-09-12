// SPDX-License-Identifier: AGPL-3.0-only

//! The launch topology: units, workers, rank groups, services, and the
//! adapter-owned payload.
//!
//! ColdSnap's `ExecutionGraph` deliberately does not assume that engine
//! parallelism is a Cartesian product, so these are the exact four shapes the
//! controller validates — no more, no fewer. `adapter.payload` stays an opaque
//! [`serde_json::Value`]: its schema belongs to the engine adapter, and
//! modelling it here would be this crate inventing a contract it does not own.

use std::collections::BTreeMap;

/// One OS/container process-tree boundary.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LaunchUnit {
    /// Unit id, matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.
    pub id: String,
    /// Caller-defined ordering.
    pub index: i64,
    /// Host the unit runs on. Placement, not artifact identity.
    pub host: String,
    /// Device ordinals or UUIDs visible to this unit.
    pub devices: Vec<String>,
    /// Digest-pinned image reference.
    pub image: String,
    /// The same image's digest.
    pub image_digest: String,
    /// The semantic command. Transport environment is placement.
    pub command: Vec<String>,
    /// Non-transport environment.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub environment: BTreeMap<String, String>,
    /// Mounts. Targets, not sources, are part of the capture contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mounts: Vec<Mount>,
}

/// A bind mount for a launch unit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Mount {
    /// Host path.
    pub source: String,
    /// Container path.
    pub target: String,
    /// Whether the mount is read-only.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub read_only: bool,
}

/// A portable process slot inside a launch unit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Worker {
    /// Worker id.
    pub id: String,
    /// Owning unit id.
    pub unit: String,
    /// Owning service id.
    pub service: String,
    /// Process slot within the unit.
    pub process_slot: i64,
    /// Indices into the owning unit's `devices` list.
    pub device_slots: Vec<i64>,
}

/// An ordered rank namespace. `kind` is adapter-owned and namespaced, e.g.
/// `vllm:world`; core ColdSnap does not enumerate parallelism modes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessGroup {
    /// Group id.
    pub id: String,
    /// Adapter-owned kind.
    pub kind: String,
    /// Owning service id.
    pub service: String,
    /// Member worker ids, in rank order.
    pub members: Vec<String>,
}

/// An independently addressable engine or cooperating role.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServiceDomain {
    /// Service id.
    pub id: String,
    /// Adapter-owned role, e.g. `vllm:serve`.
    pub role: String,
    /// Member worker ids.
    pub workers: Vec<String>,
}

/// The adapter-owned topology descriptor.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdapterTopology {
    /// Adapter-owned schema name.
    pub schema: String,
    /// `sha256:` digest of the canonicalized payload.
    pub digest: String,
    /// Opaque adapter payload. The adapter owns its schema.
    pub payload: serde_json::Value,
}

/// The captured process topology.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionGraph {
    /// Accelerator-owning engine processes.
    pub workers: Vec<Worker>,
    /// Ordered rank namespaces.
    pub groups: Vec<ProcessGroup>,
    /// Addressable engines and roles.
    pub services: Vec<ServiceDomain>,
    /// The adapter-owned descriptor.
    pub adapter: AdapterTopology,
}
