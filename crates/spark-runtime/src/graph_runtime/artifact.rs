// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    GRAPH_KEY_SCHEMA_VERSION, GraphCapabilities, GraphCost, GraphFallbackReason, GraphFingerprint,
    GraphIdentity, GraphKey, GraphMode,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const GRAPH_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const GRAPH_PREWARM_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NativeGraphSerialization {
    Unsupported { reason: String },
}

impl NativeGraphSerialization {
    pub fn deliberate_no_go() -> Self {
        Self::Unsupported {
            reason: "CUDA exposes no portable serialization contract for process-local graph executables"
                .to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphManifestEntry {
    pub key_hash: String,
    pub identity: GraphIdentity,
    pub cost: GraphCost,
    pub topology_dot: Option<PathBuf>,
    pub prewarm_eligible: bool,
    pub fallback_reason: Option<GraphFallbackReason>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphManifest {
    pub schema_version: u32,
    pub key_schema_version: u32,
    pub fingerprint: GraphFingerprint,
    pub mode: GraphMode,
    pub capabilities: GraphCapabilities,
    pub native_serialization: NativeGraphSerialization,
    pub entries: Vec<GraphManifestEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphPrewarmProfile {
    pub schema_version: u32,
    pub key_schema_version: u32,
    pub fingerprint: GraphFingerprint,
    pub mode: GraphMode,
    pub identities: Vec<GraphIdentity>,
}

#[derive(Default)]
pub(super) struct ArtifactCatalog {
    entries: BTreeMap<String, GraphManifestEntry>,
}

impl ArtifactCatalog {
    pub fn record_capture(
        &mut self,
        identity: GraphIdentity,
        cost: GraphCost,
        topology_dot: Option<PathBuf>,
        prewarm_eligible: bool,
    ) {
        let hash = graph_key_hash(&identity.key);
        self.entries.insert(
            hash.clone(),
            GraphManifestEntry {
                key_hash: hash,
                identity,
                cost,
                topology_dot,
                prewarm_eligible,
                fallback_reason: None,
            },
        );
    }

    pub fn record_fallback(&mut self, identity: GraphIdentity, reason: GraphFallbackReason) {
        let hash = graph_key_hash(&identity.key);
        self.entries
            .entry(hash.clone())
            .and_modify(|entry| entry.fallback_reason = Some(reason))
            .or_insert(GraphManifestEntry {
                key_hash: hash,
                identity,
                cost: GraphCost {
                    estimated_bytes: 0,
                    node_count: 0,
                    child_count: 0,
                    staging_bytes: 0,
                },
                topology_dot: None,
                prewarm_eligible: false,
                fallback_reason: Some(reason),
            });
    }

    pub fn entries(&self) -> Vec<GraphManifestEntry> {
        self.entries.values().cloned().collect()
    }
}

pub(super) fn identity_prewarmable(identity: &GraphIdentity) -> bool {
    match &identity.key.payload {
        super::GraphPayload::Prefill {
            request_count,
            chunk_start,
            ..
        } => *request_count == 1 && *chunk_start == 0,
        super::GraphPayload::Decode { .. } => true,
        super::GraphPayload::DecodeLoop { .. }
        | super::GraphPayload::Verify { .. }
        | super::GraphPayload::Propose { .. }
        | super::GraphPayload::Fused { .. } => false,
    }
}

pub fn graph_key_hash(key: &GraphKey) -> String {
    let encoded = serde_json::to_vec(key).expect("GraphKey serialization cannot fail");
    let mut digest = Sha256::new();
    digest.update(encoded);
    format!("{:x}", digest.finalize())
}

impl GraphManifest {
    pub fn validate_for(
        &self,
        fingerprint: &GraphFingerprint,
        mode: GraphMode,
        capabilities: GraphCapabilities,
    ) -> Result<(), String> {
        if self.schema_version != GRAPH_MANIFEST_SCHEMA_VERSION
            || self.key_schema_version != GRAPH_KEY_SCHEMA_VERSION
        {
            return Err("graph manifest schema is incompatible with this runtime".to_string());
        }
        if &self.fingerprint != fingerprint {
            return Err("graph manifest fingerprint does not match the loaded runtime".to_string());
        }
        if self.mode != mode {
            return Err(
                "graph manifest capture mode does not match the loaded runtime".to_string(),
            );
        }
        if self.capabilities.conditional_nodes && !capabilities.conditional_nodes {
            return Err("graph manifest requires CUDA conditional nodes".to_string());
        }
        if self.capabilities.while_nodes && !capabilities.while_nodes {
            return Err("graph manifest requires CUDA WHILE nodes".to_string());
        }
        for entry in &self.entries {
            entry.identity.key.validate()?;
            if entry.identity.key.fingerprint != self.fingerprint {
                return Err(format!(
                    "graph manifest entry {} has a stale fingerprint",
                    entry.key_hash
                ));
            }
            if graph_key_hash(&entry.identity.key) != entry.key_hash {
                return Err(format!(
                    "graph manifest entry {} has a stale key hash",
                    entry.key_hash
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

impl GraphPrewarmProfile {
    pub fn validate_for(
        &self,
        fingerprint: &GraphFingerprint,
        mode: GraphMode,
    ) -> Result<(), String> {
        if self.schema_version != GRAPH_PREWARM_SCHEMA_VERSION
            || self.key_schema_version != GRAPH_KEY_SCHEMA_VERSION
        {
            return Err(
                "graph prewarm profile schema is incompatible with this runtime".to_string(),
            );
        }
        if &self.fingerprint != fingerprint || self.mode != mode {
            return Err(
                "graph prewarm profile was produced by an incompatible runtime".to_string(),
            );
        }
        for identity in &self.identities {
            identity.key.validate()?;
            if identity.key.fingerprint != self.fingerprint {
                return Err("graph prewarm profile contains a stale key".to_string());
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

pub trait GraphArtifactIo: Send + Sync {
    fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>>;
    fn write_atomic(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<()>;
}

pub struct FsGraphArtifactIo;

impl GraphArtifactIo for FsGraphArtifactIo {
    fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        Ok(std::fs::read(path)?)
    }

    fn write_atomic(&self, path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
        let parent = path.parent().ok_or_else(|| {
            anyhow::anyhow!("graph artifact path has no parent: {}", path.display())
        })?;
        std::fs::create_dir_all(parent)?;
        let file_name = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("graph artifact path has no file name"))?;
        let temporary = parent.join(format!(
            ".{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id()
        ));
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(temporary, path)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "artifact_tests.rs"]
mod tests;
