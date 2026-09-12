// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    GRAPH_KEY_SCHEMA_VERSION, GRAPH_MANIFEST_SCHEMA_VERSION, GRAPH_PREWARM_SCHEMA_VERSION,
    GraphArtifactIo, GraphIdentity, GraphManifest, GraphPrewarmProfile, GraphRuntime,
    NativeGraphSerialization,
};
use std::path::Path;

impl GraphRuntime {
    pub fn capture_dot_path(&self, identity: &GraphIdentity) -> Option<std::path::PathBuf> {
        self.export_dir
            .as_ref()
            .filter(|_| self.capabilities.debug_dot)
            .map(|directory| {
                let hash = super::graph_key_hash(&identity.key);
                directory.join(format!("{}-{}.dot", identity.key.phase, &hash[..16]))
            })
    }

    pub fn take_prewarm_requests(&self) -> Vec<GraphIdentity> {
        std::mem::take(&mut *self.prewarm_requests.lock())
    }

    pub fn manifest(&self) -> GraphManifest {
        GraphManifest {
            schema_version: GRAPH_MANIFEST_SCHEMA_VERSION,
            key_schema_version: GRAPH_KEY_SCHEMA_VERSION,
            fingerprint: self.fingerprint.clone(),
            mode: self.mode,
            capabilities: self.capabilities,
            native_serialization: NativeGraphSerialization::deliberate_no_go(),
            entries: self.artifacts.lock().entries(),
        }
    }

    pub fn prewarm_profile(&self) -> GraphPrewarmProfile {
        let identities = self
            .artifacts
            .lock()
            .entries()
            .into_iter()
            .filter(|entry| entry.prewarm_eligible && entry.fallback_reason.is_none())
            .map(|entry| entry.identity)
            .collect();
        GraphPrewarmProfile {
            schema_version: GRAPH_PREWARM_SCHEMA_VERSION,
            key_schema_version: GRAPH_KEY_SCHEMA_VERSION,
            fingerprint: self.fingerprint.clone(),
            mode: self.mode,
            identities,
        }
    }

    pub fn export_configured_artifacts(
        &self,
    ) -> anyhow::Result<Option<(std::path::PathBuf, std::path::PathBuf)>> {
        let Some(directory) = &self.export_dir else {
            return Ok(None);
        };
        let manifest_path = directory.join("manifest.json");
        let prewarm_path = directory.join("prewarm.json");
        self.export_artifacts(&super::FsGraphArtifactIo, &manifest_path, &prewarm_path)?;
        Ok(Some((manifest_path, prewarm_path)))
    }

    pub fn export_artifacts(
        &self,
        io: &dyn GraphArtifactIo,
        manifest_path: &Path,
        prewarm_path: &Path,
    ) -> anyhow::Result<()> {
        io.write_atomic(manifest_path, &self.manifest().to_json()?)?;
        io.write_atomic(prewarm_path, &self.prewarm_profile().to_json()?)?;
        Ok(())
    }

    pub fn load_prewarm_profile(
        &self,
        io: &dyn GraphArtifactIo,
        path: &Path,
    ) -> anyhow::Result<Vec<GraphIdentity>> {
        let profile = GraphPrewarmProfile::from_json(&io.read(path)?)?;
        profile
            .validate_for(&self.fingerprint, self.mode)
            .map_err(anyhow::Error::msg)?;
        Ok(profile.identities)
    }

    pub fn validate_manifest(&self, manifest: &GraphManifest) -> Result<(), String> {
        manifest.validate_for(&self.fingerprint, self.mode, self.capabilities)
    }
}
