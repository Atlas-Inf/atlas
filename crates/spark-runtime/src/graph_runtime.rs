// SPDX-License-Identifier: AGPL-3.0-only

mod artifact;
mod bounded_loop;
mod cache;
mod capture_guard;
mod compatibility;
mod conditional;
mod entry;
mod fallback;
mod metrics;
mod policy;
mod runtime_artifacts;
mod runtime_identity;
mod runtime_insert;
mod types;

pub use artifact::{
    FsGraphArtifactIo, GRAPH_MANIFEST_SCHEMA_VERSION, GRAPH_PREWARM_SCHEMA_VERSION,
    GraphArtifactIo, GraphManifest, GraphManifestEntry, GraphPrewarmProfile,
    NativeGraphSerialization, graph_key_hash,
};
pub use compatibility::{CompatibilityDecision, CompatibilityRegistry, CompatibilityRule};
pub use entry::GraphLease;
pub use metrics::{GraphMetrics, GraphMetricsSnapshot};
pub use policy::{GraphPolicies, GraphRuntimeConfig, PhasePolicy};
pub use types::{
    BoundedLoopExitReason, BoundedLoopState, CaptureFailurePolicy, GRAPH_KEY_SCHEMA_VERSION,
    GraphCapabilities, GraphCost, GraphEnvironment, GraphFallbackReason, GraphFingerprint,
    GraphIdentity, GraphKey, GraphMode, GraphPayload, GraphPhase, GraphRuntimeError, GraphSegment,
    ShapeBucket, SpeculativeAlgorithm, classify_shape_bucket,
};

use artifact::ArtifactCatalog;
use cache::GraphCache;
use capture_guard::CaptureGuard;
use entry::GraphEntry;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::{GpuBackend, GraphHandle};

pub struct GraphRuntime {
    backend: Arc<dyn GpuBackend>,
    mode: GraphMode,
    algorithm: SpeculativeAlgorithm,
    capabilities: GraphCapabilities,
    policies: GraphPolicies,
    fingerprint: GraphFingerprint,
    shape_buckets: Vec<ShapeBucket>,
    resource_generation: u64,
    compatibility: CompatibilityRegistry,
    export_dir: Option<std::path::PathBuf>,
    artifacts: Mutex<ArtifactCatalog>,
    prewarm_requests: Mutex<Vec<GraphIdentity>>,
    cache: Mutex<GraphCache>,
    retirement: Mutex<Vec<Arc<GraphEntry>>>,
    streams: Mutex<HashSet<u64>>,
    capture_lock: Mutex<()>,
    metrics: Arc<GraphMetrics>,
    shutting_down: AtomicBool,
}

impl GraphRuntime {
    pub fn new(
        backend: Arc<dyn GpuBackend>,
        capabilities: GraphCapabilities,
        config: GraphRuntimeConfig,
        metrics: Arc<GraphMetrics>,
    ) -> Result<Self, String> {
        config.validate()?;
        let prewarm_requests = if let Some(profile) = &config.prewarm_profile {
            profile.validate_for(&config.fingerprint, config.mode)?;
            profile.identities.clone()
        } else {
            Vec::new()
        };
        let compatibility = CompatibilityRegistry::new(config.compatibility_rules)?;
        metrics.set_algorithm(config.speculative_algorithm);
        Ok(Self {
            backend,
            mode: config.mode,
            algorithm: config.speculative_algorithm,
            capabilities,
            policies: config.policies,
            fingerprint: config.fingerprint,
            shape_buckets: config.shape_buckets,
            resource_generation: config.resource_generation,
            compatibility,
            export_dir: config.export_dir,
            artifacts: Mutex::new(ArtifactCatalog::default()),
            prewarm_requests: Mutex::new(prewarm_requests),
            cache: Mutex::new(GraphCache::new(
                config.policies,
                config.max_cache_entries,
                config.max_cache_bytes,
            )),
            retirement: Mutex::new(Vec::new()),
            streams: Mutex::new(HashSet::new()),
            capture_lock: Mutex::new(()),
            metrics,
            shutting_down: AtomicBool::new(false),
        })
    }

    pub fn lookup(
        &self,
        identity: &GraphIdentity,
    ) -> Result<Option<GraphLease>, GraphRuntimeError> {
        self.validate_identity(identity)?;
        if !self.policies.phase(identity.key.phase).replay_enabled {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::PhaseDisabled,
                format!("{} graph replay is disabled", identity.key.phase),
            ));
        }
        self.poll_retirements();
        Ok(self.cache.lock().get(&identity.key))
    }

    pub fn capture<F>(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: GraphCost,
        dependencies: Vec<GraphLease>,
        failure_policy: CaptureFailurePolicy,
        body: F,
    ) -> Result<GraphLease, GraphRuntimeError>
    where
        F: FnOnce(&dyn GpuBackend) -> anyhow::Result<()>,
    {
        self.validate_identity(&identity)?;
        let phase_policy = self.policies.phase(identity.key.phase);
        if !phase_policy.capture_enabled {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::PhaseDisabled,
                format!("{} graph capture is disabled", identity.key.phase),
            ));
        }
        if let Some(reason) = self.cache.lock().negative_reason(&identity.key) {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::NegativeCached,
                format!("previous deterministic capture failure: {}", reason.label()),
            ));
        }
        if dependencies
            .iter()
            .any(|dependency| !dependency.is_active())
        {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::Retired,
                "conditional child graph was evicted before parent capture",
            ));
        }
        let total_cost = dependencies
            .iter()
            .fold(own_cost, |cost, dependency| cost.include(dependency.cost()));
        if phase_policy.max_entries == 0
            || total_cost.total_bytes() > phase_policy.max_estimated_bytes
            || !self
                .cache
                .lock()
                .cost_fits_global_limit(total_cost.total_bytes())
        {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::CacheQuota,
                format!(
                    "graph needs {} estimated bytes but {} phase quota is {}",
                    total_cost.total_bytes(),
                    identity.key.phase,
                    phase_policy.max_estimated_bytes
                ),
            ));
        }
        if let Some(existing) = self.cache.lock().get(&identity.key) {
            return Ok(existing);
        }

        self.poll_retirements();
        self.streams.lock().insert(stream);
        let key_hash = graph_key_hash(&identity.key);
        let topology_dot = self
            .export_dir
            .as_ref()
            .filter(|_| self.capabilities.debug_dot)
            .map(|directory| {
                directory.join(format!("{}-{}.dot", identity.key.phase, &key_hash[..16]))
            });
        let _capture_lock = self.capture_lock.lock();
        if let Some(existing) = self.cache.lock().get(&identity.key) {
            return Ok(existing);
        }
        let guard = CaptureGuard::begin(self.backend.as_ref(), stream).map_err(|error| {
            self.capture_error(
                &identity,
                GraphFallbackReason::CaptureFailed,
                failure_policy,
                error,
            )
        })?;
        if let Err(error) = body(self.backend.as_ref()) {
            return Err(self.capture_error(
                &identity,
                GraphFallbackReason::CaptureBodyFailed,
                failure_policy,
                error,
            ));
        }
        let handle = guard.finish(topology_dot.as_deref()).map_err(|error| {
            self.capture_error(
                &identity,
                GraphFallbackReason::InstantiateFailed,
                failure_policy,
                error,
            )
        })?;
        if handle.0 == 0 {
            return Err(self.capture_error(
                &identity,
                GraphFallbackReason::InstantiateFailed,
                failure_policy,
                anyhow::anyhow!("backend returned a null executable graph"),
            ));
        }

        Ok(self.insert_executable(
            identity,
            stream,
            own_cost,
            dependencies,
            handle,
            topology_dot,
        ))
    }

    pub fn prewarm<F>(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: GraphCost,
        dependencies: Vec<GraphLease>,
        failure_policy: CaptureFailurePolicy,
        body: F,
    ) -> Result<GraphLease, GraphRuntimeError>
    where
        F: FnOnce(&dyn GpuBackend) -> anyhow::Result<()>,
    {
        if !self.policies.phase(identity.key.phase).prewarm_enabled {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::PhaseDisabled,
                format!("{} graph prewarm is disabled", identity.key.phase),
            ));
        }
        self.capture(
            identity,
            stream,
            own_cost,
            dependencies,
            failure_policy,
            body,
        )
    }

    pub fn launch(&self, lease: &GraphLease, stream: u64) -> Result<(), GraphRuntimeError> {
        let phase = lease.key().phase;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::RuntimeDisabled,
                "graph runtime is shutting down",
            ));
        }
        if !self.policies.phase(phase).replay_enabled {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::PhaseDisabled,
                format!("{phase} graph replay is disabled"),
            ));
        }
        self.streams.lock().insert(stream);
        if let Err(error) = lease.0.launch(stream) {
            self.metrics.record_replay_failure();
            self.metrics.record_fallback(phase, error.reason);
            if let Some(entry) = self.cache.lock().remove(lease.key()) {
                self.retire(vec![entry]);
            }
            return Err(error);
        }
        self.metrics.record_launch(phase);
        self.metrics.record_replay(phase);
        self.cache.lock().touch(lease.key());
        Ok(())
    }

    pub fn record_eager_fallback(&self, phase: GraphPhase, reason: GraphFallbackReason) {
        self.metrics.record_fallback(phase, reason);
    }

    pub fn invalidate(&self, key: &GraphKey, reason: GraphFallbackReason) {
        if let Some(entry) = self.cache.lock().remove(key) {
            self.retire(vec![entry]);
        }
        self.metrics.record_fallback(key.phase, reason);
        self.poll_retirements();
    }

    pub fn negative_cache(&self, key: GraphKey, reason: GraphFallbackReason) {
        self.cache.lock().block(key, reason);
    }

    pub fn poll_retirements(&self) {
        let mut released = Vec::new();
        {
            let mut pending = self.retirement.lock();
            let mut index = 0;
            while index < pending.len() {
                if Arc::strong_count(&pending[index]) > 1 {
                    index += 1;
                    continue;
                }
                let entry = &pending[index];
                if entry.try_fence_retirement().is_ok() && entry.poll_retirement().unwrap_or(false)
                {
                    released.push((entry.key().phase, entry.own_cost().total_bytes()));
                    pending.swap_remove(index);
                } else {
                    index += 1;
                }
            }
        }
        for (phase, bytes) in released {
            self.metrics.remove_bytes(phase, bytes);
        }
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        if self.shutting_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let entries = self.cache.lock().drain();
        self.retire(entries);
        let streams: Vec<u64> = self.streams.lock().iter().copied().collect();
        for stream in streams {
            self.backend.synchronize(stream)?;
        }
        let mut pending = self.retirement.lock();
        pending.sort_by_key(|entry| std::cmp::Reverse(entry.dependency_count()));
        let mut released: Vec<(GraphPhase, u64)> = Vec::new();
        for entry in pending.iter() {
            if entry.force_destroy_after_sync()? {
                released.push((entry.key().phase, entry.own_cost().total_bytes()));
            }
        }
        pending.clear();
        for (phase, bytes) in released {
            self.metrics.remove_bytes(phase, bytes);
        }
        Ok(())
    }

    #[cfg(test)]
    fn resident_counts(&self) -> [usize; GraphPhase::COUNT] {
        self.cache.lock().resident_counts()
    }

    fn validate_identity(&self, identity: &GraphIdentity) -> Result<(), GraphRuntimeError> {
        let phase = identity.key.phase;
        if self.shutting_down.load(Ordering::Acquire) || self.mode == GraphMode::Disabled {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::RuntimeDisabled,
                "CUDA graph runtime is disabled",
            ));
        }
        if !self.capabilities.basic_graphs {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::DriverUnsupported,
                "backend does not expose basic CUDA graph support",
            ));
        }
        identity
            .key
            .validate()
            .map_err(|detail| self.fallback(phase, GraphFallbackReason::StaleKey, detail))?;
        if identity.key.mode != self.mode {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::StaleKey,
                format!(
                    "key mode {} does not match runtime mode {}",
                    identity.key.mode, self.mode
                ),
            ));
        }
        if identity.key.speculative_algorithm != self.algorithm {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::StaleKey,
                format!(
                    "key algorithm {} does not match runtime algorithm {}",
                    identity.key.speculative_algorithm, self.algorithm
                ),
            ));
        }
        let (tokens, requests) = identity.key.payload.shape_counts();
        if !identity.bucket.contains(tokens, requests) {
            return Err(self.fallback(
                phase,
                GraphFallbackReason::ShapeUnsupported,
                format!(
                    "concrete shape ({tokens} tokens, {requests} requests) exceeds bucket ({}, {})",
                    identity.bucket.token_limit, identity.bucket.request_limit
                ),
            ));
        }
        Ok(())
    }

    fn capture_error(
        &self,
        identity: &GraphIdentity,
        reason: GraphFallbackReason,
        policy: CaptureFailurePolicy,
        error: anyhow::Error,
    ) -> GraphRuntimeError {
        self.metrics.record_capture_failure();
        if policy == CaptureFailurePolicy::NegativeCache {
            self.cache.lock().block(identity.key.clone(), reason);
        }
        self.artifacts
            .lock()
            .record_fallback(identity.clone(), reason);
        self.fallback(identity.key.phase, reason, format!("{error:#}"))
    }

    fn fallback(
        &self,
        phase: GraphPhase,
        reason: GraphFallbackReason,
        detail: impl Into<String>,
    ) -> GraphRuntimeError {
        self.metrics.record_fallback(phase, reason);
        GraphRuntimeError::new(reason, detail)
    }

    fn retire(&self, entries: Vec<Arc<GraphEntry>>) {
        let mut pending = self.retirement.lock();
        for entry in entries {
            if entry.mark_evicted()
                && !pending
                    .iter()
                    .any(|candidate| Arc::ptr_eq(candidate, &entry))
            {
                pending.push(entry);
            }
        }
    }
}

impl Drop for GraphRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
#[path = "graph_runtime/tests.rs"]
mod tests;
