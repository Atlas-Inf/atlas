// SPDX-License-Identifier: AGPL-3.0-only

use super::{GraphEntry, GraphHandle, GraphIdentity, GraphLease, GraphRuntime};

impl GraphRuntime {
    pub fn invalidate_matching(
        &self,
        reason: super::GraphFallbackReason,
        predicate: impl FnMut(&super::GraphKey) -> bool,
    ) -> usize {
        let entries = self.cache.lock().remove_matching(predicate);
        let count = entries.len();
        let phases: Vec<super::GraphPhase> =
            entries.iter().map(|entry| entry.key().phase).collect();
        self.retire(entries);
        if count > 0 {
            for phase in phases {
                self.metrics.record_fallback(phase, reason);
            }
        }
        self.poll_retirements();
        count
    }

    pub fn clear_captured(&self, reason: super::GraphFallbackReason) -> usize {
        self.invalidate_matching(reason, |_| true)
    }

    pub fn register_captured(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: super::GraphCost,
        handle: GraphHandle,
        topology_dot: Option<std::path::PathBuf>,
    ) -> Result<GraphLease, super::GraphRuntimeError> {
        if let Err(error) = self.validate_identity(&identity) {
            let _ = self.backend.destroy_graph(handle);
            return Err(error);
        }
        if handle.0 == 0 {
            return Err(self.fallback(
                identity.key.phase,
                super::GraphFallbackReason::InstantiateFailed,
                "backend returned a null executable graph",
            ));
        }
        let policy = self.policies.phase(identity.key.phase);
        if policy.max_entries == 0
            || own_cost.total_bytes() > policy.max_estimated_bytes
            || !self
                .cache
                .lock()
                .cost_fits_global_limit(own_cost.total_bytes())
        {
            let _ = self.backend.destroy_graph(handle);
            return Err(self.fallback(
                identity.key.phase,
                super::GraphFallbackReason::CacheQuota,
                "captured graph exceeds its phase or global quota",
            ));
        }
        Ok(self.insert_executable(identity, stream, own_cost, Vec::new(), handle, topology_dot))
    }

    pub(super) fn insert_executable(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: super::GraphCost,
        dependencies: Vec<GraphLease>,
        handle: GraphHandle,
        topology_dot: Option<std::path::PathBuf>,
    ) -> GraphLease {
        let phase_policy = self.policies.phase(identity.key.phase);
        let artifact_identity = identity.clone();
        let entry = GraphEntry::new(
            identity,
            stream,
            own_cost,
            dependencies,
            self.backend.clone(),
            handle,
        );
        let phase = entry.key().phase;
        let inserted = self.cache.lock().insert(entry.clone());
        self.metrics
            .record_capture(phase, inserted.recaptured.is_some());
        self.metrics
            .add_bytes(phase, entry.own_cost().total_bytes());
        let prewarm_eligible = phase_policy.prewarm_enabled
            && super::artifact::identity_prewarmable(&artifact_identity);
        self.artifacts.lock().record_capture(
            artifact_identity,
            entry.own_cost(),
            topology_dot,
            prewarm_eligible,
        );
        for evicted in &inserted.evicted {
            self.metrics.record_eviction(evicted.key().phase);
        }
        let mut retired = inserted.evicted;
        retired.extend(inserted.recaptured);
        self.retire(retired);
        self.poll_retirements();
        GraphLease(entry)
    }
}
