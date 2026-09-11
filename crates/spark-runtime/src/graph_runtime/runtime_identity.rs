// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    CompatibilityDecision, GRAPH_KEY_SCHEMA_VERSION, GraphCapabilities, GraphFallbackReason,
    GraphIdentity, GraphKey, GraphLease, GraphMetricsSnapshot, GraphMode, GraphPayload, GraphPhase,
    GraphRuntime, GraphRuntimeError, GraphSegment, classify_shape_bucket,
};
use std::sync::atomic::Ordering;

impl GraphRuntime {
    pub fn mode(&self) -> super::GraphMode {
        self.mode
    }

    pub fn capabilities(&self) -> GraphCapabilities {
        self.capabilities
    }

    pub fn metrics(&self) -> GraphMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn identity(
        &self,
        segment: GraphSegment,
        payload: GraphPayload,
    ) -> Result<GraphIdentity, GraphRuntimeError> {
        let (tokens, requests) = payload.shape_counts();
        let phase = payload.phase();
        // An identity for a runtime that cannot capture or replay is not
        // usable, and returning one made every caller believe capture was
        // available: the model set `use_graphs = identity.is_some()` and then
        // failed later in `register_captured` with `runtime_disabled` instead
        // of running eagerly. Reject here, like `lookup`/`capture`/`register`.
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
        let bucket =
            classify_shape_bucket(&self.shape_buckets, tokens, requests).ok_or_else(|| {
                self.fallback(
                    phase,
                    GraphFallbackReason::ShapeUnsupported,
                    format!("no graph bucket covers {tokens} tokens and {requests} requests"),
                )
            })?;
        Ok(GraphIdentity {
            bucket,
            key: GraphKey {
                schema_version: GRAPH_KEY_SCHEMA_VERSION,
                phase: payload.phase(),
                mode: self.mode,
                segment,
                speculative_algorithm: self.algorithm,
                fingerprint: self.fingerprint.clone(),
                resource_generation: self.resource_generation,
                payload,
            },
        })
    }

    pub fn lookup_key(&self, key: &GraphKey) -> Result<Option<GraphLease>, GraphRuntimeError> {
        key.validate()
            .map_err(|detail| self.fallback(key.phase, GraphFallbackReason::StaleKey, detail))?;
        if key.mode != self.mode
            || key.speculative_algorithm != self.algorithm
            || key.fingerprint != self.fingerprint
            || key.resource_generation != self.resource_generation
        {
            return Err(self.fallback(
                key.phase,
                GraphFallbackReason::StaleKey,
                "graph key does not match the active runtime",
            ));
        }
        if !self.policies.phase(key.phase).replay_enabled {
            return Err(self.fallback(
                key.phase,
                GraphFallbackReason::PhaseDisabled,
                format!("{} graph replay is disabled", key.phase),
            ));
        }
        self.poll_retirements();
        Ok(self.cache.lock().get(key))
    }

    pub fn cached_graphs(&self, phase: GraphPhase) -> Vec<GraphLease> {
        self.cache.lock().leases(phase)
    }

    pub fn compatibility<'a>(
        &self,
        active_rule_ids: impl IntoIterator<Item = &'a str>,
        phase: GraphPhase,
    ) -> CompatibilityDecision {
        self.compatibility.decide(active_rule_ids, phase, self.mode)
    }
}
