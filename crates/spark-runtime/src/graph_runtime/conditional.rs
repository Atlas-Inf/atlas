// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    CaptureFailurePolicy, GraphCost, GraphFallbackReason, GraphIdentity, GraphLease, GraphRuntime,
    GraphRuntimeError, graph_key_hash,
};
use crate::gpu::{ConditionalNodeKind, DevicePtr, GpuBackend, KernelHandle};

impl GraphRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn capture_if_else<Then, Else>(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: GraphCost,
        predicate: DevicePtr,
        predicate_setter: KernelHandle,
        failure_policy: CaptureFailurePolicy,
        then_body: Then,
        else_body: Else,
    ) -> Result<GraphLease, GraphRuntimeError>
    where
        Then: FnOnce(&dyn GpuBackend) -> anyhow::Result<()>,
        Else: FnOnce(&dyn GpuBackend) -> anyhow::Result<()>,
    {
        self.validate_identity(&identity)?;
        if !self.capabilities.conditional_nodes {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::DriverUnsupported,
                "CUDA conditional graph nodes are unavailable",
            ));
        }
        let policy = self.policies.phase(identity.key.phase);
        if !policy.capture_enabled {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::PhaseDisabled,
                format!(
                    "{} conditional graph capture is disabled",
                    identity.key.phase
                ),
            ));
        }
        if own_cost.total_bytes() > policy.max_estimated_bytes
            || !self
                .cache
                .lock()
                .cost_fits_global_limit(own_cost.total_bytes())
        {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::CacheQuota,
                "conditional graph exceeds its phase memory quota",
            ));
        }
        if let Some(existing) = self.cache.lock().get(&identity.key) {
            return Ok(existing);
        }

        self.streams.lock().insert(stream);
        let topology_dot = self
            .export_dir
            .as_ref()
            .filter(|_| self.capabilities.debug_dot)
            .map(|directory| {
                let hash = graph_key_hash(&identity.key);
                directory.join(format!("{}-{}.dot", identity.key.phase, &hash[..16]))
            });
        let _capture_lock = self.capture_lock.lock();
        if let Some(existing) = self.cache.lock().get(&identity.key) {
            return Ok(existing);
        }
        let template = self
            .backend
            .create_conditional_graph(ConditionalNodeKind::If, predicate, predicate_setter, 2)
            .map_err(|error| {
                self.capture_error(
                    &identity,
                    GraphFallbackReason::CaptureFailed,
                    failure_policy,
                    error,
                )
            })?;
        let result = (|| {
            self.capture_conditional_branch(
                &identity,
                &template,
                0,
                stream,
                failure_policy,
                then_body,
            )?;
            self.capture_conditional_branch(
                &identity,
                &template,
                1,
                stream,
                failure_policy,
                else_body,
            )?;
            self.backend
                .instantiate_conditional_graph(&template, topology_dot.as_deref())
                .map_err(|error| {
                    self.capture_error(
                        &identity,
                        GraphFallbackReason::InstantiateFailed,
                        failure_policy,
                        error,
                    )
                })
        })();
        self.backend.destroy_conditional_graph_template(&template);
        let handle = result?;
        if handle.0 == 0 {
            return Err(self.capture_error(
                &identity,
                GraphFallbackReason::InstantiateFailed,
                failure_policy,
                anyhow::anyhow!("backend returned a null conditional graph"),
            ));
        }
        Ok(self.insert_executable(identity, stream, own_cost, Vec::new(), handle, topology_dot))
    }

    fn capture_conditional_branch<Body>(
        &self,
        identity: &GraphIdentity,
        template: &crate::gpu::ConditionalGraphTemplate,
        branch: usize,
        stream: u64,
        failure_policy: CaptureFailurePolicy,
        body: Body,
    ) -> Result<(), GraphRuntimeError>
    where
        Body: FnOnce(&dyn GpuBackend) -> anyhow::Result<()>,
    {
        self.backend
            .begin_conditional_branch(template, branch, stream)
            .map_err(|error| {
                self.capture_error(
                    identity,
                    GraphFallbackReason::CaptureFailed,
                    failure_policy,
                    error,
                )
            })?;
        if let Err(error) = body(self.backend.as_ref()) {
            self.backend.abort_capture_if_active(stream);
            return Err(self.capture_error(
                identity,
                GraphFallbackReason::CaptureBodyFailed,
                failure_policy,
                error,
            ));
        }
        self.backend
            .end_conditional_branch(template, branch, stream)
            .map_err(|error| {
                self.backend.abort_capture_if_active(stream);
                self.capture_error(
                    identity,
                    GraphFallbackReason::CaptureFailed,
                    failure_policy,
                    error,
                )
            })
    }
}
