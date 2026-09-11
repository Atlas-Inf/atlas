// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    BoundedLoopState, CaptureFailurePolicy, GraphCost, GraphFallbackReason, GraphIdentity,
    GraphLease, GraphPayload, GraphRuntime, GraphRuntimeError, graph_key_hash,
};
use crate::gpu::{ConditionalNodeKind, GpuBackend, KernelArg, KernelHandle};

impl GraphRuntime {
    pub fn prepare_bounded_loop(
        &self,
        state: BoundedLoopState,
        stream: u64,
        continue_initially: bool,
        clear_cancellation: bool,
    ) -> anyhow::Result<()> {
        state.validate().map_err(anyhow::Error::msg)?;
        self.backend.memset_async(state.iteration, 0, 4, stream)?;
        self.backend.memset_async(state.exit_reason, 0, 4, stream)?;
        if clear_cancellation {
            self.backend
                .memset_async(state.cancellation, 0, 4, stream)?;
        }
        self.backend.copy_h2d_async(
            &u32::from(continue_initially).to_le_bytes(),
            state.continuation,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn capture_bounded_while<Body>(
        &self,
        identity: GraphIdentity,
        stream: u64,
        own_cost: GraphCost,
        state: BoundedLoopState,
        predicate_setter: KernelHandle,
        loop_update: KernelHandle,
        failure_policy: CaptureFailurePolicy,
        body: Body,
    ) -> Result<GraphLease, GraphRuntimeError>
    where
        Body: FnOnce(&dyn GpuBackend, BoundedLoopState) -> anyhow::Result<()>,
    {
        self.validate_identity(&identity)?;
        state.validate().map_err(|detail| {
            self.fallback(
                identity.key.phase,
                GraphFallbackReason::ShapeUnsupported,
                detail,
            )
        })?;
        let shape_matches = matches!(
            &identity.key.payload,
            GraphPayload::DecodeLoop {
                max_iterations,
                output_capacity,
                ..
            } if *max_iterations == state.max_iterations
                && *output_capacity == state.output_capacity
        );
        if identity.key.segment != super::GraphSegment::LoopBody || !shape_matches {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::StaleKey,
                "bounded loop key does not encode its hard cap and output capacity",
            ));
        }
        if !self.capabilities.while_nodes {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::DriverUnsupported,
                "CUDA WHILE graph nodes are unavailable",
            ));
        }
        let policy = self.policies.phase(identity.key.phase);
        if !policy.capture_enabled
            || own_cost.total_bytes() > policy.max_estimated_bytes
            || !self
                .cache
                .lock()
                .cost_fits_global_limit(own_cost.total_bytes())
        {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::CacheQuota,
                "bounded loop is disabled or exceeds its phase quota",
            ));
        }
        if predicate_setter.0 == 0 || loop_update.0 == 0 {
            return Err(self.fallback(
                identity.key.phase,
                GraphFallbackReason::FeatureUnsupported,
                "bounded loop control kernels are unavailable",
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
            .create_conditional_graph(
                ConditionalNodeKind::While,
                state.continuation,
                predicate_setter,
                1,
            )
            .map_err(|error| {
                self.capture_error(
                    &identity,
                    GraphFallbackReason::CaptureFailed,
                    failure_policy,
                    error,
                )
            })?;
        let result = (|| {
            self.backend
                .begin_conditional_branch(&template, 0, stream)
                .map_err(|error| {
                    self.capture_error(
                        &identity,
                        GraphFallbackReason::CaptureFailed,
                        failure_policy,
                        error,
                    )
                })?;
            if let Err(error) = body(self.backend.as_ref(), state) {
                self.backend.abort_capture_if_active(stream);
                return Err(self.capture_error(
                    &identity,
                    GraphFallbackReason::CaptureBodyFailed,
                    failure_policy,
                    error,
                ));
            }
            let handle = template.conditional_handle.to_le_bytes();
            let max_iterations = state.max_iterations.to_le_bytes();
            if let Err(error) = self.backend.launch_typed(
                loop_update,
                [1, 1, 1],
                [1, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&handle),
                    KernelArg::Buffer(state.continuation),
                    KernelArg::Buffer(state.iteration),
                    KernelArg::Bytes(&max_iterations),
                    KernelArg::Buffer(state.cancellation),
                    KernelArg::Buffer(state.exit_reason),
                ],
            ) {
                self.backend.abort_capture_if_active(stream);
                return Err(self.capture_error(
                    &identity,
                    GraphFallbackReason::CaptureBodyFailed,
                    failure_policy,
                    error,
                ));
            }
            self.backend
                .end_conditional_branch(&template, 0, stream)
                .map_err(|error| {
                    self.backend.abort_capture_if_active(stream);
                    self.capture_error(
                        &identity,
                        GraphFallbackReason::CaptureFailed,
                        failure_policy,
                        error,
                    )
                })?;
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
        Ok(self.insert_executable(identity, stream, own_cost, Vec::new(), handle, topology_dot))
    }
}
