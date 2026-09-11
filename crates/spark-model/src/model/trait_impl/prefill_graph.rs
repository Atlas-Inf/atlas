// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::graph_runtime::{
    CompatibilityDecision, GraphCost, GraphFallbackReason, GraphIdentity, GraphMode, GraphPayload,
    GraphPhase, GraphRuntimeError, GraphSegment,
};

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_graph_identity(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        proc_count: usize,
        seq_len_start: usize,
        kv_write_start: usize,
        marconi_skip: bool,
        hss_engaged: bool,
    ) -> Option<GraphIdentity> {
        let mode = self.graph_runtime.mode();
        if !matches!(mode, GraphMode::Breakable | GraphMode::Piecewise) {
            return self.decline_prefill_graph(
                GraphFallbackReason::ModeUnsupported,
                "prefill requires breakable or piecewise capture",
            );
        }
        // Feature compatibility (profiling, KV offload) is enforced through
        // the runtime's registry, so the declared rule set and the actual
        // fallback cannot drift: both the rule id and its reason are recorded.
        let active_rules = [
            self.profile.then_some("profile"),
            hss_engaged.then_some("high_speed_swap"),
        ];
        if let CompatibilityDecision::EagerFallback {
            rule_id,
            reason,
            detail,
        } = self
            .graph_runtime
            .compatibility(active_rules.into_iter().flatten(), GraphPhase::Prefill)
        {
            tracing::debug!(
                target: "atlas::cuda_graph",
                rule = %rule_id,
                "prefill graph eager fallback by compatibility rule"
            );
            return self.decline_prefill_graph(reason, detail);
        }
        if self.comm.is_some() {
            return self.decline_prefill_graph(
                GraphFallbackReason::FeatureUnsupported,
                "multi-rank prefill graph capture is not qualified",
            );
        }
        if self.prefill_graph_veto {
            return self.decline_prefill_graph(
                GraphFallbackReason::ModelUnsupported,
                "a layer owns per-sequence or host-selected graph state",
            );
        }
        if self.proposer.is_some() {
            return self.decline_prefill_graph(
                GraphFallbackReason::FeatureUnsupported,
                "drafter prefill capture has separate mutable state",
            );
        }
        if self.lora.is_some() || self.lora_rotatable {
            return self.decline_prefill_graph(
                GraphFallbackReason::FeatureUnsupported,
                "prefill graph capture with LoRA is not qualified",
            );
        }
        if self.tokens_have_vision_pad(tokens) {
            return self.decline_prefill_graph(
                GraphFallbackReason::FeatureUnsupported,
                "vision embedding overlays have request-specific topology",
            );
        }
        if marconi_skip || kv_write_start != 0 || seq_len_start != 0 || proc_count != tokens.len() {
            return self.decline_prefill_graph(
                GraphFallbackReason::DynamicTopology,
                "prefix reuse changes the exact prefill compute range",
            );
        }
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return self.decline_prefill_graph(
                GraphFallbackReason::FeatureUnsupported,
                "model calibration or diagnostics suppress graph capture",
            );
        }
        let slot = seq.ssm_slot_idx().unwrap_or(seq.slot_idx) as u32;
        match self.graph_runtime.identity(
            GraphSegment::PrefillCompute,
            GraphPayload::Prefill {
                request_count: 1,
                token_count: proc_count as u32,
                sequence_lengths: vec![proc_count as u32],
                slots: vec![slot],
                block_counts: vec![seq.block_table.len() as u32],
                chunk_start: seq_len_start as u32,
                last_chunk: true,
                paged: false,
                mrope: self.config.mrope_interleaved,
            },
        ) {
            Ok(identity) => Some(identity),
            Err(error) => {
                tracing::debug!(
                    target: "atlas::cuda_graph",
                    reason = error.reason.label(),
                    detail = %error.detail,
                    "prefill graph eager fallback"
                );
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_chunk_graph_identity(
        &self,
        seq: &SequenceState,
        proc_count: usize,
        effective_seq_len_start: usize,
        kv_write_start: usize,
        marconi_skip: bool,
        is_last_chunk: bool,
        needs_paged: bool,
        use_mrope: bool,
        hss_engaged: bool,
        profile_now: bool,
        diagnostics_armed: bool,
        midchunk_capture: bool,
    ) -> Option<GraphIdentity> {
        let mode = self.graph_runtime.mode();
        let refusal = if !matches!(mode, GraphMode::Breakable | GraphMode::Piecewise) {
            Some((
                GraphFallbackReason::ModeUnsupported,
                "chunked prefill requires segmented capture",
            ))
        } else if profile_now || diagnostics_armed {
            Some((
                GraphFallbackReason::Profiling,
                "prefill diagnostics synchronize or perform host I/O",
            ))
        } else if hss_engaged {
            Some((
                GraphFallbackReason::HighSpeedSwap,
                "prefill may perform host KV offload",
            ))
        } else if self.comm.is_some() {
            Some((
                GraphFallbackReason::FeatureUnsupported,
                "multi-rank prefill graph capture is not qualified",
            ))
        } else if self.prefill_graph_veto {
            Some((
                GraphFallbackReason::ModelUnsupported,
                "a layer owns per-sequence or host-selected graph state",
            ))
        } else if self.proposer.is_some() || self.lora.is_some() || self.lora_rotatable {
            Some((
                GraphFallbackReason::FeatureUnsupported,
                "drafter or LoRA prefill state is not qualified",
            ))
        } else if self.vision_encoder.is_some() || midchunk_capture {
            Some((
                GraphFallbackReason::DynamicTopology,
                "vision or mid-chunk snapshot work changes graph topology",
            ))
        } else if marconi_skip || kv_write_start != 0 {
            Some((
                GraphFallbackReason::DynamicTopology,
                "prefix reuse changes the exact prefill compute range",
            ))
        } else if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            Some((
                GraphFallbackReason::FeatureUnsupported,
                "model calibration suppresses graph capture",
            ))
        } else {
            None
        };
        if let Some((reason, detail)) = refusal {
            return self.decline_prefill_graph(reason, detail);
        }
        let slot = seq.ssm_slot_idx().unwrap_or(seq.slot_idx) as u32;
        self.graph_runtime
            .identity(
                GraphSegment::PrefillLayers,
                GraphPayload::Prefill {
                    request_count: 1,
                    token_count: proc_count as u32,
                    sequence_lengths: vec![proc_count as u32],
                    slots: vec![slot],
                    block_counts: vec![seq.block_table.len() as u32],
                    chunk_start: effective_seq_len_start as u32,
                    last_chunk: is_last_chunk,
                    paged: needs_paged,
                    mrope: use_mrope,
                },
            )
            .map(Some)
            .unwrap_or_else(|error| {
                tracing::debug!(
                    target: "atlas::cuda_graph",
                    reason = error.reason.label(),
                    detail = %error.detail,
                    "chunked prefill graph eager fallback"
                );
                None
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_batch_graph_identity(
        &self,
        slots: Vec<u32>,
        sequence_lengths: Vec<u32>,
        block_counts: Vec<u32>,
        token_count: usize,
        chunk_start: usize,
        is_last_chunk: bool,
        paged: bool,
        mrope: bool,
        cold: bool,
        diagnostics_armed: bool,
    ) -> Option<GraphIdentity> {
        let mode = self.graph_runtime.mode();
        let refusal = if !matches!(mode, GraphMode::Breakable | GraphMode::Piecewise) {
            Some((
                GraphFallbackReason::ModeUnsupported,
                "batched prefill requires segmented capture",
            ))
        } else if self.profile || diagnostics_armed {
            Some((
                GraphFallbackReason::Profiling,
                "batched prefill diagnostics are capture-incompatible",
            ))
        } else if self.comm.is_some() || self.config.num_ssm_layers() > 0 {
            Some((
                GraphFallbackReason::FeatureUnsupported,
                "batched recurrent or multi-rank prefill needs stable per-layer pointer tables",
            ))
        } else if self.prefill_graph_veto || self.proposer.is_some() {
            Some((
                GraphFallbackReason::ModelUnsupported,
                "model or drafter owns unqualified per-sequence state",
            ))
        } else if self.lora.is_some() || self.lora_rotatable || self.vision_encoder.is_some() {
            Some((
                GraphFallbackReason::FeatureUnsupported,
                "batched LoRA or vision prefill is not qualified",
            ))
        } else if !cold {
            Some((
                GraphFallbackReason::DynamicTopology,
                "batched prefix reuse changes graph topology",
            ))
        } else {
            None
        };
        if let Some((reason, detail)) = refusal {
            return self.decline_prefill_graph(reason, detail);
        }
        self.graph_runtime
            .identity(
                GraphSegment::PrefillLayers,
                GraphPayload::Prefill {
                    request_count: slots.len() as u32,
                    token_count: token_count as u32,
                    sequence_lengths,
                    slots,
                    block_counts,
                    chunk_start: chunk_start as u32,
                    last_chunk: is_last_chunk,
                    paged,
                    mrope,
                },
            )
            .map(Some)
            .unwrap_or_else(|error| {
                tracing::debug!(
                    target: "atlas::cuda_graph",
                    reason = error.reason.label(),
                    detail = %error.detail,
                    "batched prefill graph eager fallback"
                );
                None
            })
    }

    pub(super) fn prefill_graph_cost(&self) -> GraphCost {
        GraphCost {
            estimated_bytes: (self.layers.len() as u64).saturating_mul(1024 * 1024),
            node_count: (self.layers.len() as u64)
                .saturating_mul(32)
                .saturating_add(2),
            child_count: 0,
            staging_bytes: 0,
        }
    }

    pub(super) fn execute_prefill_graph<F>(
        &self,
        identity: Option<GraphIdentity>,
        stream: u64,
        mut body: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(bool) -> anyhow::Result<()>,
    {
        let Some(identity) = identity else {
            return body(false);
        };
        match self.graph_runtime.lookup(&identity) {
            Ok(Some(graph)) => {
                self.graph_runtime.launch(&graph, stream).map_err(|error| {
                    anyhow::anyhow!("prefill CUDA graph replay failed: {error}")
                })?;
                return Ok(());
            }
            Ok(None) => {}
            Err(error) if Self::prefill_capture_may_fallback(&error) => return body(false),
            Err(error) => return Err(anyhow::anyhow!("prefill graph lookup failed: {error}")),
        }
        match self.graph_runtime.capture(
            identity,
            stream,
            self.prefill_graph_cost(),
            Vec::new(),
            spark_runtime::graph_runtime::CaptureFailurePolicy::Retry,
            |_| body(true),
        ) {
            Ok(graph) => self
                .graph_runtime
                .launch(&graph, stream)
                .map_err(|error| anyhow::anyhow!("prefill CUDA graph launch failed: {error}")),
            Err(error) if Self::prefill_capture_may_fallback(&error) => {
                tracing::warn!(
                    target: "atlas::cuda_graph",
                    reason = error.reason.label(),
                    detail = %error.detail,
                    "prefill graph capture failed; running this request eagerly"
                );
                body(false)
            }
            Err(error) => Err(anyhow::anyhow!(
                "prefill graph body failed before execution: {error}"
            )),
        }
    }

    pub(super) fn prefill_capture_may_fallback(error: &GraphRuntimeError) -> bool {
        matches!(
            error.reason,
            GraphFallbackReason::RuntimeDisabled
                | GraphFallbackReason::PhaseDisabled
                | GraphFallbackReason::ModeUnsupported
                | GraphFallbackReason::ShapeUnsupported
                | GraphFallbackReason::CacheQuota
                | GraphFallbackReason::NegativeCached
                | GraphFallbackReason::CaptureFailed
                | GraphFallbackReason::InstantiateFailed
                | GraphFallbackReason::MemoryPressure
        )
    }

    fn decline_prefill_graph(
        &self,
        reason: GraphFallbackReason,
        detail: impl Into<String>,
    ) -> Option<GraphIdentity> {
        let detail = detail.into();
        self.graph_runtime
            .record_eager_fallback(GraphPhase::Prefill, reason);
        tracing::debug!(
            target: "atlas::cuda_graph",
            reason = reason.label(),
            detail = %detail,
            "prefill graph eager fallback"
        );
        None
    }
}
