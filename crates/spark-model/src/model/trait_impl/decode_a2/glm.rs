// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 guards and lane selection for the batched decode in `decode_a2.rs`:
//! independent/C4 lane validation, the sparse multi-sequence graph gate,
//! per-step KV/shape checks and the exact distributed dispatch width.

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

/// EP peers receive the exact batch width in the decode-batch protocol, and
/// both ranks capture/replay the same width. Keep that width exact instead of
/// rounding 3 live GLM requests up to the generic 4-row graph bucket: the
/// dummy fourth row otherwise runs every KDA/MLA layer and consumes 25% of
/// the distributed decode work without producing a token.
pub(super) fn decode_dispatch_width(n: usize, glm5_distributed: bool) -> usize {
    if glm5_distributed {
        n
    } else {
        crate::traits::padded_batch_n(n)
    }
}

impl TransformerModel {
    /// Returns `(padded_n, ssm_batch)` for `decode_batch_compute_main`.
    pub(super) fn glm_padded_batch<'a>(
        &'a self,
        seqs: &[&mut SequenceState],
        n: usize,
        stream: u64,
    ) -> Result<(usize, Option<crate::layer::ssm_batch::SsmBatchView<'a>>)> {
        // Local serving uses the generic captured-graph ladder. Distributed
        // EP uses the exact protocol width so C=3 does not execute a dummy
        // fourth row on both ranks.
        let glm5_distributed = self.comm.is_some() && self.config.model_type == "glm5_next";
        let padded_n = decode_dispatch_width(n, glm5_distributed);

        // Both ranks validate their own actual slot guards/state pointers and
        // refresh the fixed ID stream before any graph lookup or replay.
        let ssm_batch =
            crate::model::ssm_indexed_decode::prepare_runtime(self, seqs, n, padded_n, stream)?;
        Ok((padded_n, ssm_batch))
    }

    /// Independent-lane checks at the top of `decode_batch_dispatch`.
    pub(super) fn glm_dispatch_guard(
        &self,
        tokens: &[u32],
        seqs: &[&mut SequenceState],
    ) -> Result<()> {
        self.validate_independent_decode(tokens, seqs)?;
        if crate::model::glm_independent::enabled(&self.config.model_type)? {
            anyhow::ensure!(
                self.comm.as_ref().is_some_and(|c| c.rank() == 0),
                "independent decode sender requires head rank0"
            );
        }
        Ok(())
    }

    /// Returns `(independent_lane, c4_lane)` after validating the C4 lane.
    pub(super) fn glm_decode_lanes(
        &self,
        tokens: &[u32],
        seqs: &[&mut SequenceState],
        n: usize,
    ) -> Result<(bool, bool)> {
        self.validate_independent_decode(tokens, seqs)?;
        let independent_lane = crate::model::glm_independent::enabled(&self.config.model_type)?;
        let c4_lane = self.config.model_type == "glm5_next"
            && !independent_lane
            && (n == 4 || crate::model::glm_c4::enabled(&self.config.model_type));
        if c4_lane {
            crate::model::glm_c4::validate_runtime(
                &self.config,
                self.comm.as_ref().map_or(0, |comm| comm.world_size()),
                self.ep_protocol_v2,
                self.proposer.is_none() && !self.self_speculative,
            )?;
            crate::model::glm_c4::validate_positions(seqs.iter().map(|s| s.seq_len), n)?;
            anyhow::ensure!(
                self.levers.max_decode_seqs == 4,
                "initial GLM C4 lane requires active cap4"
            );
            crate::model::glm_c4::validate_scratch(self.buffers.sizes(), self.config.hc_mult)?;
        }
        Ok((independent_lane, c4_lane))
    }

    /// GLM sparse multi-sequence graph gate. Returns `(glm_dynamic,
    /// glm_graphs_ok)` where `glm_graphs_ok` is the GLM share of the graph
    /// eligibility predicate.
    pub(super) fn glm_multiseq_graph_gate(&self) -> Result<(bool, bool)> {
        // The eager selector embeds host lengths. Only the explicit dynamic
        // variant has fixed topology/addresses and device-driven predicates.
        let glm_sparse =
            crate::layers::qwen3_attention::glm_multi_seq_sparse_enabled(&self.config.model_type);
        let glm_dynamic = crate::layers::qwen3_attention::glm_multi_seq_sparse_graphs_enabled(
            &self.config.model_type,
        )?;
        anyhow::ensure!(
            !glm_sparse || (self.proposer.is_none() && !self.self_speculative),
            "GLM multi-sequence sparse decode does not support speculative decoding"
        );
        let dynamic_capture_ok = !glm_dynamic
            || (!self.profile
                && !self
                    .suppress_graphs
                    .load(std::sync::atomic::Ordering::Relaxed));
        Ok((
            glm_dynamic,
            (!glm_sparse || glm_dynamic) && dynamic_capture_ok,
        ))
    }

    pub(super) fn validate_multiseq_graph_ops(
        &self,
        padded_n: usize,
        use_graphs: bool,
    ) -> Result<()> {
        // The optional M16 numerical oracle performs host readback. Reject
        // actual graph use before lookup, warmup, or beginning capture.
        crate::layers::moe::validate_m16_gate_up_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_shared_fp8_cache_graphs(&self.config.model_type, use_graphs)?;
        crate::layers::moe::validate_m5_projection_graphs(
            &self.config.model_type,
            padded_n,
            use_graphs,
        )?;
        Ok(())
    }

    pub(super) fn validate_glm_c4_kv(
        &self,
        kv_cache: &PagedKvCache,
        seqs: &[&mut SequenceState],
    ) -> Result<()> {
        use spark_runtime::kv_cache::{KvCacheDtype, SparseIndexCacheDtype};
        anyhow::ensure!(
            kv_cache.dtype() == KvCacheDtype::Bf16
                && (0..kv_cache.num_layers())
                    .all(|i| kv_cache.dtype_for_layer(i) == KvCacheDtype::Bf16)
                && kv_cache
                    .sparse_index_config()
                    .is_some_and(|s| s.dtype == SparseIndexCacheDtype::Bf16),
            "GLM C4 requires BF16 KV and semantic index"
        );
        let bs = kv_cache.block_size();
        let capacity = (self.max_blocks_per_seq as usize)
            .checked_mul(bs)
            .ok_or_else(|| anyhow::anyhow!("GLM C4 table capacity overflow"))?;
        anyhow::ensure!(
            bs > 0 && seqs.iter().all(|s| s.seq_len < capacity),
            "GLM C4 position exceeds fixed table capacity"
        );
        Ok(())
    }

    pub(super) fn validate_glm_dynamic_step(
        &self,
        kv_cache: &PagedKvCache,
        seqs: &[&mut SequenceState],
        n: usize,
        padded_n: usize,
    ) -> Result<()> {
        // These checks must run on every step, BEFORE graph lookup/replay.
        // Layer validation alone only executes on the capture/cache miss.
        anyhow::ensure!(
            padded_n == n,
            "GLM sparse graphs require exact unpadded C2/C3 width"
        );
        let shape = ops::GlmDynamicShape::new(
            self.max_blocks_per_seq,
            u32::try_from(kv_cache.block_size())?,
        )?;
        shape.validate_positions(
            seqs.iter().map(|s| s.seq_len),
            n,
            self.config.max_position_embeddings,
        )?;
        let sizes = self.buffers.sizes();
        shape.validate_arenas(sizes.expert_down_out, sizes.qkv_output)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::decode_dispatch_width;

    #[test]
    fn distributed_decode_keeps_exact_protocol_width() {
        assert_eq!(decode_dispatch_width(3, true), 3);
        assert_eq!(decode_dispatch_width(3, false), 4);
    }
}
