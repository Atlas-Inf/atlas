// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3AttentionLayer;
use crate::layer::{
    BatchedAttnMetadata, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layers::FfnComponent;

mod decode_inner;
mod diag;
mod multi_seq;
pub(crate) use multi_seq::{
    grouped_routed_decode_enabled, grouped_routed_decode_min, pairwise_moe_decode_enabled,
};
mod prefill_inner;
mod prefill_inner_glm;
pub(super) use diag::diag_norm;
pub use diag::diag_norm_f32;

#[path = "trait_impl/state.rs"]
mod state;

impl TransformerLayer for Qwen3AttentionLayer {
    fn decode_glm_long_owners(
        &self,
        owners: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        cache: &mut PagedKvCache,
        stage: &crate::layer::glm_long_owner::GlmLongStage,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_glm_long_owners_mla(owners, cache, stage, ctx, stream)
    }

    fn prefill_with_glm_passengers(
        &self,
        _hidden: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        seq_len_start: usize,
        passengers: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_glm_passengers_mla(num_tokens, seq_len_start, passengers, cache, ctx, stream)
    }

    fn uses_local_mla_prefill(&self) -> bool {
        // GLM-5 prefill (`prefill_attention_paged_glm_dense`) absorbs Q and
        // reads the complete paged latent + index history, exactly as chunk 1+
        // of any long prompt does, so a prefix-cache skip is just a later
        // chunk start. The other MLA paths attend within the current chunk.
        self.mla
            .as_ref()
            .is_some_and(|mla| mla.glm_indexer.is_none())
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn fp8_calibration_frozen(&self) -> Option<bool> {
        self.fp8_calibration
            .as_ref()
            .map(|cal| !cal.is_calibrating())
    }

    fn supports_mla_kv_only(&self) -> bool {
        self.mla.as_ref().is_some_and(|mla| mla.rope == 0)
    }

    fn prefill_mla_kv_only(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        slots: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.prefill_mla_kv_only_impl(hidden, num_tokens, kv_cache, slots, ctx, stream)
    }

    /// An indexer vetoes decode-graph capture: its ingest counter is host
    /// state, its launch parameters depend on the position, the default top-k
    /// arm sorts on the host — and a graph captured on the dense path would
    /// replay wrong attention once selection activates.
    fn decode_graph_unsupported(&self) -> bool {
        self.qsa.is_some()
    }

    /// Inert-step capture (`layers/qsa_staged.rs`): the mHC highway layer
    /// with plain BF16 KV whose decode reads only device metadata, the
    /// indexer's ingest staged. No MLA, adapters or host KV tier.
    fn qsa_inert_capturable(&self) -> bool {
        self.qsa_rows_static_ok() && self.lora.is_none()
    }

    fn qsa_commit_staged(
        &self,
        state: &mut dyn LayerState,
        row: usize,
        pos: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let qsa = self
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qsa_commit_staged without an indexer"))?;
        let st = super::helpers::qsa_seq_state(qsa, state, gpu)?;
        qsa.commit_staged_row(st, row, pos, gpu, stream)
    }

    fn has_aux_state(&self) -> bool {
        self.qsa.is_some()
    }

    fn aux_bytes_per_token(&self) -> usize {
        self.qsa.as_ref().map_or(0, |q| q.bytes_per_token())
    }

    fn reserve_aux(
        &self,
        state: &mut dyn LayerState,
        tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        let st = super::helpers::qsa_seq_state(qsa, state, gpu)?;
        // Callers reserve whole KV blocks; the last one may reach past the
        // served context, which no position ever will.
        qsa.reserve(st, tokens.min(qsa.capacity()), gpu, stream)
    }

    /// `None` when the batched path can serve an ACTIVE selection per row
    /// (`multi_seq/qsa_rows.rs`) — the same static allow-list the pre-mutation
    /// guard applies, so the scheduler and the layer cannot disagree.
    fn verify_context_limit(&self) -> Option<usize> {
        if self.qsa_rows_static_ok() {
            return None;
        }
        self.verify_context_limit_multi_seq()
    }

    fn verify_context_limit_multi_seq(&self) -> Option<usize> {
        self.qsa.as_ref().map(|q| q.inert_bound())
    }

    fn rollback_aux_verify(
        &self,
        state: &mut dyn LayerState,
        num_accepted: usize,
        k: usize,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        let Some(attn) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
        else {
            return Ok(());
        };
        if let Some(st) = attn.qsa.as_mut() {
            anyhow::ensure!(
                num_accepted <= k,
                "QSA rollback: {num_accepted} accepted of a {k}-row verify"
            );
            qsa.rewind_verify(st, k - num_accepted)?;
        }
        Ok(())
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(None);
        };
        let attn = state
            .as_any()
            .downcast_ref::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        match attn.qsa.as_ref() {
            Some(st) => Ok(Some(qsa.snapshot_aux(st, gpu, stream)?)),
            // Sequence never reached this layer's ingest: nothing to carry.
            None => Ok(None),
        }
    }

    fn snapshot_aux_into(
        &self,
        state: &dyn LayerState,
        buf: &mut Vec<u8>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(false);
        };
        let attn = state
            .as_any()
            .downcast_ref::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        match attn.qsa.as_ref() {
            Some(st) => {
                qsa.snapshot_aux_into(st, buf, gpu, stream)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let qsa = self
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restore_aux: no QSA on this layer"))?;
        let attn = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        if attn.qsa.is_none() {
            attn.qsa = Some(qsa.new_seq_state(gpu)?);
        }
        qsa.restore_aux(attn.qsa.as_mut().expect("just created"), blob, gpu, stream)
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            None, // batched_meta — single-stream
            ctx,
            stream,
        )
    }

    /// Q12 Path B: batched-mode attention prefill via `prefill_inner` with
    /// `batched_meta = Some`. The model-level `prefill_attn_batched_layer`
    /// calls this method. Per-stream block_table is unused under batched
    /// mode (block_table_ptrs from batched_meta carries them); we still
    /// pass an empty Vec to satisfy the signature.
    fn prefill_inner_batched_q12(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        batched_meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut empty_state = EmptyLayerState;
        let mut empty_block_table: Vec<u32> = Vec::new();
        let mut empty_disk_block_ids: Vec<u32> = Vec::new();
        let mut empty_disk_last: Vec<u32> = Vec::new();
        self.prefill_inner(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            &mut empty_state,
            kv_cache,
            seq_len_start,
            &mut empty_block_table,
            &mut empty_disk_block_ids,
            &mut empty_disk_last,
            0,
            Some(batched_meta),
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            active_seqs,
            states,
            None,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn decode_multi_seq_rows<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_rows: usize,
        seq_states: &'a mut [&'b mut (dyn LayerState + 'static)],
        row_owner: &[usize],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_rows,
            num_rows,
            seq_states,
            Some(row_owner),
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(crate::layer::AttnLayerState::default()))
    }

    /// Release the per-sequence QSA indexer carry.
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        state::free_attention_state(self, gpu, state)
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Vec<DevicePtr>> {
        let mut freed = Vec::new();
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            freed.extend(moe.transpose_for_prefill_unified(gpu, config)?);
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            freed.extend(moe.transpose_for_prefill_unified(gpu, config)?);
        }
        Ok(freed)
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "trait_impl/state_tests.rs"]
mod tests;
