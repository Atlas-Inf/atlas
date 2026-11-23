// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` hooks for the native Ling decoder block.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::layers::FfnComponent;

use super::*;

impl TransformerLayer for LingDecoderLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let metadata = match &self.attention {
            LingAttention::Mla { .. } => {
                let metadata = ctx
                    .attn_metadata
                    .ok_or_else(|| anyhow::anyhow!("Ling MLA decode requires AttnMetadataDev"))?;
                ensure!(
                    metadata.num_seqs == 1,
                    "Ling MLA decode requires one metadata row"
                );
                Some(metadata)
            }
            LingAttention::Kda { .. } => None,
        };
        self.run_one(
            hidden, residual, state, kv_cache, true, metadata, ctx, stream,
        )
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        if matches!(&self.attention, LingAttention::Kda { .. }) {
            return self.run_kda_prefill(hidden, residual, num_tokens, state, ctx, stream);
        }
        if matches!(&self.attention, LingAttention::Mla { .. }) {
            return self.run_mla_prefill(
                hidden,
                residual,
                num_tokens,
                state,
                kv_cache,
                seq_len_start,
                kv_write_start,
                ctx,
                stream,
            );
        }
        unreachable!("Ling decoder has neither KDA nor MLA attention")
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        if matches!(&self.attention, LingAttention::Kda { .. }) {
            let h_bytes = self.hidden_size * 2;
            for token in 0..num_tokens {
                self.decode(
                    hidden.offset(token * h_bytes),
                    residual.offset(token * h_bytes),
                    state,
                    kv_cache,
                    seq_len + token,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        let meta = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Ling MLA verify requires AttnMetadataDev"))?;
        ensure!(
            meta.num_seqs as usize >= num_tokens,
            "Ling MLA verify metadata has {} rows for {num_tokens} tokens",
            meta.num_seqs
        );
        let h_bytes = self.hidden_size * 2;
        for token in 0..num_tokens {
            self.run_one(
                hidden.offset(token * h_bytes),
                residual.offset(token * h_bytes),
                state,
                kv_cache,
                true,
                Some(Self::metadata_row(meta, token)),
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let meta = ctx.attn_metadata;
        if matches!(&self.attention, LingAttention::Mla { .. }) {
            let meta =
                meta.ok_or_else(|| anyhow::anyhow!("Ling MLA batch requires AttnMetadataDev"))?;
            ensure!(
                meta.num_seqs as usize >= num_seqs,
                "Ling MLA metadata row count is too small"
            );
        }
        ensure!(states.len() >= active_seqs && seq_lens.len() >= active_seqs);
        let h_bytes = self.hidden_size * 2;
        for i in 0..active_seqs.min(num_seqs) {
            let row_meta = meta.map(|m| Self::metadata_row(m, i));
            self.run_one(
                hidden.offset(i * h_bytes),
                residual.offset(i * h_bytes),
                states[i],
                kv_cache,
                true,
                row_meta,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        match &self.attention {
            LingAttention::Kda { .. } => {
                Self::alloc_kda_state(gpu, self.h_state_bytes, self.conv_state_bytes)
            }
            LingAttention::Mla { .. } => Ok(Box::new(EmptyLayerState)),
        }
    }

    fn uses_local_mla_prefill(&self) -> bool {
        matches!(&self.attention, LingAttention::Mla { .. })
    }

    fn decode_graph_unsupported(&self) -> bool {
        matches!(&self.ffn, FfnComponent::LingMoe(ffn) if !ffn.supports_cuda_graph())
    }

    fn multi_seq_decode_graph_unsupported(&self) -> bool {
        // Each row carries a different KDA state (and MLA cache metadata),
        // while this layer has no pre-capture state-pointer table. A graph
        // would bake the first batch's per-sequence addresses.
        true
    }
}
