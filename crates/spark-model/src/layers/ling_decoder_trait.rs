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
            let state = self.state_kda(state, ctx.config)?;
            return self.run_kda_rows(hidden, residual, num_tokens, &[state], &[], ctx, stream);
        }
        if matches!(&self.attention, LingAttention::Mla { .. }) {
            return self.run_mla_prefill(
                hidden,
                residual,
                num_tokens,
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
            if num_tokens == 2 && self.batch_verify {
                let kda = self.state_kda(state, ctx.config)?;
                let snapshot = Self::verify_snapshot(state)?;
                return self.run_kda_rows(hidden, residual, 2, &[kda], &[snapshot], ctx, stream);
            }
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
                if token + 1 < num_tokens {
                    // A rejected suffix must restore the recurrent matrix and
                    // the three convolution windows just after its last accepted row.
                    let ssm = state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("missing Ling KDA verify state"))?;
                    let h = ssm.h_state_intermediates.get(token).ok_or_else(|| {
                        anyhow::anyhow!("missing Ling KDA verify H snapshot {token}")
                    })?;
                    let conv = ssm.conv_state_intermediates.get(token).ok_or_else(|| {
                        anyhow::anyhow!("missing Ling KDA verify convolution snapshot {token}")
                    })?;
                    ctx.gpu
                        .copy_d2d_async(ssm.h_state, *h, self.h_state_bytes, stream)?;
                    ctx.gpu
                        .copy_d2d_async(ssm.conv_state, *conv, self.conv_state_bytes, stream)?;
                }
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
        if num_tokens == 2 && self.batch_verify {
            return self.mla_decode_rows(hidden, residual, num_tokens, kv_cache, meta, ctx, stream);
        }
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
        let rows = active_seqs.min(num_seqs);
        if rows == 0 {
            return Ok(());
        }
        if rows == 1 {
            return self.run_one(
                hidden,
                residual,
                states[0],
                kv_cache,
                true,
                meta.map(|m| Self::metadata_row(m, 0)),
                ctx,
                stream,
            );
        }
        match &self.attention {
            LingAttention::Kda { .. } => {
                let states: Vec<_> = states[..rows]
                    .iter_mut()
                    .map(|state| self.state_kda(*state, ctx.config))
                    .collect::<Result<_>>()?;
                self.run_kda_rows(hidden, residual, rows, &states, &[], ctx, stream)
            }
            LingAttention::Mla { .. } => self.mla_decode_rows(
                hidden,
                residual,
                rows,
                kv_cache,
                meta.expect("validated MLA batch metadata"),
                ctx,
                stream,
            ),
        }
    }

    /// The KDA layers of a batched verify: two rows per sequence (its
    /// committed token and its draft), seq-major, in one weight sweep. The
    /// attention layers take `decode_multi_seq_rows`.
    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            matches!(&self.attention, LingAttention::Kda { .. }),
            "Ling MLA verifies through decode_multi_seq_rows"
        );
        ensure!(
            n_seqs > 0
                && ks.len() == n_seqs
                && states.len() >= n_seqs
                && ks.iter().all(|&k| k == 2),
            "Ling batched verify takes two rows for each sequence, got {ks:?}"
        );
        if !self.batch_verify {
            // ATLAS_LING_SERIAL_VERIFY: each sequence's rows one at a time.
            let row_bytes = 2 * self.hidden_size * 2;
            for (i, state) in states[..n_seqs].iter_mut().enumerate() {
                self.decode_batched(
                    hidden.offset(i * row_bytes),
                    residual.offset(i * row_bytes),
                    2,
                    *state,
                    kv_cache,
                    0,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        let mut kda = Vec::with_capacity(n_seqs);
        let mut snapshots = Vec::with_capacity(n_seqs);
        for state in states[..n_seqs].iter_mut() {
            kda.push(self.state_kda(*state, ctx.config)?);
            snapshots.push(Self::verify_snapshot(*state)?);
        }
        self.run_kda_rows(hidden, residual, 2 * n_seqs, &kda, &snapshots, ctx, stream)
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
        // The graph cache keys recurrent pointers by the ordered vector of SSM
        // pool slots and declines capture if any row lacks a stable slot.
        self.sante_ffn.is_some()
            || matches!(&self.ffn, FfnComponent::LingMoe(ffn) if !ffn.supports_batched_prefill())
    }
}

impl LingDecoderLayer {
    /// A KDA sequence's verify snapshot: where its recurrent matrix and
    /// convolution windows land after the verify's first row.
    fn verify_snapshot(state: &mut dyn LayerState) -> Result<(DevicePtr, DevicePtr)> {
        let ssm = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("missing Ling verify slot"))?;
        Ok((
            *ssm.h_state_intermediates
                .first()
                .ok_or_else(|| anyhow::anyhow!("missing Ling verify H row"))?,
            *ssm.conv_state_intermediates
                .first()
                .ok_or_else(|| anyhow::anyhow!("missing Ling verify convolution row"))?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_decode_rows(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let LingAttention::Mla {
            config,
            weights,
            kernels,
            scratch,
            ..
        } = &self.attention
        else {
            anyhow::bail!("MLA decode rows require an attention layer")
        };
        let normed = ctx.buffers.norm_output();
        let attn = ctx.buffers.attn_output();
        let mut scratch = *scratch;
        scratch.projection = LingProjectionScratch::new(
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale(),
            ctx.buffers.sizes().fp8_act_scale,
        );
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            rows as u32,
            self.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        for first in (0..rows).step_by(MLA_TILE_ROWS) {
            let count = MLA_TILE_ROWS.min(rows - first);
            let metadata = AttnMetadataDev {
                num_seqs: count as u32,
                ..Self::metadata_row(meta, first)
            };
            let offset = first * self.hidden_size * 2;
            weights.forward_rows(
                *config,
                *kernels,
                ctx.gpu,
                kv_cache,
                self.attention_layer_idx.expect("Ling MLA needs a KV layer"),
                metadata,
                normed.offset(offset),
                Some(attn.offset(offset)),
                scratch,
                count as u32,
                0,
                None,
                stream,
            )?;
        }
        self.finish_prefill_ffn(hidden, residual, attn, rows, ctx, stream)
    }
}
