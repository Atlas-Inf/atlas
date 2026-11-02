// SPDX-License-Identifier: AGPL-3.0-only

//! Incremental, request-owned native MTP context; no full-history hidden arena.
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::sync::atomic::Ordering;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::speculative::DraftProposer;
use crate::traits::SequenceState;

#[path = "native_mtp_geometry.rs"]
mod geometry;

impl TransformerModel {
    fn native_mtp_forward_context(&self) -> ForwardContext<'_> {
        ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        }
    }

    /// Normalize and consume this request's fresh prompt rows while they are
    /// live. The private proposer retains one boundary row for the next chunk.
    pub(super) fn capture_native_mtp_prefill(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        start: usize,
        rows: usize,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if !self.native_glm_mtp || self.proposer.is_none() || seq.disable_mtp || rows == 0 {
            return Ok(());
        }
        let end = start
            .checked_add(rows)
            .context("native MTP prompt span overflow")?;
        let chunk = tokens
            .get(start..end)
            .context("native MTP prompt tokens unavailable")?;
        ensure!(
            rows <= self.buffers.max_batch_tokens(),
            "native MTP prompt rows exceed arena"
        );
        let normed = self.buffers.norm_output();
        crate::layers::ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            rows as u32,
            self.config.hidden_size as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        // A warm exact prefix may recompute its last target row for logits.
        // Consume only rows beyond the request's restored predictor boundary.
        let proposer = self
            .proposer
            .as_ref()
            .context("native MTP proposer missing")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("native MTP request state missing")?;
        let retained = proposer
            .native_target_hidden(state.as_ref())?
            .map(|(end, _)| end);
        let covered = geometry::covered_rows(start, rows, retained).map_err(anyhow::Error::msg)?;
        let row_bytes = self
            .config
            .hidden_size
            .checked_mul(2)
            .context("native MTP row overflow")?;
        if covered == rows {
            proposer.retain_native_target_hidden(
                normed.offset((rows - 1) * row_bytes),
                end,
                state.as_mut(),
                self.gpu.as_ref(),
                stream,
            )?;
        } else {
            self.append_native_mtp_target(
                seq,
                &chunk[covered..],
                start + covered,
                normed.offset(covered * row_bytes),
                stream,
            )?;
        }
        self.save_native_mtp_prefix(seq, tokens, end, stream)
    }

    fn append_native_mtp_target(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        start: usize,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.evict_native_mtp_prefix_for_rows(tokens.len())?;
        let proposer = self
            .proposer
            .as_ref()
            .context("native MTP proposer missing")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("native MTP request state missing")?;
        proposer.append_native_target_context(
            tokens,
            start,
            normed,
            state.as_mut(),
            &self.native_mtp_forward_context(),
            stream,
        )
    }

    /// Serial target steps append the boundary pair immediately, keeping
    /// thinking/adaptive serial stretches bounded regardless of their length.
    pub(super) fn capture_native_mtp_serial(
        &self,
        seq: &mut SequenceState,
        token: u32,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if !self.native_glm_mtp || self.proposer.is_none() || seq.disable_mtp {
            return Ok(());
        }
        let start = seq
            .seq_len
            .checked_sub(1)
            .context("native MTP serial row underflow")?;
        self.append_native_mtp_target(seq, &[token], start, normed, stream)
    }

    /// Verification retains the final committed target row. The scheduler
    /// separately trims predictor rows and repairs each accepted draft pair.
    pub(super) fn capture_native_mtp_committed(
        &self,
        seq: &mut SequenceState,
        rows: usize,
    ) -> Result<()> {
        if !self.native_glm_mtp || self.proposer.is_none() || seq.disable_mtp {
            return Ok(());
        }
        let offset = geometry::final_row_offset(
            rows,
            self.buffers.max_batch_tokens(),
            self.config.hidden_size,
        )
        .map_err(anyhow::Error::msg)?;
        let proposer = self
            .proposer
            .as_ref()
            .context("native MTP proposer missing")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("native MTP request state missing")?;
        proposer.retain_native_target_hidden(
            self.buffers.norm_output().offset(offset),
            seq.seq_len,
            state.as_mut(),
            self.gpu.as_ref(),
            self.gpu.default_stream(),
        )
    }

    pub(in crate::model) fn ensure_native_mtp_context(
        &self,
        proposer: &dyn DraftProposer,
        seq: &mut SequenceState,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        // A proposal can append up to eight rows before the next target pass.
        // Release cached prefixes before they can compete with active requests.
        self.evict_native_mtp_prefix_for_rows(16)?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("native MTP request state missing")?;
        let private = proposer.prepare_drafter_catchup(state.as_mut())?;
        let (end, _) = proposer
            .native_target_hidden(state.as_ref())?
            .context("native MTP target boundary missing")?;
        geometry::ready_context(seq.seq_len, private, end).map_err(anyhow::Error::msg)
    }

    pub(in crate::model) fn save_native_mtp_prompt_hidden_dispatch(
        &self,
        seq: &SequenceState,
    ) -> Result<()> {
        ensure!(
            self.native_glm_mtp,
            "native MTP bootstrap requires native capability"
        );
        let proposer = self
            .proposer
            .as_ref()
            .context("native MTP proposer missing")?;
        let state = seq
            .proposer_state
            .as_ref()
            .context("native MTP request state missing")?;
        let (end, hidden) = proposer
            .native_target_hidden(state.as_ref())?
            .context("native MTP target boundary missing")?;
        ensure!(
            end == seq.seq_len && end > 0,
            "native MTP target boundary is stale"
        );
        self.gpu.copy_d2d_async(
            hidden,
            self.mtp_hidden_save,
            self.config.hidden_size * 2,
            self.gpu.default_stream(),
        )?;
        self.last_mtp_hidden_idx.store(0, Ordering::Relaxed);
        Ok(())
    }
    pub(super) fn repair_mtp_after_verify_dispatch(
        &self,
        accepted_draft: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        self.repair_mtp_after_verify_row_dispatch(accepted_draft, position, 0, seq, _stream)
    }

    pub(super) fn repair_mtp_after_verify_row_dispatch(
        &self,
        accepted_draft: u32,
        position: usize,
        target_row: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let Some(state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        let ctx = ForwardContext {
            ssm_batch: None,
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };
        // The canonical verify output remains live until the subsequent
        // propose overwrites scratch. Native GLM consumes post-norm BF16 rows.
        let row_bytes = self
            .config
            .hidden_size
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("MTP repair hidden row size overflows"))?;
        let row_offset = target_row
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("MTP repair target row offset overflows"))?;
        proposer.repair_after_verify(
            accepted_draft,
            position,
            self.buffers.norm_output().offset(row_offset),
            state.as_mut(),
            &ctx,
            self.gpu.default_stream(),
        )
    }
}
