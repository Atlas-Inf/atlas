// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter head's side of the rank split (`rank_split_exec`): enabling
//! it, and the shares of a projection each rank runs.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

impl BlockDiffusionDraftHead {
    /// Turn the rank split over `ranks` ranks on for this head (any rank),
    /// for batched proposes too with `batch` (`ATLAS_GLM_DRAFT_TP_BATCH`).
    /// Refuses a head the split cannot serve: it must not run unsplit on one
    /// rank only.
    #[allow(clippy::too_many_arguments)]
    pub fn enable_rank_split(
        &mut self,
        parts: Parts,
        batch: bool,
        ctx: bool,
        topk: bool,
        ranks: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        // ATLAS_GLM_DRAFT_TP_TOPK feeds the DFlash2 selector's walk.
        ensure!(
            !topk || (parts.head && self.candidate_selector.is_some()),
            "ATLAS_GLM_DRAFT_TP_TOPK needs the head split and a DFlash2 candidate selector"
        );
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel(self.gamma as u32);
        let layers_ok = self.twins.nvfp4_tc
            && self.layers.iter().all(|l| {
                l.gate_proj_nvfp4.is_some()
                    && l.up_proj_nvfp4.is_some()
                    && l.down_proj_nvfp4.is_some()
            });
        let attn_ok = self.twins.nvfp4_tc
            && self.layers.iter().all(|l| {
                l.q_proj_nvfp4.is_some()
                    && l.k_proj_nvfp4.is_some()
                    && l.v_proj_nvfp4.is_some()
                    && l.o_proj_nvfp4.is_some()
            });
        ensure!(
            tier.0 != 0
                && (!parts.mlp || layers_ok)
                && (!parts.attn || attn_ok)
                && (!parts.head || self.twins.lm_head_q4.is_some()),
            "ATLAS_GLM_DRAFT_TP needs the drafter's NVFP4 twins on the tensor-core GEMV \
             (ATLAS_W4A16_TC=1, ATLAS_DFLASH_NVFP4_TC=1, ATLAS_DFLASH_NVFP4_HEAD=1)"
        );
        ensure!(
            self.startup.option_b_enabled
                && self.lane_count() == 1
                && !self.startup.debug_dump
                && !self.startup.graph_ineligible_diags,
            "ATLAS_GLM_DRAFT_TP needs the paged single-lane propose without diagnostics"
        );
        // ATLAS_GLM_DRAFT_TP_CTX: the context projections' NVFP4 twins.
        ensure!(
            !ctx || self.twins.ctx_q4.is_some(),
            "ATLAS_GLM_DRAFT_TP_CTX needs the context NVFP4 twins (ATLAS_DFLASH_CTX_NVFP4=1)"
        );
        let (ctx_in, ctx_kv) = if ctx {
            (
                self.target_layer_ids.len() * self.target_hidden_size,
                self.num_layers * 2 * self.num_kv_heads * self.head_dim,
            )
        } else {
            (0, 0)
        };
        let geometry = Geometry {
            gamma: self.gamma,
            hidden: self.hidden_size,
            inter: self.intermediate_size,
            vocab: self.vocab_size,
            layers: self.layers.len(),
            parts,
            ctx_in,
            ctx_kv,
            ctx: CtxRows::default(),
            topk,
            ranks,
            q_dim: self.num_q_heads * self.head_dim,
            kv_dim: self.num_kv_heads * self.head_dim,
        };
        // A batched propose runs B×gamma rows through the batch buffers.
        let capacity = if batch {
            self.gamma * self.batch_capacity
        } else {
            self.gamma
        };
        ensure!(
            !ctx || capacity <= ROWS_MAX,
            "ATLAS_GLM_DRAFT_TP_CTX announces at most {ROWS_MAX} batched rows, not {capacity}"
        );
        let mut split = RankSplit::new(geometry, capacity, gpu)?;
        let conf = self.startup.diagnostics.draft_conf;
        split.topk = topk
            .then(|| TopkSelect::new(self.gamma, capacity, conf, ranks, gpu))
            .transpose()?;
        tracing::info!(
            "DFlash rank split over {ranks} ranks: {parts:?}, {} swaps per propose, largest {} bytes; up to {capacity} rows (largest swap {} bytes); context appends {}; selector swaps {}",
            geometry.swaps().len(),
            split.max_bytes(),
            split.max_bytes_at(capacity),
            if ctx { "split" } else { "on the head" },
            if topk { "candidate lists" } else { "logits" }
        );
        self.rank_split = Some(split);
        Ok(())
    }

    /// The split and communicator a single-sequence propose on the head rank
    /// swaps over, if it splits. The one predicate the model announces by and
    /// the forward splits by.
    pub(crate) fn rank_split_with<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        grammar: bool,
    ) -> Option<(&'a RankSplit, &'a dyn CommBackend)> {
        let (split, comm) = (self.rank_split.as_ref()?, comm?);
        (!grammar && comm.rank() == 0 && split.carries(comm, split.max_bytes()))
            .then_some((split, comm))
    }

    /// `out[m, rows] = input[m, k] · W[first.., ..]ᵀ` for this rank's share
    /// of an `[n, k]` twin among `ranks`, `m` the propose's rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn split_gemv(
        &self,
        gpu: &dyn GpuBackend,
        w: Option<&QuantizedWeight>,
        input: DevicePtr,
        out: DevicePtr,
        [m, n, k]: [usize; 3],
        (rank, ranks): (usize, usize),
        stream: u64,
    ) -> Result<()> {
        let w = w.context("rank-split propose: NVFP4 twin missing")?;
        let (first, rows) = share(n, rank, ranks);
        ensure!(
            self.nvfp4_tc_rows(
                gpu,
                &row_range(w, first, k),
                input,
                out,
                m as u32,
                rows as u32,
                k as u32,
                stream,
            )?,
            "rank-split propose: tensor-core GEMV tier missing"
        );
        Ok(())
    }

    /// [`Self::split_gemv`] of each `(weight, out, n)` part over one
    /// `[m, k]` input: one `w4a16_gemv_tc8_parts` launch on the 1..=8-row
    /// tier (each part the words its own launch writes), else one launch a
    /// part.
    pub(super) fn split_gemv_parts(
        &self,
        gpu: &dyn GpuBackend,
        parts: &[(Option<&QuantizedWeight>, DevicePtr, usize)],
        input: DevicePtr,
        [m, k]: [usize; 2],
        (rank, ranks): (usize, usize),
        stream: u64,
    ) -> Result<()> {
        let tc8 = crate::layers::w4a16_gemv_tiers::tc_kernel(m as u32);
        if (1..=8).contains(&m)
            && tc8.0 != 0
            && crate::layers::w4a16_gemv_tiers::tc_rows(tc8) == Some(8)
        {
            let shares: Vec<(QuantizedWeight, DevicePtr, u32)> = parts
                .iter()
                .map(|&(w, out, n)| {
                    let w = w.context("rank-split propose: NVFP4 twin missing")?;
                    let (first, rows) = share(n, rank, ranks);
                    Ok((row_range(w, first, k), out, rows as u32))
                })
                .collect::<Result<_>>()?;
            let refs: Vec<(&QuantizedWeight, DevicePtr, u32)> =
                shares.iter().map(|(w, out, n)| (w, *out, *n)).collect();
            if ops::w4a16_gemv_tc8_parts(gpu, input, &refs, m as u32, k as u32, stream)? {
                return Ok(());
            }
        }
        for &(w, out, n) in parts {
            self.split_gemv(gpu, w, input, out, [m, n, k], (rank, ranks), stream)?;
        }
        Ok(())
    }

    /// A piece every rank runs, over `rank`'s share.
    pub(crate) fn split_piece(
        &self,
        split: &RankSplit,
        piece: Piece,
        rank: usize,
        gpu: &dyn GpuBackend,
        frame: &Frame,
        stream: u64,
    ) -> Result<()> {
        let g = &split.plan();
        let ranks = (rank, g.ranks);
        match piece {
            Piece::GateUp(l) => {
                let layer = &self.layers[l];
                let gate = split.own(Swap::Activation(l), rank);
                let parts = [
                    (layer.gate_proj_nvfp4.as_ref(), gate, g.inter),
                    (layer.up_proj_nvfp4.as_ref(), split.up, g.inter),
                ];
                self.split_gemv_parts(gpu, &parts, frame.norm, [g.gamma, g.hidden], ranks, stream)?;
                ops::silu_mul(
                    gpu,
                    self.kernels.silu_mul,
                    gate,
                    split.up,
                    gate,
                    (g.gamma * share(g.inter, rank, g.ranks).1) as u32,
                    stream,
                )
            }
            Piece::Down(l) => self.split_gemv(
                gpu,
                self.layers[l].down_proj_nvfp4.as_ref(),
                frame.inter,
                split.own(Swap::Output(l), rank),
                [g.gamma, g.hidden, g.inter],
                ranks,
                stream,
            ),
            Piece::Vocab => self.split_gemv(
                gpu,
                self.twins.lm_head_q4.as_ref(),
                frame.norm,
                split.own(Swap::Logits, rank),
                [g.gamma, g.vocab, g.hidden],
                ranks,
                stream,
            ),
            Piece::Qkv(l) => {
                let layer = &self.layers[l];
                let parts = [
                    (
                        layer.q_proj_nvfp4.as_ref(),
                        split.qkv_part(0).of(rank),
                        g.q_dim,
                    ),
                    (
                        layer.k_proj_nvfp4.as_ref(),
                        split.qkv_part(1).of(rank),
                        g.kv_dim,
                    ),
                    (
                        layer.v_proj_nvfp4.as_ref(),
                        split.qkv_part(2).of(rank),
                        g.kv_dim,
                    ),
                ];
                self.split_gemv_parts(gpu, &parts, frame.norm, [g.gamma, g.hidden], ranks, stream)
            }
            Piece::OProj(l) => self.split_gemv(
                gpu,
                self.layers[l].o_proj_nvfp4.as_ref(),
                frame.attn,
                split.own(Swap::OProj(l), rank),
                [g.gamma, g.hidden, g.q_dim],
                ranks,
                stream,
            ),
            Piece::TopK => split.topk_piece(rank, gpu, stream),
            Piece::CtxFc(_) | Piece::CtxNorm(_) | Piece::CtxKv(_) => {
                self.ctx_piece(split, piece, rank, gpu, frame, stream)
            }
            other => bail!("rank-split propose: {other:?} is not a shared piece"),
        }
    }
}
