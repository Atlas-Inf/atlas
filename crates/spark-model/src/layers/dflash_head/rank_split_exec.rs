// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split at run time (`rank_split`): a head's half buffers, its
//! swaps over the pair exchange, the pieces both ranks run, and the worker's
//! walk.

use anyhow::{Context, Result, bail, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use super::super::BlockDiffusionDraftHead;
use super::ctx::{CTX_MAX_ROWS, CtxRows, ROWS_MAX};
use super::{Frame, Geometry, Parts, Piece, Swap, half};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

#[path = "rank_split_ctx_exec.rs"]
mod ctx_exec;
pub(crate) use ctx_exec::{CtxSeq, CtxTurn};
#[path = "rank_split_worker.rs"]
mod worker;

/// A head's split state: the plan, the half buffers and how far the current
/// propose got.
pub struct RankSplit {
    /// The single-sequence plan (`gamma` rows).
    pub(crate) geometry: Geometry,
    /// Rows the half buffers hold: `gamma`, or the batch capacity times
    /// `gamma` with `ATLAS_GLM_DRAFT_TP_BATCH`.
    pub(crate) capacity: usize,
    /// Rows of the current propose.
    rows: AtomicUsize,
    /// This rank's half of gate (then of the gated activation) and of up.
    pub(super) gate: DevicePtr,
    up: DevicePtr,
    /// The peer's half of the gated activation.
    activation_peer: DevicePtr,
    /// This rank's and the peer's half of the down projection.
    pub(super) down: DevicePtr,
    down_peer: DevicePtr,
    /// This rank's and the peer's half of the logits.
    pub(super) logits: DevicePtr,
    logits_peer: DevicePtr,
    /// Where a one-way swap's unused direction lands and what it sends.
    sink: DevicePtr,
    /// Swaps issued by the current propose.
    issued: AtomicUsize,
    /// `ATLAS_GLM_DRAFT_TP_CTX`: the worker's landing rows of a context
    /// append's input, and this rank's and the peer's half of its `fc` (then
    /// fused K/V) rows. Null without the switch.
    pub(super) ctx_input: DevicePtr,
    pub(super) ctx_own: DevicePtr,
    ctx_peer: DevicePtr,
    /// The current propose's context rows (`CtxRows::pack`), the rows the
    /// head last announced, and the head's next append.
    ctx: AtomicU32,
    pub(super) announced: AtomicU32,
    pub(super) ctx_next: AtomicUsize,
}

impl RankSplit {
    /// Allocate the half buffers for `geometry` at `capacity` rows (at
    /// least `gamma`).
    pub(crate) fn new(geometry: Geometry, capacity: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        geometry.validate()?;
        let full = geometry.with_rows(capacity.max(geometry.gamma));
        let alloc = |swap: Swap, on: bool| -> Result<DevicePtr> {
            if on {
                gpu.alloc(full.bytes(swap))
            } else {
                Ok(DevicePtr::NULL)
            }
        };
        let (mlp, head) = (geometry.parts.mlp, geometry.parts.head);
        // The largest context append's input rows and wider half (none
        // without the switch).
        let (ctx_input, ctx_half) = if geometry.ctx_in > 0 {
            let wider = half(geometry.hidden, 1).1.max(half(geometry.ctx_kv, 1).1);
            (CTX_MAX_ROWS * geometry.ctx_in * 2, CTX_MAX_ROWS * wider * 2)
        } else {
            (0, 0)
        };
        let sink_bytes = full
            .swaps()
            .iter()
            .map(|&s| full.bytes(s))
            .max()
            .context("a split has swaps")?
            .max(ctx_input)
            .max(ctx_half);
        let ctx_alloc = |bytes: usize| -> Result<DevicePtr> {
            if bytes > 0 {
                gpu.alloc(bytes)
            } else {
                Ok(DevicePtr::NULL)
            }
        };
        Ok(Self {
            geometry,
            capacity: full.gamma,
            rows: AtomicUsize::new(geometry.gamma),
            gate: alloc(Swap::Activation(0), mlp)?,
            up: alloc(Swap::Activation(0), mlp)?,
            activation_peer: alloc(Swap::Activation(0), mlp)?,
            down: alloc(Swap::Output(0), mlp)?,
            down_peer: alloc(Swap::Output(0), mlp)?,
            logits: alloc(Swap::Logits, head)?,
            logits_peer: alloc(Swap::Logits, head)?,
            sink: gpu.alloc(sink_bytes)?,
            issued: AtomicUsize::new(0),
            ctx_input: ctx_alloc(ctx_input)?,
            ctx_own: ctx_alloc(ctx_half)?,
            ctx_peer: ctx_alloc(ctx_half)?,
            ctx: AtomicU32::new(0),
            announced: AtomicU32::new(0),
            ctx_next: AtomicUsize::new(0),
        })
    }

    /// The largest swap of a propose of `rows` rows, which the pair
    /// exchange must be able to carry.
    pub(crate) fn max_bytes_at(&self, rows: usize) -> usize {
        let g = self.geometry.with_rows(rows);
        g.swaps().iter().map(|&s| g.bytes(s)).max().unwrap_or(0)
    }

    /// The largest single-sequence swap.
    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes_at(self.geometry.gamma)
    }

    /// A propose of `rows` rows (at most [`Self::capacity`]) with context
    /// appends of `ctx` rows begins: no swap issued yet.
    pub(crate) fn begin(&self, rows: usize, ctx: CtxRows) -> Result<()> {
        ensure!(
            (1..=self.capacity).contains(&rows),
            "rank-split propose of {rows} rows (capacity {})",
            self.capacity
        );
        ensure!(
            ctx == CtxRows::default() || self.geometry.ctx_in > 0,
            "rank-split propose announced context rows {ctx:?} without ATLAS_GLM_DRAFT_TP_CTX"
        );
        self.rows.store(rows, Ordering::Relaxed);
        self.ctx.store(ctx.pack(), Ordering::Relaxed);
        self.ctx_next.store(0, Ordering::Relaxed);
        self.issued.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Head: a propose of `rows` rows begins with the context rows its
    /// announce carried (`BlockDiffusionDraftHead::split_ctx_rows`).
    pub(crate) fn begin_announced(&self, rows: usize) -> Result<()> {
        let ctx = CtxRows::unpack(self.announced.swap(0, Ordering::Relaxed));
        self.begin(rows, ctx)
    }

    /// The plan of the current propose: its rows, its context rows and the
    /// shared shapes.
    pub(crate) fn plan(&self) -> Geometry {
        Geometry {
            ctx: CtxRows::unpack(self.ctx.load(Ordering::Relaxed)),
            ..self.geometry.with_rows(self.rows.load(Ordering::Relaxed))
        }
    }

    /// What `rank` sends and where the peer's payload lands.
    pub(super) fn ends(&self, swap: Swap, rank: usize, frame: &Frame) -> (DevicePtr, DevicePtr) {
        match swap {
            // One way, head to worker. The rows are in `frame.norm` on both.
            Swap::Input(_) | Swap::Hidden if rank == 0 => (frame.norm, self.sink),
            Swap::Input(_) | Swap::Hidden => (self.sink, frame.norm),
            Swap::Activation(_) => (self.gate, self.activation_peer),
            Swap::Output(_) => (self.down, self.down_peer),
            Swap::Logits => (self.logits, self.logits_peer),
            // One way, head to worker: the head's rows to the worker's landing.
            Swap::CtxInput(_) if rank == 0 => (frame.ctx_in, self.sink),
            Swap::CtxInput(_) => (self.sink, frame.ctx_in),
            Swap::CtxHidden(_) | Swap::CtxKv(_) => (self.ctx_own, self.ctx_peer),
        }
    }

    /// Issue `swap` on `stream`, then place the halves it completed where
    /// the unsplit launch would have written the whole rows.
    pub(crate) fn swap(
        &self,
        swap: Swap,
        rank: usize,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        frame: &Frame,
        stream: u64,
    ) -> Result<()> {
        let (send, dst) = self.ends(swap, rank, frame);
        self.exchange(swap, send, dst, comm, stream)?;
        let g = &self.plan();
        let join = |dst, n, halves| self.join(dst, [g.rows(swap), n], halves, rank, gpu, stream);
        match swap {
            Swap::Activation(_) => join(frame.inter, g.inter, (self.gate, self.activation_peer)),
            Swap::Output(_) if rank == 0 => join(frame.acc, g.hidden, (self.down, self.down_peer)),
            Swap::Logits if rank == 0 => {
                join(frame.logits, g.vocab, (self.logits, self.logits_peer))
            }
            Swap::CtxHidden(_) => join(frame.ctx_fc, g.hidden, (self.ctx_own, self.ctx_peer)),
            Swap::CtxKv(_) if rank == 0 => {
                join(frame.ctx_kv, g.ctx_kv, (self.ctx_own, self.ctx_peer))
            }
            _ => Ok(()),
        }
    }

    fn exchange(
        &self,
        swap: Swap,
        send: DevicePtr,
        dst: DevicePtr,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<()> {
        let bytes = self.plan().bytes(swap);
        ensure!(
            comm.exchange_async(send.0, dst.0, bytes, false, stream)?,
            "rank-split propose: pair exchange refused {swap:?} ({bytes} bytes)"
        );
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Write `[m, n]` rows at `dst` from the two ranks' shares `(own,
    /// peer)`, each `[m, its rows]`: two pitched copies.
    fn join(
        &self,
        dst: DevicePtr,
        [m, n]: [usize; 2],
        (own, peer): (DevicePtr, DevicePtr),
        rank: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for (src, r) in [(own, rank), (peer, 1 - rank)] {
            let (first, rows) = half(n, r);
            gpu.copy_d2d_2d_async(
                src,
                rows * 2,
                dst.offset(first * 2),
                n * 2,
                rows * 2,
                m,
                stream,
            )?;
        }
        Ok(())
    }

    /// A propose ends, however it ended: issue every swap it did not reach,
    /// so the peer, which walks them all, is never left waiting in one.
    pub(crate) fn finish(&self, comm: &dyn CommBackend, stream: u64) -> Result<()> {
        let swaps = self.plan().swaps();
        let issued = self.issued.load(Ordering::Relaxed);
        for &swap in swaps.iter().skip(issued) {
            self.exchange(swap, self.sink, self.sink, comm, stream)?;
        }
        if issued < swaps.len() {
            tracing::warn!(
                "rank-split propose stopped after {issued} of {} swaps; the rest were drained",
                swaps.len()
            );
        }
        Ok(())
    }
}

/// Rows `[first, first + rows)` of an NVFP4 `[n, k]` twin: packed bytes and
/// group scales are row-major, the tensor scale is the twin's.
pub(super) fn row_range(w: &QuantizedWeight, first: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: w.weight.offset(first * k / 2),
        weight_scale: w.weight_scale.offset(first * k / 16),
        ..*w
    }
}

impl BlockDiffusionDraftHead {
    /// Turn the rank split on for this head (either rank), for batched
    /// proposes too with `batch` (`ATLAS_GLM_DRAFT_TP_BATCH`). Refuses a head
    /// the split cannot serve: it must not run unsplit on one rank only.
    pub fn enable_rank_split(
        &mut self,
        parts: Parts,
        batch: bool,
        ctx: bool,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let tier = crate::layers::w4a16_gemv_tiers::tc_kernel(self.gamma as u32);
        let layers_ok = self.twins.nvfp4_tc
            && self.layers.iter().all(|l| {
                l.gate_proj_nvfp4.is_some()
                    && l.up_proj_nvfp4.is_some()
                    && l.down_proj_nvfp4.is_some()
            });
        ensure!(
            tier.0 != 0
                && (!parts.mlp || layers_ok)
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
        let split = RankSplit::new(geometry, capacity, gpu)?;
        tracing::info!(
            "DFlash rank split: {parts:?}, {} swaps per propose, largest {} bytes; up to {capacity} rows (largest swap {} bytes); context appends {}",
            geometry.swaps().len(),
            split.max_bytes(),
            split.max_bytes_at(capacity),
            if ctx { "split" } else { "on the head" }
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
        (!grammar
            && comm.rank() == 0
            && comm.world_size() == 2
            && comm.supports_exchange_async(split.max_bytes()))
        .then_some((split, comm))
    }

    /// `out[m, rows] = input[m, k] · W[first.., ..]ᵀ` for this rank's half of
    /// an `[n, k]` twin, `m` the propose's rows.
    #[allow(clippy::too_many_arguments)]
    fn split_gemv(
        &self,
        gpu: &dyn GpuBackend,
        w: Option<&QuantizedWeight>,
        input: DevicePtr,
        out: DevicePtr,
        [m, n, k]: [usize; 3],
        rank: usize,
        stream: u64,
    ) -> Result<()> {
        let w = w.context("rank-split propose: NVFP4 twin missing")?;
        let (first, rows) = half(n, rank);
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

    /// A piece both ranks run, over `rank`'s half.
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
        match piece {
            Piece::GateUp(l) => {
                let layer = &self.layers[l];
                for (w, out) in [
                    (&layer.gate_proj_nvfp4, split.gate),
                    (&layer.up_proj_nvfp4, split.up),
                ] {
                    self.split_gemv(
                        gpu,
                        w.as_ref(),
                        frame.norm,
                        out,
                        [g.gamma, g.inter, g.hidden],
                        rank,
                        stream,
                    )?;
                }
                ops::silu_mul(
                    gpu,
                    self.kernels.silu_mul,
                    split.gate,
                    split.up,
                    split.gate,
                    (g.gamma * half(g.inter, rank).1) as u32,
                    stream,
                )
            }
            Piece::Down(l) => self.split_gemv(
                gpu,
                self.layers[l].down_proj_nvfp4.as_ref(),
                frame.inter,
                split.down,
                [g.gamma, g.hidden, g.inter],
                rank,
                stream,
            ),
            Piece::Vocab => self.split_gemv(
                gpu,
                self.twins.lm_head_q4.as_ref(),
                frame.norm,
                split.logits,
                [g.gamma, g.vocab, g.hidden],
                rank,
                stream,
            ),
            Piece::CtxFc(_) | Piece::CtxNorm(_) | Piece::CtxKv(_) => {
                self.ctx_piece(split, piece, rank, gpu, frame, stream)
            }
            other => bail!("rank-split propose: {other:?} is not a shared piece"),
        }
    }
}
