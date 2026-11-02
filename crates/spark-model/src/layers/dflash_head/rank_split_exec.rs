// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split at run time (`rank_split`): a head's share buffers, its
//! swaps over the pair exchange or the chain's gather and broadcast, the
//! pieces every rank runs, and the workers' walk.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::super::BlockDiffusionDraftHead;
use super::ctx::{CTX_MAX_ROWS, CtxRows, ROWS_MAX};
use super::{Frame, Geometry, Parts, Piece, Swap, share, widest};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

#[path = "rank_split_ctx_exec.rs"]
mod ctx_exec;
pub(crate) use ctx_exec::{CtxSeq, CtxTurn};
#[path = "rank_split_topk.rs"]
mod topk;
pub(crate) use topk::TopkSelect;
#[path = "rank_split_worker.rs"]
mod worker;

#[path = "rank_split_head.rs"]
mod head;

/// Every rank's share of one swap's rows, rank by rank `stride` bytes apart:
/// a rank computes into its own, and the swap fills the others'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shares {
    base: DevicePtr,
    stride: usize,
}

impl Shares {
    const NONE: Self = Self {
        base: DevicePtr::NULL,
        stride: 0,
    };

    /// Rank `rank`'s share.
    pub(crate) fn of(self, rank: usize) -> DevicePtr {
        if self.base.is_null() {
            DevicePtr::NULL
        } else {
            self.base.offset(rank * self.stride)
        }
    }
}

/// A head's split state: the plan, the share buffers and how far the current
/// propose got.
pub struct RankSplit {
    /// The single-sequence plan (`gamma` rows).
    pub(crate) geometry: Geometry,
    /// Rows the share buffers hold: `gamma`, or the batch capacity times
    /// `gamma` with `ATLAS_GLM_DRAFT_TP_BATCH`.
    pub(crate) capacity: usize,
    /// Rows of the current propose.
    rows: AtomicUsize,
    /// Every rank's share of gate (then of the gated activation), and this
    /// rank's share of up.
    activation: Shares,
    up: DevicePtr,
    /// Every rank's share of the down projection and of the logits.
    down: Shares,
    logits: Shares,
    /// The `attn` part: every rank's q, k and v shares, one part a rank
    /// (q, then k, then v, each `[rows, its rows]`), and its o_proj share.
    qkv: Shares,
    oproj: Shares,
    /// Where a one-way swap's unused direction lands and what it sends, and
    /// a drained swap's every share.
    sink: DevicePtr,
    /// Swaps issued by the current propose.
    issued: AtomicUsize,
    /// `ATLAS_GLM_DRAFT_TP_CTX`: a worker's landing rows of a context
    /// append's input, and every rank's share of its `fc` (then fused K/V)
    /// rows. Null without the switch.
    pub(super) ctx_input: DevicePtr,
    ctx_shares: Shares,
    /// The current propose's context rows (`CtxRows::pack`), the rows the
    /// head last announced, and the head's next append.
    ctx: AtomicU64,
    pub(super) announced: AtomicU64,
    pub(super) ctx_next: AtomicUsize,
    /// `ATLAS_GLM_DRAFT_TP_TOPK`: the candidate swap's lists and kernels.
    pub(crate) topk: Option<TopkSelect>,
    /// `rank_split_join_bf16` (0 where the target lacks it: the joins then
    /// take one pitched copy a rank).
    join_kernel: KernelHandle,
}

impl RankSplit {
    /// Allocate the share buffers for `geometry` at `capacity` rows (at
    /// least `gamma`).
    pub(crate) fn new(geometry: Geometry, capacity: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        geometry.validate()?;
        let ranks = geometry.ranks;
        let full = geometry.with_rows(capacity.max(geometry.gamma));
        let alloc = |bytes: usize| -> Result<DevicePtr> {
            if bytes > 0 {
                gpu.alloc(bytes)
            } else {
                Ok(DevicePtr::NULL)
            }
        };
        let shares = |stride: usize, on: bool| -> Result<Shares> {
            Ok(match on {
                true => Shares {
                    base: alloc(ranks * stride)?,
                    stride,
                },
                false => Shares::NONE,
            })
        };
        let (mlp, head, attn) = (geometry.parts.mlp, geometry.parts.head, geometry.parts.attn);
        // The largest context append's input rows and widest share (none
        // without the switch).
        let (ctx_input, ctx_share) = if geometry.ctx_in > 0 {
            let wider = widest(geometry.hidden, ranks).max(widest(geometry.ctx_kv, ranks));
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
            .max(ctx_share);
        let activation = full.bytes(Swap::Activation(0));
        Ok(Self {
            geometry,
            capacity: full.gamma,
            rows: AtomicUsize::new(geometry.gamma),
            activation: shares(activation, mlp)?,
            up: alloc(if mlp { activation } else { 0 })?,
            down: shares(full.bytes(Swap::Output(0)), mlp)?,
            logits: shares(full.bytes(Swap::Logits), head)?,
            qkv: shares(full.bytes(Swap::Qkv(0)), attn)?,
            oproj: shares(full.bytes(Swap::OProj(0)), attn)?,
            sink: gpu.alloc(ranks * sink_bytes)?,
            issued: AtomicUsize::new(0),
            ctx_input: alloc(ctx_input)?,
            ctx_shares: shares(ctx_share, ctx_share > 0)?,
            ctx: AtomicU64::new(0),
            announced: AtomicU64::new(0),
            ctx_next: AtomicUsize::new(0),
            topk: None,
            join_kernel: ops::rank_join_kernel(gpu),
        })
    }

    /// Join with the per-rank copies (a mock GPU runs copies, not kernels).
    #[cfg(test)]
    pub(crate) fn with_copy_joins(mut self) -> Self {
        self.join_kernel = KernelHandle(0);
        self
    }

    /// Whether `comm` is this split's ranks and carries swaps of `bytes`:
    /// the pair's exchange, or the chain's gather (and broadcast).
    pub(crate) fn carries(&self, comm: &dyn CommBackend, bytes: usize) -> bool {
        comm.world_size() == self.geometry.ranks
            && if self.geometry.ranks == 2 {
                comm.supports_exchange_async(bytes)
            } else {
                comm.supports_gather_async(bytes)
            }
    }

    /// The largest swap of a propose of `rows` rows, which the ranks'
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

    /// Every rank's shares of `swap`'s rows ([`Shares::NONE`] for a swap
    /// of the head's rows or of candidate lists).
    fn shares(&self, swap: Swap) -> Shares {
        match swap {
            Swap::Activation(_) => self.activation,
            Swap::Output(_) => self.down,
            Swap::Logits => self.logits,
            Swap::CtxHidden(_) | Swap::CtxKv(_) => self.ctx_shares,
            Swap::Qkv(_) => self.qkv,
            Swap::OProj(_) => self.oproj,
            Swap::Candidates => self.topk.as_ref().map_or(Shares::NONE, |t| t.lists),
            _ => Shares::NONE,
        }
    }

    /// Rank `rank`'s share of `swap`'s rows, which its piece computes into.
    pub(crate) fn own(&self, swap: Swap, rank: usize) -> DevicePtr {
        self.shares(swap).of(rank)
    }

    /// What `rank` sends and where the pair's peer payload lands (on the
    /// chain only the head's rows of a one-way swap are its ends: a share
    /// swap gathers the [`Self::shares`] in place).
    pub(super) fn ends(&self, swap: Swap, rank: usize, frame: &Frame) -> (DevicePtr, DevicePtr) {
        let peer = 1 - rank.min(1);
        match swap {
            // One way, head to worker. The rows are in `frame.norm` on both.
            Swap::Input(_) | Swap::Hidden if rank == 0 => (frame.norm, self.sink),
            Swap::Input(_) | Swap::Hidden => (self.sink, frame.norm),
            Swap::Activation(_) | Swap::Output(_) | Swap::Logits => {
                let s = self.shares(swap);
                (s.of(rank), s.of(peer))
            }
            // One way, worker to head.
            Swap::Candidates => match &self.topk {
                Some(t) if rank == 0 => (self.sink, t.lists.of(1)),
                Some(t) => (t.lists.of(rank), self.sink),
                None => (self.sink, self.sink),
            },
            // One way, head to worker: the head's rows to the worker's landing.
            Swap::CtxInput(_) if rank == 0 => (frame.ctx_in, self.sink),
            Swap::CtxInput(_) => (self.sink, frame.ctx_in),
            Swap::CtxHidden(_) | Swap::CtxKv(_) | Swap::Qkv(_) | Swap::OProj(_) => {
                let s = self.shares(swap);
                (s.of(rank), s.of(peer))
            }
            // One way, head to worker: the normed rows the projections read,
            // then the attention rows o_proj reads.
            Swap::AttnIn(_) if rank == 0 => (frame.norm, self.sink),
            Swap::AttnIn(_) => (self.sink, frame.norm),
            Swap::AttnOut(_) if rank == 0 => (frame.attn, self.sink),
            Swap::AttnOut(_) => (self.sink, frame.attn),
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
        let shares = self.shares(swap);
        self.collective(swap, rank, (send, dst), shares, comm, stream)?;
        let g = &self.plan();
        let join = |dst, n| self.join(dst, [g.rows(swap), n], shares, gpu, stream);
        match swap {
            Swap::Activation(_) => join(frame.inter, g.inter),
            Swap::Output(_) if rank == 0 => join(frame.acc, g.hidden),
            Swap::Logits if rank == 0 => join(frame.logits, g.vocab),
            Swap::CtxHidden(_) => join(frame.ctx_fc, g.hidden),
            Swap::CtxKv(_) if rank == 0 => join(frame.ctx_kv, g.ctx_kv),
            Swap::Qkv(_) if rank == 0 => {
                for (part, dst, n) in [
                    (0, frame.q, g.q_dim),
                    (1, frame.k, g.kv_dim),
                    (2, frame.v, g.kv_dim),
                ] {
                    let shares = self.qkv_part(part);
                    self.join(dst, [g.gamma, n], shares, gpu, stream)?;
                }
                Ok(())
            }
            Swap::OProj(_) if rank == 0 => join(frame.attn_proj, g.hidden),
            _ => Ok(()),
        }
    }

    /// Part `part` (0 q, 1 k, 2 v) of every rank's q/k/v shares: within a
    /// rank's part, q's `[rows, widest q share]` rows, then k's, then v's,
    /// at the current propose's rows.
    fn qkv_part(&self, part: usize) -> Shares {
        let g = self.plan();
        let (wq, wk) = (widest(g.q_dim, g.ranks), widest(g.kv_dim, g.ranks));
        let at = [0, wq, wq + wk][part] * g.gamma * 2;
        Shares {
            base: self.qkv.base.offset(at),
            stride: self.qkv.stride,
        }
    }

    /// Issue `swap`'s collective on `stream`: the pair's exchange of `send`
    /// into `dst`; on the chain the head's broadcast of the rows it sends
    /// (into `dst` elsewhere), or the in-place gather of every rank's
    /// `shares`.
    fn collective(
        &self,
        swap: Swap,
        rank: usize,
        (send, dst): (DevicePtr, DevicePtr),
        shares: Shares,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<()> {
        let bytes = self.plan().bytes(swap);
        let done = if self.geometry.ranks == 2 {
            comm.exchange_async(send.0, dst.0, bytes, false, stream)?
        } else if Geometry::from_head(swap) {
            let rows = if rank == 0 { send } else { dst };
            comm.broadcast_async(rows.0, bytes, 0, stream)?
        } else {
            comm.gather_async(shares.base.0, shares.stride as i64, bytes, stream)?
        };
        ensure!(
            done,
            "rank-split propose: the ranks' exchange refused {swap:?} ({bytes} bytes)"
        );
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Issue `swap` with nothing to say: the sink sent and landed, so a peer
    /// walking it is never left waiting.
    fn drain(&self, swap: Swap, rank: usize, comm: &dyn CommBackend, stream: u64) -> Result<()> {
        let shares = Shares {
            base: self.sink,
            stride: self.plan().bytes(swap),
        };
        self.collective(swap, rank, (self.sink, self.sink), shares, comm, stream)
    }

    /// Write `[m, n]` rows at `dst` from every rank's share in `shares`,
    /// each `[m, its rows]`: one `rank_split_join_bf16` launch, else one
    /// pitched copy a rank.
    fn join(
        &self,
        dst: DevicePtr,
        [m, n]: [usize; 2],
        shares: Shares,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let ranks = self.geometry.ranks;
        if self.join_kernel.0 != 0 {
            return ops::rank_join_bf16(
                gpu,
                self.join_kernel,
                dst,
                shares.of(0),
                shares.stride,
                [m, n],
                ranks,
                stream,
            );
        }
        for r in 0..ranks {
            let (first, rows) = share(n, r, ranks);
            gpu.copy_d2d_2d_async(
                shares.of(r),
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
    /// so the other ranks, which walk them all, are never left waiting in
    /// one.
    pub(crate) fn finish(&self, comm: &dyn CommBackend, stream: u64) -> Result<()> {
        let swaps = self.plan().swaps();
        let issued = self.issued.load(Ordering::Relaxed);
        for &swap in swaps.iter().skip(issued) {
            self.drain(swap, comm.rank(), comm, stream)?;
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
