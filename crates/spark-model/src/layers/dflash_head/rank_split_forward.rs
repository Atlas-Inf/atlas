// SPDX-License-Identifier: AGPL-3.0-only

//! Head side of the rank-split propose (`rank_split`): the layer loop and
//! tail of `forward_block` walked as pieces and swaps, with each run of
//! pieces between two swaps captured as one CUDA graph.

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::GraphHandle;

use super::forward_block_layer_paged::PagedLayerArgs;
use super::rank_split::{Frame, Graphs, Piece, RankSplit, SplitOps, Step, Swap, run_count, walk};
use super::{BlockDiffusionDraftHead, DflashGraphIdentity, DflashScratch};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// One propose's view of the head: what each piece and swap needs.
pub(super) struct HeadWalk<'a> {
    pub head: &'a BlockDiffusionDraftHead,
    pub split: &'a RankSplit,
    pub comm: &'a dyn CommBackend,
    pub ctx: &'a ForwardContext<'a>,
    pub scratch: &'a DflashScratch,
    pub stream: u64,
    /// The paged per-layer arguments of this propose.
    pub args: &'a dyn Fn(usize) -> PagedLayerArgs,
    /// The unsplit tail (final norm, head, selection).
    pub tail: &'a dyn Fn() -> Result<()>,
    /// Selection over `scratch.logits`, the last part of the tail.
    pub select: &'a dyn Fn() -> Result<()>,
}

impl SplitOps for HeadWalk<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        let (head, ctx, scratch) = (self.head, self.ctx, self.scratch);
        let layer = |l: usize| (&head.layers[l], (self.args)(l));
        match piece {
            Piece::Attention(l) => {
                let (layer, args) = layer(l);
                let (k, v) = head.forward_block_layer_pre_attn(layer, &args, ctx, scratch)?;
                head.forward_block_layer_attention(&args, ctx, k, v, scratch)
            }
            Piece::Post(l) => {
                let (layer, args) = layer(l);
                head.forward_block_layer_post_attn(layer, &args, ctx, scratch)
            }
            Piece::Project(l) => {
                let (layer, args) = layer(l);
                head.post_attn_project(layer, &args, ctx, scratch)
            }
            Piece::Residual(l) => {
                let (layer, args) = layer(l);
                head.post_attn_residual(layer, &args, ctx, scratch)
            }
            Piece::Tail => (self.tail)(),
            // Option B runs gamma rows only: the noise rows start both buffers.
            Piece::Norm => ops::rms_norm(
                ctx.gpu,
                head.kernels.rms_norm,
                scratch.stream_buf,
                &head.norm,
                scratch.norm_buf,
                head.gamma as u32,
                head.hidden_size as u32,
                head.rms_norm_eps,
                self.stream,
            ),
            Piece::Select => (self.select)(),
            Piece::CtxFc(_) | Piece::CtxNorm(_) | Piece::CtxKv(_) => {
                anyhow::bail!("rank-split propose: {piece:?} walks with its context append")
            }
            Piece::GateUp(_) | Piece::Down(_) | Piece::Vocab => {
                let frame = Frame::serial(scratch);
                head.split_piece(self.split, piece, 0, ctx.gpu, &frame, self.stream)
            }
        }
    }

    fn swap(&self, swap: Swap) -> Result<()> {
        self.split.swap(
            swap,
            0,
            self.ctx.gpu,
            self.comm,
            &Frame::serial(self.scratch),
            self.stream,
        )
    }
}

impl BlockDiffusionDraftHead {
    /// Walk `steps` for one propose. With a `graph_key` (the propose is
    /// graph-eligible) the runs replay their captured graphs, after the same
    /// eager warm-up and single capture pass as the unsplit path; without
    /// one every piece launches eagerly. The swaps are issued either way.
    pub(super) fn walk_split(
        &self,
        steps: &[Step],
        walker: &HeadWalk<'_>,
        graph_key: Option<DflashGraphIdentity>,
    ) -> Result<()> {
        let (gpu, stream) = (walker.ctx.gpu, walker.stream);
        let Some(key) = graph_key else {
            return walk(steps, Graphs::Eager, gpu, stream, walker);
        };
        let runs = run_count(steps);
        let mut graphs = self.propose_graphs.lock();
        if let Some(captured) = graphs.get(&key).filter(|g| g.len() == runs) {
            return walk(steps, Graphs::Replay(captured), gpu, stream, walker);
        }
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        if self.propose_warmup_count.load(relaxed) < self.startup.diagnostics.propose_warmup_n {
            self.propose_warmup_count.fetch_add(1, relaxed);
            return walk(steps, Graphs::Eager, gpu, stream, walker);
        }
        tracing::info!("DFlash rank-split capture: starting (key={key:?}, {runs} graphs)");
        let mut captured: Vec<GraphHandle> = Vec::with_capacity(runs);
        if let Err(e) = walk(steps, Graphs::Capture(&mut captured), gpu, stream, walker) {
            for graph in captured.into_iter().filter(|g| g.0 != 0) {
                if let Err(destroy) = gpu.destroy_graph(graph) {
                    tracing::warn!("DFlash rank-split capture: destroy: {destroy:#}");
                }
            }
            return Err(e);
        }
        tracing::info!(
            "DFlash rank-split capture: complete key={key:?} ({}/{runs} graphs captured)",
            captured.iter().filter(|g| g.0 != 0).count()
        );
        // A key first captured unsplit (a propose the split did not take)
        // holds another layout: drop its graphs with it.
        for graph in graphs.insert(key, captured).into_iter().flatten() {
            if graph.0 != 0 {
                gpu.destroy_graph(graph)?;
            }
        }
        Ok(())
    }
}
