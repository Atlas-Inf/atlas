// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split propose (`ATLAS_GLM_DRAFT_TP`, default off).
//!
//! The drafter runs on the head rank while the workers wait for the next
//! verify. A propose is the read of the drafter's NVFP4 weights, so with this
//! switch every rank reads its share of the largest ones: each of the
//! `ranks` ranks (the TP2 pair or the TP4 chain) computes a contiguous share
//! of the OUTPUT rows of a layer's gate/up and down projections and of the
//! vocabulary head, with the unchanged tensor-core GEMV over a row range of
//! the same twin, and the shares are swapped over the pair exchange or
//! gathered over the chain (the head's rows go out the same way, or by the
//! chain's broadcast). A GEMV output row depends only on its weight row and
//! the input, so every value is the one the unsplit launch writes and the
//! drafts are bit-identical.
//!
//! Every rank walks `Geometry::steps` of its own rank. The swaps are the
//! same list in the same order with the same sizes on every rank
//! (`Geometry::swaps`); only the local pieces between them differ. Pieces
//! are stream work with no host sync, so the head captures each run of pieces
//! between two swaps as one CUDA graph and the worker enqueues its whole walk
//! at once.
//!
//! `ATLAS_GLM_DRAFT_TP=1` splits the MLP and the head; `mlp`, `head` or
//! `attn` (comma separated) selects parts. `attn` (with `mlp`) also shares
//! each layer's q/k/v and o projections the same way: the head swaps the
//! normed rows out, every rank projects its share of the output rows, the
//! head joins them and runs the rest of attention alone, swaps the attention
//! rows out for o_proj, and joins that (four more swaps a layer for most of
//! the head's remaining weight reads). Both ranks must run the same value
//! (`model::startup_parity`).
//!
//! `ATLAS_GLM_DRAFT_TP_BATCH=1` (with the switch above, default off) splits
//! the batched B×gamma propose of an owner-batched step the same way
//! (`rank_split_batch`): the plan's rows are B×gamma, the halves are the
//! same tensor-core GEMV over a row range at the same row count the unsplit
//! batched launch uses, and the swaps carry the batch buffers. Without it the
//! worker idles through every batched propose (about 10 ms each in a C4
//! prose nsys profile, 2026-10-03).
//!
//! `ATLAS_GLM_DRAFT_TP_CTX=1` (with the switch above, default off) splits a
//! split propose's context append too (`rank_split_ctx`): the `fc` and fused
//! K/V projections of the context rows the propose appends, the largest
//! drafter reads the head still ran alone (about 0.5 ms a propose in a C1
//! prose nsys profile, 2026-10-03). The head announces those rows with the
//! propose and walks them as the first swaps of the plan.
//!
//! `ATLAS_GLM_DRAFT_TP_TOPK=1` (with the head split, default off) swaps
//! each rank's candidate lists instead of its logits (`Swap::Candidates`):
//! each rank reduces its share of the vocabulary to every row's exact top-16
//! on the device (`dflash2_selector_topk`, one launch a sequence), the
//! workers send their lists (1 KiB a rank at gamma 8, where the logits half
//! is 1.2 MiB and crosses the pair through the host), and the head merges
//! every rank's lists of each row and walks each sequence's selector chain
//! (`dflash2_selector_walk`, `_ranks` on the chain). Each list is exact in
//! the selector's total order, so the drafts are the unsplit selector's. A
//! batched propose (`ATLAS_GLM_DRAFT_TP_BATCH`) swaps its B sequences' lists
//! the same way, and its selector walks them in place of the joined logits.
//!
//! Prior art: splitting the drafter across both ranks follows MiaAI-Lab's
//! `DFLASH_DRAFT_TP=2` (<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks>
//! `.env.example`, `start.sh`; vLLM `draft_tensor_parallel_size`). Idea only,
//! no code. The output-row share on 16-row CTA boundaries, swapped over the
//! RDMA pair with bit-identical drafts, is ours (docs/glm-prior-art.md).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

#[path = "rank_split_exec.rs"]
mod exec;
pub use exec::RankSplit;
pub(crate) use exec::{CtxSeq, CtxTurn, TopkSelect};

#[path = "rank_split_ctx.rs"]
mod ctx;
pub(crate) use ctx::CtxRows;
use ctx::ctx_steps;
pub use ctx::{announce_extra, announce_word, ctx_requested, extra_follows};

#[path = "rank_split_parts.rs"]
mod parts;
use parts::CANDIDATE_ROW_BYTES;
pub(crate) use parts::parity_word;
pub use parts::{Parts, batch_requested, requested, topk_requested};
#[cfg(test)]
pub(crate) use parts::{parse_batch, parse_topk};

/// Where a propose's whole rows live on a rank: the MLP input and final-norm
/// rows (`norm`), the gated activation (`inter`), the down projection the
/// head's residual adds (`acc`) and the logits (`logits`). A single-sequence
/// propose uses the block scratch ([`Frame::serial`]), a batched one the
/// head's B×gamma buffers (`BlockDiffusionDraftHead::batch_frame`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub norm: DevicePtr,
    pub inter: DevicePtr,
    pub acc: DevicePtr,
    pub logits: DevicePtr,
    /// A context append's input rows (the head's accumulator rows, the
    /// worker's landing buffer), its `fc` output (normed in place) and its
    /// fused K/V output (`ATLAS_GLM_DRAFT_TP_CTX`).
    pub ctx_in: DevicePtr,
    pub ctx_fc: DevicePtr,
    pub ctx_kv: DevicePtr,
    /// The `attn` part: where the head's q/k/v projections land, the
    /// attention rows o_proj reads, and o_proj's output.
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub attn: DevicePtr,
    pub attn_proj: DevicePtr,
}

impl Frame {
    pub(crate) fn serial(scratch: &super::DflashScratch) -> Self {
        Self {
            norm: scratch.norm_buf,
            inter: scratch.mlp_intermediate,
            acc: scratch.stream_acc,
            logits: scratch.logits,
            ctx_in: DevicePtr::NULL,
            ctx_fc: scratch.fc_proj,
            ctx_kv: scratch.fused_kv_out,
            q: scratch.q_buf,
            k: scratch.k_buf,
            v: scratch.v_buf,
            attn: scratch.attn_out,
            attn_proj: scratch.stream_acc,
        }
    }
}

/// One exchange of the walk. The payload is `[gamma, width]` BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Swap {
    /// Layer input of the MLP (the post-attention norm), head to worker.
    Input(usize),
    /// Each rank's share of the SiLU-gated activation.
    Activation(usize),
    /// Each rank's share of the down projection; the head keeps them all.
    Output(usize),
    /// The final-norm rows, head to worker.
    Hidden,
    /// Each rank's share of the vocabulary logits; the head keeps them all.
    Logits,
    /// The workers' top-K candidate lists of their shares, to the head.
    Candidates,
    /// Context append `i`'s input rows, head to worker.
    CtxInput(usize),
    /// Each rank's share of its `fc` rows; every rank keeps them all.
    CtxHidden(usize),
    /// Each rank's share of its fused K/V rows; the head keeps them all.
    CtxKv(usize),
    /// `attn`: the layer's normed input rows, head to worker.
    AttnIn(usize),
    /// `attn`: each rank's shares of q, k and v; the head keeps them all.
    Qkv(usize),
    /// `attn`: the attention rows o_proj reads, head to worker.
    AttnOut(usize),
    /// `attn`: each rank's share of o_proj; the head keeps them all.
    OProj(usize),
}

/// Stream work between two swaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Layer input norm through attention (head only).
    Attention(usize),
    /// The whole unsplit post-attention half (head only).
    Post(usize),
    /// o_proj, residual and the MLP input norm (head only).
    Project(usize),
    /// This rank's share of gate and up, SiLU-gated.
    GateUp(usize),
    /// This rank's share of the down projection.
    Down(usize),
    /// The MLP residual (head only).
    Residual(usize),
    /// The whole unsplit tail (head only).
    Tail,
    /// The final norm (head only).
    Norm,
    /// This rank's share of the vocabulary projection.
    Vocab,
    /// Every row's top-K candidates of this rank's share (the head also
    /// projects the selector's hidden rows).
    TopK,
    /// Draft selection over the joined logits, or over both ranks'
    /// candidate lists (head only).
    Select,
    /// This rank's share of context append `i`'s `fc` projection.
    CtxFc(usize),
    /// `hidden_norm` over the joined `fc` rows (every rank).
    CtxNorm(usize),
    /// This rank's share of context append `i`'s fused K/V projection.
    CtxKv(usize),
    /// `attn`: the input norm and attention conv (head only).
    AttnPre(usize),
    /// `attn`: this rank's shares of the q, k and v projections.
    Qkv(usize),
    /// `attn`: q/k norms, RoPE, the cache write and attention (head only).
    AttnCore(usize),
    /// `attn`: this rank's share of o_proj.
    OProj(usize),
    /// `attn`: o_proj's residual and the MLP input norm (head only).
    ProjectRest(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Run(Piece),
    Swap(Swap),
}

/// The shapes a split is planned from; identical on every rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub gamma: usize,
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub layers: usize,
    pub parts: Parts,
    /// `fc` input width and fused K/V output rows of a context append; zero
    /// without `ATLAS_GLM_DRAFT_TP_CTX`.
    pub ctx_in: usize,
    pub ctx_kv: usize,
    /// The context rows of the current propose's appends.
    pub ctx: CtxRows,
    /// `ATLAS_GLM_DRAFT_TP_TOPK`: the head split swaps candidate lists, not
    /// logits (single-sequence proposes only).
    pub topk: bool,
    /// The ranks sharing the propose: 2 (the pair) or 4 (the chain).
    pub ranks: usize,
    /// The attention projections' widths (`attn`).
    pub q_dim: usize,
    pub kv_dim: usize,
}

/// Output rows a tensor-core GEMV CTA owns. A share starts on a CTA boundary,
/// so its launch runs exactly the CTAs the unsplit launch runs for its rows.
const CTA_ROWS: usize = 16;

/// Rank `rank`'s contiguous share of `n` output rows of `ranks`: `(first,
/// rows)`. Rank `r` starts at the largest CTA boundary at or under `r * n /
/// ranks` and the last takes the rest (on the pair: rank 0 the rows below
/// the boundary at or under `n / 2`, rank 1 the rest).
pub(crate) fn share(n: usize, rank: usize, ranks: usize) -> (usize, usize) {
    let cut = |r: usize| {
        if r == ranks {
            n
        } else {
            n * r / ranks / CTA_ROWS * CTA_ROWS
        }
    };
    (cut(rank), cut(rank + 1) - cut(rank))
}

/// The widest of the `ranks` shares of `n` rows.
pub(crate) fn widest(n: usize, ranks: usize) -> usize {
    (0..ranks).map(|r| share(n, r, ranks).1).max().unwrap_or(n)
}

impl Geometry {
    /// This plan for a propose of `rows` rows (`gamma` is the rows of a
    /// propose: a batched one runs B×gamma).
    pub(crate) fn with_rows(&self, rows: usize) -> Self {
        Self {
            gamma: rows,
            topk: self.topk && rows.is_multiple_of(self.gamma),
            ..*self
        }
    }

    /// Every row range is non-empty and starts on a 16-byte boundary of the
    /// packed twin (its K is a multiple of 32).
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.gamma > 0 && self.layers > 0 && (self.parts.mlp || self.parts.head),
            "ATLAS_GLM_DRAFT_TP: nothing to split ({self:?})"
        );
        ensure!(
            matches!(self.ranks, 2 | 4),
            "ATLAS_GLM_DRAFT_TP splits over the pair or the chain ({self:?})"
        );
        ensure!(
            self.hidden.is_multiple_of(32) && self.inter.is_multiple_of(32),
            "ATLAS_GLM_DRAFT_TP needs hidden and intermediate sizes in multiples of 32 ({self:?})"
        );
        ensure!(
            !self.parts.attn
                || self.parts.mlp
                    && self.q_dim.is_multiple_of(32)
                    && self.kv_dim.is_multiple_of(32)
                    && self.q_dim >= self.ranks * CTA_ROWS
                    && self.kv_dim >= self.ranks * CTA_ROWS,
            "ATLAS_GLM_DRAFT_TP attn needs mlp and projections the shares can serve ({self:?})"
        );
        let shares = self.ranks * CTA_ROWS;
        ensure!(
            self.vocab >= shares && self.hidden >= shares && self.inter >= shares,
            "ATLAS_GLM_DRAFT_TP: drafter too small to split {} ways ({self:?})",
            self.ranks
        );
        ensure!(
            (self.ctx_in, self.ctx_kv) == (0, 0)
                || self.ctx_in > 0 && self.ctx_in.is_multiple_of(32) && self.ctx_kv >= shares,
            "ATLAS_GLM_DRAFT_TP_CTX: context projections the shares cannot serve ({self:?})"
        );
        Ok(())
    }

    /// BF16 values per row of a swap's payload: every rank sends the widest
    /// share's width (a narrower share's tail is unused).
    fn width(&self, swap: Swap) -> usize {
        let widest = |n| widest(n, self.ranks);
        match swap {
            Swap::Input(_) | Swap::Hidden => self.hidden,
            Swap::Activation(_) => widest(self.inter),
            Swap::Output(_) => widest(self.hidden),
            Swap::Logits => widest(self.vocab),
            Swap::Candidates => CANDIDATE_ROW_BYTES / 2,
            Swap::CtxInput(_) => self.ctx_in,
            Swap::CtxHidden(_) => widest(self.hidden),
            Swap::CtxKv(_) => widest(self.ctx_kv),
            Swap::AttnIn(_) => self.hidden,
            Swap::Qkv(_) => widest(self.q_dim) + 2 * widest(self.kv_dim),
            Swap::AttnOut(_) => self.q_dim,
            Swap::OProj(_) => widest(self.hidden),
        }
    }

    /// Whether `swap` carries the head's rows out (the pair's one-way
    /// exchange, the chain's broadcast) rather than every rank's share.
    pub(crate) fn from_head(swap: Swap) -> bool {
        matches!(
            swap,
            Swap::Input(_) | Swap::Hidden | Swap::CtxInput(_) | Swap::AttnIn(_) | Swap::AttnOut(_)
        )
    }

    /// The rows a swap carries: a context append's own, else the propose's.
    pub(crate) fn rows(&self, swap: Swap) -> usize {
        match swap {
            Swap::CtxInput(i) | Swap::CtxHidden(i) | Swap::CtxKv(i) => self.ctx.get(i),
            _ => self.gamma,
        }
    }

    /// Bytes each rank sends and receives in `swap`.
    pub(crate) fn bytes(&self, swap: Swap) -> usize {
        self.rows(swap) * self.width(swap) * 2
    }

    /// Rank `rank`'s whole walk: the context appends (`ctx_steps`), then
    /// the layers and tail (`layer_steps`).
    pub(crate) fn steps(&self, rank: usize) -> Vec<Step> {
        let mut steps: Vec<Step> = self.ctx.appends().flat_map(ctx_steps).collect();
        steps.extend(self.layer_steps(rank));
        steps
    }

    /// Rank `rank`'s walk of the layers and tail. Rank 0 runs the whole
    /// propose forward; the other ranks only their shares.
    pub(crate) fn layer_steps(&self, rank: usize) -> Vec<Step> {
        use Piece::*;
        let head = rank == 0;
        let mut steps = Vec::new();
        // Pieces only the head runs, pieces both run, and swaps.
        let own = |steps: &mut Vec<Step>, piece| {
            if head {
                steps.push(Step::Run(piece));
            }
        };
        let both = |steps: &mut Vec<Step>, piece| steps.push(Step::Run(piece));
        let swap = |steps: &mut Vec<Step>, swap| steps.push(Step::Swap(swap));
        for l in 0..self.layers {
            if self.parts.attn {
                own(&mut steps, AttnPre(l));
                swap(&mut steps, Swap::AttnIn(l));
                both(&mut steps, Qkv(l));
                swap(&mut steps, Swap::Qkv(l));
                own(&mut steps, AttnCore(l));
                swap(&mut steps, Swap::AttnOut(l));
                both(&mut steps, OProj(l));
                swap(&mut steps, Swap::OProj(l));
                own(&mut steps, ProjectRest(l));
            } else {
                own(&mut steps, Attention(l));
            }
            if !self.parts.mlp {
                own(&mut steps, Post(l));
                continue;
            }
            if !self.parts.attn {
                own(&mut steps, Project(l));
            }
            swap(&mut steps, Swap::Input(l));
            both(&mut steps, GateUp(l));
            swap(&mut steps, Swap::Activation(l));
            both(&mut steps, Down(l));
            swap(&mut steps, Swap::Output(l));
            own(&mut steps, Residual(l));
        }
        if !self.parts.head {
            own(&mut steps, Tail);
            return steps;
        }
        own(&mut steps, Norm);
        swap(&mut steps, Swap::Hidden);
        both(&mut steps, Vocab);
        if self.topk {
            both(&mut steps, TopK);
            swap(&mut steps, Swap::Candidates);
        } else {
            swap(&mut steps, Swap::Logits);
        }
        own(&mut steps, Select);
        steps
    }

    /// The swaps of a propose, in order: the same list on every rank.
    pub(crate) fn swaps(&self) -> Vec<Swap> {
        swaps_of(&self.steps(1))
    }
}

pub(crate) fn swaps_of(steps: &[Step]) -> Vec<Swap> {
    steps
        .iter()
        .filter_map(|s| match s {
            Step::Swap(x) => Some(*x),
            Step::Run(_) => None,
        })
        .collect()
}

#[path = "rank_split_walk.rs"]
mod walk;
pub(crate) use walk::{Graphs, SplitOps, run_count, walk};

#[cfg(test)]
#[path = "rank_split_tests.rs"]
mod tests;
