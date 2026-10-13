// SPDX-License-Identifier: AGPL-3.0-only

//! Draft tensor parallel for the qwen4_exp MTP drafter
//! (`ATLAS_QWEN4EXP_MTP_DRAFT_TP=1`, default off; both ranks must agree,
//! `startup_parity`).
//!
//! The drafter runs on the head rank (`qwen4exp_mtp.rs`), so on the TP2 pair
//! the worker idled through every batched propose: 9.7 of 9.9 ms a C8 step
//! (nsys, 2026-10-06). The largest read of a propose position is the draft
//! head (`qwen4exp_draft_head.rs`: the 100k-row NVFP4 copy, 1.12 of ~3.1 ms a
//! position at 8 rows). With the switch the worker projects half of its rows:
//!
//! ```text
//!   head (rank 0)                              worker (rank 1)
//!   body, hc_head -> h_out [n, H]
//!   swap Hidden   h_out ----------------------> rows_in
//!   draft rows [0, width)                      draft rows [cut, rows)
//!   swap Logits   <---------- shards ---------->
//!   assemble [n, rows], argmax                 (discards)
//! ```
//!
//! The split of the rows is the LM-head split's (`model::qwen4exp_lmhead_split`:
//! [`Split`], `exchange`, `assemble`). Each rank projects `width` rows from
//! its start with the kernel the unsplit draft head runs for `n` rows, and an
//! output column depends only on its weight row and the input row
//! ([`DraftHead::project_rows_range`]), so the assembled logits, and with them
//! every draft and confidence, are the bytes the unsplit head writes. The
//! worker builds its draft head from its own (full) LM head exactly as the
//! head does: the same rows, the same NVFP4 copy.
//!
//! Lockstep. Only a batched propose splits. The head announces it at its
//! command loop (`EP_CMD_DRAFT_ASSIST`, the preamble word [`Plan::word`]: `n`
//! rows, `drafts` positions), so the worker is at its idle receive; the worker
//! enqueues its whole walk ([`Qwen4ExpDraftAssist::serve`]) and returns. Both
//! ranks walk [`Plan::swaps`], of sizes that depend only on the plan and the
//! draft head's shape, which the startup agreement pins (the switch, the
//! NVFP4 copy, an id list, `--mtp-vocab`). On the head every exit after the
//! announce, an error or a fallback to the per-sequence path included, goes
//! through [`TpRun`], which issues the swaps the propose did not reach; the
//! worker issues all of its swaps even when a projection fails. So neither
//! rank is ever left waiting in a swap the other never issues.
//!
//! Prior art: SparkGLM's `ATLAS_GLM_DRAFT_TP_BATCH` (`dflash_head/rank_split`)
//! splits the DFlash drafter's batched propose across the pair the same way.

use std::cell::Cell;

use anyhow::{Context, Result, bail, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen4ExpMtpHead;
use super::qwen4exp_mtp_batch::{PROPOSE_BATCH_MAX, PROPOSE_BATCH_MAX_DRAFTS};
use crate::layers::qwen4exp_draft_head::{self as draft_head, DraftHead};
use crate::model::qwen4exp_lmhead_split::{Split, assemble, exchange};
use crate::weight_map::DenseWeight;

const BF16: usize = 2;

/// `ATLAS_QWEN4EXP_MTP_DRAFT_TP`: `1` on; unset, empty or `0` off; anything
/// else is refused (a typo must not run unsplit on one rank).
pub fn parse(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_QWEN4EXP_MTP_DRAFT_TP must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_QWEN4EXP_MTP_DRAFT_TP`, from the profile both ranks share.
pub fn requested() -> Result<bool> {
    parse(std::env::var("ATLAS_QWEN4EXP_MTP_DRAFT_TP").ok().as_deref())
}

/// The value both ranks must agree on: the switch (bit 0), then what picks
/// the draft head's rows and kernel: the NVFP4 copy (bit 1) and an id list
/// (bit 2). 0 with the switch off. `--mtp-vocab` is agreed by the server.
pub fn parity_word() -> Result<u64> {
    Ok(if requested()? {
        1 | (draft_head::nvfp4_requested() as u64) << 1 | (draft_head::list_requested() as u64) << 2
    } else {
        0
    })
}

/// One announced batched propose: `n` rows, `drafts` positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub n: usize,
    pub drafts: usize,
}

/// The two swaps of a position, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swap {
    /// The head's final-norm rows to the worker.
    Hidden,
    /// Each rank's shard of the draft logits; both keep both.
    Logits,
}

impl Plan {
    /// High byte of an announce word, so a stray word is refused, not walked.
    const TAG: u32 = 0x51 << 24;

    fn check(self) -> Result<()> {
        ensure!(
            (1..=PROPOSE_BATCH_MAX).contains(&self.n)
                && (1..=PROPOSE_BATCH_MAX_DRAFTS).contains(&self.drafts),
            "qwen4_exp draft TP: plan {self:?} outside 1..={PROPOSE_BATCH_MAX} rows, \
             1..={PROPOSE_BATCH_MAX_DRAFTS} positions"
        );
        Ok(())
    }

    /// The announce word: tag, positions, rows.
    pub fn word(self) -> Result<u32> {
        self.check()?;
        Ok(Self::TAG | (self.drafts as u32) << 8 | self.n as u32)
    }

    pub fn from_word(word: u32) -> Result<Self> {
        ensure!(
            word & 0xFFFF_0000 == Self::TAG,
            "qwen4_exp draft TP: {word:#010x} is not a draft-TP announce"
        );
        let plan = Self {
            n: (word & 0xFF) as usize,
            drafts: (word >> 8 & 0xFF) as usize,
        };
        plan.check()?;
        Ok(plan)
    }

    /// Every swap of the propose, in order: the same list on both ranks.
    pub fn swaps(self) -> impl Iterator<Item = Swap> {
        (0..self.drafts).flat_map(|_| [Swap::Hidden, Swap::Logits])
    }
}

/// One rank's share of the draft head and the staging of its swaps, built
/// alike on both ranks.
pub struct DraftTp {
    geom: Split,
    rank: usize,
    hidden: usize,
    /// `[PROPOSE_BATCH_MAX, width]` BF16: this rank's shard (also what the
    /// worker sends in a Hidden swap), then the peer's.
    send: DevicePtr,
    recv: DevicePtr,
    /// `[PROPOSE_BATCH_MAX, hidden]` BF16: where the head's rows land on the
    /// worker, and the worker's unused direction on the head.
    rows_in: DevicePtr,
}

impl DraftTp {
    pub fn new(
        draft: &DraftHead,
        hidden: usize,
        rank: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(rank < 2, "qwen4_exp draft TP on rank {rank} of a pair");
        let geom = Split::new(draft.rows() as usize)
            .with_context(|| format!("qwen4_exp draft TP: {} draft rows", draft.rows()))?;
        let rows_in = PROPOSE_BATCH_MAX * hidden * BF16;
        let shard = PROPOSE_BATCH_MAX * geom.width() * BF16;
        Ok(Self {
            geom,
            rank,
            hidden,
            send: gpu.alloc(shard.max(rows_in))?,
            recv: gpu.alloc(shard)?,
            rows_in: gpu.alloc(rows_in)?,
        })
    }

    /// Issue `swap` of an `n`-row position on `stream`; the head sends
    /// `h_out` in a Hidden swap.
    fn swap(
        &self,
        swap: Swap,
        comm: &dyn CommBackend,
        gpu: &dyn GpuBackend,
        h_out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        match swap {
            Swap::Hidden => {
                let send = if self.rank == 0 { h_out } else { self.send };
                let bytes = n * self.hidden * BF16;
                comm.peer_exchange_async(send.0, self.rows_in.0, bytes, stream)
            }
            Swap::Logits => exchange(comm, gpu, self.geom, (self.send, self.recv), n, stream),
        }
    }

    /// This rank's shard of the draft logits of the `n` rows at `input`.
    fn project(
        &self,
        draft: &DraftHead,
        gpu: &dyn GpuBackend,
        dense_batchm_k: KernelHandle,
        input: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let rows = (self.geom.start(self.rank) as u32, self.geom.width() as u32);
        draft.project_rows_range(
            gpu,
            dense_batchm_k,
            input,
            self.send,
            n as u32,
            rows,
            self.hidden as u32,
            stream,
        )
    }
}

/// Whether `draft` serves every batched propose width (an id list does not).
fn batchable(draft: &DraftHead, dense_batchm_k: KernelHandle) -> bool {
    (2..=PROPOSE_BATCH_MAX).all(|n| draft.rows_batchable(n, dense_batchm_k))
}

impl Qwen4ExpMtpHead {
    /// Rank 0 of the pair: split the draft head of every batched propose the
    /// model announces. Refused where the batched propose cannot run.
    pub fn enable_draft_tp(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        ensure!(
            crate::model::qwen4exp_batch_fast::requested(),
            "ATLAS_QWEN4EXP_MTP_DRAFT_TP splits the batched propose, which needs \
             ATLAS_QWEN4EXP_BATCH_FAST=1"
        );
        ensure!(
            batchable(&self.draft, self.dense_gemv_batchm_k),
            "ATLAS_QWEN4EXP_MTP_DRAFT_TP: this draft head never batches (an id list?)"
        );
        self.tp = Some(DraftTp::new(
            &self.draft,
            self.module.config.hidden_size,
            0,
            gpu,
        )?);
        tracing::info!(
            "qwen4_exp MTP draft TP: the worker projects draft rows {}.. of each batched propose \
             (ATLAS_QWEN4EXP_MTP_DRAFT_TP=1)",
            self.tp.as_ref().map_or(0, |tp| tp.geom.start(1))
        );
        Ok(())
    }

    /// The announce word of a batched propose of `n` rows and `drafts`
    /// positions over `comm`, when it splits: only one `propose_batch` will
    /// run batched (`batch_admits`), or the pair walks every exchange for
    /// nothing.
    pub(super) fn tp_announce_word(
        &self,
        comm: &dyn CommBackend,
        n: usize,
        drafts: usize,
        grammar: bool,
        ctx: &crate::layer::ForwardContext,
    ) -> Option<u32> {
        let pair = comm.rank() == 0 && comm.world_size() == 2;
        (self.tp.is_some()
            && pair
            && !grammar
            && n >= 2
            && comm.supports_peer_exchange_async()
            && self.batch_admits(n, drafts, ctx))
        .then(|| Plan { n, drafts }.word().ok())
        .flatten()
    }

    /// The run of a propose the model announced (it hands the propose its
    /// communicator then, and only then).
    pub(super) fn draft_tp_run<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        gpu: &'a dyn GpuBackend,
        plan: Plan,
        stream: u64,
    ) -> Result<Option<TpRun<'a>>> {
        let (Some(tp), Some(comm)) = (self.tp.as_ref(), comm) else {
            return Ok(None);
        };
        plan.check()?;
        Ok(Some(TpRun {
            tp,
            comm,
            gpu,
            plan,
            stream,
            issued: Cell::new(0),
        }))
    }
}

/// The head's side of an announced batched propose. [`Self::finish`], or the
/// drop on any other exit, issues the swaps the propose did not reach, so the
/// worker, which walks them all, never waits in one.
pub(super) struct TpRun<'a> {
    tp: &'a DraftTp,
    comm: &'a dyn CommBackend,
    gpu: &'a dyn GpuBackend,
    plan: Plan,
    stream: u64,
    issued: Cell<usize>,
}

impl TpRun<'_> {
    /// Issue the next swap of the plan, which must be `swap`.
    fn issue(&self, swap: Swap, h_out: DevicePtr) -> Result<()> {
        let at = self.issued.get();
        let next = self.plan.swaps().nth(at);
        ensure!(
            next == Some(swap),
            "qwen4_exp draft TP: {swap:?} at swap {at} of {:?}",
            self.plan
        );
        // Counted before it is issued: a failed issue is not retried.
        self.issued.set(at + 1);
        let (tp, n) = (self.tp, self.plan.n);
        tp.swap(swap, self.comm, self.gpu, h_out, n, self.stream)
    }

    /// A position's draft head: the `n` rows at `h_out` onto the assembled
    /// logits rows at `logits` (pitch = the draft rows), the bytes
    /// [`DraftHead::project_rows`] writes.
    pub(super) fn head(
        &self,
        draft: &DraftHead,
        dense_batchm_k: KernelHandle,
        h_out: DevicePtr,
        logits: DevicePtr,
    ) -> Result<()> {
        let (tp, n, stream) = (self.tp, self.plan.n, self.stream);
        self.issue(Swap::Hidden, h_out)?;
        tp.project(draft, self.gpu, dense_batchm_k, h_out, n, stream)?;
        self.issue(Swap::Logits, DevicePtr::NULL)?;
        assemble(self.gpu, tp.geom, 0, (tp.send, tp.recv), logits, n, stream)
    }

    /// Issue every swap not yet issued (their payloads unused).
    fn drain(&self) -> Result<()> {
        while let Some(swap) = self.plan.swaps().nth(self.issued.get()) {
            self.issue(swap, self.tp.rows_in)?;
        }
        Ok(())
    }

    /// The propose is done: issue what it did not reach (nothing, normally).
    pub(super) fn finish(self) -> Result<()> {
        self.drain()
    }
}

impl Drop for TpRun<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.drain() {
            tracing::error!("qwen4_exp draft TP: draining the propose's swaps failed: {e:#}");
        }
    }
}

/// The worker's copy of the draft head: it serves the head's announced
/// batched proposes and proposes nothing.
pub struct Qwen4ExpDraftAssist {
    draft: DraftHead,
    tp: DraftTp,
    dense_batchm_k: KernelHandle,
}

impl Qwen4ExpDraftAssist {
    /// The draft head the head builds (`Qwen4ExpMtpHead::new`), from this
    /// rank's own LM head, `vocab` x `hidden` BF16.
    pub fn new(
        lm_head: &DenseWeight,
        vocab: usize,
        hidden: usize,
        mtp_vocab: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let draft = DraftHead::build(lm_head, vocab, hidden, mtp_vocab, gpu)?;
        let dense_batchm_k =
            super::super::try_kernel(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
        ensure!(
            batchable(&draft, dense_batchm_k),
            "ATLAS_QWEN4EXP_MTP_DRAFT_TP: this draft head never batches (an id list?)"
        );
        let tp = DraftTp::new(&draft, hidden, 1, gpu)?;
        Ok(Self {
            draft,
            tp,
            dense_batchm_k,
        })
    }

    /// Enqueue this rank's whole walk of the announced propose (`word`) on
    /// `stream`, with no host sync: the swaps order it against the head. A
    /// failed projection still issues every swap, then reports.
    pub fn serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
    ) -> Result<()> {
        ensure!(
            comm.rank() == 1 && comm.world_size() == 2,
            "qwen4_exp draft TP serves on rank 1 of 2"
        );
        let plan = Plan::from_word(word)?;
        let tp = &self.tp;
        let mut failed = None;
        for swap in plan.swaps() {
            tp.swap(swap, comm, gpu, DevicePtr::NULL, plan.n, stream)?;
            if swap == Swap::Hidden && failed.is_none() {
                let k = self.dense_batchm_k;
                failed = tp
                    .project(&self.draft, gpu, k, tp.rows_in, plan.n, stream)
                    .err();
            }
        }
        failed.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
#[path = "qwen4exp_mtp_tp_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "qwen4exp_mtp_tp_gpu_tests.rs"]
mod gpu_tests;
