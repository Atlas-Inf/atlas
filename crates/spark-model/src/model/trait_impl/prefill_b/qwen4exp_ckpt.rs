// SPDX-License-Identifier: AGPL-3.0-only

//! Model side of the qwen4_exp mid-chunk prefix-cache checkpoint
//! (`ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`, default off; the layer side and
//! the why are in `layers::qwen4exp_ckpt`).
//!
//! The last chunk of a prefix-caching prompt runs as ONE pass (no tail
//! split), so its numerics are the caching-off run's, and the checkpoint the
//! split existed for is captured inside the pass at `cp`: the last 64-token
//! chunk boundary of the pass at or below the tail cut (so `cp` sits up to 63
//! tokens below where the split put it; the next turn replays those). Plan:
//! reserve a snapshot slot and the per-GDN-layer destinations
//! ([`MidCapturePlan`] with `ckpt` set, which the GDN conv split and the
//! layers read through `ForwardContext::midchunk_capture`), arm the per-pass
//! layer hooks. Finalize: require every GDN layer's state and the PLE carry
//! captured, rebuild the PLE history and QSA keys at `cp`, attach them as
//! the slot's aux, and index it as the intermediate checkpoint of
//! `tokens[..cp]` -- the same registration the tail split's first pass made
//! (`register_intermediate_checkpoint`). Anything missing frees the slot.
//!
//! Every rank plans from the same tokens, pass and config; the restore
//! depth of the next turn is agreed across ranks as before
//! (`pc_policy::agree_restore`), so a rank that could not capture (no free
//! slot) only makes the pair restore shallower.
//!
//! # Off the block grid (`ATLAS_QWEN4EXP_FINISH_LEAF=1`)
//!
//! `cp` is a 64-row chunk boundary of the PASS, so it is a block boundary
//! only when the pass starts on one. A prompt's last pass starts at a chunk
//! boundary of the prefill, and an idle prefill's chunk is the arena
//! (`max_batch_tokens` = the prefill budget plus the decode rows, 16388 for a
//! 16384 budget at 4 sequences: `initial_chunk_budget`). Every chunk after
//! the first then starts 4 tokens past a block boundary (65540 for the fifth),
//! no `cp` of the last pass is a block boundary, and a multi-chunk prompt got
//! no tail checkpoint at all: its next turn restored the last chunk boundary
//! and replayed up to a whole chunk. With the switch `cp` only has to be a
//! QSA pool-block boundary (the indexer blob is rebuilt from whole pooled
//! blocks, [`qsa_blob_at`]). A checkpoint off the block grid is an ordinary
//! one: the index and the restore take any length (the chunk-boundary
//! checkpoints are already such), and a lookup restores it whenever the
//! whole-block match reaches it.

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;
use std::cell::{Cell, RefCell};

use super::super::super::types::TransformerModel;
use super::midchunk_capture::MidCapturePlan;
use crate::layers::qwen4exp_ckpt as ckpt;
use crate::traits::SequenceState;

thread_local! {
    /// `(ptr, bytes)` of this thread's PLE conv-carry staging buffer.
    static PLE_STAGE: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// Where `cp` lands for a pass over `[start, start + count)` and a tail cut
/// `cut`: the last 64-token chunk boundary of the pass at or below the cut,
/// when that is a multiple of `align` (the block size, or with
/// `ATLAS_QWEN4EXP_FINISH_LEAF` the QSA ratio) strictly inside the pass.
pub(super) fn ckpt_row(start: usize, count: usize, cut: usize, align: usize) -> Option<usize> {
    if align == 0 || !(start < cut && cut < start + count) {
        return None;
    }
    let cp = start + (cut - start) / ckpt::CHUNK * ckpt::CHUNK;
    (cp > start && cp.is_multiple_of(align)).then_some(cp)
}

/// [`ckpt_row`] with the alignment the switch picks, for a QSA ratio `ratio`
/// and blocks of `bs`.
pub(super) fn tail_ckpt_row(
    start: usize,
    count: usize,
    cut: usize,
    bs: usize,
    ratio: usize,
    off_grid: bool,
) -> Option<usize> {
    let ratio = ratio.max(1);
    let align = if off_grid { ratio } else { bs };
    ckpt_row(start, count, cut, align).filter(|cp| cp.is_multiple_of(ratio))
}

impl TransformerModel {
    /// The last chunk runs unsplit: the checkpoint is captured in-pass.
    pub(super) fn qwen4exp_ckpt_takes_tail(&self) -> bool {
        ckpt::requested() && self.config.model_type == "qwen4_exp"
    }

    /// Where the in-pass checkpoint of a pass over `[start, start + count)`
    /// lands for the tail cut `cut` ([`tail_ckpt_row`]).
    fn qwen4exp_ckpt_row(
        &self,
        start: usize,
        count: usize,
        cut: usize,
        bs: usize,
    ) -> Option<usize> {
        let off_grid = crate::model::trait_impl::finish_leaf::qwen4exp_requested();
        let ratio = self.config.indexer_compress_ratio;
        tail_ckpt_row(start, count, cut, bs, ratio, off_grid)
    }

    /// Whether a prefill of a `total`-token prompt whose last pass starts at
    /// `start` saves a tail checkpoint of its own: the tail split's when the
    /// cut lies past `start`, or the in-pass one when its row exists.
    pub(in crate::model) fn tail_checkpoint_follows(
        &self,
        total: usize,
        start: usize,
        bs: usize,
    ) -> bool {
        let cut = super::pc_policy::tail_cut(total, bs);
        if self.qwen4exp_ckpt_takes_tail() {
            let count = total.saturating_sub(start);
            self.qwen4exp_ckpt_row(start, count, cut, bs).is_some()
        } else {
            start < cut
        }
    }

    /// The pass's in-pass capture plan: the qwen4_exp checkpoint, else the
    /// generic mid-chunk tail capture (`midchunk_capture`).
    pub(super) fn plan_pass_capture(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        span: [usize; 2],
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<Option<MidCapturePlan>> {
        if let Some(p) =
            self.prepare_qwen4exp_ckpt(tokens, seq, kv_cache, span, is_last_chunk, stream)?
        {
            return Ok(Some(p));
        }
        let [start, count] = span;
        Ok(self.prepare_midchunk_capture(tokens, seq, kv_cache, start, count, stream))
    }

    /// Plan the in-pass checkpoints of a pass over
    /// `[proc_start, proc_start + proc_count)`: the last chunk's tail row,
    /// and the dense and branch-point rows (`qwen4exp_points`), or `None`.
    pub(super) fn prepare_qwen4exp_ckpt(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        span: [usize; 2],
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<Option<MidCapturePlan>> {
        let [proc_start, proc_count] = span;
        // A pass that failed after planning left its hooks armed: disarm.
        let _ = ckpt::end();
        if !self.qwen4exp_ckpt_takes_tail()
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.tokens_have_vision_pad(tokens)
        {
            return Ok(None);
        }
        let bs = kv_cache.block_size();
        let cut = super::pc_policy::tail_cut(tokens.len(), bs);
        let tail = if is_last_chunk {
            let cp = self.qwen4exp_ckpt_row(proc_start, proc_count, cut, bs);
            if cp.is_none() {
                tracing::info!(
                    "qwen4_exp mid-chunk checkpoint: no 64-token boundary in [{proc_start}, {cut}]"
                );
            }
            cp
        } else {
            None
        };
        let rows = self.qwen4exp_pass_points(tokens, seq, span, tail, bs);
        let points = self.qwen4exp_reserve_points(seq, kv_cache, &rows);
        let Some(&(first, slot, _)) = points.first() else {
            return Ok(None);
        };
        // A finish leaf's copy into these slots may still be in flight on
        // another stream (no-op without a finish-leaf flag).
        let staged = self
            .finish_leaf_wait_copies(stream)
            .and_then(|()| self.ple_stage(points.len(), stream));
        let (ple_dst, ple_bytes) = staged.inspect_err(|_| {
            points.iter().for_each(|p| self.ssm_snapshots.free(p.1));
        })?;
        let n = self.ssm_snapshots.num_ssm_layers();
        let dsts = |slot: usize| {
            let h = (0..n).map(|l| self.ssm_snapshots.tail_h_dst(l, slot));
            let c = (0..n).map(|l| self.ssm_snapshots.tail_conv_dst(l, slot));
            (h.collect::<Vec<_>>(), c.collect::<Vec<_>>())
        };
        let (h_dsts, conv_dsts) = dsts(slot);
        let extra = points[1..]
            .iter()
            .map(|&(r, slot, _)| {
                let (h_dsts, conv_dsts) = dsts(slot);
                ckpt::CapPoint {
                    cap_local: r - proc_start,
                    h_dsts,
                    conv_dsts,
                }
            })
            .collect();
        let ple: Vec<_> = (points.iter().enumerate())
            .map(|(j, p)| (p.0 - proc_start, ple_dst.offset(j * ple_bytes)))
            .collect();
        ckpt::begin_many(&ple);
        if points.len() > 1 || tail.is_none() {
            tracing::info!(
                "qwen4_exp in-pass checkpoints planned at {:?} (pass {proc_start}+{proc_count})",
                points.iter().map(|p| (p.0, p.2)).collect::<Vec<_>>()
            );
        }
        Ok(Some(MidCapturePlan {
            cap_local: first - proc_start,
            snap_slot: slot,
            tb: first,
            h_dsts,
            conv_dsts,
            h_bytes: self.ssm_snapshots.h_bytes(),
            conv_bytes: self.ssm_snapshots.conv_bytes(),
            bs,
            cap_local_early: None,
            snap_slot_early: None,
            tb_early: None,
            h_dsts_early: Vec::new(),
            conv_dsts_early: Vec::new(),
            ckpt: true,
            extra,
            ckpt_points: points,
        }))
    }

    /// Before the chunk's own save: finalize a mid-chunk checkpoint pass;
    /// the returned scope drops any pass-end aux it kept for that save.
    pub(super) fn qwen4exp_ckpt_saves(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &Option<MidCapturePlan>,
        stream: u64,
    ) -> Result<PassAuxScope> {
        let scope = PassAuxScope::new();
        if let Some(plan) = plan.as_ref().filter(|p| p.ckpt) {
            self.finalize_qwen4exp_ckpt(tokens, seq, kv_cache, plan, stream)?;
        }
        Ok(scope)
    }

    /// After the pass: attach the aux at each point and index its
    /// checkpoint, or free the slots when anything was not captured or a
    /// step fails.
    fn finalize_qwen4exp_ckpt(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &MidCapturePlan,
        stream: u64,
    ) -> Result<()> {
        let mut left: Vec<usize> = plan.ckpt_points.iter().map(|p| p.1).collect();
        let res = self.register_qwen4exp_ckpt(tokens, seq, kv_cache, plan, &mut left, stream);
        // Whatever was not registered goes back to the pool.
        left.into_iter()
            .for_each(|slot| self.ssm_snapshots.free(slot));
        res
    }

    /// [`Self::finalize_qwen4exp_ckpt`]'s work; each slot it registers is
    /// taken out of `left`.
    fn register_qwen4exp_ckpt(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        plan: &MidCapturePlan,
        left: &mut Vec<usize>,
        stream: u64,
    ) -> Result<()> {
        let (h_done, ple_done) = ckpt::end();
        // The pass wrote the slots: order the next writer after it.
        self.finish_leaf_record_copy(stream)?;
        let has_ple = self.config.ple_layer_ids.iter().any(|&id| id > 0);
        let n = self.ssm_snapshots.num_ssm_layers();
        let points = plan.ckpt_points.len();
        if h_done != n || (has_ple && ple_done != points) {
            tracing::warn!(
                "qwen4_exp in-pass checkpoints at {:?}: captured {h_done}/{n} GDN states, \
                 {ple_done}/{points} PLE carries; not registered",
                plan.ckpt_points
            );
            return Ok(());
        }
        // The pass-end aux, rebuilt at each point. With
        // ATLAS_QWEN4EXP_CKPT_AUX_SHARE the pass-end blobs are kept for the
        // final save, which would read the very same layer state back again
        // (`take_pass_aux`).
        let mut aux = self.ssm_snapshots.take_aux(plan.snap_slot);
        self.collect_aux_states_into(seq, stream, &mut aux)?;
        let (stage, _) = PLE_STAGE.with(Cell::get);
        let stage_bytes = self.ple_stage_bytes();
        let mut built = Vec::with_capacity(points);
        for (j, &(cp, slot, branch)) in plan.ckpt_points.iter().enumerate() {
            // Never index blocks past the contiguous fully-written KV (as
            // `prefill_b_save_checkpoint`).
            if seq.kv_valid_tokens / plan.bs < cp / plan.bs {
                tracing::warn!("qwen4_exp in-pass checkpoint at token {cp}: KV not valid");
                continue;
            }
            let ple_src = DevicePtr(stage).offset(j * stage_bytes);
            let mut rebuilt = Vec::with_capacity(aux.len());
            for (i, blob) in &aux {
                let at_cp = match self.config.layer_type(*i as usize) {
                    LayerType::FullAttention | LayerType::SlidingAttention => {
                        let row = self.config.indexer_head_dim * 2;
                        qsa_blob_at(blob, cp, self.config.indexer_compress_ratio, row)
                    }
                    _ => self.ple_blob_at(blob, tokens, cp, ple_src, stream)?,
                };
                match at_cp {
                    Some(b) => rebuilt.push((*i, b)),
                    None => break,
                }
            }
            if rebuilt.len() != aux.len() {
                tracing::warn!(
                    "qwen4_exp in-pass checkpoint at token {cp}: aux cannot be rebuilt; \
                     not registered"
                );
                continue;
            }
            built.push((cp, slot, branch, rebuilt));
        }
        if aux_share_requested() {
            PASS_AUX.with(|p| *p.borrow_mut() = Some(((seq.slot_idx, seq.seq_len), aux)));
        }
        for (cp, slot, branch, rebuilt) in built {
            if !rebuilt.is_empty() {
                self.ssm_snapshots.set_aux(slot, rebuilt);
            }
            left.retain(|&s| s != slot);
            // `false`: the checkpoint freed its slot itself.
            if self.register_intermediate_checkpoint(tokens, seq, kv_cache, cp, slot, branch) {
                tracing::info!(
                    "qwen4_exp mid-chunk SSM checkpoint saved at token {cp} (snapshot_id {slot}{})",
                    if branch { ", branch point" } else { "" }
                );
            }
        }
        Ok(())
    }

    /// The PLE aux blob `[n u32][history][conv]` at `cp`: the last `n` token
    /// ids below `cp`, and the conv carry captured at `cp` into `stage`.
    fn ple_blob_at(
        &self,
        blob: &[u8],
        tokens: &[u32],
        cp: usize,
        stage: DevicePtr,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        if blob.len() < 4 {
            return Ok(None);
        }
        let n = u32::from_le_bytes(blob[..4].try_into().expect("4 bytes")) as usize;
        let conv_bytes = blob.len().saturating_sub(4 + n * 4);
        if cp < n || conv_bytes == 0 || conv_bytes > self.ple_stage_bytes() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(blob.len());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for t in &tokens[cp - n..cp] {
            out.extend_from_slice(&t.to_le_bytes());
        }
        let off = out.len();
        out.resize(off + conv_bytes, 0);
        crate::layers::aux_d2h::copy(self.gpu.as_ref(), stage, &mut out[off..], stream)?;
        Ok(Some(out))
    }

    /// Bytes of one PLE conv carry: `[(k - 1) * dilation, hc_mult * hidden]`
    /// FP32; the dilation is the n-gram neighbour count
    /// (`weight_loader::qwen4_exp::ple`).
    fn ple_stage_bytes(&self) -> usize {
        let c = self.config.hc_mult * self.config.hidden_size;
        let steps = self.config.ple_conv_kernel_size.saturating_sub(1)
            * self.config.emb_neighbor_num.max(1);
        steps.max(1) * c * 4
    }

    /// The PLE conv-carry staging buffer for `points` carries (grown once to
    /// the largest count): (base, bytes a carry).
    fn ple_stage(&self, points: usize, stream: u64) -> Result<(DevicePtr, usize)> {
        let one = self.ple_stage_bytes();
        let bytes = one * points.max(1);
        let (ptr, size) = PLE_STAGE.with(Cell::get);
        if size >= bytes {
            return Ok((DevicePtr(ptr), one));
        }
        if ptr != 0 {
            self.gpu.synchronize(stream)?;
            self.gpu.free(DevicePtr(ptr))?;
        }
        let p = self.gpu.alloc(bytes)?;
        PLE_STAGE.with(|s| s.set((p.0, bytes)));
        Ok((p, one))
    }
}

/// `ATLAS_QWEN4EXP_CKPT_AUX_SHARE=1`.
fn aux_share_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_CKPT_AUX_SHARE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

thread_local! {
    /// The pass-end aux the mid-chunk finalize collected, keyed by
    /// (sequence slot, seq_len after the pass).
    static PASS_AUX: RefCell<Option<PassAux>> = const { RefCell::new(None) };
}

type PassAux = ((usize, usize), Vec<(u32, Vec<u8>)>);

/// Drops any stashed pass-end aux when the chunk's saves are done (or fail),
/// so no later chunk or request can take it.
pub(super) struct PassAuxScope(());

impl PassAuxScope {
    pub(super) fn new() -> Self {
        PASS_AUX.with(|p| p.borrow_mut().take());
        Self(())
    }
}

impl Drop for PassAuxScope {
    fn drop(&mut self) {
        PASS_AUX.with(|p| p.borrow_mut().take());
    }
}

/// Move the pass-end aux collected by this pass's mid-chunk finalize into
/// `out` for a save of the same sequence at the same length: no layer state
/// moves between the two, so these are the bytes `collect_aux_states_into`
/// would read again. `false` (nothing taken; any stale entry dropped)
/// otherwise.
pub(super) fn take_pass_aux(slot: usize, seq_len: usize, out: &mut Vec<(u32, Vec<u8>)>) -> bool {
    match PASS_AUX.with(|p| p.borrow_mut().take()) {
        Some((key, aux)) if key == (slot, seq_len) => {
            *out = aux;
            true
        }
        _ => false,
    }
}

/// The QSA aux blob `[ingested u64][pooled u64][pooled keys][raw tail]` at
/// `cp` (a block boundary): `cp` ingested, `cp / ratio` blocks pooled -- the
/// first rows of the pass-end blob's pooled keys, which a block's key depends
/// on nothing else to form -- and no raw tail. `None` when the pass-end blob
/// does not reach `cp`.
pub(super) fn qsa_blob_at(blob: &[u8], cp: usize, ratio: usize, row: usize) -> Option<Vec<u8>> {
    if blob.len() < 16 || ratio == 0 || !cp.is_multiple_of(ratio) {
        return None;
    }
    let ingested = u64::from_le_bytes(blob[..8].try_into().ok()?) as usize;
    let pooled = u64::from_le_bytes(blob[8..16].try_into().ok()?) as usize;
    let keep = cp / ratio;
    if ingested < cp || pooled < keep || blob.len() < 16 + pooled * row {
        return None;
    }
    let mut out = Vec::with_capacity(16 + keep * row);
    out.extend_from_slice(&(cp as u64).to_le_bytes());
    out.extend_from_slice(&(keep as u64).to_le_bytes());
    out.extend_from_slice(&blob[16..16 + keep * row]);
    Some(out)
}

#[cfg(test)]
#[path = "qwen4exp_ckpt_tests.rs"]
mod tests;
