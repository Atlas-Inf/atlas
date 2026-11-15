// SPDX-License-Identifier: AGPL-3.0-only

//! A prefill pass run in slices of whole layers
//! ([`Model::prefill_chunk_slice`](crate::traits::Model::prefill_chunk_slice),
//! the scheduler's `ATLAS_GLM_PREFILL_YIELD_MS`), on the GLM pair.
//!
//! A chunk of up to 8K rows holds both GPUs for its whole forward (2.9 s),
//! and every decoding sequence waits. Cutting the chunk shorter changes the
//! pass shape and so the bits (`pc_policy`, "Accumulation order"). A slice
//! does not: the pass runs its layers over the same rows with the same
//! kernels in the same order, and stops between two of them so the caller
//! can run decode steps of other sequences before the next call continues.
//!
//! Between layers a pass keeps its state in three places a decode step also
//! writes, and the pause puts each back:
//!
//! * the mHC highway rows (`hc_streams`): a verify writes its own rows from
//!   row 0, at most [`STASH_ROWS`] of them, so those rows are copied out at
//!   the pause and back at the resume (with the chunk's first token ids);
//! * the positions and slots in `scratch` (`upload_meta`, `upload_paged`),
//!   which the resume writes again from the same inputs;
//! * the KV-cache mutex, released by the unwind and taken again.
//!
//! Everything else a decode step writes in the arena (`hidden`, the norm,
//! attention, KDA and MoE scratch, the logits) a prefill layer writes before
//! it reads. The sequence-parallel scope and the per-layer capture scopes are
//! thread-locals that drop with the unwind and are entered again.
//!
//! The pause travels up as [`PassPaused`], an error only
//! `prefill_chunk_slice` sees, so every caller between (the tail-checkpoint
//! and branch-checkpoint splits, the entry's after-chunk hooks) stops where
//! it stood and the same chunk call walks back down the same path: a split
//! runs its first pass again, which finds its paused state and continues.
//!
//! Both ranks pause at the same layer: the worker runs the layer count the
//! head sent ([`EP_CMD_PREFILL_SLICE`](crate::traits::EP_CMD_PREFILL_SLICE)),
//! and whether a pass can pause at all depends only on what the ranks share
//! (the model, the pass's row count, a capture plan). A pass under
//! [`MIN_ROWS`] rows runs whole: it is short, and its rows would not cover
//! the rows a verify leaves in the per-layer scratch.

use std::cell::Cell;

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::{PausedPass, SequenceState};

/// Highway rows saved across a pause: the most rows any decode or verify
/// step writes, the owner-batched verify's
/// (`layer::glm_long_owner::MAX_ROWS`) or the wide batched verify's
/// (`verify_e2::VERIFY_ROW_CAP`).
pub(super) const STASH_ROWS: usize = {
    let (owner, wide) = (
        crate::layer::glm_long_owner::MAX_ROWS,
        super::super::verify_e2::VERIFY_ROW_CAP,
    );
    if owner > wide { owner } else { wide }
};

/// The fewest computed rows of a pass that may pause.
pub(super) const MIN_ROWS: usize = 1024;

/// A pass stopped between layers. Not a failure.
#[derive(Debug)]
pub(in crate::model) struct PassPaused;

impl std::fmt::Display for PassPaused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("prefill pass paused between layers")
    }
}

impl std::error::Error for PassPaused {}

thread_local! {
    /// Layers the current chunk call may still run (`usize::MAX`: no limit).
    static LEFT: Cell<usize> = const { Cell::new(usize::MAX) };
    /// EP worker: the layer count the head sent for the next chunk command.
    static NEXT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The layer limit of one chunk call; lifts it when the call ends.
pub(in crate::model) struct Budget(());

impl Drop for Budget {
    fn drop(&mut self) {
        LEFT.with(|c| c.set(usize::MAX));
    }
}

/// Limit this thread's chunk call to `layers` more layers.
pub(in crate::model) fn begin(layers: Option<usize>) -> Budget {
    LEFT.with(|c| c.set(layers.unwrap_or(usize::MAX)));
    Budget(())
}

/// Count one layer; true when the call may run no more.
pub(super) fn layer_done() -> bool {
    LEFT.with(|c| {
        let left = match c.get() {
            usize::MAX => return false,
            left => left.saturating_sub(1),
        };
        c.set(left);
        left == 0
    })
}

/// EP worker: the head's slice word for the next chunk command.
pub(in crate::model) fn worker_set(layers: usize) {
    NEXT.with(|c| c.set(Some(layers)));
}

/// EP worker: the layer count of this chunk command, if the head sent one.
pub(in crate::model) fn worker_take() -> Option<usize> {
    NEXT.with(Cell::take)
}

/// Where a chunk call stands against its sequence's paused pass.
pub(super) enum Standing {
    /// Nothing is paused: the call starts a pass.
    Fresh,
    /// The call is the paused pass: continue it.
    Here(Box<PausedPass>),
    /// The paused pass is the first of a branch-checkpoint split of this
    /// chunk, which cut it at this token.
    Below(usize),
}

/// What the layer loop hands back when it stops early.
pub(super) struct Stopped {
    pub next_layer: usize,
    pub highway: Vec<u8>,
    pub token_ids: Vec<u8>,
}

impl TransformerModel {
    /// Whether a pass may stop between layers on this model: the GLM pair.
    pub(in crate::model) fn pass_slices(&self) -> bool {
        let c = &self.config;
        c.model_type == "glm5_next"
            && c.hc_mult > 0
            && c.tp_world_size == 2
            && c.ep_world_size == 2
            && self.multi_rank_protocol_active()
    }

    /// Stop before `next_layer`: copy the rows a decode step may write out
    /// of the arena, after everything the pass has launched ran. `rows`: the
    /// highway rows this rank holds; `ids`: the chunk's token ids.
    pub(super) fn pass_stash(
        &self,
        next_layer: usize,
        rows: usize,
        hc_row: usize,
        ids: usize,
        stream: u64,
    ) -> Result<Stopped> {
        self.gpu.synchronize(stream)?;
        let mut highway = vec![0u8; rows.min(STASH_ROWS) * hc_row];
        self.gpu
            .copy_d2h_on_stream(self.buffers.hc_streams(), &mut highway, stream)?;
        let mut token_ids = vec![0u8; ids.min(STASH_ROWS) * 4];
        self.gpu
            .copy_d2h_on_stream(self.buffers.token_ids(), &mut token_ids, stream)?;
        Ok(Stopped {
            next_layer,
            highway,
            token_ids,
        })
    }

    /// Put them back, after everything launched since ran.
    fn pass_unstash(&self, p: &PausedPass, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)?;
        self.gpu.copy_h2d(&p.highway, self.buffers.hc_streams())?;
        self.gpu.copy_h2d(&p.token_ids, self.buffers.token_ids())
    }

    /// Record a stopped pass on `seq` and raise [`PassPaused`]; `Ok` for a
    /// pass that ran its last layer.
    pub(super) fn pass_settle(
        seq: &mut SequenceState,
        stopped: Option<Stopped>,
        pass: impl FnOnce(Stopped) -> PausedPass,
    ) -> Result<()> {
        let Some(stopped) = stopped else {
            return Ok(());
        };
        seq.paused_pass = Some(Box::new(pass(stopped)));
        Err(PassPaused.into())
    }

    /// Continue the paused pass `p` of `seq` from its next layer: the arena
    /// rows and metadata as the pause left them, then the rest of the pass
    /// exactly as `prefill_chunk_dispatch_with` runs it.
    pub(super) fn pass_resume(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        p: Box<PausedPass>,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let began = std::time::Instant::now();
        let (chunk_start, chunk_len) = p.span;
        let (proc_start, proc_count) = p.proc;
        let mut kv_cache = self.kv_cache.lock();
        self.buffers.note_rows(proc_count);
        let _det = crate::det_trace::enter(self.config.ep_rank, seq.slot_idx, proc_start);
        let meta = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            0,
            p.effective_seq_len_start,
            &kv_cache,
            stream,
        )?;
        // The slots are part of that metadata (a kernel fills them from the
        // sequence's block table, which the pause did not touch).
        if meta.needs_paged {
            self.prefill_b_upload_paged(
                seq,
                tokens.len(),
                proc_start,
                proc_count,
                meta.meta_base,
                meta.slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        self.pass_unstash(&p, stream)?;
        let stopped = self.prefill_b_forward_layers(
            seq,
            tokens,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            p.effective_seq_len_start,
            p.kv_write_start,
            p.marconi_skip,
            meta.meta_base,
            meta.slot_offset,
            meta.pos_stream_bytes,
            meta.use_mrope,
            meta.needs_paged,
            None,
            None,
            stream,
            p.next_layer,
        )?;
        Self::pass_settle(seq, stopped, |s| PausedPass {
            next_layer: s.next_layer,
            highway: s.highway,
            token_ids: s.token_ids,
            ..*p
        })?;
        self.warm_trace_sync(stream)?;
        let trace = (began, [std::time::Duration::ZERO; 2], [began.elapsed(); 5]);
        let span = (chunk_start, chunk_len, proc_start, proc_count);
        self.pass_finish(
            tokens,
            seq,
            &mut kv_cache,
            span,
            is_last_chunk,
            &None,
            trace,
            stream,
        )
    }

    /// Phases 5 to 9 of a pass whose layers all ran: the sequence takes its
    /// tokens, then the last chunk's norm, head and snapshots or an
    /// intermediate checkpoint. `span`: chunk start and length, then the
    /// computed rows' first token and count.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pass_finish(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        (chunk_start, chunk_len, proc_start, proc_count): (usize, usize, usize, usize),
        is_last_chunk: bool,
        midcap_plan: &Option<super::midchunk_capture::MidCapturePlan>,
        (began, pre, marks): (
            std::time::Instant,
            [std::time::Duration; 2],
            [std::time::Duration; 5],
        ),
        stream: u64,
    ) -> Result<DevicePtr> {
        let bs = kv_cache.block_size();
        // ── Phase 5: update sequence state incrementally ──
        // Always add chunk tokens exactly once. The early-return path for
        // fully cached non-last chunks doesn't add tokens, so this is the
        // single insertion point for all chunks that reach here.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // #155: prime the decode-checkpoint cadence gate; the last chunk
        // leaves it at the prompt's complete-block count (see prefill_a).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // ── Legacy echo+logprobs: project prompt positions while this
        // chunk's hidden rows are live, BEFORE finalize_last (which
        // re-derives norm_output + logits for the first sampled token).
        // No-op unless seq.collect_prompt_logprobs is set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        let _pass_aux = self.qwen4exp_ckpt_saves(tokens, seq, kv_cache, midcap_plan, stream)?;
        let out = if is_last_chunk {
            // ── Phase 6+7+8: final norm, lm_head, prefix-cache + snapshot save ──
            self.prefill_b_finalize_last(
                tokens,
                seq,
                kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )?
        } else {
            // ── Phase 9: intermediate Marconi checkpoint ──
            self.prefill_b_save_checkpoint(tokens, seq, kv_cache, chunk_start, chunk_len, stream)?;
            DevicePtr::NULL
        };
        let (chunk, logits) = ((chunk_start, proc_count), is_last_chunk.then_some(out));
        self.warm_trace_chunk(seq, tokens.len(), began, chunk, pre, marks, logits, stream)?;
        Ok(out)
    }

    /// Where the chunk call `[chunk_start, +chunk_len)` stands against
    /// `seq`'s paused pass (taken out of `seq` when the call continues it).
    pub(super) fn pass_standing(
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Standing> {
        let Some(span) = seq.paused_pass.as_ref().map(|p| p.span) else {
            return Ok(Standing::Fresh);
        };
        if let Some(p) = seq
            .paused_pass
            .take_if(|_| span == (chunk_start, chunk_len))
        {
            return Ok(Standing::Here(p));
        }
        anyhow::ensure!(
            span.0 == chunk_start && span.1 < chunk_len,
            "prefill pass {}+{} is paused, but the chunk call is {chunk_start}+{chunk_len}",
            span.0,
            span.1,
        );
        Ok(Standing::Below(span.0 + span.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_limit_no_layer_ends_the_call() {
        for _ in 0..100 {
            assert!(!layer_done());
        }
    }

    #[test]
    fn a_limit_ends_the_call_at_its_last_layer_and_lifts_when_the_call_ends() {
        {
            let _b = begin(Some(3));
            assert!(!layer_done());
            assert!(!layer_done());
            assert!(layer_done());
            // A second pass of the same call has nothing left.
            assert!(layer_done());
        }
        assert!(!layer_done());
        let _b = begin(None);
        assert!(!layer_done());
    }

    #[test]
    fn the_worker_uses_a_slice_word_once() {
        assert_eq!(worker_take(), None);
        worker_set(4);
        assert_eq!(worker_take(), Some(4));
        assert_eq!(worker_take(), None);
    }

    fn paused(span: (usize, usize)) -> SequenceState {
        let mut seq = SequenceState::host_only(0);
        seq.paused_pass = Some(Box::new(PausedPass {
            span,
            next_layer: 7,
            proc: span,
            effective_seq_len_start: span.0,
            kv_write_start: 0,
            marconi_skip: false,
            highway: vec![1, 2, 3],
            token_ids: vec![4],
        }));
        seq
    }

    fn standing(seq: &mut SequenceState, start: usize, len: usize) -> Result<Standing> {
        TransformerModel::pass_standing(seq, start, len)
    }

    #[test]
    fn the_same_chunk_call_continues_its_pass() {
        let mut seq = paused((8192, 8192));
        let Standing::Here(p) = standing(&mut seq, 8192, 8192).unwrap() else {
            panic!("the paused pass itself");
        };
        assert_eq!(p.next_layer, 7);
        assert!(seq.paused_pass.is_none());
        // Nothing paused: an ordinary call.
        assert!(matches!(
            standing(&mut seq, 8192, 8192),
            Ok(Standing::Fresh)
        ));
    }

    #[test]
    fn a_split_chunk_reaches_its_paused_first_pass_through_the_split() {
        let mut seq = paused((0, 3440));
        assert!(matches!(
            standing(&mut seq, 0, 8192),
            Ok(Standing::Below(3440))
        ));
        assert!(seq.paused_pass.is_some());
        assert!(matches!(standing(&mut seq, 0, 3440), Ok(Standing::Here(_))));
    }

    #[test]
    fn a_call_for_other_rows_is_refused() {
        let mut seq = paused((8192, 8192));
        assert!(standing(&mut seq, 16384, 8192).is_err());
        assert!(standing(&mut seq, 8192, 4096).is_err());
        assert!(seq.paused_pass.is_some());
    }
}
