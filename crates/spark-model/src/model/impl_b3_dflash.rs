// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::types::TransformerModel;

impl TransformerModel {
    /// DFlash prefill capture: copy `proc_count` tokens × hidden_size BF16
    /// from `self.buffers.hidden_states()` (filled by the just-completed
    /// prefill layer) into the per-sequence DFlash accumulator. Called
    /// inside the prefill layer loop after each layer. No-op when:
    ///   - DFlash is disabled (capture_layers empty)
    ///   - `layer_idx` is not in `dflash_capture_layers`
    ///   - The seq has no `DflashProposerState`
    ///   - Rank > 0 under EP/TP (drafter is rank-0 only)
    ///
    /// Layout: writes `hidden[t]` BF16 into
    /// `acc[(chunk_start + t) * 5 * h + slot_idx * h]` for each t.
    /// Per-layer call performs `proc_count` strided d2d_async copies —
    /// at typical prefill of 128–4096 tokens × 5 capture layers, total
    /// 640–20480 launches per prefill. Acceptable launch overhead for
    /// first land; replace with a strided-scatter kernel if profiling
    /// shows it's a bottleneck.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let dstate = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        if dstate.max_ctx_len == 0 {
            return Ok(()); // ctx conditioning disabled
        }
        // ATLAS_DFLASH_CTX_CARRY + first-chunk bookkeeping. Adopt BEFORE the
        // first capture write (rather than in
        // update_dflash_ctx_len_after_prefill, which the chunked path calls
        // only after the LAST chunk) so that:
        //   1. The carried accumulator install happens before any writes —
        //      captures already in the fresh buffer would be orphaned when
        //      adopt frees it.
        //   2. Captures append AFTER the carried window — no row is ever
        //      claimed by two positions.
        //   3. `ctx_prefill_origin`/`ctx_prefill_base` snapshot where the
        //      window's own coverage begins, so every chunk's capture rows
        //      derive as base + (chunk_start - origin) even though ctx_len
        //      only advances at the last chunk's update.
        if dstate.ctx_prefill_origin.is_none() {
            if let Some(ref proposer) = self.proposer {
                proposer.adopt_dflash_ctx(self.gpu.as_ref(), dstate, &seq.tokens, chunk_start);
            }
            dstate.ctx_prefill_origin = Some(chunk_start);
            dstate.ctx_prefill_base = dstate.ctx_len;
        }
        // Lazy accumulator: adopt may have installed the carried buffer
        // (ctx_hidden_acc != 0 → no-op); a rejected carry just pooled its
        // buffer, which this pops right back — either way zero device
        // allocations on the warm path.
        if dstate.ctx_hidden_acc.0 == 0
            && let Some(ref proposer) = self.proposer
        {
            proposer.acquire_ctx_acc(self.gpu.as_ref(), dstate)?;
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let acc_rows = dstate.ctx_acc_rows;
        let src_base = self.buffers.hidden_states();
        // Dense-window append: the ctx accumulator is a compacted array of
        // rows [0..ctx_len) with per-row absolute positions in
        // ctx_positions — NOT a position-indexed buffer. Carried rows sit in
        // [0..ctx_prefill_base); this prefill's captures fill rows
        // base + (chunk_start - origin) .. with positions chunk_start.. .
        let cursor = dstate.ctx_prefill_base.saturating_add(
            chunk_start.saturating_sub(dstate.ctx_prefill_origin.unwrap_or(chunk_start)),
        );
        for t in 0..proc_count {
            let row = cursor + t;
            if row >= acc_rows {
                break; // accumulator capacity; retention is enforced by the slide
            }
            let src = src_base.offset(t * h * bf16);
            let dst_offset = row * n_capture * h * bf16 + slot_idx * h * bf16;
            self.gpu
                .copy_d2d_async(src, acc_base.offset(dst_offset), h * bf16, stream)?;
        }
        Ok(())
    }

    /// After prefill completes (last chunk on the chunked path), set the
    /// seq's DFlash `ctx_len` to the full captured window — carried rows
    /// plus every position this prefill covered — and slide to the newest
    /// half-window if the window exceeds the retention cap.
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            if dstate.max_ctx_len == 0 {
                return Ok(());
            }
            let chunk_end = chunk_start + proc_count;
            // Pure/idempotent: rows [0..base) are the adopted carry (their
            // carried positions stay), rows [base..ctx_len) are this
            // prefill's captures at positions origin..chunk_end — same
            // layout the capture cursor wrote (base + chunk_start - origin).
            // A repeat call recomputes the same result.
            let base = dstate.ctx_prefill_base;
            let origin = dstate.ctx_prefill_origin.unwrap_or(chunk_start);
            dstate.ctx_len = base.saturating_add(chunk_end.saturating_sub(origin));
            dstate.ctx_positions.truncate(base);
            dstate
                .ctx_positions
                .extend((origin..chunk_end).map(|i| i as i32));
            // Retention slide — same keep-NEWEST-half policy as
            // commit_ctx/dflash_serial_ctx_append: drop_n >= keep so the
            // single D2D copy's src/dst ranges can never overlap. Absolute
            // positions survive the drain; committed K/V is re-precomputed
            // chunk-wise on the next propose.
            if dstate.ctx_len > dstate.max_ctx_len {
                let keep = dstate.max_ctx_len / 2;
                let drop_n = dstate.ctx_len - keep;
                let slot = dstate.ctx_slot_bytes;
                let src = dstate.ctx_hidden_acc.offset(drop_n * slot);
                let stream = self.gpu.default_stream();
                self.gpu
                    .copy_d2d_async(src, dstate.ctx_hidden_acc, keep * slot, stream)?;
                dstate.ctx_positions.drain(..drop_n);
                dstate.ctx_len = keep;
                dstate.ctx_committed = 0;
                tracing::info!(
                    "DFlash ctx prefill watermark: slid ctx window \
                     (dropped {drop_n} oldest, keep {keep})",
                );
            }
        }
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Captures only the latest-decoded-token's hidden, matching the
    /// `save_hidden_for_mtp` semantics. The `token_idx` argument selects
    /// which row of `self.buffers.hidden_states()` to read — pass 0 for the
    /// single-token decode path.
    ///
    /// Under EP/TP world > 1: only rank 0 owns the drafter (replicated, not
    /// sharded — same pattern as MTP under EP — see model.rs:7232 comment),
    /// so non-rank-0 ranks skip the capture. The captured hiddens are
    /// post-TP-allreduce so semantically correct on rank 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors save_hidden_for_mtp's effective behavior).
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        // The residual stream is always BF16, so DFlash hidden capture
        // copies BF16 bytes directly with no downcast.
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst_slot = dst.offset(slot * h * bf16);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        Ok(())
    }

    /// Capture `hidden_states[token_idx]` for every DFlash capture layer into
    /// `dflash_hidden_save`. Called from `verify_dflash_step` after the Phase 3
    /// D2H sync, so `token_idx` is the confirmed bonus position. Runs outside
    /// the CUDA graph so the correct accept-prefix position can be used.
    pub(super) fn save_dflash_hidden_dispatch(&self, token_idx: usize, stream: u64) -> Result<()> {
        for &layer_idx in &self.dflash_capture_layers {
            self.try_dflash_capture(layer_idx, token_idx, stream)?;
        }
        Ok(())
    }

    /// K=gamma EAGLE capture: copy the per-layer hidden of ALL `k` verify rows into
    /// the row-major `dflash_hidden_save` ([row0 | row1 | ... ], each row =
    /// n_capture * hidden_size * bf16). Called once per capture layer inside the
    /// verify graph (k is fixed per captured graph). After verify, the scheduler
    /// appends rows 0..=num_accepted to ctx so every committed position gets its
    /// target hidden (fixes the ctx-undercount) and the bonus generator (row
    /// num_accepted) is the freshest slot (EAGLE). No-op unless DFlash is on,
    /// this layer is a capture layer, and rank 0.
    pub(super) fn try_dflash_capture_all(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        debug_assert!(
            k <= kmax,
            "try_dflash_capture_all: k={k} exceeds dflash_hidden_save_rows={kmax}"
        );
        let k_capped = k.min(kmax);
        for t in 0..k_capped {
            let src = self.buffers.hidden_states().offset(t * h * bf16);
            let dst_slot = dst.offset(t * ctx_slot_bytes + slot * h * bf16);
            self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        }
        Ok(())
    }

    /// Batched UNIFIED_CTX capture: seq i owns rows
    /// `[i * kmax, i * kmax + ks[i])` of `dflash_hidden_save`.
    pub(super) fn try_dflash_capture_batched(
        &self,
        layer_idx: usize,
        ks: &[usize],
        off: &[usize],
        stream: u64,
    ) -> Result<()> {
        Self::try_dflash_capture_batched_at(self, layer_idx, ks, off, None, stream)
    }

    /// Slot-indexed variant: `slots[i]` is sequence i's STABLE SSM slot. The
    /// capture region is `slots[i] * kmax`, not the batch position — after
    /// any mid-batch finish (churn/EOS/max_tokens) the pending set reorders,
    /// and a batch-position region would hand the re-propose another (or a
    /// dead) sequence's hiddens, poisoning the drafter ctx accumulator (the
    /// release-binary churn ILA; see the wave-12 session evidence). `None`
    /// keeps the historical batch-position layout for the single-seq path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_dflash_capture_batched_at(
        &self,
        layer_idx: usize,
        ks: &[usize],
        off: &[usize],
        slots: Option<&[usize]>,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        let nseq = self.dflash_hidden_save_nseq;
        if let Some(slots) = slots {
            anyhow::ensure!(
                slots.len() == ks.len(),
                "DFlash batched capture owner-slot width {} != batch width {}",
                slots.len(),
                ks.len()
            );
        }
        let hidden = self.buffers.hidden_states();
        for (i, &k) in ks.iter().enumerate() {
            let region = slots.map(|s| s[i]).unwrap_or(i);
            anyhow::ensure!(
                region < nseq,
                "DFlash batched capture owner slot {region} exceeds capacity {nseq}"
            );
            let seq_base = dst.offset(region * kmax * ctx_slot_bytes);
            for t in 0..k.min(kmax) {
                let src = hidden.offset((off[i] + t) * h * bf16);
                let dst_slot = seq_base.offset(t * ctx_slot_bytes + slot * h * bf16);
                self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
            }
        }
        Ok(())
    }

    /// Preserve the current C=1 front before any slot-addressed batch region is
    /// packed over it. The preserve area is the extra sequence-sized region
    /// allocated after all owner slots.
    pub(super) fn preserve_dflash_save_front(&self, k: usize, stream: u64) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let ctx_slot_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        let n = k.min(kmax);
        if n == 0 {
            return Ok(());
        }
        let preserve = dst.offset(self.dflash_hidden_save_nseq * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(dst, preserve, n * ctx_slot_bytes, stream)
    }

    /// Compact a slot-addressed capture region to the C=1 front for the
    /// commit_ctx reader. The slot-0 region already is the front; its original
    /// rows are preserved once before the batch and restored after the loop.
    pub(super) fn pack_dflash_save_seq(&self, seq_i: usize, k: usize, stream: u64) -> Result<()> {
        Self::pack_dflash_save_region(self, seq_i, k, stream)
    }

    /// Slot-addressed pack: copies the sequence's slot-indexed capture
    /// region to the front for the commit_ctx reader.
    pub(super) fn pack_dflash_save_region(&self, slot: usize, k: usize, stream: u64) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        anyhow::ensure!(
            slot < self.dflash_hidden_save_nseq,
            "DFlash hidden-save owner slot {slot} exceeds capacity {}",
            self.dflash_hidden_save_nseq
        );
        let n = k.min(kmax);
        if slot == 0 || n == 0 {
            return Ok(());
        }
        let src = dst.offset(slot * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(src, dst, n * ctx_slot_bytes, stream)
    }

    /// Restore the original C=1 front after the batched commit loop packed
    /// owner regions over it. The scheduler calls this only when owner slot 0
    /// was present and therefore was explicitly preserved before the loop.
    pub(super) fn restore_dflash_save_front(&self, k: usize, stream: u64) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let ctx_slot_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        let n = k.min(kmax);
        if n == 0 {
            return Ok(());
        }
        let preserve = dst.offset(self.dflash_hidden_save_nseq * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(preserve, dst, n * ctx_slot_bytes, stream)
    }
}
