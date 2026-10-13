// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::types::TransformerModel;

impl TransformerModel {
    /// Copy `rows` target hidden rows, starting at forward row `row0`, into
    /// capture storage whose rows are `dst_stride` bytes apart. mHC targets
    /// (GLM-5) keep the residual in the FP32 highway and `hidden_states` holds
    /// only the next sublayer's mixed input, so they contract the highway by
    /// stream mean — the reference DFlash target hidden — instead.
    fn dflash_capture_rows(
        &self,
        row0: usize,
        rows: usize,
        dst: spark_runtime::gpu::DevicePtr,
        dst_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        if rows == 0 {
            return Ok(());
        }
        let trace = std::env::var("ATLAS_DFLASH_CAPTURE_TRACE").as_deref() == Ok("1");
        let result = self.dflash_capture_rows_inner(row0, rows, dst, dst_stride, stream);
        if trace && result.is_ok() {
            // Debug only: synchronous readback of the first captured row.
            self.gpu.synchronize(stream)?;
            let mut raw = vec![0u8; h * 2];
            self.gpu.copy_d2h(dst, &mut raw)?;
            let norm = raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            tracing::info!(row0, rows, dst_stride, norm, "DFlash capture trace");
        }
        result
    }

    fn dflash_capture_rows_inner(
        &self,
        row0: usize,
        rows: usize,
        dst: spark_runtime::gpu::DevicePtr,
        dst_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let hc = self.config.hc_mult;
        if hc > 1 {
            static KERNEL: std::sync::OnceLock<spark_runtime::gpu::KernelHandle> =
                std::sync::OnceLock::new();
            let kernel = match KERNEL.get() {
                Some(k) => *k,
                None => *KERNEL.get_or_init(|| {
                    self.gpu
                        .kernel(
                            "hyper_connection",
                            &crate::layers::ops::hc_kernel_name(
                                &self.config.model_type,
                                "hc_contract_strided",
                            ),
                        )
                        .unwrap_or(spark_runtime::gpu::KernelHandle(0))
                }),
            };
            anyhow::ensure!(kernel.0 != 0, "DFlash mHC capture kernel unavailable");
            anyhow::ensure!(
                dst_stride.is_multiple_of(2),
                "DFlash capture stride must be BF16-aligned"
            );
            return spark_runtime::kernel_args::KernelLaunch::new(self.gpu.as_ref(), kernel)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buffers.hc_streams().offset(
                    row0 * hc * h * crate::layers::ops::hc_elem_bytes(&self.config.model_type),
                ))
                .arg_ptr(dst)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .arg_u32((dst_stride / 2) as u32)
                .launch(stream);
        }
        let src = self.buffers.hidden_states();
        for t in 0..rows {
            self.gpu.copy_d2d_async(
                src.offset((row0 + t) * h * 2),
                dst.offset(t * dst_stride),
                h * 2,
                stream,
            )?;
        }
        Ok(())
    }

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
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let max_ctx = dstate.max_ctx_len;
        let own_row = dstate.own_row;
        // The accumulator holds the LAST `max_ctx` prompt positions (the
        // drafter's context window), not the first: slot 0 is position
        // `window_start`. Earlier rows of this chunk are dropped.
        let end = chunk_start + proc_count;
        let window_start = seq.tokens.len().max(end).saturating_sub(max_ctx);
        let mut first = window_start.max(chunk_start);
        // A sequence-parallel chunk keeps only this rank's rows (rank 0: the
        // upper half, compacted at row 0) in the highway.
        let sp_row0 = crate::layers::glm_sp::current().map_or(0, |sp| sp.row0);
        first = first.max(chunk_start + sp_row0);
        // Non-legacy `ATLAS_DFLASH_FIRST_APPEND`: keep this pass's last row —
        // after the final chunk, the last prompt position — in the sequence's
        // own spare row, whatever the window below keeps.
        if let Some(own_row) = own_row
            && let Some(row) = crate::layers::dflash_head::own_capture_row(proc_count, sp_row0)
        {
            self.dflash_capture_rows(
                row,
                1,
                own_row.offset(slot_idx * h * bf16),
                n_capture * h * bf16,
                stream,
            )?;
        }
        if first >= end {
            return Ok(());
        }
        self.dflash_capture_rows(
            first - chunk_start - sp_row0,
            end - first,
            acc_base.offset((first - window_start) * n_capture * h * bf16 + slot_idx * h * bf16),
            n_capture * h * bf16,
            stream,
        )
    }

    /// After prefill completes, advance the seq's DFlash `ctx_len` to
    /// `chunk_start + proc_count` so the drafter sees all captured prompt
    /// positions on the first propose() call.
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
            dstate.seed_prefill_ctx(seq.tokens.len(), chunk_start + proc_count, seq.seq_len);
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
        self.dflash_capture_rows(token_idx, 1, dst.offset(slot * h * bf16), h * bf16, stream)
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
        self.dflash_capture_rows(
            0,
            k.min(kmax),
            dst.offset(slot * h * bf16),
            ctx_slot_bytes,
            stream,
        )
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
        for (i, &k) in ks.iter().enumerate() {
            let region = slots.map(|s| s[i]).unwrap_or(i);
            anyhow::ensure!(
                region < nseq,
                "DFlash batched capture owner slot {region} exceeds capacity {nseq}"
            );
            let seq_base = dst.offset(region * kmax * ctx_slot_bytes);
            self.dflash_capture_rows(
                off[i],
                k.min(kmax),
                seq_base.offset(slot * h * bf16),
                ctx_slot_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Preserve the current C=1 front before any slot-addressed batch region is
    /// packed over it. The preserve area is the extra sequence-sized region
    /// allocated after all owner slots.
    pub(super) fn preserve_dflash_save_front(&self, k: usize, _stream: u64) -> Result<()> {
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
        // default_stream, not the caller arg: the front is written by the
        // capture and read by commit_ctx on the model's (non-blocking)
        // default stream. The scheduler boundary passes the legacy 0 —
        // honoring it would leave this copy unordered with both sides.
        let stream = self.gpu.default_stream();
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
    pub(super) fn pack_dflash_save_region(
        &self,
        slot: usize,
        k: usize,
        _stream: u64,
    ) -> Result<()> {
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
        // See preserve_dflash_save_front: must be ordered with the capture
        // (writer) and commit_ctx (reader), both on the default stream.
        let stream = self.gpu.default_stream();
        let src = dst.offset(slot * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(src, dst, n * ctx_slot_bytes, stream)
    }

    /// Restore the original C=1 front after the batched commit loop packed
    /// owner regions over it. The scheduler calls this only when owner slot 0
    /// was present and therefore was explicitly preserved before the loop.
    pub(super) fn restore_dflash_save_front(&self, k: usize, _stream: u64) -> Result<()> {
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
        // See preserve_dflash_save_front: must be ordered with the capture
        // (writer) and commit_ctx (reader), both on the default stream.
        let stream = self.gpu.default_stream();
        let preserve = dst.offset(self.dflash_hidden_save_nseq * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(preserve, dst, n * ctx_slot_bytes, stream)
    }
}
