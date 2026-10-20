// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::dflash_ctx_window;
use super::types::TransformerModel;

impl TransformerModel {
    /// Copy `rows` target hidden rows, starting at forward row `row0`, into
    /// capture storage whose rows are `dst_stride` bytes apart. mHC targets
    /// (GLM-5) keep the residual in the FP32 highway and `hidden_states` holds
    /// only the next sublayer's mixed input, so they contract the highway by
    /// stream mean — the reference DFlash target hidden — instead.
    pub(super) fn dflash_capture_rows(
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
                    // Only glm-5.3-flash's hyper_connection.cu carries the
                    // strided contract; probing it for another mHC model
                    // leaves a permanently-failed row in the boot audit.
                    crate::layers::try_kernel_gated(
                        self.config.model_type == "glm5_next",
                        self.gpu.as_ref(),
                        "hyper_connection",
                        &crate::layers::ops::hc_kernel_name(
                            &self.config.model_type,
                            "hc_contract_strided",
                        ),
                    )
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
        if dstate.max_ctx_len == 0 {
            return Ok(()); // ctx conditioning disabled
        }
        if self.config.model_type == "glm5_next" {
            if dstate.ctx_hidden_acc.0 == 0
                && let Some(ref proposer) = self.proposer
            {
                proposer.acquire_ctx_acc(self.gpu.as_ref(), dstate)?;
            }
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
            return self.dflash_capture_rows(
                first - chunk_start - sp_row0,
                end - first,
                acc_base
                    .offset((first - window_start) * n_capture * h * bf16 + slot_idx * h * bf16),
                n_capture * h * bf16,
                stream,
            );
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
        // First-append policy is shared by every target, independently of
        // the compacted/carry window's retention policy below.
        if let Some(own_row) = dstate.own_row
            && let Some(row) = crate::layers::dflash_head::own_capture_row(proc_count, 0)
        {
            self.dflash_capture_rows(
                row,
                1,
                own_row.offset(slot_idx * h * bf16),
                n_capture * h * bf16,
                stream,
            )?;
        }
        // Dense-window append: the ctx accumulator is a compacted array of
        // rows [0..ctx_len) with per-row absolute positions in
        // ctx_positions — NOT a position-indexed buffer. Carried rows sit in
        // [0..ctx_prefill_base); captures land at base + (chunk_start -
        // origin). Overflow → plan_capture slides to the newest
        // max_ctx_len/2 rows; a chunk bigger than keep gets move_rows = 0
        // and its leading `skip` tokens drop (the old inline code truncated
        // at acc_rows, then the NEXT chunk's slide read ~200 MB past the
        // accumulator — the GB10 crash).
        let plan = dflash_ctx_window::plan_capture(
            dstate.ctx_prefill_base,
            dstate.ctx_prefill_origin.unwrap_or(chunk_start),
            chunk_start,
            proc_count,
            acc_rows,
            dstate.max_ctx_len,
        );
        if let Some(slide) = plan.slide {
            let slot = dstate.ctx_slot_bytes;
            if slide.move_rows > 0 {
                self.gpu.copy_d2d_async(
                    acc_base.offset(slide.drop_n * slot),
                    acc_base,
                    slide.move_rows * slot,
                    stream,
                )?;
            }
            // The plan advanced ctx_prefill_origin past the dropped captures
            // so positions stay aligned with the rows.
            let pos = slide.drop_n.min(dstate.ctx_positions.len());
            dstate.ctx_positions.drain(..pos);
            dstate.ctx_prefill_base = plan.base;
            dstate.ctx_prefill_origin = Some(plan.origin);
            // Rows moved; their drafter-KV slot mapping is stale.
            dstate.ctx_committed = 0;
            let keep = dstate.max_ctx_len / 2;
            let skipped = if plan.skip > 0 {
                format!(", skipped {} chunk tokens", plan.skip)
            } else {
                String::new()
            };
            tracing::info!(
                "DFlash ctx prefill slide: dropped {} oldest rows{skipped} (mid-prefill overflow, keep {keep})",
                slide.drop_n,
            );
        }
        for t in plan.skip..proc_count {
            let row = plan.first_row + (t - plan.skip);
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
            if self.config.model_type == "glm5_next" {
                dstate.seed_prefill_ctx(seq.tokens.len(), chunk_start + proc_count, seq.seq_len);
                return Ok(());
            }
            if dstate.max_ctx_len == 0 {
                return Ok(());
            }
            dstate.first_append_at = Some(seq.seq_len);
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
                // Bound by PHYSICAL rows: the buffer is sized to
                // ctx_acc_rows (== max_ctx_len) and an all-cached-chunks
                // turn can push formula ctx_len one row past it.
                let phys = dstate.ctx_len.min(dstate.ctx_acc_rows);
                let drop_n = phys - keep;
                let slot = dstate.ctx_slot_bytes;
                let src = dstate.ctx_hidden_acc.offset(drop_n * slot);
                let stream = self.gpu.default_stream();
                self.gpu
                    .copy_d2d_async(src, dstate.ctx_hidden_acc, keep * slot, stream)?;
                let pos_drop = drop_n.min(dstate.ctx_positions.len());
                dstate.ctx_positions.drain(..pos_drop);
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
}
