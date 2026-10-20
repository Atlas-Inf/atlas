// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE pipeline stages every forward shares, and the multi-sequence
//! forward that runs the row-independent ones once for a whole batch.
//!
//! A batched MTP verify (and the multi-sequence decode) used to run the
//! whole PLE forward once per sequence: an n-gram cache resolve with its own
//! NVMe fault batch, a wait on the previous gather, a slot upload, a gather,
//! and both projection GEMMs (65 MB of BF16 weights each time). At C=8 the
//! faults alone were a serial 1-10 ms per sequence with novel text, so the
//! GPU sat idle through up to eight of them a step. Of the pipeline only the
//! gate/conv/add read a sequence's own carry; the resolve, the gather and the
//! projections are per-row functions of the row ids, so [`PleLayer::forward_seqs`]
//! runs them once over every sequence's rows (one fault batch at full queue
//! depth, one gather, two GEMMs) and keeps the per-sequence stages per
//! sequence. Each row reads the same table bytes and the GEMM computes each
//! output row from its own input row alone, so every row's injection is the
//! per-sequence forward's.
//!
//! Kill switch `ATLAS_PLE_SEQ_BATCH=0` restores the per-sequence loop.

use super::*;

/// One sequence's share of a multi-sequence PLE forward.
pub struct PleSeqRows<'a> {
    pub st: &'a mut PleSeqState,
    /// First highway row of this sequence; rows are contiguous and seq-major.
    pub row0: usize,
    /// This sequence's token ids, one per row.
    pub ids: &'a [u32],
}

/// `ATLAS_PLE_SEQ_BATCH=0` (read once): per-sequence PLE forwards.
fn seq_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_PLE_SEQ_BATCH").ok().as_deref() != Some("0"))
}

impl PleLayer {
    /// Advance `st` over `tokens` and return their row ids, `[token][head]`
    /// row-major: history ++ tokens hashed together, keeping the new tokens'
    /// rows (the reference's `[:, -input_ids.shape[1]:]`). The pre-window
    /// history and the window are kept so `rollback_verify` can rebuild the
    /// history for whatever prefix the target accepts — it is a fixed-width
    /// window, so it cannot simply be truncated back.
    pub(super) fn stage_window(&self, st: &mut PleSeqState, tokens: &[u32]) -> Vec<u64> {
        st.history_ckpt = st.history.clone();
        st.verify_tokens = tokens.to_vec();
        let mut window = st.history.clone();
        window.extend_from_slice(tokens);
        let all = ple_ngram_ids(&self.dims, &window);
        let flat = all[all.len() - tokens.len()..]
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        // Carry the last `context_len` tokens for the next step.
        let keep = self.dims.context_len();
        st.history = window[window.len() - keep..].to_vec();
        flat
    }

    /// Key and value projections of scratch rows `[0, n)` of `emb`.
    ///
    /// `dense_gemm_bf16_pipelined`, NOT `dense_gemm`: the ops wrapper and the
    /// kernel are a PAIR. `dense_gemm` launches grid [ceil(n,16), ceil(m,16)]
    /// block 16x16 for the scalar kernel, while the pipelined one wants
    /// [ceil(n,128), ceil(m,128)] block 256. Handing the pipelined kernel to
    /// the scalar launcher reads far out of bounds and produced NaN through
    /// the whole highway.
    pub(super) fn project(&self, n: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        let h = self.hidden as u32;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.key_proj,
            self.key,
            n as u32,
            c as u32,
            h,
            stream,
        )
        .context("PLE key_proj")?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.value_proj,
            self.value,
            n as u32,
            h,
            h,
            stream,
        )
        .context("PLE value_proj")
    }

    /// Gate, conv and highway add for `n` rows of ONE sequence: projections at
    /// scratch row `srow`, highway rows at `hspan`. `width` is the sequence's
    /// whole forward width. `at = (srow, base)`: `base` is the span's first
    /// forward row when a prefill checkpoint pass may capture the conv carry
    /// inside it (`conv_span`), `None` on the batched verify/decode rows.
    ///
    /// The conv carry is the one piece of PLE state a speculative verify has
    /// to be able to rewind, so at verify widths the launch is split per row
    /// and each row's resulting carry is parked. At prefill widths that would
    /// be thousands of launches for a carry nothing rolls back, so the batched
    /// form stays and `verify_snap_rows` says "no snapshots". The check is on
    /// the WHOLE forward's width, not the span's: a verify never spans
    /// (scratch >= VERIFY_SNAP_SLOTS) and a spanning forward snapshots nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inject(
        &self,
        st: &mut PleSeqState,
        hspan: DevicePtr,
        (srow, base): (usize, Option<usize>),
        n: usize,
        width: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        let cb = self.conv_bytes();
        let (gated, gated_normed, out) = (
            self.gated.offset(srow * c * 4),
            self.gated_normed.offset(srow * c * 4),
            self.out.offset(srow * c * 4),
        );
        ops::ple_gate(
            gpu,
            self.gate_k,
            hspan,
            self.key.offset(srow * c * 2),
            self.value.offset(srow * self.hidden * 2),
            self.norm_query.weight,
            self.norm_key.weight,
            self.norm_conv.weight,
            gated,
            gated_normed,
            n as u32,
            self.hidden as u32,
            self.hc_mult as u32,
            self.eps,
            stream,
        )?;
        let carry = st.conv;
        let conv = |rows: usize, at: usize| {
            ops::ple_conv(
                gpu,
                self.conv_k,
                gated_normed.offset(at * c * 4),
                gated.offset(at * c * 4),
                self.conv1d.weight,
                carry,
                out.offset(at * c * 4),
                rows as u32,
                c as u32,
                self.k_size as u32,
                self.dilation as u32,
                stream,
            )
        };
        if width < VERIFY_SNAP_SLOTS && verify_snapshots_enabled() {
            gpu.copy_d2d_async(st.conv, st.verify_snaps, cb, stream)?;
            for t in 0..n {
                conv(1, t)?;
                gpu.copy_d2d_async(st.conv, st.verify_snaps.offset((t + 1) * cb), cb, stream)?;
            }
            st.verify_snap_rows = width;
        } else {
            match base {
                Some(base) => self.conv_span(st, srow, base, n, gpu, stream)?,
                None => conv(n, 0)?,
            }
            st.verify_snap_rows = 0;
        }
        let det = crate::det_trace::on_stream(gpu, stream);
        let rows = (base.unwrap_or(0), n);
        det.tap("x_ple_key", self.key.offset(srow * c * 2), rows, c * 2);
        det.tap("x_ple_gate", gated_normed, rows, c * 4);
        det.tap("x_ple_out", out, rows, c * 4);
        ops::ple_add_highway(gpu, self.add_k, out, hspan, (n * c) as u32, stream)
    }

    /// PLE for several sequences whose rows sit seq-major in `highway`
    /// (`[rows, hc_mult*hidden]` FP32): the batched verify's ragged windows,
    /// or one row each on the multi-sequence decode. Equivalent to
    /// [`Self::forward_rows`] per sequence in order — see the module docs.
    ///
    /// Falls back to exactly that loop where the batched form does not
    /// apply: a consumable decode prestage on any member (the per-sequence
    /// forward decides whether to use it), a trellis table (its gather
    /// writes `emb` from the host per call), more rows than one scratch span,
    /// a member wide enough to skip the per-row conv snapshots, or the kill
    /// switch.
    pub fn forward_seqs(
        &self,
        seqs: &mut [PleSeqRows<'_>],
        highway: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let c = self.hc_mult * self.hidden;
        let total: usize = seqs.iter().map(|s| s.ids.len()).sum();
        let mut next = seqs.first().map_or(0, |s| s.row0);
        let contiguous = seqs.iter().all(|s| {
            let ok = s.row0 == next;
            next += s.ids.len();
            ok
        });
        let batched = seq_batch_enabled()
            && seqs.len() > 1
            && contiguous
            && self.trellis.is_none()
            && total <= self.scratch_tokens
            && total <= self.max_tokens
            && verify_snapshots_enabled()
            && seqs.iter().all(|s| {
                s.st.prestaged_va.is_none() && !s.ids.is_empty() && s.ids.len() < VERIFY_SNAP_SLOTS
            });
        if !batched {
            for s in seqs.iter_mut() {
                self.forward_rows(s.st, highway.offset(s.row0 * c * 4), s.ids, ctx, stream)?;
            }
            return Ok(());
        }
        anyhow::ensure!(
            !ctx.graph_capture,
            "PLE: multi-sequence forward inside CUDA graph capture — the \
             pageable slot upload would invalidate the recording (901)"
        );
        let gpu = ctx.gpu;
        let heads = self.dims.ngram_heads();
        let mut flat: Vec<u64> = Vec::with_capacity(total * heads);
        for s in seqs.iter_mut() {
            if s.st.history.len() != self.dims.context_len() {
                self.reset(s.st, gpu, stream)?;
            }
            flat.extend(self.stage_window(s.st, s.ids));
        }
        self.gather(&flat, total, heads, gpu, stream)?;
        self.project(total, gpu, stream)?;
        let base = seqs[0].row0;
        for s in seqs.iter_mut() {
            let n = s.ids.len();
            if let Some(w) = s.st.warm.as_ref() {
                w.note(n);
            }
            let srow = s.row0 - base;
            self.inject(
                s.st,
                highway.offset(s.row0 * c * 4),
                (srow, None),
                n,
                n,
                gpu,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
