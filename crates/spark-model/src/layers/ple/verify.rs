// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE carry across a speculative verify, and the n-gram gathers
//! that feed it. Split out of `layer.rs` for the 500-LoC cap.

use super::*;

impl PleLayer {
    pub fn rearm(&self, st: &mut PleSeqState) {
        if st.last_staged_va != 0 {
            st.prestaged_va = Some(st.last_staged_va);
        }
    }

    /// Restore the prestaged state after a failed CUDA-graph capture attempt
    /// (the eager replay re-runs `forward`, which consumed `prestaged_va`).
    /// Bytes in one conv carry — `[(k-1)*dilation, channels]` FP32.
    pub(super) fn conv_bytes(&self) -> usize {
        self.state_len * self.hc_mult * self.hidden * 4
    }

    /// Rewind this sequence's PLE carry to the `num_kept` verify tokens that
    /// were accepted (the bonus token included).
    ///
    /// Both halves of the carry move: the device conv window comes back from
    /// the per-row snapshot taken during the forward, and the host n-gram
    /// history is rebuilt from the pre-window copy plus the accepted ids.
    /// Rebuilt rather than truncated because `history` is a FIXED-WIDTH
    /// window — tokens fall off the front as it advances, so the dropped ones
    /// have to come back too.
    pub fn rollback_verify(
        &self,
        st: &mut PleSeqState,
        num_accepted: usize,
        k: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            num_accepted <= k,
            "PLE rollback: {num_accepted} accepted of a {k}-row verify"
        );
        if num_accepted == k {
            // Full accept: the live carry IS the committed one. Still finalize
            // the window so a later rollback cannot restore against it.
            st.verify_snap_rows = 0;
            st.verify_tokens.clear();
            return Ok(());
        }
        // Reaching here means rows were REJECTED, so the carry has advanced
        // past the committed prefix and must come back. If the forward took
        // the batched conv (no per-row carry) there is nothing to come back
        // to — failing loudly beats leaving the carry conditioned on drafts
        // that were thrown away, which is invisible until the n-gram
        // injections have drifted for hundreds of tokens.
        anyhow::ensure!(
            st.verify_snap_rows == k,
            "PLE rollback for a {k}-row verify but {} row(s) were snapshotted \
             — the carry advanced past the accepted prefix with nothing to \
             restore from. K must stay under VERIFY_SNAP_SLOTS ({}) and \
             ATLAS_PLE_VERIFY_SNAPSHOTS must not be 0 when MTP is on.",
            st.verify_snap_rows,
            VERIFY_SNAP_SLOTS
        );
        anyhow::ensure!(
            st.verify_tokens.len() == k,
            "PLE rollback: {} staged id(s) for a {k}-row verify — the history \
             rebuild would be short and silently wrong.",
            st.verify_tokens.len()
        );
        let num_kept = num_accepted;
        let cb = self.conv_bytes();
        gpu.copy_d2d_async(st.verify_snaps.offset(num_kept * cb), st.conv, cb, stream)?;

        let keep = self.dims.context_len();
        let mut window = st.history_ckpt.clone();
        window.extend_from_slice(&st.verify_tokens[..num_kept]);
        if window.len() >= keep {
            st.history = window[window.len() - keep..].to_vec();
        } else {
            // Shorter than one window only before the first reset; leave the
            // history invalid-length so `forward` re-seeds it via `reset`.
            st.history = window;
        }
        // A staging built for the rejected window must not be consumed by the
        // next step: it hashed ids that are no longer the sequence's history.
        st.prestaged_va = None;
        st.prestaged_n = 0;
        st.verify_snap_rows = 0;
        st.verify_tokens.clear();
        Ok(())
    }

    /// Resolve row ids to cache slots and gather them into `self.emb`.
    ///
    /// `T * ngram_heads` rows of `head_dim` land contiguously, which IS the
    /// `[T, ngram_heads * head_dim]` concatenation the projections expect —
    /// so `batched_embed` needs no PLE-specific variant.
    pub(super) fn gather(
        &self,
        ids: &[u64],
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let table_va = self.gather_host(ids, gpu, stream)?;
        self.gather_embed(table_va, num_tokens, heads, gpu, stream)
    }

    /// The HOST half of `gather`: NVMe fault-in + slot upload into the
    /// stable `slots_dev` buffer. Capture-illegal (pageable H2D), so under
    /// CUDA graphs it runs from `prestage` BEFORE replay/capture. Returns
    /// the table's device VA for the kernel half.
    pub(super) fn gather_host(
        &self,
        ids: &[u64],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<u64> {
        let mut table = self
            .table
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;

        // Release the PREVIOUS batch's pins. See `release_prev_pins`.
        Self::release_prev_pins(&mut table, gpu, stream)?;
        let table_va = match &mut *table {
            #[cfg(feature = "cuda")]
            NgramTable::Cached(cache) => {
                // Host resolves row -> slot (the ids are host-side anyway) and
                // faults missing rows off NVMe into the pinned, GPU-addressable
                // arena. The gather kernel then reads the arena BY SLOT.
                let mut slots = Vec::with_capacity(ids.len());
                let (h0, m0, _) = cache.stats();
                let t0 = std::time::Instant::now();
                cache.resolve(ids, &mut slots)?;
                // Prefill-scale gathers log the fault profile at info: the
                // misses are SERIAL blocking preads today (QD=1 under this
                // mutex), so miss-count x latency IS the prefill stall.
                // Decode-scale (16 ids) stays at debug.
                let (h1, m1, _) = cache.stats();
                let (dh, dm) = (h1 - h0, m1 - m0);
                let us = t0.elapsed().as_micros();
                if ids.len() > 64 {
                    tracing::info!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                } else {
                    tracing::debug!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                }
                let bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                let va = cache.table_dev_va()?;
                // NOTE: the pins are NOT released here. They are released at
                // the TOP of the next `gather_host`, after a stream sync — see
                // the comment there. Releasing them at this point is what
                // produced garbage input embeddings.
                DevicePtr(va)
            }
            NgramTable::Bf16(w) => {
                // Fully resident table (small fixtures / tests): the "slot" IS
                // the row id, so upload the ids truncated to u32.
                let bytes: Vec<u8> = ids.iter().flat_map(|r| (*r as u32).to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                w.weight
            }
            NgramTable::Fp8(_) => anyhow::bail!(
                "PLE: FP8 n-gram tables are not wired. This checkpoint ships BF16 \
                 rows, which are both simpler and more accurate (on LongCat, BF16 \
                 measured 0.0050 error vs FP8's 0.0247)."
            ),
        };
        Ok(table_va.0)
    }

    /// The KERNEL half of `gather`: reads `slots_dev` and the table arena —
    /// both stable device addresses — so it is graph-capture-safe.
    pub(super) fn gather_embed(
        &self,
        table_va: u64,
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.gather_embed_dispatch(table_va, num_tokens, heads, gpu, stream)
    }
}
