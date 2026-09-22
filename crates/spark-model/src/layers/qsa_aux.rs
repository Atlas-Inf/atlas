// SPDX-License-Identifier: AGPL-3.0-only

//! Indexer state that outlives one decode step: the Marconi
//! host-serialized blob, and the speculative-verify rewind.
//! Split out of `qsa.rs` for the 500-LoC cap.

use super::*;

impl QsaIndexer {
    /// Marconi aux blob: `[ingested u64][pooled u64][raw_keys bf16 bytes]`.
    /// Raw keys are a deterministic function of the token prefix, so the
    /// snapshot IS the indexer state; block keys are re-pooled on restore
    /// (one kernel) rather than serialized.
    pub fn snapshot_aux(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let mut blob = Vec::new();
        self.snapshot_aux_into(st, &mut blob, gpu, stream)?;
        Ok(blob)
    }

    /// [`Self::snapshot_aux`] writing into a caller-owned buffer: after the
    /// first call the buffer's capacity covers `16 + ingested*hd*2`, so ring
    /// and pool slots that snapshot repeatedly stop paying a fresh
    /// multi-MB malloc per save (the job-058 RSS bisect showed that churn —
    /// ~55 MB/boundary at 18K ctx — fragmenting glibc arenas).
    pub fn snapshot_aux_into(
        &self,
        st: &QsaSeqState,
        buf: &mut Vec<u8>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let hd = self.hd as usize;
        let key_bytes = st.ingested * hd * 2;
        buf.clear();
        buf.extend_from_slice(&(st.ingested as u64).to_le_bytes());
        buf.extend_from_slice(&(st.pooled as u64).to_le_bytes());
        let off = buf.len();
        buf.resize(off + key_bytes, 0);
        if key_bytes > 0 {
            gpu.copy_d2h_on_stream(st.raw_keys, &mut buf[off..], stream)?;
        }
        Ok(())
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit:
    /// upload the raw keys, reset the counters, re-pool the block keys.
    pub fn restore_aux(
        &self,
        st: &mut QsaSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 16, "QSA aux blob truncated");
        let ingested = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let pooled = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
        let hd = self.hd as usize;
        anyhow::ensure!(
            blob.len() == 16 + ingested * hd * 2,
            "QSA aux blob size mismatch"
        );
        anyhow::ensure!(ingested <= self.max_tokens, "QSA aux exceeds key cache");
        if ingested > 0 {
            gpu.copy_h2d_async(&blob[16..], st.raw_keys, stream)?;
        }
        st.ingested = ingested;
        st.pooled = 0;
        if pooled > 0 {
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                st.raw_keys,
                self.k_norm_w,
                st.block_keys,
                0,
                pooled as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            st.pooled = pooled;
        }
        Ok(())
    }

    /// Rewind the indexer by the `rejected` tail of a speculative verify.
    ///
    /// Stated as a DELTA against what the verify just ingested, not as a
    /// target derived from a pre-verify watermark: the watermark only works
    /// if every verify path pairs a checkpoint with its rollback, and the MTP
    /// path commits through `commit_accepted_prefix` while only the
    /// self-speculative path goes through `checkpoint_ssm_states`. The delta
    /// needs no pairing — the verify scanned `k` rows and kept `num_accepted`,
    /// so exactly `k - num_accepted` ingests have to come back off.
    ///
    /// Raw keys past the new end stay in `raw_keys` but are dead: the next
    /// ingest overwrites them before anything reads them. A pooled block
    /// strictly below `ingested / ratio` lies wholly inside the accepted
    /// prefix, so keeping those is exact; anything above is dropped and
    /// re-pooled from corrected keys.
    pub fn rewind_verify(&self, st: &mut QsaSeqState, rejected: usize) -> Result<()> {
        if rejected == 0 {
            return Ok(());
        }
        // Not `saturating_sub`: rewinding more than was ingested means the
        // caller's `k` and the rows actually scanned disagree, and clamping to
        // zero would turn that dispatcher bug into a sequence that silently
        // re-ingests from the start.
        anyhow::ensure!(
            st.ingested >= rejected,
            "QSA rewind of {rejected} row(s) with only {} ingested — the \
             verify width and the rows actually scanned disagree.",
            st.ingested
        );
        anyhow::ensure!(self.ratio > 0, "QSA ratio is 0");
        st.ingested -= rejected;
        st.pooled = st.pooled.min(st.ingested / self.ratio as usize);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// `snapshot_aux_into` must emit byte-identical blobs AND keep the
    /// caller's buffer allocation across calls — the capacity reuse is the
    /// entire point of the API.
    #[test]
    fn snapshot_aux_into_reuses_buffer() {
        let gpu = MockGpuBackend::new();
        let qsa = QsaIndexer::new(
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            /*n_heads*/ 2,
            /*hd*/ 8,
            /*ratio*/ 4,
            /*budget*/ 64,
            /*max_seq_len*/ 256,
            /*rot*/ 8,
            /*theta*/ 1e5,
            /*eps*/ 1e-5,
            /*hidden*/ 128,
            /*nkv_attn*/ 2,
            /*hd_attn*/ 16,
            &gpu,
        )
        .unwrap();
        let ingested = 5usize;
        let st = QsaSeqState {
            ingested,
            pooled: 0,
            table_len: 0,
            raw_keys: gpu.alloc(256 * 8 * 2).unwrap(),
            block_keys: gpu.alloc(64 * 8 * 2).unwrap(),
        };
        let keys: Vec<u8> = (0..(ingested * 8 * 2) as u32)
            .map(|v| (v % 251) as u8)
            .collect();
        gpu.copy_h2d(&keys, st.raw_keys).unwrap();

        let want = qsa.snapshot_aux(&st, &gpu, 0).unwrap();
        assert_eq!(&want[..8], &(ingested as u64).to_le_bytes());
        assert_eq!(&want[16..], &keys[..]);

        let mut buf = Vec::new();
        qsa.snapshot_aux_into(&st, &mut buf, &gpu, 0).unwrap();
        assert_eq!(buf, want);
        let (cap, ptr) = (buf.capacity(), buf.as_ptr());
        qsa.snapshot_aux_into(&st, &mut buf, &gpu, 0).unwrap();
        assert_eq!(buf, want);
        assert_eq!(buf.capacity(), cap, "buffer reallocated on reuse");
        assert_eq!(buf.as_ptr(), ptr, "buffer moved on reuse");
    }

    fn indexer(gpu: &MockGpuBackend) -> QsaIndexer {
        QsaIndexer::new(
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            /*n_heads*/ 2,
            /*hd*/ 8,
            /*ratio*/ 4,
            /*budget*/ 64,
            /*max_seq_len*/ 256,
            /*rot*/ 8,
            /*theta*/ 1e5,
            /*eps*/ 1e-5,
            /*hidden*/ 128,
            /*nkv_attn*/ 2,
            /*hd_attn*/ 16,
            gpu,
        )
        .unwrap()
    }

    fn state(gpu: &MockGpuBackend, ingested: usize, pooled: usize) -> QsaSeqState {
        QsaSeqState {
            ingested,
            pooled,
            table_len: 0,
            raw_keys: gpu.alloc(256 * 8 * 2).unwrap(),
            block_keys: gpu.alloc(64 * 8 * 2).unwrap(),
        }
    }

    /// After a verify of `rows` rows that kept `accepted`, the counters must
    /// equal those of a sequence that serially decoded the accepted prefix
    /// and nothing else: `ingested` back to the prefix, and no pooled block
    /// that holds a rejected key — for every phase of `visible % ratio`,
    /// including a rejection that lands exactly on a block boundary.
    #[test]
    fn rewind_lands_on_the_serial_state_of_the_accepted_prefix() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        for base in 100..108 {
            for rows in 1..=4 {
                for accepted in 0..=rows {
                    let after_verify = base + rows;
                    let mut st = state(&gpu, after_verify, after_verify / 4);
                    qsa.rewind_verify(&mut st, rows - accepted).unwrap();
                    let kept = base + accepted;
                    assert_eq!(
                        st.ingested, kept,
                        "base {base} rows {rows} accepted {accepted}"
                    );
                    assert_eq!(
                        st.pooled,
                        kept / 4,
                        "base {base} rows {rows} accepted {accepted}: a block holding a rejected key survived"
                    );
                }
            }
        }
    }

    /// Pooling may lag ingest (it runs at the next ingest); a rewind must
    /// never RAISE the pooled count to the new quotient.
    #[test]
    fn rewind_never_raises_the_pooled_count() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        let mut st = state(&gpu, 110, 20);
        qsa.rewind_verify(&mut st, 2).unwrap();
        assert_eq!((st.ingested, st.pooled), (108, 20));
    }

    #[test]
    fn rewinding_nothing_changes_nothing() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        let mut st = state(&gpu, 110, 27);
        qsa.rewind_verify(&mut st, 0).unwrap();
        assert_eq!((st.ingested, st.pooled), (110, 27));
    }

    /// Rewinding more rows than were ingested means the dispatcher's verify
    /// width and the rows actually scanned disagree: an error, never a clamp,
    /// and the state is left as it was.
    #[test]
    fn rewinding_past_the_start_is_an_error_and_leaves_the_state() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        let mut st = state(&gpu, 3, 0);
        let err = qsa.rewind_verify(&mut st, 4).unwrap_err().to_string();
        assert!(
            err.contains("QSA rewind of 4 row(s) with only 3 ingested"),
            "{err}"
        );
        assert_eq!((st.ingested, st.pooled), (3, 0));
    }
}
