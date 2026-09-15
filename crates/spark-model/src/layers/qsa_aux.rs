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
}
