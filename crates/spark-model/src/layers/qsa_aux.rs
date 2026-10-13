// SPDX-License-Identifier: AGPL-3.0-only

//! Indexer state that outlives one decode step: the Marconi
//! host-serialized blob, and the speculative-verify rewind.
//! Split out of `qsa.rs` for the 500-LoC cap.

use super::*;

impl QsaIndexer {
    /// Marconi aux blob: `[ingested u64][pooled u64][pooled block keys]
    /// [raw keys of the unpooled tail]`, all BF16 rows of `hd`. That IS the
    /// indexer state: scoring reads only pooled keys, and the next pool reads
    /// only the tail. Restoring writes the same bytes back, so a restored
    /// carry equals the saved one byte for byte. (It used to carry every raw
    /// key and re-pool on restore: 4x the host bytes per checkpoint.)
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
    /// first call the buffer's capacity covers the blob, so ring and pool
    /// slots that snapshot repeatedly stop paying a fresh multi-MB malloc per
    /// save (the job-058 RSS bisect showed that churn — ~55 MB/boundary at
    /// 18K ctx — fragmenting glibc arenas).
    pub fn snapshot_aux_into(
        &self,
        st: &QsaSeqState,
        buf: &mut Vec<u8>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let row = self.hd as usize * 2;
        let tail_start = st.pooled * self.ratio as usize;
        anyhow::ensure!(
            tail_start <= st.ingested && st.raw.base() <= tail_start,
            "QSA aux snapshot: pooled frontier {tail_start} outside [{}, {}]",
            st.raw.base(),
            st.ingested
        );
        let (pooled_bytes, tail_bytes) = (st.pooled * row, (st.ingested - tail_start) * row);
        buf.clear();
        buf.extend_from_slice(&(st.ingested as u64).to_le_bytes());
        buf.extend_from_slice(&(st.pooled as u64).to_le_bytes());
        let off = buf.len();
        buf.resize(off + pooled_bytes + tail_bytes, 0);
        if pooled_bytes > 0 {
            crate::layers::aux_d2h::copy(
                gpu,
                st.block_keys,
                &mut buf[off..off + pooled_bytes],
                stream,
            )?;
        }
        if tail_bytes > 0 {
            crate::layers::aux_d2h::copy(
                gpu,
                self.raw_slot(st, tail_start),
                &mut buf[off + pooled_bytes..],
                stream,
            )?;
        }
        Ok(())
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit:
    /// upload the pooled keys and the raw tail, set the counters.
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
        let row = self.hd as usize * 2;
        let tail_start = pooled * self.ratio as usize;
        anyhow::ensure!(
            tail_start <= ingested && blob.len() == 16 + (pooled + ingested - tail_start) * row,
            "QSA aux blob size mismatch"
        );
        self.reserve(st, ingested, gpu, stream)?;
        let (keys, tail) = blob[16..].split_at(pooled * row);
        if pooled > 0 {
            gpu.copy_h2d_async(keys, st.block_keys, stream)?;
        }
        (st.ingested, st.pooled) = (tail_start, pooled);
        st.raw.restart_at(tail_start);
        self.raw_room(st, ingested - tail_start, gpu, stream)?;
        if !tail.is_empty() {
            gpu.copy_h2d_async(tail, self.raw_slot(st, tail_start), stream)?;
        }
        st.ingested = ingested;
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
    /// Raw keys past the new end stay in the window but are dead: the next
    /// ingest overwrites them before anything reads them. A pooled block
    /// strictly below `ingested / ratio` lies wholly inside the accepted
    /// prefix, so keeping those is exact; anything above is dropped and
    /// re-pooled from corrected keys, which the window still holds
    /// (`qsa_window::REWIND_MARGIN`).
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
        let ingested = st.ingested - rejected;
        let pooled = st.pooled.min(ingested / self.ratio as usize);
        anyhow::ensure!(
            pooled * self.ratio as usize >= st.raw.base(),
            "QSA rewind of {rejected} row(s) reaches below the raw-key window \
             (base {}): wider than the rewind margin",
            st.raw.base()
        );
        (st.ingested, st.pooled) = (ingested, pooled);
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
        let qsa = indexer(&gpu);
        // 5 ingested, 1 block pooled: pooled key row + a 1-key raw tail.
        let st = state(&gpu, 5, 1);
        let pooled: Vec<u8> = (0..16u32).map(|v| v as u8 + 1).collect();
        let raw: Vec<u8> = (0..(5 * 16) as u32).map(|v| (v % 251) as u8).collect();
        gpu.copy_h2d(&pooled, st.block_keys).unwrap();
        gpu.copy_h2d(&raw, st.raw.bufs[0]).unwrap();

        let want = qsa.snapshot_aux(&st, &gpu, 0).unwrap();
        assert_eq!(&want[..8], &5u64.to_le_bytes());
        assert_eq!(&want[8..16], &1u64.to_le_bytes());
        assert_eq!(&want[16..32], &pooled[..]);
        assert_eq!(&want[32..], &raw[4 * 16..], "only the unpooled tail");

        let mut buf = Vec::new();
        qsa.snapshot_aux_into(&st, &mut buf, &gpu, 0).unwrap();
        assert_eq!(buf, want);
        let (cap, ptr) = (buf.capacity(), buf.as_ptr());
        qsa.snapshot_aux_into(&st, &mut buf, &gpu, 0).unwrap();
        assert_eq!(buf, want);
        assert_eq!(buf.capacity(), cap, "buffer reallocated on reuse");
        assert_eq!(buf.as_ptr(), ptr, "buffer moved on reuse");
    }

    /// Restore writes back exactly the bytes a snapshot read, and a snapshot
    /// of the restored carry is the original blob.
    #[test]
    fn restore_round_trips_the_blob() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        let st = state(&gpu, 23, 5);
        let pooled: Vec<u8> = (0..(5 * 16) as u32).map(|v| (v * 3 % 251) as u8).collect();
        let raw: Vec<u8> = (0..(23 * 16) as u32).map(|v| (v % 241) as u8).collect();
        gpu.copy_h2d(&pooled, st.block_keys).unwrap();
        gpu.copy_h2d(&raw, st.raw.bufs[0]).unwrap();
        let blob = qsa.snapshot_aux(&st, &gpu, 0).unwrap();
        assert_eq!(blob.len(), 16 + (5 + 3) * 16);

        let mut fresh = qsa.new_seq_state(&gpu).unwrap();
        qsa.restore_aux(&mut fresh, &blob, &gpu, 0).unwrap();
        assert_eq!((fresh.ingested, fresh.pooled), (23, 5));
        assert_eq!(qsa.snapshot_aux(&fresh, &gpu, 0).unwrap(), blob);
        let mut bad = blob.clone();
        bad.pop();
        assert!(qsa.restore_aux(&mut fresh, &bad, &gpu, 0).is_err());
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

    /// A carry holding positions `0..256` in its window (base 0).
    fn state(gpu: &MockGpuBackend, ingested: usize, pooled: usize) -> QsaSeqState {
        let qsa = indexer(gpu);
        let mut st = qsa.new_seq_state(gpu).unwrap();
        qsa.reserve(&mut st, 256, gpu, 0).unwrap();
        qsa.raw_room(&mut st, 256, gpu, 0).unwrap();
        (st.ingested, st.pooled) = (ingested, pooled);
        st
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
            // Up to the exact lane's 8-row window (7 drafts): a rejection
            // may un-pool two blocks.
            for rows in 1..=8 {
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
