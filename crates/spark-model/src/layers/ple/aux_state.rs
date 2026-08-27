// SPDX-License-Identifier: AGPL-3.0-only

//! PLE Marconi aux-state: serialize / restore the per-sequence lexical
//! carry (token history + conv state) that rides the SSM snapshots.
//! Split from `layer.rs` for the ≤500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{PleLayer, PleSeqState};

impl PleLayer {
    /// Marconi aux blob: `[hist_len u32][history u32s][conv f32 bytes]`.
    /// The whole per-sequence carry — a prefix hit restoring KV+SSM without
    /// this would run the n-gram hash on the PREVIOUS request's history.
    pub fn snapshot_aux(
        &self,
        st: &PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        let mut blob = Vec::with_capacity(4 + st.history.len() * 4 + conv_bytes);
        blob.extend_from_slice(&(st.history.len() as u32).to_le_bytes());
        for t in &st.history {
            blob.extend_from_slice(&t.to_le_bytes());
        }
        let off = blob.len();
        blob.resize(off + conv_bytes, 0);
        gpu.copy_d2h_on_stream(st.conv, &mut blob[off..], stream)?;
        Ok(blob)
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit.
    pub fn restore_aux(
        &self,
        st: &mut PleSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 4, "PLE aux blob truncated");
        let n = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        anyhow::ensure!(
            blob.len() == 4 + n * 4 + conv_bytes,
            "PLE aux blob size mismatch"
        );
        st.history = blob[4..4 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        st.prestaged_va = None;
        gpu.copy_h2d_async(&blob[4 + n * 4..], st.conv, stream)?;
        Ok(())
    }
}
