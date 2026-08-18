// SPDX-License-Identifier: AGPL-3.0-only

//! Exact BF16 post-layer fingerprints for Lightning K4-vs-M1 localization.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::TransformerModel;

pub(super) fn enabled() -> bool {
    std::env::var("ATLAS_LIGHTNING_VERIFY_LAYER_TRACE").as_deref() == Ok("1")
}

impl TransformerModel {
    pub(super) fn trace_lightning_hidden_rows(
        &self,
        mode: &'static str,
        seq_len: usize,
        layer_idx: usize,
        hidden: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if !enabled() {
            return Ok(());
        }
        let row_bytes = self.config.hidden_size * 2;
        self.gpu.synchronize(stream)?;
        let mut host = vec![0u8; rows * row_bytes];
        self.gpu.copy_d2h(hidden, &mut host)?;
        for row in 0..rows {
            let bytes = &host[row * row_bytes..(row + 1) * row_bytes];
            let mut hash = 0xcbf29ce484222325u64;
            for &byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            tracing::warn!(
                "LIGHTNING VERIFY LAYER TRACE mode={mode} seq_len={seq_len} layer={layer_idx} row={row} fnv64={hash:016x}"
            );
        }
        Ok(())
    }
}
