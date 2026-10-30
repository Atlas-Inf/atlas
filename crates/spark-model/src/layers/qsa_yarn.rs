// SPDX-License-Identifier: AGPL-3.0-only

use super::QsaIndexer;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

impl QsaIndexer {
    /// Borrow the owning attention layer's static table; no per-sequence copy.
    pub(crate) fn set_yarn_rope(
        &mut self,
        inv_freq: DevicePtr,
        scale: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if inv_freq.is_null() {
            return Ok(());
        }
        self.k_pool_k = gpu.kernel("qsa_indexer", "qsa_block_pool_yarn")?;
        self.k_qprep_k = gpu.kernel("qsa_indexer", "qsa_qprep_yarn")?;
        self.k_qprep_rows_k = gpu.kernel("qsa_indexer", "qsa_qprep_rows_yarn")?;
        self.yarn_inv_freq = inv_freq;
        self.yarn_attention_factor = scale;
        Ok(())
    }
}
