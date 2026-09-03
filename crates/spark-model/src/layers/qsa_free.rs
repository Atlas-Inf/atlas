// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence teardown for the QSA indexer carry.
//!
//! Allocation and release for one sequence's carry, kept TOGETHER: the pair is
//! the invariant that matters, and they drifted apart once already (alloc
//! existed, free did not). Split from `qsa.rs` for the 500-LoC cap. See
//! `TransformerLayer::free_state` for why this exists at all: `DevicePtr` has
//! no `Drop` and the backend sweeps only at process exit, so these buffers
//! leaked once per sequence on every one of the 12 full-attention layers.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSeqState};

impl QsaIndexer {
    /// Free this sequence's buffers (~2.5 MB x 12 layers). Zeroed so a second
    /// teardown is a no-op. See `TransformerLayer::free_state`.
    pub fn free_seq_state(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [&mut st.raw_keys, &mut st.block_keys] {
            if p.0 != 0 {
                gpu.free(*p)?;
                *p = DevicePtr(0);
            }
        }
        Ok(())
    }

    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<QsaSeqState> {
        let hd = self.hd as usize;
        let ratio = self.ratio as usize;
        Ok(QsaSeqState {
            ingested: 0,
            pooled: 0,
            table_len: 0,
            raw_keys: gpu.alloc(self.max_tokens * hd * 2)?,
            block_keys: gpu.alloc(self.max_tokens / ratio * hd * 2)?,
        })
    }
}
