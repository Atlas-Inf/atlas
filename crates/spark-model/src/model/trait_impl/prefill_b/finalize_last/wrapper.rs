// SPDX-License-Identifier: AGPL-3.0-only
//! `prefill_b_finalize_last` wrapper (extracted; identical content).

use super::*;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_b_finalize_last_at(
            tokens,
            seq,
            kv_cache,
            chunk_start,
            chunk_len,
            proc_count,
            0,
            0,
            None,
            stream,
        )
    }
}
