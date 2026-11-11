// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model::trait_impl) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch_with(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            stream,
            None,
        )
    }
}
