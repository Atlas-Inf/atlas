// SPDX-License-Identifier: AGPL-3.0-only

//! Paged prefill chunk causal-extent upload (per-attention-sub-chunk ends).

use anyhow::Result;

use super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Upload a paged prefill chunk's causal extents: the chunk total, then
    /// for chunks wider than [`crate::layer::PREFILL_ATTENTION_ROWS`] the end
    /// of each attention sub-chunk (read by per-sub-chunk attention).
    pub(super) fn upload_chunk_seq_lens(
        &self,
        seq: &SequenceState,
        proc_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        use crate::layer::{PREFILL_MAX_SUB_CHUNKS, prefill_attention_pieces};
        let end = proc_start + proc_count;
        let mut values = vec![end as u32];
        let pieces = prefill_attention_pieces(proc_start, proc_count, self.config.index_topk);
        if pieces.len() > 1 {
            anyhow::ensure!(
                pieces.len() <= PREFILL_MAX_SUB_CHUNKS,
                "prefill chunk of {proc_count} rows exceeds {PREFILL_MAX_SUB_CHUNKS} attention pieces"
            );
            values.extend(
                pieces
                    .iter()
                    .map(|&(row0, rows)| (proc_start + row0 + rows) as u32),
            );
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let base = seq
            .chunked_prefill_meta
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paged prefill metadata missing"))?
            .seq_len;
        self.gpu.copy_h2d_async(&bytes, base, stream)
    }
}
