// SPDX-License-Identifier: AGPL-3.0-only

//! A sequence's own vision state (`model::vision_cache`).

/// Everything a vision prefill needs across its chunks, held on the sequence
/// so two image prompts can interleave prefill chunks without overwriting
/// each other's state. The encoder output buffer (`buf_out`) stays one shared
/// scratch buffer: every chunk encodes the items it splices right before its
/// embed, and nothing in it survives to the next chunk. All empty/zero for a
/// text prompt.
#[derive(Default)]
pub struct SeqVision {
    /// The request's images (host pixels). Head only; a worker receives rows
    /// by broadcast. Released when the last prompt chunk succeeds.
    pub items: Vec<crate::VisionItem>,
    /// Per-item `(t_len, grid_h_post_merge, grid_w_post_merge)`. MRoPE
    /// positions from these; `t_len` is TEMPORAL GROUPS (1 for a still), not
    /// encoder rows — a video feeds `t_len` rows to the ViT but occupies ONE
    /// contiguous pad run.
    pub grids: Vec<(usize, usize, usize)>,
    /// Per-item content hash, parallel to `grids`. Keys the prefix cache
    /// (`SequenceState::vision_cache_key`).
    pub hashes: Vec<u128>,
    /// Leading encoder rows no chunk encodes: the cached items'. Their
    /// placeholders all sit at positions this prefill does not compute, so
    /// their rows are left as the raw token embedding no layer reads.
    pub skip_rows: usize,
    /// First prompt position the prefill computes (agreed across ranks).
    /// Every skipped row's placeholder sits before it.
    pub computed_from: usize,
    /// This prefill encodes per chunk: chunk 0 inside its prefix lookup, every
    /// later chunk right before its embed.
    pub deferred: bool,
    /// Rows the last encode of THIS prefill wrote to `buf_out`, from physical
    /// row 0. An item straddling a chunk boundary is re-encoded at every chunk
    /// holding one of its placeholders, so the rows a chunk splices are always
    /// the last encode's.
    pub rows_encoded: usize,
    /// Virtual row — the index in the prompt-order packed non-cached output —
    /// of `buf_out` physical row 0 for the last encode. The splice maps a
    /// pad's packed row `v` to physical `v - rows_base`.
    pub rows_base: usize,
    /// Co-dispatched batched-ViT slice for THIS sequence's chunk-0
    /// splice/MRoPE: when a tick batches >=2 image requests into one
    /// `buf_out`, `row_base` = first `buf_out` row, `grid_base` = first
    /// `grids` index, and `owned_images` bounds the grid scan. All 0 ⇒ the
    /// sequence's own encode (row 0 / grid 0). Set by the scheduler.
    pub row_base: usize,
    pub grid_base: usize,
    pub owned_images: usize,
    /// Encoded rows owned by this sequence's co-dispatched slice; 0 for its
    /// own encode.
    pub slice_rows: usize,
}
