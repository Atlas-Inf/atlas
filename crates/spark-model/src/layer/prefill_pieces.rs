// SPDX-License-Identifier: AGPL-3.0-only

//! Paged prefill attention sub-chunking: the per-piece row bound and the
//! `(row0, rows)` split of a chunk at the semantic top-k boundary.

/// Widest attention sub-chunk of a paged prefill chunk: the GLM native sparse
/// attention and FlashKDA are qualified up to 4100 rows. Wider chunks keep
/// their token-parallel work (projections, MoE) whole and run attention per
/// sub-chunk; chunk metadata then carries each sub-chunk's causal extent.
pub const PREFILL_ATTENTION_ROWS: usize = 4096;
/// Sub-chunk extents a chunk's `seq_len` buffer holds after its total.
pub const PREFILL_MAX_SUB_CHUNKS: usize = 8;

/// Attention pieces `(row0, rows)` of a prefill chunk starting at sequence
/// position `seq_start`. A chunk that crosses the semantic top-k boundary
/// splits there: rows before it attend their whole causal history (dense is
/// exact), rows after it select. The rest are at most
/// [`PREFILL_ATTENTION_ROWS`] rows each.
pub fn prefill_attention_pieces(seq_start: usize, rows: usize, topk: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut row0 = 0;
    if seq_start < topk && seq_start + rows > topk {
        row0 = topk - seq_start;
        pieces.push((0, row0));
    }
    while row0 < rows {
        let n = (rows - row0).min(PREFILL_ATTENTION_ROWS);
        pieces.push((row0, n));
        row0 += n;
    }
    pieces
}

#[cfg(test)]
mod prefill_piece_tests {
    use super::prefill_attention_pieces as pieces;

    #[test]
    fn first_chunk_splits_at_the_topk_boundary_then_by_attention_rows() {
        assert_eq!(pieces(0, 4096, 2048), [(0, 2048), (2048, 2048)]);
        assert_eq!(
            pieces(0, 8196, 2048),
            [(0, 2048), (2048, 4096), (6144, 2052)]
        );
        assert_eq!(pieces(1000, 4096, 2048), [(0, 1048), (1048, 3048)]);
    }

    #[test]
    fn chunks_past_or_before_the_boundary_split_only_by_attention_rows() {
        assert_eq!(pieces(4096, 4096, 2048), [(0, 4096)]);
        assert_eq!(
            pieces(8192, 8196, 2048),
            [(0, 4096), (4096, 4096), (8192, 4)]
        );
        assert_eq!(pieces(0, 1500, 2048), [(0, 1500)]);
        assert_eq!(pieces(0, 2048, 2048), [(0, 2048)]);
        assert_eq!(pieces(0, 4096, 0), [(0, 4096)]);
    }
}
