// SPDX-License-Identifier: AGPL-3.0-only

/// An intermediate chunk of `chunk_len` rows on the GDN WY4 boundary (4
/// rows) and the model's sequence-parallel row multiple
/// (`Model::prefill_row_multiple`: 1, or 512 / 1024 on the chain).
pub(in crate::scheduler) fn chunk_align(chunk_len: usize, multiple: usize) -> usize {
    chunk_align_for(chunk_len, multiple, spark_runtime::prefill_chunk_granule())
}

fn chunk_align_for(chunk_len: usize, multiple: usize, granule: usize) -> usize {
    let step = multiple.max(granule).max(1);
    if chunk_len >= step {
        chunk_len / step * step
    } else {
        chunk_len
    }
}

/// The prompt's final chunk of `chunk_len` rows for a model whose long
/// chunks run sequence-parallel over `multiple` ranks
/// (`Model::prefill_row_multiple`): a long one not a multiple of it is cut
/// to one, and the few rows past the cut become one more (the last) chunk.
/// `(rows, is_last)`.
pub(in crate::scheduler) fn sp_final_cut(chunk_len: usize, multiple: usize) -> (usize, bool) {
    // The sequence-parallel floor (`glm_prefill_sp_rows`), plus the tail.
    const SP_MIN_ROWS: usize = 4096;
    let tail = chunk_len % multiple.max(1);
    if tail == 0 || chunk_len < SP_MIN_ROWS + tail {
        return (chunk_len, true);
    }
    (chunk_len - tail, false)
}

#[cfg(test)]
mod sp_final_cut_tests {
    use super::sp_final_cut;

    #[test]
    fn long_final_chunks_land_on_the_rank_multiple() {
        assert_eq!(sp_final_cut(4193, 4), (4192, false));
        assert_eq!(sp_final_cut(8195, 4), (8192, false));
        assert_eq!(sp_final_cut(4192, 4), (4192, true));
        assert_eq!(sp_final_cut(4097, 4), (4096, false));
        // Too short to run sequence-parallel either way, and the pair.
        assert_eq!(sp_final_cut(4095, 4), (4095, true));
        assert_eq!(sp_final_cut(3001, 4), (3001, true));
        assert_eq!(sp_final_cut(8193, 1), (8193, true));
        // The chain's 128 rows a rank: whole tiles a part, the rest a tail.
        assert_eq!(sp_final_cut(8196, 512), (8192, false));
        assert_eq!(sp_final_cut(9700, 512), (9216, false));
        assert_eq!(sp_final_cut(4500, 512), (4096, false));
        assert_eq!(sp_final_cut(4095, 512), (4095, true));
    }

    #[test]
    fn intermediate_chunks_land_on_wy4_and_the_rank_multiple() {
        use super::chunk_align;
        // TP2 and TP4's SP: the WY4 boundary.
        assert_eq!(chunk_align(8196, 1), 8196);
        assert_eq!(chunk_align(8193, 4), 8192);
        // TP4's two-half pipeline: both halves on the ranks.
        assert_eq!(chunk_align(8196, 8), 8192);
        assert_eq!(chunk_align(3, 8), 3);
        // The chain's arena (8200 rows) takes 16 whole 512-row steps.
        assert_eq!(chunk_align(8200, 512), 8192);
        assert_eq!(chunk_align(300, 512), 300);
    }
}

#[cfg(test)]
#[path = "../chunk_alignment_tests.rs"]
mod chunk_alignment_tests;
