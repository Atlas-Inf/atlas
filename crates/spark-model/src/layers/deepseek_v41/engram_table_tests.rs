// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The 1017 release's Engram geometry.
fn geometry() -> DeepseekV41Engram {
    DeepseekV41Engram {
        layer_ids: vec![1, 14],
        num_embeddings: vec![384_006_168, 384_016_682],
        max_ngram_size: 4,
        n_heads: 8,
        head_dim: 256,
        vocab_size: 16_000_000,
        compressed_vocab_size: 99_092,
        pad_id: 2,
    }
}

#[test]
fn a_tokens_rows_are_the_ones_the_whole_sequence_hashes_to() {
    let geometry = geometry();
    geometry.validate(40).expect("the release's geometry");
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let tokens: Vec<u32> = (0..40)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 99_092) as u32
        })
        .collect();
    let whole: Vec<Option<u32>> = tokens.iter().copied().map(Some).collect();
    for index in 0..2 {
        let hash = EngramHash::new(&geometry, index);
        assert_eq!((hash.reach(), hash.columns()), (4, 24));
        let want = geometry.row_ids(index, &whole);
        for position in 0..tokens.len() {
            // The start of the sequence, where shifts run out, and the middle.
            let from = (position + 1).saturating_sub(hash.reach());
            let rows = hash.rows(&tokens[from..=position]);
            let want: Vec<u64> = want[position].iter().map(|&r| u64::from(r)).collect();
            assert_eq!(rows, want, "layer {index} position {position}");
        }
    }
}
