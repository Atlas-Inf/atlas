// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the `qwen4_exp` n-gram id core. Split out for the 500-LoC cap.

use super::*;

/// Qwen/Qwen3.8-Flash-Next-FP8, read off `config.json` at HF main
/// (2026-08-26). `seed` is absent from that file — the 1234 here is the
/// documented default, and the multiplier test below is what proves it is
/// the value the checkpoint was actually built with.
const FLASH_NEXT: Qwen4ExpNgram = Qwen4ExpNgram {
    ngram_size: 3,
    heads_per_ngram: 8,
    unigram_vocab_size: 248_320,
    embed_dim: 2560,
    ple_layer_index: 0,
    vocab_divisor: 128,
    eos_token_id: 248_044,
    seed: 1234,
};
const BASE: u64 = 20_000_000;

/// The 16 head vocab sizes the published checkpoint ships in
/// `model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes`,
/// read out of the safetensors shard rather than recomputed.
const SHIPPED_VOCAB_SIZES: [u64; 16] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];

/// `...ple_embedding.layer_multipliers`, likewise read from the checkpoint.
const SHIPPED_MULTIPLIERS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

/// The whole point of deriving these rather than loading them: if the
/// derivation is even slightly off, every row id is wrong and the model
/// degrades into noise instead of failing. Pinned against the real
/// checkpoint's buffers, which is the only authority that settles it.
#[test]
fn the_checkpoint_buffers_are_reproduced_from_config_alone() {
    FLASH_NEXT
        .validate(BASE)
        .expect("real config must validate");
    assert_eq!(
        FLASH_NEXT.layer_multipliers(),
        SHIPPED_MULTIPLIERS,
        "SplitMix64 multiplier draw disagrees with the checkpoint"
    );
    assert_eq!(
        FLASH_NEXT.head_vocab_sizes(BASE),
        SHIPPED_VOCAB_SIZES,
        "prime table sizes disagree with the checkpoint"
    );
    let expected_offsets: Vec<u64> = SHIPPED_VOCAB_SIZES
        .iter()
        .scan(0u64, |acc, size| {
            let start = *acc;
            *acc += size;
            Some(start)
        })
        .collect();
    assert_eq!(FLASH_NEXT.head_offsets(BASE), expected_offsets);
}

/// The shard tensors are `[2_500_012, 160]` x 128 in the published index.
/// Both numbers fall out of the geometry, so they pin it end to end: the
/// padding rule via the row count, the head count via the width.
#[test]
fn the_geometry_matches_the_published_shard_shapes() {
    assert_eq!(FLASH_NEXT.num_heads(), 16);
    assert_eq!(FLASH_NEXT.head_dim(), 160, "shard tensor width");
    let rows = FLASH_NEXT.padded_rows(BASE);
    assert_eq!(rows, 320_001_536);
    assert_eq!(rows / 128, 2_500_012, "shard tensor row count");
    // Padding is real here -- the unpadded sum is not already a multiple.
    assert_eq!(SHIPPED_VOCAB_SIZES.iter().sum::<u64>(), 320_001_446);
}

/// A decode step carries only the last N-1 tokens. If that were not
/// enough, decode would disagree with prefill at the same position and the
/// model would drift the moment a sequence left the prefill path.
#[test]
fn the_last_n_minus_one_tokens_are_enough_to_decode() {
    let eos = FLASH_NEXT.eos_token_id;
    let stream: Vec<u32> = vec![41, 9, 137, eos, 88, 5, 6002, 17, 3, eos, 71];
    let full = FLASH_NEXT.ngram_ids(BASE, &stream);
    let carry = FLASH_NEXT.ngram_size - 1;
    for t in 0..stream.len() {
        let lo = t.saturating_sub(carry);
        let window = &stream[lo..=t];
        let stepped = FLASH_NEXT.ngram_ids(BASE, window);
        for head in 0..FLASH_NEXT.num_heads() {
            assert_eq!(
                stepped[head][window.len() - 1],
                full[head][t],
                "head {head} at position {t} disagrees between prefill and decode"
            );
        }
    }
}

/// Chunked prefill must be a partition of prefill, not an approximation:
/// the same absolute position hashes identically whichever chunk holds it.
#[test]
fn chunked_prefill_agrees_with_whole_prefill() {
    let eos = FLASH_NEXT.eos_token_id;
    let stream: Vec<u32> = vec![5, 900, eos, 44, 7, 7, eos, 61, 12, 8, 300, eos];
    let full = FLASH_NEXT.ngram_ids(BASE, &stream);
    let carry = FLASH_NEXT.ngram_size - 1;
    for split_at in 1..stream.len() {
        let lo = split_at.saturating_sub(carry);
        let got = FLASH_NEXT.ngram_ids(BASE, &stream[lo..]);
        let offset = split_at - lo;
        for head in 0..FLASH_NEXT.num_heads() {
            assert_eq!(
                &got[head][offset..],
                &full[head][split_at..],
                "head {head} diverges when the stream is split at {split_at}"
            );
        }
    }
}

/// Every id must land inside its own head's slice. An id that strays into
/// the neighbouring head reads a real embedding from the wrong table --
/// invisible to any bounds check, since the tensor is one concatenation.
#[test]
fn every_id_stays_inside_its_own_head_slice() {
    let stream: Vec<u32> = vec![0, 1, 248_319, 248_044, 77, 12_345, 99, 248_318];
    let ids = FLASH_NEXT.ngram_ids(BASE, &stream);
    let sizes = FLASH_NEXT.head_vocab_sizes(BASE);
    let offsets = FLASH_NEXT.head_offsets(BASE);
    for (head, table) in ids.iter().enumerate() {
        let (lo, hi) = (offsets[head], offsets[head] + sizes[head]);
        for (pos, &id) in table.iter().enumerate() {
            assert!(
                (id as u64) >= lo && (id as u64) < hi,
                "head {head} position {pos}: id {id} outside [{lo}, {hi})"
            );
        }
    }
}

/// The two n-gram families must not be silently interchangeable. Same
/// (N, K) and the same tokens, deliberately different ids -- a checkpoint
/// routed to the wrong one would otherwise look plausible.
#[test]
fn longcat_and_qwen4exp_ids_are_not_interchangeable() {
    let stream: Vec<u32> = vec![11, 523, 9001, 44, 130_000, 7, 88, 4];
    let qwen = FLASH_NEXT.ngram_ids(BASE, &stream);
    let longcat = super::super::ngram_ids(
        &super::super::NgramDims {
            vocab_size: 248_320,
            ratio: 80,
            neighbor_num: 3,
            split_num: 8,
            eos_token_id: 248_044,
            hidden_size: 2560,
        },
        &stream,
    );
    assert_eq!(qwen.len(), longcat.len(), "same head count by construction");
    assert_ne!(qwen, longcat, "the two schemes must not coincide");
}

#[test]
fn a_config_without_a_ple_tower_declares_no_ngram_path() {
    let config = super::super::ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(config.qwen4exp_ngram(0).unwrap(), None);
    assert_eq!(config.ple_decoder_layer(0), None);
}

/// `ple_layer_ids` is one-indexed in the HF config and zero-indexed
/// everywhere inside Atlas. `[2]` is decoder layer 1 -- which is where the
/// published checkpoint really does store `layers.1.ple.*`.
#[test]
fn ple_layer_ids_are_one_indexed() {
    let mut config = super::super::ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 48;
    config.ple_layer_ids = vec![2];
    assert_eq!(config.ple_decoder_layer(0), Some(1));

    config.ple_layer_ids = vec![0];
    let err = config
        .qwen4exp_ngram(0)
        .expect_err("0 is not a valid one-indexed layer");
    assert!(format!("{err:#}").contains("ONE-indexed"), "{err:#}");
}

#[test]
fn an_embed_dim_that_does_not_divide_by_the_head_count_is_refused() {
    let bad = Qwen4ExpNgram {
        embed_dim: 2570,
        ..FLASH_NEXT
    };
    let err = bad
        .validate(BASE)
        .expect_err("indivisible embed_dim must be refused");
    assert!(format!("{err:#}").contains("divide evenly"), "{err:#}");
}

#[test]
fn a_table_wider_than_u32_is_refused() {
    // Row ids feed u32 gathers; a table past that range would truncate
    // into the wrong row instead of failing.
    let huge = Qwen4ExpNgram {
        heads_per_ngram: 200,
        embed_dim: 2400,
        ..FLASH_NEXT
    };
    let err = huge
        .validate(BASE)
        .expect_err("u32-overflowing table must be refused");
    assert!(format!("{err:#}").contains("past the u32"), "{err:#}");
}
