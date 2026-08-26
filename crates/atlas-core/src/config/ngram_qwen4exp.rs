// SPDX-License-Identifier: AGPL-3.0-only

//! N-gram hashed embeddings as `qwen4_exp` (Qwen3.8-Flash-Next) defines them.
//!
//! This is a DIFFERENT mechanism from the one in [`super::ngram`], not a
//! parameterisation of it. Both hash token ids into `K*(N-1)` tables of width
//! `embed_dim / (K*(N-1))`, both reset their shifts at document boundaries, and
//! both therefore need no more than the last `N-1` tokens to take a decode
//! step. Everything between those endpoints differs:
//!
//! |                | LongCat ([`super::ngram`])        | `qwen4_exp` (here)                     |
//! |----------------|-----------------------------------|----------------------------------------|
//! | mixing         | polynomial `Σ shift_d · (V^d mod T)` | XOR of `shift_d · m_d`               |
//! | multipliers    | powers of the vocab size          | seeded SplitMix64 draws, odd           |
//! | table rows     | `ratio·V + 2i + 1` (odd)          | consecutive PRIMES above a base        |
//! | storage        | one tensor per table              | one concatenated table + head offsets  |
//! | shift fill     | `0`                               | `eos_token_id`                         |
//!
//! Feeding a LongCat id into a `qwen4_exp` table reads an unrelated row, so the
//! two are kept apart by type rather than by a flag.
//!
//! Reference implementation: HuggingFace Transformers (Apache-2.0),
//! `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`
//! (`Qwen4ExpTextNGramEmbedding`, `_build_layer_multipliers`,
//! `_find_nth_prime_after`). Written independently against that description and
//! pinned to the published `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint, whose
//! `layer_multipliers`, `ngram_heads_vocab_sizes` and `ngram_heads_offsets`
//! buffers this module reproduces bit-exact (see the tests).

use anyhow::{Result, bail, ensure};

const MASK64: u64 = u64::MAX;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
/// Stride between PLE layers in the multiplier seed.
const SEED_STRIDE: u64 = 10007;

/// One round of SplitMix64. The multipliers are drawn from it rather than
/// stored, so a checkpoint carries three integers instead of a policy.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA) & MASK64;
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1) & MASK64;
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2) & MASK64;
    (value ^ (value >> 31)) & MASK64
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor.saturating_mul(divisor) <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The `count`-th prime strictly greater than `start`.
fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// The geometry of one PLE layer's n-gram embedding, derived entirely from
/// config. Nothing here is read from the checkpoint — the checkpoint ships the
/// same numbers as buffers, and the tests assert the two agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpNgram {
    /// Largest n-gram size N (>= 2). HF `ngram_size`.
    pub ngram_size: usize,
    /// Hash heads K per n-gram size, giving `K*(N-1)` heads. HF `heads_per_ngram`.
    pub heads_per_ngram: usize,
    /// Text vocabulary, the domain of the token ids being hashed.
    pub unigram_vocab_size: u64,
    /// Width of the concatenated n-gram embedding. HF `ple_embed_dim`.
    pub embed_dim: usize,
    /// Position of this PLE layer within `ple_layer_ids` — NOT the decoder
    /// layer index. It shifts which primes the heads draw and reseeds the
    /// multipliers, so two PLE layers never share a table layout.
    pub ple_layer_index: usize,
    /// Row count of the concatenated table is rounded up to a multiple of this.
    /// HF `make_ngram_vocab_size_divisible_by`.
    pub vocab_divisor: u64,
    pub eos_token_id: u32,
    /// HF `seed`. Fixes the SplitMix64 multiplier draw.
    pub seed: u64,
}

impl Qwen4ExpNgram {
    /// `K * (N-1)` — the number of hash heads.
    pub fn num_heads(&self) -> usize {
        self.heads_per_ngram * (self.ngram_size - 1)
    }

    /// Per-head embedding width. Exact by construction: [`Self::validate`]
    /// refuses a config where `embed_dim` does not divide evenly.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads()
    }

    /// The `ngram_size` odd multipliers, one per shift distance `0..N`.
    ///
    /// Each is bounded by `(2^63-1) / unigram_vocab_size`, which is what keeps
    /// `token * multiplier` inside the signed 64-bit range the reference mixes
    /// in — [`Self::validate`] checks that bound rather than trusting it.
    pub fn layer_multipliers(&self) -> Vec<u64> {
        let multiplier_max = (i64::MAX as u64) / self.unigram_vocab_size.max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = self
            .seed
            .wrapping_add(SEED_STRIDE * self.ple_layer_index as u64);
        (0..self.ngram_size)
            .map(|index| {
                let draw = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
                2 * (splitmix64(draw) % half_bound) + 1
            })
            .collect()
    }

    /// Row count of each head's slice of the concatenated table: consecutive
    /// primes above `base - 1`, continuing across PLE layers.
    ///
    /// Primes rather than the odd numbers LongCat uses — a collision in one
    /// head stays independent of the others because no head's modulus shares a
    /// factor with any other's.
    pub fn head_vocab_sizes(&self, base: u64) -> Vec<u64> {
        let first = self.ple_layer_index * self.num_heads();
        (0..self.num_heads())
            .map(|head| nth_prime_after(base - 1, first + head + 1))
            .collect()
    }

    /// Start row of each head within the concatenated table (running sum of
    /// [`Self::head_vocab_sizes`]).
    pub fn head_offsets(&self, base: u64) -> Vec<u64> {
        let mut total = 0u64;
        self.head_vocab_sizes(base)
            .into_iter()
            .map(|size| {
                let start = total;
                total += size;
                start
            })
            .collect()
    }

    /// Rows in the single embedding tensor: the heads concatenated, then padded
    /// up to `vocab_divisor` so the checkpoint's `split_ngram_parts` shards
    /// divide it evenly.
    pub fn padded_rows(&self, base: u64) -> u64 {
        let total: u64 = self.head_vocab_sizes(base).iter().sum();
        total.div_ceil(self.vocab_divisor) * self.vocab_divisor
    }

    /// Check every invariant the id math and the gather kernels rely on.
    pub fn validate(&self, base: u64) -> Result<()> {
        ensure!(
            self.ngram_size >= 2,
            "ngram_size must be >= 2 (it is the largest n-gram size), got {}",
            self.ngram_size
        );
        ensure!(
            self.heads_per_ngram >= 1,
            "heads_per_ngram must be >= 1, got {}",
            self.heads_per_ngram
        );
        ensure!(self.unigram_vocab_size >= 1, "vocab_size must be non-zero");
        ensure!(base >= 2, "ngram_vocab_size_base must be >= 2, got {base}");
        let heads = self.num_heads();
        ensure!(
            self.embed_dim.is_multiple_of(heads),
            "ple_embed_dim {} must divide evenly by the {} n-gram heads \
             (heads_per_ngram {} x (ngram_size {} - 1))",
            self.embed_dim,
            heads,
            self.heads_per_ngram,
            self.ngram_size
        );

        // The reference mixes in signed 64-bit. Every `token * multiplier` term
        // has to stay inside that range or the XOR mixes a wrapped value and
        // the row is silently wrong rather than out of bounds.
        let max_token = self.unigram_vocab_size - 1;
        for (shift, multiplier) in self.layer_multipliers().into_iter().enumerate() {
            ensure!(
                max_token
                    .checked_mul(multiplier)
                    .is_some_and(|v| v <= i64::MAX as u64),
                "n-gram multiplier {multiplier} for shift {shift} overflows i64 at \
                 vocab {}: the reference mixes these terms in signed 64-bit.",
                self.unigram_vocab_size
            );
        }

        // Row ids index the table through u32 gathers. A table past that range
        // is not a lint — it is a truncated lookup into the wrong row.
        let rows = self.padded_rows(base);
        if rows > u32::MAX as u64 {
            bail!(
                "the n-gram table would hold {rows} rows, past the u32 the gather \
                 kernels index with (max {}). base {base} x {heads} heads is too large.",
                u32::MAX
            );
        }
        Ok(())
    }

    /// Row ids for every head over `ctx`, head-major in reference order
    /// (`(ngram-2)*K + head`). Ids are GLOBAL — already offset into the
    /// concatenated table — so a caller gathers from one tensor, not `K*(N-1)`.
    ///
    /// `self` must have passed [`Self::validate`], and **every token in `ctx`
    /// must be `< unigram_vocab_size`**. Both are load-bearing: the multipliers
    /// are bounded by `(2^63-1)/vocab_size`, so `validate` can only promise the
    /// `token * multiplier` product stays in range for in-vocab tokens. Feed it
    /// a larger id and the product wraps — and it wraps differently here (u64)
    /// than in the reference (i64), so the two silently disagree rather than
    /// both being wrong the same way.
    pub fn ngram_ids(&self, base: u64, ctx: &[u32]) -> Vec<Vec<u32>> {
        let multipliers = self.layer_multipliers();
        let sizes = self.head_vocab_sizes(base);
        let offsets = self.head_offsets(base);
        // Shift 0 is the token itself; it still carries a multiplier.
        let shifts: Vec<Vec<u32>> = (0..self.ngram_size)
            .map(|d| {
                super::shift_right_ignore_eos_fill(ctx, d, self.eos_token_id, self.eos_token_id)
            })
            .collect();

        let mut out = Vec::with_capacity(self.num_heads());
        for ngram in 2..=self.ngram_size {
            // The mix for an n-gram size folds in shifts 0..ngram-1 only.
            let mixed: Vec<u64> = (0..ctx.len())
                .map(|pos| {
                    (0..ngram).fold(0u64, |acc, d| {
                        acc ^ (shifts[d][pos] as u64 * multipliers[d])
                    })
                })
                .collect();
            for head in 0..self.heads_per_ngram {
                let index = (ngram - 2) * self.heads_per_ngram + head;
                let (size, offset) = (sizes[index], offsets[index]);
                out.push(
                    mixed
                        .iter()
                        // < padded_rows <= u32::MAX, guaranteed by validate().
                        .map(|m| ((m % size) + offset) as u32)
                        .collect(),
                );
            }
        }
        out
    }
}

impl super::ModelConfig {
    /// The `qwen4_exp` n-gram geometry for one PLE layer, if this checkpoint
    /// declares a PLE tower at all.
    ///
    /// `Ok(None)` means it does not — the ordinary case for every other
    /// family. `ple_layer_index` is the position within `ple_layer_ids`, not a
    /// decoder layer index, because that is what shifts the prime draw.
    pub fn qwen4exp_ngram(&self, ple_layer_index: usize) -> Result<Option<Qwen4ExpNgram>> {
        if self.ple_layer_ids.is_empty() {
            return Ok(None);
        }
        ensure!(
            ple_layer_index < self.ple_layer_ids.len(),
            "PLE layer index {ple_layer_index} is past the {} declared in ple_layer_ids",
            self.ple_layer_ids.len()
        );
        // HF validates the same thing, and for the same reason: an id outside
        // this range silently selects the wrong decoder layer.
        for &one_indexed in &self.ple_layer_ids {
            ensure!(
                one_indexed >= 1 && one_indexed <= self.num_hidden_layers,
                "ple_layer_ids are ONE-indexed decoder layers in [1, {}], got {one_indexed}",
                self.num_hidden_layers
            );
        }
        let dims = Qwen4ExpNgram {
            ngram_size: self.ngram_size,
            heads_per_ngram: self.heads_per_ngram,
            unigram_vocab_size: self.vocab_size as u64,
            embed_dim: self.ple_embed_dim,
            ple_layer_index,
            vocab_divisor: self.make_ngram_vocab_size_divisible_by,
            eos_token_id: self.eos_token_id,
            seed: self.ngram_seed,
        };
        dims.validate(self.ngram_vocab_size_base)?;
        Ok(Some(dims))
    }

    /// Decoder layer index (zero-based) hosting PLE tower `ple_layer_index`.
    ///
    /// `ple_layer_ids` is one-indexed in the HF config; every consumer inside
    /// Atlas indexes layers from zero, and the off-by-one is exactly the kind
    /// that loads a real tensor from the wrong layer rather than failing.
    pub fn ple_decoder_layer(&self, ple_layer_index: usize) -> Option<usize> {
        self.ple_layer_ids.get(ple_layer_index).map(|id| id - 1)
    }
}

#[cfg(test)]
mod tests {
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
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];

    /// `...ple_embedding.layer_multipliers`, likewise read from the checkpoint.
    const SHIPPED_MULTIPLIERS: [u64; 3] =
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

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
}
