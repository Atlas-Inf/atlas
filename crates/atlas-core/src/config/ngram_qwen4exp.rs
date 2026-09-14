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
#[path = "ngram_qwen4exp_tests.rs"]
mod tests;
