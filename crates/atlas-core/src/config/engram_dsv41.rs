// SPDX-License-Identifier: AGPL-3.0-only

//! Engram n-gram hash ids as DeepSeek-V4.1 defines them.
//!
//! A third n-gram mechanism beside [`super::ngram`] (LongCat) and
//! [`super::ngram_qwen4exp`], kept apart by type for the same reason those two
//! are: an id from one scheme reads an unrelated row of another's table.
//!
//! |              | `qwen4_exp`                    | `deepseek_v41` Engram (here)                       |
//! |--------------|--------------------------------|----------------------------------------------------|
//! | token domain | raw token ids                  | a COMPRESSED vocab: tokens that normalize alike (case, accents, whitespace) share an id |
//! | multipliers  | seeded SplitMix64, odd         | numpy `default_rng(10007 * layer_id)`, odd         |
//! | table rows   | consecutive primes, padded     | consecutive primes, one table per Engram layer, unpadded |
//! | shift fill   | `eos_token_id`                 | the compressed id of `engram_pad_token_id`; a fill also blocks every longer shift |
//! | breaks       | document boundaries            | sequence start and dead (image-span) tokens         |
//!
//! Building the compressed vocab needs the tokenizer's normalizers and is the
//! caller's job; everything here starts from compressed ids.
//!
//! Reference: `inference/engram.py` of deepseek-ai/DeepSeek-V4.1-Flash-1017
//! (MIT), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Written
//! independently against it; the tests pin the published config's table sizes
//! and multipliers, and row ids from a transcription of the reference.

use anyhow::{Result, ensure};

use super::ngram_qwen4exp::nth_prime_after;

/// Stride between Engram layers in the multiplier seed.
const SEED_STRIDE: u64 = 10007;

/// The geometry of a checkpoint's Engram tables, derived entirely from config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepseekV41Engram {
    /// Decoder layers carrying an Engram, in order. HF `engram_layer_ids`.
    pub layer_ids: Vec<usize>,
    /// Rows of each layer's table as published. HF `engram_num_embeddings`.
    pub num_embeddings: Vec<u64>,
    /// Largest n-gram size N; a position hashes the 2..=N-grams ending there.
    pub max_ngram_size: usize,
    /// Hash heads per n-gram size.
    pub n_heads: usize,
    /// Width of one table row.
    pub head_dim: usize,
    /// Every head's prime search starts above `vocab_size - 1`.
    pub vocab_size: u64,
    /// Size of the compressed vocab; every multiplier is derived from it.
    pub compressed_vocab_size: u64,
    /// Compressed id that fills shifts with no history.
    pub pad_id: u32,
}

impl DeepseekV41Engram {
    /// Row ids per position and layer: `(N-1) * n_heads`, n-gram-size major.
    pub fn n_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }

    /// Bucket modulus of every `(n-gram size, head)` column of Engram layer
    /// `index` (the position in `layer_ids`). The reference hands out the
    /// smallest unused prime above the base, layer by layer, so the columns
    /// take consecutive primes and no two share one.
    pub fn primes(&self, index: usize) -> Vec<u64> {
        let mut prime = nth_prime_after(self.vocab_size - 1, index * self.n_cols());
        (0..self.n_cols())
            .map(|_| {
                prime = nth_prime_after(prime, 1);
                prime
            })
            .collect()
    }

    /// Start row of each column within layer `index`'s table.
    pub fn offsets(&self, index: usize) -> Vec<u64> {
        let mut total = 0;
        self.primes(index)
            .into_iter()
            .map(|p| {
                let start = total;
                total += p;
                start
            })
            .collect()
    }

    /// The `max_ngram_size` odd multipliers of Engram layer `index`, one per
    /// lookback. The bound keeps `token * multiplier` inside i64, as the
    /// reference mixes in signed 64-bit.
    pub fn multipliers(&self, index: usize) -> Vec<u64> {
        let bound = (i64::MAX as u64 / self.compressed_vocab_size) / 2;
        let mut rng = Pcg64::from_seed(SEED_STRIDE * self.layer_ids[index] as u64);
        (0..self.max_ngram_size)
            .map(|_| 2 * rng.below(bound) + 1)
            .collect()
    }

    /// Check every invariant the ids and the table gathers rely on, including
    /// that the bucket ranges tile each published table exactly: a mismatch
    /// there means the config and the derivation disagree, and every lookup
    /// would read a wrong row.
    pub fn validate(&self, num_hidden_layers: usize) -> Result<()> {
        ensure!(
            self.max_ngram_size >= 2,
            "engram_max_ngram_size must be >= 2"
        );
        ensure!(
            self.n_heads >= 1 && self.head_dim >= 1,
            "engram heads must be non-empty"
        );
        ensure!(
            self.layer_ids.len() == self.num_embeddings.len(),
            "engram_layer_ids ({}) and engram_num_embeddings ({}) differ in length",
            self.layer_ids.len(),
            self.num_embeddings.len()
        );
        ensure!(
            self.layer_ids.windows(2).all(|w| w[0] < w[1])
                && self.layer_ids.iter().all(|&l| l < num_hidden_layers),
            "engram_layer_ids must be increasing decoder layers below {num_hidden_layers}"
        );
        // The draw seeds numpy's SeedSequence with one 32-bit word.
        ensure!(
            self.layer_ids
                .iter()
                .all(|&l| SEED_STRIDE * l as u64 <= u32::MAX as u64),
            "an Engram layer id is too large for the multiplier seed"
        );
        ensure!(
            self.compressed_vocab_size >= 1 && self.compressed_vocab_size <= u32::MAX as u64,
            "engram_compressed_vocab_size must be in 1..2^32"
        );
        ensure!(
            (self.pad_id as u64) < self.compressed_vocab_size,
            "the Engram pad id is outside the compressed vocab"
        );
        // Lemire's 64-bit path; the reference takes it for any vocab below 2^30.
        ensure!(
            (i64::MAX as u64 / self.compressed_vocab_size) / 2 > u32::MAX as u64,
            "engram_compressed_vocab_size is too large for the multiplier draw"
        );
        for (index, &rows) in self.num_embeddings.iter().enumerate() {
            let tiled: u64 = self.primes(index).iter().sum();
            ensure!(
                tiled == rows && rows <= u32::MAX as u64,
                "Engram layer {}: the prime buckets cover {tiled} rows, the config \
                 declares {rows}",
                self.layer_ids[index]
            );
        }
        Ok(())
    }

    /// Row ids of Engram layer `index` for every position of `tokens`, each
    /// `n_cols` long. `tokens` are compressed ids from the start of the
    /// sequence, `None` for a dead token (inside an image span): an n-gram
    /// never reaches across one or past the start, and the slots it cannot
    /// fill take `pad_id`.
    ///
    /// `self` must have passed [`Self::validate`], and every token must be
    /// below `compressed_vocab_size`: the multipliers are bounded so that only
    /// in-vocab products stay inside the i64 range the reference mixes in.
    pub fn row_ids(&self, index: usize, tokens: &[Option<u32>]) -> Vec<Vec<u32>> {
        debug_assert!(
            tokens
                .iter()
                .flatten()
                .all(|&t| (t as u64) < self.compressed_vocab_size)
        );
        let primes = self.primes(index);
        let offsets = self.offsets(index);
        let multipliers = self.multipliers(index);
        (0..tokens.len())
            .map(|pos| {
                let mut blocked = false;
                let products: Vec<u64> = (0..self.max_ngram_size)
                    .map(|shift| {
                        let source = pos.checked_sub(shift).and_then(|p| tokens[p]);
                        blocked |= source.is_none();
                        let token = if blocked {
                            self.pad_id
                        } else {
                            source.unwrap_or(0)
                        };
                        token as u64 * multipliers[shift]
                    })
                    .collect();
                let mut rolling = products[0];
                let mut cols = Vec::with_capacity(self.n_cols());
                for (n, product) in products.iter().enumerate().skip(1) {
                    rolling ^= product;
                    for head in 0..self.n_heads {
                        let col = (n - 1) * self.n_heads + head;
                        cols.push((rolling % primes[col] + offsets[col]) as u32);
                    }
                }
                cols
            })
            .collect()
    }
}

impl super::ModelConfig {
    /// The Engram geometry, if this checkpoint declares Engram layers.
    ///
    /// `pad_id` is the COMPRESSED id of `engram_pad_token_id`: the config names
    /// a raw token, and only the caller's compressed vocab knows what it maps to.
    pub fn deepseek_v41_engram(&self, pad_id: u32) -> Result<Option<DeepseekV41Engram>> {
        if self.engram_layer_ids.is_empty() {
            return Ok(None);
        }
        let engram = DeepseekV41Engram {
            layer_ids: self.engram_layer_ids.clone(),
            num_embeddings: self.engram_num_embeddings.clone(),
            max_ngram_size: self.engram_max_ngram_size,
            n_heads: self.engram_n_heads,
            head_dim: self.engram_head_dim,
            vocab_size: self.engram_vocab_size,
            compressed_vocab_size: self.engram_compressed_vocab_size,
            pad_id,
        };
        engram.validate(self.num_hidden_layers)?;
        Ok(Some(engram))
    }
}

/// numpy's `default_rng(seed)`: PCG64 (XSL-RR) seeded through `SeedSequence`,
/// with `Generator.integers`' unbiased Lemire draw. Only what the multipliers
/// need: a non-negative integer seed below 2^32 and 64-bit bounds.
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    fn from_seed(seed: u64) -> Self {
        let words = seed_sequence_u64(seed as u32, 4);
        let initstate = (words[0] as u128) << 64 | words[1] as u128;
        let initseq = (words[2] as u128) << 64 | words[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: initseq << 1 | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let xored = ((self.state >> 64) as u64) ^ self.state as u64;
        xored.rotate_right((self.state >> 122) as u32)
    }

    /// Uniform in `0..bound` (`integers(0, bound)`), for bounds above 2^32.
    fn below(&mut self, bound: u64) -> u64 {
        let mut m = self.next_u64() as u128 * bound as u128;
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * bound as u128;
            }
        }
        (m >> 64) as u64
    }
}

/// `SeedSequence(seed).generate_state(n, np.uint64)` for a one-word entropy.
fn seed_sequence_u64(seed: u32, n: usize) -> Vec<u64> {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_L: u32 = 0xca01_f9dd;
    const MIX_R: u32 = 0x4973_f715;
    const POOL: usize = 4;

    let mut hash_a = INIT_A;
    let mut hashmix = |value: u32| {
        let mut value = value ^ hash_a;
        hash_a = hash_a.wrapping_mul(MULT_A);
        value = value.wrapping_mul(hash_a);
        value ^ (value >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool: [u32; POOL] = std::array::from_fn(|i| hashmix(if i == 0 { seed } else { 0 }));
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src]));
            }
        }
    }
    let mut hash_b = INIT_B;
    let words: Vec<u32> = (0..2 * n)
        .map(|i| {
            let mut value = pool[i % POOL] ^ hash_b;
            hash_b = hash_b.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_b);
            value ^ (value >> 16)
        })
        .collect();
    words
        .chunks(2)
        .map(|w| (w[1] as u64) << 32 | w[0] as u64)
        .collect()
}

#[cfg(test)]
#[path = "engram_dsv41_tests.rs"]
mod tests;
