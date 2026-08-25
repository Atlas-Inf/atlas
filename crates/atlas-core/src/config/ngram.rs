// SPDX-License-Identifier: AGPL-3.0-only

//! N-gram scaled embeddings: the id arithmetic and its validity envelope.
//!
//! Mechanism (arXiv:2601.21204; the reference implementation Atlas is checked
//! against lives in `bench/ngram_embed/`). `K*(N-1)` lookup tables, where
//! `K = emb_split_num` and `N = emb_neighbor_num`. Table `index = (i-2)*K + j`
//! for n-gram size `i` and split `j` holds
//!
//! ```text
//!   T(index) = ratio * vocab + 2*index + 1
//! ```
//!
//! rows of width `hidden / (K*(N-1))`. The consecutive ODD offsets give the K
//! tables of one n-gram size mutually near-coprime row counts, so a collision in
//! one split is independent of the others.
//!
//! A row id is a polynomial rolling hash over TOKEN IDS ONLY — never hidden
//! state:
//!
//! ```text
//!   id_t(i,j) = ( x_t + Σ_{d=1..i-1} shift_d(x)_t * (V^d mod T) ) mod T
//! ```
//!
//! `shift_d` right-shifts by `d` and RESETS at document boundaries: a position
//! within `d` tokens of a segment start contributes 0 rather than reaching
//! across an EOS (segments end AT an EOS, inclusive). Depending only on token
//! ids is what makes the lookups deterministic, prefetchable and safe under
//! speculative decoding — and it is why a decode step needs no more than the
//! last `N-1` tokens of history.
//!
//! This module is deliberately pure integer math in the config crate: the
//! arithmetic envelope below has to be checked when a config is parsed, long
//! before any GPU layer exists, and the hash must not be written twice.

use anyhow::{Result, bail, ensure};

/// The n-gram trio plus everything derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramDims {
    pub vocab_size: u64,
    pub ratio: u64,
    /// Largest n-gram size N (>= 2).
    pub neighbor_num: usize,
    /// Hash splits K per n-gram size.
    pub split_num: usize,
    pub eos_token_id: u32,
    pub hidden_size: usize,
}

impl NgramDims {
    /// `K * (N-1)` — the number of lookup tables.
    pub fn num_tables(&self) -> usize {
        self.split_num * (self.neighbor_num - 1)
    }

    /// Per-table embedding width. Exact by construction: `validate` refuses a
    /// config where `hidden_size` does not divide evenly.
    pub fn table_dim(&self) -> usize {
        self.hidden_size / self.num_tables()
    }

    /// Row count of table `index`.
    pub fn table_rows(&self, index: usize) -> u64 {
        self.ratio * self.vocab_size + 2 * index as u64 + 1
    }

    /// `[V^1 mod T, ..., V^(ngram-1) mod T]` for table `(ngram, split)`.
    pub fn vocab_mods(&self, ngram: usize, split: usize) -> Vec<u64> {
        let t = self.table_rows((ngram - 2) * self.split_num + split);
        let mut mods = Vec::with_capacity(ngram - 1);
        let mut power: u64 = 1;
        for _ in 0..ngram - 1 {
            power = (power % t) * (self.vocab_size % t) % t;
            mods.push(power);
        }
        mods
    }

    /// Largest value the pre-modulo accumulator can reach, over every table —
    /// `None` if computing that bound itself overflows `u64`.
    ///
    /// This is the number that decides whether the hash is evaluable in 64-bit
    /// integers at all, and it is checked rather than asserted in a comment:
    /// the whole point of this work is a model family whose embedding tables
    /// are far larger than the one it was developed against, and a wrapped
    /// accumulator would surface as subtly wrong logits rather than as a crash.
    pub fn max_accumulator(&self) -> Option<u64> {
        let max_tok = self.vocab_size.checked_sub(1)?;
        let mut worst: u64 = 0;
        for ngram in 2..=self.neighbor_num {
            for split in 0..self.split_num {
                let mut acc = max_tok;
                for m in self.vocab_mods(ngram, split) {
                    acc = acc.checked_add(max_tok.checked_mul(m)?)?;
                }
                worst = worst.max(acc);
            }
        }
        Some(worst)
    }

    /// Check every invariant the id math and the GPU gathers rely on.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.neighbor_num >= 2,
            "emb_neighbor_num must be >= 2 (it is the largest n-gram size), got {}",
            self.neighbor_num
        );
        ensure!(
            self.split_num >= 1,
            "emb_split_num must be >= 1, got {}",
            self.split_num
        );
        ensure!(
            self.ratio >= 1,
            "ngram_vocab_size_ratio must be >= 1, got {}",
            self.ratio
        );
        ensure!(self.vocab_size >= 1, "vocab_size must be non-zero");
        let tables = self.num_tables();
        ensure!(
            self.hidden_size.is_multiple_of(tables),
            "hidden_size {} must divide evenly by the {} n-gram tables \
             (emb_split_num {} x (emb_neighbor_num {} - 1))",
            self.hidden_size,
            tables,
            self.split_num,
            self.neighbor_num
        );

        // Row ids index the tables through the gather kernels, which take u32.
        // A table larger than u32 addresses is not a lint — it is a silently
        // truncated lookup into the wrong row.
        let largest = self.table_rows(tables - 1);
        ensure!(
            largest <= u32::MAX as u64,
            "n-gram table {} would hold {} rows, past the u32 the gather \
             kernels index with (max {}). ratio {} x vocab {} is too large.",
            tables - 1,
            largest,
            u32::MAX,
            self.ratio,
            self.vocab_size
        );

        match self.max_accumulator() {
            Some(_) => Ok(()),
            None => bail!(
                "the n-gram rolling hash would overflow u64 at vocab {} / ratio {} / \
                 emb_neighbor_num {}: the accumulator sums up to {} terms of \
                 (vocab-1) x (V^d mod T). Reduce the n-gram size or the table ratio.",
                self.vocab_size,
                self.ratio,
                self.neighbor_num,
                self.neighbor_num - 1
            ),
        }
    }
}

/// `out[t] = ctx[t-n]`, zero wherever `[t-n, t]` would cross a document
/// boundary. Segments end AT an EOS (inclusive); a segment no longer than `n`
/// contributes nothing, having no position far enough from its start.
pub fn shift_right_ignore_eos(ctx: &[u32], n: usize, eos: u32) -> Vec<u32> {
    let mut out = vec![0u32; ctx.len()];
    let mut prev = 0usize;
    for (pos, &tok) in ctx.iter().enumerate() {
        if tok == eos {
            let end = pos + 1;
            if end - prev > n {
                out[prev + n..end].copy_from_slice(&ctx[prev..end - n]);
            }
            prev = end;
        }
    }
    if ctx.len() - prev > n {
        out[prev + n..].copy_from_slice(&ctx[prev..ctx.len() - n]);
    }
    out
}

/// Row ids for every table over `ctx`, table-major in reference index order
/// (`(ngram-2)*K + split`). Callers serving a chunk slice the last `seq_len`.
///
/// `dims` must have passed [`NgramDims::validate`]; the arithmetic below is
/// only sound inside that envelope.
pub fn ngram_ids(dims: &NgramDims, ctx: &[u32]) -> Vec<Vec<u32>> {
    // A shift by d is shared across every split of every n-gram size using it.
    let shifts: Vec<Vec<u32>> = (1..dims.neighbor_num)
        .map(|d| shift_right_ignore_eos(ctx, d, dims.eos_token_id))
        .collect();

    let mut out = Vec::with_capacity(dims.num_tables());
    for ngram in 2..=dims.neighbor_num {
        for split in 0..dims.split_num {
            let t = dims.table_rows((ngram - 2) * dims.split_num + split);
            let mods = dims.vocab_mods(ngram, split);
            out.push(
                ctx.iter()
                    .enumerate()
                    .map(|(pos, &x)| {
                        let mut acc = x as u64;
                        for (d, &m) in mods.iter().enumerate() {
                            acc += shifts[d][pos] as u64 * m;
                        }
                        // < T <= u32::MAX, guaranteed by validate().
                        (acc % t) as u32
                    })
                    .collect(),
            );
        }
    }
    out
}

impl super::ModelConfig {
    /// The n-gram dimensions this checkpoint declares, if any.
    ///
    /// `Ok(None)` means the checkpoint declares no n-gram embeddings — the
    /// ordinary case for every other family. `Err` means it declared them
    /// incoherently, which is refused rather than partially honoured: the trio
    /// travels together, and a checkpoint naming two thirds of it is one whose
    /// remaining third we would otherwise have to invent.
    pub fn ngram_dims(&self) -> Result<Option<NgramDims>> {
        let present = [
            ("ngram_vocab_size_ratio", self.ngram_vocab_size_ratio),
            ("emb_neighbor_num", self.emb_neighbor_num),
            ("emb_split_num", self.emb_split_num),
        ];
        let named: Vec<&str> = present
            .iter()
            .filter(|(_, v)| *v != 0)
            .map(|(k, _)| *k)
            .collect();
        if named.is_empty() {
            return Ok(None);
        }
        ensure!(
            named.len() == 3,
            "n-gram config is partial: {} present, {} missing. All three of \
             ngram_vocab_size_ratio / emb_neighbor_num / emb_split_num must be \
             declared together.",
            named.join(", "),
            present
                .iter()
                .filter(|(_, v)| *v == 0)
                .map(|(k, _)| *k)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let dims = NgramDims {
            vocab_size: self.vocab_size as u64,
            ratio: self.ngram_vocab_size_ratio as u64,
            neighbor_num: self.emb_neighbor_num,
            split_num: self.emb_split_num,
            eos_token_id: self.eos_token_id,
            hidden_size: self.hidden_size,
        };
        dims.validate()?;
        Ok(Some(dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LITE: NgramDims = NgramDims {
        vocab_size: 131072,
        ratio: 78,
        neighbor_num: 4,
        split_num: 4,
        eos_token_id: 2,
        hidden_size: 3072,
    };

    #[derive(serde::Deserialize)]
    struct IdCase {
        name: String,
        vocab_size: u64,
        hidden_size: usize,
        ngram_vocab_size_ratio: u64,
        emb_neighbor_num: usize,
        emb_split_num: usize,
        eos_token_id: u32,
        tokens: Vec<u32>,
        max_accumulator: u64,
        expected_ids: Vec<Vec<u32>>,
    }

    #[derive(serde::Deserialize)]
    struct Fixtures {
        id_cases: Vec<IdCase>,
    }

    fn fixtures() -> Fixtures {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../bench/ngram_embed/fixtures.json"
        );
        let raw = std::fs::read_to_string(path)
            .expect("regenerate with bench/ngram_embed/make_fixtures.py");
        serde_json::from_str(&raw).expect("fixtures parse")
    }

    /// The ids are the whole contract: a table row chosen even slightly
    /// differently from the reference reads an unrelated embedding, and the
    /// model degrades rather than fails. Checked bit-exact, at real
    /// LongCat-Flash-Lite dimensions and on synthetic (N, K) shapes, over token
    /// streams built around the document-boundary resets.
    #[test]
    fn ids_are_bit_exact_against_the_reference() {
        let f = fixtures();
        assert!(!f.id_cases.is_empty(), "fixture file has no id cases");
        for c in &f.id_cases {
            let dims = NgramDims {
                vocab_size: c.vocab_size,
                ratio: c.ngram_vocab_size_ratio,
                neighbor_num: c.emb_neighbor_num,
                split_num: c.emb_split_num,
                eos_token_id: c.eos_token_id,
                hidden_size: c.hidden_size,
            };
            dims.validate()
                .unwrap_or_else(|e| panic!("{}: {e}", c.name));
            assert_eq!(
                dims.max_accumulator(),
                Some(c.max_accumulator),
                "{}: accumulator bound disagrees with the reference",
                c.name
            );
            let got = ngram_ids(&dims, &c.tokens);
            assert_eq!(got.len(), dims.num_tables(), "{}: table count", c.name);
            for (index, ids) in got.iter().enumerate() {
                assert_eq!(ids, &c.expected_ids[index], "{} table {index}", c.name);
            }
        }
    }

    /// A decode step carries only the last N-1 tokens. If that were not
    /// sufficient, decode would disagree with prefill on the same position and
    /// the model would drift the moment a sequence left the prefill path —
    /// so this pins the window size the KV/context plumbing has to honour.
    #[test]
    fn the_last_n_minus_one_tokens_are_enough_to_decode() {
        let stream: Vec<u32> = vec![41, 9, 137, 2, 88, 5, 6002, 17, 3, 2, 71];
        let full = ngram_ids(&LITE, &stream);
        let carry = LITE.neighbor_num - 1;

        for t in 0..stream.len() {
            let lo = t.saturating_sub(carry);
            let window = &stream[lo..=t];
            let stepped = ngram_ids(&LITE, window);
            for index in 0..LITE.num_tables() {
                assert_eq!(
                    stepped[index][window.len() - 1],
                    full[index][t],
                    "table {index} at position {t} disagrees between prefill and decode"
                );
            }
        }
    }

    /// Chunked prefill must be a partition of prefill, not an approximation of
    /// it: the same absolute position has to hash identically whichever chunk
    /// happens to contain it.
    #[test]
    fn chunked_prefill_agrees_with_whole_prefill() {
        let stream: Vec<u32> = vec![5, 900, 2, 44, 7, 7, 2, 61, 12, 8, 300, 2];
        let full = ngram_ids(&LITE, &stream);
        let carry = LITE.neighbor_num - 1;

        for split_at in 1..stream.len() {
            let lo = split_at.saturating_sub(carry);
            let chunk = &stream[lo..];
            let got = ngram_ids(&LITE, chunk);
            let offset = split_at - lo;
            for index in 0..LITE.num_tables() {
                assert_eq!(
                    &got[index][offset..],
                    &full[index][split_at..],
                    "table {index} diverges when the stream is split at {split_at}"
                );
            }
        }
    }

    /// With every token an EOS, no segment is ever longer than the shift
    /// distance, so every shifted term must vanish and the hash must collapse
    /// to the bare token id. Catches a shift that silently reaches across a
    /// boundary — which a random-token fixture can mask.
    #[test]
    fn an_all_eos_stream_contributes_no_shifted_terms() {
        let eos = LITE.eos_token_id;
        let ids = ngram_ids(&LITE, &[eos; 6]);
        for (index, table) in ids.iter().enumerate() {
            for (pos, &id) in table.iter().enumerate() {
                assert_eq!(id, eos, "table {index} position {pos} crossed a boundary");
            }
        }
    }

    #[test]
    fn a_table_wider_than_u32_is_refused() {
        // Row ids feed u32 gather kernels; a table past that range would
        // truncate into the wrong row instead of failing.
        let huge = NgramDims {
            ratio: 40_000,
            ..LITE
        };
        let err = huge
            .validate()
            .expect_err("u32-overflowing table must be refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("past the u32"), "wrong rejection: {msg}");
    }

    #[test]
    fn hidden_size_must_divide_by_the_table_count() {
        let bad = NgramDims {
            hidden_size: 3070,
            ..LITE
        };
        let err = bad
            .validate()
            .expect_err("indivisible hidden_size must be refused");
        assert!(format!("{err:#}").contains("divide evenly"));
    }

    #[test]
    fn a_partial_trio_is_refused_rather_than_guessed() {
        let config = |ratio, neighbor, split| -> super::super::ModelConfig {
            serde_json::from_value(serde_json::json!({
                "vocab_size": 131072,
                "hidden_size": 3072,
                "eos_token_id": 2,
                "ngram_vocab_size_ratio": ratio,
                "emb_neighbor_num": neighbor,
                "emb_split_num": split,
            }))
            .expect("minimal config deserializes")
        };
        assert_eq!(config(78, 4, 4).ngram_dims().unwrap(), Some(LITE));

        let err = config(78, 4, 0)
            .ngram_dims()
            .expect_err("two thirds of the trio must not configure");
        let msg = format!("{err:#}");
        assert!(msg.contains("partial"), "wrong rejection: {msg}");
        assert!(
            msg.contains("emb_split_num"),
            "missing key not named: {msg}"
        );

        // All three absent is the ordinary case for every other family.
        assert_eq!(config(0, 0, 0).ngram_dims().unwrap(), None);
    }
}
