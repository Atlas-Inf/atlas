// SPDX-License-Identifier: AGPL-3.0-only
//! Sampled verify picks on the GLM vocabulary-split head (the TP2 pair).
//!
//! The split head (`glm_vocab_split`) leaves each rank only its part of every
//! verify row's logits, which serves an argmax but not a sample. A sampled
//! verify row is picked from the target's processed distribution instead
//! (exact rejection sampling for a one-hot draft: a draft is accepted while
//! the row's sample equals it, and the first other sample is the correction).
//! To sample without gathering whole rows, each rank sends, per row, its best
//! [`CANDIDATES`] logits in the sampler's order (value descending, then the
//! lower id), the logits of up to [`SPECIALS`] tokens the host pipeline may
//! mask (the think and tool tags) and the mass of the rest of its part at the
//! row's temperature (`argmax_bf16_candidates`). The head merges them into the
//! present tokens of each row and an [`Absent`] summary of the others: every
//! absent token lies below the best of the ranks' last candidates, and their
//! total mass is known. `sample_with_absent` then picks exactly what the full
//! row would, or says the candidates do not decide it (a top-k, min-p or top-p
//! cut that could reach an absent token), in which case the head fetches the
//! rows whole ([`TransformerModel::glm_verify_full_rows`]) and samples those.
//! The decision depends on the row alone, before the draw, so the fallback
//! keeps the distribution exact.
//!
//! Wire contract: both requests ride a verify's verdict word. While a worker
//! waits for it (`ep_worker_verdict`), the head may send
//! [`EP_VERIFY_CANDIDATES`] (then rows, temperature and the special ids) or
//! [`EP_VERIFY_FULL_ROWS`] (then rows), any number of times; both run on the
//! rows `[0, rows)` the verify (or an owner tail's restore) left in the logits
//! buffer. No verdict reaches these values (a verdict is under 64).
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::sampler::Absent;
use std::sync::OnceLock;

use super::types::TransformerModel;

/// Verdict-word request: per-row candidates follow.
pub const EP_VERIFY_CANDIDATES: u32 = 0xFFFF_FFD0;
/// Verdict-word request: the head fetches the rows whole.
pub const EP_VERIFY_FULL_ROWS: u32 = 0xFFFF_FFD1;
/// Each rank's candidates per row (`GLM_SAMPLE_C`).
pub const CANDIDATES: usize = 64;
/// Special tokens whose logits every row reports.
pub const SPECIALS: usize = 4;
/// u32 words per row record (`GLM_SAMPLE_REC_WORDS`).
const REC_WORDS: usize = 8 + 2 * CANDIDATES;
/// Widest sampled verify (the worker's generic-verify width bound).
const MAX_ROWS: usize = 32;
/// One rank's records.
const RANK_BYTES: usize = MAX_ROWS * REC_WORDS * 4;
/// A special id slot left empty, and a special logit another rank holds.
const NONE: u32 = u32::MAX;

/// Every rank's records, rank by rank (allocated at load, before KV sizing).
static RECORDS: OnceLock<DevicePtr> = OnceLock::new();

/// One sampled verify row as the head sees it: the tokens present, with their
/// logits, and what the absent ones add up to (`None`: nothing is absent).
#[derive(Debug, Clone, PartialEq)]
pub struct SplitRow {
    pub present: Vec<(u32, f32)>,
    pub absent: Option<Absent>,
}

/// Attach the records on every rank whose verify head may be the split.
pub(crate) fn prepare(
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<()> {
    if config.model_type != "glm5_next" || !super::glm_vocab_split::serves(config.ep_world_size) {
        return Ok(());
    }
    let buf = gpu.alloc(config.ep_world_size * RANK_BYTES)?;
    ensure!(
        RECORDS.set(buf).is_ok(),
        "GLM sampled verify records attached twice"
    );
    Ok(())
}

/// Merge every rank's records (`per_rank[q]`: rank q's first `rows` records)
/// into each row's present tokens and absent summary, at `temperature`, for
/// the `specials` the records were taken with.
pub fn merge_records(
    per_rank: &[&[u32]],
    rows: usize,
    temperature: f32,
    specials: &[u32],
) -> Result<Vec<SplitRow>> {
    let inv_t = 1.0 / temperature as f64;
    (0..rows)
        .map(|r| {
            let mut row = SplitRow {
                present: Vec::new(),
                absent: None,
            };
            let mut log_mass = f64::NEG_INFINITY;
            for rank in per_rank {
                let rec = rank
                    .get(r * REC_WORDS..(r + 1) * REC_WORDS)
                    .ok_or_else(|| anyhow::anyhow!("GLM sampled verify: short records"))?;
                let c = rec[0] as usize;
                ensure!(
                    (1..=CANDIDATES).contains(&c),
                    "GLM sampled verify: {c} candidates in row {r}"
                );
                let tail = f32::from_bits(rec[2]) as f64;
                if tail > 0.0 {
                    let best = f32::from_bits(rec[1]) as f64;
                    log_mass = log_add(log_mass, tail.ln() + best * inv_t);
                }
                let cands = &rec[8..8 + 2 * c];
                row.present
                    .extend(cands.chunks_exact(2).map(|p| (p[1], f32::from_bits(p[0]))));
                // A rank that filled its candidates holds more after its last;
                // the absent tokens all come after the first such last.
                if c == CANDIDATES {
                    let (id, v) = (cands[2 * c - 1], f32::from_bits(cands[2 * c - 2]));
                    if row
                        .absent
                        .is_none_or(|a| v > a.bound || (v == a.bound && id < a.bound_id))
                    {
                        row.absent = Some(Absent {
                            bound: v,
                            bound_id: id,
                            log_mass: f32::NEG_INFINITY,
                        });
                    }
                }
                for (s, &id) in specials.iter().enumerate().take(SPECIALS) {
                    let w = rec[4 + s];
                    if w != NONE && !row.present.iter().any(|&(t, _)| t == id) {
                        row.present.push((id, f32::from_bits(w)));
                    }
                }
            }
            if let Some(a) = row.absent.as_mut() {
                a.log_mass = log_mass as f32;
            }
            Ok(row)
        })
        .collect()
}

/// `ln(e^a + e^b)`.
fn log_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    if lo == f64::NEG_INFINITY {
        hi
    } else {
        hi + (lo - hi).exp().ln_1p()
    }
}

impl TransformerModel {
    /// Whether sampled verify rows can be picked here: the split head serves,
    /// and the ranks can exchange records and whole rows.
    pub(crate) fn glm_verify_sample_supported(&self) -> bool {
        RECORDS.get().is_some()
            && self.glm_verify_logits_argmax_only()
            && self.candidates_kernel().0 != 0
            && self.comm.as_ref().is_some_and(|c| {
                c.world_size() == 2
                    && c.supports_peer_exchange_async()
                    && self.buffers.sizes().moe_output >= self.config.vocab_size * 2
            })
    }

    fn candidates_kernel(&self) -> KernelHandle {
        static K: OnceLock<KernelHandle> = OnceLock::new();
        *K.get_or_init(|| {
            self.gpu
                .kernel("argmax", "argmax_bf16_candidates")
                .unwrap_or(KernelHandle(0))
        })
    }

    /// Head: the candidates of verify rows `[0, rows)` at `temperature`,
    /// with the logits of `specials` (at most [`SPECIALS`]).
    pub(crate) fn glm_verify_candidates(
        &self,
        rows: usize,
        temperature: f32,
        specials: &[u32],
    ) -> Result<Vec<SplitRow>> {
        ensure!(
            self.glm_verify_sample_supported() && temperature > 0.0 && specials.len() <= SPECIALS,
            "GLM sampled verify: temperature {temperature}, {} specials",
            specials.len()
        );
        let mut words = [NONE; 2 + SPECIALS];
        words[0] = rows as u32;
        words[1] = temperature.to_bits();
        words[2..2 + specials.len()].copy_from_slice(specials);
        self.ep_broadcast_u32(EP_VERIFY_CANDIDATES)?;
        self.ep_broadcast_tokens(&words)?;
        let (base, ranks) = self.exchange_candidates(&words)?;
        let stream = self.gpu.default_stream();
        self.gpu.synchronize(stream)?;
        let mut host = vec![0u8; ranks * RANK_BYTES];
        self.gpu.copy_d2h(base, &mut host)?;
        let words: Vec<u32> = host
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let per_rank: Vec<&[u32]> = words.chunks_exact(RANK_BYTES / 4).collect();
        merge_records(&per_rank, rows, temperature, specials)
    }

    /// Head: verify rows `[0, rows)` whole into `out` (`rows * vocab` BF16).
    pub(crate) fn glm_verify_full_rows(&self, rows: usize, out: &mut [u8]) -> Result<()> {
        let bytes = rows * self.config.vocab_size * 2;
        ensure!(
            self.glm_verify_sample_supported()
                && out.len() == bytes
                && (1..=MAX_ROWS).contains(&rows),
            "GLM sampled verify: {} bytes for {rows} full rows",
            out.len()
        );
        self.ep_broadcast_u32(EP_VERIFY_FULL_ROWS)?;
        self.ep_broadcast_u32(rows as u32)?;
        self.glm_split_head_full_rows(rows, self.gpu.default_stream())?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.gpu.copy_d2h(self.buffers.logits(), out)
    }

    /// Worker side of the verdict wait (`ep_worker_verdict`): serve `request`.
    pub(super) fn glm_verify_sample_serve(&self, request: u32) -> Result<()> {
        if request == EP_VERIFY_FULL_ROWS {
            let rows = self.ep_broadcast_u32(0)? as usize;
            ensure!(
                (1..=MAX_ROWS).contains(&rows),
                "GLM sampled verify: {rows} full rows"
            );
            return self.glm_split_head_full_rows(rows, self.gpu.default_stream());
        }
        let words = self.ep_broadcast_tokens(&[0u32; 2 + SPECIALS])?;
        self.exchange_candidates(&words).map(|_| ())
    }

    /// Both ranks: this rank's records of the rows `words` describe, then
    /// both ranks' on each. Returns the records' base and rank count.
    fn exchange_candidates(&self, words: &[u32]) -> Result<(DevicePtr, usize)> {
        let rows = words[0] as usize;
        ensure!(
            (1..=MAX_ROWS).contains(&rows) && self.glm_verify_sample_supported(),
            "GLM sampled verify of {rows} rows is not available"
        );
        let comm = self.comm.as_ref().expect("checked by supported");
        let (rank, ranks) = (comm.rank(), comm.world_size());
        let base = *RECORDS.get().expect("checked by supported");
        let local = base.offset(rank * RANK_BYTES);
        let (vocab, shard) = (self.config.vocab_size, self.config.vocab_size / ranks);
        let start = rank * shard;
        let stream = self.gpu.default_stream();
        KernelLaunch::new(self.gpu.as_ref(), self.candidates_kernel())
            .grid([rows as u32, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.buffers.logits().offset(start * 2))
            .arg_ptr(local)
            .arg_u32(shard as u32)
            .arg_u32(vocab as u32)
            .arg_u32(start as u32)
            .arg_f32(1.0 / f32::from_bits(words[1]))
            .arg_u32(words[2])
            .arg_u32(words[3])
            .arg_u32(words[4])
            .arg_u32(words[5])
            .launch(stream)?;
        let peer = base.offset((1 - rank) * RANK_BYTES);
        comm.peer_exchange_async(local.0, peer.0, rows * REC_WORDS * 4, stream)?;
        Ok((base, ranks))
    }
}

#[cfg(test)]
#[path = "glm_verify_sample_tests.rs"]
mod tests;
