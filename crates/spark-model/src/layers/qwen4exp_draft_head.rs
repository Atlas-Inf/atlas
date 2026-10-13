// SPDX-License-Identifier: AGPL-3.0-only

//! The qwen4_exp MTP drafter's argmax head: which rows of the SHARED LM head
//! a draft step projects, and in what precision.
//!
//! Every draft is re-verified by the target's own full head, so this head
//! decides only WHICH token is proposed — acceptance, never output.
//!
//! - Default: the first `--mtp-vocab` ids (100,000; 0 = all), BF16, read in
//!   place from the shared head.
//! - `ATLAS_QWEN4EXP_DRAFT_VOCAB=<file>`: an explicit id list (decimal ids,
//!   whitespace/comma separated, `#` comments). The rows are gathered ONCE at
//!   load into a compact `[n, hidden]` BF16 matrix, the draft argmax runs over
//!   it, and the winning row maps back to its full id. Ids are sorted, so the
//!   "lowest index wins a tie" rule of `argmax_bf16` stays "lowest full id".
//!   `scripts/dev/qwen4exp_draft_vocab.py` derives a list (ids below a bound,
//!   every added/special id, then the most frequent ids of a corpus).
//! - `ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4=1`: an NVFP4 copy of those rows for
//!   drafting (a quarter of the BF16 bytes; lossy, so it can move acceptance).
//!
//! Cost per draft (one `dense_gemv_bf16` over `[n, 2560]`, ~244 GB/s on
//! GB10, `scripts/dev/qwen4exp_exact_verify_bench.cu`): 5.21 ms at the full
//! 248,320 rows, 2.1 ms at the 100k default prefix, 0.99 ms at 47k.
//!
//! Rank 0 drafts (`weight_loader/qwen4_exp/mtp.rs`). Under
//! `ATLAS_QWEN4EXP_MTP_DRAFT_TP` the worker builds the same head from its own
//! copy of the LM head and projects half of its rows
//! ([`DraftHead::project_rows_range`], `qwen4exp_mtp_tp.rs`).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{DenseWeight, QuantizedWeight};

const BF16: usize = 2;

/// The rows a draft step projects.
pub struct DraftHead {
    /// Full-vocabulary id of each row, ascending; `None` = row `i` is id `i`.
    ids: Option<Vec<u32>>,
    rows: u32,
    /// BF16 rows: a compact gather for a list, else the shared head itself.
    bf16: DenseWeight,
    nvfp4: Option<QuantizedWeight>,
    w4a16_gemv_k: KernelHandle,
    /// The NVFP4 copy's multi-row tiers (`project_rows`): scalar
    /// `w4a16_gemv_batch{M}`, row `r` byte-identical to `w4a16_gemv`.
    w4a16_tiers: W4a16BatchmTiers,
    /// `ATLAS_QWEN4EXP_W4_ROWS`: the persistent 4..8-row tier for the copy.
    w4_rows: ops::Qwen4ExpW4Rows,
}

impl DraftHead {
    /// `prefix` = `--mtp-vocab` (0 = the whole vocabulary).
    pub fn build(
        lm_head: &DenseWeight,
        vocab: usize,
        hidden: usize,
        prefix: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::build_with(lm_head, vocab, hidden, prefix, nvfp4_requested(), gpu)
    }

    /// [`Self::build`] with the NVFP4 copy chosen by the caller.
    pub fn build_with(
        lm_head: &DenseWeight,
        vocab: usize,
        hidden: usize,
        prefix: u32,
        want_nvfp4: bool,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let list = match std::env::var("ATLAS_QWEN4EXP_DRAFT_VOCAB") {
            Ok(path) if !path.is_empty() => {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("ATLAS_QWEN4EXP_DRAFT_VOCAB: read {path}"))?;
                Some(parse_ids(&text, vocab).with_context(|| format!("draft vocab {path}"))?)
            }
            _ => None,
        };
        let stream = gpu.default_stream();
        let (ids, rows, bf16) = match list {
            Some(ids) => {
                let rows = ids.len();
                let dst = gpu.alloc(rows * hidden * BF16)?;
                let mut at = 0usize;
                for (start, len) in runs(&ids) {
                    let bytes = len as usize * hidden * BF16;
                    gpu.copy_d2d_async(
                        lm_head.weight.offset(start as usize * hidden * BF16),
                        dst.offset(at),
                        bytes,
                        stream,
                    )?;
                    at += bytes;
                }
                gpu.synchronize(stream)?;
                tracing::info!(
                    "qwen4_exp MTP draft head: {rows} of {vocab} ids \
                     (ATLAS_QWEN4EXP_DRAFT_VOCAB, {:.1} MB BF16 gathered)",
                    (rows * hidden * BF16) as f64 / 1e6
                );
                (Some(ids), rows as u32, DenseWeight { weight: dst })
            }
            None => {
                let rows = if prefix > 0 {
                    prefix.min(vocab as u32)
                } else {
                    vocab as u32
                };
                (None, rows, *lm_head)
            }
        };
        let nvfp4 = if want_nvfp4 {
            let q = crate::weight_map::quantize_to_nvfp4(
                &bf16,
                rows as usize,
                hidden,
                gpu,
                gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
                gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
                stream,
            )
            .context("qwen4_exp MTP draft head: NVFP4 copy")?;
            tracing::info!(
                "qwen4_exp MTP draft head: NVFP4 copy of {rows} rows for drafting \
                 (ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4, lossy: acceptance only)"
            );
            Some(q)
        } else {
            None
        };
        // The gathered BF16 rows are the NVFP4 copy's source only.
        let bf16 = if nvfp4.is_some() && ids.is_some() {
            gpu.free(bf16.weight)?;
            *lm_head
        } else {
            bf16
        };
        Ok(Self {
            ids,
            rows,
            bf16,
            nvfp4,
            w4a16_gemv_k: if want_nvfp4 {
                gpu.kernel("w4a16_gemv", "w4a16_gemv")?
            } else {
                KernelHandle(0)
            },
            w4a16_tiers: if want_nvfp4 {
                W4a16BatchmTiers::resolve(gpu)
            } else {
                W4a16BatchmTiers::default()
            },
            w4_rows: if want_nvfp4 {
                ops::Qwen4ExpW4Rows::resolve(gpu)
            } else {
                ops::Qwen4ExpW4Rows::OFF
            },
        })
    }

    /// Rows a draft step projects (the argmax width).
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Whether the draft rows are the NVFP4 copy.
    pub fn is_nvfp4(&self) -> bool {
        self.nvfp4.is_some()
    }

    /// Project `input` (`[hidden]` BF16) onto the draft rows, into `logits`.
    pub fn project(
        &self,
        gpu: &dyn GpuBackend,
        dense_gemv_k: KernelHandle,
        input: DevicePtr,
        logits: DevicePtr,
        hidden: u32,
        stream: u64,
    ) -> Result<()> {
        match self.nvfp4.as_ref() {
            Some(q) => ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                input,
                q,
                logits,
                self.rows,
                hidden,
                stream,
            ),
            None => ops::dense_gemv(
                gpu,
                dense_gemv_k,
                input,
                &self.bf16,
                logits,
                self.rows,
                hidden,
                stream,
            ),
        }
    }

    /// Whether [`Self::project_rows`] can serve `m` rows, and every draft row
    /// IS its token id, so a device argmax is already the drafted token
    /// (the batched propose embeds it without a host round trip). A gathered
    /// id list (`ATLAS_QWEN4EXP_DRAFT_VOCAB`) would need a device remap, so
    /// it keeps the per-sequence path.
    pub fn rows_batchable(&self, m: usize, dense_batchm_k: KernelHandle) -> bool {
        self.ids.is_none()
            && match self.nvfp4 {
                Some(_) => self.w4a16_tiers.scalar_kernel(m as u32).0 != 0,
                None => dense_batchm_k.0 != 0 && m as u32 <= ops::DENSE_GEMV_BATCHM_MAX_M,
            }
    }

    /// [`Self::project`] for `m` contiguous `[hidden]` input rows into `m`
    /// contiguous `[rows]` logit rows, the head read once. Row `r` is
    /// byte-identical to `project` on that row alone: `dense_gemv_bf16_batchm`
    /// equals `dense_gemv_bf16` per row, the scalar `w4a16_gemv_batch{M}`
    /// tiers equal `w4a16_gemv`. Gate on [`Self::rows_batchable`].
    #[allow(clippy::too_many_arguments)]
    pub fn project_rows(
        &self,
        gpu: &dyn GpuBackend,
        dense_batchm_k: KernelHandle,
        input: DevicePtr,
        logits: DevicePtr,
        m: u32,
        hidden: u32,
        stream: u64,
    ) -> Result<()> {
        self.project_rows_range(
            gpu,
            dense_batchm_k,
            input,
            logits,
            m,
            (0, self.rows),
            hidden,
            stream,
        )
    }

    /// [`Self::project_rows`] onto draft rows `first .. first + n` only, into
    /// `m` output rows `n` apart: output column `c` is column `first + c` of
    /// `project_rows`, byte for byte (each column is its weight row against
    /// the input row; the kernels' arithmetic for a column depends on neither
    /// its position nor the column count). A pointer offset into the rows.
    #[allow(clippy::too_many_arguments)]
    pub fn project_rows_range(
        &self,
        gpu: &dyn GpuBackend,
        dense_batchm_k: KernelHandle,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        (first, n): (u32, u32),
        hidden: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            n > 0 && first.checked_add(n).is_some_and(|end| end <= self.rows),
            "draft head rows {first}..+{n} outside 0..{}",
            self.rows
        );
        match self.nvfp4.as_ref() {
            Some(q) => {
                let w = q.rows_from(first as usize, hidden as usize);
                self.w4_rows
                    .launch_or(gpu, input, &w, out, (m, n, hidden), stream, || {
                        ops::w4a16_gemv_batchm(
                            gpu,
                            self.w4a16_tiers.scalar_kernel(m),
                            input,
                            &w,
                            out,
                            m,
                            n,
                            hidden,
                            stream,
                        )
                    })
            }
            None => ops::dense_gemv_batchm(
                gpu,
                dense_batchm_k,
                input,
                &DenseWeight {
                    weight: self
                        .bf16
                        .weight
                        .offset(first as usize * hidden as usize * BF16),
                },
                out,
                m,
                n,
                hidden,
                n,
                stream,
            ),
        }
    }

    /// Full-vocabulary id of draft row `row`.
    pub fn token(&self, row: u32) -> u32 {
        self.ids.as_ref().map_or(row, |ids| ids[row as usize])
    }

    /// The draft row of full-vocabulary id `id`, if the head projects it
    /// (the inverse of [`Self::token`]).
    pub fn row_of(&self, id: u32) -> Option<u32> {
        row_of(self.ids.as_deref(), self.rows, id)
    }
}

/// [`DraftHead::row_of`] over a head's id list (`None` = row `i` is id `i`)
/// and row count.
fn row_of(ids: Option<&[u32]>, rows: u32, id: u32) -> Option<u32> {
    match ids {
        Some(ids) => ids.binary_search(&id).ok().map(|r| r as u32),
        None => (id < rows).then_some(id),
    }
}

/// `ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4=1`: draft from an NVFP4 copy of the rows.
pub fn nvfp4_requested() -> bool {
    std::env::var("ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4").as_deref() == Ok("1")
}

/// `ATLAS_QWEN4EXP_DRAFT_VOCAB`: an explicit draft id list is named.
pub fn list_requested() -> bool {
    std::env::var("ATLAS_QWEN4EXP_DRAFT_VOCAB").is_ok_and(|p| !p.is_empty())
}

/// Parse a draft id list: decimal ids separated by whitespace or commas,
/// `#` to end of line a comment. Sorted, de-duplicated, each `< vocab`.
pub fn parse_ids(text: &str, vocab: usize) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        for tok in line.split(|c: char| c.is_whitespace() || c == ',') {
            if tok.is_empty() {
                continue;
            }
            let id: u32 = tok
                .parse()
                .with_context(|| format!("not a token id: {tok:?}"))?;
            if id as usize >= vocab {
                bail!("token id {id} is outside the {vocab}-id vocabulary");
            }
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ensure!(!ids.is_empty(), "no token ids");
    Ok(ids)
}

/// Maximal runs of consecutive ids in a sorted list, `(first, len)`: one row
/// copy per run when gathering.
pub fn runs(ids: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &id in ids {
        match out.last_mut() {
            Some((start, len)) if *start + *len == id => *len += 1,
            _ => out.push((id, 1)),
        }
    }
    out
}

/// Softmax probability of the top logit of a BF16 row (host side, for the
/// confidence stop): `1 / sum(exp(x - max))`.
pub fn top1_prob_bf16(bytes: &[u8]) -> f32 {
    let at = |c: &[u8]| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
    let max = bytes
        .chunks_exact(2)
        .map(at)
        .filter(|x| !x.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let z: f64 = bytes
        .chunks_exact(2)
        .map(at)
        .filter(|x| !x.is_nan())
        .map(|x| ((x - max) as f64).exp())
        .sum();
    (1.0 / z.max(1.0)) as f32
}

/// Parse `ATLAS_QWEN4EXP_MTP_CONFIDENCE`, the drafter's confidence stop
/// (`Qwen4ExpMtpHead::propose`): unset/empty = 0 = off, else a probability in
/// (0, 1). The first draft is always kept; a later draft whose draft-head
/// probability (`top1_prob_bf16` over the draft rows) is below it ends the
/// chain.
pub fn conf_stop_from(v: Option<String>) -> Result<f32> {
    let Some(v) = v.filter(|v| !v.trim().is_empty()) else {
        return Ok(0.0);
    };
    let p: f32 = v
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("ATLAS_QWEN4EXP_MTP_CONFIDENCE={v}: not a number"))?;
    ensure!(
        (0.0..1.0).contains(&p),
        "ATLAS_QWEN4EXP_MTP_CONFIDENCE={v}: a probability in [0, 1)"
    );
    if p > 0.0 {
        tracing::info!("qwen4_exp MTP confidence stop at p < {p} (drafts after the first)");
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_stop_parses_off_and_probabilities() {
        assert_eq!(conf_stop_from(None).unwrap(), 0.0);
        assert_eq!(conf_stop_from(Some(" ".into())).unwrap(), 0.0);
        assert_eq!(conf_stop_from(Some("0.6".into())).unwrap(), 0.6);
        assert!(conf_stop_from(Some("1.5".into())).is_err());
        assert!(conf_stop_from(Some("x".into())).is_err());
    }

    #[test]
    fn ids_parse_sorted_unique_and_bounded() {
        let ids = parse_ids("# header\n5 3, 3\n10\t# tail\n0\n", 11).unwrap();
        assert_eq!(ids, vec![0, 3, 5, 10]);
        assert!(parse_ids("11", 11).is_err());
        assert!(parse_ids("x", 11).is_err());
        assert!(parse_ids("# nothing\n", 11).is_err());
    }

    #[test]
    fn row_of_inverts_the_draft_rows() {
        // The default 100k prefix holds no Qwen end token (<|im_end|> 248046).
        assert_eq!(row_of(None, 100_000, 248_046), None);
        assert_eq!(row_of(None, 100_000, 99_999), Some(99_999));
        // A list maps an id to its sorted row, or to nothing.
        let ids = [0u32, 3, 5, 248_044, 248_046];
        assert_eq!(row_of(Some(&ids), 5, 248_046), Some(4));
        assert_eq!(row_of(Some(&ids), 5, 248_045), None);
    }

    #[test]
    fn runs_cover_each_id_once_in_order() {
        assert_eq!(runs(&[0, 1, 2, 5, 7, 8]), vec![(0, 3), (5, 1), (7, 2)]);
        assert_eq!(runs(&[]), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn top1_prob_matches_a_softmax() {
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let row: Vec<u8> = [2.0f32, 0.0, 0.0].iter().flat_map(|&x| bf(x)).collect();
        let want = 2f32.exp() / (2f32.exp() + 2.0);
        assert!((top1_prob_bf16(&row) - want).abs() < 1e-6);
        let flat: Vec<u8> = [1.0f32; 4].iter().flat_map(|&x| bf(x)).collect();
        assert!((top1_prob_bf16(&flat) - 0.25).abs() < 1e-6);
    }
}
