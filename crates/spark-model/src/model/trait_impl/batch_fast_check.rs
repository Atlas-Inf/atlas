// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_FAST_CHECK=1`: before a batched multi-sequence
//! decode step or batched multi-sequence verify, decode each sequence's rows
//! through single-sequence decode on its live state, put the state back, and
//! compare the batched step's logits and final hidden rows with those byte
//! for byte (design: `model/qwen4exp_batch_fast.rs`). A verify row is
//! checked whether or not its draft is then accepted: the serial pass
//! decodes the drafts too.
//!
//! Every rank runs the check inside its own `decode_batch_compute_main`, the
//! head's and the worker's alike, so the serial steps' collectives pair
//! across ranks as the batched step's do; the switch is carried by
//! `startup_parity`. It reuses the exact-verify check's serial machinery
//! (`exact_verify_check.rs`), one sequence at a time through one scratch.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::qwen4exp_batch_fast::check_active;
use super::super::types::TransformerModel;
use super::exact_verify_check::{argmax, bf16s};
use crate::traits::SequenceState;

const BF16: usize = 2;
/// Summary period, in checked batched steps.
const SUMMARY_EVERY: u64 = 64;

static STEPS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static BAD_ROWS: AtomicU64 = AtomicU64::new(0);
static ARGMAX_FLIPS: AtomicU64 = AtomicU64::new(0);

/// What the serial pass left for the comparison: `n` logits rows, then `n`
/// final-hidden rows, and each row's position.
pub(super) struct BatchSerialRows {
    logits: DevicePtr,
    hidden: DevicePtr,
    positions: Vec<usize>,
}

impl TransformerModel {
    /// Before a batched step over `seqs` (sequence `i` owns `ks[i]` rows of
    /// `tokens`, seq-major; `None` = one row each, a decode): decode each
    /// sequence's rows serially on its live state and keep their logits and
    /// final hidden rows. `None` when the check is off.
    pub(super) fn batch_fast_serial_rows(
        &self,
        tokens: &[u32],
        ks: Option<&[usize]>,
        seqs: &mut [&mut SequenceState],
    ) -> Result<Option<BatchSerialRows>> {
        if !check_active(&self.config.model_type, self.levers.qwen4exp_batch_fast)
            || seqs.is_empty()
        {
            return Ok(None);
        }
        let n = tokens.len();
        let (h, vocab) = (self.config.hidden_size, self.config.vocab_size);
        let (logits, hidden) = self.serial_check_rows(&*seqs[0], n, "BATCH_FAST_CHECK")?;
        let mut positions = Vec::with_capacity(n);
        let mut row = 0usize;
        for (i, seq) in seqs.iter_mut().enumerate() {
            let k = ks.map_or(1, |ks| ks[i]);
            positions.extend((0..k).map(|j| seq.seq_len + j));
            self.serial_decode_restore(
                &tokens[row..row + k],
                seq,
                logits.offset(row * vocab * BF16),
                hidden.offset(row * h * BF16),
            )?;
            row += k;
        }
        Ok(Some(BatchSerialRows {
            logits,
            hidden,
            positions,
        }))
    }

    /// After the batched forward: compare its logits rows (`logits`, row `i`
    /// at `i * vocab`) and final hidden rows (`norm_output`) with the serial
    /// ones, log every mismatching row and a running summary.
    pub(super) fn batch_fast_compare(
        &self,
        serial: BatchSerialRows,
        logits: DevicePtr,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let BatchSerialRows {
            logits: s_logits,
            hidden: s_hidden,
            positions,
        } = serial;
        let n = positions.len();
        let (h, vocab) = (self.config.hidden_size, self.config.vocab_size);
        let read = |p: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
            let mut v = vec![0u8; bytes];
            self.gpu.copy_d2h_on_stream(p, &mut v, stream)?;
            Ok(v)
        };
        let (sl, bl) = (
            read(s_logits, n * vocab * BF16)?,
            read(logits, n * vocab * BF16)?,
        );
        let (sh, bh) = (
            read(s_hidden, n * h * BF16)?,
            read(self.buffers.norm_output(), n * h * BF16)?,
        );
        let mut bad = 0u64;
        let mut flips = 0u64;
        for (t, &pos) in positions.iter().enumerate() {
            let row = |v: &[u8], w: usize| v[t * w * BF16..(t + 1) * w * BF16].to_vec();
            let (s, b) = (bf16s(&row(&sl, vocab)), bf16s(&row(&bl, vocab)));
            let logit_diff = s
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let hidden_diff = row(&sh, h)
                .chunks_exact(2)
                .zip(row(&bh, h).chunks_exact(2))
                .filter(|(x, y)| x != y)
                .count();
            if logit_diff == 0 && hidden_diff == 0 {
                continue;
            }
            bad += 1;
            let max_abs = s
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .filter(|d| d.is_finite())
                .fold(0f32, f32::max);
            let (sa, ba) = (argmax(&s), argmax(&b));
            flips += u64::from(sa != ba);
            tracing::warn!(
                "BATCH_FAST_CHECK row {t}/{n} pos {pos}: {logit_diff}/{vocab} logits and \
                 {hidden_diff}/{h} final-hidden values differ from single-sequence decode \
                 (max |dlogit| {max_abs:.3e}); argmax serial {sa} batched {ba}"
            );
        }
        let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
        let rows = ROWS.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
        let bad_rows = BAD_ROWS.fetch_add(bad, Ordering::Relaxed) + bad;
        let all_flips = ARGMAX_FLIPS.fetch_add(flips, Ordering::Relaxed) + flips;
        if steps.is_multiple_of(SUMMARY_EVERY) || steps == 1 {
            tracing::info!(
                "BATCH_FAST_CHECK summary: {steps} batched steps, {rows} rows, \
                 {bad_rows} rows differ from single-sequence decode, {all_flips} argmax flips"
            );
        }
        Ok(())
    }
}
