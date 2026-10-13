// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK=1`: run a K=2/3/4 verify's tokens
//! through serial decode first, put the state back, and compare the verify's
//! rows with the serial ones byte for byte (design and how to read it:
//! `model/qwen4exp_exact_verify.rs`).
//!
//! Every rank runs the check inside its own verify dispatch, so the serial
//! steps' collectives pair across ranks like the verify's do; the switch is
//! carried by `startup_parity`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;

use super::super::qwen4exp_exact_verify::check_active;
use super::super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::SequenceState;

const BF16: usize = 2;
/// Widest verify the check serves (`verify_c`).
const MAX_ROWS: usize = 4;
/// Summary period, in checked verify steps.
const SUMMARY_EVERY: u64 = 64;

/// One process-lifetime device scratch: the GDN states of one sequence, then
/// `MAX_ROWS` logits rows and final-hidden rows. Allocated on first use.
static SCRATCH: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);

static STEPS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static BAD_ROWS: AtomicU64 = AtomicU64::new(0);
static ARGMAX_FLIPS: AtomicU64 = AtomicU64::new(0);

/// What the serial pass left for the comparison.
pub(super) struct SerialRows {
    k: usize,
    logits: DevicePtr,
    hidden: DevicePtr,
}

impl TransformerModel {
    /// Before a K-row verify of `tokens`: decode them serially on the live
    /// state, keep each step's logits and final hidden row, and restore the
    /// state. `None` when the check is off.
    pub(super) fn exact_verify_serial_rows(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
    ) -> Result<Option<SerialRows>> {
        if !check_active(&self.config.model_type) {
            return Ok(None);
        }
        let k = tokens.len();
        ensure!(
            k <= MAX_ROWS,
            "exact verify check: {k} rows, serves {MAX_ROWS}"
        );
        let (logits, hidden) = self.serial_check_rows(seq, MAX_ROWS, "EXACT_VERIFY_CHECK")?;
        self.serial_decode_restore(tokens, seq, logits, hidden)?;
        Ok(Some(SerialRows { k, logits, hidden }))
    }

    /// The check scratch, grown to hold one sequence's GDN states plus `rows`
    /// logits and final-hidden rows: returns where the rows go.
    pub(super) fn serial_check_rows(
        &self,
        seq: &SequenceState,
        rows: usize,
        who: &str,
    ) -> Result<(DevicePtr, DevicePtr)> {
        ensure!(
            !self.use_fp32_logits,
            "{who}: FP32 decode logits are not the checked step's BF16 rows"
        );
        let (h, vocab) = (self.config.hidden_size, self.config.vocab_size);
        let rows_off = self.serial_state_bytes(seq).next_multiple_of(256);
        let need = rows_off + rows * (vocab + h) * BF16;
        let base = {
            let mut slot = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            match *slot {
                Some((p, n)) if n >= need => p,
                _ => {
                    if let Some((p, _)) = slot.take() {
                        self.gpu.free(p)?;
                    }
                    let p = self.gpu.alloc(need)?;
                    tracing::warn!(
                        "{who} armed: {:.1} MiB scratch, every checked step is \
                         preceded by serial decode steps (diagnostic, slow)",
                        need as f64 / 1048576.0
                    );
                    *slot = Some((p, need));
                    p
                }
            }
        };
        let logits = base.offset(rows_off);
        Ok((logits, logits.offset(rows * vocab * BF16)))
    }

    /// One sequence's GDN recurrent state (h + conv per layer).
    fn serial_ssm_states(&self, seq: &SequenceState) -> Vec<(DevicePtr, DevicePtr)> {
        seq.layer_states
            .iter()
            .enumerate()
            .filter(|(i, _)| self.config.layer_type(*i) == LayerType::LinearAttention)
            .filter_map(|(_, s)| s.as_any().downcast_ref::<SsmLayerState>())
            .map(|s| (s.h_state, s.conv_state))
            .collect()
    }

    fn serial_state_bytes(&self, seq: &SequenceState) -> usize {
        let per = self.ssm_pool.h_stored_bytes + self.config.ssm_conv_state_bytes();
        self.serial_ssm_states(seq).len() * per
    }

    /// Decode `tokens` serially on `seq`'s live state, landing step `t`'s
    /// logits row at `logits + t * vocab` and its final hidden row at
    /// `hidden + t * h`, then put the sequence back exactly where it was
    /// (GDN h/conv copied back through the check scratch, PLE carry restored
    /// from its aux blob, QSA ingest rewound as a rejected draft is). The
    /// rows must sit past the scratch's state region
    /// ([`Self::serial_check_rows`] lays them out).
    pub(super) fn serial_decode_restore(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        logits: DevicePtr,
        hidden: DevicePtr,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let k = tokens.len();
        let h_bytes = self.ssm_pool.h_stored_bytes;
        let conv_bytes = self.config.ssm_conv_state_bytes();
        let ssm = self.serial_ssm_states(seq);
        let base = SCRATCH
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map(|(p, _)| p)
            .ok_or_else(|| anyhow::anyhow!("serial check: scratch not allocated"))?;
        let save = |i: usize| base.offset(i * (h_bytes + conv_bytes));
        for (i, &(hs, cs)) in ssm.iter().enumerate() {
            self.gpu.copy_d2d_async(hs, save(i), h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(cs, save(i).offset(h_bytes), conv_bytes, stream)?;
        }
        // Host-side carries: PLE's n-gram history + conv (an aux blob). QSA's
        // ingest is rewound below exactly as a rejected draft is.
        let mut ple_blobs: Vec<(usize, Vec<u8>)> = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention && l.has_aux_state() {
                let mut blob = Vec::new();
                if l.snapshot_aux_into(
                    seq.layer_states[i].as_ref(),
                    &mut blob,
                    self.gpu.as_ref(),
                    stream,
                )? {
                    ple_blobs.push((i, blob));
                }
            }
        }
        let tokens_len = seq.tokens.len();
        let seq_len = seq.seq_len;

        for (t, &tok) in tokens.iter().enumerate() {
            let out = self.decode_dispatch(tok, seq, stream)?;
            self.gpu
                .copy_d2d_async(out, logits.offset(t * vocab * BF16), vocab * BF16, stream)?;
            self.gpu.copy_d2d_async(
                self.buffers.norm_output(),
                hidden.offset(t * h * BF16),
                h * BF16,
                stream,
            )?;
        }

        // Put the sequence back where the checked step expects it.
        seq.tokens.truncate(tokens_len);
        seq.seq_len = seq_len;
        for (i, &(hs, cs)) in ssm.iter().enumerate() {
            self.gpu.copy_d2d_async(save(i), hs, h_bytes, stream)?;
            self.gpu
                .copy_d2d_async(save(i).offset(h_bytes), cs, conv_bytes, stream)?;
        }
        for (i, blob) in &ple_blobs {
            self.layers[*i].restore_aux(
                seq.layer_states[*i].as_mut(),
                blob,
                self.gpu.as_ref(),
                stream,
            )?;
        }
        for (i, l) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::FullAttention {
                l.rollback_aux_verify(
                    seq.layer_states[i].as_mut(),
                    0,
                    k,
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        // The restores read host blobs; finish them before those drop.
        self.gpu.synchronize(stream)?;
        Ok(())
    }

    /// After the verify forward: compare its logits and final hidden rows
    /// with the serial ones, log every mismatching row and a running summary.
    pub(super) fn exact_verify_compare(&self, serial: SerialRows, seq_len: usize) -> Result<()> {
        let stream = self.gpu.default_stream();
        let SerialRows { k, logits, hidden } = serial;
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let read = |p: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
            let mut v = vec![0u8; bytes];
            self.gpu.copy_d2h_on_stream(p, &mut v, stream)?;
            Ok(v)
        };
        let (sl, vl) = (
            read(logits, k * vocab * BF16)?,
            read(self.buffers.logits(), k * vocab * BF16)?,
        );
        let (sh, vh) = (
            read(hidden, k * h * BF16)?,
            read(self.buffers.norm_output(), k * h * BF16)?,
        );
        let mut bad = 0u64;
        let mut flips = 0u64;
        for t in 0..k {
            let row = |v: &[u8], w: usize| v[t * w * BF16..(t + 1) * w * BF16].to_vec();
            let (s, v) = (bf16s(&row(&sl, vocab)), bf16s(&row(&vl, vocab)));
            let (shid, vhid) = (row(&sh, h), row(&vh, h));
            let logit_diff = s
                .iter()
                .zip(&v)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let hidden_diff = shid
                .chunks_exact(2)
                .zip(vhid.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            if logit_diff == 0 && hidden_diff == 0 {
                continue;
            }
            bad += 1;
            let max_abs = s
                .iter()
                .zip(&v)
                .map(|(a, b)| (a - b).abs())
                .filter(|d| d.is_finite())
                .fold(0f32, f32::max);
            let (sa, va) = (argmax(&s), argmax(&v));
            flips += u64::from(sa != va);
            tracing::warn!(
                "EXACT_VERIFY_CHECK pos {} row {t}/{k}: {logit_diff}/{vocab} logits and \
                 {hidden_diff}/{h} final-hidden values differ (max |dlogit| {max_abs:.3e}); \
                 argmax serial {sa} verify {va}",
                seq_len + t,
            );
        }
        let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
        let rows = ROWS.fetch_add(k as u64, Ordering::Relaxed) + k as u64;
        let bad_rows = BAD_ROWS.fetch_add(bad, Ordering::Relaxed) + bad;
        let all_flips = ARGMAX_FLIPS.fetch_add(flips, Ordering::Relaxed) + flips;
        if steps.is_multiple_of(SUMMARY_EVERY) {
            tracing::info!(
                "EXACT_VERIFY_CHECK summary: {steps} verify steps, {rows} rows, \
                 {bad_rows} rows differ from serial decode, {all_flips} argmax flips"
            );
        }
        Ok(())
    }
}

pub(super) fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Lowest index of the maximum, the tie rule of `argmax_bf16` and the host
/// greedy sampler.
pub(super) fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{argmax, bf16s};

    #[test]
    fn argmax_takes_the_lowest_index_of_a_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn bf16s_widens_little_endian_rows() {
        // 1.0 = 0x3f80, -2.0 = 0xc000
        assert_eq!(bf16s(&[0x80, 0x3f, 0x00, 0xc0]), vec![1.0, -2.0]);
    }
}
