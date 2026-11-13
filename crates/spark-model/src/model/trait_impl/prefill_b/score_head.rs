// SPDX-License-Identifier: AGPL-3.0-only

//! The head of a score pass (`score.rs`): final norm on each branch's last
//! row, then the option logits — gathered FP32 dot products from a BF16 or
//! NVFP4 LM head, else (FP8 head) the full-vocabulary head — and a softmax
//! over each branch's options.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::score_plan::ScorePlan;

impl TransformerModel {
    /// Final norm on each branch's last row, LM head, softmax over options.
    pub(super) fn score_head(
        &self,
        plan: &ScorePlan,
        options: &[Vec<u32>],
        stream: u64,
    ) -> Result<Vec<Vec<f32>>> {
        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let hidden = self.buffers.hidden_states();
        let normed = self.buffers.norm_output();
        let mut last_rows = Vec::with_capacity(options.len());
        let mut end = 0usize;
        for i in 0..plan.streams() {
            end += plan.rows(i);
            if i >= plan.heads {
                last_rows.push(end - 1);
            }
        }
        for (j, &row) in last_rows.iter().enumerate() {
            self.final_norm_rows(
                hidden.offset(row * h * bf16),
                normed.offset(j * h * bf16),
                1,
                stream,
            )?;
        }
        if self.lm_head_nvfp4.is_none() && self.lm_head_fp8.is_none() {
            let head = self.lm_head_weight.weight;
            return self.score_head_gathered(normed, options, stream, head.0, |t, buf| {
                let mut raw = vec![0u8; h * 2];
                self.gpu
                    .copy_d2h(head.offset(t as usize * h * 2), &mut raw)?;
                *buf = bf16_to_f32(&raw);
                Ok(())
            });
        }
        if let Some(q) = self.lm_head_nvfp4.as_ref() {
            // A pre-packed NVFP4 head (NVIDIA's checkpoints) has no BF16 copy:
            // dequantize just the option rows instead of running the
            // full-vocabulary GEMM over every branch row.
            return self.score_head_gathered(normed, options, stream, q.weight.0, |t, buf| {
                *buf = self.nvfp4_head_row(q, t, h)?;
                Ok(())
            });
        }
        // The logits arena holds a bounded number of rows; run the LM head
        // in groups of that many and keep only each branch's option logits.
        let group = self.buffers.sizes().logits / (v * bf16);
        anyhow::ensure!(group > 0, "score: the logits arena holds no full row");
        let mut out = Vec::with_capacity(options.len());
        let mut row = vec![0u8; v * bf16];
        for g in (0..last_rows.len()).step_by(group) {
            let k = group.min(last_rows.len() - g);
            let dst = self.buffers.logits();
            self.lm_head_batched(normed.offset(g * h * bf16), k as u32, dst, stream)?;
            self.gpu.synchronize(stream)?;
            for j in 0..k {
                self.gpu.copy_d2h(dst.offset(j * v * bf16), &mut row)?;
                out.push(softmax_options(&row, &options[g + j]));
            }
        }
        Ok(out)
    }
}

impl TransformerModel {
    /// The option logits only, in FP32, from the LM head rows `load` returns
    /// (BF16 `--lm-head-dtype bf16`, or a dequantized NVFP4 row): `N x K` dot
    /// products instead of a full-vocabulary head, and no full rows copied to
    /// the host. `head_key` identifies the head the cached rows came from.
    fn score_head_gathered(
        &self,
        normed: DevicePtr,
        options: &[Vec<u32>],
        stream: u64,
        head_key: u64,
        load: impl Fn(u32, &mut Vec<f32>) -> Result<()>,
    ) -> Result<Vec<Vec<f32>>> {
        let h = self.config.hidden_size;
        let mut rows = vec![0u8; options.len() * h * 2];
        self.gpu.synchronize(stream)?;
        self.gpu.copy_d2h(normed, &mut rows)?;
        let mut cache = self
            .score_head_rows
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache.0 != head_key {
            *cache = (head_key, std::collections::HashMap::new());
        }
        let mut out = Vec::with_capacity(options.len());
        for (j, opts) in options.iter().enumerate() {
            let x = bf16_to_f32(&rows[j * h * 2..(j + 1) * h * 2]);
            let mut logits = Vec::with_capacity(opts.len());
            for &t in opts {
                // Bounded: start over rather than grow past the cap.
                if cache.1.len() >= MAX_HEAD_ROWS && !cache.1.contains_key(&t) {
                    cache.1.clear();
                }
                let w = match cache.1.entry(t) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let mut w = Vec::new();
                        load(t, &mut w)?;
                        e.insert(w)
                    }
                };
                logits.push(x.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>());
            }
            out.push(softmax_f32(&logits));
        }
        Ok(out)
    }
}

impl TransformerModel {
    /// Row `t` of a pre-packed NVFP4 LM head, dequantized to FP32: packed
    /// E2M1 `[N, K/2]` (low nibble = even k), E4M3 group scales `[N, K/16]`,
    /// and the per-tensor (or per-row) FP32 scale 2. The same values the
    /// `w4a16_gemv` LUT path multiplies.
    fn nvfp4_head_row(
        &self,
        q: &crate::weight_map::QuantizedWeight,
        t: u32,
        h: usize,
    ) -> Result<Vec<f32>> {
        let row = t as usize;
        let mut packed = vec![0u8; h / 2];
        self.gpu
            .copy_d2h(q.weight.offset(row * h / 2), &mut packed)?;
        let mut scales = vec![0u8; h / 16];
        self.gpu
            .copy_d2h(q.weight_scale.offset(row * h / 16), &mut scales)?;
        let scale2 = if q.weight_scale_2_vec.is_null() {
            q.weight_scale_2
        } else {
            let mut b = [0u8; 4];
            self.gpu
                .copy_d2h(q.weight_scale_2_vec.offset(row * 4), &mut b)?;
            f32::from_le_bytes(b)
        };
        Ok(dequant_nvfp4_row(&packed, &scales, scale2))
    }
}

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP E4M3 (e4m3fn) to f32; the NaN code reads as 0, like the kernels' decode.
fn e4m3_to_f32(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 0x7);
    let v = if e == 0 {
        m as f32 * 0.001_953_125
    } else if e == 15 && m == 7 {
        0.0
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20))
    };
    if s == 1 { -v } else { v }
}

fn dequant_nvfp4_row(packed: &[u8], scales: &[u8], scale2: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for (i, &b) in packed.iter().enumerate() {
        let s = e4m3_to_f32(scales[i / 8]) * scale2;
        out.push(E2M1[(b & 0xF) as usize] * s);
        out.push(E2M1[(b >> 4) as usize] * s);
    }
    out
}

/// Most LM-head rows a model keeps on the host (20 KiB each at hidden 5120):
/// option sets repeat, so a small cache serves them all.
const MAX_HEAD_ROWS: usize = 1024;

/// LM-head rows already read to the host, keyed by token id, for the head
/// they came from (its device pointer). Owned by the model, so a reloaded
/// model never sees another model's rows; cleared on release.
#[derive(Default)]
pub(in crate::model) struct HeadRows(
    std::sync::Mutex<(u64, std::collections::HashMap<u32, Vec<f32>>)>,
);

impl HeadRows {
    pub(in crate::model) fn clear(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
    }
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn softmax_f32(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let es: Vec<f32> = xs.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = es.iter().sum();
    es.into_iter().map(|e| e / s).collect()
}

/// Softmax over the BF16 logits of `opts`, in f32 (max-subtracted).
pub(super) fn softmax_options(row_bf16: &[u8], opts: &[u32]) -> Vec<f32> {
    let logit = |t: u32| {
        let i = t as usize * 2;
        f32::from_bits((u16::from_le_bytes([row_bf16[i], row_bf16[i + 1]]) as u32) << 16)
    };
    let xs: Vec<f32> = opts.iter().map(|&t| logit(t)).collect();
    softmax_f32(&xs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_row(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn nvfp4_row_dequant_matches_layout() {
        // 32 values = 16 packed bytes, two 16-value groups with scales 1.0 (0x38) and 2.0 (0x40).
        let mut packed = vec![0u8; 16];
        packed[0] = 0x2 | (0x9 << 4); // k0 = 1.0, k1 = -0.5
        packed[8] = 0x7; // k16 = 6.0 (second group)
        let row = dequant_nvfp4_row(&packed, &[0x38, 0x40], 0.5);
        assert_eq!(row.len(), 32);
        assert_eq!(row[0], 0.5);
        assert_eq!(row[1], -0.25);
        assert_eq!(row[16], 6.0);
        assert_eq!(e4m3_to_f32(0x7F), 0.0);
        assert_eq!(e4m3_to_f32(0x01), 0.001_953_125);
    }

    #[test]
    fn softmax_picks_option_logits_only() {
        let row = bf16_row(&[9.0, 1.0, 0.0, 1.0, -50.0]);
        let p = softmax_options(&row, &[1, 3]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        let p = softmax_options(&row, &[2, 4]);
        assert!(p[0] > 0.999_999 && (p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
