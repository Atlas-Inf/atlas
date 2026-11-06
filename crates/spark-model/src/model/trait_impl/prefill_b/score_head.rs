// SPDX-License-Identifier: AGPL-3.0-only

//! The head of a score pass (`score.rs`): final norm on each branch's last
//! row, then the option logits — gathered FP32 dot products from a BF16 LM
//! head when one is kept, else the full-vocabulary head — and a softmax over
//! each branch's options.

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
            return self.score_head_gathered(normed, options, stream);
        }
        // The logits arena holds a bounded number of rows; run the LM head
        // in groups and keep only each branch's option logits.
        const GROUP: usize = 32;
        let mut out = Vec::with_capacity(options.len());
        let mut row = vec![0u8; v * bf16];
        for g in (0..last_rows.len()).step_by(GROUP) {
            let k = GROUP.min(last_rows.len() - g);
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
    /// The option logits only, in FP32, from the BF16 LM head
    /// (`--lm-head-dtype bf16`): `N x K` dot products instead of a
    /// full-vocabulary head, and no full rows copied to the host.
    fn score_head_gathered(
        &self,
        normed: DevicePtr,
        options: &[Vec<u32>],
        stream: u64,
    ) -> Result<Vec<Vec<f32>>> {
        let h = self.config.hidden_size;
        let mut rows = vec![0u8; options.len() * h * 2];
        self.gpu.synchronize(stream)?;
        self.gpu.copy_d2h(normed, &mut rows)?;
        let head = self.lm_head_weight.weight;
        let mut cache = HEAD_ROWS.lock().unwrap_or_else(|e| e.into_inner());
        if cache.0 != head.0 {
            *cache = (head.0, std::collections::HashMap::new());
        }
        let mut buf = vec![0u8; h * 2];
        let mut out = Vec::with_capacity(options.len());
        for (j, opts) in options.iter().enumerate() {
            let x = bf16_to_f32(&rows[j * h * 2..(j + 1) * h * 2]);
            let mut logits = Vec::with_capacity(opts.len());
            for &t in opts {
                if !cache.1.contains_key(&t) {
                    self.gpu.copy_d2h(head.offset(t as usize * h * 2), &mut buf)?;
                    cache.1.insert(t, bf16_to_f32(&buf));
                }
                let w = &cache.1[&t];
                logits.push(x.iter().zip(w).map(|(a, b)| a * b).sum::<f32>());
            }
            out.push(softmax_f32(&logits));
        }
        Ok(out)
    }
}

/// LM-head rows already read to the host, keyed by token id, for the head
/// they came from (its device pointer).
type HeadRows = (u64, std::collections::HashMap<u32, Vec<f32>>);
static HEAD_ROWS: std::sync::LazyLock<std::sync::Mutex<HeadRows>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new((0, std::collections::HashMap::new())));

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
    fn softmax_picks_option_logits_only() {
        let row = bf16_row(&[9.0, 1.0, 0.0, 1.0, -50.0]);
        let p = softmax_options(&row, &[1, 3]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6);
        let p = softmax_options(&row, &[2, 4]);
        assert!(p[0] > 0.999_999 && (p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
