// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 bilinear CandidateSelector.
//!
//! Evaluates candidate tokens along the sequential chain from the anchor:
//! `Score(b) = Unary(b) + <Pred[prev] ⊙ H_proj(h_t), Succ[b]>` applied to the
//! top-k unary candidates at each step t; the highest-scoring candidate
//! advances the chain.

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::layers::ops::{DFLASH2_SELECTOR_MAX_RANK, DFLASH2_SELECTOR_MAX_TOP_K};
use crate::weight_map::DenseWeight;

#[derive(Clone)]
pub struct Dflash2CandidateSelector {
    pub hidden_projection: DenseWeight,
    pub predecessor_codebook: DenseWeight,
    pub successor_codebook: DenseWeight,
    pub predecessor_host: Option<Vec<bf16>>,
    pub successor_host: Option<Vec<bf16>>,
    pub rank: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl Dflash2CandidateSelector {
    pub fn new(
        hidden_projection: DenseWeight,
        predecessor_codebook: DenseWeight,
        successor_codebook: DenseWeight,
        rank: usize,
        top_k: usize,
        vocab_size: usize,
        hidden_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        // The on-device selector's shared-memory arrays are compile-time
        // sized; a checkpoint past the caps must fail loudly, not silently
        // degrade to a truncated selector.
        anyhow::ensure!(
            (1..=DFLASH2_SELECTOR_MAX_RANK).contains(&rank),
            "DFlash2 selector rank {rank} exceeds the on-device selector's cap {DFLASH2_SELECTOR_MAX_RANK}"
        );
        anyhow::ensure!(
            (1..=DFLASH2_SELECTOR_MAX_TOP_K).contains(&top_k),
            "DFlash2 selector top_k {top_k} exceeds the on-device selector's cap {DFLASH2_SELECTOR_MAX_TOP_K}"
        );
        let n_elements = vocab_size * rank;
        let mut pred_buf = vec![0u8; n_elements * 2];
        let mut succ_buf = vec![0u8; n_elements * 2];

        gpu.copy_d2h(predecessor_codebook.weight, &mut pred_buf)?;
        gpu.copy_d2h(successor_codebook.weight, &mut succ_buf)?;

        let predecessor_host: Vec<bf16> = pred_buf
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let successor_host: Vec<bf16> = succ_buf
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        Ok(Self {
            hidden_projection,
            predecessor_codebook,
            successor_codebook,
            predecessor_host: Some(predecessor_host),
            successor_host: Some(successor_host),
            rank,
            top_k,
            vocab_size,
            hidden_size,
        })
    }

    /// Select candidate tokens using the bilinear codebook path walk.
    pub fn select_candidates(
        &self,
        last_token: u32,
        hidden_buf: DevicePtr,
        logits_buf: DevicePtr,
        projected_hidden_buf: DevicePtr,
        draft_tokens_dev: DevicePtr,
        gamma: usize,
        gpu: &dyn GpuBackend,
        gemm: &dyn Fn(DevicePtr, &DenseWeight, DevicePtr, u32, u32, u32) -> Result<()>,
        candidate_selector_kernel: Option<KernelHandle>,
        stream: u64,
    ) -> Result<()> {
        let r = self.rank as u32;
        let h = self.hidden_size as u32;
        let g = gamma as u32;

        // 1. GEMM: projected_hidden_buf [γ, rank] = hidden_buf [γ, H] @ hidden_projection^T [rank, H]
        //    `gemm` is `drafter_dense_gemm`: γ-row fits the batched GEMV arm.
        gemm(
            hidden_buf,
            &self.hidden_projection,
            projected_hidden_buf,
            g,
            r,
            h,
        )?;

        // 2. On-Device GPU Candidate Selector (Zero D2H, Zero CPU Loop, <0.05ms)
        if let Some(kernel) = candidate_selector_kernel {
            return ops::dflash2_candidate_selector(
                gpu,
                kernel,
                logits_buf,
                projected_hidden_buf,
                self.predecessor_codebook.weight,
                self.successor_codebook.weight,
                draft_tokens_dev,
                last_token,
                g,
                self.vocab_size as u32,
                r,
                self.top_k as u32,
                stream,
            );
        }

        // Host fallback (if GPU kernel unavailable)
        let mut proj_bytes = vec![0u8; gamma * self.rank * 2];
        let mut logits_bytes = vec![0u8; gamma * self.vocab_size * 2];

        gpu.synchronize(stream)?;
        gpu.copy_d2h(projected_hidden_buf, &mut proj_bytes)?;
        gpu.copy_d2h(logits_buf, &mut logits_bytes)?;

        let proj_hiddens: Vec<f32> = proj_bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();

        let pred_codebook = self.predecessor_host.as_ref().expect("predecessor host");
        let succ_codebook = self.successor_host.as_ref().expect("successor host");

        // Row 0 is the anchor token. Rows 1..gamma are the mask tokens.
        // Step 0..gamma-1 selects candidate for mask rows 1..gamma, with
        // step 0's predecessor = last_token.
        let num_draft_steps = gamma.saturating_sub(1);
        let mut mask_drafts = Vec::with_capacity(num_draft_steps);
        let mut prev_token = last_token as usize;

        for step in 0..num_draft_steps {
            let mask_row = step + 1;
            let logit_row_offset = mask_row * self.vocab_size * 2;
            let logit_slice =
                &logits_bytes[logit_row_offset..logit_row_offset + self.vocab_size * 2];

            // Extract top-k candidate token IDs from unary logits, ordered
            // (value desc, index asc) — the same total order the kernel uses.
            let logits_row: Vec<f32> = logit_slice
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            let top_candidates = topk_unary(&logits_row, self.top_k);

            // Context vector: c_r = pred[prev, r] * H_proj[mask_row, r]
            let h_step = &proj_hiddens[mask_row * self.rank..(mask_row + 1) * self.rank];
            let pred_row = &pred_codebook[prev_token.min(self.vocab_size - 1) * self.rank
                ..(prev_token.min(self.vocab_size - 1) + 1) * self.rank];

            let mut context = vec![0.0f32; self.rank];
            for i in 0..self.rank {
                context[i] = pred_row[i].to_f32() * h_step[i];
            }

            // Score candidates; strict `>` in list order, so a score tie
            // resolves to the earlier (higher-unary) candidate — same rule
            // as the kernel's tid-0 pick.
            let (best_token, best_score) = pick_best(&top_candidates, &mut |cand_id| {
                let succ_row = &succ_codebook[cand_id * self.rank..(cand_id + 1) * self.rank];
                let mut dot = 0.0f32;
                for i in 0..self.rank {
                    dot += context[i] * succ_row[i].to_f32();
                }
                dot
            });

            if step == 0 {
                let unary_top = top_candidates.first().map(|c| c.1).unwrap_or(0);
                tracing::trace!(
                    "DFLASH SELECTOR draft 0: prev_token={} unary_top1={} best_token={} best_score={:.2}",
                    prev_token,
                    unary_top,
                    best_token,
                    best_score,
                );
            }

            mask_drafts.push(best_token as u32);
            prev_token = best_token;
        }

        // Anchor row 0: unary argmax
        let anchor_logit_slice = &logits_bytes[0..self.vocab_size * 2];
        let mut anchor_best_val = f32::NEG_INFINITY;
        let mut anchor_token = 0usize;
        for (idx, chunk) in anchor_logit_slice.chunks_exact(2).enumerate() {
            let val = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            if val > anchor_best_val {
                anchor_best_val = val;
                anchor_token = idx;
            }
        }

        // Write row_order to device: row 0 (anchor), then rows 1..gamma (mask drafts).
        // forward_block's (1..gamma).chain(once(0)) will place mask drafts first in verify order.
        let mut dev_bytes = Vec::with_capacity(gamma * 4);
        dev_bytes.extend_from_slice(&(anchor_token as u32).to_le_bytes());
        for token in mask_drafts {
            dev_bytes.extend_from_slice(&token.to_le_bytes());
        }
        gpu.copy_h2d(&dev_bytes, draft_tokens_dev)?;

        Ok(())
    }
}

/// Top-k of a unary logit row under the selector's total order: value
/// descending, index ascending (the lower vocab index wins ties) —
/// identical to the on-device kernel's insert/merge order.
fn topk_unary(logits: &[f32], top_k: usize) -> Vec<(f32, usize)> {
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(top_k + 1);
    for (idx, &val) in logits.iter().enumerate() {
        if top.len() == top_k {
            let last = top[top_k - 1];
            let better = val > last.0 || (val == last.0 && idx < last.1);
            if !better {
                continue;
            }
        }
        top.push((val, idx));
        top.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        top.truncate(top_k);
    }
    top
}

/// Score each candidate as `unary + context·succ[cand]` and pick the max by
/// strict `>` in list order — a score tie resolves to the earlier
/// (higher-unary) candidate, matching the kernel's tid-0 pick.
fn pick_best(top_candidates: &[(f32, usize)], dot: &mut dyn FnMut(usize) -> f32) -> (usize, f32) {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = top_candidates.first().map(|c| c.1).unwrap_or(0);
    for &(unary, cand_id) in top_candidates {
        let total = unary + dot(cand_id);
        if total > best_score {
            best_score = total;
            best_token = cand_id;
        }
    }
    (best_token, best_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact ties in the unary row must order (value desc, index asc) — the
    /// kernel's total order and the engine's first-index-wins contract.
    #[test]
    fn topk_orders_value_desc_then_index_asc() {
        let logits = [0.5f32, 1.0, -2.0, 1.0, 1.0, 0.25];
        let top = topk_unary(&logits, 3);
        assert_eq!(top, vec![(1.0, 1), (1.0, 3), (1.0, 4)]);
    }

    /// Fewer logits than k: everything is a candidate, still ordered.
    #[test]
    fn topk_shorter_than_k() {
        let logits = [-1.0f32, 3.0, 0.0];
        let top = topk_unary(&logits, 16);
        assert_eq!(top, vec![(3.0, 1), (0.0, 2), (-1.0, 0)]);
    }

    /// With a zero context every candidate's score is its unary value, so
    /// the pick must be the lowest-index max — not whichever tied entry the
    /// scan happened to see last.
    #[test]
    fn pick_best_under_zero_context_is_lowest_index_max() {
        let logits = [0.5f32, 2.0, -1.0, 2.0, 2.0, 0.1];
        let top = topk_unary(&logits, 4);
        let (best, _) = pick_best(&top, &mut |_| 0.0);
        assert_eq!(best, 1);
    }

    /// A context that flips the ranking must beat the unary leader.
    #[test]
    fn pick_best_context_can_overtake_unary() {
        let top = vec![(2.0f32, 7usize), (1.9f32, 3usize)];
        let (best, score) = pick_best(&top, &mut |c| if c == 3 { 0.5 } else { 0.0 });
        assert_eq!(best, 3);
        assert!((score - 2.4).abs() < 1e-6);
    }
}
