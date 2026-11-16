// SPDX-License-Identifier: AGPL-3.0-only

//! The indexer of a DeepSeek-V4.1 block on one token: the index key of a
//! group that has just completed, and which compressed positions the token's
//! query attends to.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::{Block, Candidates};
use super::cache_recipe::{bf16_to_f32, visible_groups};
use super::index::{self, INDEX_DIM, head_weight_scale};
use super::index_recipe::{candidate_blocks, keep_candidates, top_positions};
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM};
use super::side;
use super::state::SeqState;
use super::weights::{IndexKeyWeights, IndexerWeights};
use crate::layers::ops::mxfp8_gemv_rows;

impl Block {
    /// `key` (`[128]` BF16) = the index key of the group whose compressed
    /// key, normed and not yet rotated, is at `latent`. `at` holds the
    /// group's position.
    pub(super) fn index_key(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexKeyWeights,
        latent: DevicePtr,
        key: DevicePtr,
        at: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        side::gemv_f32(
            gpu,
            k.side.gemv_f32,
            latent,
            w.wk,
            sc.index_key_f32,
            INDEX_DIM,
            HEAD_DIM,
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.index_key_f32,
            sc.index_key,
            INDEX_DIM,
            stream,
        )?;
        self.norm(gpu, sc.index_key, w.norm, key, INDEX_DIM, stream)?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            key,
            at,
            self.freqs,
            1,
            1,
            INDEX_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp4_quant_dequant(
            gpu,
            k.cache.fp4_quant_dequant,
            key,
            INDEX_DIM,
            Fp4Scale::PowerOfTwoPer32,
            stream,
        )
    }

    /// What each of the first `visible` index keys scores for the token whose
    /// attention input is in `scratch.normed` and whose low-rank query, in
    /// FP8, is in `scratch.q_low_quantized`. `at` holds the token's position.
    fn index_scores(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexerWeights,
        keys: DevicePtr,
        visible: usize,
        at: DevicePtr,
        stream: u64,
    ) -> Result<Vec<f32>> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        ensure!(
            visible <= sc.max_groups && !keys.is_null(),
            "layer {}: {visible} compressed positions to score, room for {}",
            self.index,
            sc.max_groups
        );
        ensure!(
            w.wq_b.n == s.index_heads * INDEX_DIM && w.wq_b.k == s.q_rank,
            "indexer wq_b [{}, {}]",
            w.wq_b.n,
            w.wq_b.k
        );
        // The query heads: projected, rotated, rounded to FP4.
        mxfp8_gemv_rows(
            gpu,
            &k.tiers,
            &w.wq_b.w,
            sc.q_low_quantized,
            sc.index_q,
            1,
            w.wq_b.n,
            w.wq_b.k,
            w.wq_b.n,
            stream,
        )?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            sc.index_q,
            at,
            self.freqs,
            1,
            s.index_heads,
            INDEX_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp4_quant_dequant(
            gpu,
            k.cache.fp4_quant_dequant,
            sc.index_q,
            s.index_heads * INDEX_DIM,
            Fp4Scale::PowerOfTwoPer32,
            stream,
        )?;
        // A weight per head, from the attention's input.
        side::gemv_f32(
            gpu,
            k.side.gemv_f32,
            sc.normed,
            w.head_weights,
            sc.index_weights_f32,
            s.index_heads,
            s.hidden,
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.index_weights_f32,
            sc.index_weights,
            s.index_heads,
            stream,
        )?;
        index::score(
            gpu,
            k.index.score,
            sc.index_q,
            keys,
            sc.index_weights,
            sc.scores_f32,
            s.index_heads,
            visible,
            head_weight_scale(s.index_heads),
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.scores_f32,
            sc.scores,
            visible,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; visible * 2];
        gpu.copy_d2h(sc.scores, &mut bytes)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    /// The compressed positions, of `visible`, that an index source's query
    /// attends to, in ascending order. The candidate source also leaves the
    /// positions the index sources after it may choose from.
    fn choose(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexerWeights,
        keys: DevicePtr,
        candidates: &mut Vec<bool>,
        visible: usize,
        at: DevicePtr,
        stream: u64,
    ) -> Result<Vec<i32>> {
        let s = &self.shared.shape;
        if visible <= s.index_topk {
            // Nothing to choose: every position is read, and all are
            // candidates (`Shape::from_config` checks that they would be).
            if self.candidates == Candidates::Source {
                *candidates = vec![true; visible];
            }
            return Ok((0..visible as i32).collect());
        }
        let mut scores = self.index_scores(gpu, w, keys, visible, at, stream)?;
        match self.candidates {
            Candidates::None => {}
            Candidates::Source => {
                *candidates = candidate_blocks(&scores, s.candidate_blocks, s.candidate_block_size);
            }
            Candidates::Reader => {
                ensure!(
                    candidates.len() == visible,
                    "layer {}: {} candidate flags for {visible} compressed positions",
                    self.index,
                    candidates.len()
                );
                keep_candidates(&mut scores, candidates);
            }
        }
        Ok(top_positions(&scores, s.index_topk))
    }

    /// The compressed keys the query of row `row` at `position` reads, and
    /// the positions among them it attends to. A source layer publishes its
    /// own for the layers after it; the others read what the latest source
    /// published. `at` holds `position`.
    pub(super) fn compressed_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &SeqState,
        position: usize,
        at: DevicePtr,
        stream: u64,
    ) -> Result<(DevicePtr, Vec<i32>)> {
        let visible = visible_groups(position, self.ratio);
        let mut handoff = self.shared.handoff.lock().expect("handoff");
        if handoff.rows.len() <= row {
            handoff.rows.resize_with(row + 1, Default::default);
        }
        let mine = &mut handoff.rows[row];
        if let Some(held) = &state.compressed {
            mine.compressed = held.keys;
            mine.index_keys = held.index_keys;
        }
        ensure!(
            !mine.compressed.is_null(),
            "layer {}: no source layer has run for row {row}",
            self.index
        );
        if let Some(w) = &self.weights.indexer {
            let keys = mine.index_keys;
            mine.chosen = self.choose(gpu, w, keys, &mut mine.candidates, visible, at, stream)?;
        }
        ensure!(
            mine.chosen.len() == visible.min(self.shared.shape.index_topk),
            "layer {}: row {row} was handed {} compressed positions, and its query sees {visible}",
            self.index,
            mine.chosen.len()
        );
        Ok((mine.compressed, mine.chosen.clone()))
    }
}

#[cfg(test)]
#[path = "block_index_test_gpu.rs"]
mod test_gpu;
