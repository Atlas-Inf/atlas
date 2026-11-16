// SPDX-License-Identifier: AGPL-3.0-only

//! The compressed keys of a DeepSeek-V4.1 block over a batch of rows: the
//! keys and index keys of the groups the batch completes, and which
//! compressed positions each row's query attends to.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::{Block, Candidates};
use super::cache_recipe::{bf16_to_f32, group_position, visible_groups};
use super::index::{self, INDEX_DIM, Queries, head_weight_scale};
use super::index_recipe::{candidate_blocks, keep_candidates, top_positions};
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM};
use super::parts;
use super::shape::MAX_RATIO;
use super::side;
use super::state::SeqState;
use super::weights::{IndexKeyWeights, IndexerWeights};

impl Block {
    /// The compressor over `rows` tokens from `position`, whose attention
    /// inputs are `scratch.normed`: project each, pool every group the batch
    /// completes into its key and, on a layer that makes them, its index
    /// key; and keep the tokens of a group it leaves open.
    pub(super) fn compress(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut SeqState,
        position: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        let w = self
            .weights
            .compressor
            .as_ref()
            .context("a source layer has a compressor")?;
        let held = state
            .compressed
            .as_mut()
            .context("a source layer holds compressed keys")?;
        // Tokens of a group the batch before left open come first.
        let (ratio, token) = (w.ratio, HEAD_DIM * 4);
        ensure!(
            ratio <= MAX_RATIO && rows <= sc.rows,
            "layer {}: {rows} rows at {ratio} tokens a key",
            self.index
        );
        let carried = position % ratio;
        let (first, groups) = (position / ratio, (carried + rows) / ratio);
        let open = carried + rows - groups * ratio;
        held.reserve(gpu, first + groups, stream)?;
        let held = &*held;
        let projections = [(Some(w.wkv), held.kv_state, sc.pool_kv)]
            .into_iter()
            .chain([(w.wgate, held.score_state, sc.pool_score)]);
        for (weights, kept, pool) in projections {
            let Some(weights) = weights else { continue };
            if carried > 0 {
                gpu.copy_d2d_async(kept, pool, carried * token, stream)?;
            }
            side::gemv_f32_rows(
                gpu,
                k.side.gemv_f32,
                sc.normed,
                weights,
                pool.offset(carried * token),
                rows,
                HEAD_DIM,
                s.hidden,
                stream,
            )?;
            if open > 0 {
                let from = pool.offset(groups * ratio * token);
                gpu.copy_d2d_async(from, kept, open * token, stream)?;
            }
        }
        if groups == 0 {
            return Ok(());
        }
        match w.wgate {
            Some(_) => side::compress_pool_groups(
                gpu,
                k.side.compress_pool,
                sc.pool_kv,
                sc.pool_score,
                sc.pooled,
                groups,
                ratio,
                HEAD_DIM,
                stream,
            )?,
            // One token a group: nothing to pool, the projection is the key.
            None => moe::round_bf16(
                gpu,
                k.moe.round_bf16,
                sc.pool_kv,
                sc.pooled,
                groups * HEAD_DIM,
                stream,
            )?,
        }
        let keys = held.keys.offset(first * HEAD_DIM * 2);
        self.norm(gpu, sc.pooled, w.norm, keys, groups, HEAD_DIM, stream)?;
        // A group is turned to its first token's position.
        let at: Vec<u8> = (first..first + groups)
            .flat_map(|group| (group_position(group, ratio) as u32).to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&at, sc.group_positions, stream)?;
        // The index key comes from the key as it is before its rotation.
        if let Some(index) = self.weights.indexer.as_ref().and_then(|i| i.key.as_ref()) {
            let index_keys = held.index_keys.offset(first * INDEX_DIM * 2);
            self.index_keys(gpu, index, keys, index_keys, groups, stream)?;
        }
        ops::rotate(
            gpu,
            k.cache.rotate,
            keys,
            sc.group_positions,
            self.freqs,
            groups,
            1,
            HEAD_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp4_quant_dequant(
            gpu,
            k.cache.fp4_quant_dequant,
            keys,
            groups * HEAD_DIM,
            Fp4Scale::E4m3Per16,
            stream,
        )
    }

    /// `keys` (`[groups, 128]` BF16) = the index keys of the groups whose
    /// compressed keys, normed and not yet rotated, are `latents`. Their
    /// positions are in `scratch.group_positions`.
    fn index_keys(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexKeyWeights,
        latents: DevicePtr,
        keys: DevicePtr,
        groups: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        side::gemv_f32_rows(
            gpu,
            k.side.gemv_f32,
            latents,
            w.wk,
            sc.index_key_f32,
            groups,
            INDEX_DIM,
            HEAD_DIM,
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.index_key_f32,
            sc.index_key,
            groups * INDEX_DIM,
            stream,
        )?;
        self.norm(gpu, sc.index_key, w.norm, keys, groups, INDEX_DIM, stream)?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            keys,
            sc.group_positions,
            self.freqs,
            groups,
            1,
            INDEX_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp4_quant_dequant(
            gpu,
            k.cache.fp4_quant_dequant,
            keys,
            groups * INDEX_DIM,
            Fp4Scale::PowerOfTwoPer32,
            stream,
        )
    }

    /// What the index keys score for each of `count` tokens of the batch,
    /// from its row `first` at `position`: a row of `stride` scores a token,
    /// of which the token's visible groups are meaningful.
    #[allow(clippy::too_many_arguments)]
    fn index_scores(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexerWeights,
        keys: DevicePtr,
        first: usize,
        count: usize,
        position: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, usize)> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        let stride = visible_groups(position + count - 1, self.ratio);
        ensure!(
            stride <= sc.max_groups && count <= sc.score_rows && !keys.is_null(),
            "layer {}: {count} queries over {stride} compressed positions, room for {} over {}",
            self.index,
            sc.score_rows,
            sc.max_groups
        );
        ensure!(
            w.wq_b.n == s.index_heads * INDEX_DIM && w.wq_b.k == s.q_rank,
            "indexer wq_b [{}, {}]",
            w.wq_b.n,
            w.wq_b.k
        );
        // The query heads: projected from the query's FP8 form, rotated,
        // rounded to FP4.
        parts::project_rows(
            gpu,
            k.parts.fp8_q,
            sc.q_low_normed.offset(first * s.q_rank * 2),
            w.wq_b.w.data,
            w.wq_b.w.scales,
            sc.index_q,
            2,
            w.wq_b.n,
            count,
            w.wq_b.n,
            w.wq_b.k,
            stream,
        )?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            sc.index_q,
            sc.positions.offset(position * 4),
            self.freqs,
            count,
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
            count * s.index_heads * INDEX_DIM,
            Fp4Scale::PowerOfTwoPer32,
            stream,
        )?;
        // A weight per head, from the attention's input.
        side::gemv_f32_rows(
            gpu,
            k.side.gemv_f32,
            sc.normed.offset(first * s.hidden * 2),
            w.head_weights,
            sc.index_weights_f32,
            count,
            s.index_heads,
            s.hidden,
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.index_weights_f32,
            sc.index_weights,
            count * s.index_heads,
            stream,
        )?;
        let queries = Queries {
            count,
            position,
            ratio: self.ratio,
            stride,
        };
        index::score_rows(
            gpu,
            k.index.score,
            sc.index_q,
            keys,
            sc.index_weights,
            sc.scores_f32,
            s.index_heads,
            queries,
            head_weight_scale(s.index_heads),
            stream,
        )?;
        moe::round_bf16(
            gpu,
            k.moe.round_bf16,
            sc.scores_f32,
            sc.scores,
            count * stride,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; count * stride * 2];
        gpu.copy_d2h(sc.scores, &mut bytes)?;
        let scores = bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Ok((scores, stride))
    }

    /// The compressed positions, of `visible`, that an index source's query
    /// attends to from its `scores`, in ascending order. The candidate source
    /// also leaves the positions the index sources after it may choose from.
    fn choose(&self, scores: &mut [f32], candidates: &mut Vec<bool>) -> Result<Vec<i32>> {
        let s = &self.shared.shape;
        match self.candidates {
            Candidates::None => {}
            Candidates::Source => {
                *candidates = candidate_blocks(scores, s.candidate_blocks, s.candidate_block_size);
            }
            Candidates::Reader => {
                ensure!(
                    candidates.len() == scores.len(),
                    "layer {}: {} candidate flags for {} compressed positions",
                    self.index,
                    candidates.len(),
                    scores.len()
                );
                keep_candidates(scores, candidates);
            }
        }
        Ok(top_positions(scores, s.index_topk))
    }

    /// The compressed keys the queries of `rows` rows from row `row` at
    /// `position` read, and the positions among them each attends to. A
    /// source layer publishes its own for the layers after it; the others
    /// read what the latest source published.
    pub(super) fn compressed_rows(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        rows: usize,
        state: &SeqState,
        position: usize,
        stream: u64,
    ) -> Result<(DevicePtr, Vec<Vec<i32>>)> {
        let (topk, score_rows) = (self.shared.shape.index_topk, self.shared.scratch.score_rows);
        let visible = |r: usize| visible_groups(position + r, self.ratio);
        let mut handoff = self.shared.handoff.lock().expect("handoff");
        if handoff.rows.len() < row + rows {
            handoff.rows.resize_with(row + rows, Default::default);
        }
        let batch = &mut handoff.rows[row..row + rows];
        if let Some(held) = &state.compressed {
            for mine in batch.iter_mut() {
                mine.compressed = held.keys;
                mine.index_keys = held.index_keys;
            }
        }
        let (keys, index_keys) = (batch[0].compressed, batch[0].index_keys);
        ensure!(
            !keys.is_null(),
            "layer {}: no source layer has run for row {row}",
            self.index
        );
        if let Some(w) = &self.weights.indexer {
            // Below `index_topk` positions there is nothing to choose: every
            // position is read, and all are candidates (`Shape::from_config`
            // checks that they would be). The rows that see more come last.
            let scored = (0..rows).find(|&r| visible(r) > topk).unwrap_or(rows);
            for (r, mine) in batch.iter_mut().enumerate().take(scored) {
                if self.candidates == Candidates::Source {
                    mine.candidates = vec![true; visible(r)];
                }
                mine.chosen = (0..visible(r) as i32).collect();
            }
            for first in (scored..rows).step_by(score_rows) {
                let count = (rows - first).min(score_rows);
                let (mut scores, stride) =
                    self.index_scores(gpu, w, index_keys, first, count, position + first, stream)?;
                for (i, mine) in batch[first..first + count].iter_mut().enumerate() {
                    let of_row = &mut scores[i * stride..i * stride + visible(first + i)];
                    mine.chosen = self.choose(of_row, &mut mine.candidates)?;
                }
            }
        }
        let mut chosen = Vec::with_capacity(rows);
        for (r, mine) in batch.iter().enumerate() {
            ensure!(
                mine.chosen.len() == visible(r).min(topk),
                "layer {}: row {} was handed {} compressed positions, and its query sees {}",
                self.index,
                row + r,
                mine.chosen.len(),
                visible(r)
            );
            chosen.push(mine.chosen.clone());
        }
        Ok((keys, chosen))
    }
}

#[cfg(test)]
#[path = "block_index_test_gpu.rs"]
mod test_gpu;
