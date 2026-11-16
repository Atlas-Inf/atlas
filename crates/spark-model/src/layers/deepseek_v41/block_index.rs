// SPDX-License-Identifier: AGPL-3.0-only

//! Which compressed positions the queries of a batch attend to: the
//! indexer's scores and the choice among them. A row alone is chosen for on
//! the host (`index_recipe.rs`); a batch on the device (`dsv41_choose.cu`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::{Block, Candidates};
use super::cache_recipe::{bf16_to_f32, visible_groups};
use super::gemm::{GEMM_ROWS, Input};
use super::index::{self, INDEX_DIM, Queries, head_weight_scale};
use super::index_recipe::{candidate_blocks, keep_candidates, top_positions};
use super::moe;
use super::ops::{self, Fp4Scale};
use super::parts;
use super::state::SeqState;
use super::weights::IndexerWeights;

/// The compressed positions the queries of a batch read.
pub(super) enum Chosen {
    /// One query's, on the host, ascending.
    Host(Vec<i32>),
    /// On the device: `[rows, index_topk]` i32, a row ascending and then
    /// -1. Null on a layer that attends over its window only.
    Table(DevicePtr),
}

impl Block {
    /// The indexer's queries for `count` tokens of the batch, from its row
    /// `first` at `position`, in `scratch.index_q` and
    /// `scratch.index_weights`; and how many compressed positions the last
    /// of them sees.
    fn index_queries(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexerWeights,
        first: usize,
        count: usize,
        position: usize,
        stream: u64,
    ) -> Result<Queries> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        let stride = visible_groups(position + count - 1, self.ratio);
        ensure!(
            stride <= sc.max_groups && count <= sc.score_rows,
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
        let q_low = sc.q_low_normed.offset(first * s.q_rank * 2);
        let gemm = self.gemm.as_ref().filter(|_| count >= GEMM_ROWS);
        match gemm.and_then(|g| g.index_wq_b.as_ref()) {
            Some(wq) => {
                let x = Input::rows(q_low, s.q_rank);
                sc.gemm.project(gpu, x, wq, sc.index_q, count, stream)?
            }
            None => parts::project_rows(
                gpu,
                k.parts.fp8_q,
                q_low,
                w.wq_b.w.data,
                w.wq_b.w.scales,
                sc.index_q,
                2,
                w.wq_b.n,
                count,
                w.wq_b.n,
                w.wq_b.k,
                stream,
            )?,
        }
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
        self.project_f32(
            gpu,
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
        Ok(Queries {
            count,
            position,
            ratio: self.ratio,
            stride,
        })
    }

    /// What the index keys score for each of `count` tokens of the batch,
    /// from its row `first` at `position`, read back: a row of `stride`
    /// scores a token, of which the token's visible groups are meaningful.
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
        let queries = self.index_queries(gpu, w, first, count, position, stream)?;
        let stride = queries.stride;
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

    /// The positions each of `rows` queries from `position` reads, chosen
    /// on the host as a row alone is, into `table`.
    #[allow(clippy::too_many_arguments)]
    fn choose_on_host(
        &self,
        gpu: &dyn GpuBackend,
        w: &IndexerWeights,
        index_keys: DevicePtr,
        candidates: &mut [Vec<bool>],
        table: DevicePtr,
        rows: usize,
        position: usize,
        stream: u64,
    ) -> Result<()> {
        let (topk, score_rows) = (self.shared.shape.index_topk, self.shared.scratch.score_rows);
        let visible = |r: usize| visible_groups(position + r, self.ratio);
        let mut words: Vec<u8> = Vec::with_capacity(rows * topk * 4);
        for first in (0..rows).step_by(score_rows) {
            let count = (rows - first).min(score_rows);
            let (mut scores, stride) =
                self.index_scores(gpu, w, index_keys, first, count, position + first, stream)?;
            for i in 0..count {
                let of_row = &mut scores[i * stride..i * stride + visible(first + i)];
                let chosen = self.choose(of_row, &mut candidates[first + i])?;
                words.extend(chosen.iter().flat_map(|p| p.to_le_bytes()));
                words.extend((chosen.len()..topk).flat_map(|_| (-1i32).to_le_bytes()));
            }
        }
        gpu.copy_h2d_async(&words, table, stream)
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
    ) -> Result<(DevicePtr, Chosen)> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        let topk = s.index_topk;
        let visible = |r: usize| visible_groups(position + r, self.ratio);
        let mut handoff = self.shared.handoff.lock().expect("handoff");
        if handoff.rows.len() < row + rows {
            handoff.rows.resize_with(row + rows, Default::default);
        }
        // A batch's choices stay on the device, a row of a table each; and
        // so do a row's alone where batches take the GEMMs, which is every
        // decode step of a served model. A row alone otherwise, as the tests
        // of one layer run it, is chosen for on the host, where they can
        // hand it a choice.
        let on_device = rows > 1 || self.gemm.is_some();
        let table = match on_device {
            false => DevicePtr::NULL,
            true => handoff
                .table(gpu, row + rows, topk, stream)?
                .offset(row * topk * 4),
        };
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
        let Some(w) = &self.weights.indexer else {
            // The latest index source chose for these rows.
            return Ok(match on_device {
                false => {
                    ensure!(
                        batch[0].chosen.len() == visible(0).min(topk),
                        "layer {}: row {row} was handed {} compressed positions, and its query sees {}",
                        self.index,
                        batch[0].chosen.len(),
                        visible(0)
                    );
                    (keys, Chosen::Host(batch[0].chosen.clone()))
                }
                true => (keys, Chosen::Table(table)),
            });
        };
        ensure!(
            !index_keys.is_null(),
            "layer {}: no index keys for row {row}",
            self.index
        );
        // Below `index_topk` positions there is nothing to choose: every
        // position is read, and all are candidates (`Shape::from_config`
        // checks that they would be). The rows that see more come last.
        let scored = (0..rows).find(|&r| visible(r) > topk).unwrap_or(rows);
        if !on_device {
            let mine = &mut batch[0];
            mine.chosen = match scored {
                0 => {
                    let (mut scores, _) =
                        self.index_scores(gpu, w, index_keys, 0, 1, position, stream)?;
                    self.choose(&mut scores, &mut mine.candidates)?
                }
                _ => {
                    if self.candidates == Candidates::Source {
                        mine.candidates = vec![true; visible(0)];
                    }
                    (0..visible(0) as i32).collect()
                }
            };
            return Ok((keys, Chosen::Host(mine.chosen.clone())));
        }
        // While every position is a candidate, the device chooses. Past
        // that the two-level choice is the host's.
        let all_candidates = s.candidate_blocks * s.candidate_block_size;
        if self.candidates != Candidates::None && visible(rows - 1) > all_candidates {
            let mut candidates: Vec<Vec<bool>> = batch
                .iter_mut()
                .enumerate()
                .map(|(r, mine)| match (self.candidates, r < scored) {
                    (Candidates::Source, true) => vec![true; visible(r)],
                    _ => std::mem::take(&mut mine.candidates),
                })
                .collect();
            self.choose_on_host(
                gpu,
                w,
                index_keys,
                &mut candidates,
                table,
                rows,
                position,
                stream,
            )?;
            for (mine, flags) in batch.iter_mut().zip(candidates) {
                mine.candidates = flags;
            }
            return Ok((keys, Chosen::Table(table)));
        }
        if self.candidates == Candidates::Source {
            // What a row alone after this batch would be handed.
            for (r, mine) in batch.iter_mut().enumerate() {
                mine.candidates = vec![true; visible(r)];
            }
        }
        if scored > 0 {
            let all = Queries {
                count: scored,
                position,
                ratio: self.ratio,
                stride: 0,
            };
            index::choose_rows(
                gpu,
                k.index.choose,
                DevicePtr::NULL,
                table,
                topk,
                all,
                stream,
            )?;
        }
        for first in (scored..rows).step_by(sc.score_rows) {
            let count = (rows - first).min(sc.score_rows);
            let queries = self.index_queries(gpu, w, first, count, position + first, stream)?;
            index::score_rows_tc(
                gpu,
                k.index.score_tc,
                sc.index_q,
                index_keys,
                sc.index_weights,
                sc.scores_f32,
                s.index_heads,
                queries,
                head_weight_scale(s.index_heads),
                stream,
            )?;
            index::choose_rows(
                gpu,
                k.index.choose,
                sc.scores_f32,
                table.offset(first * topk * 4),
                topk,
                queries,
                stream,
            )?;
        }
        Ok((keys, Chosen::Table(table)))
    }
}

#[cfg(test)]
#[path = "block_index_test_gpu.rs"]
mod test_gpu;
