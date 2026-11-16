// SPDX-License-Identifier: AGPL-3.0-only

//! The attention sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::cache_recipe::{
    WINDOW, completed_group, group_position, prefill_window, window_row, window_slot,
};
use super::index::INDEX_DIM;
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM, MAX_KEYS, Turn};
use super::parts::{self, PARTS};
use super::profile;
use super::side;
use super::state::SeqState;

impl Block {
    /// The device word holding `position`, for a rotation to read.
    fn at(&self, position: usize) -> Result<DevicePtr> {
        let sc = &self.shared.scratch;
        ensure!(
            position < sc.max_groups,
            "layer {}: position {position} of at most {}",
            self.index,
            sc.max_groups
        );
        Ok(sc.positions.offset(position * 4))
    }

    /// The compressor at `position`: project, hold in the group's slot, and
    /// when the group completes write its key into `keys` and, on a layer
    /// that makes them, its index key.
    fn compress(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut SeqState,
        position: usize,
        normed: DevicePtr,
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
        if let Some(group) = completed_group(position, w.ratio) {
            held.reserve(gpu, group + 1, stream)?;
        }
        let held = &*held;
        let slot = (position % w.ratio) * HEAD_DIM * 4;
        side::gemv_f32(
            gpu,
            k.side.gemv_f32,
            normed,
            w.wkv,
            held.kv_state.offset(slot),
            HEAD_DIM,
            s.hidden,
            stream,
        )?;
        if let Some(wgate) = w.wgate {
            side::gemv_f32(
                gpu,
                k.side.gemv_f32,
                normed,
                wgate,
                held.score_state.offset(slot),
                HEAD_DIM,
                s.hidden,
                stream,
            )?;
        }
        let Some(group) = completed_group(position, w.ratio) else {
            return Ok(());
        };
        let key = held.keys.offset(group * HEAD_DIM * 2);
        match w.wgate {
            Some(_) => side::compress_pool(
                gpu,
                k.side.compress_pool,
                held.kv_state,
                held.score_state,
                sc.pooled,
                w.ratio,
                HEAD_DIM,
                stream,
            )?,
            // One token a group: nothing to pool, the projection is the key.
            None => moe::round_bf16(
                gpu,
                k.moe.round_bf16,
                held.kv_state,
                sc.pooled,
                HEAD_DIM,
                stream,
            )?,
        }
        self.norm(gpu, sc.pooled, w.norm, key, HEAD_DIM, stream)?;
        let at = self.at(group_position(group, w.ratio))?;
        // The index key comes from the key as it is before its rotation.
        if let Some(index) = self.weights.indexer.as_ref().and_then(|i| i.key.as_ref()) {
            let index_key = held.index_keys.offset(group * INDEX_DIM * 2);
            self.index_key(gpu, index, key, index_key, at, stream)?;
        }
        ops::rotate(
            gpu,
            k.cache.rotate,
            key,
            at,
            self.freqs,
            1,
            1,
            HEAD_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp4_quant_dequant(
            gpu,
            k.cache.fp4_quant_dequant,
            key,
            HEAD_DIM,
            Fp4Scale::E4m3Per16,
            stream,
        )
    }

    /// The window's keys for a batch of more than one row from `position`,
    /// in `scratch.chunk_keys`: the ring unrolled into position order, with
    /// room after it for the batch's own, which `front` writes there and this
    /// copies on into the ring. A row then reads its 128 by position, and no
    /// row's key displaces one an earlier row of the batch still reads.
    fn unroll_window(
        &self,
        gpu: &dyn GpuBackend,
        state: &SeqState,
        position: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let sc = &self.shared.scratch;
        let key = HEAD_DIM * 2;
        // Entry i of the unrolled keys is position `position - 128 + i`.
        let turn = window_slot(position);
        let copy = |from: usize, to: usize, count: usize, back: bool| -> Result<()> {
            if count == 0 {
                return Ok(());
            }
            let (ring, run) = (
                state.window.offset(from * key),
                sc.chunk_keys.offset(to * key),
            );
            match back {
                false => gpu.copy_d2d_async(ring, run, count * key, stream),
                true => gpu.copy_d2d_async(run, ring, count * key, stream),
            }
        };
        copy(turn, 0, WINDOW - turn, false)?;
        copy(0, WINDOW - turn, turn, false)?;
        // The batch's keys, once `front` has written them, into the ring.
        let first = rows.min(WINDOW - turn);
        copy(turn, WINDOW, first, true)?;
        copy(0, WINDOW + first, rows - first, true)
    }

    /// Attention of the `rows` inputs in `scratch.normed`, and in FP8 in
    /// `scratch.quantized`, into the parts of [`Block::attention_parts`].
    /// The heads, and the output projections behind them, are in four parts;
    /// `comm` reaches the ranks that hold the others.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        rows: usize,
        state: &mut SeqState,
        position: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.attn,
        );
        let held = self.shared.held;
        let began = profile::begin(gpu, stream)?;
        // The low-rank query and the window key, both from the input's FP8
        // form; then the query normed, and the key normed, turned to the
        // token's position and rounded into the window.
        parts::project_two(
            gpu,
            k.parts.fp8_two,
            sc.quantized,
            (w.wq_a.rows_from(0), w.wq_a.n, sc.q_low),
            (w.wkv.rows_from(0), w.wkv.n, sc.key),
            s.hidden,
            rows,
            stream,
        )?;
        self.at(position + rows - 1)?;
        let turn = Turn {
            positions: self.at(position)?,
            freqs: self.freqs,
            rope: s.rope,
        };
        // One row writes its key into the ring and reads the ring. More read
        // the window unrolled, with their own keys after it.
        let (window, slot) = match rows {
            1 => (
                state.window,
                state.window.offset(window_slot(position) * HEAD_DIM * 2),
            ),
            _ => (sc.chunk_keys, sc.chunk_keys.offset(WINDOW * HEAD_DIM * 2)),
        };
        ops::front(
            gpu,
            k.cache.front,
            sc.q_low,
            w.q_norm,
            sc.q_low_normed,
            sc.key,
            w.kv_norm,
            slot,
            turn,
            s.q_rank,
            rows,
            s.norm_eps,
            stream,
        )?;
        if rows > 1 {
            self.unroll_window(gpu, state, position, rows, stream)?;
        }
        profile::end(gpu, stream, "query and window key", began)?;
        let began = profile::begin(gpu, stream)?;
        // Compressed keys: a source writes its own and chooses which a query
        // reads; the layers after it read the same keys at the same choice.
        let mut compressed = state.window;
        let mut index: Vec<Vec<i32>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let at = position + r;
            let mut idx = match rows {
                1 => window_row(at),
                // Position q is entry q - position + 128 of the unrolled keys.
                _ => {
                    let mut row: Vec<i32> = prefill_window(at)
                        .map(|q| (q + WINDOW - position) as i32)
                        .collect();
                    row.resize(WINDOW, -1);
                    row
                }
            };
            if self.ratio > 0 {
                let normed = sc.normed.offset(r * s.hidden * 2);
                if self.weights.compressor.is_some() {
                    self.compress(gpu, state, at, normed, stream)?;
                }
                let (keys, chosen) = self.compressed_row(
                    gpu,
                    row + r,
                    state,
                    at,
                    turn.positions.offset(r * 4),
                    normed,
                    sc.q_low_normed.offset(r * s.q_rank * 2),
                    stream,
                )?;
                compressed = keys;
                idx.extend(chosen);
            }
            index.push(idx);
        }
        profile::end(gpu, stream, "compress and choose", began)?;
        // Every row's keys in one table, the shorter rows ending in none.
        let keys = index.iter().map(Vec::len).max().unwrap_or(0);
        ensure!(
            keys <= MAX_KEYS,
            "layer {}: {keys} keys for one query",
            self.index
        );
        let mut bytes = Vec::with_capacity(rows * keys * 4);
        for idx in &index {
            bytes.extend(idx.iter().flat_map(|i| i.to_le_bytes()));
            bytes.extend((idx.len()..keys).flat_map(|_| (-1i32).to_le_bytes()));
        }
        // In stream order, behind the kernels of the batch before.
        gpu.copy_h2d_async(&bytes, sc.idx, stream)?;

        let (heads, groups) = (s.heads / PARTS, s.o_groups / PARTS);
        let group_in = s.heads / s.o_groups * HEAD_DIM;
        ensure!(
            w.wo_a.k == group_in
                && w.wo_a.n == s.o_groups * s.o_rank
                && w.wq_b.n == s.heads * HEAD_DIM,
            "wo_a [{}, {}], wq_b [{}, {}]",
            w.wo_a.n,
            w.wo_a.k,
            w.wq_b.n,
            w.wq_b.k
        );
        for part in held.range() {
            let began = profile::begin(gpu, stream)?;
            // This part's heads: projected from the query's FP8 form, and
            // attended, turned to the position and back inside the kernel.
            let q = sc.q.offset(part * sc.rows * heads * HEAD_DIM * 2);
            let attended = sc.attended.offset(part * sc.rows * heads * HEAD_DIM * 2);
            let wq = w.wq_b.rows_from(part * heads * HEAD_DIM);
            parts::project_rows(
                gpu,
                k.parts.fp8_q,
                sc.q_low_normed,
                wq.data,
                wq.scales,
                q,
                2,
                rows,
                heads * HEAD_DIM,
                w.wq_b.k,
                stream,
            )?;
            ops::sparse_attn(
                gpu,
                &k.cache,
                q,
                window,
                compressed,
                sc.idx,
                w.sink.offset(part * heads * 4),
                attended,
                sc.attn_records,
                rows,
                heads,
                keys,
                WINDOW,
                Some(turn),
                stream,
            )?;
            profile::end(gpu, stream, "sparse attention", began)?;
            let began = profile::begin(gpu, stream)?;
            // Project out, each projection rounding its input to FP8: this
            // part's groups of heads through their blocks of wo_a, then its
            // columns of wo_b, unrounded.
            let (low, width) = (part * groups * s.o_rank, groups * s.o_rank);
            let o_low = sc.o_low.offset(part * sc.rows * width * 2);
            parts::project_groups(
                gpu,
                k.parts.fp8_groups_q,
                attended,
                w.wo_a.rows_from(low),
                o_low,
                rows,
                groups,
                s.o_rank,
                group_in,
                stream,
            )?;
            let wo = w.wo_b[part].context("a held part of wo_b")?;
            parts::project_rows(
                gpu,
                k.parts.fp8_f32_q,
                o_low,
                wo.data,
                wo.scales,
                sc.attn_parts.offset(part * sc.rows * s.hidden * 4),
                4,
                rows,
                s.hidden,
                width,
                stream,
            )?;
            profile::end(gpu, stream, "output projections", began)?;
        }
        profile::stage(gpu, stream, "gather", || {
            parts::gather(
                gpu,
                comm,
                held,
                sc.attn_parts,
                sc.gather_recv,
                sc.rows * s.hidden * 4,
                rows * s.hidden * 4,
                stream,
            )
        })
    }
}
