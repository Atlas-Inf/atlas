// SPDX-License-Identifier: AGPL-3.0-only

//! The attention sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::cache_recipe::{WINDOW, completed_group, group_position, window_row, window_slot};
use super::index::INDEX_DIM;
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM, MAX_KEYS};
use super::parts::{self, PARTS};
use super::profile;
use super::side;
use super::state::SeqState;
use super::weights::row_part;
use crate::layers::ops::mxfp8_gemv_rows;
use crate::weight_map::Mxfp8Weight;

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
            sc.normed,
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
                sc.normed,
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

    /// Attention of the input in `scratch.normed`, and in FP8 in
    /// `scratch.quantized`, into the parts of [`Block::attention_parts`].
    /// The heads, and the output projections behind them, are in four parts;
    /// `comm` reaches the ranks that hold the others.
    pub(super) fn attention(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
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
        // The low-rank query, normed, and its FP8 form, which every head's
        // projection and the indexer's read.
        mxfp8_gemv_rows(
            gpu,
            &k.tiers,
            &w.wq_a.w,
            sc.quantized,
            sc.q_low,
            1,
            w.wq_a.n,
            w.wq_a.k,
            w.wq_a.n,
            stream,
        )?;
        self.norm(gpu, sc.q_low, w.q_norm, sc.q_low_normed, s.q_rank, stream)?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.q_low_normed,
            sc.q_low_quantized,
            s.q_rank,
            stream,
        )?;
        // The window key, from the same FP8 input, into the ring.
        mxfp8_gemv_rows(
            gpu,
            &k.tiers,
            &w.wkv.w,
            sc.quantized,
            sc.key,
            1,
            w.wkv.n,
            w.wkv.k,
            w.wkv.n,
            stream,
        )?;
        let slot = state.window.offset(window_slot(position) * HEAD_DIM * 2);
        self.norm(gpu, sc.key, w.kv_norm, sc.pooled, HEAD_DIM, stream)?;
        let at = self.at(position)?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            sc.pooled,
            at,
            self.freqs,
            1,
            1,
            HEAD_DIM,
            s.rope,
            false,
            stream,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.pooled,
            slot,
            HEAD_DIM,
            stream,
        )?;
        profile::end(gpu, stream, "query and window key", began)?;
        let began = profile::begin(gpu, stream)?;
        // Compressed keys: a source writes its own and chooses which a query
        // reads; the layers after it read the same keys at the same choice.
        let mut idx = window_row(position);
        let mut compressed = state.window;
        if self.ratio > 0 {
            if self.weights.compressor.is_some() {
                self.compress(gpu, state, position, stream)?;
            }
            let (keys, chosen) = self.compressed_row(gpu, row, state, position, at, stream)?;
            compressed = keys;
            idx.extend(chosen);
        }
        profile::end(gpu, stream, "compress and choose", began)?;
        ensure!(
            idx.len() <= MAX_KEYS,
            "layer {}: {} keys for one query",
            self.index,
            idx.len()
        );
        // In stream order, behind the kernels of the token before.
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
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
            // This part's heads: project, rotate, attend, and undo the
            // rotation.
            let q = sc.q.offset(part * heads * HEAD_DIM * 2);
            let attended = sc.attended.offset(part * heads * HEAD_DIM * 2);
            let wq = row_part(&w.wq_b, part);
            parts::project(
                gpu,
                k.parts.fp8,
                sc.q_low_quantized,
                wq.data,
                wq.scales,
                q,
                heads * HEAD_DIM,
                w.wq_b.k,
                stream,
            )?;
            ops::rotate(
                gpu,
                k.cache.rotate,
                q,
                at,
                self.freqs,
                1,
                heads,
                HEAD_DIM,
                s.rope,
                false,
                stream,
            )?;
            ops::sparse_attn(
                gpu,
                k.cache.sparse_attn,
                q,
                state.window,
                compressed,
                sc.idx,
                w.sink.offset(part * heads * 4),
                attended,
                1,
                heads,
                idx.len(),
                WINDOW,
                stream,
            )?;
            ops::rotate(
                gpu,
                k.cache.rotate,
                attended,
                at,
                self.freqs,
                1,
                heads,
                HEAD_DIM,
                s.rope,
                true,
                stream,
            )?;
            profile::end(gpu, stream, "sparse attention", began)?;
            let began = profile::begin(gpu, stream)?;
            // Quantize, and project out: each group of heads through its own
            // block of wo_a, then this part's columns of wo_b, unrounded.
            ops::fp8_quant_dequant(
                gpu,
                k.cache.fp8_quant_dequant,
                attended,
                attended,
                heads * HEAD_DIM,
                stream,
            )?;
            for group in part * groups..(part + 1) * groups {
                let block = Mxfp8Weight {
                    data: w.wo_a.w.data.offset(group * s.o_rank * group_in),
                    scales: w.wo_a.w.scales.offset(group * s.o_rank * group_in / 32),
                    tiled: false,
                };
                mxfp8_gemv_rows(
                    gpu,
                    &k.tiers,
                    &block,
                    sc.attended.offset(group * group_in * 2),
                    sc.o_low.offset(group * s.o_rank * 2),
                    1,
                    s.o_rank,
                    group_in,
                    s.o_rank,
                    stream,
                )?;
            }
            let (low, width) = (part * groups * s.o_rank, groups * s.o_rank);
            ops::fp8_quant_dequant(
                gpu,
                k.cache.fp8_quant_dequant,
                sc.o_low.offset(low * 2),
                sc.o_low_quantized.offset(low * 2),
                width,
                stream,
            )?;
            let wo = w.wo_b[part].context("a held part of wo_b")?;
            parts::project(
                gpu,
                k.parts.fp8_f32,
                sc.o_low_quantized.offset(low * 2),
                wo.data,
                wo.scales,
                sc.attn_parts.offset(part * s.hidden * 4),
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
                s.hidden * 4,
                stream,
            )
        })
    }
}
