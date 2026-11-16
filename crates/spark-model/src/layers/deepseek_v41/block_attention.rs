// SPDX-License-Identifier: AGPL-3.0-only

//! The attention sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;

use super::block::Block;
use super::cache_recipe::{WINDOW, completed_group, group_position, window_row, window_slot};
use super::index::INDEX_DIM;
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM};
use super::profile;
use super::side;
use super::state::SeqState;
use crate::layers::ops::mxfp8_gemv_rows;
use crate::weight_map::Mxfp8Weight;

impl Block {
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
        let at = self.at(
            gpu,
            sc.group_position,
            group_position(group, w.ratio),
            stream,
        )?;
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

    /// Attention of the normed input in `scratch.normed` into
    /// `scratch.sublayer_out`.
    pub(super) fn attention(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &mut SeqState,
        position: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.attn,
        );
        let began = profile::begin(gpu, stream)?;
        // q: low-rank projection, norm, per-head projection, rotation.
        self.fp8(gpu, &w.wq_a, sc.normed, sc.quantized, sc.q_low, stream)?;
        self.norm(gpu, sc.q_low, w.q_norm, sc.q_low_normed, s.q_rank, stream)?;
        self.fp8(
            gpu,
            &w.wq_b,
            sc.q_low_normed,
            sc.q_low_quantized,
            sc.q,
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
        let at = self.at(gpu, sc.position, position, stream)?;
        ops::rotate(
            gpu,
            k.cache.rotate,
            sc.q,
            at,
            self.freqs,
            1,
            s.heads,
            HEAD_DIM,
            s.rope,
            false,
            stream,
        )?;
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
        let began = profile::begin(gpu, stream)?;
        gpu.synchronize(stream)?;
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, sc.idx)?;
        ops::sparse_attn(
            gpu,
            k.cache.sparse_attn,
            sc.q,
            state.window,
            compressed,
            sc.idx,
            w.sink,
            sc.attended,
            1,
            s.heads,
            idx.len(),
            WINDOW,
            stream,
        )?;
        profile::end(gpu, stream, "sparse attention", began)?;
        let began = profile::begin(gpu, stream)?;

        // Undo the query's rotation, quantize, and project out: each group of
        // heads through its own block of wo_a, then wo_b.
        ops::rotate(
            gpu,
            k.cache.rotate,
            sc.attended,
            at,
            self.freqs,
            1,
            s.heads,
            HEAD_DIM,
            s.rope,
            true,
            stream,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.attended,
            sc.attended,
            s.heads * HEAD_DIM,
            stream,
        )?;
        let group_in = s.heads / s.o_groups * HEAD_DIM;
        ensure!(
            w.wo_a.k == group_in && w.wo_a.n == s.o_groups * s.o_rank,
            "wo_a [{}, {}]",
            w.wo_a.n,
            w.wo_a.k
        );
        for group in 0..s.o_groups {
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
        self.fp8(
            gpu,
            &w.wo_b,
            sc.o_low,
            sc.o_low_quantized,
            sc.sublayer_out,
            stream,
        )?;
        profile::end(gpu, stream, "output projections", began)
    }
}
