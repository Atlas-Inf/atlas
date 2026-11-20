// SPDX-License-Identifier: AGPL-3.0-only

//! The compressor of a DeepSeek-V4.1 key source over a batch of rows: the
//! keys, and the index keys, of the groups the batch completes.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::cache_recipe::group_position;
use super::index::INDEX_DIM;
use super::moe;
use super::ops::{self, Fp4Scale, HEAD_DIM};
use super::shape::MAX_RATIO;
use super::side;
use super::state::{KEPT_ROWS, SeqState};
use super::weights::IndexKeyWeights;

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
        let projections = [(Some(w.wkv), held.kv_state, sc.pool_kv, held.kept_kv)]
            .into_iter()
            .chain([(w.wgate, held.score_state, sc.pool_score, held.kept_score)]);
        for (weights, kept, pool, saved) in projections {
            let Some(weights) = weights else { continue };
            if carried > 0 {
                gpu.copy_d2d_async(kept, pool, carried * token, stream)?;
            }
            self.project_f32(
                gpu,
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
            // A small batch may be a verify whose rows are taken back.
            if rows <= KEPT_ROWS {
                gpu.copy_d2d_async(pool, saved, (carried + rows) * token, stream)?;
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
        self.project_f32(
            gpu,
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
}
