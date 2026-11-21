// SPDX-License-Identifier: AGPL-3.0-only

use super::cache::AttentionCache;
use crate::layers::hy4::timing;
use anyhow::{Context, Result, ensure};
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl AttentionCache<'_, '_> {
    pub(super) fn run_index(
        &mut self,
        x: DevicePtr,
        position: [i64; 3],
        stream: u64,
    ) -> Result<()> {
        self.run_index_projected(x, position, None, stream)
    }

    pub(super) fn run_index_projected(
        &mut self,
        x: DevicePtr,
        position: [i64; 3],
        projected: Option<[DevicePtr; 3]>,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let s = &self.scratch;
        let gpu = a.memory.gpu;
        let ix = a.index.as_ref().context("Hy4 SWA has no IndexPool")?;
        if self.len.is_multiple_of(4) {
            self.first_group_position = Some(position);
        }
        let phase = timing::span("dsa_key_projection", stream)?;
        let [key, query, hw] = projected.unwrap_or([s.key, s.query, s.hw]);
        if projected.is_none() {
            a.linear(x, ix.key, key, 128, a.hidden, stream)?;
        }
        drop(phase);
        a.norm(
            key,
            ix.key_norm,
            self.tail.offset((self.len % 4) * 128 * 2),
            128,
            stream,
        )?;
        if self.len % 4 == 3 {
            let phase = timing::span("dsa_pool", stream)?;
            KernelLaunch::new(gpu, a.kernels.pool)
                .grid([1, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(self.tail)
                .arg_ptr(ix.value)
                .arg_ptr(ix.gates)
                .arg_ptr(ix.ape)
                .arg_ptr(ix.norm)
                .arg_ptr(s.pooled)
                .arg_f32(a.eps)
                .launch(stream)?;
            drop(phase);
            a.rotary(
                s.pooled,
                s.pooled,
                1,
                [128, 128, 64, 64],
                self.first_group_position
                    .context("Hy4 missing partial-group position")?,
                true,
                stream,
            )?;
            let group = self.len / 4;
            let page = group / 64;
            let row = group % 64;
            let phase = timing::span("dsa_pool_fwht", stream)?;
            KernelLaunch::new(gpu, a.kernels.fwht)
                .grid([1, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(s.pooled)
                .arg_ptr(self.pool.offset(page * 8448 + row * 128))
                .arg_ptr(self.pool.offset(page * 8448 + 8192 + row * 4))
                .launch(stream)?;
            drop(phase);
        }
        let phase = timing::span("dsa_query_projection", stream)?;
        if projected.is_none() {
            a.linear(s.qc, ix.query, query, ix.heads * 128, a.qrank, stream)?;
        }
        drop(phase);
        a.rotary(
            query,
            query,
            ix.heads,
            [128, 128, 64, 64],
            position,
            true,
            stream,
        )?;
        let phase = timing::span("dsa_query_fwht", stream)?;
        KernelLaunch::new(gpu, a.kernels.fwht)
            .grid([ix.heads as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(query)
            .arg_ptr(s.quant)
            .arg_ptr(s.scales)
            .launch(stream)?;
        drop(phase);
        let phase = timing::span("dsa_head_projection", stream)?;
        if projected.is_none() {
            a.linear(x, ix.head, hw, ix.heads, a.hidden, stream)?;
        }
        drop(phase);
        let groups = self.len / 4;
        if groups > 0 {
            let _phase = timing::span("dsa_score", stream)?;
            KernelLaunch::new(gpu, a.kernels.score)
                .grid([groups as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.quant)
                .arg_ptr(s.scales)
                .arg_ptr(hw)
                .arg_ptr(self.pool)
                .arg_ptr(self.pool_table)
                .arg_ptr(s.scores)
                .arg_u32(groups as u32)
                .arg_u32(ix.heads as u32)
                .launch(stream)?;
        }
        let _phase = timing::span("dsa_select", stream)?;
        KernelLaunch::new(gpu, a.kernels.select)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.scores)
            .arg_ptr(s.indices)
            .arg_u32(groups as u32)
            .arg_u32(self.len as u32)
            .launch(stream)
    }

    /// Batched IndexPool step for one causal tile (prefill/verify batch path).
    /// Per row this performs the same state updates as `run_index_projected`
    /// (key norm into the tail, pool group write every fourth row, query
    /// rotary), but the FWHT, score and select kernels are launched once for
    /// the whole tile: `hy4_fwht_batch`, `hy4_index_scores_batch` and
    /// `hy4_index_select_batch` are grid-parallel across rows and the select
    /// is O(groups log groups) instead of O(groups^2). Selection output is
    /// identical to the serial path, so attention accumulation order and the
    /// recorded prefixes are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_index_tile(
        &mut self,
        positions: &[[i64; 3]],
        key: DevicePtr,
        query: DevicePtr,
        hw: DevicePtr,
        quant: DevicePtr,
        scales: DevicePtr,
        scores: DevicePtr,
        indices: DevicePtr,
        max_groups: usize,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let gpu = a.memory.gpu;
        let ix = a.index.as_ref().context("Hy4 SWA has no IndexPool")?;
        let rows = positions.len();
        ensure!(
            rows > 0 && rows <= crate::layers::hy4::batch::MAX_ROWS,
            "Hy4 index tile rows"
        );
        let len_base = self.len;
        for (r, &position) in positions.iter().enumerate() {
            let len = len_base + r;
            if len.is_multiple_of(4) {
                self.first_group_position = Some(position);
            }
            a.norm(
                key.offset(r * 128 * 2),
                ix.key_norm,
                self.tail.offset((len % 4) * 128 * 2),
                128,
                stream,
            )?;
            if len % 4 == 3 {
                let phase = timing::span("dsa_pool", stream)?;
                KernelLaunch::new(gpu, a.kernels.pool)
                    .grid([1, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(self.tail)
                    .arg_ptr(ix.value)
                    .arg_ptr(ix.gates)
                    .arg_ptr(ix.ape)
                    .arg_ptr(ix.norm)
                    .arg_ptr(self.scratch.pooled)
                    .arg_f32(a.eps)
                    .launch(stream)?;
                drop(phase);
                a.rotary(
                    self.scratch.pooled,
                    self.scratch.pooled,
                    1,
                    [128, 128, 64, 64],
                    self.first_group_position
                        .context("Hy4 missing partial-group position")?,
                    true,
                    stream,
                )?;
                let group = len / 4;
                let page = group / 64;
                let rowg = group % 64;
                let phase = timing::span("dsa_pool_fwht", stream)?;
                KernelLaunch::new(gpu, a.kernels.fwht)
                    .grid([1, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(self.scratch.pooled)
                    .arg_ptr(self.pool.offset(page * 8448 + rowg * 128))
                    .arg_ptr(self.pool.offset(page * 8448 + 8192 + rowg * 4))
                    .launch(stream)?;
                drop(phase);
            }
            let q = query.offset(r * ix.heads * 128 * 2);
            a.rotary(q, q, ix.heads, [128, 128, 64, 64], position, true, stream)?;
        }
        {
            let _phase = timing::span("dsa_query_fwht", stream)?;
            KernelLaunch::new(gpu, a.kernels.fwht_batch)
                .grid([ix.heads as u32, rows as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(query)
                .arg_ptr(quant)
                .arg_ptr(scales)
                .arg_u32(ix.heads as u32)
                .arg_u32(rows as u32)
                .launch(stream)?;
        }
        let groups_max = (len_base + rows) / 4 + 1;
        if groups_max > 1 {
            let _phase = timing::span("dsa_score", stream)?;
            KernelLaunch::new(gpu, a.kernels.score_batch)
                .grid([groups_max as u32, rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(quant)
                .arg_ptr(scales)
                .arg_ptr(hw)
                .arg_ptr(self.pool)
                .arg_ptr(self.pool_table)
                .arg_ptr(scores)
                .arg_u32(len_base as u32)
                .arg_u32(ix.heads as u32)
                .arg_u32(max_groups as u32)
                .arg_u32(rows as u32)
                .launch(stream)?;
        }
        {
            let _phase = timing::span("dsa_select", stream)?;
            KernelLaunch::new(gpu, a.kernels.select_batch)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(scores)
                .arg_ptr(indices)
                .arg_u32(len_base as u32)
                .arg_u32(len_base as u32)
                .arg_u32(max_groups as u32)
                .arg_u32(rows as u32)
                .launch(stream)?;
        }
        for r in 0..rows {
            if a.window.is_none() {
                gpu.copy_d2d_async(
                    indices.offset(r * 2048 * 4),
                    self.scratch.indices,
                    2048 * 4,
                    stream,
                )?;
            }
            self.len += 1;
            self.record_prefix(stream)?;
        }
        Ok(())
    }
}
