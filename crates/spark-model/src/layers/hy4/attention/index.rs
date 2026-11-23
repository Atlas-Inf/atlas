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
    /// Every stage is launched once for the whole tile instead of once per row
    /// or per pool group: `hy4_tail_norm_batch` (key norm into the rolling
    /// tail plus a per-tile snapshot), `hy4_pool_fwht_batch` (pool + rotary +
    /// FWHT fused per group), `hy4_batch_rotary` (query rotary, grid-parallel
    /// over rows), then `hy4_fwht_batch`, `hy4_index_scores_batch` and
    /// `hy4_index_select_batch` as before. The math per row is identical to
    /// the serial launches, so selection output, accumulation order and the
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
        let base = positions[0];
        // Per-tile snapshot of the normalized keys (three-row headroom for
        // groups that start before this tile), consumed by hy4_pool_fwht_batch
        // in the same stream-ordered region it is written in.
        let tail_tile = gpu.op_cache().scratch(
            gpu,
            "hy4_index_tile_tail",
            (crate::layers::hy4::batch::MAX_ROWS + 3) * 128 * 2,
        )?;
        // Preload the rolling tail's three pre-tile members into the snapshot
        // headroom so fused pool groups that start before this tile read the
        // same values the serial per-row path left in the tail slots.
        if !len_base.is_multiple_of(4) {
            for k in 1..=3usize {
                let slot = (len_base as i64 - k as i64).rem_euclid(4) as usize;
                gpu.copy_d2d_async(
                    self.tail.offset(slot * 128 * 2),
                    tail_tile.offset((3 - k) * 128 * 2),
                    128 * 2,
                    stream,
                )?;
            }
        }
        {
            let _norm = timing::span("attn_norm", stream)?;
            KernelLaunch::new(gpu, a.kernels.tail_norm_batch)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(key)
                .arg_ptr(ix.key_norm)
                .arg_ptr(self.tail)
                .arg_ptr(tail_tile)
                .arg_u32(len_base as u32)
                .arg_u32(128)
                .arg_f32(a.eps)
                .launch(stream)?;
        }
        let r0 = (7 - len_base % 4) % 4;
        let groups = if rows > r0 {
            (rows - r0).div_ceil(4)
        } else {
            0
        };
        if groups > 0 {
            let _pool = timing::span("dsa_pool", stream)?;
            KernelLaunch::new(gpu, a.kernels.pool_fwht_batch)
                .grid([groups as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(tail_tile)
                .arg_ptr(ix.value)
                .arg_ptr(ix.gates)
                .arg_ptr(ix.ape)
                .arg_ptr(ix.norm)
                .arg_ptr(self.pool)
                .arg_u32(len_base as u32)
                .arg_u32(r0 as u32)
                .arg_u32(rows as u32)
                .arg_u64(base[0] as u64)
                .arg_f32(a.rope.rope_theta as f32)
                .arg_f32(a.eps)
                .launch(stream)?;
        }
        {
            let _rope = timing::span("attn_rope", stream)?;
            KernelLaunch::new(
                gpu,
                gpu.op_cache().kernel(gpu, "hy4_mla", "hy4_batch_rotary")?,
            )
            .grid([ix.heads as u32, rows as u32, 1])
            .block([32, 1, 1])
            .arg_ptr(query)
            .arg_ptr(query)
            .arg_u32(128)
            .arg_u32(128)
            .arg_u32(64)
            .arg_u32(64)
            .arg_u32(ix.heads as u32 * 128)
            .arg_u32(ix.heads as u32 * 128)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(0)
            .arg_u32(1)
            .arg_u32(1)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u64(base[0] as u64)
            .arg_u64(base[1] as u64)
            .arg_u64(base[2] as u64)
            .arg_f32(a.rope.rope_theta as f32)
            .arg_u32(a.rope.mrope_section[1] as u32)
            .arg_u32(a.rope.mrope_section[2] as u32)
            .arg_u32(u32::from(a.rope.mrope_interleaved))
            .arg_u32(1)
            .launch(stream)?;
        }
        for (r, &position) in positions.iter().enumerate() {
            if (len_base + r).is_multiple_of(4) {
                self.first_group_position = Some(position);
            }
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
        // Per-row state copies are only needed while prefix capture is active
        // (speculative VERIFY); otherwise the cache scratch keeps the last
        // row's selection, which is all the DFly target reads.
        let capturing = self.prefix_capturing();
        for r in 0..rows {
            if capturing && a.window.is_none() {
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
        if !capturing && a.window.is_none() {
            gpu.copy_d2d_async(
                indices.offset((rows - 1) * 2048 * 4),
                self.scratch.indices,
                2048 * 4,
                stream,
            )?;
        }
        Ok(())
    }
}
