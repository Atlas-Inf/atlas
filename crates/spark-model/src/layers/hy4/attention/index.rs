// SPDX-License-Identifier: AGPL-3.0-only

use super::cache::AttentionCache;
use anyhow::{Context, Result};
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl AttentionCache<'_, '_> {
    pub(super) fn run_index(
        &mut self,
        x: DevicePtr,
        position: [i64; 3],
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let s = &self.scratch;
        let gpu = a.memory.gpu;
        let ix = a.index.as_ref().context("Hy4 SWA has no IndexPool")?;
        if self.len.is_multiple_of(4) {
            self.first_group_position = Some(position);
        }
        a.linear(x, ix.key, s.key, 128, a.hidden, stream)?;
        a.norm(s.key, ix.key_norm, s.key, 128, stream)?;
        KernelLaunch::new(gpu, a.kernels.tail)
            .grid([1, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(s.key)
            .arg_ptr(self.tail)
            .arg_u32((self.len % 4) as u32)
            .launch(stream)?;
        if self.len % 4 == 3 {
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
            KernelLaunch::new(gpu, a.kernels.fwht)
                .grid([1, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(s.pooled)
                .arg_ptr(self.pool.offset(page * 8448 + row * 128))
                .arg_ptr(self.pool.offset(page * 8448 + 8192 + row * 4))
                .launch(stream)?;
        }
        a.linear(s.qc, ix.query, s.query, ix.heads * 128, a.qrank, stream)?;
        a.rotary(
            s.query,
            s.query,
            ix.heads,
            [128, 128, 64, 64],
            position,
            true,
            stream,
        )?;
        KernelLaunch::new(gpu, a.kernels.fwht)
            .grid([ix.heads as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(s.query)
            .arg_ptr(s.quant)
            .arg_ptr(s.scales)
            .launch(stream)?;
        a.linear(x, ix.head, s.hw, ix.heads, a.hidden, stream)?;
        let groups = self.len / 4;
        if groups > 0 {
            KernelLaunch::new(gpu, a.kernels.score)
                .grid([groups.div_ceil(128) as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(s.quant)
                .arg_ptr(s.scales)
                .arg_ptr(s.hw)
                .arg_ptr(self.pool)
                .arg_ptr(self.pool_table)
                .arg_ptr(s.scores)
                .arg_u32(groups as u32)
                .arg_u32(ix.heads as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, a.kernels.select)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(s.scores)
            .arg_ptr(s.indices)
            .arg_u32(groups as u32)
            .arg_u32(self.len as u32)
            .launch(stream)
    }
}
