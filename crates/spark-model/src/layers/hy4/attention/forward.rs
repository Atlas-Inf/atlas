// SPDX-License-Identifier: AGPL-3.0-only

use super::cache::AttentionCache;
use crate::layers::hy4::timing;
use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl AttentionCache<'_, '_> {
    /// BF16 `[positions.len(),H]` buffers, distinct and non-overlapping.
    /// Input is already the decoder input RMSNorm output, after iHC collapse.
    /// This accepted-token path is eager; it does not advertise graph capture.
    pub fn forward(
        &mut self,
        input: DevicePtr,
        output: DevicePtr,
        positions: &[[i64; 3]],
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "Hy4 cache poisoned by a prior dispatch error"
        );
        ensure!(
            self.len
                .checked_add(positions.len())
                .is_some_and(|n| n <= self.max_tokens),
            "Hy4 cache capacity exceeded"
        );
        ensure!(
            positions
                .iter()
                .flatten()
                .all(|p| *p >= 0 && (*p as usize) < self.layer.max_context),
            "Hy4 rotary position out of bounds"
        );
        if positions.is_empty() {
            return Ok(());
        }
        ensure!(
            !input.is_null() && !output.is_null() && input != output,
            "Hy4 attention null/aliased input"
        );
        ensure!(
            self.stream.is_none_or(|s| s == stream),
            "Hy4 request cache used on another stream"
        );
        let comm = if self.layer.world > 1 {
            let c = comm.context("Hy4 TP2 attention requires communicator")?;
            ensure!(
                c.world_size() == self.layer.world && c.rank() == self.layer.rank,
                "Hy4 TP attention communicator mismatch"
            );
            Some(c)
        } else {
            None
        };
        self.stream = Some(stream);
        for (row, &position) in positions.iter().enumerate() {
            let offset = row * self.layer.hidden * 2;
            if let Err(e) = self.forward_one(
                input.offset(offset),
                output.offset(offset),
                position,
                comm,
                stream,
            ) {
                self.poisoned = true;
                return Err(e);
            }
            self.len += 1;
        }
        Ok(())
    }
    fn forward_one(
        &mut self,
        x: DevicePtr,
        out: DevicePtr,
        position: [i64; 3],
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let s = &self.scratch;
        let gpu = a.memory.gpu;
        a.linear(x, a.qa, s.qc, a.qrank, a.hidden, stream)?;
        a.norm(s.qc, a.qan, s.qc, a.qrank, stream)?;
        a.linear(s.qc, a.qb, s.q, a.heads * 256, a.qrank, stream)?;
        a.linear(x, a.kva, s.kv, 576, a.hidden, stream)?;
        a.norm(s.kv, a.kvn, s.kv_norm, 512, stream)?;
        a.gemv(s.q, a.wk, s.absorbed, [512, 192, 256, 576], stream)?;
        a.rotary(
            s.q,
            s.absorbed,
            a.heads,
            [256, 576, 192, 512],
            position,
            false,
            stream,
        )?;
        let row = self.latent.offset(self.layout.slot(self.len)? * 576 * 2);
        KernelLaunch::new(gpu, a.kernels.cache)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.kv_norm)
            .arg_ptr(row)
            .launch(stream)?;
        a.rotary(s.kv, row, 1, [576, 576, 512, 512], position, false, stream)?;
        if a.index.is_some() {
            let _index = timing::span("dsa_index", stream)?;
            self.run_index(x, position, stream)?;
        }
        let s = &self.scratch;
        let (indices, count) = match a.window {
            Some(window) => (DevicePtr::NULL, (self.len + 1).min(window)),
            None => (s.indices, 2048),
        };
        KernelLaunch::new(gpu, a.kernels.attention)
            .grid([a.heads as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.absorbed)
            .arg_ptr(self.latent)
            .arg_ptr(self.table)
            .arg_ptr(indices)
            .arg_ptr(a.sink)
            .arg_ptr(s.latent_out)
            .arg_u32(self.len as u32)
            .arg_u32(self.layout.page_size as u32)
            .arg_u32(self.layout.table.len() as u32)
            .arg_u32(a.window.unwrap_or(0) as u32)
            .arg_u32(count as u32)
            .launch(stream)?;
        a.gemv(s.latent_out, a.wv, s.values, [256, 512, 512, 256], stream)?;
        a.linear(x, a.gate, s.gates, a.heads * 256, a.hidden, stream)?;
        KernelLaunch::new(gpu, a.kernels.gate)
            .grid([(a.heads * 256).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.values)
            .arg_ptr(s.gates)
            .arg_u32((a.heads * 256) as u32)
            .launch(stream)?;
        a.linear(
            s.values,
            a.output,
            s.partial,
            a.hidden,
            a.heads * 256,
            stream,
        )?;
        if let Some(comm) = comm {
            let _reduce = timing::span("all_reduce", stream)?;
            comm.all_reduce_async(s.partial.0, a.hidden * 2, stream)?;
        }
        a.norm(s.partial, a.norm, out, a.hidden, stream)
    }
}
