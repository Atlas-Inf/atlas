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
        let phase = timing::span("attn_q_a", stream)?;
        let share = s.qkv.offset(a.rank * (a.qrank + 576) / a.world * 2);
        let qc = if a.world == 1 { s.qc } else { share };
        let kv = if a.world == 1 {
            s.kv
        } else {
            share.offset(a.qrank / a.world * 2)
        };
        a.linear(x, a.qa, qc, a.qrank / a.world, a.hidden, stream)?;
        drop(phase);
        let phase = timing::span("attn_kv_a", stream)?;
        a.linear(x, a.kva, kv, 576 / a.world, a.hidden, stream)?;
        drop(phase);
        if let Some(comm) = comm {
            let phase = timing::span("attn_qkv_gather", stream)?;
            crate::layers::glm_kv_shard::pair_exchange(
                comm,
                share,
                s.qkv.offset((1 - a.rank) * (a.qrank + 576)),
                (a.qrank + 576) * 2 / a.world,
                stream,
            )?;
            KernelLaunch::new(gpu, a.kernels.join)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.qkv)
                .arg_ptr(s.qc)
                .arg_ptr(s.kv)
                .arg_u32(a.qrank as u32)
                .launch(stream)?;
            drop(phase);
        }
        a.norm(s.qc, a.qan, s.qc, a.qrank, stream)?;
        let phase = timing::span("attn_q_b", stream)?;
        a.linear(s.qc, a.qb, s.q, a.heads * 256, a.qrank, stream)?;
        drop(phase);
        let row = self.latent.offset(self.layout.slot(self.len)? * 576 * 2);
        a.norm(s.kv, a.kvn, row, 512, stream)?;
        let phase = timing::span("attn_q_absorb", stream)?;
        a.gemv(s.q, a.wk, s.absorbed, [512, 192, 256, 576], stream)?;
        drop(phase);
        a.rotary(
            s.q,
            s.absorbed,
            a.heads,
            [256, 576, 192, 512],
            position,
            false,
            stream,
        )?;
        a.rotary(s.kv, row, 1, [576, 576, 512, 512], position, false, stream)?;
        if a.index.is_some() {
            let _index = timing::span("dsa_index", stream)?;
            self.run_index(x, position, stream)?;
        }
        let s = &self.scratch;
        let (indices, count) = match a.window {
            Some(window) => (DevicePtr::NULL, (self.len + 1).min(window)),
            None => (s.indices, (self.len / 4).min(511) * 4 + self.len % 4 + 1),
        };
        let splits = count.div_ceil(128);
        let phase = timing::span("attn_latent", stream)?;
        KernelLaunch::new(gpu, a.kernels.attention)
            .grid([a.heads as u32, splits as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(s.absorbed)
            .arg_ptr(self.latent)
            .arg_ptr(self.table)
            .arg_ptr(indices)
            .arg_ptr(s.softmax_parts)
            .arg_u32(self.len as u32)
            .arg_u32(self.layout.page_size as u32)
            .arg_u32(self.layout.table.len() as u32)
            .arg_u32(a.window.unwrap_or(0) as u32)
            .arg_u32(count as u32)
            .arg_u32(splits as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, a.kernels.merge)
            .grid([a.heads as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.softmax_parts)
            .arg_ptr(a.sink)
            .arg_ptr(s.latent_out)
            .arg_u32(splits as u32)
            .launch(stream)?;
        drop(phase);
        let phase = timing::span("attn_value_extract", stream)?;
        a.gemv(s.latent_out, a.wv, s.values, [256, 512, 512, 256], stream)?;
        drop(phase);
        let phase = timing::span("attn_gate_projection", stream)?;
        a.linear(x, a.gate, s.gates, a.heads * 256, a.hidden, stream)?;
        drop(phase);
        let phase = timing::span("attn_gate", stream)?;
        KernelLaunch::new(gpu, a.kernels.gate)
            .grid([(a.heads * 256).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.values)
            .arg_ptr(s.gates)
            .arg_u32((a.heads * 256) as u32)
            .launch(stream)?;
        drop(phase);
        let phase = timing::span("attn_output_projection", stream)?;
        a.linear(
            s.values,
            a.output,
            s.partial,
            a.hidden,
            a.heads * 256,
            stream,
        )?;
        drop(phase);
        if let Some(comm) = comm {
            let _reduce = timing::span("all_reduce", stream)?;
            comm.all_reduce_async(s.partial.0, a.hidden * 2, stream)?;
        }
        a.norm(s.partial, a.norm, out, a.hidden, stream)
    }
}
