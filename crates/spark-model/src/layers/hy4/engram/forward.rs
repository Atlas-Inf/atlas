// SPDX-License-Identifier: AGPL-3.0-only

use super::{EngramScratch, Hy4Engram};
use crate::layers::hy4::timing;
use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl Hy4Engram<'_> {
    pub fn forward(
        &self,
        hidden: DevicePtr,
        history: &[u32],
        token: u32,
        scratch: &mut EngramScratch<'_>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.mem.is_live()
                && scratch.mem.is_live()
                && scratch.signature == [self.h, self.heads.start, self.heads.end]
                && !hidden.is_null(),
            "Hy4 Engram released/incompatible scratch or hidden"
        );
        let comm = if self.topology[1] > 1 {
            let c = comm.context("Hy4 Engram TP requires communicator")?;
            ensure!(
                [c.rank(), c.world_size()] == self.topology,
                "Hy4 Engram communicator topology mismatch"
            );
            Some(c)
        } else {
            None
        };
        let previous = history.last().copied().unwrap_or(u32::MAX);
        let older = history
            .len()
            .checked_sub(2)
            .map(|i| history[i])
            .unwrap_or(u32::MAX);
        KernelLaunch::new(self.mem.gpu, self.kernels[0])
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_u32(token)
            .arg_u32(previous)
            .arg_u32(older)
            .arg_ptr(self.map)
            .arg_ptr(self.multipliers)
            .arg_ptr(self.sizes)
            .arg_ptr(self.offsets)
            .arg_ptr(scratch.hashes)
            .arg_u32(self.layout.map.len() as u32)
            .arg_u32(self.layout.sizes.len() as u32)
            .arg_u32(self.layout.heads_per_order as u32)
            .arg_u32(self.layout.pad)
            .launch(stream)?;
        let lookup = KernelLaunch::new(self.mem.gpu, self.kernels[1])
            .grid([self.heads.len() as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(scratch.hashes)
            .arg_ptr(self.table);
        let lookup = if self.fp8_table {
            lookup.arg_ptr(self.scales)
        } else {
            lookup
        };
        lookup
            .arg_ptr(self.lookup_norm)
            .arg_ptr(scratch.lookup)
            .arg_u32(self.heads.start as u32)
            .arg_u32(self.rows.start as u32)
            .arg_u32(self.rows.len() as u32)
            .arg_u32(self.dim as u32)
            .launch(stream)?;
        for i in 0..5 {
            let out = if i == 4 {
                scratch.value
            } else {
                scratch.keys.offset(i * self.h * 2)
            };
            KernelLaunch::new(self.mem.gpu, self.kernels[2])
                .grid([(self.h as u32).div_ceil(128), 1, 1])
                .block([128, 1, 1])
                .arg_ptr(scratch.lookup)
                .arg_ptr(self.projections[i])
                .arg_ptr(self.biases[i])
                .arg_ptr(out)
                .arg_u32(self.h as u32)
                .arg_u32((self.layout.sizes.len() * self.dim) as u32)
                .arg_u32((self.heads.start * self.dim) as u32)
                .arg_u32((self.heads.len() * self.dim) as u32)
                .arg_u32(u32::from(self.topology[0] == 0))
                .launch(stream)?;
            if let Some(c) = comm {
                let _reduce = timing::span("all_reduce", stream)?;
                c.all_reduce_async(out.0, self.h * 2, stream)?;
            }
        }
        for (input, weight, out) in [
            (scratch.keys, self.key_norm, scratch.norm_keys),
            (hidden, self.query_norm, scratch.queries),
        ] {
            KernelLaunch::new(self.mem.gpu, self.kernels[3])
                .grid([4, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(weight)
                .arg_ptr(out)
                .arg_u32(self.h as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(self.mem.gpu, self.kernels[4])
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden)
            .arg_ptr(scratch.norm_keys)
            .arg_ptr(scratch.queries)
            .arg_ptr(scratch.value)
            .arg_ptr(self.gamma)
            .arg_ptr(self.hc_scales)
            .arg_ptr(scratch.hashes)
            .arg_u32(self.h as u32)
            .launch(stream)
    }
}
