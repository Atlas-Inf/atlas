// SPDX-License-Identifier: AGPL-3.0-only

use super::{EngramScratch, Hy4Engram};
use crate::layers::hy4::{attention::memory::Owned, batch};
use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

pub(super) struct BatchScratch<'a> {
    _memory: Owned<'a>,
    hashes: DevicePtr,
    lookup: DevicePtr,
    keys: DevicePtr,
    norm_keys: DevicePtr,
    queries: DevicePtr,
    value: DevicePtr,
}
impl<'a> BatchScratch<'a> {
    fn new(e: &Hy4Engram<'a>) -> Result<Self> {
        let mut m = Owned::new(e.mem.gpu);
        let rows = batch::MAX_ROWS;
        Ok(Self {
            hashes: m.alloc(rows * e.layout.sizes.len() * 4)?,
            lookup: m.alloc(rows * e.heads.len() * e.dim * 2)?,
            keys: m.alloc(rows * 4 * e.h * 2)?,
            norm_keys: m.alloc(rows * 4 * e.h * 2)?,
            queries: m.alloc(rows * 4 * e.h * 2)?,
            value: m.alloc(rows * e.h * 2)?,
            _memory: m,
        })
    }
}

impl<'a> Hy4Engram<'a> {
    pub fn forward_batch(
        &self,
        hidden: DevicePtr,
        history: &[u32],
        start: usize,
        scratch: &mut EngramScratch<'a>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let rows = history
            .len()
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("Hy4 Engram batch start"))?;
        ensure!(
            rows > 0
                && rows <= batch::MAX_ROWS
                && !hidden.is_null()
                && self.mem.is_live()
                && scratch.mem.is_live()
                && scratch.signature == [self.h, self.heads.start, self.heads.end],
            "Hy4 Engram batch geometry"
        );
        if self.topology[1] > 1 {
            ensure!(
                comm.is_some_and(|c| [c.rank(), c.world_size()] == self.topology),
                "Hy4 Engram batch communicator"
            );
        }
        let b = match scratch.batch.take() {
            Some(b) => b,
            None => BatchScratch::new(self)?,
        };
        let result = self.batch_rows(hidden, history, start, &b, comm, stream);
        scratch.batch = Some(b);
        result
    }

    fn batch_rows(
        &self,
        hidden: DevicePtr,
        history: &[u32],
        start: usize,
        b: &BatchScratch<'_>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let gpu = self.mem.gpu;
        let rows = history.len() - start;
        let heads = self.layout.sizes.len();
        let local = self.heads.len() * self.dim;
        for row in 0..rows {
            let _lookup = super::super::timing::span("engram_lookup", stream)?;
            let p = start + row;
            let hashes = b.hashes.offset(row * heads * 4);
            KernelLaunch::new(gpu, self.kernels[0])
                .grid([1, 1, 1])
                .block([32, 1, 1])
                .arg_u32(history[p])
                .arg_u32(p.checked_sub(1).map_or(u32::MAX, |i| history[i]))
                .arg_u32(p.checked_sub(2).map_or(u32::MAX, |i| history[i]))
                .arg_ptr(self.map)
                .arg_ptr(self.multipliers)
                .arg_ptr(self.sizes)
                .arg_ptr(self.offsets)
                .arg_ptr(hashes)
                .arg_u32(self.layout.map.len() as u32)
                .arg_u32(heads as u32)
                .arg_u32(self.layout.heads_per_order as u32)
                .arg_u32(self.layout.pad)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.kernels[1])
                .grid([self.heads.len() as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(hashes)
                .arg_ptr(self.table)
                .arg_ptr(self.scales)
                .arg_ptr(self.lookup_norm)
                .arg_ptr(b.lookup.offset(row * local * 2))
                .arg_u32(self.heads.start as u32)
                .arg_u32(self.rows.start as u32)
                .arg_u32(self.rows.len() as u32)
                .arg_u32(self.dim as u32)
                .launch(stream)?;
        }
        for i in 0..5 {
            let _project = super::super::timing::span("engram_project", stream)?;
            let (out, stride) = if i == 4 {
                (b.value, self.h)
            } else {
                (b.keys.offset(i * self.h * 2), 4 * self.h)
            };
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_engram", "hy4_engram_batch_project")?,
            )
            .grid([self.h.div_ceil(128) as u32, rows.div_ceil(8) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(b.lookup)
            .arg_ptr(self.projections[i])
            .arg_ptr(self.biases[i])
            .arg_ptr(out)
            .arg_u32(self.h as u32)
            .arg_u32((heads * self.dim) as u32)
            .arg_u32((self.heads.start * self.dim) as u32)
            .arg_u32(local as u32)
            .arg_u32(u32::from(self.topology[0] == 0))
            .arg_u32(rows as u32)
            .arg_u32(stride as u32)
            .launch(stream)?;
        }
        if let Some(comm) = comm.filter(|_| self.topology[1] > 1) {
            let _reduce = super::super::timing::span("all_reduce", stream)?;
            let bytes = rows * self.h * 2;
            for part in 0..4 {
                comm.all_reduce_async(b.keys.offset(part * bytes).0, bytes, stream)?;
            }
            comm.all_reduce_async(b.value.0, rows * self.h * 2, stream)?;
        }
        for row in 0..rows {
            let _finish = super::super::timing::span("engram_norm_gate", stream)?;
            let highway = hidden.offset(row * 4 * self.h * 2);
            let norm_keys = b.norm_keys.offset(row * 4 * self.h * 2);
            let queries = b.queries.offset(row * 4 * self.h * 2);
            for (input, weight, out) in [
                (
                    b.keys.offset(row * 4 * self.h * 2),
                    self.key_norm,
                    norm_keys,
                ),
                (highway, self.query_norm, queries),
            ] {
                KernelLaunch::new(gpu, self.kernels[3])
                    .grid([4, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(weight)
                    .arg_ptr(out)
                    .arg_u32(self.h as u32)
                    .launch(stream)?;
            }
            KernelLaunch::new(gpu, self.kernels[4])
                .grid([4, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(highway)
                .arg_ptr(norm_keys)
                .arg_ptr(queries)
                .arg_ptr(b.value.offset(row * self.h * 2))
                .arg_ptr(self.gamma)
                .arg_ptr(self.hc_scales)
                .arg_ptr(b.hashes.offset(row * heads * 4))
                .arg_u32(self.h as u32)
                .launch(stream)?;
        }
        Ok(())
    }
}
