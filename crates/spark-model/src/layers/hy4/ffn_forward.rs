// SPDX-License-Identifier: AGPL-3.0-only

use super::{Hy4FfnScratch, ffn::Hy4Ffn};
use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl Hy4Ffn<'_> {
    /// BF16 `[rows,H]` -> `[rows,H]`, distinct from all input/scratch buffers.
    /// Routed partials are all-reduced AFTER latent up, BEFORE shared addition
    /// and sandwich norm. No shared output is ever all-reduced.
    pub fn forward(
        &self,
        x: DevicePtr,
        out: DevicePtr,
        rows: u32,
        scratch: &mut Hy4FfnScratch<'_>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        ensure!(!self.table.is_null(), "Hy4 FFN already released");
        ensure!(
            scratch.signature == self.scratch_signature() && rows as usize <= scratch.capacity,
            "Hy4 scratch shape/capacity mismatch"
        );
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            !x.is_null()
                && !out.is_null()
                && x != out
                && !scratch.ptrs.contains(&x)
                && !scratch.ptrs.contains(&out),
            "Hy4 FFN invalid/aliased buffer"
        );
        let comm = if !self.dense && self.ep.world > 1 {
            let c = comm.context("Hy4 EP2 requires a communicator")?;
            ensure!(
                c.rank() == self.ep.rank && c.world_size() == self.ep.world,
                "Hy4 EP communicator topology mismatch"
            );
            Some(c)
        } else {
            None
        };
        let [
            logits,
            ids,
            weights,
            latent,
            activated,
            latent_out,
            routed,
            shared,
        ] = scratch.ptrs;
        if self.dense {
            self.mlp(
                x,
                activated,
                shared,
                DevicePtr::NULL,
                DevicePtr::NULL,
                [self.h, self.inter, 1, 0],
                0.0,
                rows,
                stream,
            )?;
        } else {
            // BF16 checkpoint router, FP32 dot products/logits.
            self.linear(
                x,
                self.router,
                logits,
                [self.ep.experts, self.h],
                true,
                rows,
                stream,
            )?;
            KernelLaunch::new(self.gpu, self.kernels[1])
                .grid([rows, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(self.bias)
                .arg_ptr(ids)
                .arg_ptr(weights)
                .arg_u32(self.ep.experts as u32)
                .arg_u32(self.routing.top_k as u32)
                .arg_u32(u32::from(self.routing.renormalize))
                .arg_f32(self.routing.scaling)
                .launch(stream)?;
            self.linear(
                x,
                self.down,
                latent,
                [self.latent, self.h],
                false,
                rows,
                stream,
            )?;
            self.mlp(
                latent,
                activated,
                latent_out,
                ids,
                weights,
                [self.latent, self.inter, self.routing.top_k, 0],
                self.limit,
                rows,
                stream,
            )?;
            self.linear(
                latent_out,
                self.up,
                routed,
                [self.h, self.latent],
                false,
                rows,
                stream,
            )?;
            if let Some(comm) = comm {
                comm.all_reduce_async(routed.0, rows as usize * self.h * 2, stream)?;
            }
            self.mlp(
                x,
                activated,
                shared,
                DevicePtr::NULL,
                DevicePtr::NULL,
                [self.h, self.shared_inter, 1, self.ep.experts],
                0.0,
                rows,
                stream,
            )?;
        }
        KernelLaunch::new(self.gpu, self.kernels[4])
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(if self.dense { DevicePtr::NULL } else { routed })
            .arg_ptr(shared)
            .arg_ptr(self.norm)
            .arg_ptr(out)
            .arg_u32(self.h as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dims: [usize; 2],
        float_output: bool,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let [n, k] = dims;
        KernelLaunch::new(self.gpu, self.kernels[0])
            .grid([n.div_ceil(128) as u32, rows, 1])
            .block([128, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(0)
            .arg_u32(u32::from(float_output))
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp(
        &self,
        x: DevicePtr,
        activated: DevicePtr,
        out: DevicePtr,
        ids: DevicePtr,
        weights: DevicePtr,
        dims: [usize; 4],
        limit: f32,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let [h, i, slots, group] = dims;
        KernelLaunch::new(self.gpu, self.kernels[2])
            .grid([i.div_ceil(128) as u32, rows, slots as u32])
            .block([128, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.table)
            .arg_ptr(ids)
            .arg_ptr(activated)
            .arg_u32(h as u32)
            .arg_u32(i as u32)
            .arg_u32(slots as u32)
            .arg_u32(group as u32)
            .arg_f32(limit)
            .launch(stream)?;
        KernelLaunch::new(self.gpu, self.kernels[3])
            .grid([h.div_ceil(128) as u32, rows, 1])
            .block([128, 1, 1])
            .arg_ptr(activated)
            .arg_ptr(self.table)
            .arg_ptr(ids)
            .arg_ptr(weights)
            .arg_ptr(out)
            .arg_u32(h as u32)
            .arg_u32(i as u32)
            .arg_u32(slots as u32)
            .arg_u32(group as u32)
            .launch(stream)
    }
}
