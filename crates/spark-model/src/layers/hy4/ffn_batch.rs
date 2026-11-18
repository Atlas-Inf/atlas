// SPDX-License-Identifier: AGPL-3.0-only

use super::ffn::Hy4Ffn;
use anyhow::Result;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl Hy4Ffn<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn batch_mlp(
        &self,
        x: DevicePtr,
        activated: DevicePtr,
        out: DevicePtr,
        ids: DevicePtr,
        weights: DevicePtr,
        dims: [usize; 4],
        limit: f32,
        rows: u32,
        batch: [DevicePtr; 2],
        stream: u64,
    ) -> Result<()> {
        let [h, i, slots, group] = dims;
        let [groups, partials] = batch;
        let experts = if ids.is_null() { 1 } else { self.ep.experts };
        KernelLaunch::new(self.gpu, self.gpu.kernel("hy4_ffn", "hy4_group_init")?)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(groups)
            .arg_u32(rows)
            .arg_u32(experts as u32)
            .launch(stream)?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("hy4_ffn", "hy4_group_rows")?)
            .grid([experts as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids)
            .arg_ptr(groups)
            .arg_u32(rows)
            .arg_u32(slots as u32)
            .launch(stream)?;
        for (input, output, n, k, down) in [(x, activated, i, h, 0), (activated, partials, h, i, 1)]
        {
            KernelLaunch::new(self.gpu, self.gpu.kernel("hy4_ffn", "hy4_batch_expert")?)
                .grid([
                    n.div_ceil(8) as u32,
                    1,
                    experts.min(rows as usize * slots) as u32,
                ])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(self.table)
                .arg_ptr(groups)
                .arg_ptr(output)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(rows)
                .arg_u32(slots as u32)
                .arg_u32(group as u32)
                .arg_u32(down)
                .arg_f32(limit)
                .arg_u32(experts as u32)
                .launch(stream)?;
        }
        KernelLaunch::new(self.gpu, self.gpu.kernel("hy4_ffn", "hy4_batch_combine")?)
            .grid([h.div_ceil(256) as u32, rows, 1])
            .block([256, 1, 1])
            .arg_ptr(partials)
            .arg_ptr(weights)
            .arg_ptr(out)
            .arg_u32(h as u32)
            .arg_u32(slots as u32)
            .launch(stream)
    }
}
