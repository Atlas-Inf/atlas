// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};

pub const MAX_ROWS: usize = 128;

pub(crate) fn gemm(
    gpu: &dyn GpuBackend,
    buffers: [DevicePtr; 4],
    dimensions: [usize; 9],
    flags: [u32; 2],
    stream: u64,
) -> Result<()> {
    let [n, _, rows, heads, ..] = dimensions;
    let mut launch = KernelLaunch::new(
        gpu,
        gpu.op_cache().kernel(gpu, "hy4_ffn", "hy4_batch_linear")?,
    )
    .grid([n.div_ceil(8) as u32, rows.div_ceil(8) as u32, heads as u32])
    .block([256, 1, 1]);
    for ptr in buffers {
        launch = launch.arg_ptr(ptr);
    }
    for n in dimensions {
        launch = launch.arg_u32(u32::try_from(n)?);
    }
    for flag in flags {
        launch = launch.arg_u32(flag);
    }
    launch.launch(stream)
}

pub(crate) fn linear(
    gpu: &dyn GpuBackend,
    buffers: [DevicePtr; 4],
    dimensions: [usize; 3],
    flags: [u32; 2],
    stream: u64,
) -> Result<()> {
    let [n, k, rows] = dimensions;
    gemm(gpu, buffers, [n, k, rows, 1, 0, 0, k, n, k], flags, stream)
}
