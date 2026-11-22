// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};

pub const MAX_ROWS: usize = 256;

pub(crate) fn gemm(
    gpu: &dyn GpuBackend,
    buffers: [DevicePtr; 4],
    dimensions: [usize; 9],
    flags: [u32; 2],
    stream: u64,
) -> Result<()> {
    let [n, k, rows, heads, ..] = dimensions;
    let [kind, float_out] = flags;
    // Tensor-core path for fp8-e4m3 weights, gated by
    // ATLAS_HY4_LINEAR_MMA=1: bf16 mma with in-kernel e4m3->bf16 decode,
    // numerics identical to the GEMV. Requires K%32 and bf16 output.
    // Crossover vs the GEMV is rows≈32 on GB10 (kernel bench); decode
    // stays on the GEMV.
    let mma = kind == 1
        && float_out == 0
        && k % 32 == 0
        && rows >= 32
        && std::env::var("ATLAS_HY4_LINEAR_MMA").ok().as_deref() == Some("1");
    if mma {
        let mut launch = KernelLaunch::new(
            gpu,
            gpu.op_cache()
                .kernel(gpu, "hy4_ffn", "hy4_batch_linear_mma")?,
        )
        .grid([
            n.div_ceil(128) as u32,
            rows.div_ceil(64) as u32,
            heads as u32,
        ])
        .block([256, 1, 1]);
        for ptr in buffers {
            launch = launch.arg_ptr(ptr);
        }
        for n in dimensions {
            launch = launch.arg_u32(u32::try_from(n)?);
        }
        return launch.launch(stream);
    }
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
