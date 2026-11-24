// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    kernel_args::KernelLaunch,
};

pub const MAX_ROWS: usize = 256;
/// Prefill chunk width: the wide workspace and the expert-FFN batch gate.
/// The expert FFN streams its weights once per chunk, so wider chunks cut
/// per-token weight traffic; attention and Engram still process each chunk
/// in MAX_ROWS slices, and VERIFY keeps using the MAX_ROWS-wide workspace.
pub const PREFILL_ROWS: usize = 1024;

pub(crate) fn gemm(
    gpu: &dyn GpuBackend,
    buffers: [DevicePtr; 4],
    dimensions: [usize; 9],
    flags: [u32; 2],
    stream: u64,
) -> Result<()> {
    let [n, k, rows, heads, ..] = dimensions;
    let [kind, float_out] = flags;
    // Tensor-core paths gated by ATLAS_HY4_LINEAR_MMA=1: fp8-e4m3 weights
    // decode to bf16 in-kernel (numerics identical to the GEMV, bf16
    // output only), and bf16 weights run mma directly. Requires K%32;
    // crossover vs the GEMV is rows~32 on GB10 (kernel bench), so decode
    // stays on the GEMV.
    // The fp32 router GEMM (the only float_out==1 caller) stays on the
    // scalar GEMV by default because fp32 accumulation order flips top-k
    // expert ties (parity break). ATLAS_HY4_ROUTER_MMA=1 opts it into the
    // same bf16 mma kernel with fp32 accumulate and fp32 logits output —
    // top-k selection still reads fp32 logits, unchanged — accepting
    // top-k drift for prefill speed evals.
    let linear_mma = std::env::var("ATLAS_HY4_LINEAR_MMA").ok().as_deref() == Some("1");
    let router_mma = std::env::var("ATLAS_HY4_ROUTER_MMA").ok().as_deref() == Some("1");
    let mma_fp8 = kind == 1 && float_out == 0 && linear_mma;
    let mma_bf16 = kind == 0 && float_out == 0 && linear_mma;
    let mma_router = kind == 0 && float_out == 1 && router_mma;
    if (mma_fp8 || mma_bf16 || mma_router) && k % 32 == 0 && rows >= 32 {
        let name = if mma_fp8 {
            "hy4_batch_linear_mma"
        } else if buffers[2].is_null() {
            "hy4_batch_linear_bf16_mma"
        } else {
            "hy4_batch_linear_bf16_mma_bias"
        };
        let mut launch = KernelLaunch::new(gpu, gpu.op_cache().kernel(gpu, "hy4_ffn", name)?)
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
        if mma_bf16 || mma_router {
            launch = launch.arg_u32(float_out);
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
