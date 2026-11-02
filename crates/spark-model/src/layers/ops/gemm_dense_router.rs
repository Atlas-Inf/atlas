// SPDX-License-Identifier: AGPL-3.0-only

//! Order-preserving router GEMMs for decode/verify row counts.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// Order-preserving router GEMM for `m <= 32` rows (kernel
/// `dense_gemm_bf16_router_rows`): one warp per output column, one lane per
/// row, strict k order — bit-identical to [`super::dense_gemm`] at decode/verify
/// widths where the tiled kernels launch only a handful of blocks.
///
/// Grid: (ceil(N/4), 1, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_router_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=32).contains(&m),
        "dense_gemm_router_rows: m={m} outside 1..=32"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Exact-M=5 order-preserving router GEMM. The kernel retains one sequential
/// FP32 accumulator per output while eliminating the generic tile's eleven
/// padded row lanes. Callers must guard `m == 5`.
pub fn dense_gemm_router_m5(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    debug_assert_eq!(m, 5);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([16, 5, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
