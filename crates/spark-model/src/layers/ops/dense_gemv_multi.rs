// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row BF16 dense GEMV: tensor-core tiers and fused dual/triple projections
//! (plus the prefill GEMM triple, which shares the triple argument list).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

use super::DENSE_GEMV_BATCHM_MAX_M;

/// Widest row count [`dense_gemv_bf16_tc`] serves (`dense_gemv_bf16_tc32`).
pub const DENSE_GEMV_TC_MAX_M: u32 = 32;

/// Most CTAs a load-ahead dense GEMV grid may span (`AHEAD` in GLM's
/// `dense_gemv_bf16_batchm.cu`): that body's registers leave two CTAs per SM
/// (96 on GB10), and a wider grid runs slower than the plain tiers.
pub const DENSE_GEMV_AHEAD_MAX_CTAS: u32 = 96;

/// Whether the fused verify triples' three-plane grid fits the load-ahead
/// body they run; a wider shape takes one plain launch per projection.
pub fn dense_gemv_triple_fits(first_n: u32, other_n: u32) -> bool {
    div_ceil(first_n.max(other_n), 4) <= DENSE_GEMV_AHEAD_MAX_CTAS / 3
}

/// `dense_gemv_bf16_tc16` / `_tc32` for `m` rows (9..=32) when the tensor-core
/// verify tiers are enabled (`ATLAS_W4A16_TC=1`), else a zero handle.
pub fn dense_tc_kernel(gpu: &dyn GpuBackend, m: u32) -> KernelHandle {
    static TC: std::sync::OnceLock<(KernelHandle, KernelHandle)> = std::sync::OnceLock::new();
    if !(9..=DENSE_GEMV_TC_MAX_M).contains(&m)
        || std::env::var("ATLAS_W4A16_TC").as_deref() != Ok("1")
    {
        return KernelHandle(0);
    }
    let (tc16, tc32) = *TC.get_or_init(|| {
        let k = |name| {
            gpu.kernel("dense_gemv_bf16_batchm", name)
                .unwrap_or(KernelHandle(0))
        };
        (k("dense_gemv_bf16_tc16"), k("dense_gemv_bf16_tc32"))
    });
    if m <= 16 { tc16 } else { tc32 }
}

/// BF16 `C[m, n] = A[m, k] · W[n, k]ᵀ` for 9..=32 rows on the tensor cores:
/// one weight pass for all rows (see `dense_gemv_bf16_tc_impl`). `kernel` is
/// `dense_gemv_bf16_tc16` (m <= 16) or `dense_gemv_bf16_tc32` (m <= 32).
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_bf16_tc(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DENSE_GEMV_TC_MAX_M).contains(&m) && k.is_multiple_of(8),
        "dense_gemv_bf16_tc: m={m} k={k} unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Two same-shape exact-M=5 BF16 projections in one two-plane launch.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batch5_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    first_input: DevicePtr,
    second_input: DevicePtr,
    first_weight: &DenseWeight,
    second_weight: &DenseWeight,
    first_output: DevicePtr,
    second_output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    dense_gemv_dual(
        gpu,
        kernel,
        [first_input, second_input],
        [first_weight, second_weight],
        [first_output, second_output],
        None,
        n,
        k,
        4,
        stream,
    )
}

/// `dense_gemv_bf16_batchm_dual_k128`, the bit-identical K = 128 tier of the
/// dual (16 outputs per CTA), or a zero handle when the target lacks it.
/// Memoized on the backend (the handle dies with its registry);
/// `Glm5KdaLayer::new` resolves it first, before the boot audit seals.
pub fn dense_gemv_dual_k128_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    gpu.op_cache()
        .kernel(
            gpu,
            "dense_gemv_bf16_batchm",
            "dense_gemv_bf16_batchm_dual_k128",
        )
        .unwrap_or(KernelHandle(0))
}

/// Two same-shape BF16 projections of `m` (<= 8) rows in one grid
/// (`dense_gemv_bf16_batchm_dual`, or its K = 128 tier), each bit-identical
/// to its batchm launch.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    inputs: [DevicePtr; 2],
    weights: [&DenseWeight; 2],
    outputs: [DevicePtr; 2],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm_dual takes 1..={DENSE_GEMV_BATCHM_MAX_M} rows, got {m}"
    );
    let k128 = (k == 128)
        .then(|| dense_gemv_dual_k128_kernel(gpu))
        .filter(|h| h.0 != 0);
    let (kernel, outs_per_cta) = k128.map_or((kernel, 4), |h| (h, 16));
    dense_gemv_dual(
        gpu,
        kernel,
        inputs,
        weights,
        outputs,
        Some(m),
        n,
        k,
        outs_per_cta,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_gemv_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    inputs: [DevicePtr; 2],
    weights: [&DenseWeight; 2],
    outputs: [DevicePtr; 2],
    m: Option<u32>,
    n: u32,
    k: u32,
    outs_per_cta: u32,
    stream: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outs_per_cta), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(inputs[0])
        .arg_ptr(inputs[1])
        .arg_ptr(weights[0].weight)
        .arg_ptr(weights[1].weight)
        .arg_ptr(outputs[0])
        .arg_ptr(outputs[1]);
    if let Some(m) = m {
        l = l.arg_u32(m);
    }
    l.arg_u32(n).arg_u32(k).launch(stream)
}

/// Three same-input exact-M=5 BF16 projections; the first may have a smaller N.
/// A load-ahead grid: the widths must satisfy [`dense_gemv_triple_fits`].
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batch5_triple_n(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    first_weight: &DenseWeight,
    second_weight: &DenseWeight,
    third_weight: &DenseWeight,
    first_output: DevicePtr,
    second_output: DevicePtr,
    third_output: DevicePtr,
    first_n: u32,
    other_n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    dense_gemv_triple_n(
        gpu,
        kernel,
        input,
        [first_weight, second_weight, third_weight],
        [first_output, second_output, third_output],
        None,
        [first_n, other_n],
        k,
        stream,
    )
}

/// Three same-input BF16 projections of `m` (<= 8) rows in one grid
/// (`dense_gemv_bf16_batchm_triple_n`); the first may have a smaller N.
/// A load-ahead grid: the widths must satisfy [`dense_gemv_triple_fits`].
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_triple_n(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weights: [&DenseWeight; 3],
    outputs: [DevicePtr; 3],
    m: u32,
    [first_n, other_n]: [u32; 2],
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm_triple_n takes 1..={DENSE_GEMV_BATCHM_MAX_M} rows, got {m}"
    );
    dense_gemv_triple_n(
        gpu,
        kernel,
        input,
        weights,
        outputs,
        Some(m),
        [first_n, other_n],
        k,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_gemv_triple_n(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weights: [&DenseWeight; 3],
    outputs: [DevicePtr; 3],
    m: Option<u32>,
    [first_n, other_n]: [u32; 2],
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        first_n.is_multiple_of(4) && other_n.is_multiple_of(4),
        "dense GEMV triple requires output widths divisible by 4 (got {first_n} and {other_n})"
    );
    ensure!(
        dense_gemv_triple_fits(first_n, other_n),
        "dense GEMV triple is a load-ahead grid of at most {DENSE_GEMV_AHEAD_MAX_CTAS} CTAs \
         (got widths {first_n} and {other_n}); launch the projections separately"
    );
    launch_triple_n(
        gpu,
        kernel,
        [div_ceil(first_n.max(other_n), 4), 1, 3],
        input,
        weights,
        outputs,
        m,
        [first_n, other_n],
        k,
        stream,
    )
}

/// Prefill twin of [`dense_gemv_batchm_triple_n`] (`dense_gemm_bf16_pipelined_triple_n`):
/// each plane runs the unchanged `dense_gemm_bf16_pipelined` body, so it is
/// bit-identical to three [`super::dense_gemm_bf16_pipelined`] launches, while
/// the planes' CTAs over one 128-row activation tile run back to back and
/// share it in L2 instead of streaming the activation three times.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_pipelined_triple_n(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weights: [&DenseWeight; 3],
    outputs: [DevicePtr; 3],
    m: u32,
    [first_n, other_n]: [u32; 2],
    k: u32,
    stream: u64,
) -> Result<()> {
    let tiles = |n| div_ceil(n, 128);
    launch_triple_n(
        gpu,
        kernel,
        [tiles(first_n) + 2 * tiles(other_n), div_ceil(m, 128), 1],
        input,
        weights,
        outputs,
        Some(m),
        [first_n, other_n],
        k,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_triple_n(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    input: DevicePtr,
    weights: [&DenseWeight; 3],
    outputs: [DevicePtr; 3],
    m: Option<u32>,
    [first_n, other_n]: [u32; 2],
    k: u32,
    stream: u64,
) -> Result<()> {
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weights[0].weight)
        .arg_ptr(weights[1].weight)
        .arg_ptr(weights[2].weight)
        .arg_ptr(outputs[0])
        .arg_ptr(outputs[1])
        .arg_ptr(outputs[2]);
    if let Some(m) = m {
        l = l.arg_u32(m);
    }
    l.arg_u32(first_n)
        .arg_u32(other_n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
#[path = "dense_gemv_multi_tests.rs"]
mod tests;
