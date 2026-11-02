// SPDX-License-Identifier: AGPL-3.0-only

//! FP8-weight dual-GEMV (batch=2) dispatch.
//!
//! `dense_gemv_fp8w_batch2` computes two output rows from one pass over the
//! FP8 weight matrix — the batch=2 sibling of `dense_gemv_fp8w`. It halves
//! FP8 weight bandwidth vs two M=1 GEMV launches and is bit-identical to
//! running `dense_gemv_fp8w` twice (per-token reduction order unchanged).
//! Used by the K=2 MTP verify path where the two verify positions share
//! weights but have distinct activations (lm_head, attention Q/K/V/O, SSM
//! out_proj).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

/// FP8-weight dual-GEMV. `input` is `[2, K]` BF16, `output` is `[2, N]` BF16.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv_fp8w_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Per-row-scaled FP8 batched GEMV (M<=8). `input` is `[M, K]` BF16,
/// `output` is M rows at `output + t*out_stride` BF16. One pass over the
/// FP8 weight serves all M rows — the M<=8 sibling of `dense_gemv_fp8w`,
/// with the same per-row K-order and end-of-dot-product scale application.
/// `out_stride` decouples output row stride from N (e.g. the multi-token
/// verify projection buffer). Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_fp8w_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Block-scaled FP8 batched GEMV (M<=4). `input` is `[M, K]` BF16, `output` is
/// `[M, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers (2D
/// block-scaled FP8). One pass over the FP8 weight serves all M rows — the M=4
/// sibling of `w8a16_gemv`, replacing `w8a16_gemm_pipelined` for n<=4 batched
/// decode (which pads M to a 128-row MMA tile). Bit-identical per-row to
/// `w8a16_gemv`. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Kernel for the block-scaled FP8 verify GEMV at `m` rows (5..=16) and the
/// output-columns-per-block divisor for its grid: 4 = plain batchm twin
/// (`w8a16_gemv_batch4`), 8 = `_vl2` twin (`w8a16_gemv_batch4_vl2`), 16 = the
/// n2 twin (two outputs per thread). 0-handles = not linked.
pub fn fp8_verify_gemv_tier(
    m: usize,
    vl2_on: bool,
    batch8: KernelHandle,
    batch8_dyn_vl2: KernelHandle,
    batch8_dyn_vl2_n2: KernelHandle,
    batch16: KernelHandle,
) -> (KernelHandle, u32) {
    if m <= 8 {
        if vl2_on {
            // n2 is bit-identical to dyn_vl2 (same per-output lane->k16 map
            // and reduction order) and measured 1.15-1.31x on the verify
            // shapes; it handles any m <= 8 through its row guards.
            if batch8_dyn_vl2_n2.0 != 0 {
                return (batch8_dyn_vl2_n2, 16);
            }
            if batch8_dyn_vl2.0 != 0 {
                return (batch8_dyn_vl2, 8);
            }
        }
        if batch8.0 != 0 {
            return (batch8, 4);
        }
    }
    (batch16, 4)
}

/// Same call surface as `w8a16_gemv_batch4` but with the output-columns-per-
/// block divisor passed in: 4 (plain twins), 8 (`*_vl2`), 16 (`*_n2`).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4_div(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    div: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, div), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// vl2 twin of `w8a16_gemv_batch4`-family calls: the `*_vl2` kernels cover
/// 8 outputs per block — grid `ceil(n/8)`.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4_vl2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block-scaled FP8 dual-GEMV (batch=2). `input` is `[2, K]` BF16, `output` is
/// `[2, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_verify_tier_pick() {
        let b8 = KernelHandle(1);
        let d8 = KernelHandle(3);
        let n2 = KernelHandle(5);
        let b16 = KernelHandle(4);
        let z = KernelHandle(0);
        let pick = |m: usize, on: bool, a, b, c, d| {
            let (k, v) = fp8_verify_gemv_tier(m, on, a, b, c, d);
            (k.0, v)
        };
        assert_eq!(pick(8, true, b8, d8, n2, b16), (5, 16));
        assert_eq!(pick(6, true, b8, d8, n2, b16), (5, 16));
        assert_eq!(pick(8, true, b8, d8, z, b16), (3, 8));
        assert_eq!(pick(8, false, b8, d8, n2, b16), (1, 4));
        assert_eq!(pick(8, true, b8, z, z, b16), (1, 4));
        assert_eq!(pick(8, true, z, z, z, b16), (4, 4));
        assert_eq!(pick(12, true, b8, d8, n2, b16), (4, 4));
    }
}
