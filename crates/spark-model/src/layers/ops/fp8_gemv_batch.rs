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

/// Kernel for the block-scaled FP8 verify GEMV at `m` rows (5..=16) and whether it is a
/// `_vl2` twin (8 outputs/block -> launch with `w8a16_gemv_batch4_vl2`). 0-handles = not linked.
pub fn fp8_verify_gemv_tier(
    m: usize,
    vl2_on: bool,
    batch8: KernelHandle,
    batch8_vl2: KernelHandle,
    batch8_dyn_vl2: KernelHandle,
    batch16: KernelHandle,
) -> (KernelHandle, bool) {
    if m <= 8 {
        if vl2_on && m == 8 && batch8_vl2.0 != 0 {
            return (batch8_vl2, true);
        }
        if vl2_on && batch8_dyn_vl2.0 != 0 {
            return (batch8_dyn_vl2, true);
        }
        if batch8.0 != 0 {
            return (batch8, false);
        }
    }
    (batch16, false)
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
        let v8 = KernelHandle(2);
        let d8 = KernelHandle(3);
        let b16 = KernelHandle(4);
        let z = KernelHandle(0);
        let pick = |m: usize, on: bool, a, b, c, d| {
            let (k, v) = fp8_verify_gemv_tier(m, on, a, b, c, d);
            (k.0, v)
        };
        assert_eq!(pick(8, true, b8, v8, d8, b16), (2, true));
        assert_eq!(pick(6, true, b8, v8, d8, b16), (3, true));
        assert_eq!(pick(8, true, b8, z, d8, b16), (3, true));
        assert_eq!(pick(7, false, b8, v8, d8, b16), (1, false));
        assert_eq!(pick(8, true, b8, z, z, b16), (1, false));
        assert_eq!(pick(8, true, z, z, z, b16), (4, false));
        assert_eq!(pick(12, true, b8, v8, d8, b16), (4, false));
    }
}
