// SPDX-License-Identifier: AGPL-3.0-only

//! The wide projections in four parts.
//!
//! Every projection that is large enough to matter is cut in four: its
//! output rows where rows are independent (an expert's w1 and w3, the query
//! heads, the output groups), its input columns where it sums over them (an
//! expert's w2, wo_b). A column part's sum is an FP32 partial, and the four
//! partials are added in part order before the one rounding
//! (`dsv41_parts.cu`).
//!
//! A rank holds `PARTS / world` consecutive parts. One Spark computes all
//! four, four Sparks one each and gather the partials. The kernels, their
//! operands and the order of the sum are the same either way, so the bytes
//! are too, and each of four Sparks reads a quarter of the weights.

use std::ops::Range;

use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::weights::Slice;

pub const PARTS: usize = 4;
/// Rows of activations one launch of a projection takes; more go in turns,
/// and a row's bytes do not depend on which rows share its launch.
pub const TILE: usize = 8;
/// The widest row a kernel that stages its weights has room for
/// (`F4_MAX_BLOCKS` scale blocks in `dsv41_tc.cuh`).
pub const STAGED_K: usize = 5120;

/// The parts one rank computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    pub first: usize,
    pub count: usize,
}

impl Held {
    /// A rank alone holds every part.
    pub const ALL: Self = Self {
        first: 0,
        count: PARTS,
    };

    pub fn of_rank(rank: usize, world: usize) -> Result<Self> {
        ensure!(
            matches!(world, 1 | 2 | 4) && rank < world,
            "deepseek_v41: rank {rank} of {world}; the parts split over 1, 2 or 4 ranks"
        );
        let count = PARTS / world;
        Ok(Self {
            first: rank * count,
            count,
        })
    }

    pub fn range(&self) -> Range<usize> {
        self.first..self.first + self.count
    }

    pub fn holds(&self, part: usize) -> bool {
        self.range().contains(&part)
    }
}

#[derive(Clone, Copy)]
pub struct PartKernels {
    /// FP4 weights, BF16 rows out; and the same with the FP32 sum unrounded.
    pub fp4: KernelHandle,
    pub fp4_f32: KernelHandle,
    /// FP8 weights with a scale per row and 32 columns.
    pub fp8: KernelHandle,
    pub fp8_f32: KernelHandle,
    /// The same, rounding their input to FP8 on the way in.
    pub fp8_q: KernelHandle,
    pub fp8_f32_q: KernelHandle,
    /// Two FP8 projections of one vector in a launch, and a block-diagonal
    /// one that rounds its input.
    pub fp8_two: KernelHandle,
    pub fp8_groups_q: KernelHandle,
    pub round: KernelHandle,
    pub combine: KernelHandle,
    pub slice: KernelHandle,
}

impl PartKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let tc = |name: &str| gpu.kernel("dsv41_tc", name);
        let parts = |name: &str| gpu.kernel("dsv41_parts", name);
        Ok(Self {
            fp4: tc("dsv41_fp4_gemv_tc8")?,
            fp4_f32: tc("dsv41_fp4_gemv_tc8_f32")?,
            fp8: tc("dsv41_fp8_gemv_tc8")?,
            fp8_f32: tc("dsv41_fp8_gemv_tc8_f32")?,
            fp8_q: tc("dsv41_fp8_gemv_tc8_q")?,
            fp8_f32_q: tc("dsv41_fp8_gemv_tc8_f32_q")?,
            fp8_two: tc("dsv41_fp8_gemv_two")?,
            fp8_groups_q: tc("dsv41_fp8_gemv_groups_q")?,
            round: parts("dsv41_parts_round")?,
            combine: parts("dsv41_moe_combine")?,
            slice: parts("dsv41_slice_columns")?,
        })
    }
}

/// `rows` vectors `x` (`[rows, k]` BF16) through a tensor-core projection `w`
/// (`[n, k]`, FP4 or FP8 by `kernel`) under `scales` (`[n, k / 32]` E8M0),
/// into `out` (BF16 or FP32 by `kernel`: `out_bytes` a value), whose rows are
/// `out_stride` values apart, `n` of each written. `x` is FP8-rounded
/// already, or with a `_q` kernel as it was computed.
#[allow(clippy::too_many_arguments)]
pub fn project_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    out_bytes: usize,
    out_stride: usize,
    rows: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(k.is_multiple_of(32), "dsv41_gemv_tc: k {k}");
    for first in (0..rows).step_by(TILE) {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 16), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x.offset(first * k * 2))
            .arg_ptr(w)
            .arg_ptr(scales)
            .arg_ptr(out.offset(first * out_stride * out_bytes))
            .arg_u32((rows - first).min(TILE) as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(out_stride as u32)
            .launch(stream)?;
    }
    Ok(())
}

/// [`project_rows`] of one vector.
#[allow(clippy::too_many_arguments)]
pub fn project(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    // One row: the width of a value does not enter.
    project_rows(gpu, kernel, x, w, scales, out, 0, n, 1, n, k, stream)
}

/// A projection's weights, rows and output, as `dsv41_fp8_gemv_two` takes
/// each of its two.
fn projection<'a>(
    launch: KernelLaunch<'a>,
    (w, n, out): (Slice, usize, DevicePtr),
) -> KernelLaunch<'a> {
    launch
        .arg_ptr(w.data)
        .arg_ptr(w.scales)
        .arg_ptr(out)
        .arg_u32(n as u32)
}

/// Two FP8 projections of the same `rows` FP8-rounded vectors `x` (`[rows,
/// k]` BF16) in one launch a tile of rows, each as [`project_rows`] would
/// compute it: `a` and `b` are `(weights, n, output [rows, n])`.
#[allow(clippy::too_many_arguments)]
pub fn project_two(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    a: (Slice, usize, DevicePtr),
    b: (Slice, usize, DevicePtr),
    k: usize,
    rows: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(32) && k <= STAGED_K,
        "dsv41_fp8_gemv_two: k {k}"
    );
    let blocks = div_ceil(a.1 as u32, 16) + div_ceil(b.1 as u32, 16);
    for first in (0..rows).step_by(TILE) {
        let from = |(w, n, out): (Slice, usize, DevicePtr)| (w, n, out.offset(first * n * 2));
        let launch = KernelLaunch::new(gpu, kernel)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x.offset(first * k * 2));
        projection(projection(launch, from(a)), from(b))
            .arg_u32(k as u32)
            .arg_u32((rows - first).min(TILE) as u32)
            .launch(stream)?;
    }
    Ok(())
}

/// A block-diagonal FP8 projection, a launch a tile of rows: group `g` of
/// `groups` is `out[row][g]` (`[n]` BF16) = `w[g]` (`[n, k]`) times the FP8
/// rounding of `x[row][g]` (`[k]` BF16, as it was computed). `w` holds the
/// groups one after another, `x` and `out` a row's groups one after another.
#[allow(clippy::too_many_arguments)]
pub fn project_groups(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w: Slice,
    out: DevicePtr,
    rows: usize,
    groups: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(32) && k <= STAGED_K,
        "dsv41_fp8_gemv_groups_q: k {k}"
    );
    for first in (0..rows).step_by(TILE) {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 16), groups as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x.offset(first * groups * k * 2))
            .arg_ptr(w.data)
            .arg_ptr(w.scales)
            .arg_ptr(out.offset(first * groups * n * 2))
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32((rows - first).min(TILE) as u32)
            .launch(stream)?;
    }
    Ok(())
}

/// `out` (`[slots, n]` BF16) = the four parts of `parts` added in part order
/// and rounded. A part is `[slots, n]` FP32 and the next starts `stride`
/// values on.
#[allow(clippy::too_many_arguments)]
pub fn round(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    parts: DevicePtr,
    out: DevicePtr,
    stride: usize,
    slots: usize,
    n: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil((slots * n) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(parts)
        .arg_ptr(out)
        .arg_u32(slots as u32)
        .arg_u32(n as u32)
        .arg_u32(stride as u32)
        .launch(stream)
}

/// The MoE's output `y` (`[n]` BF16): every slot of `parts` rounded as
/// [`round`] rounds it, the slots added in FP32 in slot order, and the sum
/// rounded.
#[allow(clippy::too_many_arguments)]
pub fn combine(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    parts: DevicePtr,
    y: DevicePtr,
    stride: usize,
    slots: usize,
    n: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(parts)
        .arg_ptr(y)
        .arg_u32(slots as u32)
        .arg_u32(n as u32)
        .arg_u32(stride as u32)
        .launch(stream)
}

/// `dst` (`[rows, width]` bytes) = columns `first..first + width` of `src`
/// (`[rows, src_stride]` bytes).
#[allow(clippy::too_many_arguments)]
pub fn slice_columns(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    rows: usize,
    src_stride: usize,
    first: usize,
    width: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil((rows * width) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(rows as u32)
        .arg_u32(src_stride as u32)
        .arg_u32(first as u32)
        .arg_u32(width as u32)
        .launch(stream)
}

/// Bring the other ranks' parts of `base` here, and this rank's there. A
/// part starts every `stride` bytes and `bytes` of it are in use. `recv` is
/// scratch of `PARTS * bytes`, for the transport that cannot gather in place.
#[allow(clippy::too_many_arguments)]
pub fn gather(
    gpu: &dyn GpuBackend,
    comm: Option<&dyn CommBackend>,
    held: Held,
    base: DevicePtr,
    recv: DevicePtr,
    stride: usize,
    bytes: usize,
    stream: u64,
) -> Result<()> {
    let Some(comm) = comm.filter(|_| held.count < PARTS) else {
        return Ok(());
    };
    let (mine, stride) = (held.count * bytes, held.count * stride);
    ensure!(
        held.count == 1 || stride == mine,
        "deepseek_v41: a rank that holds {} parts gathers whole parts",
        held.count
    );
    if comm.supports_gather_async(mine) && comm.gather_async(base.0, stride as i64, mine, stream)? {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    comm.all_gather(base.offset(held.first * stride).0, recv.0, mine)?;
    for rank in 0..PARTS / held.count {
        gpu.copy_d2d(recv.offset(rank * mine), base.offset(rank * stride), mine)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "parts_test_gpu.rs"]
mod test_gpu;
