// SPDX-License-Identifier: AGPL-3.0-only

//! Checked launch foundation for independent GLM KDA rows; not decode wiring.

use crate::layer::ssm_batch::SsmBatchLayer;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const PLANE: usize = 32 * 128;
const CHANNELS: usize = 3 * PLANE;

/// Geometry selection only: callers retain existing model/independence,
/// absent-view and optional-kernel fallback policy. Present invalid data errors.
#[derive(Clone, Copy, Debug)]
pub struct KdaIndexedShape {
    rows: u32,
}

impl KdaIndexedShape {
    pub fn select(rows: usize, heads: usize, dim: usize, conv_width: usize) -> Option<Self> {
        (matches!(rows, 2..=8) && heads == 32 && dim == 128 && conv_width == 4)
            .then_some(Self { rows: rows as u32 })
    }
    fn grid(self, recurrent: bool) -> [u32; 3] {
        [if recurrent { 32 } else { 48 }, self.rows, 1]
    }
    fn block(self, recurrent: bool) -> [u32; 3] {
        [if recurrent { 128 } else { 256 }, 1, 1]
    }
    fn validate_launch(
        self,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        recurrent: bool,
    ) -> Result<()> {
        ensure!(
            grid == self.grid(recurrent) && block == self.block(recurrent) && shared == 0,
            "indexed KDA launch geometry mismatch"
        );
        Ok(())
    }
    fn pool(self, layer: SsmBatchLayer, recurrent: bool) -> Result<KdaBuffer> {
        ensure!(
            layer.rows() == self.rows && layer.slot_capacity() > 0,
            "indexed KDA state view width/capacity mismatch"
        );
        KdaBuffer {
            ptr: layer.slots(),
            bytes: self.rows as usize * 4,
        }
        .validate(self.rows as usize * 4, 4)?;
        let stride = if recurrent {
            layer.h_stride_elements()
        } else {
            layer.conv_stride_elements()
        };
        let live = if recurrent { PLANE * 128 } else { CHANNELS * 4 };
        ensure!(
            stride >= live as u64,
            "indexed KDA state stride is too small"
        );
        let bytes = stride
            .checked_mul(4)
            .and_then(|n| n.checked_mul(u64::from(layer.slot_capacity())))
            .ok_or_else(|| anyhow::anyhow!("indexed KDA pool byte overflow"))?;
        let pool = KdaBuffer {
            ptr: if recurrent {
                layer.h_base()
            } else {
                layer.conv_base()
            },
            bytes: usize::try_from(bytes)?,
        };
        pool.validate(pool.bytes, 4)?;
        Ok(pool)
    }
}

/// Borrowed byte capacity starting at ptr, not an owning allocation.
#[derive(Clone, Copy, Debug)]
pub struct KdaBuffer {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

impl KdaBuffer {
    fn validate(self, required: usize, alignment: u64) -> Result<()> {
        ensure!(
            !self.ptr.is_null() && self.ptr.0.is_multiple_of(alignment),
            "indexed KDA null/misaligned buffer"
        );
        ensure!(
            self.bytes >= required,
            "indexed KDA buffer capacity is too small"
        );
        self.ptr
            .0
            .checked_add(u64::try_from(self.bytes)?)
            .ok_or_else(|| anyhow::anyhow!("indexed KDA buffer address overflow"))?;
        Ok(())
    }
}

fn row_bytes(rows: usize, stride: usize, width: usize) -> Result<usize> {
    ensure!(
        rows > 0 && stride >= width && u32::try_from(stride).is_ok(),
        "indexed KDA invalid BF16 row stride"
    );
    (rows - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(width))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("indexed KDA row capacity overflow"))
}

#[derive(Clone, Copy)]
pub struct KdaIndexedConv {
    pub input: KdaBuffer,
    pub weight: KdaBuffer,
    pub bias: Option<KdaBuffer>,
    pub output: KdaBuffer,
    /// BF16 elements, not bytes.
    pub input_stride: usize,
    pub output_stride: usize,
}

#[derive(Clone, Copy)]
pub struct KdaIndexedRecurrent {
    pub qkv: KdaBuffer,
    pub gate: KdaBuffer,
    pub beta: KdaBuffer,
    pub a_log: KdaBuffer,
    pub dt_bias: KdaBuffer,
    pub output: KdaBuffer,
    pub lower_bound: f32,
}

// One descriptor representation is used for both ABI tests and submission.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arg {
    Ptr(DevicePtr),
    U32(u32),
    U64(u64),
    F32(f32),
}

fn conv_args(shape: KdaIndexedShape, layer: SsmBatchLayer, x: KdaIndexedConv) -> Result<[Arg; 13]> {
    let pool = shape.pool(layer, false)?;
    x.input
        .validate(row_bytes(shape.rows as usize, x.input_stride, CHANNELS)?, 2)?;
    x.output.validate(
        row_bytes(shape.rows as usize, x.output_stride, CHANNELS)?,
        2,
    )?;
    x.weight.validate(CHANNELS * 4 * 2, 2)?;
    if let Some(bias) = x.bias {
        bias.validate(CHANNELS * 4, 4)?;
    }
    Ok([
        Arg::Ptr(pool.ptr),
        Arg::Ptr(x.input.ptr),
        Arg::Ptr(x.weight.ptr),
        Arg::Ptr(x.bias.map_or(DevicePtr::NULL, |b| b.ptr)),
        Arg::Ptr(x.output.ptr),
        Arg::U32(CHANNELS as u32),
        Arg::U32(4),
        Arg::U32(shape.rows),
        Arg::U32(x.input_stride as u32),
        Arg::U32(x.output_stride as u32),
        Arg::Ptr(layer.slots()),
        Arg::U32(layer.slot_capacity()),
        Arg::U64(layer.conv_stride_elements()),
    ])
}

fn recurrent_args(
    shape: KdaIndexedShape,
    layer: SsmBatchLayer,
    x: KdaIndexedRecurrent,
) -> Result<[Arg; 14]> {
    let pool = shape.pool(layer, true)?;
    let rows = shape.rows as usize;
    x.qkv.validate(rows * CHANNELS * 2, 2)?;
    x.gate.validate(rows * PLANE * 2, 2)?;
    x.beta.validate(rows * 32 * 2, 2)?;
    x.a_log.validate(32 * 4, 4)?;
    x.dt_bias.validate(PLANE * 4, 4)?;
    x.output.validate(rows * PLANE * 2, 2)?;
    ensure!(
        x.lower_bound.is_finite() && x.lower_bound < 0.0,
        "indexed KDA decay bound must be finite and negative"
    );
    Ok([
        Arg::Ptr(x.qkv.ptr),
        Arg::Ptr(x.gate.ptr),
        Arg::Ptr(x.beta.ptr),
        Arg::Ptr(x.a_log.ptr),
        Arg::Ptr(x.dt_bias.ptr),
        Arg::Ptr(pool.ptr),
        Arg::Ptr(x.output.ptr),
        Arg::U32(shape.rows),
        Arg::U32(32),
        Arg::U32(128),
        Arg::F32(x.lower_bound),
        Arg::Ptr(layer.slots()),
        Arg::U32(layer.slot_capacity()),
        Arg::U64(layer.h_stride_elements()),
    ])
}

fn submit(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: KdaIndexedShape,
    recurrent: bool,
    args: &[Arg],
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "indexed KDA kernel is unavailable");
    let grid = shape.grid(recurrent);
    let block = shape.block(recurrent);
    shape.validate_launch(grid, block, 0, recurrent)?;
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .shared_mem(0);
    for arg in args {
        launch = match *arg {
            Arg::Ptr(v) => launch.arg_ptr(v),
            Arg::U32(v) => launch.arg_u32(v),
            Arg::U64(v) => launch.arg_u64(v),
            Arg::F32(v) => launch.arg_f32(v),
        };
    }
    launch.launch(stream)
}

/// Call before either stateful launch. A malformed recurrent span or missing
/// second handle must never be discovered after convolution state advances.
/// The shared shape/layer enforce identical row count, slots, and pool identity.
pub fn validate_kda_indexed_pair(
    shape: KdaIndexedShape,
    layer: SsmBatchLayer,
    conv_kernel: KernelHandle,
    recurrent_kernel: KernelHandle,
    conv: KdaIndexedConv,
    recurrent: KdaIndexedRecurrent,
) -> Result<()> {
    ensure!(
        conv_kernel.0 != 0 && recurrent_kernel.0 != 0,
        "indexed KDA kernel pair is unavailable"
    );
    ensure!(
        conv.output.ptr == recurrent.qkv.ptr && conv.output_stride == CHANNELS,
        "indexed KDA recurrence needs the contiguous convolution output"
    );
    conv_args(shape, layer, conv)?;
    recurrent_args(shape, layer, recurrent)?;
    Ok(())
}

pub fn kda_conv_indexed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: KdaIndexedShape,
    layer: SsmBatchLayer,
    args: KdaIndexedConv,
    stream: u64,
) -> Result<()> {
    submit(
        gpu,
        kernel,
        shape,
        false,
        &conv_args(shape, layer, args)?,
        stream,
    )
}

pub fn kda_recurrent_indexed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: KdaIndexedShape,
    layer: SsmBatchLayer,
    args: KdaIndexedRecurrent,
    stream: u64,
) -> Result<()> {
    submit(
        gpu,
        kernel,
        shape,
        true,
        &recurrent_args(shape, layer, args)?,
        stream,
    )
}

#[cfg(test)]
#[path = "kda_indexed_tests.rs"]
mod tests;
