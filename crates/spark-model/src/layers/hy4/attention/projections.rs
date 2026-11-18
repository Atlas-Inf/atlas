// SPDX-License-Identifier: AGPL-3.0-only

use super::memory::Owned;
use anyhow::Result;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};
use std::collections::HashMap;

pub(super) type Scales = HashMap<DevicePtr, DevicePtr>;

pub(super) fn retain(
    memory: &mut Owned<'_>,
    source: DevicePtr,
    n: usize,
    k: usize,
    fp8: bool,
    scales: &mut Scales,
) -> Result<DevicePtr> {
    let stream = memory.gpu.default_stream();
    let result = memory.alloc(n * k * if fp8 { 1 } else { 2 })?;
    if fp8 {
        let scale = memory.alloc(n * 4)?;
        KernelLaunch::new(
            memory.gpu,
            memory.gpu.kernel("hy4_mla", "quant_rowwise_fp8")?,
        )
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(source)
        .arg_ptr(result)
        .arg_ptr(scale)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
        scales.insert(result, scale);
    } else {
        memory
            .gpu
            .copy_d2d_async(source, result, n * k * 2, stream)?;
    }
    memory.gpu.synchronize(stream)?;
    Ok(result)
}

pub(super) fn upload(
    memory: &mut Owned<'_>,
    bytes: &[u8],
    n: usize,
    k: usize,
    fp8: bool,
    scales: &mut Scales,
) -> Result<DevicePtr> {
    let source = memory.upload(bytes)?;
    if !fp8 {
        return Ok(source);
    }
    let result = retain(memory, source, n, k, true, scales)?;
    memory.free(source)?;
    Ok(result)
}
