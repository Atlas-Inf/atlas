// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 Hyper-Connections (`dsv41_hc.cu`): the mix a site computes
//! from the stream, the collapse to one vector, and the expansion back.
//!
//! A collapse uses the `pre` of the site before it, not its own: see
//! [`IDENTITY_PRE`] and the kernel file.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// The `pre` layer 0's attention is collapsed with: the first copy alone.
pub const IDENTITY_PRE: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

const THREADS: u32 = 256;
const MAX_COPIES: usize = 4;

#[derive(Clone, Copy)]
pub struct HcKernels {
    pub mix: KernelHandle,
    pub collapse: KernelHandle,
    pub expand: KernelHandle,
}

impl HcKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            mix: gpu.kernel("dsv41_hc", "dsv41_hc_mix")?,
            collapse: gpu.kernel("dsv41_hc", "dsv41_hc_collapse")?,
            expand: gpu.kernel("dsv41_hc", "dsv41_hc_expand")?,
        })
    }
}

/// One site's learned mixing weights, all FP32: `fn_` `[(2 + hc) * hc, hc *
/// hidden]`, `scale` `[3]`, `base` `[(2 + hc) * hc]`.
#[derive(Clone, Copy)]
pub struct HcSite {
    pub fn_: DevicePtr,
    pub scale: DevicePtr,
    pub base: DevicePtr,
}

/// The stream's geometry and the two constants of the mix.
#[derive(Clone, Copy)]
pub struct HcShape {
    pub hidden: usize,
    pub copies: usize,
    pub sinkhorn_iters: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
}

/// A site's coefficients from `streams` (`[tokens, copies, hidden]` BF16):
/// `pre` and `post` `[tokens, copies]`, `comb` `[tokens, copies, copies]`,
/// FP32.
#[allow(clippy::too_many_arguments)]
pub fn mix(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &HcShape,
    site: &HcSite,
    streams: DevicePtr,
    pre: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        shape.copies <= MAX_COPIES,
        "dsv41_hc: {} copies",
        shape.copies
    );
    KernelLaunch::new(gpu, kernel)
        .grid([tokens as u32, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(site.fn_)
        .arg_ptr(site.scale)
        .arg_ptr(site.base)
        .arg_ptr(pre)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(shape.sinkhorn_iters as u32)
        .arg_f32(shape.norm_eps)
        .arg_f32(shape.hc_eps)
        .launch(stream)
}

/// `y` (`[tokens, hidden]` BF16) = the copies of `streams` weighted by `pre`.
pub fn collapse(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &HcShape,
    streams: DevicePtr,
    pre: DevicePtr,
    y: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens as u32, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(pre)
        .arg_ptr(y)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .launch(stream)
}

/// `out` = a sublayer's output `x` (`[tokens, hidden]` BF16) spread over the
/// copies by `post`, plus `residual` mixed through `comb`. `out` may be
/// `residual`.
#[allow(clippy::too_many_arguments)]
pub fn expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &HcShape,
    x: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens as u32, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "hc_test_gpu.rs"]
mod test_gpu;
