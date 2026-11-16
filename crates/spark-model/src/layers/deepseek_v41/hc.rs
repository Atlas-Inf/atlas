// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 Hyper-Connections (`dsv41_hc.cu`): the mix a site computes
//! from the stream, the collapse to one vector, and the expansion back; and
//! [`site`], which is all of a site in two launches.
//!
//! A collapse uses the `pre` of the site before it, not its own: see
//! [`IDENTITY_PRE`] and the kernel file.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// The `pre` layer 0's attention is collapsed with: the first copy alone.
pub const IDENTITY_PRE: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// A block is this many values of a token, in all its copies.
const THREADS: u32 = 256;
/// `dsv41_hc_site_finish` runs a token on one block this wide.
const FINISH_THREADS: u32 = 1024;
const MAX_COPIES: usize = 4;
/// The sums a block leaves the mix: its rows, and the sum of squares.
const MIX_SUMS: usize = (2 + MAX_COPIES) * MAX_COPIES + 1;

#[derive(Clone, Copy)]
pub struct HcKernels {
    mix_sums: KernelHandle,
    mix_finish: KernelHandle,
    pub collapse: KernelHandle,
    pub expand: KernelHandle,
    pub expand_parts: KernelHandle,
    site_sums: KernelHandle,
    site_finish: KernelHandle,
}

impl HcKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = |name: &str| gpu.kernel("dsv41_hc", name);
        Ok(Self {
            mix_sums: kernel("dsv41_hc_mix_sums")?,
            mix_finish: kernel("dsv41_hc_mix_finish")?,
            collapse: kernel("dsv41_hc_collapse")?,
            expand: kernel("dsv41_hc_expand")?,
            expand_parts: kernel("dsv41_hc_expand_parts")?,
            site_sums: kernel("dsv41_hc_site_sums")?,
            site_finish: kernel("dsv41_hc_site_finish")?,
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

impl HcShape {
    /// The blocks of one token, and the grid of `tokens`.
    fn blocks(&self) -> u32 {
        self.hidden as u32 / THREADS
    }

    fn grid(&self, tokens: usize) -> [u32; 3] {
        [self.blocks(), tokens as u32, 1]
    }

    /// Bytes of the scratch the mix of `tokens` tokens leaves its blocks'
    /// partial sums in.
    pub fn partials(&self, tokens: usize) -> usize {
        tokens * self.blocks() as usize * MIX_SUMS * 4
    }

    fn check(&self) -> Result<()> {
        ensure!(
            self.copies <= MAX_COPIES && self.hidden.is_multiple_of(THREADS as usize),
            "dsv41_hc: {} copies of {}",
            self.copies,
            self.hidden
        );
        Ok(())
    }
}

/// A site's coefficients from `streams` (`[tokens, copies, hidden]` BF16):
/// `pre` and `post` `[tokens, copies]`, `comb` `[tokens, copies, copies]`,
/// FP32. Two launches; `partials` is scratch of [`HcShape::partials`] bytes.
#[allow(clippy::too_many_arguments)]
pub fn mix(
    gpu: &dyn GpuBackend,
    kernels: &HcKernels,
    shape: &HcShape,
    site: &HcSite,
    streams: DevicePtr,
    partials: DevicePtr,
    pre: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    shape.check()?;
    KernelLaunch::new(gpu, kernels.mix_sums)
        .grid(shape.grid(tokens))
        .block([THREADS, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(site.fn_)
        .arg_ptr(partials)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.mix_finish)
        .grid([tokens as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(partials)
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
        .grid(shape.grid(tokens))
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
        .grid(shape.grid(tokens))
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

/// A sublayer's output as the four parts it was computed in (`parts.rs`):
/// `[PARTS, slots, hidden]` FP32, a slot per expert of the MoE and one for
/// the attention.
#[derive(Clone, Copy)]
pub struct SublayerParts {
    pub parts: DevicePtr,
    pub slots: usize,
}

/// [`expand`] of one token in place from a sublayer's parts, combined and
/// rounded in the same launch.
#[allow(clippy::too_many_arguments)]
pub fn expand_parts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &HcShape,
    x: SublayerParts,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid(shape.grid(1))
        .block([THREADS, 1, 1])
        .arg_ptr(x.parts)
        .arg_ptr(streams)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(x.slots as u32)
        .launch(stream)
}

/// The vectors a site leaves for its sublayer, `[hidden]` BF16 each: the
/// collapsed stream, its norm, and the norm's FP8 form. `partials` is the
/// mix's scratch ([`HcShape::partials`] bytes).
#[derive(Clone, Copy)]
pub struct SiteOut {
    pub collapsed: DevicePtr,
    pub normed: DevicePtr,
    pub quantized: DevicePtr,
    pub partials: DevicePtr,
}

/// A whole site of one token in two launches, with the bytes of the
/// launches they stand for: [`expand_parts`] of the sublayer before
/// (`before`, with the `post` and `comb` the site before left), then [`mix`]
/// (`pre_out`, and `post` and `comb` replaced), [`collapse`] with `pre_in`,
/// the RMS norm under `norm_weight` and its FP8 rounding.
#[allow(clippy::too_many_arguments)]
pub fn site(
    gpu: &dyn GpuBackend,
    kernels: &HcKernels,
    shape: &HcShape,
    weights: &HcSite,
    before: Option<SublayerParts>,
    streams: DevicePtr,
    pre_in: DevicePtr,
    pre_out: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    norm_weight: DevicePtr,
    out: SiteOut,
    stream: u64,
) -> Result<()> {
    shape.check()?;
    ensure!(
        shape.hidden.is_multiple_of(64),
        "dsv41_hc_site: hidden {}",
        shape.hidden
    );
    let before = before.unwrap_or(SublayerParts {
        parts: DevicePtr::NULL,
        slots: 0,
    });
    KernelLaunch::new(gpu, kernels.site_sums)
        .grid(shape.grid(1))
        .block([THREADS, 1, 1])
        .arg_ptr(before.parts)
        .arg_ptr(streams)
        .arg_ptr(weights.fn_)
        .arg_ptr(pre_in)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out.collapsed)
        .arg_ptr(out.partials)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(before.slots as u32)
        .launch(stream)?;
    // The norm and its rounding on one block, the coefficients on another.
    KernelLaunch::new(gpu, kernels.site_finish)
        .grid([2, 1, 1])
        .block([FINISH_THREADS, 1, 1])
        .arg_ptr(out.partials)
        .arg_ptr(weights.scale)
        .arg_ptr(weights.base)
        .arg_ptr(pre_out)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(norm_weight)
        .arg_ptr(out.collapsed)
        .arg_ptr(out.normed)
        .arg_ptr(out.quantized)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(shape.sinkhorn_iters as u32)
        .arg_f32(shape.norm_eps)
        .arg_f32(shape.hc_eps)
        .launch(stream)
}

#[cfg(test)]
#[path = "hc_test_gpu.rs"]
mod test_gpu;
