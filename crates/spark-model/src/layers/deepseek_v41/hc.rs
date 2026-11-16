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
    /// A batch's site around a GEMM ([`site_rows`]).
    site_rows: KernelHandle,
    mix_rows: KernelHandle,
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
            site_rows: kernel("dsv41_hc_site_rows")?,
            mix_rows: kernel("dsv41_hc_mix_rows")?,
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
/// `[PARTS, rows, hidden / PARTS]` BF16, a part a quarter of every token's
/// values, with room for `rows` tokens in a part.
#[derive(Clone, Copy)]
pub struct SublayerParts {
    pub parts: DevicePtr,
    pub rows: usize,
}

/// [`expand`] of `tokens` tokens in place from a sublayer's parts.
#[allow(clippy::too_many_arguments)]
pub fn expand_parts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &HcShape,
    x: SublayerParts,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid(shape.grid(tokens))
        .block([THREADS, 1, 1])
        .arg_ptr(x.parts)
        .arg_ptr(streams)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(x.rows as u32)
        .launch(stream)
}

/// The vectors a site leaves for its sublayer, `[tokens, hidden]` BF16 each: the
/// collapsed stream, its norm, and the norm's FP8 form. `partials` is the
/// mix's scratch ([`HcShape::partials`] bytes).
#[derive(Clone, Copy)]
pub struct SiteOut {
    pub collapsed: DevicePtr,
    pub normed: DevicePtr,
    pub quantized: DevicePtr,
    pub partials: DevicePtr,
}

/// A whole site of `tokens` tokens in two launches, with the bytes of the
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
    tokens: usize,
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
        rows: 0,
    });
    KernelLaunch::new(gpu, kernels.site_sums)
        .grid(shape.grid(tokens))
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
        .arg_u32(before.rows as u32)
        .launch(stream)?;
    // The norm and its rounding on one block, the coefficients on another.
    KernelLaunch::new(gpu, kernels.site_finish)
        .grid([2, tokens as u32, 1])
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

/// The rows of a site's mix: `pre`, `post` and `comb`.
pub fn mix_rows(copies: usize) -> usize {
    (2 + copies) * copies
}

/// A site's `fn_` (FP32, as `words` of `[rows, columns]`) as three BF16
/// matrices that add up to it, one after another: BF16 bits of `[3 * rows,
/// columns]`. An FP32 value is its eight leading bits, the next eight and
/// the last, each a BF16 value, so a BF16 GEMM against the three multiplies
/// what the FP32 weights would. `None` if some value is not such a sum (one
/// too near the smallest FP32 can hold).
pub fn split_fn(words: &[f32]) -> Option<Vec<u16>> {
    use atlas_core::numeric::f32_to_bf16;
    let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut out = vec![0u16; 3 * words.len()];
    for (i, &w) in words.iter().enumerate() {
        let mut rest = w;
        let mut sum = 0.0f32;
        for share in 0..3 {
            let bits = f32_to_bf16(rest);
            out[share * words.len() + i] = bits;
            rest -= widen(bits);
            sum += widen(bits);
        }
        if rest != 0.0 || sum != w {
            return None;
        }
    }
    Some(out)
}

/// [`site`] for a batch: the same site with the mix as a GEMM against
/// `split` (the site's `fn_` as [`split_fn`] leaves it, on the device).
/// `sums` is scratch of `[tokens, 3 * mix_rows]` FP32, and `out.partials` of
/// `[tokens, MIX_SUMS]`. The collapse, the norm and its rounding are
/// [`site`]'s value for value; the mix's FP32 sums have another order.
#[allow(clippy::too_many_arguments)]
pub fn site_rows(
    gpu: &dyn GpuBackend,
    kernels: &HcKernels,
    shape: &HcShape,
    weights: &HcSite,
    split: DevicePtr,
    before: Option<SublayerParts>,
    streams: DevicePtr,
    pre_in: DevicePtr,
    pre_out: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    norm_weight: DevicePtr,
    out: SiteOut,
    sums: DevicePtr,
    tokens: usize,
    stream: u64,
) -> Result<()> {
    shape.check()?;
    ensure!(
        shape.hidden.is_multiple_of(64),
        "dsv41_hc_site_rows: hidden {}",
        shape.hidden
    );
    if let Some(before) = before {
        expand_parts(
            gpu,
            kernels.expand_parts,
            shape,
            before,
            streams,
            post,
            comb,
            tokens,
            stream,
        )?;
    }
    let (m, n) = (tokens as u32, 3 * mix_rows(shape.copies) as u32);
    let k = (shape.copies * shape.hidden) as u32;
    crate::layers::ops::bf16_gemm_f32out(streams, split.0, sums, m, n, k, stream)?;
    KernelLaunch::new(gpu, kernels.site_rows)
        .grid([tokens as u32, 1, 1])
        .block([FINISH_THREADS, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(sums)
        .arg_ptr(pre_in)
        .arg_ptr(norm_weight)
        .arg_ptr(out.collapsed)
        .arg_ptr(out.normed)
        .arg_ptr(out.quantized)
        .arg_ptr(out.partials)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_f32(shape.norm_eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.mix_rows)
        .grid([tokens as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(out.partials)
        .arg_ptr(weights.scale)
        .arg_ptr(weights.base)
        .arg_ptr(pre_out)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_u32(shape.hidden as u32)
        .arg_u32(shape.copies as u32)
        .arg_u32(shape.sinkhorn_iters as u32)
        .arg_f32(shape.norm_eps)
        .arg_f32(shape.hc_eps)
        .launch(stream)
}

#[cfg(test)]
mod split_tests {
    use super::split_fn;

    #[test]
    fn an_fp32_weight_is_the_sum_of_its_three_bf16_shares() {
        let widen = |bits: u16| f32::from_bits(u32::from(bits) << 16);
        // Every mantissa, over the exponents a weight has.
        let words: Vec<f32> = (0..4096u32)
            .map(|i| {
                let bits = i.wrapping_mul(0x9e37_79b9);
                f32::from_bits((bits & 0x807f_ffff) | ((100 + i % 30) << 23))
            })
            .chain([0.0, -0.0, 1.0, -1.5, 3.0e-3])
            .collect();
        let shares = split_fn(&words).expect("every value splits");
        for (i, &w) in words.iter().enumerate() {
            let [a, b, c] = [0, 1, 2].map(|s| widen(shares[s * words.len() + i]));
            assert_eq!((a + b) + c, w, "{w:e}");
        }
    }
}

#[cfg(test)]
#[path = "hc_test_gpu.rs"]
mod test_gpu;
