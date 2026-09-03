// SPDX-License-Identifier: AGPL-3.0-only

//! `qwen4_exp` block ops: the hyper-connection collapse and the PLE tower.
//!
//! Both are novel to this family — Atlas has nothing to reuse for either — and
//! both are checked against a CPU oracle that is itself checked against
//! HuggingFace at real weights (`atlas_core::qwen4exp_reference`, 1.6e-7 and
//! 5.1e-7). The microtest that does the checking is
//! `examples/qwen4exp_grouped_norm_microtest.rs`.
//!
//! The residual stream these operate on is `hc_count * hidden` wide for the
//! whole trunk. A block collapses it to `hidden`, computes, and the result is
//! scattered back scaled per stream; the model is never `hidden`-wide except
//! inside a block.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_map::DenseWeight;

/// Kernel handles the `qwen4_exp` blocks need, resolved once at layer build.
#[derive(Clone, Copy)]
pub struct Qwen4ExpKernels {
    pub gemv: KernelHandle,
    pub rms_norm_grouped: KernelHandle,
    pub hc_expand: KernelHandle,
    pub hc_lowrank_act: KernelHandle,
    pub hc_stream_mix: KernelHandle,
    pub hc_injection: KernelHandle,
    pub hc_scatter_add: KernelHandle,
    pub ple_gate: KernelHandle,
    pub ple_conv_add: KernelHandle,
    /// The 12 full-attention layers' decode step: causal softmax over a
    /// contiguous K/V buffer, then the per-head `[query | gate]` sigmoid gate.
    pub attn_decode: KernelHandle,
}

impl Qwen4ExpKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            rms_norm_grouped: gpu.kernel("norm", "rms_norm_grouped")?,
            // `qwen4exp_hc`, NOT `hyper_connection` — that module belongs to
            // DeepSeek-V4's Sinkhorn mHC, which is a different formulation.
            hc_expand: gpu.kernel("qwen4exp_hc", "q4e_hc_expand")?,
            hc_lowrank_act: gpu.kernel("qwen4exp_hc", "q4e_hc_lowrank_act")?,
            hc_stream_mix: gpu.kernel("qwen4exp_hc", "q4e_hc_stream_mix")?,
            hc_injection: gpu.kernel("qwen4exp_hc", "q4e_hc_injection")?,
            hc_scatter_add: gpu.kernel("qwen4exp_hc", "q4e_hc_scatter_add")?,
            ple_gate: gpu.kernel("qwen4exp_ple", "q4e_ple_gate")?,
            ple_conv_add: gpu.kernel("qwen4exp_ple", "q4e_ple_conv_add")?,
            attn_decode: gpu.kernel("qwen4exp_attn", "q4e_attn_decode")?,
        })
    }
}

/// Tile one hidden state across all `hc_count` residual streams — the trunk
/// entry, run once after the embedding lookup rather than per layer.
///
/// The streams start IDENTICAL and diverge only once the first block's
/// injection lands. Zero-initialising them instead makes the first collapse
/// read a zero mean, and the model does not recover from it.
pub fn hc_expand(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: usize,
    hidden_size: usize,
    hc_count: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([num_tokens as u32, hc_count as u32, 1])
        .block([hidden_size.min(1024) as u32, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size as u32)
        .arg_u32(hc_count as u32)
        .launch(stream)
}

/// Geometry shared by both blocks.
#[derive(Clone, Copy)]
pub struct Qwen4ExpDims {
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub eps: f32,
}

impl Qwen4ExpDims {
    pub fn wide(&self) -> usize {
        self.hidden * self.hc_count
    }
}

/// Grouped RMS norm: each of `hc_count` streams normalises over its own
/// `hidden` slice, then the row is scaled by the offset-from-1 weight.
pub fn rms_norm_grouped(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    dims: &Qwen4ExpDims,
    num_tokens: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([num_tokens as u32, dims.hc_count as u32, 1])
        .block([dims.hidden.min(1024) as u32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(out)
        .arg_u32(dims.hidden as u32)
        .arg_u32(dims.hc_count as u32)
        .arg_f32(dims.eps)
        .launch(stream)
}

/// Weights of one hyper-connection block.
pub struct HyperConnectionWeights<'a> {
    pub hc_norm: DevicePtr,
    pub mix_down: &'a DenseWeight,
    pub mix_up: &'a DenseWeight,
    /// Absent on the trunk and MTP mixers, which mix without injecting.
    pub block_inject: Option<&'a DenseWeight>,
}

/// Scratch the collapse needs. Sized `wide`, `lowrank`, `hidden`, `hc_count`.
pub struct HyperConnectionScratch {
    pub normed: DevicePtr,
    pub lowrank: DevicePtr,
    pub gate: DevicePtr,
    pub mixed: DevicePtr,
    pub raw_injection: DevicePtr,
    pub injection: DevicePtr,
}

/// Collapse the residual to `hidden` and produce the per-stream injection
/// gains, for ONE token.
///
/// Leaves `scratch.mixed` holding the block input and `scratch.injection` the
/// gains; the caller runs the block and then [`scatter_add`]s the result onto
/// the un-normalised residual.
pub fn hyper_connection_collapse(
    gpu: &dyn GpuBackend,
    k: &Qwen4ExpKernels,
    dims: &Qwen4ExpDims,
    w: &HyperConnectionWeights<'_>,
    residual: DevicePtr,
    scratch: &HyperConnectionScratch,
    stream: u64,
) -> Result<()> {
    let wide = dims.wide();
    rms_norm_grouped(
        gpu,
        k.rms_norm_grouped,
        residual,
        w.hc_norm,
        scratch.normed,
        dims,
        1,
        stream,
    )?;

    super::dense_gemv(
        gpu,
        k.gemv,
        scratch.normed,
        w.mix_down,
        scratch.lowrank,
        dims.hc_lowrank as u32,
        wide as u32,
        stream,
    )?;
    // silu(x / hc_count) — divided BEFORE the activation.
    KernelLaunch::new(gpu, k.hc_lowrank_act)
        .grid([1, 1, 1])
        .block([dims.hc_lowrank.min(1024) as u32, 1, 1])
        .arg_ptr(scratch.lowrank)
        .arg_u32(dims.hc_lowrank as u32)
        .arg_u32(dims.hc_count as u32)
        .launch(stream)?;
    super::dense_gemv(
        gpu,
        k.gemv,
        scratch.lowrank,
        w.mix_up,
        scratch.gate,
        wide as u32,
        dims.hc_lowrank as u32,
        stream,
    )?;

    // MEAN across streams, not a sum.
    KernelLaunch::new(gpu, k.hc_stream_mix)
        .grid([1, 1, 1])
        .block([dims.hidden.min(1024) as u32, 1, 1])
        .arg_ptr(scratch.gate)
        .arg_ptr(scratch.normed)
        .arg_ptr(scratch.mixed)
        .arg_u32(dims.hidden as u32)
        .arg_u32(dims.hc_count as u32)
        .launch(stream)?;

    if let Some(inject) = w.block_inject {
        super::dense_gemv(
            gpu,
            k.gemv,
            scratch.normed,
            inject,
            scratch.raw_injection,
            dims.hc_count as u32,
            wide as u32,
            stream,
        )?;
        // 2 * sigmoid(x / hc_count) — centred on 1.
        KernelLaunch::new(gpu, k.hc_injection)
            .grid([1, 1, 1])
            .block([dims.hc_count as u32, 1, 1])
            .arg_ptr(scratch.raw_injection)
            .arg_ptr(scratch.injection)
            .arg_u32(dims.hc_count as u32)
            .launch(stream)?;
    }
    Ok(())
}

/// Scatter a block's `hidden`-wide output back onto the residual, scaled per
/// stream. Accumulates onto the UN-normalised residual — the normalised copy
/// exists only to compute the gate.
pub fn scatter_add(
    gpu: &dyn GpuBackend,
    k: &Qwen4ExpKernels,
    dims: &Qwen4ExpDims,
    block_out: DevicePtr,
    injection: DevicePtr,
    residual: DevicePtr,
    num_tokens: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.hc_scatter_add)
        .grid([num_tokens as u32, dims.hc_count as u32, 1])
        .block([dims.hidden.min(1024) as u32, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(injection)
        .arg_ptr(residual)
        .arg_u32(dims.hidden as u32)
        .arg_u32(dims.hc_count as u32)
        .launch(stream)
}
