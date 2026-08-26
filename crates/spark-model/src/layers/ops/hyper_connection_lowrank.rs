// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next low-rank mHC dispatch.
//!
//! Companion to `hyper_connection.rs`, which drives DeepSeek-V4's Sinkhorn
//! mixer. Both families share the `[T, hc_mult, H]` FP32 highway and the same
//! four kernel NAMES — a model shadow overrides the whole
//! `hyper_connection.cu` file, so `qwen3.8-flash-next` resolves
//! `hyper_connection::hc_pre` to the low-rank kernel while
//! `deepseek-v4-flash` resolves it to the Sinkhorn one. The two take
//! DIFFERENT argument lists, which is why the launches live apart.
//!
//! `hc_expand` is byte-identical across both and is not duplicated here.
//!
//! Selection is by WEIGHTS, not by model name: `HcSiteWeights::lowrank`
//! being `Some` is what routes here. A model that somehow carried both would
//! be a load-time bug, not a silent dispatch coin-flip.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;

/// Collapse the `hc_mult` streams to one, and emit the per-stream injection
/// weights the matching [`hc_post_lowrank`] needs.
///
/// `streams [T, hc, H] -> y_out [T, H]`, `inj_out [T, hc]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !w.inject_w.is_null(),
        "hc_pre_lowrank needs block_inject_weight; a site loaded without one \
         is the model-level mixer and must use hc_head_lowrank"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(w.inject_w)
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// The model-level mixer (`use_combine=False`): the same collapse with no
/// injection vector.
///
/// This is also the model's FINAL NORMALIZATION — the checkpoint ships no
/// `model.norm.weight` because `hc_norm` here plays that role.
#[allow(clippy::too_many_arguments)]
pub fn hc_head_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// Inject the block output back into every stream:
/// `out[t, s*H + d] = residual[t, s*H + d] + block_out[t, d] * inj[t, s]`.
///
/// Note there is no `comb` argument: DeepSeek mixes streams with a full
/// `[hc, hc]` combine matrix on the way back, Qwen scales by one scalar per
/// stream. Passing a combine matrix here would not type-check, which is the
/// point of keeping the two launches separate.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    inj: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(inj)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}
