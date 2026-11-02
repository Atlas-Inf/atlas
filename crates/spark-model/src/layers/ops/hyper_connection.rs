// SPDX-License-Identifier: AGPL-3.0-only

//! Manifold-Constrained Hyper-Connections (mHC) kernel dispatch (DeepSeek-V4).
//!
//! Wraps the `hyper_connection` module kernels (`hc_pre`, `hc_post`,
//! `hc_head`). The hidden state is stored BF16 as `[T, hc_mult, H]`
//! (stream-major per token). HC parameters are float32 device buffers.

use anyhow::Result;

/// GLM keeps its mHC highway BF16 (`ATLAS_GLM_HC_BF16=1`, the storage vLLM
/// serves) on the DFlash lane; every other model and lane stays FP32.
pub fn hc_bf16_for(model_type: &str) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    model_type == "glm5_next"
        && *ON.get_or_init(|| std::env::var("ATLAS_GLM_HC_BF16").as_deref() == Ok("1"))
        && crate::speculative::glm_repair_policy::dflash_enabled()
}

/// Bytes per highway element for `model_type`.
pub fn hc_elem_bytes(model_type: &str) -> usize {
    if hc_bf16_for(model_type) { 2 } else { 4 }
}

/// The `hyper_connection` / `glm_hc_prefill_vec` kernel serving this model's
/// highway storage (BF16 twins carry a `_bf16` suffix).
pub fn hc_kernel_name(model_type: &str, base: &str) -> String {
    if hc_bf16_for(model_type) {
        format!("{base}_bf16")
    } else {
        base.to_string()
    }
}
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Broadcast a single hidden state into `hc_mult` identical streams:
/// `streams[t, i, d] = hidden[t, d]`. One block per token.
pub fn hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Average the final FP32 HC streams into the BF16 model hidden state.
pub fn hc_contract(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    hidden: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hidden)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Collapse `hc_mult` streams to one (RMS-rescaled mix → sigmoid `pre`
/// weighted sum) and emit `post` / `comb` (Sinkhorn) for the matching
/// `hc_post`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Largest batch routed through [`hc_pre_split`]; beyond it `hc_pre`'s one
/// block per token already fills the GPU (prefill uses the TF32 GEMM path).
pub const HC_PRE_SPLIT_MAX_TOKENS: u32 = 64;

/// `hc_pre` as two launches with bit-identical results: `hc_pre_mix` spreads
/// the 24 mix dot products of every token over the grid, `hc_pre_from_raw_mix`
/// finalizes. `raw_mix` needs `tokens * (2 + hc) * hc` FP32 elements.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_split(
    gpu: &dyn GpuBackend,
    mix_kernel: KernelHandle,
    finalize_kernel: KernelHandle,
    raw_mix: DevicePtr,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, mix_kernel)
        .grid([(2 + hc_mult) * hc_mult, num_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(raw_mix)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)?;
    hc_pre_from_raw_mix(
        gpu,
        finalize_kernel,
        streams,
        raw_mix,
        hc_scale,
        hc_base,
        y_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        stream,
    )
}

/// Decode/verify `hc_pre` through [`hc_pre_split`] when it applies
/// (`ATLAS_GLM_HC_SPLIT`, default on; kernels present; `tokens` within
/// [`HC_PRE_SPLIT_MAX_TOKENS`]; `scratch` large enough). Returns false when the
/// caller must launch the one-kernel `hc_pre` itself.
///
/// `ATLAS_GLM_HC_SPLIT_CHECK=1` also runs `hc_pre` and requires byte-identical
/// `y`/`post`/`comb` (synchronizing; diagnostic only).
#[allow(clippy::too_many_arguments)]
pub fn try_hc_pre_split(
    gpu: &dyn GpuBackend,
    pre_kernel: KernelHandle,
    mix_kernel: KernelHandle,
    finalize_kernel: KernelHandle,
    scratch: DevicePtr,
    scratch_bytes: usize,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<bool> {
    static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    let (enabled, check) = *MODE.get_or_init(|| {
        (
            std::env::var("ATLAS_GLM_HC_SPLIT").as_deref() != Ok("0"),
            std::env::var("ATLAS_GLM_HC_SPLIT_CHECK").as_deref() == Ok("1"),
        )
    });
    let t = num_tokens as usize;
    let (h, hc) = (hidden_size as usize, hc_mult as usize);
    let raw_bytes = t * (2 + hc) * hc * 4;
    let (y_bytes, post_bytes, comb_bytes) = (t * h * 2, t * hc * 4, t * hc * hc * 4);
    let needed = raw_bytes
        + if check {
            y_bytes + post_bytes + comb_bytes
        } else {
            0
        };
    if !enabled
        || mix_kernel.0 == 0
        || finalize_kernel.0 == 0
        || num_tokens == 0
        || num_tokens > HC_PRE_SPLIT_MAX_TOKENS
        || scratch.is_null()
        || scratch_bytes < needed
    {
        return Ok(false);
    }
    hc_pre_split(
        gpu,
        mix_kernel,
        finalize_kernel,
        scratch,
        streams,
        hc_fn,
        hc_scale,
        hc_base,
        y_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        stream,
    )?;
    if check {
        let saved = scratch.offset(raw_bytes);
        let spans = [
            (y_out, saved, y_bytes),
            (post_out, saved.offset(y_bytes), post_bytes),
            (comb_out, saved.offset(y_bytes + post_bytes), comb_bytes),
        ];
        for &(src, dst, bytes) in &spans {
            gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        hc_pre(
            gpu,
            pre_kernel,
            streams,
            hc_fn,
            hc_scale,
            hc_base,
            y_out,
            post_out,
            comb_out,
            num_tokens,
            hidden_size,
            hc_mult,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            stream,
        )?;
        gpu.synchronize(stream)?;
        for (name, &(reference, split, bytes)) in ["y", "post", "comb"].iter().zip(&spans) {
            let (mut a, mut b) = (vec![0u8; bytes], vec![0u8; bytes]);
            gpu.copy_d2h(reference, &mut a)?;
            gpu.copy_d2h(split, &mut b)?;
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            anyhow::ensure!(
                diff == 0,
                "hc_pre split differs from hc_pre in {diff}/{bytes} {name} bytes (T={num_tokens})"
            );
        }
        tracing::debug!("hc_pre split byte-identical (T={num_tokens})");
    }
    Ok(true)
}

/// Finalize `hc_pre` from an FP32 `[tokens, (2 + hc) * hc]` pre-mix GEMM.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_from_raw_mix(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    raw_mix: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(raw_mix)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Expand the sublayer output back into `hc_mult` streams, mixing the saved
/// residual streams through the doubly-stochastic `comb`. `out` may alias
/// `residual`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    let ptrs = [block_out, DevicePtr::NULL, residual, post, comb, out];
    if super::glm_decode_fuse::hc_post(
        gpu,
        kernel,
        ptrs,
        [num_tokens, hidden_size, hc_mult],
        stream,
    )? {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// K=5 two-rank KDA seam: add the local and peer BF16 projections, then feed
/// the exact rounded result into mHC post-mixing without materialising the
/// reduced BF16 buffer. The kernel uses the same `__hadd` operation as Atlas's
/// existing two-rank all-reduce fast path.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_bf16_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    local_block_out: DevicePtr,
    peer_block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    let ptrs = [local_block_out, peer_block_out, residual, post, comb, out];
    if super::glm_decode_fuse::hc_post(
        gpu,
        kernel,
        ptrs,
        [num_tokens, hidden_size, hc_mult],
        stream,
    )? {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(local_block_out)
        .arg_ptr(peer_block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Fuse GLM's exact BF16 shared-expert blend into the immediately following
/// mHC post-step. The kernel retains the blend's explicit BF16 rounding before
/// feeding the value into the FP32 hyperconnection highway.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_moe_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    routed: DevicePtr,
    shared: DevicePtr,
    normed: DevicePtr,
    gate_weight: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(routed)
        .arg_ptr(shared)
        .arg_ptr(normed)
        .arg_ptr(gate_weight)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Final collapse before the LM head: a single learned sigmoid-weighted sum
/// over the `hc_mult` streams. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head_fn: DevicePtr,
    head_scale: DevicePtr,
    head_base: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(head_fn)
        .arg_ptr(head_scale)
        .arg_ptr(head_base)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}
