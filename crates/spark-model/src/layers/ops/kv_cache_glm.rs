// SPDX-License-Identifier: AGPL-3.0-only

//! GLM MLA ops: exact-row MLA GEMV and `fp8_g128` latent-cache kernels.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Exact-row MLA GEMV. The selected M=2/3/5 kernel shares weights across rows.
#[allow(clippy::too_many_arguments)]
pub fn mla_batched_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    n_out: u32,
    k: u32,
    num_heads: u32,
    input_head_stride: u32,
    output_head_stride: u32,
    input_row_stride: u32,
    output_row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 8), num_heads, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_u32(input_head_stride)
        .arg_u32(output_head_stride)
        .arg_u32(input_row_stride)
        .arg_u32(output_row_stride)
        .launch(stream)
}

/// FP8 fake-quant of cached GLM NoPE-512 latents, one CTA per token.
pub fn glm_latent_qdq_fp8g128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(cache)
        .arg_ptr(slot_mapping)
        .launch(stream)
}

/// Write GLM NoPE-512 latents into an `fp8_g128` paged cache, one CTA per
/// token (see `glm_latent_cache_write_fp8g128`).
#[allow(clippy::too_many_arguments)]
pub fn glm_latent_cache_write_fp8g128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    block_size: u32,
    key_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(key)
        .arg_ptr(cache)
        .arg_ptr(slot_mapping)
        .arg_u32(block_size)
        .arg_u32(key_stride)
        .launch(stream)
}

/// Token capacity of the BF16 view an `fp8_g128` GLM owner is dequantized
/// into for the BF16 dense (<=2048) and native sparse (<=32768) prefill.
pub const GLM_LATENT_BF16_VIEW_TOKENS: usize = 32768;

/// Dequantize an owner's `fp8_g128` latents for logical tokens `[0, tokens)`
/// into contiguous BF16 rows `[tokens, 512]`.
#[allow(clippy::too_many_arguments)]
pub fn glm_latent_dequant_fp8g128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    cache: DevicePtr,
    block_table: DevicePtr,
    out: DevicePtr,
    tokens: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens, 1, 1])
        .block([64, 1, 1])
        .arg_ptr(cache)
        .arg_ptr(block_table)
        .arg_ptr(out)
        .arg_u32(tokens)
        .arg_u32(block_size)
        .launch(stream)
}
