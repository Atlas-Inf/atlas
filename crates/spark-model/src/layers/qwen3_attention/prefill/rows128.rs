// SPDX-License-Identifier: AGPL-3.0-only

//! Batched paged prefill attention on 128-row tiles (`prefill_paged_rows128.cu`,
//! lever `ATLAS_ATTN_ROWS128=1`): the BF16-KV, HDIM-256 replacement of
//! `inferspark_prefill_paged_batched_64`, same arguments and bit-identical
//! output, FlashAttention-2 layout (every warp owns its rows end to end, K
//! and V tiles shared by 128 rows).
//!
//! The handle is resolved on first use and only with the lever set (only the
//! 27B dense targets ship the module), so other boot audits are unchanged.
//! It is process-wide: the lever is read once and a process drives one GPU
//! backend.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Query rows a CTA (`rows128::BR` in the kernel).
const ROWS: u32 = 128;

/// The kernel when the lever is set and the target ships it.
pub(super) fn kernel(gpu: &dyn GpuBackend) -> Option<KernelHandle> {
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        let on = std::env::var("ATLAS_ATTN_ROWS128").as_deref() == Ok("1");
        crate::layers::try_kernel_gated(
            on,
            gpu,
            "prefill_paged_rows128",
            "prefill_paged_bf16_rows128",
        )
    });
    (k.0 != 0).then_some(k)
}

/// `ops::prefill_attention_paged_batched_64`'s launch, on 128-row tiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    [q, k_cache, v_cache, output, block_table_ptrs]: [DevicePtr; 5],
    batch_size: u32,
    [cu_seqlens, kv_lens, state_end]: [DevicePtr; 3],
    [q_len, kv_len, q_offset]: [u32; 3],
    [
        num_q_heads,
        num_kv_heads,
        head_dim,
        cache_block_size,
        sliding_window,
    ]: [u32; 5],
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        head_dim == 256,
        "prefill_paged_rows128: head_dim {head_dim} != 256"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, ROWS), batch_size])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_ptr(state_end)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
