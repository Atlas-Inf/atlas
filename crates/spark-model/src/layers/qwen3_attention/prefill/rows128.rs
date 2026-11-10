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

/// `[cp.async, TMA]` kernels when the lever is set and the target ships
/// them (handle 0 where absent).
fn kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 2] {
    static K: OnceLock<[KernelHandle; 2]> = OnceLock::new();
    *K.get_or_init(|| {
        let on = std::env::var("ATLAS_ATTN_ROWS128").as_deref() == Ok("1");
        [
            "prefill_paged_bf16_rows128",
            "prefill_paged_bf16_rows128_tma",
        ]
        .map(|f| crate::layers::try_kernel_gated(on, gpu, "prefill_paged_rows128", f))
    })
}

/// Whether the lever is set and the target ships the kernel.
pub(super) fn available(gpu: &dyn GpuBackend) -> bool {
    kernels(gpu)[0].0 != 0
}

/// A layer's BF16 KV pools as the TMA kernel reads them: K and V base,
/// `blocks` pages of 16 rows, `block_stride` bytes a page.
pub(super) struct Pools {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub blocks: usize,
    pub block_stride: usize,
}

/// `ops::prefill_attention_paged_batched_64`'s launch, on 128-row tiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    gpu: &dyn GpuBackend,
    pools: Pools,
    [q, output, block_table_ptrs]: [DevicePtr; 3],
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
    let [plain, tma] = kernels(gpu);
    // The TMA kernel's boxes are 16-row pages of [num_kv_heads * 256] BF16 rows.
    let row = num_kv_heads as usize * 256;
    let maps = (tma.0 != 0 && cache_block_size == 16 && pools.block_stride == 16 * row * 2)
        .then(|| tensor_maps(&pools, row))
        .transpose()?
        .flatten();
    let kernel = if maps.is_some() { tma } else { plain };
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, ROWS), batch_size])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(pools.k)
        .arg_ptr(pools.v)
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
        .arg_f32(inv_sqrt_d);
    if let Some([km, vm]) = &maps {
        l = l.arg_tensormap(km).arg_tensormap(vm);
    }
    l.launch(stream)
}

/// The K and V tensor maps (`[blocks * 16] x [row]` BF16, 16 x 64 boxes,
/// 128-byte swizzle); `None` on a build without CUDA tensor maps.
#[cfg(feature = "cuda")]
fn tensor_maps(p: &Pools, row: usize) -> Result<Option<[[u8; 128]; 2]>> {
    use spark_runtime::cuda_backend::tensormap::TensorMap;
    let rows = (p.blocks * 16) as u64;
    let map = |base| TensorMap::tiled_2d_bf16_swizzle128(base, rows, row as u64, row as u64, 16);
    Ok(Some([*map(p.k)?.bytes(), *map(p.v)?.bytes()]))
}
#[cfg(not(feature = "cuda"))]
fn tensor_maps(_: &Pools, _: usize) -> Result<Option<[[u8; 128]; 2]>> {
    Ok(None)
}
