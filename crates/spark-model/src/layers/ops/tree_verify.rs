// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrapper for the tree verify's KV slot copies
//! (`kv_slot_copy_bf16`, kernels/gb10/qwen3.6-27b/nvfp4/tree_verify.cu; see
//! `speculative::draft_tree`). Looked up on first use, so targets that never
//! verify a tree issue no lookup.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

/// Whether this kernel set ships `kv_slot_copy_bf16` (the qwen3.6-27b tree).
pub fn kv_slot_copy_available(gpu: &dyn GpuBackend) -> bool {
    kernel(gpu).0 != 0
}

fn kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static K: OnceLock<KernelHandle> = OnceLock::new();
    *K.get_or_init(|| crate::layers::try_kernel(gpu, "tree_verify", "kv_slot_copy_bf16"))
}

/// Copy, in one layer's BF16 K and V pools, the first `*count` of the
/// `max` (src slot, dst slot) int64 pairs at `entries` (`count` in device
/// memory, so a captured graph replays any count up to `max`).
#[allow(clippy::too_many_arguments)]
pub fn kv_slot_copy_bf16(
    gpu: &dyn GpuBackend,
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    entries: DevicePtr,
    count: DevicePtr,
    max: u32,
    n_elems: u32,
    stream: u64,
) -> Result<()> {
    let k = kernel(gpu);
    ensure!(k.0 != 0, "kv_slot_copy_bf16 is not in this kernel set");
    if max == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, k)
        .grid([max, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(k_pool)
        .arg_ptr(v_pool)
        .arg_ptr(entries)
        .arg_ptr(count)
        .arg_u32(n_elems)
        .launch(stream)
}
