// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-scan paged decode attention for a BF16 KV cache
//! (`ATLAS_ATTN_SHARED_SCAN=1`; `attn_shared_scan` and
//! `attn_shared_scan_reduce`, kernels/gb10/qwen3.6-27b/nvfp4/attn_shared_scan.cu).
//!
//! `paged_decode_attn` walks a KV head's whole K and V once per (query head,
//! row). The shared scan reads them once per KV head for every row that
//! attends the same blocks and scores on tensor cores; the kernel's header
//! has the design and why a row's bits do not depend on the rows beside it,
//! which is what `ATLAS_EXACT_VERIFY` needs (`model/exact_verify.rs`).
//!
//! The switch and the shape test are fixed for the life of the process, so
//! serial decode and every verify width take the same arithmetic.

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

/// Superblocks a row (`SS_MAX_SB`).
pub const SHARED_SCAN_MAX_SB: u32 = 32;
/// FP32 words a superblock partial: `m`, `l`, `o[256]` (`SS_REC`).
pub const SHARED_SCAN_REC: usize = 258;
/// Head dim the kernel is built for (`SS_HD`).
pub const SHARED_SCAN_HEAD_DIM: u32 = 256;
/// Rows a scan CTA walks (`SS_ROWS_MAX`).
const WINDOW: u32 = 8;
/// Queries a CTA scores (`SS_QMAX`): the bound on query heads per KV head.
const QMAX: u32 = 48;
/// Rows the first workspace holds; a wider launch gets a larger one.
const WS_MIN_ROWS: usize = 64;

/// `ATLAS_ATTN_SHARED_SCAN=1`. Read once.
pub fn attn_shared_scan_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_ATTN_SHARED_SCAN").as_deref() == Ok("1");
        if on {
            tracing::info!(
                "ATLAS_ATTN_SHARED_SCAN: BF16 paged decode attention reads each KV head once \
                 for the rows that share its blocks"
            );
        }
        on
    })
}

fn kernels(gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
    static K: OnceLock<(KernelHandle, KernelHandle)> = OnceLock::new();
    *K.get_or_init(|| {
        (
            crate::layers::try_kernel(gpu, "attn_shared_scan", "attn_shared_scan"),
            crate::layers::try_kernel(gpu, "attn_shared_scan", "attn_shared_scan_reduce"),
        )
    })
}

/// Whether this kernel set ships the shared scan (the qwen3.6-27b tree).
pub fn attn_shared_scan_available(gpu: &dyn GpuBackend) -> bool {
    let (scan, reduce) = kernels(gpu);
    scan.0 != 0 && reduce.0 != 0
}

/// Shapes the kernel serves. A function of the layer's constants only.
pub fn attn_shared_scan_shape_ok(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> bool {
    head_dim == SHARED_SCAN_HEAD_DIM
        && block_size.is_power_of_two()
        && (16..=256).contains(&block_size)
        && num_kv_heads != 0
        && num_q_heads.is_multiple_of(num_kv_heads)
        && num_q_heads / num_kv_heads <= QMAX
}

/// The switch is on, the kernels are loaded and the shape fits: the BF16
/// decode arm takes the shared scan for every launch of this layer.
pub fn attn_shared_scan_on(
    gpu: &dyn GpuBackend,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
) -> bool {
    attn_shared_scan_enabled()
        && attn_shared_scan_shape_ok(num_q_heads, num_kv_heads, head_dim, block_size)
        && attn_shared_scan_available(gpu)
}

/// Bytes of partials for `rows` rows of `num_q_heads` query heads.
pub fn attn_shared_scan_ws_bytes(rows: usize, num_q_heads: u32) -> usize {
    rows * num_q_heads as usize * SHARED_SCAN_MAX_SB as usize * SHARED_SCAN_REC * 4
}

/// The partials workspace, shared by every layer. Like the model's other
/// decode scratch it relies on the scheduler running one forward at a time:
/// a scan and its reduce are adjacent on their stream and no other attention
/// launch runs between them. A captured graph holds the pointer it was
/// recorded with, so a workspace is never freed or moved: a launch wider than
/// every one so far allocates a larger one beside it.
fn workspace(gpu: &dyn GpuBackend, rows: usize, num_q_heads: u32) -> Result<DevicePtr> {
    static WS: Mutex<Vec<(DevicePtr, usize)>> = Mutex::new(Vec::new());
    let need = attn_shared_scan_ws_bytes(rows, num_q_heads);
    let mut ws = WS.lock();
    if let Some(&(ptr, _)) = ws.iter().find(|&&(_, bytes)| bytes >= need) {
        return Ok(ptr);
    }
    let bytes = attn_shared_scan_ws_bytes(rows.max(WS_MIN_ROWS), num_q_heads);
    let ptr = gpu.alloc(bytes)?;
    tracing::info!(
        "shared-scan attention workspace: {} MiB ({} rows)",
        bytes >> 20,
        rows.max(WS_MIN_ROWS)
    );
    ws.push((ptr, bytes));
    Ok(ptr)
}

/// Paged decode attention over a BF16 NHD cache: one scan launch (partials)
/// and one reduce launch. Arguments as `paged_decode_attn_bf16`, full
/// attention only.
#[allow(clippy::too_many_arguments)]
pub fn attn_shared_scan_bf16(
    gpu: &dyn GpuBackend,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    stream: u64,
) -> Result<()> {
    let (scan, reduce) = kernels(gpu);
    ensure!(
        scan.0 != 0 && reduce.0 != 0,
        "attn_shared_scan is not in this kernel set"
    );
    ensure!(
        attn_shared_scan_shape_ok(num_q_heads, num_kv_heads, head_dim, block_size),
        "attn_shared_scan: unsupported shape (q heads {num_q_heads}, kv heads {num_kv_heads}, \
         head dim {head_dim}, block size {block_size})"
    );
    if num_seqs == 0 {
        return Ok(());
    }
    let ws = workspace(gpu, num_seqs as usize, num_q_heads)?;
    KernelLaunch::new(gpu, scan)
        .grid([
            num_kv_heads * SHARED_SCAN_MAX_SB,
            num_seqs.div_ceil(WINDOW),
            1,
        ])
        .block([96, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(ws)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_seqs)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .launch(stream)?;
    KernelLaunch::new(gpu, reduce)
        .grid([num_q_heads, num_seqs, 1])
        .block([SHARED_SCAN_HEAD_DIM, 1, 1])
        .arg_ptr(ws)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_test_admits_the_27b_attention_and_nothing_it_was_not_built_for() {
        assert!(attn_shared_scan_shape_ok(24, 4, 256, 16));
        assert!(attn_shared_scan_shape_ok(24, 4, 256, 32));
        assert!(!attn_shared_scan_shape_ok(24, 4, 128, 16));
        assert!(!attn_shared_scan_shape_ok(24, 4, 256, 8));
        assert!(!attn_shared_scan_shape_ok(24, 4, 256, 24));
        assert!(!attn_shared_scan_shape_ok(24, 5, 256, 16));
        assert!(!attn_shared_scan_shape_ok(24, 0, 256, 16));
        assert!(!attn_shared_scan_shape_ok(98, 2, 256, 16));
    }

    #[test]
    fn workspace_size_covers_every_partial() {
        assert_eq!(attn_shared_scan_ws_bytes(8, 24), 8 * 24 * 32 * 258 * 4);
    }
}
