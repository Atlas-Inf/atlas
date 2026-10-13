// SPDX-License-Identifier: AGPL-3.0-only

//! Verify block-table rows: one `[max_blocks]` i32 row per verify row.
//!
//! `max_blocks` is the table's full capacity (32769 at 512K context) while a
//! sequence holds `seq_len / block_size + 1` live blocks, so a zero-padded
//! host image of the rows is almost all padding: 1 MiB allocated, zero-filled
//! and uploaded per 8-row step. The rows are zeroed on the device instead and
//! only each live prefix is uploaded, straight from the sequence's table.
//!
//! The device ends up with the same bytes as the full image left. The fill is
//! unconditional because the rows sit in the shared scratch arena: decode and
//! prefill metadata, the other verify lanes and prefill MoE routing all write
//! inside them between two verify steps, so what an earlier call left there
//! is not known, and the fused chunk moves its owner blocks from call to call.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Write one block-table row per entry of `tables` at `dst`, rows `max_blocks`
/// i32 apart: the table's first `max_blocks` entries, then zeros.
/// Stream-ordered; the tables may change as soon as this returns.
pub(super) fn upload_block_table_rows<'a>(
    gpu: &dyn GpuBackend,
    dst: DevicePtr,
    max_blocks: usize,
    tables: impl ExactSizeIterator<Item = &'a [u32]>,
    stream: u64,
) -> Result<()> {
    let row_bytes = max_blocks * 4;
    gpu.memset_zero_async(dst, tables.len() * row_bytes, stream)?;
    for (row, table) in tables.enumerate() {
        let live = &table[..table.len().min(max_blocks)];
        if live.is_empty() {
            continue;
        }
        // SAFETY: `u32` is POD with the bit pattern of the `i32` the kernels
        // read; the byte view covers exactly `live`.
        let bytes = unsafe {
            std::slice::from_raw_parts(live.as_ptr().cast::<u8>(), std::mem::size_of_val(live))
        };
        gpu.copy_h2d_async(bytes, dst.offset(row * row_bytes), stream)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "block_table_upload_tests.rs"]
mod tests;
