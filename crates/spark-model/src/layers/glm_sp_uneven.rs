// SPDX-License-Identifier: AGPL-3.0-only
//! Reduce-scatter and all-gather over an UNEVEN two-rank row split
//! (`glm_sp::SpRows::split_at`, qwen4_exp sequence-parallel prefill).
//!
//! The pair exchange (`CommBackend::exchange_async`) moves the same byte
//! count each way. With halves of `lo` and `hi` rows, each rank sends a
//! WINDOW of `m = max(lo, hi)` rows that contains the region it means to
//! send: `[0, m)` for the lower region, `[total - m, total)` for the upper
//! one -- both in bounds, and both containing their region. Where the
//! incoming window is exactly the region it carries (the receiving region is
//! the longer one) the peer's rows land in place, added or copied, as on an
//! even split. Otherwise they land in a staging buffer and the region's rows
//! are added (`bf16_add_inplace`, the kernel the pair's own reduce uses:
//! `dst = dst + peer`, so every sum is the bits an all-reduce gives) or
//! copied out of it.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::cell::Cell;

use super::glm_sp::{SpRows, exchange_rows};
use crate::layer::ForwardContext;

/// How one rank runs one uneven exchange, in rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    /// Rows each way.
    pub(super) m: usize,
    /// First row of the window this rank sends.
    pub(super) send0: usize,
    /// The region the peer's rows are for, and its rows.
    pub(super) recv0: usize,
    pub(super) recv_n: usize,
    /// That region's offset inside the peer's window.
    pub(super) skip: usize,
}

/// The windows of one exchange (an even split's are its regions).
pub(super) fn plan(sp: SpRows, add: bool) -> Plan {
    let total = sp.total();
    let m = sp.rows.max(sp.peer_rows);
    let window = |r0: usize| if r0 == 0 { 0 } else { total - m };
    // Reduce-scatter sends this rank's partial of the PEER's rows and
    // receives the peer's partial of its own; all-gather the other way round.
    let (send_region, recv0, recv_n) = if add {
        (sp.peer0, sp.row0, sp.rows)
    } else {
        (sp.row0, sp.peer0, sp.peer_rows)
    };
    Plan {
        m,
        send0: window(send_region),
        recv0,
        recv_n,
        skip: recv0 - window(recv0),
    }
}

/// `[total, width]` BF16 at `ptr`: reduce-scatter (`add`) or all-gather.
pub(super) fn exchange(
    sp: SpRows,
    ptr: DevicePtr,
    width: usize,
    add: bool,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let row = width * 2;
    let p = plan(sp, add);
    let send = ptr.offset(p.send0 * row);
    let dst = ptr.offset(p.recv0 * row);
    if p.recv_n == p.m {
        return exchange_rows(send, dst, p.m, row, add, ctx, stream);
    }
    let stage = staging(ctx.gpu, p.m * row, stream)?;
    exchange_rows(send, stage, p.m, row, false, ctx, stream)?;
    let src = stage.offset(p.skip * row);
    if add {
        let n = (p.recv_n * width) as u32;
        KernelLaunch::new(ctx.gpu, ctx.gpu.kernel("bf16_add", "bf16_add_inplace")?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n)
            .launch(stream)
    } else {
        ctx.gpu.copy_d2d_async(src, dst, p.recv_n * row, stream)
    }
}

/// Give the rank that does not own chunk row 0 the owner's `row_bytes` row 0
/// of `ptr`, at its own row 0: after a split forward both ranks' row 0 holds
/// what an unsplit forward leaves there.
pub fn share_row0(
    sp: SpRows,
    ptr: DevicePtr,
    row_bytes: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let stage = staging(ctx.gpu, row_bytes, stream)?;
    let (send, dst) = if sp.row0 == 0 {
        (ptr, stage)
    } else {
        (stage, ptr)
    };
    exchange_rows(send, dst, 1, row_bytes, false, ctx, stream)
}

thread_local! {
    /// `(ptr, bytes)` of the staging buffer. Per thread: a forward runs on
    /// one thread, and its exchanges on one stream.
    static STAGING: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// One device buffer of at least `bytes`, grown on demand. A grow waits for
/// `stream`, the only stream the exchanges run on, before freeing the old one.
fn staging(gpu: &dyn GpuBackend, bytes: usize, stream: u64) -> Result<DevicePtr> {
    let (ptr, size) = STAGING.with(Cell::get);
    if size >= bytes {
        return Ok(DevicePtr(ptr));
    }
    if ptr != 0 {
        gpu.synchronize(stream)?;
        gpu.free(DevicePtr(ptr))?;
        STAGING.with(|s| s.set((0, 0)));
    }
    let p = gpu.alloc(bytes)?;
    STAGING.with(|s| s.set((p.0, bytes)));
    Ok(p)
}

#[cfg(test)]
#[path = "glm_sp_uneven_tests.rs"]
pub(crate) mod tests;
