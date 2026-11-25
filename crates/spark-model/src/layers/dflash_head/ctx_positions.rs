// SPDX-License-Identifier: AGPL-3.0-only

use super::super::{BlockDiffusionDraftHead, DflashScratch};

impl BlockDiffusionDraftHead {
    /// #58: carve `bytes` from the lane scratch's pinned position-staging
    /// region. Returns the region start pointer, or `None` on overflow —
    /// callers fall back to the synchronous `copy_h2d` in that case.
    ///
    /// The cursor is reset at each public propose entry (`propose_drafts`,
    /// `propose_on_lanes`, `propose_batch`). Reuse across calls is safe
    /// because every propose ends in the step's readback
    /// (`synchronize` + `copy_d2h` of the drafted tokens), which completes
    /// all enqueued `copy_h2d_async_retained` copies before the next call
    /// overwrites the region.
    pub(in crate::layers::dflash_head) fn ctx_positions_region(
        scratch: &DflashScratch,
        bytes: usize,
    ) -> Option<*mut u8> {
        let base = scratch
            .ctx_positions_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        if base.is_null() {
            return None;
        }
        carve_region(
            &scratch.ctx_positions_cursor,
            scratch.ctx_positions_pinned_bytes,
            bytes,
        )
        .map(|off| unsafe { base.add(off) })
    }

    /// Reset this lane scratch's carve cursor — call at each public
    /// propose entry (see `ctx_positions_region` for the lifetime rule).
    pub(in crate::layers::dflash_head) fn ctx_positions_reset(scratch: &DflashScratch) {
        scratch
            .ctx_positions_cursor
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Pure bookkeeping for the pinned carve: returns the offset of a fresh
/// `bytes`-sized region, or `None` when the buffer is exhausted. Atomic so
/// propose lanes sharing a scratch's staging can carve without a lock.
pub(in crate::layers::dflash_head) fn carve_region(
    cursor: &std::sync::atomic::AtomicUsize,
    capacity: usize,
    bytes: usize,
) -> Option<usize> {
    let mut cur = cursor.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        let next = cur.checked_add(bytes)?;
        if next > capacity {
            return None;
        }
        match cursor.compare_exchange_weak(
            cur,
            next,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => return Some(cur),
            Err(c) => cur = c,
        }
    }
}
