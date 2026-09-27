// SPDX-License-Identifier: AGPL-3.0-only

//! Fault-queue machinery for the n-gram row cache, split out of
//! `ngram_cache.rs` to keep it under the 500-LoC cap: the QD-tuned
//! worker count, per-fault bookkeeping, and the row fetch itself.

use super::*;

/// Queue depth for the fault pass. The misses of one prefill are independent
/// 4 KiB O_DIRECT reads, and O_DIRECT means no page cache and no kernel
/// readahead -- the device only overlaps what we hand it at once. Measured on
/// one prefill of 4656 tokens: 22,462 misses x ~74us ISSUED SERIALLY = 1657 ms,
/// 17% of that request's TTFT. Raising the cache does not help; the misses are
/// compulsory first-touch (65536 -> 1048576 slots changed the miss count by
/// zero), so the depth is the whole lever.
///
/// Measured on that prefill, resolve time by depth -- monotone, so the default
/// is the deepest measured rather than the knee:
///
/// ```text
///     QD    resolve    vs serial
///      1    1631 ms      1.00x
///      8     424 ms      3.85x
///     16     257 ms      6.35x
///     32     171 ms      9.55x
/// ```
///
/// `1` restores the old strictly-serial behaviour for a bisect.
pub(super) fn fault_threads() -> usize {
    std::env::var("ATLAS_PLE_FAULT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32)
}

/// One scheduled miss: read row `id` into `slot`. Produced by the decision
/// pass, consumed by the fault pass.
pub(super) struct Fault {
    pub(super) id: u64,
    pub(super) slot: u32,
}

/// Where a worker writes. Raw because the slots a batch owns are disjoint, so
/// the workers do not alias and cannot be expressed as split `&mut` slices
/// without carving the arena up first.
///
/// SAFETY (the `Send`/`Sync`): both pointers are into pinned allocations that
/// outlive the `thread::scope` below -- the cache owns them and is borrowed for
/// the whole scope. Every worker writes ONLY `[slot*stride, +stride)` for slots
/// this batch pinned, and `victim` never hands the same slot out twice while
/// pinned, so no two workers touch the same bytes.
pub(super) struct ArenaPtrs {
    pub(super) rows: *mut u8,
    pub(super) scales: Option<*mut u8>,
}
unsafe impl Send for ArenaPtrs {}
unsafe impl Sync for ArenaPtrs {}

/// The immutable half of the cache a worker needs: which file holds a row, and
/// where. Split out so the fault pass can borrow it shared across threads while
/// each worker keeps its OWN bounce buffer (the old single `self.bounce` was a
/// second serialisation point, independent of the I/O).
pub(super) struct RowSource<'a> {
    pub(super) file: &'a File,
    pub(super) base_offset: u64,
    pub(super) segments: Option<&'a Segments>,
    pub(super) row_stride: usize,
    pub(super) scale_file: Option<&'a File>,
}

impl RowSource<'_> {
    /// The file holding row `id`, and the row's byte offset within it. Same
    /// divide as the method it replaces -- a shard carries its own base offset
    /// AND its own backing file, and resolving one without the other was the
    /// bug this signature exists to prevent.
    fn row_loc(&self, id: u64) -> (&File, u64) {
        match self.segments {
            None => (self.file, self.base_offset + id * self.row_stride as u64),
            Some(seg) => {
                let shard = (id / seg.rows_per) as usize;
                let local = id % seg.rows_per;
                let file = &seg.files[seg.shard_file[shard] as usize];
                (file, seg.bases[shard] + local * self.row_stride as u64)
            }
        }
    }
}

/// Read one row (and its FP8 scale, when the table has a per-row scale file)
/// into `slot`. Free function, not a method: it must be callable from several
/// worker threads at once, each with its own `bounce`.
pub(super) fn fetch_row(
    src: &RowSource<'_>,
    ptrs: &ArenaPtrs,
    bounce: &mut AlignedBlock,
    id: u64,
    slot: u32,
) -> Result<()> {
    let (file, byte) = src.row_loc(id);
    let block_off = byte - (byte % BLOCK as u64);
    let within = (byte - block_off) as usize;
    // One block unless the row crosses the boundary (possible whenever the
    // table's base offset is not 4 KiB-aligned, i.e. reading in place from a
    // safetensors shard).
    let nblocks = if within + src.row_stride > BLOCK {
        2
    } else {
        1
    };
    atlas_tier::pio::read_exact_at(file, bounce.blocks(nblocks), block_off)
        .with_context(|| format!("NgramRowCache: read row {id}"))?;
    // SAFETY: slot is one this batch pinned, so it is < slots and no other
    // worker writes it; the arena holds slots*row_stride bytes.
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            ptrs.rows.add(slot as usize * src.row_stride),
            src.row_stride,
        )
    };
    dst.copy_from_slice(&bounce.blocks(nblocks)[within..within + src.row_stride]);

    // A constant per-tensor scale needs no per-row refresh: every slot already
    // holds it (see `set_constant_scale`), and `scale_file` is None then.
    if let (Some(sfile), Some(sbase)) = (src.scale_file, ptrs.scales) {
        let sbyte = id * 4;
        let sblock = sbyte - (sbyte % BLOCK as u64);
        let swithin = (sbyte - sblock) as usize;
        atlas_tier::pio::read_exact_at(sfile, bounce.blocks(1), sblock)
            .with_context(|| format!("NgramRowCache: read scale {id}"))?;
        // SAFETY: as above; the scale arena holds slots*4 bytes.
        let sdst = unsafe { std::slice::from_raw_parts_mut(sbase.add(slot as usize * 4), 4) };
        sdst.copy_from_slice(&bounce.blocks(1)[swithin..swithin + 4]);
    }
    Ok(())
}
