// SPDX-License-Identifier: AGPL-3.0-only

//! The two tier seams — [`SlotArena`] (hot) and [`SwapStore`] (cold) — plus
//! [`SwapStats`], the counters [`crate::Residency`] keeps over them.

use anyhow::Result;

/// The hot tier: a RAM arena as a set of `num_slots` fixed-size slots. The
/// cache peer implements this over its `mmap`'d MR (page-aligned →
/// O_DIRECT-safe); in-process consumers use [`crate::VecSlotArena`].
pub trait SlotArena: Send {
    fn slot_bytes(&self) -> usize;
    fn num_slots(&self) -> usize;
    /// Copy arena slot → `out` (for spilling a victim to disk). `out.len()`
    /// MUST equal `slot_bytes()`.
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()>;
    /// Copy `bytes` → arena slot (for faulting a record back in). `bytes.len()`
    /// MUST equal `slot_bytes()`.
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()>;
}

/// The cold tier: an unbounded fixed-stride record store addressed by a
/// monotonic `disk_slot` index. The peer implements this over an O_DIRECT NVMe
/// file ([`crate::DirectSwapFile`]); [`crate::MemSwapStore`] is the host-RAM
/// variant.
pub trait SwapStore: Send {
    fn record_bytes(&self) -> usize;
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()>;
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()>;
    /// Read the `out.len() / record_bytes()` CONSECUTIVE records starting at
    /// `first_slot` into `out`, in slot order. Default: one `read_record` per
    /// record; a store with a cheaper contiguous path (one large O_DIRECT
    /// `pread`) overrides it. `out.len()` must be a record multiple.
    fn read_records(&self, first_slot: usize, out: &mut [u8]) -> Result<()> {
        let rb = self.record_bytes();
        if rb == 0 || !out.len().is_multiple_of(rb) {
            anyhow::bail!(
                "read_records: {} bytes is not a multiple of {rb}",
                out.len()
            );
        }
        for (i, rec) in out.chunks_exact_mut(rb).enumerate() {
            self.read_record(first_slot + i, rec)?;
        }
        Ok(())
    }
    /// Optional: reclaim disk space for a freed slot (default no-op; a hole in
    /// a preallocated file is fine — the free-list reuses the index).
    fn discard_record(&mut self, _disk_slot: usize) {}
}

/// A record store whose I/O is positional and may be issued from several
/// threads at once (`&self`) — what a worker pool needs for queue depth > 1.
///
/// A *run* is the `buf.len() / record_bytes()` CONSECUTIVE slots starting at
/// `low_slot`. With `reversed` the buffer holds them highest slot first: a
/// chain spilled leaf-first restores root-first, and one vectored request
/// moves the whole run without reordering the caller's staging.
pub trait ConcurrentSwapStore: Send + Sync {
    fn record_bytes(&self) -> usize;
    fn read_run(&self, low_slot: usize, reversed: bool, out: &mut [u8]) -> Result<()>;
    fn write_run(&self, low_slot: usize, reversed: bool, bytes: &[u8]) -> Result<()>;
}

/// Any [`SwapStore`] behind a mutex: correct, one request at a time.
impl<S: SwapStore> ConcurrentSwapStore for std::sync::Mutex<S> {
    fn record_bytes(&self) -> usize {
        self.lock().map_or(0, |s| s.record_bytes())
    }

    fn read_run(&self, low_slot: usize, reversed: bool, out: &mut [u8]) -> Result<()> {
        let s = self.lock().map_err(|_| anyhow::anyhow!("store poisoned"))?;
        if !reversed {
            return s.read_records(low_slot, out);
        }
        let rb = run_records(s.record_bytes(), out.len())?.0;
        for (i, rec) in out.chunks_exact_mut(rb).rev().enumerate() {
            s.read_record(low_slot + i, rec)?;
        }
        Ok(())
    }

    fn write_run(&self, low_slot: usize, reversed: bool, bytes: &[u8]) -> Result<()> {
        let mut s = self.lock().map_err(|_| anyhow::anyhow!("store poisoned"))?;
        let (rb, n) = run_records(s.record_bytes(), bytes.len())?;
        for (i, rec) in bytes.chunks_exact(rb).enumerate() {
            let slot = if reversed { n - 1 - i } else { i };
            s.write_record(low_slot + slot, rec)?;
        }
        Ok(())
    }
}

/// `(record_bytes, records)` of a run buffer; rejects a ragged or empty one.
pub(crate) fn run_records(record_bytes: usize, len: usize) -> Result<(usize, usize)> {
    if record_bytes == 0 || len == 0 || !len.is_multiple_of(record_bytes) {
        anyhow::bail!("record run: {len} bytes is not a positive multiple of {record_bytes}");
    }
    Ok((record_bytes, len / record_bytes))
}

// Boxed trait objects compose (lets a consumer pick arena/swap impls at
// runtime: `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>`).
impl<T: SlotArena + ?Sized> SlotArena for Box<T> {
    fn slot_bytes(&self) -> usize {
        (**self).slot_bytes()
    }
    fn num_slots(&self) -> usize {
        (**self).num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_slot(slot, bytes)
    }
}

impl<T: SwapStore + ?Sized> SwapStore for Box<T> {
    fn record_bytes(&self) -> usize {
        (**self).record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_record(disk_slot, out)
    }
    fn read_records(&self, first_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_records(first_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        (**self).discard_record(disk_slot)
    }
}

#[derive(Default, Debug, Clone)]
pub struct SwapStats {
    pub puts: u64,
    pub gets: u64,
    pub get_miss: u64,
    pub spills_to_disk: u64,
    pub faults_from_disk: u64,
    pub resident_hits: u64,
    /// Cold on-disk snapshots dropped because the disk cap was hit (a later GET
    /// for one cleanly misses → recompute).
    pub disk_evictions: u64,
}
