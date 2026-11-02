// SPDX-License-Identifier: AGPL-3.0-only

//! The NVMe prefix tier's synchronous path (no `ATLAS_GLM_NVME_FAST`): gather
//! → stamp → one write per record on the evicting thread, and read → verify →
//! scatter in staging batches. Moved verbatim out of `nvme_spill.rs` (500-LoC
//! cap); the record format and the dispatch live there.

use anyhow::Result;

use super::nvme_spill::{NvmeSpill, note_write_failure, run_layout, stamp, verify};
use crate::gpu::GpuBackend;
use crate::prefix_cache::{DiskRef, SpillOrder};

pub(super) fn write_batch(
    spill: &mut NvmeSpill,
    batch: &[SpillOrder],
    gpu: &dyn GpuBackend,
    stream: u64,
    failed: &mut Vec<SpillOrder>,
) -> Result<()> {
    let (record, payload) = (spill.record, spill.payload);
    // SAFETY: the staging buffer is owned by `spill` and disjoint from
    // `spill.store` / `spill.segments`, which are borrowed alongside it.
    let staging = unsafe { std::slice::from_raw_parts_mut(spill.staging, spill.staging_bytes) };
    // Device-wide, not just `stream`: the victims' last writes may still be in
    // flight on ANOTHER stream (the scheduler's prefill stream, MoE/secondary
    // streams — all non-blocking), and a record gathered before they land
    // would checksum and later restore torn KV. Paid once per staging batch.
    gpu.synchronize_device()?;
    let mut enqueue = || -> Result<()> {
        for (i, o) in batch.iter().enumerate() {
            let mut off = i * record;
            for s in &spill.segments {
                let src = s.at(o.block);
                gpu.copy_d2h_async(src, &mut staging[off..off + s.stride], stream)?;
                off += s.stride;
            }
        }
        Ok(())
    };
    let enq = enqueue();
    // Drain even after a failed enqueue: earlier chunks still DMA into staging.
    gpu.synchronize(stream)?;
    enq?;
    for (i, o) in batch.iter().enumerate() {
        let rec = &mut staging[i * record..(i + 1) * record];
        rec[payload..].fill(0);
        stamp(rec, payload, o.tag);
        if let Err(e) = spill.store.write_run(o.slot as usize, false, rec) {
            note_write_failure(o.slot, &e);
            failed.push(*o);
        }
    }
    Ok(())
}

/// Read `disk` into staging; returns each entry's staging record index for
/// the leading entries that were read (stops at the first unreadable one).
fn read_runs(spill: &NvmeSpill, disk: &[DiskRef], staging: &mut [u8]) -> Vec<usize> {
    let record = spill.record;
    let slots: Vec<u32> = disk.iter().map(|d| d.slot).collect();
    let mut pos = Vec::with_capacity(disk.len());
    for (start, len, first) in run_layout(&slots) {
        let window = &mut staging[start * record..(start + len) * record];
        if len > 1 && spill.store.read_run(first as usize, false, window).is_ok() {
            pos.extend((start..start + len).map(|k| start + (slots[k] - first) as usize));
            continue;
        }
        // Single record, or the ranged read failed: find the exact failure.
        for k in start..start + len {
            let rec = &mut staging[k * record..(k + 1) * record];
            if let Err(e) = spill.store.read_run(slots[k] as usize, false, rec) {
                tracing::warn!("NVMe KV restore: read of slot {} failed ({e:#})", slots[k]);
                return pos;
            }
            pos.push(k);
        }
    }
    pos
}

pub(super) fn read_batch(
    spill: &mut NvmeSpill,
    disk: &[DiskRef],
    blocks: &[u32],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<usize> {
    let (record, payload) = (spill.record, spill.payload);
    // SAFETY: as in `write_batch`.
    let staging = unsafe { std::slice::from_raw_parts_mut(spill.staging, spill.staging_bytes) };
    let pos = read_runs(spill, disk, staging);
    let mut ok = 0;
    for (d, &p) in disk.iter().zip(&pos) {
        if !verify(&staging[p * record..(p + 1) * record], payload, d.tag) {
            tracing::warn!(
                "NVMe KV restore: slot {} failed verification (tag/checksum) — recomputing",
                d.slot
            );
            break;
        }
        ok += 1;
    }
    let enqueue = || -> Result<()> {
        for (&b, &p) in blocks[..ok].iter().zip(&pos) {
            let mut off = p * record;
            for s in &spill.segments {
                let dst = s.at(b);
                gpu.copy_h2d_async_retained(&staging[off..off + s.stride], dst, stream)?;
                off += s.stride;
            }
        }
        Ok(())
    };
    let enq = enqueue();
    gpu.synchronize(stream)?;
    enq?;
    Ok(ok)
}
