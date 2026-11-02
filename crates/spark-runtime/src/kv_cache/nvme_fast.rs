// SPDX-License-Identifier: AGPL-3.0-only

//! Fast path of the NVMe prefix tier (`ATLAS_GLM_NVME_FAST=1`): the same
//! records, tags and checksums as the synchronous path in `nvme_spill.rs`,
//! moved in three cheaper shapes (measured on a GB10; see
//! `docs/glm-nvme-prefix-cache.md` §9):
//!
//! 1. **Pitched copies.** A block's record is ~22 small device regions, and a
//!    small async copy costs several microseconds of copy-engine time whatever
//!    its size. Staging is laid out in BLOCK order, so a run of consecutive
//!    blocks is one pitched copy per region instead of one copy per region per
//!    block.
//! 2. **Run-sized I/O.** A run of consecutive slots is one request in either
//!    direction (`ConcurrentSwapStore`), not one `pwrite` per record.
//! 3. **Write-behind and read-ahead.** Checksums and disk I/O run on a small
//!    worker pool over a fixed ring of pinned staging chunks. A spill returns
//!    once the blocks' bytes are in staging — the blocks may be reused, their
//!    bytes are owned by the queued write — and a restore reads ahead while
//!    the serving thread scatters.
//!
//! Ordering rules the rest of the tier relies on:
//! * every queued write is on disk before any read is issued ([`FastIo::flush`]);
//! * no two queued writes touch the same slot — a spill that re-uses a slot
//!   still in flight drains the queue first — so the workers may write in any
//!   order and a re-issued slot still ends up holding its later record;
//! * a failed write is reported on the next [`write`] / `nvme_take_failed`,
//!   and until then its record fails verification (wrong tag), never restores.

use std::collections::{HashSet, VecDeque};

use anyhow::Result;

use super::nvme_io::{IoPool, Outcome, Work};
use super::nvme_spill::{NvmeIoStats, NvmeSpill};
use crate::gpu::{GpuBackend, Pitched, copy_d2h_pitched_async, copy_h2d_pitched_async_retained};
use crate::prefix_cache::{DiskRef, SpillOrder};

/// Records per staging chunk (one worker job).
pub(super) const CHUNK_RECORDS: usize = 16;
/// Chunks in the pinned staging ring: the write-behind depth and the restore
/// read-ahead. Fixed — the ring is the tier's only staging memory.
pub(super) const RING_CHUNKS: usize = 8;
/// Checksum / I/O threads (idle unless a spill or restore is in flight): one
/// per chunk, so a fragmented restore keeps the whole ring in flight.
const IO_WORKERS: usize = RING_CHUNKS;
/// Chunks gathered per stream sync on the spill path (one evict batch).
const ROUND_CHUNKS: usize = 2;

pub(super) struct FastIo {
    pool: IoPool,
    /// Chunks no job owns.
    pub(super) free: Vec<usize>,
    /// Queued writes: `(chunk, slots)`.
    writing: Vec<(usize, Vec<u32>)>,
    /// Failed writes not yet reported to the tree.
    failed: Vec<SpillOrder>,
    /// Offset of each segment inside a record.
    seg_off: Vec<usize>,
    /// Microseconds the serving thread has spent waiting for the writer.
    waited: u64,
}

impl FastIo {
    /// `spill.staging` must hold `RING_CHUNKS × CHUNK_RECORDS` records.
    pub(super) fn new(spill: &NvmeSpill) -> Result<Self> {
        let ring = RING_CHUNKS * CHUNK_RECORDS * spill.record;
        anyhow::ensure!(
            spill.staging_bytes >= ring,
            "NVMe staging of {} B cannot hold the {ring} B ring",
            spill.staging_bytes
        );
        let seg_off = spill
            .segments
            .iter()
            .scan(0, |off, s| {
                let at = *off;
                *off += s.stride;
                Some(at)
            })
            .collect();
        Ok(Self {
            pool: IoPool::new(
                spill.store.clone(),
                spill.staging,
                CHUNK_RECORDS * spill.record,
                (spill.record, spill.payload),
                IO_WORKERS,
            )?,
            free: (0..RING_CHUNKS).collect(),
            writing: Vec::new(),
            failed: Vec::new(),
            seg_off,
            waited: 0,
        })
    }

    /// Collect finished writes (with `block`: at least one).
    fn reap(&mut self, block: bool) {
        if self.writing.is_empty() {
            return;
        }
        let t0 = block.then(std::time::Instant::now);
        for (chunk, failed) in self.pool.take_writes(block) {
            self.free.push(chunk);
            self.writing.retain(|(c, _)| *c != chunk);
            self.failed.extend(failed);
        }
        self.waited += t0.map_or(0, |t| t.elapsed().as_micros() as u64);
    }

    /// A chunk nobody owns — waits for the writer when the ring is full,
    /// which is what bounds the write-behind. An error when there is neither
    /// a free chunk nor a write to wait for (a leaked chunk): the caller
    /// fails its batch — waiting would hang the serving thread, and with it
    /// the peer rank's next collective.
    fn chunk(&mut self) -> Result<usize> {
        loop {
            if let Some(c) = self.free.pop() {
                return Ok(c);
            }
            anyhow::ensure!(
                !self.writing.is_empty(),
                "staging ring has no free chunk and no write in flight"
            );
            self.reap(true);
        }
    }

    /// Every queued write has reached the disk (or failed) on return.
    pub(super) fn flush(&mut self) {
        while !self.writing.is_empty() {
            self.reap(true);
        }
    }

    fn is_writing(&self, slot: u32) -> bool {
        self.writing.iter().any(|(_, slots)| slots.contains(&slot))
    }

    pub(super) fn take_failed(&mut self) -> Vec<SpillOrder> {
        self.reap(false);
        std::mem::take(&mut self.failed)
    }
}

/// Maximal runs of ascending blocks `step` ids apart (a lane's neighbours;
/// `step` = 1 unsharded): `(start, len)`.
pub(super) fn block_runs(blocks: &[u32], step: u32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < blocks.len() {
        let mut j = i + 1;
        while j < blocks.len() && blocks[j - 1].checked_add(step) == Some(blocks[j]) {
            j += 1;
        }
        runs.push((i, j - i));
        i = j;
    }
    runs
}

/// Enqueue the copies between `blocks` (record `i` of `chunk` ↔ `blocks[i]`)
/// and staging: one pitched copy per segment per run of consecutive blocks —
/// and for a run of ONE block the plain copies of the synchronous path (a
/// one-row pitched copy buys nothing, so a fully fragmented pool costs what
/// it costs there). Returns the number of runs.
fn copy_blocks(
    spill: &NvmeSpill,
    seg_off: &[usize],
    chunk: usize,
    blocks: &[u32],
    (gpu, stream): (&dyn GpuBackend, u64),
    to_device: bool,
) -> Result<usize> {
    let record = spill.record;
    anyhow::ensure!(
        chunk < RING_CHUNKS && blocks.len() <= CHUNK_RECORDS,
        "NVMe staging: {} blocks do not fit chunk {chunk}",
        blocks.len()
    );
    // SAFETY: the caller owns `chunk` (no job holds it), `blocks` fits a
    // chunk (checked above; `FastIo::new` checked the ring), and the ring is
    // disjoint from everything else borrowed here.
    let stage = unsafe {
        std::slice::from_raw_parts_mut(
            spill.staging.add(chunk * CHUNK_RECORDS * record),
            blocks.len() * record,
        )
    };
    let runs = block_runs(blocks, spill.step as u32);
    for &(start, len) in &runs {
        for (seg, &off) in spill.segments.iter().zip(seg_off) {
            let shape = Pitched {
                host_pitch: record,
                dev_pitch: seg.pitch(spill.step),
                width: seg.stride,
                height: len,
            };
            let host = &mut stage[start * record + off..][..shape.host_span()];
            let dev = seg.at(blocks[start]);
            match (to_device, len) {
                (true, 1) => gpu.copy_h2d_async_retained(host, dev, stream)?,
                (false, 1) => gpu.copy_d2h_async(dev, host, stream)?,
                (true, _) => copy_h2d_pitched_async_retained(gpu, host, dev, shape, stream)?,
                (false, _) => copy_d2h_pitched_async(gpu, dev, host, shape, stream)?,
            }
        }
    }
    Ok(runs.len())
}

/// Write-behind spill; same contract as the synchronous `nvme_write` except
/// that a failed WRITE is reported by a later call (see the module doc).
pub(super) fn write(
    spill: &NvmeSpill,
    (fast, io): (&mut FastIo, &mut NvmeIoStats),
    orders: &[SpillOrder],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Vec<SpillOrder> {
    let waited = fast.waited;
    let mut failed = fast.take_failed();
    // The last order per slot is the live one: a slot the budget re-issued
    // inside this batch belongs to a node that is already gone.
    let mut seen = HashSet::new();
    let mut live: Vec<SpillOrder> = orders
        .iter()
        .rev()
        .filter(|o| seen.insert(o.slot))
        .copied()
        .collect();
    live.sort_unstable_by_key(|o| o.block);
    // Device-wide, as on the synchronous path: a victim's last writes may
    // still be in flight on another stream.
    if let Err(e) = gpu.synchronize_device() {
        tracing::warn!("NVMe KV spill: device sync failed ({e:#}); dropping batch");
        failed.extend(live);
        return failed;
    }
    for round in live.chunks(ROUND_CHUNKS * CHUNK_RECORDS) {
        // A slot re-issued while its earlier write is still queued: that
        // write must land first (see the module doc).
        if round.iter().any(|o| fast.is_writing(o.slot)) {
            fast.flush();
        }
        let mut groups: Vec<(usize, &[SpillOrder])> = Vec::new();
        let gathered = round.chunks(CHUNK_RECORDS).try_fold(0, |runs, g| {
            let chunk = fast.chunk()?;
            groups.push((chunk, g));
            let blocks: Vec<u32> = g.iter().map(|o| o.block).collect();
            copy_blocks(spill, &fast.seg_off, chunk, &blocks, (gpu, stream), false)
                .map(|n| runs + n)
        });
        // Drain even after a failed enqueue: earlier copies still DMA into
        // staging.
        match gpu.synchronize(stream).and(gathered) {
            Ok(runs) => io.gather_runs += runs as u64,
            Err(e) => {
                tracing::warn!("NVMe KV spill: gather failed ({e:#}); dropping batch");
                failed.extend_from_slice(round);
                fast.free.extend(groups.iter().map(|(chunk, _)| *chunk));
                continue;
            }
        }
        for (chunk, g) in groups {
            fast.writing
                .push((chunk, g.iter().map(|o| o.slot).collect()));
            fast.pool.submit(chunk, Work::Write(g.to_vec()));
        }
    }
    io.spill_wait_micros += fast.waited - waited;
    failed
}

/// Pipelined restore; same contract as the synchronous `nvme_read`, plus:
/// `blocks[..n]` is sorted first, so the restored run lands on ascending
/// blocks and scatters in pitched runs (a latent-shard lane's blocks share
/// one residue, so sorting them keeps every block on its residue).
pub(super) fn read(
    spill: &NvmeSpill,
    (fast, io): (&mut FastIo, &mut NvmeIoStats),
    disk: &[DiskRef],
    blocks: &mut [u32],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> (usize, bool) {
    let waited = fast.waited;
    fast.flush();
    io.flush_micros += fast.waited - waited;
    let n = disk.len().min(blocks.len());
    blocks[..n].sort_unstable();
    let mut in_flight: VecDeque<(usize, usize, usize)> = VecDeque::new();
    let (mut next, mut done, mut failed) = (0, 0, false);
    loop {
        while next < n
            && !failed
            && let Some(chunk) = fast.free.pop()
        {
            let end = (next + CHUNK_RECORDS).min(n);
            fast.pool
                .submit(chunk, Work::Read(disk[next..end].to_vec()));
            in_flight.push_back((chunk, next, end));
            next = end;
        }
        let Some((chunk, first, end)) = in_flight.pop_front() else {
            break;
        };
        let ok = match fast.pool.wait(chunk) {
            Outcome::Read(ok) => ok,
            Outcome::Wrote(_) => 0,
        };
        // After a failure the remaining read-ahead is only drained.
        if !failed {
            let run = &blocks[first..first + ok];
            let scattered = copy_blocks(spill, &fast.seg_off, chunk, run, (gpu, stream), true);
            match gpu.synchronize(stream).and(scattered) {
                Ok(runs) => {
                    io.scatter_runs += runs as u64;
                    done += ok;
                    failed = ok < end - first;
                }
                Err(e) => {
                    tracing::warn!("NVMe KV restore: scatter failed ({e:#})");
                    failed = true;
                }
            }
        }
        fast.free.push(chunk);
    }
    (done, failed)
}
