// SPDX-License-Identifier: AGPL-3.0-only

//! Byte movement for the prefix cache's NVMe spill tier (`ATLAS_KV_NVME_DIR`;
//! bookkeeping lives in the radix tree — see `crate::prefix_cache::nvme`).
//!
//! One record = one physical block's bytes on THIS rank, gathered from every
//! per-block device region the cache owns for that block:
//!
//! ```text
//! [layer0 K][layer0 V if not aliased][layer0 index values][layer0 index scales]
//! [layer1 …] … [zero pad] [trailer: magic u64 | tag u64 | checksum u64]
//! ```
//!
//! padded to a 4 KiB multiple (O_DIRECT). The raw sparse-index TAILS are not
//! part of a record: they only hold the raw keys of a pool still being
//! assembled, and only FULL blocks (every pool finalised) are ever spilled.
//! The trailer binds the record to the node that spilled it (the tree's tag)
//! and to its bytes (checksum); any mismatch fails the restore — the caller
//! then recomputes, so a torn, stale or misdirected record is never served.
//!
//! This file holds the record format and the synchronous path (one `pwrite`
//! per record, one small copy per region per block). `ATLAS_GLM_NVME_FAST=1`
//! moves the SAME records through `nvme_fast.rs` instead.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail, ensure};
use atlas_tier::ConcurrentSwapStore;

use super::PagedKvCache;
use super::nvme_fast::{self, FastIo};
use super::nvme_sync::{read_batch, write_batch};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::prefix_cache::{DiskRef, SpillOrder};

/// Spill writes that failed (process-wide), for throttled logging.
static WRITE_FAILURES: AtomicU64 = AtomicU64::new(0);

const MAGIC: u64 = u64::from_le_bytes(*b"ATLKVNV1");
const TRAILER: usize = 3 * std::mem::size_of::<u64>();
const RECORD_ALIGN: usize = 4096;
/// Records staged per gather/scatter round (one stream sync per round).
const STAGING_RECORDS: usize = 32;

/// One per-block device region: block `b` lives at `base + b·stride`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Segment {
    pub(super) base: DevicePtr,
    pub(super) stride: usize,
}

pub(super) struct NvmeSpill {
    pub(super) store: Arc<dyn ConcurrentSwapStore>,
    pub(super) segments: Vec<Segment>,
    pub(super) payload: usize,
    pub(super) record: usize,
    pub(super) staging: *mut u8,
    pub(super) staging_bytes: usize,
    /// `ATLAS_GLM_NVME_FAST`: write-behind / pipelined I/O over the staging
    /// ring. `None` = the synchronous path below.
    pub(super) fast: Option<FastIo>,
    io: NvmeIoStats,
}

/// What the tier's I/O has cost the SERVING thread so far, and how
/// contiguous the blocks it moved were. `spill_micros` is the time inside
/// `nvme_write`: all of a spill on the synchronous path; on the fast path the
/// gather plus `spill_wait_micros`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeIoStats {
    pub fast: bool,
    pub spilled_blocks: u64,
    pub spill_micros: u64,
    /// Of `spill_micros`, waiting for the write-behind: a full staging ring
    /// (the spill is then throttled to the disk's write speed) or a re-issued
    /// slot whose earlier write is still queued. Fast path only.
    pub spill_wait_micros: u64,
    /// Restores waiting for queued writes before their first read. Fast path
    /// only.
    pub flush_micros: u64,
    pub restored_blocks: u64,
    /// Device copy runs so far: one per run of consecutive blocks on the
    /// fast path, one per block on the synchronous path. Blocks ÷ runs says
    /// how contiguous the pool was.
    pub gather_runs: u64,
    pub scatter_runs: u64,
}

impl NvmeIoStats {
    /// What was added since `earlier` (an older reading of the same cache).
    pub fn since(self, earlier: Self) -> Self {
        Self {
            fast: self.fast,
            spilled_blocks: self.spilled_blocks - earlier.spilled_blocks,
            spill_micros: self.spill_micros - earlier.spill_micros,
            spill_wait_micros: self.spill_wait_micros - earlier.spill_wait_micros,
            flush_micros: self.flush_micros - earlier.flush_micros,
            restored_blocks: self.restored_blocks - earlier.restored_blocks,
            gather_runs: self.gather_runs - earlier.gather_runs,
            scatter_runs: self.scatter_runs - earlier.scatter_runs,
        }
    }
}

// SAFETY: `staging` is a uniquely owned host allocation; the cache (and so
// this struct) is only ever used behind the model's `Mutex<PagedKvCache>`.
unsafe impl Send for NvmeSpill {}
unsafe impl Sync for NvmeSpill {}

/// Fast 64-bit checksum over the payload: four independent multiply-fold
/// lanes over 32-byte strides (so the multiply latency chains overlap — a
/// ~104 KB GLM record costs a few µs), folded together with the length.
/// Detects torn / partial / misdirected records; not a cryptographic MAC.
fn checksum(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let word = |w: &[u8]| u64::from_le_bytes(w.try_into().expect("8-byte word"));
    let mut lanes = [
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    let mut strides = bytes.chunks_exact(32);
    for st in &mut strides {
        for (l, w) in lanes.iter_mut().zip(st.chunks_exact(8)) {
            *l = (*l ^ word(w)).wrapping_mul(K).rotate_left(27);
        }
    }
    let mut tail = [0u8; 32];
    tail[..strides.remainder().len()].copy_from_slice(strides.remainder());
    let mut h = bytes.len() as u64;
    for (l, w) in lanes.iter().zip(tail.chunks_exact(8)) {
        h = (h ^ l ^ word(w)).wrapping_mul(K).rotate_left(31);
    }
    h
}

pub(super) fn stamp(rec: &mut [u8], payload: usize, tag: u64) {
    let sum = checksum(&rec[..payload]);
    let t = rec.len() - TRAILER;
    rec[t..t + 8].copy_from_slice(&MAGIC.to_le_bytes());
    rec[t + 8..t + 16].copy_from_slice(&tag.to_le_bytes());
    rec[t + 16..].copy_from_slice(&sum.to_le_bytes());
}

pub(super) fn verify(rec: &[u8], payload: usize, tag: u64) -> bool {
    let t = rec.len() - TRAILER;
    let word = |o: usize| u64::from_le_bytes(rec[t + o..t + o + 8].try_into().expect("8 bytes"));
    word(0) == MAGIC && word(8) == tag && word(16) == checksum(&rec[..payload])
}

/// A spill write failed: count it and log at 1, 2, 4, 8, … (a full or failing
/// disk fails EVERY spill).
pub(super) fn note_write_failure(slot: u32, e: &anyhow::Error) {
    let n = WRITE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::warn!(
            "NVMe KV spill: write of slot {slot} failed ({e:#}) — {n} failed spills so far; \
             those blocks are evicted as without the tier (recompute on reuse)"
        );
    }
}

impl NvmeSpill {
    pub(super) fn free_staging(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        // Joins the workers (after their queued jobs) before staging goes away.
        self.fast = None;
        let ptr = std::mem::replace(&mut self.staging, std::ptr::null_mut());
        gpu.free_host_pinned(ptr, self.staging_bytes)
    }
}

impl PagedKvCache {
    fn nvme_segments(&self) -> Vec<Segment> {
        let mut segs = Vec::new();
        for l in &self.layers {
            let mut push = |base: DevicePtr, stride: usize| {
                if !base.is_null() && stride > 0 {
                    segs.push(Segment { base, stride });
                }
            };
            push(l.k_pool, l.k_block_stride);
            push(l.owned_v_pool(), l.v_block_stride);
            push(l.sparse_index_values, l.sparse_index_values_block_stride);
            push(l.sparse_index_scales, l.sparse_index_scales_block_stride);
        }
        segs
    }

    /// Bytes of one spill record for this cache's current layout (attach the
    /// sparse index FIRST). Size the store's records with this.
    pub fn nvme_record_bytes(&self) -> usize {
        let payload: usize = self.nvme_segments().iter().map(|s| s.stride).sum();
        (payload + TRAILER).next_multiple_of(RECORD_ALIGN)
    }

    /// [`Self::nvme_record_bytes`] for a cache that is not built yet: KV
    /// sizing needs the tier's host-memory reserve before the pool exists.
    pub fn nvme_record_bytes_for(
        config: &super::KvCacheConfig,
        v_aliases_k: bool,
        index: Option<super::SparseIndexCacheConfig>,
    ) -> usize {
        let bs = config.block_size;
        let index = index.map_or(0, |i| i.values_block_bytes(bs) + i.scales_block_bytes(bs));
        let payload: usize = (0..config.num_layers)
            .map(|l| {
                let v = if v_aliases_k {
                    0
                } else {
                    config.v_block_bytes_for_layer(l)
                };
                config.k_block_bytes_for_layer(l) + v + index
            })
            .sum();
        (payload + TRAILER).next_multiple_of(RECORD_ALIGN)
    }

    /// Pinned staging the tier allocates for a record of `record` bytes.
    pub fn nvme_staging_bytes(record: usize, fast: bool) -> usize {
        let records = if fast {
            nvme_fast::RING_CHUNKS * nvme_fast::CHUNK_RECORDS
        } else {
            STAGING_RECORDS
        };
        record * records
    }

    /// Attach the spill store (records of [`Self::nvme_record_bytes`]) on the
    /// synchronous path.
    pub fn attach_nvme_spill(
        &mut self,
        store: Box<dyn atlas_tier::SwapStore>,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.attach_nvme(Arc::new(Mutex::new(store)), gpu, false)
    }

    /// Attach a store that takes concurrent requests, on the fast path
    /// (`nvme_fast.rs`: pitched copies, run-sized I/O, write-behind).
    pub fn attach_nvme_fast(
        &mut self,
        store: Arc<dyn ConcurrentSwapStore>,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.attach_nvme(store, gpu, true)
    }

    fn attach_nvme(
        &mut self,
        store: Arc<dyn ConcurrentSwapStore>,
        gpu: &dyn GpuBackend,
        fast: bool,
    ) -> Result<()> {
        ensure!(self.nvme.is_none(), "NVMe spill store already attached");
        ensure!(
            self.config.cache_blocks_per_seq.is_none(),
            "the NVMe prefix spill tier cannot be combined with --high-speed-swap"
        );
        let segments = self.nvme_segments();
        let payload: usize = segments.iter().map(|s| s.stride).sum();
        let record = self.nvme_record_bytes();
        ensure!(payload > 0, "KV cache has no per-block regions to spill");
        if store.record_bytes() != record {
            bail!(
                "NVMe spill store record size {} != KV record size {record}",
                store.record_bytes()
            );
        }
        let staging_bytes = Self::nvme_staging_bytes(record, fast);
        let staging = gpu.alloc_host_pinned(staging_bytes)?;
        if !(staging as usize).is_multiple_of(RECORD_ALIGN) {
            // Correct either way — the store bounces — but every record then
            // pays an extra copy, and the synchronous path loses its ranged reads.
            tracing::warn!(
                "NVMe spill tier: pinned staging is not 4 KiB-aligned; O_DIRECT I/O will bounce"
            );
        }
        let mut spill = NvmeSpill {
            store,
            segments,
            payload,
            record,
            staging,
            staging_bytes,
            fast: None,
            io: NvmeIoStats {
                fast,
                ..NvmeIoStats::default()
            },
        };
        if fast {
            match FastIo::new(&spill) {
                Ok(io) => spill.fast = Some(io),
                Err(e) => {
                    spill.free_staging(gpu)?;
                    return Err(e.context("NVMe spill tier: starting the I/O workers"));
                }
            }
        }
        self.nvme = Some(spill);
        Ok(())
    }

    pub fn nvme_io_stats(&self) -> NvmeIoStats {
        self.nvme
            .as_ref()
            .map_or_else(NvmeIoStats::default, |s| s.io)
    }

    /// Spill writes that failed since the last report (fast path: a write
    /// fails after `nvme_write` returned). The tree must drop those nodes.
    pub fn nvme_take_failed(&mut self) -> Vec<SpillOrder> {
        let fast = self.nvme.as_mut().and_then(|s| s.fast.as_mut());
        fast.map_or_else(Vec::new, FastIo::take_failed)
    }

    pub fn nvme_attached(&self) -> bool {
        self.nvme.is_some()
    }

    /// Prefix-cache blocks to evict per allocation miss: one without the
    /// tier (unchanged); a staging batch with it, so a spill pays its two
    /// stream syncs once per batch instead of once per block.
    pub fn evict_batch(&self) -> usize {
        if self.nvme.is_some() {
            STAGING_RECORDS
        } else {
            1
        }
    }

    /// Write each order's block to its record. Must run BEFORE the blocks go
    /// back to the free list. Returns the orders that did NOT reach disk (the
    /// tree must drop those nodes). Work in flight on EVERY stream is drained
    /// first (a device-wide sync); cached blocks are complete and never
    /// rewritten after that.
    pub fn nvme_write(
        &mut self,
        orders: &[SpillOrder],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Vec<SpillOrder> {
        let Some(mut spill) = self.nvme.take() else {
            return orders.to_vec();
        };
        let t0 = std::time::Instant::now();
        let mut failed = Vec::new();
        let mut io = spill.io;
        if let Some(mut fast) = spill.fast.take() {
            failed = nvme_fast::write(&spill, (&mut fast, &mut io), orders, gpu, stream);
            spill.fast = Some(fast);
        } else {
            io.gather_runs += orders.len() as u64;
            for batch in orders.chunks(STAGING_RECORDS) {
                if let Err(e) = write_batch(&mut spill, batch, gpu, stream, &mut failed) {
                    // Nothing of this batch reached disk (writes follow the gather).
                    tracing::warn!("NVMe KV spill: gather failed ({e:#}); dropping batch");
                    failed.extend_from_slice(batch);
                }
            }
        }
        io.spilled_blocks += orders.len() as u64;
        io.spill_micros += t0.elapsed().as_micros() as u64;
        spill.io = io;
        self.nvme = Some(spill);
        failed
    }

    /// Read `disk[i]` into freshly allocated `blocks[i]` (in order), verifying
    /// every trailer. Returns how many leading blocks hold verified bytes and
    /// whether the next one FAILED (I/O error or bad record) — as opposed to
    /// simply not being attempted. Scatter completes before returning. The
    /// fast path re-orders `blocks` (ascending) before pairing them up.
    pub fn nvme_read(
        &mut self,
        disk: &[DiskRef],
        blocks: &mut [u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> (usize, bool) {
        let Some(mut spill) = self.nvme.take() else {
            return (0, false);
        };
        let mut io = spill.io;
        if let Some(mut fast) = spill.fast.take() {
            let r = nvme_fast::read(&spill, (&mut fast, &mut io), disk, blocks, gpu, stream);
            io.restored_blocks += r.0 as u64;
            spill.fast = Some(fast);
            spill.io = io;
            self.nvme = Some(spill);
            return r;
        }
        let n = disk.len().min(blocks.len());
        let mut done = 0;
        let mut failed = false;
        while done < n && !failed {
            let end = (done + STAGING_RECORDS).min(n);
            match read_batch(
                &mut spill,
                &disk[done..end],
                &blocks[done..end],
                gpu,
                stream,
            ) {
                Ok(k) => {
                    failed = k < end - done;
                    done += k;
                }
                Err(e) => {
                    tracing::warn!("NVMe KV restore: scatter failed ({e:#})");
                    failed = true;
                }
            }
        }
        spill.io.restored_blocks += done as u64;
        spill.io.scatter_runs += done as u64;
        self.nvme = Some(spill);
        (done, failed)
    }
}

/// Staging position of each of `slots` after the synchronous path's `read_runs`: maximal runs of
/// consecutive slots (either direction — a chain spills leaf-first, so its
/// path order is usually DEScending) are read with one `read_records` each and
/// land in slot order within the run's window. Pure, for testing.
pub(super) fn run_layout(slots: &[u32]) -> Vec<(usize, usize, u32)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let step = slots
            .get(i + 1)
            .map_or(0, |&n| i64::from(n) - i64::from(slots[i]));
        let mut j = i;
        if step == 1 || step == -1 {
            while j + 1 < slots.len() && i64::from(slots[j + 1]) - i64::from(slots[j]) == step {
                j += 1;
            }
        }
        let first = slots[i..=j].iter().copied().min().expect("non-empty run");
        runs.push((i, j + 1 - i, first));
        i = j + 1;
    }
    runs
}
