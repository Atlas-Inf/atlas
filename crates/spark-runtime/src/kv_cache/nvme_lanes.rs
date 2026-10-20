// SPDX-License-Identifier: AGPL-3.0-only

//! The NVMe prefix tier's record LANES: one per slot class (see
//! `nvme_spill.rs`). Unsharded there is one lane and `nvme_write` /
//! `nvme_read` hand it the tree's orders as they are. Under a latent shard
//! (`ATLAS_GLM_KV_SHARD=1`) each class's orders go to its lane at
//! `slot / classes`: the own class's records carry this rank's latents and
//! the index rows, the peer class's the index rows only.

use super::PagedKvCache;
use super::nvme_fast;
use super::nvme_spill::{NvmeSpill, STAGING_RECORDS};
use super::nvme_sync::{read_batch, write_batch};
use crate::gpu::GpuBackend;
use crate::prefix_cache::{DiskRef, SpillOrder};

/// Record sizes of the tier's slot classes on this rank (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvmeGeometry {
    /// A record of a block whose latents this rank stores: every block
    /// unless latent-sharded.
    pub own: usize,
    /// Latent-sharded: a record of a block the peer stores (its index rows
    /// only). `None` unsharded.
    pub peer: Option<usize>,
}

impl NvmeGeometry {
    /// Slot classes (= record lanes): 1, or the shard's world (a pair).
    pub fn classes(&self) -> usize {
        if self.peer.is_some() { 2 } else { 1 }
    }

    /// Record bytes of class `class` on rank `rank`.
    pub fn class_bytes(&self, class: usize, rank: usize) -> usize {
        match self.peer {
            Some(peer) if class != rank => peer,
            _ => self.own,
        }
    }

    /// Disk bytes of one slot of EVERY class: what a budget is divided by to
    /// get the slots per class.
    pub fn row_bytes(&self) -> usize {
        self.own + self.peer.unwrap_or(0)
    }

    /// Pinned staging of every lane (see [`PagedKvCache::nvme_staging_bytes`]).
    pub fn staging_bytes(&self, fast: bool) -> usize {
        PagedKvCache::nvme_staging_bytes(self.own, fast)
            + self
                .peer
                .map_or(0, |p| PagedKvCache::nvme_staging_bytes(p, fast))
    }
}

impl PagedKvCache {
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
        if self.nvme.len() <= 1 {
            let Some(spill) = self.nvme.first_mut() else {
                return orders.to_vec();
            };
            return write_lane(spill, orders, gpu, stream);
        }
        // One lane per slot class: each writes its orders at `slot / classes`.
        let classes = self.nvme.len() as u32;
        let mut failed = Vec::new();
        for class in 0..classes {
            let mut lane_orders = Vec::new();
            for o in orders.iter().filter(|o| o.slot % classes == class) {
                if o.block % classes == class {
                    lane_orders.push(SpillOrder {
                        slot: o.slot / classes,
                        ..*o
                    });
                } else {
                    // The tree classes a spill by its block's residue.
                    tracing::warn!(
                        "NVMe KV spill: block {} cannot take slot {} (class {class}); dropped",
                        o.block,
                        o.slot
                    );
                    failed.push(*o);
                }
            }
            if lane_orders.is_empty() {
                continue;
            }
            let lost = write_lane(&mut self.nvme[class as usize], &lane_orders, gpu, stream);
            failed.extend(lost.into_iter().map(|o| SpillOrder {
                slot: o.slot * classes + class,
                ..o
            }));
        }
        failed
    }

    /// Read `disk[i]` into freshly allocated `blocks[i]` (in order), verifying
    /// every trailer. Returns how many leading blocks hold verified bytes and
    /// whether the next one FAILED (I/O error or bad record) — as opposed to
    /// simply not being attempted. Scatter completes before returning. The
    /// fast path re-orders `blocks` (ascending) before pairing them up — under
    /// a latent shard only among blocks of one residue, so every block keeps
    /// the residue of its position.
    pub fn nvme_read(
        &mut self,
        disk: &[DiskRef],
        blocks: &mut [u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> (usize, bool) {
        if self.nvme.len() <= 1 {
            let Some(spill) = self.nvme.first_mut() else {
                return (0, false);
            };
            return read_lane(spill, disk, blocks, gpu, stream);
        }
        let classes = self.nvme.len() as u32;
        let n = disk.len().min(blocks.len());
        // A target block of the wrong residue cannot take the record (it was
        // not allocated at its logical index): restore up to it, then fail.
        let limit = (0..n)
            .find(|&i| blocks[i] % classes != disk[i].slot % classes)
            .unwrap_or(n);
        if limit < n {
            tracing::warn!(
                "NVMe KV restore: block {} cannot take slot {}; recomputing from there",
                blocks[limit],
                disk[limit].slot
            );
        }
        let (mut done, mut failed) = (limit, limit < n);
        for class in 0..classes {
            let at: Vec<usize> = (0..limit)
                .filter(|&i| disk[i].slot % classes == class)
                .collect();
            // Even an empty lane is asked: a fast-path read first waits for
            // that lane's queued writes, which callers rely on.
            let lane_disk: Vec<DiskRef> = at
                .iter()
                .map(|&i| DiskRef {
                    slot: disk[i].slot / classes,
                    ..disk[i]
                })
                .collect();
            let mut lane_blocks: Vec<u32> = at.iter().map(|&i| blocks[i]).collect();
            let lane = &mut self.nvme[class as usize];
            let (ok, lane_failed) = read_lane(lane, &lane_disk, &mut lane_blocks, gpu, stream);
            // The fast path sorts a lane's blocks: same residue, any position.
            for (&i, &b) in at.iter().zip(&lane_blocks) {
                blocks[i] = b;
            }
            // This lane's first unread record bounds the restored prefix (the
            // lanes' positions are disjoint, and all lie below `limit`).
            if let Some(&first_missing) = at.get(ok)
                && first_missing < done
            {
                (done, failed) = (first_missing, lane_failed);
            }
        }
        (done, failed)
    }
}

/// [`PagedKvCache::nvme_write`] on one lane (lane-local slots).
fn write_lane(
    spill: &mut NvmeSpill,
    orders: &[SpillOrder],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Vec<SpillOrder> {
    let t0 = std::time::Instant::now();
    let mut failed = Vec::new();
    let mut io = spill.io;
    if let Some(mut fast) = spill.fast.take() {
        failed = nvme_fast::write(spill, (&mut fast, &mut io), orders, gpu, stream);
        spill.fast = Some(fast);
    } else {
        io.gather_runs += orders.len() as u64;
        for batch in orders.chunks(STAGING_RECORDS) {
            if let Err(e) = write_batch(spill, batch, gpu, stream, &mut failed) {
                // Nothing of this batch reached disk (writes follow the gather).
                tracing::warn!("NVMe KV spill: gather failed ({e:#}); dropping batch");
                failed.extend_from_slice(batch);
            }
        }
    }
    io.spilled_blocks += orders.len() as u64;
    io.spill_micros += t0.elapsed().as_micros() as u64;
    spill.io = io;
    failed
}

/// [`PagedKvCache::nvme_read`] on one lane (lane-local slots).
fn read_lane(
    spill: &mut NvmeSpill,
    disk: &[DiskRef],
    blocks: &mut [u32],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> (usize, bool) {
    let mut io = spill.io;
    if let Some(mut fast) = spill.fast.take() {
        let r = nvme_fast::read(spill, (&mut fast, &mut io), disk, blocks, gpu, stream);
        io.restored_blocks += r.0 as u64;
        spill.fast = Some(fast);
        spill.io = io;
        return r;
    }
    let n = disk.len().min(blocks.len());
    let mut done = 0;
    let mut failed = false;
    while done < n && !failed {
        let end = (done + STAGING_RECORDS).min(n);
        match read_batch(spill, &disk[done..end], &blocks[done..end], gpu, stream) {
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
    (done, failed)
}
