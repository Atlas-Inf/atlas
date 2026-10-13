// SPDX-License-Identifier: AGPL-3.0-only

//! The NVMe tier's fast path (`attach_nvme_fast`): the same records as the
//! synchronous path, moved as pitched copies and slot runs, written behind the
//! evicting thread and never served before they are safe.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::nvme_spill::{dump, fill, glm_cache};
use super::*;
use crate::prefix_cache::{DiskRef, SpillOrder};

/// An in-memory [`atlas_tier::ConcurrentSwapStore`] that counts requests, can
/// fail chosen slots and can hold every write at a gate.
#[derive(Default)]
struct TestStore {
    record: usize,
    recs: Mutex<HashMap<usize, Vec<u8>>>,
    reads: AtomicUsize,
    writes: AtomicUsize,
    fail_writes: Mutex<HashSet<usize>>,
    gate_closed: Mutex<bool>,
    gate: Condvar,
}

impl TestStore {
    fn new(record: usize) -> Arc<Self> {
        Arc::new(Self {
            record,
            ..Self::default()
        })
    }

    fn set_gate(&self, closed: bool) {
        *self.gate_closed.lock().unwrap() = closed;
        self.gate.notify_all();
    }

    /// Slot of record `i` of a run buffer.
    fn slot(low: usize, n: usize, reversed: bool, i: usize) -> usize {
        if reversed { low + n - 1 - i } else { low + i }
    }
}

impl atlas_tier::ConcurrentSwapStore for TestStore {
    fn record_bytes(&self) -> usize {
        self.record
    }

    fn read_run(&self, low: usize, reversed: bool, out: &mut [u8]) -> anyhow::Result<()> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let n = out.len() / self.record;
        let recs = self.recs.lock().unwrap();
        for (i, rec) in out.chunks_exact_mut(self.record).enumerate() {
            let slot = Self::slot(low, n, reversed, i);
            let Some(bytes) = recs.get(&slot) else {
                anyhow::bail!("no record {slot}");
            };
            rec.copy_from_slice(bytes);
        }
        Ok(())
    }

    fn write_run(&self, low: usize, reversed: bool, bytes: &[u8]) -> anyhow::Result<()> {
        let mut closed = self.gate_closed.lock().unwrap();
        while *closed {
            closed = self.gate.wait(closed).unwrap();
        }
        drop(closed);
        self.writes.fetch_add(1, Ordering::Relaxed);
        let n = bytes.len() / self.record;
        let slots: Vec<usize> = (0..n).map(|i| Self::slot(low, n, reversed, i)).collect();
        if slots
            .iter()
            .any(|s| self.fail_writes.lock().unwrap().contains(s))
        {
            anyhow::bail!("injected write failure");
        }
        let mut recs = self.recs.lock().unwrap();
        for (slot, rec) in slots.into_iter().zip(bytes.chunks_exact(self.record)) {
            recs.insert(slot, rec.to_vec());
        }
        Ok(())
    }
}

fn fast_cache(gpu: &MockGpuBackend, blocks: usize) -> (PagedKvCache, Arc<TestStore>) {
    let mut c = glm_cache(gpu, blocks);
    let store = TestStore::new(c.nvme_record_bytes());
    c.attach_nvme_fast(store.clone(), gpu).unwrap();
    (c, store)
}

fn order(block: u32, slot: u32) -> SpillOrder {
    SpillOrder {
        block,
        slot,
        tag: 0x1000 + slot as u64,
    }
}

fn disk(slot: u32) -> DiskRef {
    DiskRef {
        slot,
        tag: 0x1000 + slot as u64,
    }
}

/// Wait for the write-behind and hand back what it reported.
fn settle(c: &mut PagedKvCache, gpu: &MockGpuBackend) -> Vec<SpillOrder> {
    assert_eq!(c.nvme_read(&[], &mut [], gpu, 0), (0, false));
    c.nvme_take_failed()
}

#[test]
fn a_leaf_first_chain_moves_as_runs_and_restores_in_path_order() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 64);
    // A 12-block chain on blocks 10..22, spilled leaf first into fresh slots.
    let mut want = Vec::new();
    for b in 10..22u32 {
        fill(&c, &gpu, b, b as u8);
        want.push(dump(&c, &gpu, b));
    }
    let orders: Vec<SpillOrder> = (0..12).map(|i| order(21 - i, i)).collect();
    let (pitched, plain) = (gpu.host_pitched_count(), gpu.d2h_async_count());
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    assert!(settle(&mut c, &gpu).is_empty());
    // 2 layers × (K, index) = 4 regions: ONE pitched copy each for the run.
    assert_eq!(gpu.host_pitched_count() - pitched, 4);
    assert_eq!(gpu.d2h_async_count(), plain, "no per-block copies");
    assert_eq!(store.writes.load(Ordering::Relaxed), 1, "one slot run");
    // Restore root first (slots 11..=0) into whatever blocks came free.
    let refs: Vec<DiskRef> = (0..12).rev().map(disk).collect();
    let mut blocks = [40u32, 33, 35, 34, 36, 37, 39, 38, 41, 42, 44, 43];
    let pitched = gpu.host_pitched_count();
    assert_eq!(c.nvme_read(&refs, &mut blocks, &gpu, 0), (12, false));
    assert_eq!(blocks, [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]);
    assert_eq!(gpu.host_pitched_count() - pitched, 4);
    assert_eq!(store.reads.load(Ordering::Relaxed), 1);
    for (i, &b) in blocks.iter().enumerate() {
        assert_eq!(dump(&c, &gpu, b), want[i], "path block {i}");
    }
    let io = c.nvme_io_stats();
    assert!(io.fast);
    assert_eq!((io.spilled_blocks, io.gather_runs), (12, 1));
    assert_eq!((io.restored_blocks, io.scatter_runs), (12, 1));
}

#[test]
fn both_paths_write_the_same_records() {
    /// The synchronous path's view of the shared test store.
    struct Boxed(Arc<TestStore>);
    impl atlas_tier::SwapStore for Boxed {
        fn record_bytes(&self) -> usize {
            self.0.record
        }
        fn write_record(&mut self, slot: usize, bytes: &[u8]) -> anyhow::Result<()> {
            atlas_tier::ConcurrentSwapStore::write_run(&*self.0, slot, false, bytes)
        }
        fn read_record(&self, slot: usize, out: &mut [u8]) -> anyhow::Result<()> {
            atlas_tier::ConcurrentSwapStore::read_run(&*self.0, slot, false, out)
        }
    }
    let gpu = MockGpuBackend::new();
    let (mut fast, store) = fast_cache(&gpu, 16);
    let mut sync = glm_cache(&gpu, 16);
    sync.attach_nvme_spill(Box::new(Boxed(store.clone())), &gpu)
        .unwrap();
    // Scattered blocks and slots: no run on either axis.
    let plan = [(3u32, 9u32), (7, 2), (12, 5)];
    for &(b, _) in &plan {
        fill(&fast, &gpu, b, 0x30 + b as u8);
        fill(&sync, &gpu, b, 0x30 + b as u8);
    }
    let orders: Vec<SpillOrder> = plan.iter().map(|&(b, s)| order(b, s)).collect();
    let (pitched, plain) = (gpu.host_pitched_count(), gpu.d2h_async_count());
    assert!(fast.nvme_write(&orders, &gpu, 0).is_empty());
    assert!(settle(&mut fast, &gpu).is_empty());
    // A run of one block is the synchronous path's plain copies (4 regions
    // per block), never a one-row pitched copy.
    assert_eq!(gpu.host_pitched_count(), pitched);
    assert_eq!(gpu.d2h_async_count() - plain, 3 * 4);
    assert_eq!(fast.nvme_io_stats().gather_runs, 3);
    let written = store.recs.lock().unwrap().clone();
    assert!(sync.nvme_write(&orders, &gpu, 0).is_empty());
    assert_eq!(
        *store.recs.lock().unwrap(),
        written,
        "byte-identical records"
    );
    // Either path restores the other's records.
    let refs: Vec<DiskRef> = plan.iter().map(|&(_, s)| disk(s)).collect();
    assert_eq!(sync.nvme_read(&refs, &mut [0, 1, 2], &gpu, 0), (3, false));
    assert_eq!(fast.nvme_read(&refs, &mut [0, 1, 2], &gpu, 0), (3, false));
    // Blocks 0..3 are one run on the fast path, three on the synchronous one.
    let (f, s) = (fast.nvme_io_stats(), sync.nvme_io_stats());
    assert_eq!((f.restored_blocks, f.scatter_runs), (3, 1));
    assert_eq!((s.restored_blocks, s.scatter_runs), (3, 3));
    assert_eq!((s.spilled_blocks, s.gather_runs), (3, 3));
    assert_eq!((s.spill_wait_micros, s.flush_micros), (0, 0));
    for (i, &(b, _)) in plan.iter().enumerate() {
        assert_eq!(dump(&sync, &gpu, i as u32), dump(&sync, &gpu, b));
        assert_eq!(dump(&fast, &gpu, i as u32), dump(&fast, &gpu, b));
    }
}

#[test]
fn a_spill_returns_before_its_write_and_a_restore_waits_for_it() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 8);
    fill(&c, &gpu, 2, 0x5A);
    let want = dump(&c, &gpu, 2);
    store.set_gate(true);
    assert!(c.nvme_write(&[order(2, 0)], &gpu, 0).is_empty());
    assert!(store.recs.lock().unwrap().is_empty(), "write still queued");
    // The block is free to be reused the moment `nvme_write` returns.
    fill(&c, &gpu, 2, 0xEE);
    let opener = {
        let store = store.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            store.set_gate(false);
        })
    };
    // The read is only issued once the queued write is on disk.
    assert_eq!(c.nvme_read(&[disk(0)], &mut [5], &gpu, 0), (1, false));
    opener.join().unwrap();
    assert_eq!(dump(&c, &gpu, 5), want, "the bytes gathered at spill time");
    // That wait is accounted to the restore's flush, not to the spill path.
    let io = c.nvme_io_stats();
    assert!(io.flush_micros >= 40_000, "{io:?}");
    assert_eq!(io.spill_wait_micros, 0);
}

#[test]
fn a_failed_write_is_reported_later_and_never_restores() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 8);
    store.fail_writes.lock().unwrap().insert(7);
    fill(&c, &gpu, 1, 1);
    fill(&c, &gpu, 3, 3);
    // Slots 2 and 7 are separate runs: only the second fails.
    assert!(
        c.nvme_write(&[order(1, 2), order(3, 7)], &gpu, 0)
            .is_empty()
    );
    let before = dump(&c, &gpu, 6);
    assert_eq!(c.nvme_read(&[disk(7)], &mut [6], &gpu, 0), (0, true));
    assert_eq!(dump(&c, &gpu, 6), before, "nothing scattered");
    assert_eq!(c.nvme_take_failed(), vec![order(3, 7)]);
    assert!(c.nvme_take_failed().is_empty(), "reported once");
    assert_eq!(c.nvme_read(&[disk(2)], &mut [6], &gpu, 0), (1, false));
}

#[test]
fn a_slot_reissued_while_its_write_is_queued_keeps_the_later_record() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 8);
    fill(&c, &gpu, 1, 0x11);
    fill(&c, &gpu, 2, 0x22);
    fill(&c, &gpu, 3, 0x33);
    let (first, later) = (
        order(1, 4),
        SpillOrder {
            tag: 0xBEEF,
            ..order(2, 4)
        },
    );
    // Within one batch: the earlier order for the slot is dead.
    assert!(c.nvme_write(&[first, later], &gpu, 0).is_empty());
    let got = DiskRef {
        slot: 4,
        tag: 0xBEEF,
    };
    assert_eq!(c.nvme_read(&[got], &mut [6], &gpu, 0), (1, false));
    assert_eq!(dump(&c, &gpu, 6), dump(&c, &gpu, 2));
    // Across batches, the first still queued: the second spill waits for it
    // rather than race it to the slot.
    store.set_gate(true);
    let last = SpillOrder {
        tag: 0xF00D,
        ..order(3, 4)
    };
    assert!(c.nvme_write(&[first], &gpu, 0).is_empty());
    let before = store.writes.load(Ordering::Relaxed);
    let opener = {
        let store = store.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let held = store.writes.load(Ordering::Relaxed) == before;
            store.set_gate(false);
            held
        })
    };
    assert!(c.nvme_write(&[last], &gpu, 0).is_empty());
    assert!(
        opener.join().unwrap(),
        "the first write was still at the gate"
    );
    assert!(
        store.writes.load(Ordering::Relaxed) > before,
        "and it landed before the second spill was queued"
    );
    let got = DiskRef {
        slot: 4,
        tag: 0xF00D,
    };
    assert_eq!(c.nvme_read(&[got], &mut [7], &gpu, 0), (1, false));
    assert_eq!(dump(&c, &gpu, 7), dump(&c, &gpu, 3));
    assert!(c.nvme_take_failed().is_empty());
    // The spill path's wait for the writer is reported as such.
    let io = c.nvme_io_stats();
    assert!(io.spill_wait_micros >= 40_000, "{io:?}");
    assert!(io.spill_micros >= io.spill_wait_micros);
}

#[test]
fn a_leaked_staging_ring_fails_the_batch_instead_of_hanging() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 8);
    fill(&c, &gpu, 1, 0x11);
    // No free chunk and no write in flight: nothing will ever free one.
    let ring = std::mem::take(&mut c.nvme.as_mut().unwrap().fast.as_mut().unwrap().free);
    let orders = [order(1, 0), order(2, 1)];
    assert_eq!(c.nvme_write(&orders, &gpu, 0), orders.to_vec());
    assert_eq!(store.writes.load(Ordering::Relaxed), 0);
    // A restore attempts nothing (the records stay planned for a recompute).
    assert_eq!(c.nvme_read(&[disk(0)], &mut [5], &gpu, 0), (0, false));
    // With the ring back the tier works again.
    c.nvme.as_mut().unwrap().fast.as_mut().unwrap().free = ring;
    assert!(c.nvme_write(&orders[..1], &gpu, 0).is_empty());
    assert_eq!(c.nvme_read(&[disk(0)], &mut [5], &gpu, 0), (1, false));
    assert_eq!(dump(&c, &gpu, 5), dump(&c, &gpu, 1));
}

#[test]
fn more_blocks_than_the_staging_ring_spill_and_restore_in_order() {
    let gpu = MockGpuBackend::new();
    let n = 300u32; // ring = 8 chunks × 16 records
    let (mut c, _store) = fast_cache(&gpu, 2 * n as usize);
    let mut want = Vec::new();
    // Fragmented on both axes: every other block, slots shuffled by a stride.
    let plan: Vec<(u32, u32)> = (0..n).map(|i| (2 * i, (i * 7) % n)).collect();
    for &(b, _) in &plan {
        fill(&c, &gpu, b, (b % 251) as u8);
        want.push(dump(&c, &gpu, b));
    }
    let orders: Vec<SpillOrder> = plan.iter().map(|&(b, s)| order(b, s)).collect();
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    let refs: Vec<DiskRef> = plan.iter().map(|&(_, s)| disk(s)).collect();
    let mut blocks: Vec<u32> = (0..n).map(|i| 2 * i + 1).rev().collect();
    assert_eq!(
        c.nvme_read(&refs, &mut blocks, &gpu, 0),
        (n as usize, false)
    );
    for (i, &b) in blocks.iter().enumerate() {
        assert_eq!(dump(&c, &gpu, b), want[i], "path block {i}");
    }
    assert!(c.nvme_take_failed().is_empty());
}

#[test]
fn a_bad_record_stops_the_restore_at_the_verified_prefix() {
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 128);
    let n = 40u32; // three chunks
    for b in 0..n {
        fill(&c, &gpu, b, b as u8);
    }
    let orders: Vec<SpillOrder> = (0..n).map(|b| order(b, b)).collect();
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    assert!(settle(&mut c, &gpu).is_empty());
    store.recs.lock().unwrap().get_mut(&21).unwrap()[100] ^= 1;
    let refs: Vec<DiskRef> = (0..n).map(disk).collect();
    let mut blocks: Vec<u32> = (64..64 + n).collect();
    let before = dump(&c, &gpu, 64 + 21);
    assert_eq!(c.nvme_read(&refs, &mut blocks, &gpu, 0), (21, true));
    assert_eq!(dump(&c, &gpu, 64 + 20), dump(&c, &gpu, 20));
    assert_eq!(
        dump(&c, &gpu, 64 + 21),
        before,
        "the bad record is not scattered"
    );
    assert_eq!(dump(&c, &gpu, 64 + 30), before, "nor anything after it");
    // A record that was never written fails the same way.
    let gone = [disk(0), disk(999)];
    assert_eq!(c.nvme_read(&gone, &mut [70, 71], &gpu, 0), (1, true));
}

#[test]
fn release_joins_the_workers_and_frees_the_ring() {
    use atlas_core::scope::ModelResource;
    let gpu = MockGpuBackend::new();
    let (mut c, store) = fast_cache(&gpu, 8);
    assert_eq!(
        PagedKvCache::nvme_staging_bytes(c.nvme_record_bytes(), true),
        8 * 16 * c.nvme_record_bytes()
    );
    fill(&c, &gpu, 1, 9);
    assert!(c.nvme_write(&[order(1, 0)], &gpu, 0).is_empty());
    c.release(&gpu).unwrap();
    assert!(!c.nvme_attached());
    // The queued write still ran before the workers exited.
    assert_eq!(store.writes.load(Ordering::Relaxed), 1);
}

#[test]
fn record_size_is_predictable_before_the_cache_exists() {
    let gpu = MockGpuBackend::new();
    let c = glm_cache(&gpu, 4);
    let index = Some(SparseIndexCacheConfig::bf16(4, 128));
    assert_eq!(
        PagedKvCache::nvme_record_bytes_for(&c.config, true, index),
        c.nvme_record_bytes()
    );
    // Separate V pools and no index: both sides of every layer, nothing else.
    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 3,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let predicted = PagedKvCache::nvme_record_bytes_for(&cfg, false, None);
    let plain = PagedKvCache::new(cfg, 4, &gpu).unwrap();
    assert_eq!(predicted, plain.nvme_record_bytes());
}
