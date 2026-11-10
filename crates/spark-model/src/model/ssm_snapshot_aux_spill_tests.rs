// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for a spilled snapshot's aux layer state: it travels to the tier
//! with the recurrent state and comes back with it, so an aux-carrying model
//! (qwen4_exp: PLE history and carry, QSA indexer keys) can restore a
//! faulted-in snapshot instead of declining it at the restore gate.

use super::*;
use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

/// Small Marconi-only pool, the shape of the other snapshot test modules: a
/// 3-layer blob is 144 B, so a few hundred bytes of aux span several chunks.
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, /*h_bytes*/ 32, /*conv_bytes*/ 16, layers, /*decode_ring*/ 0,
        /*decode_max_seqs*/ 0, /*hidden_bytes*/ 8, gpu,
    )
    .unwrap()
}

/// Aux of several layers, one longer than a chunk, one empty, every byte
/// distinct per layer so a misplaced chunk would show.
fn aux() -> Vec<(u32, Vec<u8>)> {
    vec![
        (2, (0..300u32).map(|i| (i % 251) as u8).collect()),
        (5, vec![0xA5; 10]),
        (7, Vec::new()),
        (11, (0..150u32).map(|i| (i % 7) as u8 + 0x40).collect()),
    ]
}

fn aux_of(p: &SsmSnapshotPool, slot: usize) -> Option<Vec<(u32, Vec<u8>)>> {
    p.with_aux(slot, |blobs| Ok(blobs.to_vec()))
        .map(Result::unwrap)
}

/// The defect: a spilled snapshot came back without its aux, so the restore
/// gate (`requires_aux_state` && !`has_aux`) declined every disk restore.
#[test]
fn aux_travels_with_a_spilled_snapshot() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = MemBlobStore::new(0);
    let key = 0x5EED_0001;
    p.set_aux(1, aux());

    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    p.free(1);
    assert!(
        store.len() > 1,
        "the aux must be in the tier, not only in host RAM"
    );

    assert!(p.fault_in_slot(2, key, &store, &gpu, 0).unwrap());
    assert!(p.has_aux(2), "a faulted-in snapshot must carry its aux");
    assert_eq!(aux_of(&p, 2).unwrap(), aux());
}

/// A model without aux (GLM, Holo) writes exactly the state blob, as before,
/// and a faulted-in slot has no aux.
#[test]
fn no_aux_spills_only_the_state_blob() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = MemBlobStore::new(0);
    assert!(p.spill_slot(1, 0x77, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 1);
    assert_eq!(store.bytes_resident(), p.spill_blob_bytes());
    assert!(p.fault_in_slot(2, 0x77, &store, &gpu, 0).unwrap());
    assert!(!p.has_aux(2));
}

/// The tier dropped one aux chunk (a disk cap evicts records one at a time):
/// that is a MISS, so the caller reaps the key, never a restore with part of
/// the aux.
#[test]
fn a_missing_aux_chunk_is_a_miss() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = MemBlobStore::new(0);
    let key = 0x5EED_0002;
    p.set_aux(1, aux());
    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    p.free(1);

    store.remove(aux_key(key, 1));
    assert!(!p.fault_in_slot(2, key, &store, &gpu, 0).unwrap());
}

/// Retiring a spilled key removes its aux chunks too: they would otherwise
/// hold tier budget and never be read again.
#[test]
fn removing_a_spilled_key_removes_its_aux() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = MemBlobStore::new(0);
    let key = 0x5EED_0003;
    p.set_aux(1, aux());
    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());

    p.remove_spilled(&store, key);
    assert_eq!(store.len(), 0);
    assert_eq!(store.bytes_resident(), 0);
}

/// A key spilled again (after a fault-in promoted it) with less aux keeps no
/// stale chunks from the first spill.
#[test]
fn respilling_a_key_drops_its_stale_chunks() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = MemBlobStore::new(0);
    let key = 0x5EED_0004;
    p.set_aux(1, aux());
    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    let first = store.len();

    let small = vec![(2, vec![1u8; 8])];
    p.set_aux(1, small.clone());
    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 2, "one state blob and one aux chunk");
    assert!(first > store.len());
    assert!(p.fault_in_slot(2, key, &store, &gpu, 0).unwrap());
    assert_eq!(aux_of(&p, 2).unwrap(), small);

    // And a re-spill with no aux leaves only the state blob.
    p.free(1);
    assert!(p.spill_slot(1, key, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 1);
}

/// A tier that takes the first `n` puts and refuses the rest.
struct Refusing {
    inner: MemBlobStore,
    left: std::sync::atomic::AtomicUsize,
}

impl SnapshotBlobStore for Refusing {
    fn put(&self, key: u64, bytes: &[u8]) -> anyhow::Result<bool> {
        use std::sync::atomic::Ordering;
        if self
            .left
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
        {
            return Ok(false);
        }
        self.inner.put(key, bytes)
    }
    fn get(&self, key: u64, out: &mut [u8]) -> anyhow::Result<bool> {
        self.inner.get(key, out)
    }
    fn remove(&self, key: u64) {
        self.inner.remove(key)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn bytes_resident(&self) -> usize {
        self.inner.bytes_resident()
    }
}

/// A tier that refuses part of the aux refuses the whole spill and keeps
/// nothing of it: the caller then drops the snapshot as for any refusal.
#[test]
fn a_refused_aux_chunk_refuses_the_spill() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 4, 3);
    let store = Refusing {
        inner: MemBlobStore::new(0),
        left: 2.into(),
    };
    p.set_aux(1, aux());
    assert!(!p.spill_slot(1, 0x5EED_0005, &store, &gpu, 0).unwrap());
    assert_eq!(store.len(), 0);
    assert!(!p.fault_in_slot(2, 0x5EED_0005, &store, &gpu, 0).unwrap());
}
