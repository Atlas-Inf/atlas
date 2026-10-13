// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill records: layout (GLM MLA: V aliases K, pooled index, no tails),
//! write → read round trip across blocks, and trailer verification.

use super::*;
use crate::prefix_cache::{DiskRef, SpillOrder};

/// GLM-5.3 shape: one 512-wide FP8-G128 latent head, V aliasing K, a BF16
/// four-token pooled index of width 128 with per-block raw tails.
pub(super) fn glm_cache(gpu: &MockGpuBackend, blocks: usize) -> PagedKvCache {
    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 2,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut c = PagedKvCache::new_with_v_alias(cfg, blocks, gpu, true).unwrap();
    c.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    c
}

/// Every spilled region of `block`, in record order (K, index values).
fn regions(c: &PagedKvCache, block: u32) -> Vec<(DevicePtr, usize)> {
    (0..c.num_layers())
        .flat_map(|l| {
            let k = c.k_block_stride_bytes_for_layer(l);
            let v = c.sparse_index_block_stride_bytes(l);
            [
                (c.k_cache_ptr(l, block), k),
                (c.sparse_index_pool_ptr(l).offset(block as usize * v), v),
            ]
        })
        .collect()
}

pub(super) fn fill(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32, seed: u8) {
    for (i, (ptr, len)) in regions(c, block).into_iter().enumerate() {
        let bytes: Vec<u8> = (0..len)
            .map(|j| seed.wrapping_add((i * 31 + j) as u8))
            .collect();
        gpu.copy_h2d(&bytes, ptr).unwrap();
    }
}

pub(super) fn dump(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (ptr, len) in regions(c, block) {
        let mut b = vec![0u8; len];
        gpu.copy_d2h(ptr, &mut b).unwrap();
        out.extend(b);
    }
    out
}

fn attach(c: &mut PagedKvCache, gpu: &MockGpuBackend) {
    let store = atlas_tier::MemSwapStore::new(c.nvme_record_bytes());
    c.attach_nvme_spill(Box::new(store), gpu).unwrap();
}

#[test]
fn glm_record_is_latent_plus_pooled_index_without_tails() {
    let gpu = MockGpuBackend::new();
    let c = glm_cache(&gpu, 4);
    // Per layer: 16×512 FP8 + 16×512/128 f32 scales = 8448, index 4×128×2 = 1024.
    // No V (aliased) and no raw tails; + 24 B trailer, padded to 4 KiB.
    let payload: usize = 2 * (8448 + 1024);
    assert_eq!(c.nvme_record_bytes(), (payload + 24).next_multiple_of(4096));
}

#[test]
fn spill_and_restore_round_trip_into_other_blocks() {
    let gpu = MockGpuBackend::new();
    let mut c = glm_cache(&gpu, 8);
    attach(&mut c, &gpu);
    fill(&c, &gpu, 1, 7);
    fill(&c, &gpu, 2, 99);
    let (want1, want2) = (dump(&c, &gpu, 1), dump(&c, &gpu, 2));
    let orders = [
        SpillOrder {
            block: 1,
            slot: 5,
            tag: 0xA,
        },
        SpillOrder {
            block: 2,
            slot: 0,
            tag: 0xB,
        },
    ];
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    fill(&c, &gpu, 1, 0);
    let disk = [DiskRef { slot: 5, tag: 0xA }, DiskRef { slot: 0, tag: 0xB }];
    assert_eq!(c.nvme_read(&disk, &mut [6, 3], &gpu, 0), (2, false));
    assert_eq!(dump(&c, &gpu, 6), want1);
    assert_eq!(dump(&c, &gpu, 3), want2);
}

#[test]
fn wrong_tag_or_missing_record_fails_without_scattering() {
    let gpu = MockGpuBackend::new();
    let mut c = glm_cache(&gpu, 8);
    attach(&mut c, &gpu);
    fill(&c, &gpu, 1, 3);
    let good = SpillOrder {
        block: 1,
        slot: 0,
        tag: 42,
    };
    assert!(c.nvme_write(&[good], &gpu, 0).is_empty());
    let before = dump(&c, &gpu, 4);
    // Stale record (tag from another spill): verified prefix = 0, failed.
    let stale = [DiskRef { slot: 0, tag: 43 }];
    assert_eq!(c.nvme_read(&stale, &mut [4], &gpu, 0), (0, true));
    assert_eq!(dump(&c, &gpu, 4), before, "nothing scattered");
    // A good record followed by a never-written one: 1 restored, then failed.
    let run = [DiskRef { slot: 0, tag: 42 }, DiskRef { slot: 9, tag: 1 }];
    assert_eq!(c.nvme_read(&run, &mut [4, 5], &gpu, 0), (1, true));
}

#[test]
fn checksum_catches_payload_corruption() {
    let payload = vec![5u8; 1000];
    let mut rec = vec![0u8; 4096];
    rec[..1000].copy_from_slice(&payload);
    super::super::nvme_spill::stamp(&mut rec, 1000, 77);
    assert!(super::super::nvme_spill::verify(&rec, 1000, 77));
    rec[500] ^= 1;
    assert!(!super::super::nvme_spill::verify(&rec, 1000, 77));
    rec[500] ^= 1;
    // The unaligned tail (1000 = 31·32 + 8) is covered too.
    rec[996] ^= 0x80;
    assert!(!super::super::nvme_spill::verify(&rec, 1000, 77));
    rec[996] ^= 0x80;
    assert!(
        !super::super::nvme_spill::verify(&rec, 1000, 78),
        "tag bound"
    );
}

#[test]
fn unattached_cache_refuses_everything() {
    let gpu = MockGpuBackend::new();
    let mut c = glm_cache(&gpu, 4);
    assert!(!c.nvme_attached());
    let o = SpillOrder {
        block: 1,
        slot: 0,
        tag: 1,
    };
    assert_eq!(c.nvme_write(&[o], &gpu, 0), vec![o]);
    assert_eq!(
        c.nvme_read(&[DiskRef { slot: 0, tag: 1 }], &mut [2], &gpu, 0),
        (0, false)
    );
}

#[test]
fn store_with_wrong_record_size_is_rejected() {
    let gpu = MockGpuBackend::new();
    let mut c = glm_cache(&gpu, 4);
    let store = atlas_tier::MemSwapStore::new(c.nvme_record_bytes() + 4096);
    assert!(c.attach_nvme_spill(Box::new(store), &gpu).is_err());
}

#[test]
fn consecutive_slots_group_into_runs_in_either_direction() {
    use super::super::nvme_spill::run_layout as runs;
    // A leaf-first spilled chain restores in DEScending slot order.
    assert_eq!(
        runs(&[9, 8, 7, 3, 4, 20]),
        vec![(0, 3, 7), (3, 2, 3), (5, 1, 20)]
    );
    assert_eq!(runs(&[5]), vec![(0, 1, 5)]);
    assert!(runs(&[]).is_empty());
    assert_eq!(runs(&[1, 1]), vec![(0, 1, 1), (1, 1, 1)]);
}

/// Counts ranged reads so the test can see a run was ONE store call.
struct CountingStore {
    inner: atlas_tier::MemSwapStore,
    ranged: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl atlas_tier::SwapStore for CountingStore {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, slot: usize, bytes: &[u8]) -> anyhow::Result<()> {
        self.inner.write_record(slot, bytes)
    }
    fn read_record(&self, slot: usize, out: &mut [u8]) -> anyhow::Result<()> {
        self.inner.read_record(slot, out)
    }
    fn read_records(&self, first: usize, out: &mut [u8]) -> anyhow::Result<()> {
        self.ranged
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.read_records(first, out)
    }
}

#[test]
fn descending_chain_restores_with_one_ranged_read() {
    let gpu = MockGpuBackend::new();
    let mut c = glm_cache(&gpu, 8);
    let ranged = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let store = CountingStore {
        inner: atlas_tier::MemSwapStore::new(c.nvme_record_bytes()),
        ranged: ranged.clone(),
    };
    c.attach_nvme_spill(Box::new(store), &gpu).unwrap();
    // Leaf-first spill of a 3-block chain [1, 2, 3]: slots 0, 1, 2 for 3, 2, 1.
    let mut want = Vec::new();
    for (b, seed) in [(1u32, 11u8), (2, 22), (3, 33)] {
        fill(&c, &gpu, b, seed);
        want.push(dump(&c, &gpu, b));
    }
    let orders = [
        SpillOrder {
            block: 3,
            slot: 0,
            tag: 3,
        },
        SpillOrder {
            block: 2,
            slot: 1,
            tag: 2,
        },
        SpillOrder {
            block: 1,
            slot: 2,
            tag: 1,
        },
    ];
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    // Restore in path order (root first) = slots 2, 1, 0.
    let disk = [
        DiskRef { slot: 2, tag: 1 },
        DiskRef { slot: 1, tag: 2 },
        DiskRef { slot: 0, tag: 3 },
    ];
    assert_eq!(c.nvme_read(&disk, &mut [5, 6, 7], &gpu, 0), (3, false));
    assert_eq!(ranged.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(dump(&c, &gpu, 5), want[0]);
    assert_eq!(dump(&c, &gpu, 6), want[1]);
    assert_eq!(dump(&c, &gpu, 7), want[2]);
}
