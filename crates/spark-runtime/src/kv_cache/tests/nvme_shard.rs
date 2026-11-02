// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill records under a latent shard (`ATLAS_GLM_KV_SHARD=1`): one lane
//! per slot class, each rank writing and reading only the latents it stores
//! (at their local slots) plus every block's index rows, on both I/O paths.

use std::sync::{Arc, Mutex};

use super::*;
use crate::prefix_cache::{DiskRef, SpillOrder};
use atlas_core::scope::ModelResource;

const LAYERS: usize = 2;

fn config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: LAYERS,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

fn spec(rank: usize) -> LatentShardSpec {
    LatentShardSpec {
        rank,
        world: 2,
        scratch_bytes: 4096,
        view_blocks: 8,
        write_rows: 4,
        lane: false,
    }
}

/// Rank `rank`'s GLM-5.3-shaped cache (FP8-G128 latent, V aliasing K, BF16
/// pooled index), with the tier's lanes attached when `path` is set.
fn sharded(gpu: &MockGpuBackend, rank: usize, blocks: usize, fast: Option<bool>) -> PagedKvCache {
    let mut c = PagedKvCache::new_latent_sharded(config(), blocks, gpu, spec(rank)).unwrap();
    c.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    if let Some(fast) = fast {
        for class in 0..c.nvme_classes() {
            let store = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(class));
            if fast {
                c.attach_nvme_fast(Arc::new(Mutex::new(store)), gpu)
            } else {
                c.attach_nvme_spill(Box::new(store), gpu)
            }
            .unwrap();
        }
    }
    c
}

/// Every region this rank holds for `block`: the latents when it owns the
/// block (at its local slot), then each layer's index rows.
fn regions(c: &PagedKvCache, block: u32) -> Vec<(DevicePtr, usize)> {
    let shard = c.latent_shard().unwrap();
    let mut out = Vec::new();
    for l in 0..c.num_layers() {
        let k = c.k_block_stride_bytes_for_layer(l);
        if let Some(slot) = shard.local_slot(block) {
            out.push((c.latent_pool_ptr(l).offset(slot as usize * k), k));
        }
        let ix = c.sparse_index_block_stride_bytes(l);
        out.push((c.sparse_index_pool_ptr(l).offset(block as usize * ix), ix));
    }
    out
}

fn fill(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32, seed: u8) {
    for (i, (ptr, len)) in regions(c, block).into_iter().enumerate() {
        let bytes: Vec<u8> = (0..len)
            .map(|j| seed.wrapping_add((i * 31 + j) as u8))
            .collect();
        gpu.copy_h2d(&bytes, ptr).unwrap();
    }
}

fn dump(c: &PagedKvCache, gpu: &MockGpuBackend, block: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (ptr, len) in regions(c, block) {
        let mut b = vec![0u8; len];
        gpu.copy_d2h(ptr, &mut b).unwrap();
        out.extend(b);
    }
    out
}

/// The slot the tree would hand logical block `logical` (class = residue).
fn slot_for(logical: usize, k: u32) -> u32 {
    2 * k + (logical % 2) as u32
}

fn tag(slot: u32) -> u64 {
    0x5000 + u64::from(slot)
}

#[test]
fn each_rank_has_a_full_lane_and_an_index_only_lane() {
    let gpu = MockGpuBackend::new();
    let index = SparseIndexCacheConfig::bf16(4, 128);
    let unsharded = PagedKvCache::nvme_geometry_for(&config(), true, Some(index), false);
    assert_eq!(unsharded.peer, None);
    assert_eq!(unsharded.classes(), 1);
    // Per layer: latent 16×528 = 8448, index 4×128×2 = 1024; + 24 B trailer.
    let full = (LAYERS * (8448 + 1024) + 24).next_multiple_of(4096);
    let index_only = (LAYERS * 1024 + 24).next_multiple_of(4096);
    assert_eq!(unsharded.own, full);
    for rank in 0..2 {
        let c = sharded(&gpu, rank, 8, None);
        let g = c.nvme_geometry();
        assert_eq!(c.nvme_classes(), 2);
        assert_eq!(c.nvme_class_record_bytes(rank), full, "rank {rank} own");
        assert_eq!(c.nvme_class_record_bytes(1 - rank), index_only);
        assert_eq!((g.own, g.peer), (full, Some(index_only)));
        assert_eq!(g.class_bytes(rank, rank), full);
        assert_eq!(g.class_bytes(1 - rank, rank), index_only);
        assert_eq!(g.row_bytes(), full + index_only);
        assert_eq!(
            PagedKvCache::nvme_geometry_for(&config(), true, Some(index), true),
            g,
            "sized before the cache exists exactly as after"
        );
        assert_eq!(c.nvme_block_record_bytes(), (full + index_only) / 2);
    }
}

/// A 6-block chain (logical 0..6, ids drawn at their logical index) spills
/// and comes back into other blocks of the same residues, byte for byte, on
/// both ranks and both I/O paths.
#[test]
fn both_ranks_round_trip_their_halves_on_both_paths() {
    for fast in [false, true] {
        for rank in 0..2 {
            let gpu = MockGpuBackend::new();
            let mut c = sharded(&gpu, rank, 24, Some(fast));
            assert!(c.nvme_attached());
            let chain: Vec<u32> = (0..6).map(|l| c.alloc_block_at(l).unwrap()).collect();
            let mut want = Vec::new();
            for (l, &b) in chain.iter().enumerate() {
                fill(&c, &gpu, b, 0x30 + l as u8);
                want.push(dump(&c, &gpu, b));
            }
            // Leaf first, as the tree spills a chain.
            let orders: Vec<SpillOrder> = (0..6)
                .rev()
                .map(|l| {
                    let slot = slot_for(l, (5 - l as u32) / 2);
                    SpillOrder {
                        block: chain[l],
                        slot,
                        tag: tag(slot),
                    }
                })
                .collect();
            assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
            for &b in &chain {
                fill(&c, &gpu, b, 0xEE);
            }
            let disk: Vec<DiskRef> = (0..6)
                .map(|l| {
                    let o = orders.iter().find(|o| o.block == chain[l]).unwrap();
                    DiskRef {
                        slot: o.slot,
                        tag: o.tag,
                    }
                })
                .collect();
            let mut blocks: Vec<u32> = (0..6).map(|l| c.alloc_block_at(l).unwrap()).collect();
            assert_eq!(
                c.nvme_read(&disk, &mut blocks, &gpu, 0),
                (6, false),
                "fast={fast} rank={rank}"
            );
            for (l, &b) in blocks.iter().enumerate() {
                assert_eq!(b as usize % 2, l % 2, "block keeps its logical residue");
                assert_eq!(
                    dump(&c, &gpu, b),
                    want[l],
                    "fast={fast} rank={rank} block {l}"
                );
            }
            let io = c.nvme_io_stats();
            assert_eq!(
                (io.fast, io.spilled_blocks, io.restored_blocks),
                (fast, 6, 6)
            );
            assert!(c.nvme_take_failed().is_empty());
            c.release(&gpu).unwrap();
        }
    }
}

/// The peer's blocks carry no latents on this rank: restoring one leaves
/// the whole latent pool alone, and its record is the index-only size.
#[test]
fn a_peer_block_restores_index_rows_only() {
    for fast in [false, true] {
        let gpu = MockGpuBackend::new();
        let mut c = sharded(&gpu, 0, 8, Some(fast));
        let peer = c.alloc_block_at(1).unwrap();
        fill(&c, &gpu, peer, 9);
        let want = dump(&c, &gpu, peer);
        let order = SpillOrder {
            block: peer,
            slot: 1,
            tag: 77,
        };
        assert!(c.nvme_write(&[order], &gpu, 0).is_empty());
        let stride = c.k_block_stride_bytes_for_layer(0);
        let local = c.latent_shard().unwrap().local_blocks;
        for l in 0..LAYERS {
            gpu.memset(c.latent_pool_ptr(l), 0xAB, local * stride)
                .unwrap();
        }
        let mut target = [c.alloc_block_at(1).unwrap()];
        let disk = [DiskRef { slot: 1, tag: 77 }];
        assert_eq!(c.nvme_read(&disk, &mut target, &gpu, 0), (1, false));
        assert_eq!(dump(&c, &gpu, target[0]), want);
        for l in 0..LAYERS {
            let pool = gpu.read_alloc(c.latent_pool_ptr(l)).unwrap();
            assert!(pool.iter().all(|&b| b == 0xAB), "fast={fast} layer {l}");
        }
    }
}

/// Owned blocks two ids apart are neighbours in their lane: one pitched copy
/// per region (local slots for the latents, a doubled pitch for the index).
#[test]
fn a_run_of_owned_blocks_moves_as_pitched_copies() {
    let gpu = MockGpuBackend::new();
    let mut c = sharded(&gpu, 0, 24, Some(true));
    let blocks = [4u32, 6, 8, 10];
    let mut want = Vec::new();
    for &b in &blocks {
        fill(&c, &gpu, b, b as u8);
        want.push(dump(&c, &gpu, b));
    }
    // Class-0 slots 0, 2, 4, 6: lane slots 0..4, one run.
    let orders: Vec<SpillOrder> = blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| SpillOrder {
            block: b,
            slot: 2 * i as u32,
            tag: tag(2 * i as u32),
        })
        .collect();
    let (pitched, plain) = (gpu.host_pitched_count(), gpu.d2h_async_count());
    assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
    assert_eq!(gpu.host_pitched_count() - pitched, 2 * LAYERS);
    assert_eq!(gpu.d2h_async_count(), plain, "no per-block copies");
    let disk: Vec<DiskRef> = orders
        .iter()
        .map(|o| DiskRef {
            slot: o.slot,
            tag: o.tag,
        })
        .collect();
    let mut targets = [16u32, 12, 14, 18];
    assert_eq!(c.nvme_read(&disk, &mut targets, &gpu, 0), (4, false));
    assert_eq!(targets, [12, 14, 16, 18], "sorted within the residue");
    for (i, &b) in targets.iter().enumerate() {
        assert_eq!(dump(&c, &gpu, b), want[i]);
    }
}

/// A block whose residue is not its slot's class cannot take that record:
/// the write is reported failed, and a restore stops in front of it.
#[test]
fn a_block_off_its_slot_class_is_refused() {
    for fast in [false, true] {
        let gpu = MockGpuBackend::new();
        let mut c = sharded(&gpu, 1, 8, Some(fast));
        let (even, odd) = (c.alloc_block_at(0).unwrap(), c.alloc_block_at(1).unwrap());
        fill(&c, &gpu, even, 1);
        fill(&c, &gpu, odd, 2);
        let good = SpillOrder {
            block: even,
            slot: 0,
            tag: 5,
        };
        let bad = SpillOrder {
            block: odd,
            slot: 2,
            tag: 6,
        };
        assert_eq!(c.nvme_write(&[good, bad], &gpu, 0), vec![bad]);
        let disk = [DiskRef { slot: 0, tag: 5 }, DiskRef { slot: 2, tag: 6 }];
        let mut targets = [c.alloc_block_at(0).unwrap(), c.alloc_block_at(1).unwrap()];
        assert_eq!(c.nvme_read(&disk, &mut targets, &gpu, 0), (1, true));
        assert!(c.nvme_take_failed().is_empty());
    }
}

/// A failure in either lane bounds the restored prefix at its position,
/// whatever the other lane read beyond it.
#[test]
fn the_first_failure_in_any_lane_ends_the_restore() {
    for fast in [false, true] {
        let gpu = MockGpuBackend::new();
        let mut c = sharded(&gpu, 0, 16, Some(fast));
        let chain: Vec<u32> = (0..4).map(|l| c.alloc_block_at(l).unwrap()).collect();
        let orders: Vec<SpillOrder> = chain
            .iter()
            .enumerate()
            .map(|(l, &b)| SpillOrder {
                block: b,
                slot: l as u32,
                tag: tag(l as u32),
            })
            .collect();
        assert!(c.nvme_write(&orders, &gpu, 0).is_empty());
        let disk = |bad: usize| -> Vec<DiskRef> {
            (0..4u32)
                .map(|s| DiskRef {
                    slot: s,
                    tag: if s as usize == bad { 1 } else { tag(s) },
                })
                .collect()
        };
        for bad in 0..4 {
            let mut targets: Vec<u32> = (0..4).map(|l| c.alloc_block_at(l).unwrap()).collect();
            assert_eq!(
                c.nvme_read(&disk(bad), &mut targets, &gpu, 0),
                (bad, true),
                "fast={fast} bad={bad}"
            );
            c.free_blocks(&targets);
        }
        let mut targets: Vec<u32> = (0..4).map(|l| c.alloc_block_at(l).unwrap()).collect();
        assert_eq!(c.nvme_read(&disk(9), &mut targets, &gpu, 0), (4, false));
    }
}

#[test]
fn every_class_needs_its_store_and_one_io_path() {
    let gpu = MockGpuBackend::new();
    let mut c = sharded(&gpu, 0, 8, None);
    let own = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(0));
    c.attach_nvme_spill(Box::new(own), &gpu).unwrap();
    assert!(!c.nvme_attached(), "one of two classes");
    // Class 1 takes the index-only record, on the same path.
    let wrong = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(0));
    assert!(c.attach_nvme_spill(Box::new(wrong), &gpu).is_err());
    let peer = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(1));
    assert!(
        c.attach_nvme_fast(Arc::new(Mutex::new(peer)), &gpu)
            .is_err()
    );
    let peer = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(1));
    c.attach_nvme_spill(Box::new(peer), &gpu).unwrap();
    assert!(c.nvme_attached());
    let extra = atlas_tier::MemSwapStore::new(c.nvme_class_record_bytes(1));
    assert!(c.attach_nvme_spill(Box::new(extra), &gpu).is_err());
    c.release(&gpu).unwrap();
}

#[test]
fn an_unsharded_cache_keeps_its_one_lane() {
    let gpu = MockGpuBackend::new();
    let c = super::nvme_spill::glm_cache(&gpu, 4);
    assert_eq!(c.nvme_classes(), 1);
    let g = c.nvme_geometry();
    assert_eq!((g.own, g.peer), (c.nvme_record_bytes(), None));
    assert_eq!(c.nvme_class_record_bytes(0), c.nvme_record_bytes());
    assert_eq!(c.nvme_block_record_bytes(), c.nvme_record_bytes());
}
