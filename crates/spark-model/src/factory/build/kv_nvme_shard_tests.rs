// SPDX-License-Identifier: AGPL-3.0-only

//! The NVMe tier's sizing under a latent shard (`ATLAS_GLM_KV_SHARD=1`): two
//! record classes per rank, slots per class from the budget, the rank word,
//! the host reserve and the per-class record files (`kv_nvme.rs`). The
//! unsharded sizing is pinned alongside, unchanged.

use super::tests::{DISK, FAST, disk_scratch_dir, tier};
use super::*;

/// GLM-5.3's record per rank, unsharded.
pub(super) const GLM: NvmeGeometry = NvmeGeometry {
    own: 106_496,
    peer: None,
};
/// The same under a latent shard: the peer's blocks keep their index rows.
pub(super) const GLM_SHARD: NvmeGeometry = NvmeGeometry {
    own: 106_496,
    peer: Some(12_288),
};

#[test]
fn a_shard_buys_slots_of_both_classes_within_the_budget() {
    let budget = 24u64 << 30;
    let row = 106_496 + 12_288;
    let per_class = slots_per_class(budget, GLM_SHARD).unwrap();
    assert_eq!(per_class as u64, budget / row as u64);
    assert!(per_class as u64 * row as u64 <= budget, "both files fit");
    // ~1.8x the blocks per GiB of the unsharded tier (two per slot index).
    let plain = slots_per_class(budget, GLM).unwrap();
    assert_eq!(plain, max_slots(budget, 106_496).unwrap(), "unchanged");
    let gain = 2.0 * per_class as f64 / plain as f64;
    assert!((1.75..1.85).contains(&gain), "{gain}");
    // A budget below one slot of each class is refused.
    assert!(slots_per_class(row as u64 - 1, GLM_SHARD).is_err());
    assert!(slots_per_class(row as u64, GLM_SHARD).is_ok());
}

#[test]
fn the_rank_word_tells_a_sharded_tier_from_a_plain_one() {
    let cfg = tier("24", true);
    let plain = word_for(Some(&cfg), GLM, DISK, true).unwrap();
    // Unsharded: the very word this rank sent before classes existed.
    let slots = max_slots(cfg.budget_bytes, 106_496).unwrap();
    assert_eq!(plain, rank_fingerprint(slots, 106_496, DISK, FAST));
    let sharded = word_for(Some(&cfg), GLM_SHARD, DISK, true).unwrap();
    assert_ne!(plain, sharded);
    assert!(sharded != TIER_OFF && sharded != FAILED_RANK);
    // Both ranks of the pair size the same records: one word.
    assert_eq!(
        sharded,
        word_for(Some(&cfg), GLM_SHARD, DISK, true).unwrap()
    );
    assert_eq!(word_for(None, GLM_SHARD, None, true).unwrap(), TIER_OFF);
}

#[test]
fn host_reserve_covers_staging_index_and_the_ssm_tier() {
    use spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK;
    let record = GLM; // GLM-5.3, per rank
    assert_eq!(reserve_for(None, record, 1 << 30), 0, "tier off: nothing");
    let cfg = |fast| NvmeKvConfig {
        dir: PathBuf::from("/nvme"),
        budget_bytes: 24 << 30,
        fast,
        keep: false,
    };
    let slots = (24usize << 30) / record.own;
    let index = slots * NVME_HOST_BYTES_PER_BLOCK;
    assert_eq!(
        reserve_for(Some(&cfg(false)), record, 0),
        32 * record.own + index
    );
    assert_eq!(
        reserve_for(Some(&cfg(true)), record, 5000),
        128 * record.own + index + 5000
    );
    // 24 GiB of records costs about 148 MiB of host RAM for the index alone.
    assert!((140 << 20..150 << 20).contains(&index), "{index}");
    // A budget below one record is refused at attach; the reserve stays 0.
    let tiny = NvmeKvConfig {
        budget_bytes: 100,
        ..cfg(true)
    };
    assert_eq!(reserve_for(Some(&tiny), record, 5000), 0);
    // Sharded: both lanes' staging and every slot of both classes indexed.
    let per_class = (24usize << 30) / (106_496 + 12_288);
    assert_eq!(
        reserve_for(Some(&cfg(true)), GLM_SHARD, 0),
        128 * (106_496 + 12_288) + 2 * per_class * NVME_HOST_BYTES_PER_BLOCK
    );
}

fn sharded_glm_kv(gpu: &spark_runtime::gpu::mock::MockGpuBackend, rank: usize) -> PagedKvCache {
    use spark_runtime::kv_cache::{
        KvCacheConfig, KvCacheDtype, LatentShardSpec, SparseIndexCacheConfig,
    };
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
    let spec = LatentShardSpec {
        rank,
        world: 2,
        scratch_bytes: 4096,
        view_blocks: 8,
        write_rows: 4,
        lane: false,
    };
    let mut kv = PagedKvCache::new_latent_sharded(cfg, 8, gpu, spec).unwrap();
    kv.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    kv
}

/// Both ranks of a shard attach one unlinked record file per class, enable
/// the tree's classes, and move a block of each class through the real files.
#[test]
fn a_sharded_rank_attaches_a_file_per_class() {
    use spark_runtime::prefix_cache::{DiskRef, NvmePrefixTier, SpillOrder};
    for (rank, fast) in [(0, true), (1, false)] {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut kv = sharded_glm_kv(&gpu, rank);
        let tree = spark_runtime::radix_tree::RadixTree::new();
        let Some(dir) = disk_scratch_dir(&format!("shard{rank}")) else {
            return;
        };
        let cfg = NvmeKvConfig {
            dir: dir.clone(),
            budget_bytes: 64 << 20,
            fast,
            keep: false,
        };
        let geometry = kv.nvme_geometry();
        assert_eq!(geometry.classes(), 2);
        let slots = setup_local(Some(cfg), rank, geometry, &mut kv, &tree, &gpu).unwrap();
        let per_class = slots_per_class(64 << 20, geometry).unwrap();
        assert_eq!(slots, 2 * per_class);
        assert!(kv.nvme_attached());
        assert_eq!(tree.nvme_stats().max_slots, slots);
        assert!(!tree.enable_classes(1, 2), "enabled once, with classes");
        #[cfg(unix)]
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "unlinked");
        // An own block (residue = rank) and a peer block, out and back.
        let own = kv.alloc_block_at(rank).unwrap();
        let peer = kv.alloc_block_at(rank + 1).unwrap();
        let ix = kv.sparse_index_block_stride_bytes(0);
        let pattern: Vec<u8> = (0..ix).map(|i| (i % 251) as u8).collect();
        for b in [own, peer] {
            let at = kv.sparse_index_pool_ptr(0).offset(b as usize * ix);
            gpu.copy_h2d(&pattern, at).unwrap();
        }
        let order = |block: u32| SpillOrder {
            block,
            slot: block % 2,
            tag: 40 + u64::from(block),
        };
        assert!(
            kv.nvme_write(&[order(own), order(peer)], &gpu, 0)
                .is_empty()
        );
        let disk = |block: u32| DiskRef {
            slot: block % 2,
            tag: 40 + u64::from(block),
        };
        let mut back = [
            kv.alloc_block_at(rank).unwrap(),
            kv.alloc_block_at(rank + 1).unwrap(),
        ];
        assert_eq!(
            kv.nvme_read(&[disk(own), disk(peer)], &mut back, &gpu, 0),
            (2, false)
        );
        for b in back {
            let mut got = vec![0u8; ix];
            let at = kv.sparse_index_pool_ptr(0).offset(b as usize * ix);
            gpu.copy_d2h(at, &mut got).unwrap();
            assert_eq!(got, pattern, "rank {rank} block {b}");
        }
        assert!(kv.nvme_take_failed().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_tier_sized_for_one_class_refuses_a_sharded_cache() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let mut kv = sharded_glm_kv(&gpu, 0);
    let tree = spark_runtime::radix_tree::RadixTree::new();
    let cfg = NvmeKvConfig {
        dir: PathBuf::from("/nonexistent-never-created"),
        budget_bytes: 64 << 20,
        fast: false,
        keep: false,
    };
    let e = setup_local(Some(cfg), 0, GLM, &mut kv, &tree, &gpu).unwrap_err();
    assert!(format!("{e:#}").contains("record class"), "{e:#}");
    assert!(!kv.nvme_attached());
}
