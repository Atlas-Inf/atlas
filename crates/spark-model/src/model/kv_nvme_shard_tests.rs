// SPDX-License-Identifier: AGPL-3.0-only

//! The spill → restore cycle under a latent shard (`ATLAS_GLM_KV_SHARD=1`),
//! run for BOTH ranks of a simulated pair over the real radix tree (record
//! slots in two classes) and mock-GPU caches whose block ids diverge: each
//! rank spills and restores only the latents it stores, every block's index
//! rows on both, and the ranks agree on every record slot.

use std::sync::{Arc, Mutex};

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{
    KvCacheConfig, KvCacheDtype, LatentShardSpec, PagedKvCache, SparseIndexCacheConfig,
};
use spark_runtime::prefix_cache::{DiskRef, NvmePrefixTier, NvmeStats, PrefixCache};
use spark_runtime::radix_tree::RadixTree;

use super::restore_prefix;
use crate::model::block_mgmt::{alloc_block_evicting, apply_evicted_blocks, cache_acquires_refs};
use crate::model::prefix_share::cap_prefix_match;

const BS: usize = 16;
const POOL: usize = 8;

fn sharded_kv(gpu: &MockGpuBackend, rank: usize, fast: bool) -> PagedKvCache {
    let cfg = KvCacheConfig {
        block_size: BS,
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
    let mut kv = PagedKvCache::new_latent_sharded(cfg, POOL, gpu, spec).unwrap();
    kv.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    for class in 0..kv.nvme_classes() {
        let store = atlas_tier::MemSwapStore::new(kv.nvme_class_record_bytes(class));
        if fast {
            kv.attach_nvme_fast(Arc::new(Mutex::new(store)), gpu)
        } else {
            kv.attach_nvme_spill(Box::new(store), gpu)
        }
        .unwrap();
    }
    kv
}

fn classed_tree(per_class: u32) -> RadixTree {
    let tree = RadixTree::new();
    assert!(tree.enable_classes(per_class, 2));
    tree
}

/// What this rank holds of `b`: latents if it owns the block, index rows.
fn regions(kv: &PagedKvCache, b: u32) -> Vec<(DevicePtr, usize)> {
    let shard = kv.latent_shard().unwrap();
    let mut out = Vec::new();
    for l in 0..kv.num_layers() {
        let k = kv.k_block_stride_bytes_for_layer(l);
        if let Some(slot) = shard.local_slot(b) {
            out.push((kv.latent_pool_ptr(l).offset(slot as usize * k), k));
        }
        let ix = kv.sparse_index_block_stride_bytes(l);
        out.push((kv.sparse_index_pool_ptr(l).offset(b as usize * ix), ix));
    }
    out
}

fn dump(kv: &PagedKvCache, gpu: &MockGpuBackend, b: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (p, n) in regions(kv, b) {
        let mut v = vec![0u8; n];
        gpu.copy_d2h(p, &mut v).unwrap();
        out.extend(v);
    }
    out
}

/// Bytes a rank derives for logical block `l` of `tokens` (the same on both
/// ranks for the index rows, as the real index is replicated).
fn fill(kv: &PagedKvCache, gpu: &MockGpuBackend, b: u32, seed: u8) {
    for (i, (p, n)) in regions(kv, b).into_iter().enumerate() {
        let v: Vec<u8> = (0..n).map(|j| seed ^ (i * 7 + j) as u8).collect();
        gpu.copy_h2d(&v, p).unwrap();
    }
}

/// Churn the free lists in a rank-specific order, so the ranks' ids for the
/// same logical blocks differ from here on (as on hardware).
fn diverge(kv: &mut PagedKvCache, rank: usize) {
    let mut held: Vec<u32> = (0..4).map(|l| kv.alloc_block_at(l).unwrap()).collect();
    if rank == 1 {
        held.reverse();
    }
    kv.free_blocks(&held);
}

/// A finished request cached `tokens`' full blocks; returns their bytes.
fn cache_request(
    kv: &mut PagedKvCache,
    tree: &RadixTree,
    gpu: &MockGpuBackend,
    tokens: &[u32],
) -> Vec<Vec<u8>> {
    cache_request_on(kv, tree, gpu, tokens).1
}

/// [`cache_request`], with the blocks the request held.
fn cache_request_on(
    kv: &mut PagedKvCache,
    tree: &RadixTree,
    gpu: &MockGpuBackend,
    tokens: &[u32],
) -> (Vec<u32>, Vec<Vec<u8>>) {
    let n = tokens.len() / BS;
    let blocks: Vec<u32> = (0..n).map(|l| kv.alloc_block_at(l).unwrap()).collect();
    for (l, &b) in blocks.iter().enumerate() {
        assert_eq!(b as usize % 2, l % 2, "drawn at its logical index");
        fill(kv, gpu, b, 0x40 + l as u8);
    }
    let acquired = tree.insert(tokens, &blocks, &[], BS, 0, 0);
    cache_acquires_refs(&acquired, kv);
    tree.release(tokens, BS, 0);
    kv.free_blocks(&blocks);
    let bytes = blocks.iter().map(|&b| dump(kv, gpu, b)).collect();
    (blocks, bytes)
}

/// Another workload takes every block (forcing the cache out), then frees.
fn pressure(kv: &mut PagedKvCache, tree: &RadixTree, gpu: &MockGpuBackend) {
    let mut held = Vec::new();
    while let Some(b) = alloc_block_evicting(kv, tree, gpu, held.len()) {
        held.push(b);
    }
    assert_eq!(held.len(), POOL, "the whole pool is reclaimable");
    kv.free_blocks(&held);
}

/// The run a restore of `t` would read (slots and tags), without reading it.
fn planned(tree: &RadixTree, t: &[u32]) -> Vec<DiskRef> {
    let plan = tree.plan_restore(t, BS, 0);
    assert!(
        tree.complete_restore(t, BS, 0, &plan, &[], false)
            .is_empty()
    );
    plan.disk
}

/// What one rank saw: the planned run, the restored blocks, and the stats.
#[derive(Debug, PartialEq)]
struct RankView {
    plan: Vec<DiskRef>,
    stats: NvmeStats,
}

fn run_rank(rank: usize, fast: bool, t: &[u32]) -> (RankView, Vec<u32>) {
    let gpu = MockGpuBackend::new();
    let mut kv = sharded_kv(&gpu, rank, fast);
    let tree = classed_tree(64);
    diverge(&mut kv, rank);
    let (cached, want) = cache_request_on(&mut kv, &tree, &gpu, t);
    pressure(&mut kv, &tree, &gpu);
    assert!(tree.lookup(t, BS, 0, 0).is_empty(), "evicted from GPU");
    let plan = planned(&tree, t);
    for (l, d) in plan.iter().enumerate() {
        assert_eq!(d.slot as usize % 2, l % 2, "slot class = logical residue");
    }
    let r = restore_prefix(&tree, &mut kv, &gpu, t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!((r.restored, r.failed), (want.len(), false), "rank {rank}");
    let m = tree.lookup(t, BS, 0, 0);
    assert_eq!(m.matched_tokens, t.len());
    for (l, &b) in m.matched_blocks.iter().enumerate() {
        assert_eq!(b as usize % 2, l % 2, "restored at its logical residue");
        assert_eq!(dump(&kv, &gpu, b), want[l], "rank {rank} block {l}");
        assert_eq!(kv.ref_count(b), 1, "only the cache's ref");
    }
    // Owned latents went through the full lane, peer blocks through the
    // index-only one: the I/O counters cover every block once each way.
    let io = kv.nvme_io_stats();
    assert_eq!((io.spilled_blocks, io.restored_blocks), (5, 5));
    tree.release(t, BS, 0);
    assert_eq!(kv.num_free_in_all(), POOL - want.len(), "no leak");
    let view = RankView {
        plan,
        stats: tree.nvme_stats(),
    };
    (view, cached)
}

#[test]
fn both_ranks_restore_their_halves_and_agree_on_every_slot() {
    let t: Vec<u32> = (0..5 * BS as u32).collect();
    for fast in [false, true] {
        let (v0, b0) = run_rank(0, fast, &t);
        let (v1, b1) = run_rank(1, fast, &t);
        assert_eq!(v0, v1, "same slots, tags and accounting (fast={fast})");
        assert_ne!(b0, b1, "while the ranks cached on different block ids");
        assert_eq!(v0.stats.spills, 5);
        assert_eq!(v0.stats.restores, 5);
    }
}

/// A budget smaller than the chain: each class keeps at most its own slots,
/// and both ranks drop the same records.
#[test]
fn eviction_accounting_is_per_class_and_identical_on_both_ranks() {
    let t: Vec<u32> = (0..6 * BS as u32).collect();
    let u: Vec<u32> = (10_000..10_000 + 2 * BS as u32).collect();
    let rank = |rank: usize| {
        let gpu = MockGpuBackend::new();
        let mut kv = sharded_kv(&gpu, rank, true);
        let tree = classed_tree(2);
        diverge(&mut kv, rank);
        cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        cache_request(&mut kv, &tree, &gpu, &u);
        pressure(&mut kv, &tree, &gpu);
        let s = tree.nvme_stats();
        assert!(s.slots_used <= 4, "{s:?}");
        let runs = [planned(&tree, &t), planned(&tree, &u)];
        for class in 0..2 {
            let n = runs
                .iter()
                .flatten()
                .filter(|d| d.slot % 2 == class)
                .count();
            assert!(n <= 2, "class {class} holds {n} records");
        }
        // Whatever is still on disk restores, and nothing leaks.
        for x in [&t, &u] {
            if let Some(r) = restore_prefix(&tree, &mut kv, &gpu, x, 0, 0, |p| p.disk.len()) {
                assert!(!r.failed, "rank {rank}");
            }
        }
        pressure(&mut kv, &tree, &gpu);
        assert_eq!(kv.num_free_in_all(), POOL);
        (s, runs)
    };
    assert_eq!(rank(0), rank(1));
}

/// One rank restores more than the other (its pool or anchor gave out): the
/// F83 cap brings it to the agreed match, which the other rank holds too.
#[test]
fn the_rank_that_restored_more_is_capped_to_blocks_both_hold() {
    let t: Vec<u32> = (0..4 * BS as u32).collect();
    let rank = |rank: usize, restore: usize| {
        let gpu = MockGpuBackend::new();
        let mut kv = sharded_kv(&gpu, rank, false);
        let tree = classed_tree(64);
        diverge(&mut kv, rank);
        let want = cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |_| restore).unwrap();
        assert_eq!(r.restored, restore);
        let local = tree.lookup_whole_blocks(&t, BS, 0, 0);
        assert_eq!(local.matched_tokens, restore * BS);
        (gpu, kv, tree, local, want)
    };
    let mut ranks = [rank(0, 4), rank(1, 3)];
    let agreed = 3 * BS;
    for (gpu, kv, tree, local, want) in &mut ranks {
        let capped = cap_prefix_match(tree, &t, BS, 0, 0, local.clone(), agreed);
        assert_eq!(capped.matched_tokens, agreed);
        for (l, &b) in capped.matched_blocks.iter().enumerate() {
            assert_eq!(b as usize % 2, l % 2);
            assert_eq!(dump(kv, gpu, b), want[l]);
        }
        let evicted = tree.evict(POOL);
        apply_evicted_blocks(evicted, kv, tree, gpu);
        tree.release_matched(&t, BS, agreed, 0);
        pressure(kv, tree, gpu);
        assert_eq!(kv.num_free_in_all(), POOL);
    }
}

/// `ATLAS_GLM_NVME_KEEP` under the shard: a restored block evicted again
/// writes nothing, in either lane, and restores from the kept record.
#[test]
fn kept_records_cost_no_write_in_either_lane() {
    let t: Vec<u32> = (0..4 * BS as u32).collect();
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let mut kv = sharded_kv(&gpu, rank, true);
        let tree = classed_tree(64);
        tree.set_keep_restored(true);
        let want = cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
        assert_eq!(r.restored, 4);
        pressure(&mut kv, &tree, &gpu);
        assert_eq!(kv.nvme_io_stats().spilled_blocks, 4, "nothing rewritten");
        let s = tree.nvme_stats();
        assert_eq!((s.spills, s.clean_evictions), (4, 4));
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
        assert_eq!((r.restored, r.failed), (4, false));
        let m = tree.lookup(&t, BS, 0, 0);
        for (l, &b) in m.matched_blocks.iter().enumerate() {
            assert_eq!(dump(&kv, &gpu, b), want[l], "rank {rank} block {l}");
        }
        tree.release(&t, BS, 0);
    }
}
