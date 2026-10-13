// SPDX-License-Identifier: AGPL-3.0-only

//! The spill → restore cycle end to end over the real radix tree, a mock-GPU
//! KV cache and an in-memory record store.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache, SparseIndexCacheConfig};
use spark_runtime::prefix_cache::{NvmePrefixTier, PrefixCache};
use spark_runtime::radix_tree::RadixTree;

use super::{RestoreOutcome, restore_prefix};
use crate::model::block_mgmt::{alloc_block_evicting, cache_acquires_refs};
use crate::model::prefix_share::cap_prefix_match;

pub(super) const BS: usize = 16;
pub(super) const POOL: usize = 6;

/// Which I/O path the tier runs on (`ATLAS_GLM_NVME_FAST`).
#[derive(Clone, Copy)]
pub(super) enum Path {
    Sync,
    Fast,
}

fn attach(
    kv: &mut PagedKvCache,
    gpu: &MockGpuBackend,
    store: Box<dyn atlas_tier::SwapStore>,
    path: Path,
) {
    match path {
        Path::Sync => kv.attach_nvme_spill(store, gpu),
        Path::Fast => kv.attach_nvme_fast(std::sync::Arc::new(std::sync::Mutex::new(store)), gpu),
    }
    .unwrap();
}

fn glm_kv(gpu: &MockGpuBackend, store: Option<Box<dyn atlas_tier::SwapStore>>) -> PagedKvCache {
    glm_kv_on(gpu, store, Path::Sync)
}

/// GLM-5.3 geometry (FP8-G128 latent, V aliases K, pooled BF16 index).
pub(super) fn glm_kv_on(
    gpu: &MockGpuBackend,
    store: Option<Box<dyn atlas_tier::SwapStore>>,
    path: Path,
) -> PagedKvCache {
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
    let mut kv = PagedKvCache::new_with_v_alias(cfg, POOL, gpu, true).unwrap();
    kv.attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
        .unwrap();
    let store =
        store.unwrap_or_else(|| Box::new(atlas_tier::MemSwapStore::new(kv.nvme_record_bytes())));
    attach(&mut kv, gpu, store, path);
    kv
}

fn regions(kv: &PagedKvCache, b: u32) -> Vec<(spark_runtime::gpu::DevicePtr, usize)> {
    (0..kv.num_layers())
        .flat_map(|l| {
            let is = kv.sparse_index_block_stride_bytes(l);
            [
                (kv.k_cache_ptr(l, b), kv.k_block_stride_bytes_for_layer(l)),
                (kv.sparse_index_pool_ptr(l).offset(b as usize * is), is),
            ]
        })
        .collect()
}

pub(super) fn fill(kv: &PagedKvCache, gpu: &MockGpuBackend, b: u32, seed: u8) {
    for (i, (p, n)) in regions(kv, b).into_iter().enumerate() {
        let v: Vec<u8> = (0..n).map(|j| seed ^ (i * 7 + j) as u8).collect();
        gpu.copy_h2d(&v, p).unwrap();
    }
}

pub(super) fn dump(kv: &PagedKvCache, gpu: &MockGpuBackend, b: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (p, n) in regions(kv, b) {
        let mut v = vec![0u8; n];
        gpu.copy_d2h(p, &mut v).unwrap();
        out.extend(v);
    }
    out
}

/// A finished request cached `n` blocks of `tokens`: returns their bytes.
pub(super) fn cache_request(
    kv: &mut PagedKvCache,
    tree: &RadixTree,
    gpu: &MockGpuBackend,
    tokens: &[u32],
) -> Vec<Vec<u8>> {
    let n = tokens.len() / BS;
    let blocks: Vec<u32> = (0..n).map(|_| kv.try_alloc_block().unwrap()).collect();
    for (i, &b) in blocks.iter().enumerate() {
        fill(kv, gpu, b, 0x40 + i as u8);
    }
    let acquired = tree.insert(tokens, &blocks, &[], BS, 0, 0);
    cache_acquires_refs(&acquired, kv);
    tree.release(tokens, BS, 0);
    kv.free_blocks(&blocks); // the sequence's own refs
    blocks.iter().map(|&b| dump(kv, gpu, b)).collect()
}

/// Another workload takes every block (forcing the cache out), then frees them.
pub(super) fn pressure(kv: &mut PagedKvCache, tree: &RadixTree, gpu: &MockGpuBackend) {
    let mut held = Vec::new();
    while let Some(b) = alloc_block_evicting(kv, tree, gpu, held.len()) {
        held.push(b);
    }
    assert_eq!(held.len(), POOL, "the whole pool is reclaimable");
    kv.free_blocks(&held);
}

pub(super) fn tree_with_tier(slots: u32) -> RadixTree {
    let tree = RadixTree::new();
    assert!(tree.enable(slots));
    tree
}

#[test]
fn evicted_prefix_restores_byte_identical() {
    evicted_prefix_restores(Path::Sync);
}

#[test]
fn evicted_prefix_restores_byte_identical_on_the_fast_path() {
    evicted_prefix_restores(Path::Fast);
}

fn evicted_prefix_restores(path: Path) {
    let gpu = MockGpuBackend::new();
    let mut kv = glm_kv_on(&gpu, None, path);
    let tree = tree_with_tier(64);
    let t: Vec<u32> = (0..3 * BS as u32).collect();
    let want = cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    assert!(tree.lookup(&t, BS, 0, 0).is_empty(), "evicted from GPU");
    assert_eq!(tree.nvme_stats().spills, 3);

    let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!((r.restored, r.failed), (3, false));
    let m = tree.lookup(&t, BS, 0, 0);
    assert_eq!(m.matched_tokens, 3 * BS);
    for (i, &b) in m.matched_blocks.iter().enumerate() {
        assert_eq!(dump(&kv, &gpu, b), want[i], "block {i} bytes");
        // Exactly the cache's own ref (the prefill adds the sequence's).
        assert_eq!(kv.ref_count(b), 1, "restored block ownership");
    }
    tree.release(&t, BS, 0);
    assert_eq!(
        kv.num_free_blocks(),
        POOL - 3,
        "no leak: only the cached blocks held"
    );
}

/// Two ranks restore different amounts (each pages in up to its own anchor,
/// or its pool gives out early). The production cap (`cap_prefix_match`, the
/// F83 minimum) brings the rank that restored more down to the agreed match:
/// it holds the agreed whole blocks and nothing deeper, what it restored past
/// them stays an ordinary cache entry, and nothing leaks on either rank.
#[test]
fn a_rank_that_restored_more_is_capped_to_the_agreed_match() {
    let t: Vec<u32> = (0..4 * BS as u32).collect();
    let rank = |restore: usize| {
        let gpu = MockGpuBackend::new();
        let mut kv = glm_kv(&gpu, None);
        let tree = tree_with_tier(64);
        cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |_| restore).unwrap();
        assert_eq!((r.restored, r.failed), (restore, false));
        let local = tree.lookup_whole_blocks(&t, BS, 0, 0);
        assert_eq!(local.matched_tokens, restore * BS);
        (gpu, kv, tree, local)
    };
    let mut ranks = [rank(4), rank(2)];
    let agreed = 2 * BS;
    for (gpu, kv, tree, local) in &mut ranks {
        let restored = local.matched_blocks.clone();
        let capped = cap_prefix_match(tree, &t, BS, 0, 0, local.clone(), agreed);
        assert_eq!(capped.matched_blocks, restored[..2]);
        // The sequence's radix refs are on the agreed blocks alone: the rest
        // of what this rank restored is evictable again at once.
        let spills = tree.nvme_stats().spills;
        let evicted = tree.evict(POOL);
        assert_eq!(
            evicted.physical,
            restored[2..].iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(tree.nvme_stats().spills - spills, restored.len() as u64 - 2);
        super::super::block_mgmt::apply_evicted_blocks(evicted, kv, tree, gpu);
        tree.release_matched(&t, BS, agreed, 0);
        pressure(kv, tree, gpu);
    }
}

#[test]
fn restore_policy_can_decline_and_nothing_leaks() {
    for path in [Path::Sync, Path::Fast] {
        let gpu = MockGpuBackend::new();
        let mut kv = glm_kv_on(&gpu, None, path);
        let tree = tree_with_tier(64);
        let t: Vec<u32> = (0..2 * BS as u32).collect();
        cache_request(&mut kv, &tree, &gpu, &t);
        pressure(&mut kv, &tree, &gpu);
        let io = kv.nvme_io_stats();
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |_| 0).unwrap();
        // Declined: no block taken, no I/O of any kind — only the pin goes.
        let declined = RestoreOutcome {
            on_disk: 2,
            ..RestoreOutcome::default()
        };
        assert_eq!(r, declined);
        assert_eq!(kv.nvme_io_stats(), io);
        assert_eq!(kv.num_free_blocks(), POOL);
        // Still on disk and restorable later (the pin was released).
        let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
        assert_eq!(r.restored, 2);
    }
}

/// Record store whose writes or reads fail on demand.
struct FlakyStore {
    inner: atlas_tier::MemSwapStore,
    fail_writes: bool,
    fail_reads: bool,
}

impl atlas_tier::SwapStore for FlakyStore {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(!self.fail_writes, "injected ENOSPC");
        self.inner.write_record(slot, bytes)
    }
    fn read_record(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        anyhow::ensure!(!self.fail_reads, "injected EIO");
        self.inner.read_record(slot, out)
    }
}

fn flaky(record: usize, fail_writes: bool, fail_reads: bool) -> Box<dyn atlas_tier::SwapStore> {
    Box::new(FlakyStore {
        inner: atlas_tier::MemSwapStore::new(record),
        fail_writes,
        fail_reads,
    })
}

fn record_bytes() -> usize {
    let gpu = MockGpuBackend::new();
    glm_kv(&gpu, None).nvme_record_bytes()
}

#[test]
fn disk_full_degrades_to_plain_eviction() {
    let gpu = MockGpuBackend::new();
    let mut kv = glm_kv(&gpu, Some(flaky(record_bytes(), true, false)));
    let tree = tree_with_tier(64);
    let t: Vec<u32> = (0..2 * BS as u32).collect();
    cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    let s = tree.nvme_stats();
    assert_eq!((s.spill_failures, s.slots_used), (2, 0));
    assert!(restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).is_none());
    assert_eq!(kv.num_free_blocks(), POOL);
}

#[test]
fn read_error_recomputes_and_forgets_the_record() {
    let gpu = MockGpuBackend::new();
    let mut kv = glm_kv(&gpu, Some(flaky(record_bytes(), false, true)));
    let tree = tree_with_tier(64);
    let t: Vec<u32> = (0..2 * BS as u32).collect();
    cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!((r.restored, r.failed), (0, true));
    assert!(tree.lookup(&t, BS, 0, 0).is_empty());
    assert_eq!(
        kv.num_free_blocks(),
        POOL,
        "allocated restore targets returned"
    );
    assert!(restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).is_none());
    assert_eq!(tree.nvme_stats().slots_used, 0);
}

/// Write-behind: the failure surfaces after the evictions returned, and the
/// tree still ends up exactly where a plain eviction leaves it.
#[test]
fn disk_full_on_the_fast_path_degrades_to_plain_eviction() {
    let gpu = MockGpuBackend::new();
    let store = flaky(record_bytes(), true, false);
    let mut kv = glm_kv_on(&gpu, Some(store), Path::Fast);
    let tree = tree_with_tier(64);
    let t: Vec<u32> = (0..2 * BS as u32).collect();
    cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    // Either the failures were already in (nothing to plan) or the records
    // fail verification; never a restore.
    let first = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len());
    assert!(first.is_none_or(|r| r.restored == 0));
    assert!(restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).is_none());
    assert!(tree.lookup(&t, BS, 0, 0).is_empty());
    assert_eq!(tree.nvme_stats().slots_used, 0);
    assert_eq!(kv.num_free_blocks(), POOL);
}

#[test]
fn restore_reports_where_its_time_went() {
    let gpu = MockGpuBackend::new();
    let mut kv = glm_kv_on(&gpu, None, Path::Fast);
    let tree = tree_with_tier(64);
    let t: Vec<u32> = (0..3 * BS as u32).collect();
    cache_request(&mut kv, &tree, &gpu, &t);
    pressure(&mut kv, &tree, &gpu);
    let io = kv.nvme_io_stats();
    assert!(io.fast);
    assert_eq!(io.spilled_blocks, 3);
    let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!(r.restored, 3);
    // Its share of the cache's counters: three blocks scattered as one run,
    // the evictions it caused, and the waits for the writer kept apart from
    // the gather (evict) and from the read.
    assert_eq!((r.io.restored_blocks, r.io.scatter_runs), (3, 1));
    assert_eq!(r.io.spilled_blocks, kv.nvme_io_stats().spilled_blocks - 3);
    assert!(r.io.gather_runs <= r.io.spilled_blocks);
    assert!(r.io.spill_wait_micros <= r.evict_micros);
    assert_eq!(kv.nvme_io_stats().since(io).restored_blocks, 3);
    // The restored run sits on ascending blocks, in path order.
    let m = tree.lookup(&t, BS, 0, 0);
    assert!(m.matched_blocks.is_sorted(), "{:?}", m.matched_blocks);
    tree.release(&t, BS, 0);
}

/// Slotted index tails (prefix caching + ATLAS_MARCONI_PREFILL_ONLY): every
/// tail a request was lent comes back — through spill and restore alike — and
/// a restored block is published tail-less, never aliasing a live slot.
#[test]
fn slotted_tails_are_released_by_spill_and_absent_after_restore() {
    slotted_tails(Path::Sync);
}

#[test]
fn slotted_tails_survive_the_fast_path() {
    slotted_tails(Path::Fast);
}

fn slotted_tails(path: Path) {
    use spark_runtime::kv_cache::{NO_TAIL, TailSlotPlan};
    let gpu = MockGpuBackend::new();
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
    let mut kv = PagedKvCache::new_with_v_alias(cfg, POOL, &gpu, true).unwrap();
    let plan = TailSlotPlan {
        lag_blocks: 1,
        sequences: 1,
    }; // 3 slots
    kv.attach_sparse_index_with_tail_slots(SparseIndexCacheConfig::bf16(4, 128), Some(plan), &gpu)
        .unwrap();
    let record = kv.nvme_record_bytes();
    let store = Box::new(atlas_tier::MemSwapStore::new(record));
    attach(&mut kv, &gpu, store, path);
    let tree = tree_with_tier(64);
    let tail_map = |kv: &PagedKvCache| {
        let mut b = vec![0u8; POOL * 4];
        gpu.copy_d2h(kv.sparse_index_tail_map_ptr(), &mut b)
            .unwrap();
        b.chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect::<Vec<u32>>()
    };

    // A request writes 3 blocks, each lent a tail, and caches them.
    let t: Vec<u32> = (0..3 * BS as u32).collect();
    let blocks: Vec<u32> = (0..3).map(|_| kv.try_alloc_block().unwrap()).collect();
    kv.lend_tail_slots(&blocks, &gpu, 0).unwrap();
    for (i, &b) in blocks.iter().enumerate() {
        fill(&kv, &gpu, b, 0x60 + i as u8);
        assert_ne!(tail_map(&kv)[b as usize], NO_TAIL);
    }
    let want: Vec<Vec<u8>> = blocks.iter().map(|&b| dump(&kv, &gpu, b)).collect();
    let acquired = tree.insert(&t, &blocks, &[], BS, 0, 0);
    cache_acquires_refs(&acquired, &mut kv);
    tree.release(&t, BS, 0);
    kv.free_blocks(&blocks);

    pressure(&mut kv, &tree, &gpu);
    assert_eq!(tree.nvme_stats().spills, 3);

    let r = restore_prefix(&tree, &mut kv, &gpu, &t, 0, 0, |p| p.disk.len()).unwrap();
    assert_eq!(r.restored, 3);
    let m = tree.lookup(&t, BS, 0, 0);
    let map = tail_map(&kv);
    for (i, &b) in m.matched_blocks.iter().enumerate() {
        assert_eq!(dump(&kv, &gpu, b), want[i], "block {i} bytes");
        assert_eq!(map[b as usize], NO_TAIL, "restored block {i} owns no tail");
    }
    tree.release(&t, BS, 0);
    // Every slot is free again: a new request can take the whole pool.
    let fresh: Vec<u32> = (0..3).map(|_| kv.try_alloc_block().unwrap()).collect();
    kv.lend_tail_slots(&fresh, &gpu, 0)
        .expect("all tail slots returned");
}
