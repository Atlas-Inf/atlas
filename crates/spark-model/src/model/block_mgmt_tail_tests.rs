// SPDX-License-Identifier: AGPL-3.0-only

//! Slotted sparse-index tails through the block allocation helpers. Every
//! block a helper pushes into `block_table` must be zeroed and lent a tail
//! before a kernel writes it — including the blocks pushed before a failed
//! allocation, which a preempt-and-retry re-enters with already in-window.

use super::*;
use crate::model::prefix_share::adopt_prefix_match;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{
    KvCacheConfig, KvCacheDtype, NO_TAIL, SparseIndexCacheConfig, TailSlotPlan,
};
use spark_runtime::prefix_cache::{NoPrefixCaching, PrefixCache};
use spark_runtime::radix_tree::RadixTree;

const BLOCKS: usize = 8;
const PLAN: TailSlotPlan = TailSlotPlan {
    lag_blocks: 8,
    sequences: 1,
};

/// A one-layer GLM-style cache whose pooled keys all hold a previous owner's
/// bytes; `plan: None` keeps one tail per block.
fn cache_with_stale_index(gpu: &MockGpuBackend, plan: Option<TailSlotPlan>) -> PagedKvCache {
    stale_index_cache(gpu, plan, None)
}

fn stale_index_cache(
    gpu: &MockGpuBackend,
    plan: Option<TailSlotPlan>,
    cache_blocks_per_seq: Option<u32>,
) -> PagedKvCache {
    let config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 64,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq,
    };
    let mut cache = PagedKvCache::new(config, BLOCKS, gpu).unwrap();
    cache
        .attach_sparse_index_with_tail_slots(SparseIndexCacheConfig::bf16(4, 128), plan, gpu)
        .unwrap();
    let stride = cache.sparse_index_block_stride_bytes(0);
    gpu.memset(cache.sparse_index_pool_ptr(0), 0xAB, BLOCKS * stride)
        .unwrap();
    cache
}

/// Each block owns a published tail slot and zeroed pooled keys.
fn assert_published(cache: &PagedKvCache, gpu: &MockGpuBackend, blocks: &[u32]) {
    let mut map = vec![0u8; BLOCKS * 4];
    gpu.copy_d2h(cache.sparse_index_tail_map_ptr(), &mut map)
        .unwrap();
    let stride = cache.sparse_index_block_stride_bytes(0);
    for &b in blocks {
        let slot = u32::from_le_bytes(map[b as usize * 4..][..4].try_into().unwrap());
        assert_ne!(slot, NO_TAIL, "block {b} owns no tail slot");
        let mut pooled = vec![0xFFu8; stride];
        gpu.copy_d2h(
            cache.sparse_index_pool_ptr(0).offset(b as usize * stride),
            &mut pooled,
        )
        .unwrap();
        assert!(pooled.iter().all(|&x| x == 0), "block {b} kept stale keys");
    }
}

fn prefill(
    seq: &mut SequenceState,
    abs: usize,
    cache: &mut PagedKvCache,
    gpu: &MockGpuBackend,
) -> Result<()> {
    ensure_blocks_through_prefill(seq, abs, cache, &NoPrefixCaching, gpu, 0, false)
}

fn decode(
    seq: &mut SequenceState,
    abs: usize,
    cache: &mut PagedKvCache,
    gpu: &MockGpuBackend,
) -> Result<()> {
    ensure_blocks_through_decode(seq, abs, cache, &NoPrefixCaching, gpu, 0, false)
}

#[test]
fn prefill_exhaustion_publishes_the_blocks_it_pushed() {
    let gpu = MockGpuBackend::new();
    let mut cache = cache_with_stale_index(&gpu, Some(PLAN));
    let held: Vec<u32> = (0..5).map(|_| cache.alloc_block().unwrap()).collect();
    let mut seq = SequenceState::host_only(0);

    let err = prefill(&mut seq, 4, &mut cache, &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("KV cache exhausted"), "{err:#}");
    assert_eq!(seq.block_table.len(), 3);
    assert_published(&cache, &gpu, &seq.block_table);

    // Preempt-and-retry: a victim frees blocks and the same chunk re-enters
    // with the first three already in-window.
    held[..2].iter().for_each(|&b| cache.free_block(b));
    prefill(&mut seq, 4, &mut cache, &gpu).unwrap();
    assert_eq!(seq.block_table.len(), 5);
    assert_published(&cache, &gpu, &seq.block_table);
}

#[test]
fn prefill_before_a_nonzero_window_is_a_noop() {
    let gpu = MockGpuBackend::new();
    let mut cache = cache_with_stale_index(&gpu, None);
    let mut seq = SequenceState::host_only(0);
    seq.block_table.push(cache.alloc_block().unwrap());
    // Model the retained HSS prefix: two disk ids and one resident block make
    // the current window start one. The requested absolute block is before it;
    // bulk-reclaim bookkeeping must not underflow or allocate a new block.
    seq.disk_block_ids.extend([101, 102]);
    let before = seq.block_table.clone();
    prefill(&mut seq, 0, &mut cache, &gpu).unwrap();
    assert_eq!(seq.block_table, before);
}

#[test]
fn prefill_bulk_reclaims_exact_shortfall_from_real_prefix_cache() {
    const BS: usize = 16;
    let gpu = MockGpuBackend::new();
    let mut cache = cache_with_stale_index(&gpu, Some(PLAN));
    let tree = RadixTree::new();
    let mut prefixes = Vec::with_capacity(BLOCKS);

    // Fill the pool with one-block real radix entries. Drop each creator's
    // temporary KV reference after publishing so only the cache owns it.
    for i in 0..BLOCKS {
        let base = 1_000 + (i * BS) as u32;
        let tokens: Vec<u32> = (base..base + BS as u32).collect();
        let block = cache.alloc_block().unwrap();
        let acquired = tree.insert(&tokens, &[block], &[], BS, 0, 0);
        super::super::cache_acquires_refs(&acquired, &mut cache);
        cache.free_block(block);
        // `insert(..., matched_tokens=0)` also records the inserting
        // sequence's temporary tree reference. Finalization releases that
        // reference while the cache's baseline ref remains evictable.
        tree.release(&tokens, BS, 0);
        prefixes.push((tokens, block));
    }
    assert_eq!(cache.num_free_blocks(), 0);

    // Keep one cached block live through a production prefix lookup/adoption;
    // bulk reclaim must evict exactly the two blocks needed by this prefill,
    // leaving that live block and the other cache entries intact.
    let mut live_seq = SequenceState::host_only(7);
    let live_match = tree.lookup_whole_blocks(&prefixes[0].0, BS, 0, 0);
    adopt_prefix_match(&mut live_seq, &live_match, &mut cache).unwrap();
    assert_eq!(cache.ref_count(prefixes[0].1), 2);

    let mut seq = SequenceState::host_only(8);
    ensure_blocks_through_prefill(&mut seq, 1, &mut cache, &tree, &gpu, 0, false).unwrap();
    assert_eq!(seq.block_table.len(), 2);
    assert_eq!(cache.num_free_blocks(), 0);
    assert_eq!(cache.ref_count(prefixes[0].1), 2);
    assert_eq!(tree.peek_matched_tokens(&prefixes[0].0, BS, 0), BS);
    let survivors = prefixes
        .iter()
        .filter(|(tokens, _)| tree.peek_matched_tokens(tokens, BS, 0) == BS)
        .count();
    assert_eq!(survivors, BLOCKS - 2);
    assert_published(&cache, &gpu, &seq.block_table);
}

#[test]
fn prefill_disk_id_failure_publishes_the_block_it_pushed() {
    let gpu = MockGpuBackend::new();
    // HSS is engaged (a per-sequence cap) but no orchestrator is installed:
    // the disk-id step fails after the block is already in `block_table`.
    let mut cache = stale_index_cache(&gpu, Some(PLAN), Some(4));
    let mut seq = SequenceState::host_only(0);

    let err = prefill(&mut seq, 0, &mut cache, &gpu).unwrap_err();
    assert!(
        format!("{err:#}").contains("orchestrator not installed"),
        "{err:#}"
    );
    assert_eq!(seq.block_table.len(), 1);
    assert_published(&cache, &gpu, &seq.block_table);
}

#[test]
fn decode_fill_failure_leaves_the_block_with_the_sequence() {
    let gpu = MockGpuBackend::new();
    let plan = TailSlotPlan {
        lag_blocks: 0,
        sequences: 1,
    };
    let mut cache = cache_with_stale_index(&gpu, Some(plan));
    // Another sequence holds both tail slots.
    let other: Vec<u32> = (0..2).map(|_| cache.alloc_block().unwrap()).collect();
    cache.lend_tail_slots(&other, &gpu, 0).unwrap();
    let mut seq = SequenceState::host_only(0);

    let err = decode(&mut seq, 0, &mut cache, &gpu).unwrap_err();
    assert!(
        format!("{err:#}").contains("tail slots exhausted"),
        "{err:#}"
    );
    // Freed with the sequence rather than leaked from the pool.
    assert_eq!(
        cache.num_free_blocks() + other.len() + seq.block_table.len(),
        BLOCKS
    );
}

#[test]
fn write_window_refuses_an_owned_block_without_a_tail() {
    let gpu = MockGpuBackend::new();
    for plan in [Some(PLAN), None] {
        let mut cache = cache_with_stale_index(&gpu, plan);
        let mut seq = SequenceState::host_only(0);
        // A path that pushed a block without filling it (a swap restore).
        seq.block_table.push(cache.alloc_block().unwrap());
        let res = decode(&mut seq, 0, &mut cache, &gpu);
        match plan {
            Some(_) => {
                let err = res.unwrap_err();
                assert!(format!("{err:#}").contains("no index tail"), "{err:#}");
            }
            // Per-block tails: nothing is lent, nothing can be missing.
            None => res.unwrap(),
        }
    }
}

#[test]
fn write_window_leaves_prefix_cache_blocks_alone() {
    let gpu = MockGpuBackend::new();
    let mut cache = cache_with_stale_index(&gpu, Some(PLAN));
    let mut seq = SequenceState::host_only(0);
    // A matched block the radix cache still holds, and one whose node was
    // evicted since the lookup: neither was ever lent a tail.
    let shared = cache.alloc_block().unwrap();
    cache.inc_ref(shared);
    let evicted = cache.alloc_block().unwrap();
    seq.block_table.extend([shared, evicted]);
    seq.cached_prefix_blocks = 2;
    prefill(&mut seq, 2, &mut cache, &gpu).unwrap();
    assert_published(&cache, &gpu, &seq.block_table[2..]);
}

#[test]
fn write_window_verdict_ignores_reference_counts() {
    let gpu = MockGpuBackend::new();
    let mut cache = cache_with_stale_index(&gpu, Some(PLAN));
    let mut seq = SequenceState::host_only(0);
    prefill(&mut seq, 0, &mut cache, &gpu).unwrap();
    // A frontier block that lost its tail while another holder kept a
    // reference. Reference counts follow each rank's own radix cache, so a
    // verdict keyed on them could fail one rank of a pair and hang the other.
    let frontier = seq.block_table[0];
    cache.inc_ref(frontier);
    cache.inc_ref(frontier);
    cache.free_blocks(&[frontier]);
    assert!(cache.tail_slot_missing(frontier) && cache.ref_count(frontier) == 2);
    let err = decode(&mut seq, 0, &mut cache, &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("no index tail"), "{err:#}");
}
