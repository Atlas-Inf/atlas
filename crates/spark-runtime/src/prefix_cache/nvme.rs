// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill tier for the prefix cache (`ATLAS_KV_NVME_DIR`, default OFF).
//!
//! With the tier enabled, a cached block that LRU eviction would delete is
//! instead written to an NVMe record and its radix node stays in the tree as
//! "on disk". A later prefix hit reads those records back into freshly
//! allocated GPU blocks BEFORE the ordinary resident-only `lookup`, so every
//! existing consumer of [`super::PrefixMatch`] keeps seeing resident blocks
//! only. Tree bookkeeping (slots, disk LRU, budget) lives behind this trait;
//! the byte movement lives with the KV cache (`PagedKvCache::nvme_*`), so the
//! tree never performs I/O.
//!
//! Tree invariant (spill mode): a node on disk has only on-disk descendants.
//! Spill picks resident nodes with no RESIDENT children; restore promotes a
//! run top-down starting right below the resident prefix; `insert` re-homes an
//! on-disk node onto the inserting sequence's block top-down.

/// Host RAM the tree keeps per ON-DISK block, for sizing: a disk budget of N
/// records can grow the process by about `N ×` this. Per node (16-token
/// blocks, 64-bit): the `RadixNode` itself (152 B, in a `Vec` that doubles —
/// 1.5× on average), its `parent_key` (64 B + allocator header), its entry in
/// the parent's `children` map (a 4-bucket table of 32 B buckets, plus the
/// 64 B key), and the slot index (owner + tag + disk-LRU entry, ~60 B).
/// `host_bytes_per_disk_block_covers_a_node` pins the inline part.
pub const NVME_HOST_BYTES_PER_BLOCK: usize = 640;

/// One eviction-time write the caller MUST perform before returning `block`
/// to the free list: copy the block's bytes to record `slot`, stamped `tag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillOrder {
    pub block: u32,
    pub slot: u32,
    pub tag: u64,
}

/// One on-disk block of a restore run: its record slot and the tag the record
/// must carry (a mismatch means the record is not this node's — never restore
/// it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskRef {
    pub slot: u32,
    pub tag: u64,
}

/// What [`NvmePrefixTier::plan_restore`] found and pinned.
///
/// `disk[i]` covers tokens `[resident_tokens + i·bs, resident_tokens + (i+1)·bs)`.
/// The whole path (`pinned_tokens`) carries one extra radix ref so eviction
/// during the restore's own block allocation cannot take it; the matching
/// [`NvmePrefixTier::complete_restore`] releases it. An empty plan pins
/// nothing and needs no completion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestorePlan {
    pub resident_tokens: usize,
    pub disk: Vec<DiskRef>,
    pub pinned_tokens: usize,
}

/// Counters for logs/telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmeStats {
    pub max_slots: u32,
    pub slots_used: u32,
    /// Blocks written to disk at eviction.
    pub spills: u64,
    /// Spill writes that failed (the node was dropped — plain eviction).
    pub spill_failures: u64,
    /// Blocks read back and promoted to resident.
    pub restores: u64,
    /// Restore reads that failed or mismatched (the node and its on-disk
    /// subtree were dropped — the prefix recomputes).
    pub restore_failures: u64,
    /// On-disk blocks dropped by the disk budget (LRU).
    pub disk_drops: u64,
    /// Evicted blocks deleted instead of spilled because they were colder than
    /// every droppable on-disk block (budget full).
    pub cold_drops: u64,
    /// Evictions that needed no write: the block still had the record it was
    /// restored from ([`NvmePrefixTier::set_keep_restored`]).
    pub clean_evictions: u64,
}

/// The tree side of the NVMe tier. Every method is a no-op / empty result
/// until [`Self::enable`] has been called.
pub trait NvmePrefixTier: Send + Sync {
    /// Turn spill-on-evict on with a budget of `max_slots` records. Must be
    /// called before any block is cached. Returns `false` if already enabled
    /// or `max_slots == 0`.
    fn enable(&self, max_slots: u32) -> bool {
        self.enable_classes(max_slots, 1)
    }

    /// [`Self::enable`] with the record slots split into `classes` classes of
    /// `per_class` slots: a block `b` spills into a slot `s` with
    /// `s % classes == b % classes`, and the budget drops only on-disk blocks
    /// of the class that is full. A latent-sharded KV cache
    /// (`ATLAS_GLM_KV_SHARD=1`) takes one class per rank, since its records
    /// differ in size by owner (`PagedKvCache::nvme_classes`). `classes = 1`
    /// is [`Self::enable`].
    fn enable_classes(&self, per_class: u32, classes: u32) -> bool;

    fn is_enabled(&self) -> bool;

    /// Keep a block's record when the block is restored (default: release
    /// it), so evicting the block again costs no I/O at all. The record stays
    /// valid only while nothing rewrites the block: a cached full block is
    /// never APPENDED to, but a prefill that resumes below its match
    /// recomputes the rows in between in place, with equivalent rather than
    /// identical values — the caller must report that ([`Self::forget_kept`]).
    /// Kept records count against the budget, so they are only kept while at
    /// most half of it is in use.
    fn set_keep_restored(&self, keep: bool);

    /// Blocks `blocks` (indices into `tokens`' cached full-block path) are
    /// about to be rewritten in place: release the record any of them kept
    /// from its restore, so its next eviction writes the block again. No-op
    /// unless records are kept.
    fn forget_kept(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
        blocks: std::ops::Range<usize>,
    );

    /// Find the on-disk run that continues the resident prefix of `tokens`
    /// (full blocks only) and pin the path. Empty plan ⇒ nothing to restore.
    /// The run keeps its place in the disk LRU until it is restored: a plan
    /// the caller declines must not keep records the budget could drop.
    fn plan_restore(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> RestorePlan;

    /// Finish a restore: `restored[i]` (a freshly allocated block holding
    /// `plan.disk[i]`'s bytes, KV ref 1) becomes node `i`'s block. When
    /// `failed` is set, node `restored.len()` could not be read — it and its
    /// on-disk subtree are dropped. Releases the plan's pin.
    ///
    /// Returns blocks the caller must `dec_ref` once: restored blocks the tree
    /// did not adopt, plus any resident block freed by a subtree drop.
    fn complete_restore(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
        plan: &RestorePlan,
        restored: &[u32],
        failed: bool,
    ) -> Vec<u32>;

    /// The writes for these orders failed: drop their nodes (and on-disk
    /// subtrees). Returns blocks the caller must `dec_ref` once.
    fn spill_failed(&self, orders: &[SpillOrder]) -> Vec<u32>;

    /// Deepest SSM snapshot anchor (resident or tiered) with depth ≤ `limit`
    /// that a lookup of `tokens` could use. Read-only: no LRU, stats or lease
    /// side effects. 0 when none.
    fn snapshot_anchor_depth(
        &self,
        tokens: &[u32],
        limit: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> usize;

    fn nvme_stats(&self) -> NvmeStats;
}
