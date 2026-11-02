// SPDX-License-Identifier: AGPL-3.0-only

//! Single-threaded radix-tree state — wrapped behind a `Mutex` in the
//! [`super::RadixTree`] public API. Token sequences are chunked at
//! `block_size` granularity; each node holds one physical KV block index.

use std::collections::HashMap;

mod evict;
mod nvme;
mod nvme_index;

pub(super) use nvme_index::NvmeIndex;

type NodeId = usize;

pub(super) struct RadixNode {
    /// Children keyed by the token chunk (block_size tokens).
    children: HashMap<Vec<u32>, NodeId>,
    /// Physical KV cache block index stored at this node. `u32::MAX` on a
    /// non-root node means the block is ON DISK at `nvme_slot` (spill tier).
    block_idx: u32,
    /// NVMe spill-tier record slot; `u32::MAX` when the node has no record.
    /// A resident node may have one too (a record kept across its restore).
    nvme_slot: u32,
    /// `--high-speed-swap` disk-block ID (Phase 6.1.e). `u32::MAX` when HSS
    /// is not in use. The cache holds a refcount on this disk_id (bumped
    /// at insert time, dropped at evict time) parallel to its physical-
    /// block ref bookkeeping.
    disk_block_id: u32,
    /// Context hash: FNV-1a chain of this node's tokens + parent's context_hash.
    /// Ensures a block is only reused when the ENTIRE causal prefix matches,
    /// preventing cross-request KV contamination (vLLM-style parent-hash chain).
    context_hash: u64,
    /// Number of active sequences referencing this node.
    ref_count: u32,
    /// Monotonic access counter for LRU eviction.
    last_access: u64,
    /// Parent node (None for root).
    parent: Option<NodeId>,
    /// Token chunk that leads from parent to this node (for cleanup).
    parent_key: Option<Vec<u32>>,
}

/// Combine a parent context hash with a token chunk to produce a child context hash.
/// Uses FNV-1a for speed (~ns per block). The chain guarantees that two nodes with
/// identical token chunks but different prefixes get different context_hashes.
fn context_hash_combine(parent_hash: u64, tokens: &[u32]) -> u64 {
    let mut h = parent_hash;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

/// Inner state protected by mutex.
pub(super) struct RadixTreeInner {
    nodes: Vec<RadixNode>,
    /// Indices of deleted nodes available for reuse.
    free_nodes: Vec<NodeId>,
    /// Task #24 (adapter-correct KV): one radix ROOT per stable `adapter_id`.
    /// Adapter A's and adapter B's node chains live in physically disjoint
    /// subtrees, so a cross-adapter walk can never reach the wrong adapter's
    /// blocks (correct MISS) and an insert under one root can never clobber
    /// another adapter's node (the children map is keyed by token chunk, so
    /// seeding the context-hash alone would NOT prevent that cross-adapter
    /// insert-collision — disjoint roots do). `adapter_id == 0` is the base /
    /// no-adapter root created in `new()`, so base keying is byte-identical to
    /// the pre-LoRA single-root tree.
    roots: HashMap<u64, NodeId>,
    access_counter: u64,
    /// NVMe spill tier (`None` = off: eviction deletes, the default).
    pub(super) nvme: Option<NvmeIndex>,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        let root = RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX, // sentinel — root has no block
            nvme_slot: u32::MAX,
            disk_block_id: u32::MAX, // sentinel — root has no disk slot either
            context_hash: 0,         // root context hash is 0
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
        };
        let mut roots = HashMap::new();
        roots.insert(0u64, 0usize); // base / no-adapter root
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            roots,
            access_counter: 0,
            nvme: None,
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Read-side root for `adapter_id`: `None` when this adapter has no cached
    /// blocks yet (walk/inc_refs/dec_refs then no-op → clean MISS).
    fn root_for_read(&self, adapter_id: u64) -> Option<NodeId> {
        self.roots.get(&adapter_id).copied()
    }

    /// Insert-side root for `adapter_id`: creates a fresh disjoint root the first
    /// time an adapter caches anything. Roots carry `block_idx == u32::MAX` and
    /// `parent == None`, so they are excluded from eviction and `num_entries`.
    fn root_for_insert(&mut self, adapter_id: u64) -> NodeId {
        if let Some(&id) = self.roots.get(&adapter_id) {
            return id;
        }
        let id = self.alloc_node(RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,
            nvme_slot: u32::MAX,
            disk_block_id: u32::MAX,
            context_hash: 0,
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
        });
        self.roots.insert(adapter_id, id);
        id
    }

    pub(super) fn alloc_node(&mut self, node: RadixNode) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// Walk the tree matching whole `block_size` token chunks.
    /// Returns (matched_blocks, matched_disk_block_ids, matched_tokens). The
    /// disk-id vec parallels matched_blocks; entries are `u32::MAX` on nodes
    /// that were inserted before HSS engaged (caller's responsibility to
    /// filter / treat MAX as "no disk-side ref to bump"). Snapshot lookup is
    /// now handled by the separate SsmSnapshotIndex after the walk completes.
    ///
    /// `matched_tokens` is always a multiple of `block_size`: a match never
    /// ends inside a cached block. The sequence that acquired such a block
    /// would write the rest of it (its own decode rows, and every row of a
    /// speculative verify) under the block's other holders. The trailing
    /// `tokens.len() % block_size` tokens are recomputed into a block the
    /// sequence owns alone.
    pub(super) fn walk(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return (Vec::new(), Vec::new(), 0), // adapter has nothing cached
        };
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_tokens = 0;
        let mut parent_ctx_hash: u64 = 0; // root context hash

        let num_full_blocks = tokens.len() / block_size;
        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0
                        && self.nodes[child].block_idx != u32::MAX =>
                {
                    // Context chain matches AND block is still live (resident,
                    // not spilled to NVMe) — safe to reuse.
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_tokens += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break, // Token match but context mismatch or freed block — stop.
            }
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// Whether the live cached path through the block-aligned prefix
    /// `tokens[..depth]` continues with a full block other than
    /// `tokens[depth..depth + block_size]`, i.e. another request diverged
    /// from this one exactly there. Read-only (no refs, no recency).
    pub(super) fn forks_at(
        &self,
        tokens: &[u32],
        depth: usize,
        block_size: usize,
        adapter_id: u64,
    ) -> bool {
        let (Some(mut current), Some(ours)) = (
            self.root_for_read(adapter_id),
            tokens.get(depth..depth + block_size),
        ) else {
            return false;
        };
        if block_size == 0 || !depth.is_multiple_of(block_size) {
            return false;
        }
        for chunk in tokens[..depth].chunks_exact(block_size) {
            match self.nodes[current].children.get(chunk) {
                Some(&child) if self.nodes[child].ref_count > 0 => current = child,
                _ => return false,
            }
        }
        self.nodes[current]
            .children
            .iter()
            .any(|(key, &child)| key.as_slice() != ours && self.nodes[child].ref_count > 0)
    }

    /// Increment ref_count on all nodes along the matched path.
    pub(super) fn inc_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        num_matched: usize,
        adapter_id: u64,
    ) {
        let access = self.next_access();
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
        let num_blocks = num_matched / block_size;

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                self.nodes[child].ref_count += 1;
                self.nodes[child].last_access = access;
                current = child;
            } else {
                break;
            }
        }
    }

    /// Decrement ref_count on up to `matched_tokens` along the matched path.
    pub(super) fn dec_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
        let num_full_blocks = (matched_tokens / block_size).min(tokens.len() / block_size);

        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                if self.nodes[child].ref_count > 0 {
                    self.nodes[child].ref_count -= 1;
                }
                current = child;
            } else {
                break;
            }
        }
    }

    /// Insert the whole blocks of `tokens` into the tree. Skips blocks that
    /// already exist. Snapshot storage is handled externally by
    /// SsmSnapshotIndex.
    ///
    /// The block holding the trailing `tokens.len() % block_size` tokens is
    /// never published: the inserting sequence still writes it, and so would
    /// any sequence that later acquired it. It enters the cache once every
    /// position in it is a committed token, through a later insert.
    ///
    /// `matched_tokens` is the prefix length the inserting sequence already
    /// acquired via `lookup()`'s `inc_refs` (those nodes' ref_count was
    /// bumped by walk and will be balanced by the eventual `release`). For
    /// blocks past that offset the inserting sequence holds no walk-ref but
    /// `release` will still dec_ref them — so we pre-bump here to keep the
    /// cache's baseline ref (=1) alive across the release. Without this the
    /// very first sequence to cache a prompt immediately renders its own
    /// cache entry un-findable on exit (`walk` requires `ref_count > 0`).
    pub(super) fn insert(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) -> crate::prefix_cache::InsertAcquired {
        let access = self.next_access();
        let mut current = self.root_for_insert(adapter_id);
        let mut parent_ctx_hash: u64 = 0; // root context hash
        let num_full_blocks = tokens.len() / block_size;
        let num_blocks = num_full_blocks.min(block_table.len());
        // disk_block_ids may be empty (HSS off) or parallel to block_table.
        // When empty, every node's disk_block_id stays at u32::MAX sentinel.
        let hss_active = !disk_block_ids.is_empty();
        debug_assert!(
            !hss_active || disk_block_ids.len() == block_table.len(),
            "disk_block_ids length mismatch: {} vs {}",
            disk_block_ids.len(),
            block_table.len(),
        );

        // Issue #17: collect disk_ids on which the cache newly takes an
        // ownership ref. Caller `inc_disk_ref`s these so the swap allocator's
        // refcount tracks the cache's reachability. Two acquisition events:
        //   1. New node creation with a real disk_id.
        //   2. Existing node, MAX → real disk_id (first-time HSS population).
        // Re-insertion of an already-cached (node, disk_id) pair is NOT an
        // acquisition — the cache's ref already covers it.
        let mut newly_acquired: Vec<u32> = Vec::new();
        // Blocks the cache starts storing here; the caller takes exactly one
        // KV ref each. See `InsertAcquired`.
        let mut newly_owned_blocks: Vec<u32> = Vec::new();

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let ctx_hash = context_hash_combine(parent_ctx_hash, chunk);
            let token_start = i * block_size;
            let is_seq_owned = token_start >= matched_tokens;
            let disk_id = if hss_active {
                disk_block_ids[i]
            } else {
                u32::MAX
            };

            if let Some(&child) = self.nodes[current].children.get(chunk) {
                // A node spilled to NVMe takes this sequence's freshly computed
                // block instead (before its LRU key changes below).
                if self.nvme_rehome(child, block_table[i], false) {
                    newly_owned_blocks.push(block_table[i]);
                }
                // Node exists — update access time, context_hash, and ensure
                // the cache's ref is still held (ref_count >= 1).
                self.nodes[child].last_access = access;
                self.nodes[child].context_hash = ctx_hash;
                if self.nodes[child].ref_count == 0 {
                    self.nodes[child].ref_count = 1; // restore cache's ref
                }
                if is_seq_owned {
                    self.nodes[child].ref_count += 1;
                }
                // First-time HSS population on a pre-existing node: stash
                // the disk_id. (This happens when the cache was built
                // pre-HSS and a later HSS-enabled sequence touches the
                // same prefix — defensive, shouldn't fire under normal
                // operation since cache_blocks_per_seq is set at startup.)
                if hss_active && self.nodes[child].disk_block_id == u32::MAX && disk_id != u32::MAX
                {
                    self.nodes[child].disk_block_id = disk_id;
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child;
            } else {
                let node = RadixNode {
                    children: HashMap::new(),
                    block_idx: block_table[i],
                    nvme_slot: u32::MAX,
                    disk_block_id: disk_id,
                    context_hash: ctx_hash,
                    ref_count: if is_seq_owned { 2 } else { 1 },
                    last_access: access,
                    parent: Some(current),
                    parent_key: Some(chunk.to_vec()),
                };
                let child_id = self.alloc_node(node);
                newly_owned_blocks.push(block_table[i]);
                self.nodes[current]
                    .children
                    .insert(chunk.to_vec(), child_id);
                if hss_active && disk_id != u32::MAX {
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child_id;
            }
        }

        crate::prefix_cache::InsertAcquired {
            disk_block_ids: newly_acquired,
            blocks: newly_owned_blocks,
        }
    }
}
