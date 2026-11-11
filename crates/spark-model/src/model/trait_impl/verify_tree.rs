// SPDX-License-Identifier: AGPL-3.0-only

//! Exact DFlash tree verify on the dense hybrid qwen3_5
//! (`ATLAS_DFLASH_TREE`, `speculative::draft_tree`): the K=γ verify
//! (`verify_d.rs`) over a draft tree's rows, then the accepted path's
//! commit.
//!
//! The verify is the chain's, row by row, with three differences, none of
//! which touches a row's arithmetic:
//!
//! * Attention: row `r` sits at position `prefix + depth(r)` and reads its
//!   own block table: the sequence's blocks, then private blocks from the
//!   block holding the prefix's end. Each attention layer writes the rows'
//!   K/V at their private slots and copies into every row's private blocks
//!   the prefix entries they share with the sequence's block and the K/V of
//!   the row's ancestors at the ancestors' positions (`kv_slot_copy_bf16`).
//!   So row `r`'s paged attention reads, slot for slot, what serial decode
//!   reads after the path ending at `r`, over the same key count (the same
//!   split bounds).
//! * GDN: every row continues its parent's recurrence and conv window
//!   (`qwen35_gdn_verify_tree`, needs `ATLAS_QWEN38_VERIFY_FUSE=1`).
//! * The graph is keyed apart from the chain's (`TREE_KEY`); a tree's
//!   positions, slots, tables, parents and copies live in device memory
//!   written before the replay, so one graph serves every tree of a width.
//!
//! The commit (`commit_dflash_tree`) copies the accepted path's entries from
//! the deepest accepted row's private blocks into the sequence's slots,
//! frees the private blocks, takes the GDN state after the path
//! (`commit_accepted_rows`: the last row's rollback slot, or the path
//! replayed from H0 under `ATLAS_QWEN38_VERIFY_DEFER`),
//! moves the accepted rows' drafter-context captures and logits into path
//! order, and leaves `seq.tokens` / `seq.seq_len` as the chain would.

use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::DevicePtr;
use std::sync::OnceLock;

use super::super::block_table_upload::upload_block_table_rows;
use super::super::types::TransformerModel;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::speculative::draft_tree::{DraftTree, MAX_ROWS, TreeDev};
use crate::traits::SequenceState;

/// Graph-cache key bit that keeps a tree verify's graphs apart from the
/// chain's of the same width.
pub(super) const TREE_KEY: usize = 1 << 20;
/// KV copies a tree verify can need per attention layer: every row's share
/// of the prefix's last block and every row's ancestors.
const KV_MAX: usize = 512;
/// Descriptor layout: parents `[MAX_ROWS]` i32 | count u32 at 64 | entries
/// `[KV_MAX]` (i64 src, i64 dst) at 128.
const DESC_COUNT: usize = 64;
const DESC_ENTRIES: usize = 128;
const DESC_BYTES: usize = DESC_ENTRIES + KV_MAX * 16;

/// One planned tree verify: per-row metadata and the private blocks.
pub(super) struct TreePlan {
    pub positions: Vec<u32>,
    pub slots: Vec<i64>,
    pub seq_lens: Vec<i32>,
    pub tables: Vec<Vec<u32>>,
    pub dev: TreeDev,
}

/// What the commit needs from the verify it follows.
struct Pending {
    slot_idx: usize,
    pre_len: usize,
    /// `seq.tokens.len()` before the verify pushed the rows.
    tokens_len: usize,
    tree: DraftTree,
    /// Row `r`'s private blocks, for logical blocks `first_block..`.
    private: Vec<Vec<u32>>,
    first_block: usize,
    /// `ATLAS_DFLASH_TREE_CHECK`: the serial reference.
    check: Option<super::verify_tree_check::TreeSerial>,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);

fn le<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a plain integer type here (u32, i32, i64); the slice is
    // fully initialised and viewed for exactly its byte length.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Where a tree verify's rows live in the KV cache.
#[derive(Debug)]
pub(super) struct TreeLayout {
    pub positions: Vec<u32>,
    /// Each row's own slot (in its private blocks).
    pub slots: Vec<i64>,
    pub seq_lens: Vec<i32>,
    /// Each row's block table: the sequence's, its private blocks from the
    /// block holding the prefix's end.
    pub tables: Vec<Vec<u32>>,
    /// (src slot, dst slot) copies after the rows' KV writes: each row's
    /// share of the prefix in its first private block, then its ancestors'
    /// K/V at their positions.
    pub entries: Vec<i64>,
}

/// The layout of `tree` verified after `pre_len` committed positions, with
/// `private[r]` row `r`'s blocks for logical blocks `pre_len / bs..`.
pub(super) fn tree_layout(
    tree: &DraftTree,
    pre_len: usize,
    bs: usize,
    block_table: &[u32],
    private: &[Vec<u32>],
) -> TreeLayout {
    let k = tree.rows();
    let b0 = pre_len / bs;
    let real = |p: usize| (block_table[p / bs] as i64) * bs as i64 + (p % bs) as i64;
    let own = |r: usize, p: usize| (private[r][p / bs - b0] as i64) * bs as i64 + (p % bs) as i64;
    let pos = |r: usize| pre_len + tree.depth[r] as usize;
    let mut entries = Vec::new();
    for r in 0..k {
        for p in b0 * bs..pre_len {
            entries.extend([real(p), own(r, p)]);
        }
        for &a in tree.path_to(r).iter().filter(|&&a| a != r) {
            entries.extend([own(a, pos(a)), own(r, pos(a))]);
        }
    }
    TreeLayout {
        positions: (0..k).map(|r| pos(r) as u32).collect(),
        slots: (0..k).map(|r| own(r, pos(r))).collect(),
        seq_lens: (0..k).map(|r| (pos(r) + 1) as i32).collect(),
        tables: (0..k)
            .map(|r| {
                let mut t = block_table.to_vec();
                for (i, &b) in private[r].iter().enumerate() {
                    t[b0 + i] = b;
                }
                t
            })
            .collect(),
        entries,
    }
}

/// The commit's copies: positions `pre_len..=pre_len + accepted` from the
/// deepest accepted row's private blocks (`deepest`, for logical blocks
/// `first_block..`) into the sequence's slots.
pub(super) fn commit_entries(
    pre_len: usize,
    bs: usize,
    block_table: &[u32],
    deepest: &[u32],
    first_block: usize,
    accepted: usize,
) -> Vec<i64> {
    (pre_len..=pre_len + accepted)
        .flat_map(|pos| {
            [
                deepest[pos / bs - first_block] as i64 * bs as i64 + (pos % bs) as i64,
                block_table[pos / bs] as i64 * bs as i64 + (pos % bs) as i64,
            ]
        })
        .collect()
}

impl TransformerModel {
    fn tree_desc(&self) -> Result<DevicePtr> {
        static DESC: OnceLock<u64> = OnceLock::new();
        if let Some(&p) = DESC.get() {
            return Ok(DevicePtr(p));
        }
        let p = self.gpu.alloc(DESC_BYTES)?;
        Ok(DevicePtr(*DESC.get_or_init(|| p.0)))
    }

    /// Whether this model can verify a tree exactly.
    pub(super) fn dflash_tree_supported_impl(&self) -> bool {
        let kv = self.kv_cache.lock();
        self.config.model_type == "qwen3_5"
            && super::super::exact_verify::verify_bit_exact(&self.levers)
            && ops::qwen38_decode_fuse::verify_on().unwrap_or(false)
            && !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
            && self.comm.is_none()
            && self.lora.is_none()
            && self.lightning_dspark_identity.policy().is_none()
            && self.proposer.is_some()
            && self.dflash_hidden_save.is_some()
            && kv.dtype() == spark_runtime::kv_cache::KvCacheDtype::Bf16
            && kv.config().cache_blocks_per_seq.is_none()
            && kv.latent_shard().is_none()
            && ops::kv_slot_copy_available(self.gpu.as_ref())
    }

    /// Plan, upload and run the tree verify. Returns each row's argmax.
    pub(super) fn decode_verify_dflash_tree_impl(
        &self,
        tree: &DraftTree,
        seq: &mut SequenceState,
    ) -> Result<Vec<u32>> {
        tree.validate()?;
        let k = tree.rows();
        ensure!(
            (1..=MAX_ROWS).contains(&k),
            "tree verify: {k} rows (serves 1..={MAX_ROWS})"
        );
        self.drop_stale_tree()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let check = if super::verify_tree_check::active() {
            Some(self.tree_check_serial(tree, seq)?)
        } else {
            None
        };
        // ATLAS_QWEN38_VERIFY_DEFER: mark the step's GDN layers, outside
        // any capture (the chain's verify entry does the same).
        self.mark_gdn_deferred_commit(seq, k)?;
        let stream = self.gpu.default_stream();
        let pre_len = seq.seq_len;
        let (plan, private, first_block) = {
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            // The chain's reservation: every position a row can commit.
            for t in 0..k {
                self.reserve_decode_blocks(seq, (pre_len + t) / bs, &mut kv, stream)?;
            }
            let b0 = pre_len / bs;
            let mut private: Vec<Vec<u32>> = Vec::with_capacity(k);
            for r in 0..k {
                let last = (pre_len + tree.depth[r] as usize) / bs;
                let mut blocks = Vec::with_capacity(last + 1 - b0);
                for b in b0..=last {
                    match kv.alloc_block_at(b) {
                        Ok(id) => blocks.push(id),
                        Err(e) => {
                            for id in private.iter().flatten().chain(&blocks) {
                                kv.free_block(*id);
                            }
                            return Err(e.context("tree verify: private KV blocks"));
                        }
                    }
                }
                private.push(blocks);
            }
            let lay = tree_layout(tree, pre_len, bs, &seq.block_table, &private);
            let count = lay.entries.len() / 2;
            ensure!(
                count <= KV_MAX,
                "tree verify: {count} KV copies (serves {KV_MAX})"
            );
            let desc = self.tree_desc()?;
            let parents: Vec<i32> = tree.parent.clone();
            self.gpu.copy_h2d_async(le(&parents), desc, stream)?;
            self.gpu
                .copy_h2d_async(le(&[count as u32]), desc.offset(DESC_COUNT), stream)?;
            if count > 0 {
                self.gpu
                    .copy_h2d_async(le(&lay.entries), desc.offset(DESC_ENTRIES), stream)?;
            }
            let plan = TreePlan {
                positions: lay.positions,
                slots: lay.slots,
                seq_lens: lay.seq_lens,
                tables: lay.tables,
                dev: TreeDev {
                    parents: desc,
                    kv_entries: desc.offset(DESC_ENTRIES),
                    kv_count: desc.offset(DESC_COUNT),
                    kv_max: KV_MAX as u32,
                },
            };
            (plan, private, b0)
        };
        *PENDING.lock() = Some(Pending {
            slot_idx: seq.slot_idx,
            pre_len,
            tokens_len: seq.tokens.len(),
            tree: tree.clone(),
            private,
            first_block,
            check: None,
        });
        let rows = self.decode_verify_graphed_kgamma_dispatch_with(
            &tree.tokens,
            seq,
            stream,
            None,
            Some(&plan),
        );
        if rows.is_err() {
            self.drop_stale_tree()?;
            return rows;
        }
        if let Some(serial) = check {
            self.tree_check_rows(&serial, pre_len)?;
            if let Some(p) = PENDING.lock().as_mut() {
                p.check = Some(serial);
            }
        }
        rows
    }

    /// Free a planned tree's private blocks that no commit took (an error
    /// between the verify and its commit).
    fn drop_stale_tree(&self) -> Result<()> {
        if let Some(p) = PENDING.lock().take() {
            let mut kv = self.kv_cache.lock();
            for &b in p.private.iter().flatten() {
                kv.free_block(b);
            }
        }
        Ok(())
    }

    /// K-entry metadata of a tree verify (the `kgamma_upload_meta` layout).
    pub(super) fn kgamma_upload_meta_tree(
        &self,
        seq: &SequenceState,
        plan: &TreePlan,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let k = plan.positions.len();
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        self.gpu
            .copy_h2d_async(le(&plan.positions), meta_base, stream)?;
        self.gpu
            .copy_h2d_async(le(&plan.slots), meta_base.offset(256), stream)?;
        self.gpu
            .copy_h2d_async(le(&plan.seq_lens), meta_base.offset(512), stream)?;
        upload_block_table_rows(
            self.gpu.as_ref(),
            meta_base.offset(768),
            max_blocks as usize,
            plan.tables.iter().map(|t| &t[..]),
            stream,
        )?;
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, k, meta_base.offset(128), stream)?;
        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
            seq_slot,
            moe_row_adapter: DevicePtr::NULL,
        })
    }

    /// Move logits rows (and final-hidden rows) `rows[i]` to row `i`, the
    /// order the scheduler's per-row pick pipeline reads. `rows` is a path:
    /// `rows[i] >= i`, so copying in order never reads an overwritten row.
    pub(super) fn dflash_tree_gather_rows_impl(&self, rows: &[usize]) -> Result<()> {
        let stream = self.gpu.default_stream();
        let (vocab, h) = (self.config.vocab_size, self.config.hidden_size);
        for (i, &r) in rows.iter().enumerate() {
            ensure!(r >= i, "tree gather: row {r} before position {i}");
            if r == i {
                continue;
            }
            let lb = vocab * 2;
            self.gpu.copy_d2d_async(
                self.buffers.logits().offset(r * lb),
                self.buffers.logits().offset(i * lb),
                lb,
                stream,
            )?;
            let hb = h * 2;
            self.gpu.copy_d2d_async(
                self.buffers.norm_output().offset(r * hb),
                self.buffers.norm_output().offset(i * hb),
                hb,
                stream,
            )?;
        }
        Ok(())
    }

    /// Commit the accepted `path` (rows, root first) of the tree verify that
    /// just ran on `seq`.
    pub(super) fn commit_dflash_tree_impl(
        &self,
        seq: &mut SequenceState,
        path: &[usize],
    ) -> Result<()> {
        let Some(p) = PENDING.lock().take() else {
            bail!("tree commit without a tree verify");
        };
        let stream = self.gpu.default_stream();
        let k = p.tree.rows();
        let free_private = |kv: &mut spark_runtime::kv_cache::PagedKvCache| {
            for &b in p.private.iter().flatten() {
                kv.free_block(b);
            }
        };
        if p.slot_idx != seq.slot_idx || path.is_empty() || path[0] != 0 {
            free_private(&mut self.kv_cache.lock());
            bail!("tree commit: not the verify's sequence or path");
        }
        for w in path.windows(2) {
            ensure!(
                p.tree.parent[w[1]] == w[0] as i32,
                "tree commit: {path:?} is not a root path"
            );
        }
        let accepted = path.len() - 1;
        let deepest = path[accepted];
        let pre_len = p.pre_len;
        {
            // KV: the deepest row's private blocks hold the whole path at
            // its positions; copy them into the sequence's slots.
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            let entries = commit_entries(
                pre_len,
                bs,
                &seq.block_table,
                &p.private[deepest],
                p.first_block,
                accepted,
            );
            let desc = self.tree_desc()?;
            self.gpu.copy_h2d_async(
                le(&[(entries.len() / 2) as u32]),
                desc.offset(DESC_COUNT),
                stream,
            )?;
            self.gpu
                .copy_h2d_async(le(&entries), desc.offset(DESC_ENTRIES), stream)?;
            let n_elems = (self.config.num_key_value_heads * self.config.head_dim) as u32;
            for layer in 0..kv.num_layers() {
                ops::kv_slot_copy_bf16(
                    self.gpu.as_ref(),
                    kv.k_pool_ptr(layer),
                    kv.v_pool_ptr(layer),
                    desc.offset(DESC_ENTRIES),
                    desc.offset(DESC_COUNT),
                    (entries.len() / 2) as u32,
                    n_elems,
                    stream,
                )?;
            }
            // The copies read the private blocks: free them after.
            self.gpu.synchronize(stream)?;
            free_private(&mut kv);
        }
        // GDN: the state after the accepted path (row `deepest`'s rollback
        // slot or, as the last row, the live state; replayed from H0 when the
        // verify deferred).
        self.commit_accepted_rows_dispatch(seq, path, k)?;
        // Drafter context: the accepted rows' captures in path order.
        if let Some(base) = self.dflash_hidden_save
            && !self.dflash_capture_layers.is_empty()
        {
            let row_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
            for (i, &r) in path.iter().enumerate() {
                if r != i {
                    self.gpu.copy_d2d_async(
                        base.offset(r * row_bytes),
                        base.offset(i * row_bytes),
                        row_bytes,
                        stream,
                    )?;
                }
            }
        }
        // Tokens: the prefix, then the path (the verify pushed the rows).
        seq.tokens.truncate(p.tokens_len);
        for &r in path {
            seq.tokens.push(p.tree.tokens[r]);
        }
        seq.seq_len = pre_len + accepted + 1;
        if let Some(serial) = &p.check {
            self.tree_check_commit(serial, seq, path, pre_len)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "verify_tree_tests.rs"]
mod tests;
