// SPDX-License-Identifier: AGPL-3.0-only

//! Serving-path side of the prefix cache's NVMe spill tier
//! (`ATLAS_KV_NVME_DIR`; see `spark_runtime::prefix_cache::nvme`): paging an
//! evicted prefix back in before the prefix lookup. A restore is rank-local
//! and issues no collective: the ranks then agree on the match (F83) and, with
//! the tier on, on the Marconi restore depth (`prefill_b/pc_policy.rs`).
//!
//! Only the per-stream prefill (`prefill_b_prefix_lookup`) restores. The
//! batched admission (`prefill_b_reserve_batched_prefix_matches`, single-rank
//! worlds only) looks the tree up as it is: concurrent arrivals whose prefix
//! is on disk are admitted cold and recompute it.
//!
//! Construction and env parsing live in `factory/build/kv_nvme.rs`.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::{NvmePrefixTier, PrefixCache, RestorePlan};

use super::block_mgmt::alloc_block_evicting;
use super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// The NVMe prefix tier, when it is enabled (the same on every rank:
    /// startup refuses a pair that differs).
    pub(in crate::model) fn nvme_tier(&self) -> Option<&dyn NvmePrefixTier> {
        self.prefix_cache.nvme().filter(|t| t.is_enabled())
    }

    /// How many blocks of `plan.disk` to read back: all of them, or none.
    ///
    /// The rule: after a restore the prefill must find exactly what it would
    /// have found had nothing been evicted, and must not page in what it is
    /// about to overwrite.
    ///
    /// Pure attention: all (every restored block is a skipped prefill block).
    /// Hybrid SSM (GLM-5.3's KDA layers): a prefill resumes from an SSM
    /// snapshot anchor.
    ///
    /// * With a usable anchor (resident OR spill-tiered) the pass replays
    ///   `[anchor, matched)` under the KV write floor: those rows are READ
    ///   from the cache, not rewritten (`prefill_b/forward_layers.rs`; GLM
    ///   honours the floor since `integ/next`). So the whole run comes back,
    ///   past the anchor too — stopping at the anchor would make the pass
    ///   recompute rows a resident cache serves as cached, and its output
    ///   could differ bitwise from the run that was never evicted.
    /// * Without one the pass recomputes the prompt and rewrites every
    ///   matched row ("Prefix cache hit … but no SSM snapshot"): nothing is
    ///   read back — unless `ATLAS_GLM_PC_WRITE_FLOOR` keeps the matched rows
    ///   of that pass as cached, too.
    ///
    /// "Usable" mirrors the gates the lookup applies to the anchor afterwards:
    /// the Marconi minimum, the snapshot tier's fault-in minimum
    /// (`ssm_fault_in`; applied to a resident anchor too — a prefix that short
    /// is not worth a restore), and the exact-hit bypass (an anchor AT the
    /// prompt's end is declined unless `ATLAS_MARCONI_EXACT=1`;
    /// `pc_policy::marconi_restorable`).
    fn nvme_blocks_worth_restoring(
        &self,
        tier: &dyn NvmePrefixTier,
        plan: &RestorePlan,
        tokens: &[u32],
        seq: &SequenceState,
        bs: usize,
    ) -> usize {
        let run = plan.disk.len();
        if self.config.num_ssm_layers() == 0 {
            return run;
        }
        let limit = plan.resident_tokens + run * bs;
        let depth = tier.snapshot_anchor_depth(tokens, limit, seq.session_hash, seq.adapter_id);
        let fault_min = match self.ssm_tier_store {
            Some(_) => super::trait_impl::ssm_fault_in::fault_in_min_tokens(),
            None => 0,
        };
        if restore_reads_cached_rows(
            depth,
            tokens.len(),
            crate::model::mtp_carry::marconi_min_tokens().max(fault_min),
            std::env::var("ATLAS_MARCONI_EXACT").as_deref() == Ok("1"),
            self.pc_write_floor_keeps_matched(),
        ) {
            run
        } else {
            0
        }
    }

    /// Page the on-disk continuation of `tokens`' cached prefix back into
    /// fresh KV blocks, so the resident-only `lookup` that follows matches it.
    /// No-op when the tier is off. See [`restore_prefix`].
    pub(in crate::model) fn nvme_restore_prefix(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) {
        let Some(tier) = self.nvme_tier() else {
            return;
        };
        let bs = kv_cache.block_size();
        let t0 = std::time::Instant::now();
        let Some(r) = restore_prefix(
            self.prefix_cache.as_ref(),
            kv_cache,
            self.gpu.as_ref(),
            tokens,
            seq.adapter_id,
            stream,
            |plan| self.nvme_blocks_worth_restoring(tier, plan, tokens, seq, bs),
        ) else {
            return;
        };
        if r.wanted == 0 {
            tracing::debug!(
                "NVMe prefix restore skipped: {} on-disk blocks past {} resident tokens but no \
                 usable SSM anchor — the prefill rewrites its whole prefix",
                r.on_disk,
                r.resident_tokens,
            );
            return;
        }
        let s = tier.nvme_stats();
        let io = kv_cache.nvme_io_stats();
        let ms = |micros: u64| micros as f64 / 1e3;
        let read_ms = ms(r.read_micros);
        // Blocks per device copy run: how contiguous the pool was (1.0 = a
        // copy per region per block, as on the synchronous path).
        let per_run = |blocks: u64, runs: u64| blocks as f64 / runs.max(1) as f64;
        tracing::info!(
            "NVMe prefix restore: {}/{} blocks ({} tokens after {} resident) in {:.1} ms{}; \
             tier {}/{} slots, {} spills, {} restores, {} failures; evict {:.1} ms + read {:.1} ms \
             ({:.0} MB/s); spill path ({}): {} blocks in {:.1} ms on the serving thread, \
             {} evictions without a write; this restore: writer wait {:.1} ms in evict, flush \
             {:.1} ms before read, {:.1} blocks/scatter run, {} spilled at {:.1} blocks/gather run; \
             spill path waited {:.1} ms for the writer",
            r.restored,
            r.wanted,
            r.restored * bs,
            r.resident_tokens,
            t0.elapsed().as_secs_f64() * 1e3,
            if r.failed {
                " — a record FAILED (I/O or verification): recomputing from there"
            } else if r.allocated < r.wanted {
                " — KV pool exhausted: recomputing the rest"
            } else {
                ""
            },
            s.slots_used,
            s.max_slots,
            s.spills,
            s.restores,
            s.spill_failures + s.restore_failures,
            ms(r.evict_micros),
            read_ms,
            (r.restored * kv_cache.nvme_record_bytes()) as f64 / 1e3 / read_ms.max(1e-3),
            if io.fast { "fast" } else { "sync" },
            io.spilled_blocks,
            ms(io.spill_micros),
            s.clean_evictions,
            ms(r.io.spill_wait_micros),
            ms(r.io.flush_micros),
            per_run(r.io.restored_blocks, r.io.scatter_runs),
            r.io.spilled_blocks,
            per_run(r.io.spilled_blocks, r.io.gather_runs),
            ms(io.spill_wait_micros),
        );
    }
}

impl TransformerModel {
    /// A prefill that resumes at `skip_to` under a `matched`-token cached
    /// prefix may recompute rows `[skip_to, matched)` into the shared blocks
    /// (a full-prompt hit: its last row), with equivalent values, not the
    /// recorded bytes (`prefix_share`). A record one of those blocks kept from
    /// its restore (`ATLAS_GLM_NVME_KEEP`) would then be stale: release it, so
    /// the block's next eviction writes what the block holds, as without KEEP.
    ///
    /// Deliberately not exact: a replay under the KV write floor leaves those
    /// rows as cached, and which passes honour the floor is the layers'
    /// business (`ATLAS_GLM_KV_WRITE_FLOOR_LEGACY`, the recompute-all pass).
    /// Giving up a record that was still good costs one write of that block —
    /// the few blocks between a turn's anchor and its match, not the prefix
    /// below the anchor, which no pass touches.
    /// Same inputs on every rank (the match and the restore depth are agreed).
    pub(in crate::model) fn nvme_forget_rewritten(
        &self,
        tokens: &[u32],
        adapter_id: u64,
        skip_to: usize,
        matched: usize,
        bs: usize,
    ) {
        let blocks = skip_to.min(tokens.len().saturating_sub(1)) / bs..matched / bs;
        if !blocks.is_empty()
            && let Some(tier) = self.nvme_tier()
        {
            tier.forget_kept(tokens, bs, adapter_id, blocks);
        }
    }
}

/// Spill writes that did not reach the disk: the tree forgets those nodes (a
/// plain eviction). The fast path reports a failure after the fact, so this
/// also runs around a restore.
pub(crate) fn drop_failed_spills(
    failed: &[spark_runtime::prefix_cache::SpillOrder],
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
) {
    if !failed.is_empty()
        && let Some(tier) = prefix_cache.nvme()
    {
        for block in tier.spill_failed(failed) {
            kv_cache.return_evicted_block(block);
        }
    }
}

/// Whether the prefill that follows will READ its cached prefix rather than
/// rewrite all of it (see `nvme_blocks_worth_restoring`): it restores the
/// snapshot anchor at `anchor` tokens (0 = none; at least `min_anchor` deep,
/// and not the exact-hit anchor at the prompt's end, which is bypassed
/// without `exact`), or the write floor keeps the matched rows of a
/// recompute (`floor_keeps_matched`).
fn restore_reads_cached_rows(
    anchor: usize,
    total: usize,
    min_anchor: usize,
    exact: bool,
    floor_keeps_matched: bool,
) -> bool {
    let usable = anchor > 0 && anchor >= min_anchor && (anchor < total || exact);
    usable || floor_keeps_matched
}

/// What one [`restore_prefix`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RestoreOutcome {
    pub resident_tokens: usize,
    pub on_disk: usize,
    pub wanted: usize,
    pub allocated: usize,
    pub restored: usize,
    pub failed: bool,
    /// Allocating the target blocks — which, on a full pool, is spilling that
    /// many victims (the gather, plus `io.spill_wait_micros` for the writer).
    pub evict_micros: u64,
    /// Reading, verifying and scattering the records — NOT the wait for the
    /// victims' queued writes before the first read (`io.flush_micros`).
    pub read_micros: u64,
    /// What this restore added to the cache's I/O counters.
    pub io: spark_runtime::kv_cache::NvmeIoStats,
}

/// The restore cycle: plan (pins the path) → allocate up to `want(plan)`
/// blocks (evicting/spilling others; the pin protects this path) → read and
/// verify → promote in the tree → hand back everything not adopted.
///
/// Best effort by construction: any shortfall (no free blocks, I/O error, a
/// record failing verification) only shortens the match; the prefill
/// recomputes the rest, and unverified bytes are never adopted. `None` when
/// there is nothing on disk to restore (or the tier is off).
pub(crate) fn restore_prefix(
    prefix_cache: &dyn PrefixCache,
    kv_cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
    tokens: &[u32],
    adapter_id: u64,
    stream: u64,
    want: impl FnOnce(&RestorePlan) -> usize,
) -> Option<RestoreOutcome> {
    let tier = prefix_cache.nvme().filter(|t| t.is_enabled())?;
    if !kv_cache.nvme_attached() {
        return None;
    }
    // Fast path: a write-behind failure since the last eviction. Forget those
    // nodes before planning over them (a record still in flight is caught by
    // its tag instead).
    drop_failed_spills(&kv_cache.nvme_take_failed(), kv_cache, prefix_cache);
    let bs = kv_cache.block_size();
    let plan = tier.plan_restore(tokens, bs, adapter_id);
    if plan.disk.is_empty() {
        return None;
    }
    let wanted = want(&plan).min(plan.disk.len());
    if wanted == 0 {
        // Declined: only the pin to release. No block is allocated and
        // nothing is read (on the fast path a read first waits for every
        // queued write).
        let unused = tier.complete_restore(tokens, bs, adapter_id, &plan, &[], false);
        debug_assert!(unused.is_empty(), "a declined restore adopted nothing");
        return Some(RestoreOutcome {
            resident_tokens: plan.resident_tokens,
            on_disk: plan.disk.len(),
            ..RestoreOutcome::default()
        });
    }
    let io0 = kv_cache.nvme_io_stats();
    let t0 = std::time::Instant::now();
    let mut blocks = Vec::with_capacity(wanted);
    // Each block's index in the sequence (the tier is refused beside a latent
    // shard, whose allocator draws by it: `factory::build::glm::shard_plan`).
    let first = plan.resident_tokens / bs;
    while blocks.len() < wanted {
        match alloc_block_evicting(kv_cache, prefix_cache, gpu, first + blocks.len()) {
            Some(b) => blocks.push(b),
            None => break,
        }
    }
    let evict_micros = t0.elapsed().as_micros() as u64;
    // The fast path pairs the run with the blocks in ascending order.
    let (mut ok, mut failed) =
        kv_cache.nvme_read(&plan.disk[..blocks.len()], &mut blocks, gpu, stream);
    let io = kv_cache.nvme_io_stats().since(io0);
    let read_micros =
        (t0.elapsed().as_micros() as u64 - evict_micros).saturating_sub(io.flush_micros);
    // Slotted index tails: a restored block is a shared cached block and must
    // own NO tail (the index kernels then skip it). Its tail was released when
    // it was last freed, but that release reaches the device map only with the
    // next publish — do it now, on the stream the prefill will read on.
    if ok > 0
        && let Err(e) = kv_cache.lend_tail_slots(&[], gpu, stream)
    {
        tracing::warn!("NVMe prefix restore: tail-map publish failed ({e:#}) — recomputing");
        (ok, failed) = (0, false);
    }
    let give_back = tier.complete_restore(tokens, bs, adapter_id, &plan, &blocks[..ok], failed);
    // Allocated but not adopted (read failed / not reached): ref 1 → free.
    for &b in blocks[ok..].iter().chain(&give_back) {
        kv_cache.return_evicted_block(b);
    }
    // Spills queued by this restore's own evictions (the read waited for them).
    drop_failed_spills(&kv_cache.nvme_take_failed(), kv_cache, prefix_cache);
    Some(RestoreOutcome {
        resident_tokens: plan.resident_tokens,
        on_disk: plan.disk.len(),
        wanted,
        allocated: blocks.len(),
        restored: ok
            - give_back
                .iter()
                .filter(|b| blocks[..ok].contains(b))
                .count(),
        failed,
        evict_micros,
        read_micros,
        io,
    })
}

#[cfg(test)]
#[path = "kv_nvme_keep_tests.rs"]
mod keep_tests;
#[cfg(test)]
#[path = "kv_nvme_tests.rs"]
mod tests;
