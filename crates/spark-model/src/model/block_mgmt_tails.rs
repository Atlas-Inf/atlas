// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill window growth and the slotted index-tail guard behind the
//! allocation helpers (split from block_mgmt.rs for the 500-LoC cap).

use super::*;

/// Grow `block_table` through `abs_block_idx` for a prefill chunk, recording
/// every block it pushes in `fresh`.
pub(super) fn grow_prefill_window(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    fresh: &mut Vec<u32>,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            return Ok(());
        }
        // Issue #31: NEVER slide during prefill. block_table grows
        // monotonically until the chunk's full token range is in-window.
        // HBM headroom is preserved by the existing prefix-cache eviction
        // fallback below when `try_alloc_block` returns None. The cap
        // (when HSS is engaged) is enforced lazily on the first decode
        // step, where attention reads through the orchestrator's tiled
        // path and slides are correctness-safe.

        // Try alloc; on failure, evict prefix-cache entries and retry once.
        // Ask the prefix cache to free a block via LRU eviction, as many times
        // as it takes (one eviction can free zero blocks — see
        // `alloc_block_evicting`).
        let blk = alloc_block_evicting(kv_cache, prefix_cache)
            .ok_or_else(|| anyhow::anyhow!("KV cache exhausted: no free blocks"))?;
        fresh.push(blk);
        seq.block_table.push(blk);
        if cap.is_some() {
            let id = spark_storage::with_local(|hss| {
                hss.alloc_disk_block_id().ok_or_else(|| {
                    anyhow::anyhow!(
                        "high-speed-swap: disk-block-id pool exhausted; \
                         increase --high-speed-swap-bytes or shorten --max-seq-len"
                    )
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
                )
            })??;
            seq.disk_block_ids.push(id);
        }
    }
}

/// Host guard for slotted index tails, run before any step writes the window:
/// the tail kernels silently skip a block without a slot (`NO_TAIL`), leaving
/// its pooled keys from the block's previous owner. Every block this step can
/// write for new tokens — committed length through `abs_block_idx`, past the
/// matched prefix, whose pools are already finalized — must hold a tail. A
/// prefix match covers whole blocks and a partly filled block is never
/// published (`prefix_share`), so no block in that window is shared with
/// another sequence or with the prefix cache. (A snapshot replay below the
/// match is outside the window: it needs no tail there.) The verdict reads no
/// reference counts: they follow each rank's own radix cache, and a check that
/// failed one rank of a pair would hang the other instead of failing both.
pub(super) fn check_write_window_tails(
    seq: &SequenceState,
    abs_block_idx: usize,
    kv_cache: &PagedKvCache,
) -> Result<()> {
    let ws = seq.hss_window_start();
    let first = (seq.seq_len / kv_cache.block_size())
        .max(seq.cached_prefix_blocks)
        .max(ws);
    for abs in first..=abs_block_idx {
        let block = seq.block_table[abs - ws];
        if kv_cache.tail_slot_missing(block) {
            bail!(
                "block {block} (logical {abs}, seq_len {}) would be written with no index \
                 tail: its pooled keys would never be finalized",
                seq.seq_len
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "block_mgmt_tail_tests.rs"]
mod tests;
