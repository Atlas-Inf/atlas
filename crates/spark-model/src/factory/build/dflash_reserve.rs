// SPDX-License-Identifier: AGPL-3.0-only

//! What a DFlash drafter allocates per sequence after the KV pool is sized.
//!
//! The drafter head is built after the target (it needs the target's embed
//! and LM head), so its paged KV pool, sized `per-seq blocks × max batch`,
//! lands after the KV budget; and every sequence that speculates takes a
//! context accumulator (`acquire_ctx_acc`: `ctx_window` rows of the captured
//! target hiddens) at request time. Neither was in the budget. At eight
//! sequences the OOM guard absorbed them (about 2.3 GB on the 27B); at 64 they
//! are 18 GB for DFlash2 and twice that for two drafters, and the unified
//! memory ran out mid-run (swap, then a stalled server). The budget now
//! reserves them.

use crate::factory::DflashBuildArgs;
use crate::weight_loader::DflashConfig;

const BLOCK: usize = 16;
const BF16: usize = 2;

/// The drafter context window (`from_weights`: `ATLAS_DFLASH_CTX_WINDOW`,
/// 4096), capped at `max_seq_len`.
fn ctx_capacity(max_seq_len: usize) -> usize {
    std::env::var("ATLAS_DFLASH_CTX_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096usize)
        .min(max_seq_len)
}

/// One drafter's paged KV per sequence, as `from_weights` sizes its pool.
fn drafter_kv_per_seq(c: &DflashConfig, ctx: usize, gamma: usize) -> usize {
    let blocks = (ctx + gamma + 1) / BLOCK + 1;
    blocks * BLOCK * c.num_hidden_layers * 2 * c.num_key_value_heads * c.head_dim * BF16
}

/// Bytes the drafter(s) take after the KV budget for `max_batch` sequences.
pub(super) fn bytes(
    args: &DflashBuildArgs<'_>,
    capture_layers: usize,
    target_hidden: usize,
    max_seq_len: usize,
    max_batch: usize,
) -> usize {
    let ctx = ctx_capacity(max_seq_len);
    let gamma = |c: &DflashConfig| args.gamma.map_or(c.block_size(), |g| g.min(c.block_size()));
    let mut per_seq = drafter_kv_per_seq(&args.drafter_config, ctx, gamma(&args.drafter_config));
    if let Some((_, alt)) = args.alt.as_ref() {
        per_seq += drafter_kv_per_seq(alt, ctx, gamma(alt));
    }
    // The accumulator holds the union capture (one per sequence).
    per_seq += (ctx + 1) * capture_layers * target_hidden * BF16;
    per_seq * max_batch.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 27B DFlash2 pool as the log reports it: 257 blocks a sequence of
    /// 5 layers x 8 KV heads x 128 (84 MB), plus the 200 MiB accumulator.
    #[test]
    fn matches_the_27b_dflash2_log() {
        let c: DflashConfig = serde_json::from_str(
            r#"{"hidden_size":5120,"num_hidden_layers":5,"intermediate_size":17408,
                "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,
                "vocab_size":248320,"block_size":8}"#,
        )
        .unwrap();
        assert_eq!(drafter_kv_per_seq(&c, 4096, 8), 257 * 16 * 5 * 2 * 8 * 128 * 2);
        let acc = 4097 * 5 * 5120 * 2;
        assert_eq!(acc / (1 << 20), 200);
    }
}
