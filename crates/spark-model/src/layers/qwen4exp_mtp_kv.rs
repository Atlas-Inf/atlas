// SPDX-License-Identifier: AGPL-3.0-only

//! What the qwen4_exp drafter keeps of its own KV after a verdict
//! (`ATLAS_QWEN4EXP_MTP_KV_KEEP=1`, default off: an acceptance A/B switch).
//!
//! A propose at position `Q` writes one drafter row per draft it keeps, row
//! `i` at position `Q + i` from token `t_{Q+i}` (the last verified token,
//! then the drafts) and the stream highway before it (the target's for row
//! 0, the drafter's own after that). After a verdict accepting `na` drafts
//! the committed tokens are `t_Q, d_0 .. d_{na-1}` at `Q .. Q + na`, and the
//! next propose writes position `Q + na + 1`.
//!
//! The default keeps `na` rows (positions `Q .. Q + na - 1`): the row for
//! the last accepted token is dropped though its token was committed, and
//! on a full reject row 0 — the target's own stream and the committed
//! token, the one exact pair the propose wrote — goes too. So every step
//! leaves a hole in the drafter's attention context at `Q + na`, the
//! position nearest the next draft. With the switch the drafter keeps
//! `min(na + 1, drafted)` rows: every committed position it wrote (only a
//! full accept leaves `Q + na`, which no propose computed, unwritten).
//!
//! The switch also drops the rows of drafts that were never verified (the
//! scheduler cleared them: MTP gated off, a prefill joined the batch) before
//! the next propose, keeping row 0: without it those rows stay in context at
//! positions the sequence then fills with other tokens.

use super::Qwen4ExpMtpProposerState;
// Ported from Enntity/atlas 166e2b1 (AGPL-3.0-only).
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

/// `ATLAS_QWEN4EXP_MTP_KV_KEEP=1`, read once. The absorb
/// (`ATLAS_QWEN4EXP_MTP_ABSORB=1`) implies it: it appends after row 0.
pub(crate) fn keep_committed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_MTP_KV_KEEP").as_deref() == Ok("1")
            || super::qwen4exp_mtp_absorb::enabled()
    })
}

/// Rows of the last propose (`drafted`, at least 1) kept after a verdict
/// accepting `accepted` drafts.
pub(crate) fn kept_rows(drafted: usize, accepted: usize, keep_committed: bool) -> usize {
    if keep_committed {
        (accepted + 1).min(drafted)
    } else {
        accepted.min(drafted)
    }
}

/// Apply a verdict of `accepted` drafts to the drafter's row count.
pub(crate) fn after_verdict(st: &mut Qwen4ExpMtpProposerState, accepted: usize) {
    let drafted = st.last_num_drafted.max(1);
    let keep = kept_rows(drafted, accepted, keep_committed());
    st.seq_len = st.seq_len.saturating_sub(drafted - keep);
    st.awaiting_verdict = false;
    st.absorb.verdict = Some(accepted);
}

/// Before a propose: drafts the scheduler dropped unverified count as
/// rejected (switch on only; off, the rows stay as before).
pub(crate) fn settle_unverified(st: &mut Qwen4ExpMtpProposerState) {
    if st.awaiting_verdict && keep_committed() {
        after_verdict(st, 0);
    }
    st.awaiting_verdict = false;
}

/// The drafter's KV geometry: `slot` placeholder layer slots, then the one
/// layer it writes. The placeholders exist only so the body's
/// `attn_idx = slot` lands on a real pool; they are never read or written,
/// so they get the smallest geometry the allocator accepts (one 1-wide head,
/// 32 B per block) instead of a full layer each. Full layers there cost
/// 12/13 of the pool: 6.5 GB at --max-seq-len 262144 where 0.5 GB is used.
pub(super) fn drafter_kv_config(
    slot: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> KvCacheConfig {
    let mut layer_dims = vec![(1, 1); slot];
    layer_dims.push((num_kv_heads, head_dim));
    KvCacheConfig {
        block_size: 16,
        num_kv_heads,
        head_dim,
        num_layers: slot + 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims,
        cache_blocks_per_seq: None,
    }
}

#[cfg(test)]
mod tests {
    use super::kept_rows;

    #[test]
    fn the_default_drops_the_last_committed_row() {
        assert_eq!(kept_rows(3, 0, false), 0);
        assert_eq!(kept_rows(3, 2, false), 2);
        assert_eq!(kept_rows(3, 3, false), 3);
    }

    #[test]
    fn keep_committed_keeps_every_committed_row_it_wrote() {
        // Full reject: row 0 (target stream + committed token) stays.
        assert_eq!(kept_rows(3, 0, true), 1);
        assert_eq!(kept_rows(3, 1, true), 2);
        assert_eq!(kept_rows(7, 6, true), 7);
        // Full accept: the last accepted draft was never fed.
        assert_eq!(kept_rows(3, 3, true), 3);
        // Confidence stop left one draft.
        assert_eq!(kept_rows(1, 0, true), 1);
        assert_eq!(kept_rows(1, 1, true), 1);
    }
}
