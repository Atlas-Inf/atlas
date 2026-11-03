// SPDX-License-Identifier: AGPL-3.0-only

//! Pure cursor and bounds helpers for the GLM-5 MTP draft cache.
//!
//! Keeping these rules on the host makes the cache lifecycle independently
//! testable. The GPU forward must fail closed on overflow or invalid proposer
//! state; it never silently reuses a rejected row or wraps a sequence.

use anyhow::{Result, bail};

/// Blocks required before appending at `seq_len` (one-based when len is 0).
pub(crate) fn required_blocks(seq_len: usize, block_size: usize) -> Result<usize> {
    ensure_block_size(block_size)?;
    seq_len
        .checked_div(block_size)
        .and_then(|complete| complete.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP: required block count overflows"))
}

/// Validate the block table and return its cursor plus the physical slot for
/// the row about to be written at `seq_len`.
pub(crate) fn append_cursor(
    block_table: &[u32],
    seq_len: usize,
    block_size: usize,
    capacity_blocks: usize,
) -> Result<(usize, i64)> {
    ensure_block_size(block_size)?;
    let cursor = seq_len / block_size;
    let block = block_table.get(cursor).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "GLM-5 MTP: block table has {} blocks but cache cursor is {cursor}",
            block_table.len()
        )
    })?;
    ensure_block_capacity(cursor, capacity_blocks)?;
    let slot = checked_slot(block, seq_len % block_size, block_size)?;
    Ok((cursor, slot))
}

/// Roll back rejected drafts while retaining rows represented by accepted
/// drafts. `accepted` must be known (`last_num_drafted > 0`) and no larger
/// than the number of drafts written.
pub(crate) fn rewind_len(
    seq_len: usize,
    last_num_drafted: usize,
    num_accepted: usize,
) -> Result<usize> {
    if last_num_drafted == 0 {
        bail!("GLM-5 MTP: verification called before a non-empty draft");
    }
    if num_accepted > last_num_drafted {
        bail!(
            "GLM-5 MTP: accepted {num_accepted} drafts but only {last_num_drafted} were proposed"
        );
    }
    if seq_len < last_num_drafted {
        bail!("GLM-5 MTP: cache length {seq_len} cannot contain {last_num_drafted} drafted rows");
    }
    let rejected = last_num_drafted - num_accepted;
    seq_len
        .checked_sub(rejected)
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP: cache rewind underflows"))
}

/// C1 verification keeps the row for the verified input token on rejection.
/// `rewind_len` models ordinary draft rows only; GLM's first row is also the
/// `(last_token, verified-hidden)` pair needed by the next proposal.
pub(crate) fn rewind_len_c1(
    seq_len: usize,
    last_num_drafted: usize,
    num_accepted: usize,
) -> Result<usize> {
    if num_accepted == 0 && last_num_drafted > 0 {
        rewind_len(seq_len, last_num_drafted, 1)
    } else {
        rewind_len(seq_len, last_num_drafted, num_accepted)
    }
}

/// Roll back a native GLM verify while retaining the verified input pair.
/// Every proposed row is predictor-derived and is rebuilt from canonical
/// target hidden rows after verification.
pub(crate) fn rewind_len_native_gamma(
    seq_len: usize,
    last_num_drafted: usize,
    num_accepted: usize,
) -> Result<usize> {
    if last_num_drafted == 0 {
        bail!("GLM-5 MTP: native verification called before a non-empty draft");
    }
    if last_num_drafted > 8 {
        bail!("GLM-5 MTP: native gamma admits at most eight drafts (got {last_num_drafted})");
    }
    if num_accepted > last_num_drafted {
        bail!(
            "GLM-5 MTP: accepted {num_accepted} drafts but only {last_num_drafted} were proposed"
        );
    }
    if seq_len < last_num_drafted {
        bail!("GLM-5 MTP: cache length {seq_len} cannot contain {last_num_drafted} drafted rows");
    }
    if last_num_drafted == 1 {
        return rewind_len_c1(seq_len, last_num_drafted, num_accepted);
    }
    seq_len
        .checked_sub(last_num_drafted)
        .and_then(|base| base.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP: native gamma rewind underflows"))
}

/// Drop an unverified proposal without recording acceptance or retaining its root.
pub(crate) fn abandon_draft_tail(seq_len: usize, pending: usize) -> Result<usize> {
    anyhow::ensure!(pending <= 8, "GLM-5 MTP abandoned tail exceeds draft cap");
    seq_len
        .checked_sub(pending)
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP abandoned tail underflows"))
}

/// A propose span must fit both the u32 RoPE metadata and the private cache.
pub(crate) fn validate_propose_span(
    position: usize,
    num_drafts: usize,
    max_seq_len: usize,
) -> Result<()> {
    let end = position
        .checked_add(num_drafts)
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP: draft position range overflows"))?;
    if end > max_seq_len {
        bail!("GLM-5 MTP: drafts through position {end} exceed cache capacity {max_seq_len}");
    }
    if end > u32::MAX as usize {
        bail!("GLM-5 MTP: draft position {end} exceeds u32 RoPE metadata");
    }
    Ok(())
}

/// Bound proposal width for target verification, which consumes the pending
/// root token as well as every proposed token. The root is not a draft.
pub(crate) fn bounded_draft_width(
    position: usize,
    requested: usize,
    capacity: usize,
) -> Result<usize> {
    validate_propose_span(position, 0, capacity)?;
    Ok(requested.min(capacity.saturating_sub(position).saturating_sub(1)))
}

fn ensure_block_size(block_size: usize) -> Result<()> {
    if block_size == 0 {
        bail!("GLM-5 MTP: cache block size must be nonzero");
    }
    Ok(())
}

fn ensure_block_capacity(cursor: usize, capacity_blocks: usize) -> Result<()> {
    if cursor >= capacity_blocks {
        bail!(
            "GLM-5 MTP: cache cursor {cursor} exceeds allocated block capacity {capacity_blocks}"
        );
    }
    Ok(())
}

fn checked_slot(block: u32, offset: usize, block_size: usize) -> Result<i64> {
    let block = usize::try_from(block)
        .map_err(|_| anyhow::anyhow!("GLM-5 MTP: block id {block} is not a usize"))?;
    let blocks = block
        .checked_mul(block_size)
        .and_then(|start| start.checked_add(offset))
        .ok_or_else(|| anyhow::anyhow!("GLM-5 MTP: cache slot overflows"))?;
    i64::try_from(blocks).map_err(|_| anyhow::anyhow!("GLM-5 MTP: cache slot exceeds i64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_derives_one_block_per_partial_cache_block() {
        assert_eq!(required_blocks(0, 16).unwrap(), 1);
        assert_eq!(required_blocks(15, 16).unwrap(), 1);
        assert_eq!(required_blocks(16, 16).unwrap(), 2);
        assert!(required_blocks(0, 0).is_err());
    }

    #[test]
    fn cursor_validates_table_capacity_and_slot_arithmetic() {
        let (cursor, slot) = append_cursor(&[7, 9], 19, 16, 2).unwrap();
        assert_eq!(cursor, 1);
        assert_eq!(slot, 9 * 16 + 3);
        assert!(append_cursor(&[7], 19, 16, 2).is_err());
        assert!(append_cursor(&[7, 9], 19, 16, 1).is_err());
    }

    #[test]
    fn rewind_preserves_accepted_rows_and_rejects_unknown_state() {
        assert_eq!(rewind_len(4, 3, 1).unwrap(), 2);
        assert_eq!(rewind_len(3, 3, 0).unwrap(), 0);
        assert!(rewind_len(3, 0, 0).is_err());
        assert!(rewind_len(3, 3, 4).is_err());
        assert!(rewind_len(2, 3, 1).is_err());
    }

    #[test]
    fn c1_rejection_retains_verified_input_pair() {
        assert_eq!(rewind_len_c1(1, 1, 0).unwrap(), 1);
        assert_eq!(rewind_len_c1(2, 1, 1).unwrap(), 2);
        assert_eq!(rewind_len_c1(4, 3, 0).unwrap(), 2);
    }

    #[test]
    fn native_gamma_retains_input_pair_for_every_verdict() {
        for gamma in 1..=8 {
            let seq_len = 9 + gamma;
            for accepted in 0..=gamma {
                assert_eq!(
                    rewind_len_native_gamma(seq_len, gamma, accepted).unwrap(),
                    10,
                    "gamma={gamma} accepted={accepted}"
                );
            }
            assert!(rewind_len_native_gamma(seq_len, gamma, gamma + 1).is_err());
        }
    }

    #[test]
    fn native_gamma_rejects_empty_and_wide_drafts() {
        assert!(rewind_len_native_gamma(1, 0, 0).is_err());
        assert!(rewind_len_native_gamma(9, 9, 0).is_err());
        assert!(rewind_len_native_gamma(2, 3, 1).is_err());
    }

    #[test]
    fn abandoned_tail_preserves_committed_base() {
        for pending in 0..=8 {
            assert_eq!(abandon_draft_tail(39 + pending, pending).unwrap(), 39);
        }
        assert!(abandon_draft_tail(48, 9).is_err());
        assert!(abandon_draft_tail(2, 3).is_err());
    }

    #[test]
    fn proposal_width_reserves_the_target_root_at_context_boundary() {
        for capacity in [8, 8192, 1_048_576] {
            for remaining in 0..=8 {
                let position = capacity - remaining;
                for requested in 0..=8 {
                    let drafts = bounded_draft_width(position, requested, capacity).unwrap();
                    assert_eq!(drafts, requested.min(remaining.saturating_sub(1)));
                    if remaining > 0 {
                        assert!(position + 1 + drafts <= capacity);
                    } else {
                        assert_eq!(drafts, 0);
                    }
                }
            }
            assert_eq!(bounded_draft_width(capacity - 8, 8, capacity).unwrap(), 7);
            assert_eq!(bounded_draft_width(capacity - 1, 8, capacity).unwrap(), 0);
            assert!(bounded_draft_width(capacity + 1, 8, capacity).is_err());
        }
        assert_eq!(bounded_draft_width(0, 8, 0).unwrap(), 0);
    }

    #[test]
    fn propose_span_is_bounded() {
        assert!(validate_propose_span(7, 3, 10).is_ok());
        assert!(validate_propose_span(0, 0, 10).is_ok());
        assert!(validate_propose_span(8, 3, 10).is_err());
        assert!(validate_propose_span(usize::MAX, 1, usize::MAX).is_err());
    }
}
