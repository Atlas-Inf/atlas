// SPDX-License-Identifier: AGPL-3.0-only

//! Latch MTP off BEFORE a sequence can cross the QSA inert bound mid-verify.
//!
//! The dispatch gate declines spec for a batch with any sequence already at
//! or past the bound (`verify_ctx_limit`), but admission and verify are
//! separate steps: a sequence admitted BELOW the bound gets a verify step
//! that ingests `num_drafts + 1` rows ACROSS it, so the NEXT verify runs
//! with the QSA selection ACTIVE on the batched multi-seq path — which
//! refuses it ("QSA selection active for row N on the batched ms path",
//! defect D-2a: 10 requests died at 2051 visible tokens with
//! `--speculative --num-drafts 1`). Latching `disable_mtp` one step early —
//! as soon as the NEXT verify could land past the bound — keeps the
//! crossing off the batched path entirely; the sequence simply decodes
//! serially from there (the per-seq lane handles active QSA fine).

/// Whether the next verify step for a sequence at `seq_len` with `num_drafts`
/// drafted tokens could ingest a row at or past the inert bound `lim`: the
/// verify ingests rows `seq_len ..= seq_len + num_drafts`, so the crossing
/// happens iff `seq_len + num_drafts + 1 >= lim`.
pub fn crosses_inert_bound(seq_len: usize, num_drafts: usize, lim: usize) -> bool {
    seq_len + num_drafts + 1 >= lim
}

/// The context limit the MTP lane obeys this step.
///
/// A sequence speculating ALONE may verify past the inert bound when the
/// model serves an active selection per row (`single_seq` is then `None`).
/// A multi-sequence verify batch keeps the conservative limit: past the bound
/// `decode_batch` routes per-sequence and stages only logits, KV exhaustion
/// inside a batched verify finishes the whole batch, and none of that has
/// been exercised at long context. A sequence latched while it shared the
/// batch stays serial afterwards — conservative by design.
pub fn mtp_verify_limit(
    num_active: usize,
    single_seq: Option<usize>,
    multi_seq: Option<usize>,
) -> Option<usize> {
    if num_active == 1 {
        single_seq
    } else {
        multi_seq
    }
}

#[cfg(test)]
mod tests {
    use super::{crosses_inert_bound, mtp_verify_limit};

    #[test]
    fn only_a_lone_sequence_gets_the_lifted_limit() {
        // Model serves active QSA per row: single = None, multi = the bound.
        assert_eq!(mtp_verify_limit(1, None, Some(2051)), None);
        assert_eq!(mtp_verify_limit(2, None, Some(2051)), Some(2051));
        assert_eq!(mtp_verify_limit(8, None, Some(2051)), Some(2051));
        // Switch off / unsupported: both are the bound, nothing changes.
        assert_eq!(mtp_verify_limit(1, Some(2051), Some(2051)), Some(2051));
        assert_eq!(mtp_verify_limit(3, Some(2051), Some(2051)), Some(2051));
        // No indexer at all.
        assert_eq!(mtp_verify_limit(1, None, None), None);
        assert_eq!(mtp_verify_limit(4, None, None), None);
    }

    #[test]
    fn latch_table_around_bound() {
        // index_topk 2048 + ratio 4 - 1 = 2051 on the affected card.
        const LIM: usize = 2051;
        // Expected latch state, written out by hand: [num_drafts 1, 2, 3] per
        // row. The step that FIRST latches is the one whose last verify row
        // (seq_len + num_drafts) lands at or past the bound — e.g. at
        // seq_len=2048 with 2 drafts the rows are 2048..=2050, so the NEXT
        // verify starts at 2051 with selection active.
        let table: [(usize, [bool; 3]); 5] = [
            (2048, [false, true, true]),
            (2049, [true, true, true]),
            (2050, [true, true, true]),
            (2051, [true, true, true]),
            (2052, [true, true, true]),
        ];
        for (seq_len, expected) in table {
            for (j, &num_drafts) in [1usize, 2, 3].iter().enumerate() {
                assert_eq!(
                    crosses_inert_bound(seq_len, num_drafts, LIM),
                    expected[j],
                    "seq_len={seq_len} num_drafts={num_drafts} lim={LIM}",
                );
            }
        }
    }
}
