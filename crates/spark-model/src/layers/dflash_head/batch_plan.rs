// SPDX-License-Identifier: AGPL-3.0-only

//! Pure planning helpers for the B×gamma propose seam: the scheduler's
//! admission envelope, and which execution path each sequence takes when a
//! batched run fails partway through per-sequence prepare.

/// Admission width: the batched seam serves every admitted sequence when the
/// Lightning product, the generic authoritative mode, parity, or multi-lane
/// overlap is active; the plain generic head keeps the serial 1.
pub(super) fn propose_batch_width(
    native_authoritative: bool,
    batch_parity: bool,
    generic_authoritative: bool,
    multi_lane: bool,
    capacity: usize,
) -> usize {
    if native_authoritative || batch_parity || generic_authoritative || multi_lane {
        capacity
    } else {
        1
    }
}

/// Admission floor: Lightning and parity must see single-sequence batches;
/// generic DFlash (authoritative or serial) stays at 2, so a lone pending
/// sequence keeps the graph-captured serial path.
pub(super) fn propose_batch_floor(native_authoritative: bool, batch_parity: bool) -> usize {
    if native_authoritative || batch_parity {
        1
    } else {
        2
    }
}

/// Per-sequence execution after a batched run failed with `prepared` of `n`
/// sequences' drafter state already advanced (lifecycle stepped, ctx slot
/// appended, precompute committed). Prepared sequences MUST NOT re-prepare —
/// that would double-advance the lifecycle and duplicate a ctx slot — so they
/// run the post-prepare forward only; the rest run the full serial propose.
/// `failed_at` is the sequence whose own `prepare_drafts_state` failed: its
/// state may be partially advanced (advance ran, then a later step such as
/// block allocation failed), so it takes NO drafts this step and re-proposes
/// next step instead of retrying a prepare it may not survive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchSeqPlan {
    /// State already prepared: run `forward_prepared` only.
    ForwardPrepared,
    /// Prepare failed after a partial advance: empty drafts this step.
    Skip,
    /// State untouched: run the full serial `propose_drafts`.
    SerialPropose,
}

pub(super) fn plan_prepared_fallback(
    n: usize,
    prepared: usize,
    failed_at: Option<usize>,
) -> Vec<BatchSeqPlan> {
    (0..n)
        .map(|i| {
            if Some(i) == failed_at {
                BatchSeqPlan::Skip
            } else if i < prepared {
                BatchSeqPlan::ForwardPrepared
            } else {
                BatchSeqPlan::SerialPropose
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_capacity_for_any_batched_mode() {
        assert_eq!(propose_batch_width(true, false, false, false, 16), 16);
        assert_eq!(propose_batch_width(false, true, false, false, 16), 16);
        assert_eq!(propose_batch_width(false, false, true, false, 16), 16);
        assert_eq!(propose_batch_width(false, false, false, true, 16), 16);
        assert_eq!(propose_batch_width(false, false, false, false, 16), 1);
    }

    #[test]
    fn floor_is_one_for_lightning_and_parity_two_for_generic() {
        assert_eq!(propose_batch_floor(true, false), 1);
        assert_eq!(propose_batch_floor(false, true), 1);
        // Generic authoritative and plain generic both floor at 2: the
        // single-sequence case stays on the serial path either way.
        assert_eq!(propose_batch_floor(false, false), 2);
    }

    #[test]
    fn prepared_fallback_splits_at_the_failure_boundary() {
        assert_eq!(
            plan_prepared_fallback(4, 0, None),
            vec![BatchSeqPlan::SerialPropose; 4]
        );
        assert_eq!(
            plan_prepared_fallback(4, 4, None),
            vec![BatchSeqPlan::ForwardPrepared; 4]
        );
        assert_eq!(
            plan_prepared_fallback(4, 2, None),
            vec![
                BatchSeqPlan::ForwardPrepared,
                BatchSeqPlan::ForwardPrepared,
                BatchSeqPlan::SerialPropose,
                BatchSeqPlan::SerialPropose,
            ]
        );
        // The failing index itself may be partially advanced — skip it.
        assert_eq!(
            plan_prepared_fallback(4, 1, Some(1)),
            vec![
                BatchSeqPlan::ForwardPrepared,
                BatchSeqPlan::Skip,
                BatchSeqPlan::SerialPropose,
                BatchSeqPlan::SerialPropose,
            ]
        );
    }
}
