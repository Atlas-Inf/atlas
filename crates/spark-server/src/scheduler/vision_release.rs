// SPDX-License-Identifier: AGPL-3.0-only

//! Release a sequence's pixels once its prompt is in.
//!
//! A request's preprocessed pixels (`SeqVision::items`, which carry its
//! in-flight vision permits, `spark_model::vision_budget`) are needed only
//! until its prefill ends, but a sequence keeps decoding for minutes. Every
//! prefill path (chunked, non-chunked, two-phase, fused with decode) ends with
//! the sequence in `active`, so the scheduler drops the pixels there, once a
//! tick, instead of each model entry point remembering to.

use super::types::ActiveSeq;

/// Drop the pixels of every active sequence; returns how many held some.
pub(super) fn release_prefilled_pixels(active: &mut [ActiveSeq]) -> usize {
    let mut released = 0;
    for a in active.iter_mut() {
        if !a.seq.vision.items.is_empty() {
            a.seq.vision.items = Vec::new();
            released += 1;
        }
    }
    released
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_model::VisionItem;
    use spark_model::vision_budget::VisionRowBudget;

    #[test]
    fn an_active_sequence_returns_its_vision_rows() {
        // Whichever prefill path admitted it, a sequence in `active` holds no
        // pixels after the next tick's sweep, so its permit is back in the
        // budget while it decodes.
        let budget = VisionRowBudget::new(10_000);
        let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1], 7, None, 8);
        let item = VisionItem::image(vec![0.0; 4], 2, 2).holding(budget.try_acquire(4096).unwrap());
        a.seq.vision.items = vec![item];
        let (b, _rx2) = crate::scheduler::test_support::test_seq(vec![1], 7, None, 8);
        let mut active = vec![a, b];
        assert_eq!(budget.in_use(), 4096);
        assert_eq!(release_prefilled_pixels(&mut active), 1);
        assert_eq!(budget.in_use(), 0);
        assert!(active[0].seq.vision.items.is_empty());
        assert_eq!(release_prefilled_pixels(&mut active), 0, "idempotent");
    }
}
