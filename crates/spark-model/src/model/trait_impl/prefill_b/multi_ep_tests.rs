// SPDX-License-Identifier: AGPL-3.0-only

use super::{Lookup, multi_segment_start};

fn lookup(skip: bool, skip_to: usize, shares_blocks: bool) -> Lookup {
    Lookup {
        skip,
        skip_to,
        shares_blocks,
        exact_snap: false,
        vision_pad: false,
    }
}

#[test]
fn shipped_behaviour_sends_every_cache_hit_alone() {
    assert_eq!(
        multi_segment_start(false, 70, &lookup(false, 0, false)),
        Some(0)
    );
    assert_eq!(
        multi_segment_start(false, 70, &lookup(false, 0, true)),
        None
    );
    assert_eq!(
        multi_segment_start(false, 700, &lookup(true, 256, true)),
        None
    );
}

#[test]
fn a_match_without_a_restore_recomputes_from_zero_in_the_pass() {
    assert_eq!(
        multi_segment_start(true, 70, &lookup(false, 0, true)),
        Some(0)
    );
}

#[test]
fn a_restore_starts_its_segment_at_the_restored_depth() {
    assert_eq!(
        multi_segment_start(true, 700, &lookup(true, 256, true)),
        Some(256)
    );
    // Nothing left to compute, or the exact-snapshot fixup: alone.
    assert_eq!(
        multi_segment_start(true, 256, &lookup(true, 256, true)),
        None
    );
    let mut exact = lookup(true, 256, true);
    exact.exact_snap = true;
    assert_eq!(multi_segment_start(true, 700, &exact), None);
}

#[test]
fn vision_pads_always_prefill_alone() {
    let mut l = lookup(false, 0, false);
    l.vision_pad = true;
    assert_eq!(multi_segment_start(true, 70, &l), None);
    assert_eq!(multi_segment_start(false, 70, &l), None);
}
