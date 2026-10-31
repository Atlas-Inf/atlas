// SPDX-License-Identifier: AGPL-3.0-only

use super::{cached_items, first_pad_positions, items_in_range, last_pad_positions};

const PAD: u32 = 9;

#[test]
fn only_items_wholly_before_the_computed_position_are_cached() {
    // text, item 0 (2 rows) at 1..3, text, item 1 (3 rows) at 4..7, text.
    let t = [1, PAD, PAD, 2, PAD, PAD, PAD, 3];
    let pad = |x| x == PAD;
    assert_eq!(cached_items(&t, pad, &[2, 3], 0), (0, 0));
    assert_eq!(
        cached_items(&t, pad, &[2, 3], 2),
        (0, 0),
        "item 0 is half computed"
    );
    assert_eq!(cached_items(&t, pad, &[2, 3], 3), (1, 2));
    assert_eq!(
        cached_items(&t, pad, &[2, 3], 6),
        (1, 2),
        "item 1 is partly computed"
    );
    assert_eq!(cached_items(&t, pad, &[2, 3], 7), (2, 5));
    assert_eq!(cached_items(&t, pad, &[2, 3], 99), (2, 5));
}

#[test]
fn split_video_runs_count_by_item() {
    // One item of 3 placeholders split by timestamp text.
    let t = [PAD, 5, PAD, PAD, 6];
    assert_eq!(cached_items(&t, |x| x == PAD, &[3], 3), (0, 0));
    assert_eq!(cached_items(&t, |x| x == PAD, &[3], 4), (1, 3));
}

fn walks(t: &[u32], counts: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let pad = |x: u32| x == PAD;
    (
        first_pad_positions(t, pad, counts),
        last_pad_positions(t, pad, counts),
    )
}

#[test]
fn a_straddling_item_is_re_encoded_at_every_chunk_it_spans() {
    // item 0 (4 rows) pads at 2..6, item 1 (2 rows) pads at 7..9. Chunks
    // [0,5) and [5,10): item 0 straddles (pads 2,3,4 in chunk 0; pad 5 in
    // chunk 1), so chunk 1 re-encodes it together with item 1.
    let t = [1, 2, PAD, PAD, PAD, PAD, 3, PAD, PAD, 4];
    let (first, last) = walks(&t, &[4, 2]);
    assert_eq!(first, vec![2, 7]);
    assert_eq!(last, vec![5, 8]);
    assert_eq!(items_in_range(&first, &last, 0, 0, 5), vec![0]);
    assert_eq!(
        items_in_range(&first, &last, 0, 5, 10),
        vec![0, 1],
        "item 0's tail pad sits in chunk 1 — re-encode it there"
    );
    // A chunk past every pad encodes nothing.
    assert!(items_in_range(&first, &last, 0, 10, 99).is_empty());
}

#[test]
fn cached_leading_items_are_never_selected() {
    // item 0 (2 rows) wholly cached (computed_from 3), item 1 (2 rows) at
    // 4..6, item 2 (2 rows) at 7..9. Chunk [3,6) selects item 1 only.
    let t = [1, PAD, PAD, 2, PAD, PAD, 3, PAD, PAD, 4];
    let (first, last) = walks(&t, &[2, 2, 2]);
    assert_eq!(items_in_range(&first, &last, 3, 3, 6), vec![1]);
    assert_eq!(
        items_in_range(&first, &last, 3, 6, 9),
        vec![2],
        "item 1's pads all sit before 6 — chunk [6,9) encodes only item 2"
    );
    assert!(items_in_range(&first, &last, 3, 9, 99).is_empty());
}

#[test]
fn a_video_run_split_by_timestamp_text_is_one_item() {
    // One item of 4 placeholders split by timestamp text: [PAD, 5, PAD,
    // 6, PAD, 7, PAD]. Its pads span chunks [0,3) and [3,7) — the item is
    // re-encoded at both.
    let t = [PAD, 5, PAD, 6, PAD, 7, PAD];
    let (first, last) = walks(&t, &[4]);
    assert_eq!(first, vec![0]);
    assert_eq!(last, vec![6]);
    assert_eq!(items_in_range(&first, &last, 0, 0, 3), vec![0]);
    assert_eq!(items_in_range(&first, &last, 0, 3, 7), vec![0]);
    assert!(items_in_range(&first, &last, 0, 7, 99).is_empty());
}

#[test]
fn an_item_without_pads_is_never_selected() {
    let t = [1, 2, 3];
    let (first, last) = walks(&t, &[2]);
    assert_eq!(first, vec![usize::MAX]);
    assert_eq!(last, vec![0]);
    assert!(items_in_range(&first, &last, 0, 0, 3).is_empty());
}
