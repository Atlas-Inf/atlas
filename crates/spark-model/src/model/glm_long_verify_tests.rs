// SPDX-License-Identifier: AGPL-3.0-only
//! Metadata layout and E9/EA width words (host-side only).
use super::*;

const WIDTHS: [usize; 3] = [2, K3_ROWS, owner::MAX_OWNER_ROWS];
/// 262K context at 16-token blocks, and a tiny table.
const MAX_BLOCKS: [usize; 2] = [16_384, 3];

#[test]
fn k3_metadata_layout_matches_the_fixed_width_layout() {
    for mb in MAX_BLOCKS {
        for owners in 1..=MAX_OWNERS {
            let l = MetaLayout::new(owners, K3_ROWS, mb);
            let joint = align(META_BLOCK_TABLE + owners * 3 * mb * 4);
            let own = align(META_BLOCK_TABLE + 3 * mb * 4);
            for o in 0..owners {
                assert_eq!(l.owner_offset(o), META_BASE + joint + o * own);
            }
            assert_eq!(l.end(), META_BASE + joint + owners * own);
        }
    }
}

#[test]
fn metadata_blocks_are_aligned_disjoint_and_hold_their_rows() {
    for mb in MAX_BLOCKS {
        for rows in WIDTHS {
            for owners in 1..=MAX_OWNERS {
                let l = MetaLayout::new(owners, rows, mb);
                let block = |r: usize| META_BLOCK_TABLE + r * mb * 4;
                assert!(l.joint_bytes >= block(owners * rows));
                assert!(l.owner_bytes >= block(rows));
                assert_eq!(META_BASE % 256, 0);
                let mut end = META_BASE + l.joint_bytes;
                for o in 0..owners {
                    let at = l.owner_offset(o);
                    assert_eq!(at % 256, 0, "{owners}x{rows} owner {o}");
                    assert!(at >= end, "{owners}x{rows} owner {o} overlaps");
                    end = at + block(rows);
                }
                assert!(end <= l.end());
            }
        }
    }
    // Wider blocks never shrink the layout.
    let widest = MetaLayout::new(MAX_OWNERS, owner::MAX_OWNER_ROWS, MAX_BLOCKS[0]);
    assert!(widest.end() > MetaLayout::new(MAX_OWNERS, K3_ROWS, MAX_BLOCKS[0]).end());
}

#[test]
fn width_word_keeps_the_k3_wire_and_round_trips_other_widths() {
    for word in 0..MAX_OWNERS + 1 {
        assert_eq!(encode_width(word, K3_ROWS), word as u32);
        assert_eq!(decode_width(word as u32), (word, K3_ROWS));
        for rows in WIDTHS {
            assert_eq!(decode_width(encode_width(word, rows)), (word, rows));
        }
    }
    assert_eq!(encode_width(4, 8), 0x0008_0004);
    assert_eq!(encode_width(1, 2), 0x0002_0001);
}

#[test]
fn owner_index_is_bounded_by_owners_and_width() {
    for rows in WIDTHS {
        for o in 0..MAX_OWNERS {
            assert_eq!(
                owner_supported(o, rows),
                (o + 1) * rows <= MAX_ROWS,
                "{o}x{rows}"
            );
        }
        assert!(!owner_supported(MAX_OWNERS, rows));
    }
    for rows in [0, 1, owner::MAX_OWNER_ROWS + 1] {
        assert!(!owner_supported(0, rows));
    }
}

#[test]
fn alias_check_ignores_the_shared_read_only_prefix() {
    // Two owners matched the same two cached prompt blocks (7, 8) and decode
    // at position 37 into their own third blocks.
    let (a, b) = ([7, 8, 20], [7, 8, 30]);
    assert!(write_window_aliases(&[(37, &a), (37, &b)], 3, 16).is_empty());
    // A verify that crosses into a fresh block writes two blocks per owner.
    let (a, b) = ([7, 8, 20, 21], [7, 8, 30, 31]);
    assert!(write_window_aliases(&[(46, &a), (46, &b)], 3, 16).is_empty());
    // An owner on a block boundary writes only the block it is about to open.
    assert!(write_window_aliases(&[(32, &a[..3]), (32, &b[..3])], 3, 16).is_empty());
}

#[test]
fn alias_check_reports_a_block_two_owners_can_write() {
    // Both owners hold block 20 as the block their next rows land in.
    let (a, b) = ([7, 8, 20], [7, 8, 20]);
    assert_eq!(
        write_window_aliases(&[(37, &a), (38, &b)], 3, 16),
        vec![(20, (0, 2), (1, 2))]
    );
}

#[test]
fn alias_check_reports_a_written_block_inside_another_owners_prefix() {
    // Owner 0 retries a short prompt and writes block 9, which owner 1 holds
    // as a read-only prefix block of its longer prompt.
    let (retry, long) = ([7, 8, 9], [7, 8, 9, 40, 41]);
    assert_eq!(
        write_window_aliases(&[(37, &retry), (70, &long)], 3, 16),
        vec![(9, (0, 2), (1, 2))]
    );
}

#[test]
fn alias_check_reports_a_block_listed_twice_in_one_table() {
    // A prefix adopted twice: block 8 sits at index 1 and again at index 2,
    // where the owner's next rows land.
    let twice = [7, 8, 8];
    assert_eq!(
        write_window_aliases(&[(37, &twice)], 3, 16),
        vec![(8, (0, 2), (0, 1))]
    );
    // Both entries inside the write window are reported once.
    let twice = [7, 8, 20, 20];
    assert_eq!(
        write_window_aliases(&[(46, &twice)], 3, 16),
        vec![(20, (0, 2), (0, 3))]
    );
}
