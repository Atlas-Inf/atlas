// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The merge regions of a chain owner, as `(offset, bytes)`.
fn regions(m: &ChainLayout, rows: usize, world: usize) -> Vec<(usize, usize)> {
    let local = local_heads(world) as usize;
    let (lat, splits) = (LATENT as usize, shard::MERGE_SPLITS as usize);
    let (go, gl) = (rows * HEADS as usize * lat * 4, rows * HEADS as usize * 4);
    vec![
        (m.q_all, world * m.q_part),
        (m.q_groups, 2 * rows * HEADS as usize * lat * 2),
        (m.ids, rows * WIDTH as usize * 4),
        (m.counts, rows * 4),
        (m.part_o, 2 * splits * go),
        (m.part_lse, 2 * splits * gl),
        (m.gathered, world * m.part),
        (m.mine_o, world * rows * local * lat * 4),
        (m.mine_lse, world * rows * local * 4),
        (m.out_lse, rows * local * 4),
    ]
}

#[test]
fn chain_layout_regions_are_disjoint_and_hold_their_payloads() {
    for world in [4usize] {
        for rows in [1u32, 3, 8, 9, 64] {
            let m = ChainLayout::new(rows, world);
            let r = rows as usize;
            let mut spans = regions(&m, r, world);
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].0 + w[0].1 <= w[1].0, "rows {rows}: {w:?} overlap");
            }
            let last = spans.last().unwrap();
            assert!(last.0 + last.1 <= m.total);
            // A rank's queries and merged partials fit their gathered parts.
            assert!(r * local_heads(world) as usize * LATENT as usize * 2 <= m.q_part);
            let lse = r * HEADS as usize * 4;
            assert_eq!(ChainLayout::group_lse(rows, 1) + lse, m.part_bytes);
            assert!(m.part_bytes <= m.part && m.part_bytes.is_multiple_of(2));
            assert!(ChainLayout::group_o(rows, 1) >= ChainLayout::group_lse(rows, 0) + lse);
        }
    }
}

#[test]
fn every_rank_owns_a_disjoint_slice_of_the_two_head_groups() {
    let world = 4;
    let slots: Vec<(usize, usize)> = (0..world).map(|r| head_slot(r, world)).collect();
    assert_eq!(slots, [(0, 0), (0, 16), (1, 0), (1, 16)]);
    // The pair's ranks are its two groups.
    assert_eq!([head_slot(0, 2), head_slot(1, 2)], [(0, 0), (1, 0)]);
}

#[test]
fn the_scratch_holds_the_chain_merge_and_the_view_pieces() {
    let bytes = 16 * 528;
    let l = ScratchLayout::for_world(4, 32772, 8256, bytes);
    let merge = ChainLayout::new(shard::MERGE_MAX_ROWS as u32, 4).total;
    assert!(l.total - l.work >= merge);
    assert_eq!(l.piece_bytes, shard::piece_blocks(4) * bytes);
    assert!(l.total - l.work >= 4 * l.piece_bytes.next_multiple_of(256));
    // The pair's layout is unchanged.
    assert_eq!(
        ScratchLayout::new(32772, 8256, bytes),
        ScratchLayout::for_world(2, 32772, 8256, bytes)
    );
    assert_eq!(shard::piece_blocks(2), shard::PIECE_BLOCKS);
}

#[test]
fn view_counts_cover_the_history_once() {
    for blocks in [1usize, 2, 3, 4, 5, 4099] {
        let parts = ViewParts {
            pool: DevicePtr::NULL,
            view: DevicePtr::NULL,
            mine_slot: DevicePtr::NULL,
            blocks,
            block_bytes: 16,
        };
        let counts: Vec<usize> = (0..4).map(|o| parts.count(o, 4)).collect();
        assert_eq!(counts.iter().sum::<usize>(), blocks);
        for (o, &n) in counts.iter().enumerate() {
            assert_eq!(n, (0..blocks).filter(|l| l % 4 == o).count());
        }
    }
}
