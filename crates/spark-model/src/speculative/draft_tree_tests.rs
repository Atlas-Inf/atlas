// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A lattice of `d` positions whose candidates are `base + 10 * d + j`, with
/// scores that depend on the predecessor token (so children differ by
/// parent).
fn lattice(d: usize, k: usize) -> Lattice {
    Lattice {
        anchor: 7,
        ban_depth: 0,
        end_ids: [u32::MAX; 4],
        ids: (0..d)
            .map(|p| (0..k).map(|j| 100 + 10 * p as u32 + j as u32).collect())
            .collect(),
        unary: (0..d)
            .map(|_| (0..k).map(|j| -(j as f32)).collect())
            .collect(),
    }
}

/// Scores: unary, plus a bump on candidate `pred % k`.
fn score_of(lat: &Lattice) -> impl FnMut(usize, u32) -> Vec<f32> + '_ {
    move |d, pred| {
        let k = lat.ids[d].len();
        let mut s = lat.unary[d].clone();
        s[pred as usize % k] += 0.5;
        s
    }
}

#[test]
fn chain_layout_and_walk() {
    let t = DraftTree::chain(5, &[6, 7, 8]);
    assert!(t.is_chain());
    t.validate().unwrap();
    assert_eq!(t.parent, vec![-1, 0, 1, 2]);
    // Picks: row 0 picks 6 (accept), row 1 picks 7 (accept), row 2 picks 9.
    assert_eq!(t.walk(&[6, 7, 9, 1]), vec![0, 1, 2]);
    assert_eq!(t.walk(&[6, 7, 8, 1]), vec![0, 1, 2, 3]);
    assert_eq!(t.walk(&[1, 7, 8, 1]), vec![0]);
}

#[test]
fn best_first_is_preorder_with_the_spine_first() {
    let lat = lattice(7, 16);
    let mut score = score_of(&lat);
    for rows in [1, 2, 8, 12, 16] {
        let t = best_first(&lat, rows, 1_000, &mut score);
        t.validate().unwrap();
        assert_eq!(t.rows(), rows);
        assert_eq!(t.tokens[0], 7);
        // Row r + 1 is row r's most probable child down the first branch.
        let mut r = 0;
        while let Some(c) = (r + 1..t.rows()).find(|&c| t.parent[c] == r as i32) {
            assert_eq!(c, r + 1, "first child must follow its parent");
            r = c;
        }
    }
}

#[test]
fn best_first_admits_by_path_probability() {
    // Two positions, three candidates; position 0 nearly certain, so the
    // second row is its top child and depth 2 comes before depth-1 hedges.
    let lat = Lattice {
        anchor: 1,
        ban_depth: 0,
        end_ids: [u32::MAX; 4],
        ids: vec![vec![10, 11, 12], vec![20, 21, 22]],
        unary: vec![vec![10.0, 0.0, 0.0], vec![0.0, -0.1, -5.0]],
    };
    let mut score = |d: usize, _pred: u32| lat.unary[d].clone();
    let t = best_first(&lat, 4, 100, &mut score);
    t.validate().unwrap();
    assert_eq!(t.tokens, vec![1, 10, 20, 21]);
    assert_eq!(t.parent, vec![-1, 0, 1, 1]);
}

#[test]
fn banned_and_invalid_candidates_never_enter() {
    let lat = Lattice {
        anchor: 1,
        ban_depth: 1,
        end_ids: [10, u32::MAX, u32::MAX, u32::MAX],
        ids: vec![vec![10, 11, 5000], vec![20, 21, 22]],
        unary: vec![vec![5.0, 0.0, 9.0], vec![0.0, 0.0, f32::NEG_INFINITY]],
    };
    let mut score = |d: usize, _pred: u32| lat.unary[d].clone();
    let t = best_first(&lat, 16, 1000, &mut score);
    t.validate().unwrap();
    assert!(!t.tokens[1..].contains(&10), "end token below the floor");
    assert!(!t.tokens.contains(&5000), "id past the vocabulary");
    assert!(!t.tokens.contains(&22), "non-finite unary");
    // 11 at depth 1, then 20 and 21 under it: four rows in all.
    assert_eq!(t.rows(), 4);
}

#[test]
fn duplicate_candidates_merge_into_one_child() {
    // The selector may list a token twice for one position: the tree must
    // carry it once (siblings never share a token), with the duplicates'
    // summed mass. 10 holds 2/(2 + e^0.5) = 0.55 against 11's 0.45, so it
    // is the first child although each of its entries scores below 11.
    let lat = Lattice {
        anchor: 1,
        ban_depth: 0,
        end_ids: [u32::MAX; 4],
        ids: vec![vec![10, 11, 10], vec![20, 20, 21]],
        unary: vec![vec![0.0, 0.5, 0.0], vec![0.0, 0.0, 0.0]],
    };
    let mut score = |d: usize, _pred: u32| lat.unary[d].clone();
    for rows in 2..=8 {
        let t = best_first(&lat, rows, 100, &mut score);
        t.validate().unwrap();
        assert_eq!(t.tokens[1], 10, "rows {rows}");
    }
    let t = best_first(&lat, 16, 100, &mut score);
    t.validate().unwrap();
    // Root, two depth-1 children, two depth-2 children under each.
    assert_eq!(t.rows(), 7);
}

#[test]
fn walk_follows_branches() {
    // root -> a(1) -> c(2); root -> b(3)
    let t = DraftTree {
        tokens: vec![0, 10, 20, 11],
        parent: vec![-1, 0, 1, 0],
        depth: vec![0, 1, 2, 1],
    };
    t.validate().unwrap();
    assert_eq!(t.walk(&[11, 0, 0, 30]), vec![0, 3]);
    assert_eq!(t.walk(&[10, 20, 5, 0]), vec![0, 1, 2]);
    assert_eq!(t.path_to(2), vec![0, 1, 2]);
}

#[test]
fn validate_rejects_broken_trees() {
    let bad_order = DraftTree {
        // row 3's parent is row 1, but row 2 (a child of the root) sits
        // between: row 1's subtree is not contiguous.
        tokens: vec![0, 10, 11, 20],
        parent: vec![-1, 0, 0, 1],
        depth: vec![0, 1, 1, 2],
    };
    assert!(bad_order.validate().is_err());
    let twins = DraftTree {
        tokens: vec![0, 10, 10],
        parent: vec![-1, 0, 0],
        depth: vec![0, 1, 1],
    };
    assert!(twins.validate().is_err());
}

#[test]
fn scope_is_visible_until_dropped() {
    assert!(current().is_none());
    {
        let _s = enter(TreeDev {
            parents: DevicePtr(16),
            kv_entries: DevicePtr(32),
            kv_count: DevicePtr(48),
            kv_max: 4,
        });
        assert_eq!(current().unwrap().kv_max, 4);
    }
    assert!(current().is_none());
}

#[test]
fn rows_lever_is_off_unless_a_valid_width() {
    assert_eq!(parse_rows(None), 0);
    assert_eq!(parse_rows(Some("0")), 0);
    assert_eq!(parse_rows(Some("12")), 12);
    assert_eq!(parse_rows(Some(" 16 ")), 16);
    for bad in ["1", "17", "x", "-3"] {
        assert_eq!(parse_rows(Some(bad)), 0, "{bad:?}");
    }
}
