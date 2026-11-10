// SPDX-License-Identifier: AGPL-3.0-only

//! The tree verify's KV layout on a simulated cache: every row's paged
//! attention must read, position for position, what serial decode reads
//! after the path ending at that row, and the commit must leave the
//! sequence's slots as serial decode of the accepted path does.

use std::collections::HashMap;

use super::{commit_entries, tree_layout};
use crate::speculative::draft_tree::DraftTree;

/// What a KV slot holds: a committed prefix position, or a row's own K/V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kv {
    Prefix(usize),
    Row(usize),
}

fn tree() -> DraftTree {
    // root -> 1 -> 2 -> 3 ; 2 -> 4 ; 1 -> 5 ; root -> 6 -> 7
    DraftTree {
        tokens: vec![0, 10, 20, 30, 21, 11, 40, 50],
        parent: vec![-1, 0, 1, 2, 2, 1, 0, 6],
        depth: vec![0, 1, 2, 3, 3, 2, 1, 2],
    }
}

fn run(pre_len: usize, bs: usize) {
    let t = tree();
    t.validate().unwrap();
    let k = t.rows();
    let max_pos = pre_len + t.max_depth();
    // The sequence's blocks (ids 100..), every position reserved.
    let table: Vec<u32> = (0..=max_pos / bs).map(|b| 100 + b as u32).collect();
    // Row r's private blocks for logical blocks b0..=last(r) (ids 1000 + ...).
    let b0 = pre_len / bs;
    let mut next = 1000u32;
    let private: Vec<Vec<u32>> = (0..k)
        .map(|r| {
            let last = (pre_len + t.depth[r] as usize) / bs;
            (b0..=last)
                .map(|_| {
                    next += 1;
                    next
                })
                .collect()
        })
        .collect();
    let lay = tree_layout(&t, pre_len, bs, &table, &private);

    let slot = |table: &[u32], p: usize| table[p / bs] as i64 * bs as i64 + (p % bs) as i64;
    let mut kv: HashMap<i64, Kv> = HashMap::new();
    for p in 0..pre_len {
        kv.insert(slot(&table, p), Kv::Prefix(p));
    }
    // The rows' own writes, then the copies (independent: no dst is a src).
    for r in 0..k {
        kv.insert(lay.slots[r], Kv::Row(r));
    }
    let srcs: Vec<i64> = lay.entries.chunks(2).map(|e| e[0]).collect();
    for e in lay.entries.chunks(2) {
        assert!(
            !srcs.contains(&e[1]),
            "copy dst {} is another copy's src",
            e[1]
        );
    }
    let snapshot = kv.clone();
    for e in lay.entries.chunks(2) {
        kv.insert(e[1], snapshot[&e[0]]);
    }

    for r in 0..k {
        let path = t.path_to(r);
        assert_eq!(lay.positions[r] as usize, pre_len + path.len() - 1);
        assert_eq!(lay.seq_lens[r] as usize, pre_len + path.len());
        for p in 0..lay.seq_lens[r] as usize {
            let want = if p < pre_len {
                Kv::Prefix(p)
            } else {
                Kv::Row(path[p - pre_len])
            };
            assert_eq!(
                kv.get(&slot(&lay.tables[r], p)).copied(),
                Some(want),
                "row {r} position {p} (pre_len {pre_len}, bs {bs})"
            );
        }
    }

    // Commit the path 0 -> 1 -> 2 -> 4.
    let path = [0usize, 1, 2, 4];
    let entries = commit_entries(pre_len, bs, &table, &private[4], b0, path.len() - 1);
    let snapshot = kv.clone();
    for e in entries.chunks(2) {
        kv.insert(e[1], snapshot[&e[0]]);
    }
    for p in 0..pre_len + path.len() {
        let want = if p < pre_len {
            Kv::Prefix(p)
        } else {
            Kv::Row(path[p - pre_len])
        };
        assert_eq!(
            kv.get(&slot(&table, p)).copied(),
            Some(want),
            "committed position {p}"
        );
    }
}

#[test]
fn rows_read_their_paths_and_the_commit_lands_the_accepted_one() {
    // Prefix ends mid-block, at a block boundary, and so the tree spans two
    // blocks; block sizes 4 and 16.
    for (pre_len, bs) in [
        (0, 4),
        (3, 4),
        (4, 4),
        (6, 4),
        (7, 4),
        (17, 16),
        (31, 16),
        (32, 16),
    ] {
        run(pre_len, bs);
    }
}
