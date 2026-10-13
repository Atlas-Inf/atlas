// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DRAFT_TP_BATCH`: the switch, and a batched propose's swaps at
//! B×gamma rows through the batch frame.

use super::*;

#[test]
fn the_batch_switch_is_0_or_1_and_sets_its_own_parity_bit() {
    for off in [None, Some(""), Some("0"), Some(" ")] {
        assert!(!parse_batch(off).unwrap());
    }
    assert!(parse_batch(Some("1")).unwrap());
    for bad in ["2", "true", "on", "mlp"] {
        assert!(parse_batch(Some(bad)).is_err(), "{bad}");
    }
    // Off, both ranks compare the split's own word; on, a word of its own.
    for parts in [None, Some(MLP), Some(HEAD), Some(ALL)] {
        assert_eq!(parity_word(parts, false, false), Parts::word(parts));
        assert_ne!(parity_word(parts, true, false), Parts::word(parts));
    }
}

/// A batch frame of `rows` rows on `rank`'s device.
fn frame(rank: &Rank, g: Geometry, rows: usize) -> Frame {
    let alloc = |width: usize| rank.gpu.alloc(rows * width * 2).unwrap();
    Frame {
        norm: alloc(g.hidden),
        inter: alloc(g.inter),
        acc: alloc(g.hidden),
        logits: alloc(g.vocab),
        ..Frame::serial(&rank.scratch)
    }
}

#[test]
fn a_batched_propose_swaps_its_rows_through_the_batch_frame() {
    let g = geometry(ALL);
    let capacity = 4 * g.gamma;
    let (r0, r1) = (
        Rank::with_capacity(g, capacity),
        Rank::with_capacity(g, capacity),
    );
    let (f0, f1) = (frame(&r0, g, capacity), frame(&r1, g, capacity));
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    // Two sequences of the four the buffers hold.
    let rows = 2 * g.gamma;
    let b = g.with_rows(rows);
    for rank in [&r0, &r1] {
        rank.split.begin(rows, CtxRows::default()).unwrap();
        assert_eq!(rank.split.plan(), b);
        assert!(
            rank.split.begin(capacity + 1, CtxRows::default()).is_err(),
            "over capacity"
        );
        rank.split.begin(rows, CtxRows::default()).unwrap();
    }
    let hidden = rows * g.hidden * 2;
    for (i, &swap) in b.swaps().iter().enumerate() {
        let bytes = b.bytes(swap);
        let (send0, _) = r0.split.ends(swap, 0, &f0);
        let (send1, _) = r1.split.ends(swap, 1, &f1);
        r0.fill(send0, bytes, 0x10 + i as u8);
        r1.fill(send1, bytes, 0x80 + i as u8);
        let (sent0, sent1) = (r0.read(send0, bytes), r1.read(send1, bytes));
        p0.peer.borrow_mut().push_back(sent1.clone());
        p1.peer.borrow_mut().push_back(sent0.clone());
        r0.split.swap(swap, 0, &r0.gpu, &p0, &f0, 0).unwrap();
        r1.split.swap(swap, 1, &r1.gpu, &p1, &f1, 0).unwrap();
        match swap {
            // The head's rows reach the worker's batch norm rows.
            Swap::Input(_) | Swap::Hidden => assert_eq!(r1.read(f1.norm, hidden), sent0),
            // Both hold the whole activation at B×gamma rows.
            Swap::Activation(_) => {
                let whole = joined(rows, g.inter, &sent0, &sent1);
                assert_eq!(r0.read(f0.inter, whole.len()), whole);
                assert_eq!(r1.read(f1.inter, whole.len()), whole);
            }
            Swap::Output(_) => {
                assert_eq!(
                    r0.read(f0.acc, hidden),
                    joined(rows, g.hidden, &sent0, &sent1)
                )
            }
            Swap::Logits => {
                let whole = joined(rows, g.vocab, &sent0, &sent1);
                assert_eq!(r0.read(f0.logits, whole.len()), whole);
            }
            Swap::CtxInput(_) | Swap::CtxHidden(_) | Swap::CtxKv(_) => {
                unreachable!("no context append was announced")
            }
        }
    }
    // Same sizes, same order, both ranks: the plan at B×gamma rows.
    let sizes: Vec<usize> = b.swaps().iter().map(|&s| b.bytes(s)).collect();
    assert_eq!(*p0.sizes.borrow(), sizes);
    assert_eq!(*p1.sizes.borrow(), sizes);
    assert_eq!(r0.split.max_bytes_at(rows), *sizes.iter().max().unwrap());

    // A batched propose that stops early drains the rest at its own rows,
    // and the next single-sequence propose plans gamma rows again.
    r0.split.begin(rows, CtxRows::default()).unwrap();
    r0.split.finish(&p0, 0).unwrap();
    assert_eq!(p0.sizes.borrow()[sizes.len()..], sizes[..]);
    r0.split.begin(g.gamma, CtxRows::default()).unwrap();
    assert_eq!(r0.split.plan(), g);
}
