// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split over the pair: what a propose's swaps leave.

use super::*;

/// `[gamma, n]` rows from rank 0's and rank 1's shares, each `[gamma, its
/// rows]` at the start of the payload it sent.
pub(super) fn joined(gamma: usize, n: usize, lo: &[u8], hi: &[u8]) -> Vec<u8> {
    let (row0, row1) = (share(n, 0, 2).1 * 2, share(n, 1, 2).1 * 2);
    (0..gamma)
        .flat_map(|r| {
            lo[r * row0..(r + 1) * row0]
                .iter()
                .chain(&hi[r * row1..(r + 1) * row1])
                .copied()
        })
        .collect()
}

#[test]
fn swapped_halves_land_where_the_unsplit_launch_writes_whole_rows() {
    let g = geometry(ALL);
    let (r0, r1) = (Rank::new(g), Rank::new(g));
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    // Run `swap` on both ranks, each landing what the other sent.
    let run = |swap: Swap| {
        let bytes = g.bytes(swap);
        let sent0 = r0.read(r0.split.ends(swap, 0, &Frame::serial(&r0.scratch)).0, bytes);
        let sent1 = r1.read(r1.split.ends(swap, 1, &Frame::serial(&r1.scratch)).0, bytes);
        p0.peer.borrow_mut().push_back(sent1.clone());
        p1.peer.borrow_mut().push_back(sent0.clone());
        r0.split
            .swap(swap, 0, &r0.gpu, &p0, &Frame::serial(&r0.scratch), 0)
            .unwrap();
        r1.split
            .swap(swap, 1, &r1.gpu, &p1, &Frame::serial(&r1.scratch), 0)
            .unwrap();
        (sent0, sent1)
    };

    // The head's MLP input reaches the worker's `norm_buf`.
    let hidden = g.gamma * g.hidden * 2;
    r0.fill(r0.scratch.norm_buf, hidden, 0x11);
    let (x, _) = run(Swap::Input(0));
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), x);
    assert_eq!(
        r0.read(r0.scratch.norm_buf, hidden),
        x,
        "the head keeps its own"
    );

    // Both ranks hold the whole gated activation, rank 0's half first.
    let act = g.bytes(Swap::Activation(0));
    r0.fill(r0.split.own(Swap::Activation(0), 0), act, 0x21);
    r1.fill(r1.split.own(Swap::Activation(0), 1), act, 0x22);
    let (a0, a1) = run(Swap::Activation(0));
    let whole = joined(g.gamma, g.inter, &a0, &a1);
    assert_eq!(r0.read(r0.scratch.mlp_intermediate, whole.len()), whole);
    assert_eq!(r1.read(r1.scratch.mlp_intermediate, whole.len()), whole);

    // The head holds the whole down projection; the worker's residual stream
    // is not its to write.
    let out = g.bytes(Swap::Output(0));
    r0.fill(r0.split.own(Swap::Output(0), 0), out, 0x31);
    r1.fill(r1.split.own(Swap::Output(0), 1), out, 0x32);
    r1.fill(r1.scratch.stream_acc, hidden, 0x3f);
    let (y0, y1) = run(Swap::Output(0));
    assert_eq!(
        r0.read(r0.scratch.stream_acc, hidden),
        joined(g.gamma, g.hidden, &y0, &y1)
    );
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden).len(), hidden);
    let untouched: Vec<u8> = (0..hidden)
        .map(|i| 0x3f ^ (i as u8).wrapping_mul(31))
        .collect();
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden), untouched);

    // The final norm rows reach the worker; the head gets the whole logits.
    r0.fill(r0.scratch.norm_buf, hidden, 0x41);
    let (h, _) = run(Swap::Hidden);
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), h);
    let half_logits = g.bytes(Swap::Logits);
    r0.fill(r0.split.own(Swap::Logits, 0), half_logits, 0x51);
    r1.fill(r1.split.own(Swap::Logits, 1), half_logits, 0x52);
    let (l0, l1) = run(Swap::Logits);
    assert_eq!(
        r0.read(r0.scratch.logits, g.gamma * g.vocab * 2),
        joined(g.gamma, g.vocab, &l0, &l1)
    );

    // Same sizes, same order, on both ranks.
    assert_eq!(*p0.sizes.borrow(), *p1.sizes.borrow());
}

#[test]
fn a_propose_that_stops_early_still_issues_every_swap() {
    let g = geometry(ALL);
    let rank = Rank::new(g);
    let pair = Pair::new(&rank.gpu, 0);
    let sizes: Vec<usize> = g.swaps().iter().map(|&s| g.bytes(s)).collect();

    // Stopped after two swaps: the rest are drained, sized as planned.
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    for &swap in &g.swaps()[..2] {
        rank.split
            .swap(swap, 0, &rank.gpu, &pair, &Frame::serial(&rank.scratch), 0)
            .unwrap();
    }
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(*pair.sizes.borrow(), sizes);

    // A propose that reached every swap drains nothing, and the next one
    // starts from zero.
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(pair.sizes.borrow().len(), sizes.len());
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(pair.sizes.borrow().len(), 2 * sizes.len());
    assert!(rank.split.max_bytes() >= *sizes.iter().max().unwrap());
}
