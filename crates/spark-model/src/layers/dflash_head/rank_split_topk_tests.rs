// SPDX-License-Identifier: AGPL-3.0-only

//! The candidate swap (`ATLAS_GLM_DRAFT_TP_TOPK`): a single-sequence propose
//! swaps the worker's top-K lists where it swapped logits, a batched one
//! keeps the logits, and both ranks still walk the same swaps.

use super::super::exec::TopkSelect;
use super::*;
use crate::layers::ops;

fn topk(parts: Parts) -> Geometry {
    Geometry {
        topk: true,
        ..geometry(parts)
    }
}

#[test]
fn the_switch_reads_zero_or_one_and_rides_bit_4_of_the_parity_word() {
    for off in [None, Some(""), Some("0"), Some(" ")] {
        assert!(!parse_topk(off).unwrap());
    }
    assert!(parse_topk(Some("1")).unwrap());
    for bad in ["2", "on", "true"] {
        assert!(parse_topk(Some(bad)).is_err(), "{bad}");
    }
    for (batch, ctx) in [(false, false), (true, true)] {
        let off = parity_word(Some(ALL), batch, ctx, false);
        assert_eq!(parity_word(Some(ALL), batch, ctx, true), off | 16);
    }
}

#[test]
fn a_single_sequence_swaps_candidate_lists_where_it_swapped_logits() {
    use Piece::*;
    for parts in [ALL, HEAD] {
        let g = topk(parts);
        let (head, worker) = (g.steps(0), g.steps(1));
        assert_eq!(swaps_of(&head), swaps_of(&worker), "{parts:?}");
        assert_eq!(swaps_of(&head).last(), Some(&Swap::Candidates));
        assert!(!swaps_of(&head).contains(&Swap::Logits));
        assert_eq!(swaps_of(&head).len(), geometry(parts).swaps().len());
        let tail = |steps: &[Step], n: usize| steps[steps.len() - n..].to_vec();
        assert_eq!(
            tail(&head, 6),
            [
                Step::Run(Norm),
                Step::Swap(Swap::Hidden),
                Step::Run(Vocab),
                Step::Run(TopK),
                Step::Swap(Swap::Candidates),
                Step::Run(Select)
            ]
        );
        assert_eq!(
            tail(&worker, 3),
            [
                Step::Run(Vocab),
                Step::Run(TopK),
                Step::Swap(Swap::Candidates)
            ]
        );
    }
    // One row's list: K f32 values and K u32 ids.
    let g = topk(ALL);
    assert_eq!(
        g.bytes(Swap::Candidates),
        ops::dflash2_selector_lists_bytes(g.gamma)
    );
    // The same plan at a sequence's rows; a batched propose swaps logits.
    assert!(g.with_rows(g.gamma).topk);
    let batched = g.with_rows(2 * g.gamma);
    assert!(!batched.topk);
    assert!(batched.swaps().contains(&Swap::Logits));
    assert!(!batched.swaps().contains(&Swap::Candidates));
}

#[test]
fn the_worker_lists_reach_the_head_and_nothing_else_moves() {
    let g = topk(ALL);
    let (mut r0, mut r1) = (Rank::new(g), Rank::new(g));
    r0.split.topk = Some(TopkSelect::new(g.gamma, false, &r0.gpu).unwrap());
    r1.split.topk = Some(TopkSelect::new(g.gamma, true, &r1.gpu).unwrap());
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    let bytes = g.bytes(Swap::Candidates);
    let lists = |r: &Rank| {
        r.split
            .topk
            .as_ref()
            .map(|t| (t.lists, t.lists_peer))
            .unwrap()
    };
    let ((own0, peer0), (own1, _)) = (lists(&r0), lists(&r1));
    r0.fill(own0, bytes, 0x61);
    r1.fill(own1, bytes, 0x62);
    let (head_own, worker_own) = (r0.read(own0, bytes), r1.read(own1, bytes));

    let frame = |r: &Rank| Frame::serial(&r.scratch);
    let sent0 = r0.read(r0.split.ends(Swap::Candidates, 0, &frame(&r0)).0, bytes);
    let sent1 = r1.read(r1.split.ends(Swap::Candidates, 1, &frame(&r1)).0, bytes);
    assert_eq!(sent1, worker_own, "the worker sends its lists");
    p0.peer.borrow_mut().push_back(sent1);
    p1.peer.borrow_mut().push_back(sent0);
    r0.split
        .swap(Swap::Candidates, 0, &r0.gpu, &p0, &frame(&r0), 0)
        .unwrap();
    r1.split
        .swap(Swap::Candidates, 1, &r1.gpu, &p1, &frame(&r1), 0)
        .unwrap();
    assert_eq!(r0.read(peer0, bytes), worker_own);
    assert_eq!(r0.read(own0, bytes), head_own, "the head keeps its own");
    assert_eq!(
        r1.read(own1, bytes),
        worker_own,
        "the worker's lands in its sink"
    );
    assert_eq!(*p0.sizes.borrow(), [bytes]);
    assert_eq!(*p1.sizes.borrow(), [bytes]);
}

#[test]
fn each_rank_reduces_its_own_share_of_the_logits() {
    let g = topk(ALL);
    let mut r = Rank::new(g);
    r.split.topk = Some(TopkSelect::new(g.gamma, false, &r.gpu).unwrap());
    r.split.begin(g.gamma, CtxRows::default()).unwrap();
    for rank in 0..2 {
        let before = r.gpu.launches_snapshot().len();
        r.split.topk_piece(rank, &r.gpu, 0).unwrap();
        let launches = r.gpu.launches_snapshot();
        assert_eq!(launches.len(), before + 1);
        let l = launches.last().unwrap();
        // Seven arguments; one block a (slice, row).
        assert_eq!(l.args, 7);
        assert_eq!(l.grid[1], g.gamma as u32);
        assert_eq!(
            l.block,
            [ops::DFLASH2_SELECTOR_THREADS, 1, 1],
            "rank {rank}"
        );
    }
}
