// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DRAFT_TP_CTX`: the switch, the announce word, a context
//! append's swaps ahead of the layers on both ranks, where its halves land,
//! and an append that does not match its announce.

use std::sync::atomic::Ordering;

use super::super::ctx::{CTX_MAX_ROWS, ROWS_MAX, parse_ctx, read_announce};
use super::*;

/// The test drafter with a context append: 96 input columns, 80 K/V rows.
fn ctx_geometry() -> Geometry {
    Geometry {
        ctx_in: 96,
        ctx_kv: 80,
        ..geometry(ALL)
    }
}

const ROWS: CtxRows = CtxRows([3, 0, 2, 0]);

#[test]
fn the_ctx_switch_is_0_or_1_and_sets_its_own_parity_bit() {
    for off in [None, Some(""), Some("0"), Some(" ")] {
        assert!(!parse_ctx(off).unwrap());
    }
    assert!(parse_ctx(Some("1")).unwrap());
    for bad in ["2", "true", "on", "fc"] {
        assert!(parse_ctx(Some(bad)).is_err(), "{bad}");
    }
    for batch in [false, true] {
        let off = parity_word(Some(ALL), batch, false);
        assert_eq!(off, Parts::word(Some(ALL)) | (batch as u64) << 2);
        assert_eq!(parity_word(Some(ALL), batch, true), off | 8);
    }
}

#[test]
fn the_announce_carries_the_rows_and_the_context_rows() {
    // Without context rows the word is the rows, as before the switch, and
    // a pair without the switch reads every word that way.
    assert_eq!(announce_word(0, 0).unwrap(), 0);
    assert_eq!(announce_word(96, 0).unwrap(), 96);
    assert_eq!(announce_word(1 << 20, 0).unwrap(), 1 << 20);
    assert_eq!(read_announce(1 << 20, false), (1 << 20, CtxRows::default()));
    // With them, both fields come back on a pair with the switch.
    for (rows, ctx) in [
        (0, CtxRows::single(7)),
        (32, ROWS),
        (ROWS_MAX, CtxRows([31; 4])),
    ] {
        let word = announce_word(rows, ctx.pack()).unwrap();
        assert_eq!(read_announce(word, true), (rows, ctx));
    }
    assert!(announce_word(ROWS_MAX + 1, ROWS.pack()).is_err());
    assert_eq!(CtxRows::unpack(ROWS.pack()), ROWS);
    assert_eq!(ROWS.appends().collect::<Vec<_>>(), [0, 2]);
    assert_eq!(CtxRows::single(40).get(0), CTX_MAX_ROWS);
}

#[test]
fn context_appends_lead_both_walks_with_the_same_swaps() {
    use Piece::*;
    let g = Geometry {
        ctx: ROWS,
        ..ctx_geometry()
    };
    let (head, worker) = (g.steps(0), g.steps(1));
    assert_eq!(swaps_of(&head), swaps_of(&worker));
    let ctx = [
        Swap::CtxInput(0),
        Swap::CtxHidden(0),
        Swap::CtxKv(0),
        Swap::CtxInput(2),
        Swap::CtxHidden(2),
        Swap::CtxKv(2),
    ];
    assert_eq!(g.swaps()[..6], ctx);
    assert_eq!(
        g.swaps()[6..],
        g.layer_steps(1).iter().filter_map(swap).collect::<Vec<_>>()
    );
    // Both ranks run every context piece; the head's layers follow as before.
    assert_eq!(head[..12], worker[..12]);
    assert_eq!(head[12..], g.layer_steps(0));
    assert_eq!(head[1], Step::Run(CtxFc(0)));
    assert_eq!(head[3..5], [Step::Run(CtxNorm(0)), Step::Run(CtxKv(0))]);
    // An append's swaps carry its own rows: input rows, then the halves.
    assert_eq!(g.bytes(Swap::CtxInput(0)), 3 * 96 * 2);
    assert_eq!(g.bytes(Swap::CtxHidden(2)), 2 * 32 * 2);
    assert_eq!(g.bytes(Swap::CtxKv(0)), 3 * half(80, 1).1 * 2);
    assert_eq!(g.bytes(Swap::Input(0)), g.gamma * 64 * 2);
    // No context rows: the walk is the layers alone.
    assert_eq!(ctx_geometry().steps(1), ctx_geometry().layer_steps(1));
    assert!(ctx_geometry().validate().is_ok());
    for bad in [(96, 0), (0, 80), (80, 80), (96, 16)] {
        let (ctx_in, ctx_kv) = bad;
        let g = Geometry {
            ctx_in,
            ctx_kv,
            ..geometry(ALL)
        };
        assert!(g.validate().is_err(), "{bad:?}");
    }
}

fn swap(step: &Step) -> Option<Swap> {
    match step {
        Step::Swap(s) => Some(*s),
        Step::Run(_) => None,
    }
}

/// A rank of the pair with the context buffers of [`ctx_geometry`].
fn ctx_rank() -> Rank {
    let mut rank = Rank::new(ctx_geometry());
    rank.scratch.fc_proj = rank.gpu.alloc(CTX_MAX_ROWS * 64 * 2).unwrap();
    rank.scratch.fused_kv_out = rank.gpu.alloc(CTX_MAX_ROWS * 80 * 2).unwrap();
    rank
}

#[test]
fn context_halves_land_where_the_unsplit_projections_write_whole_rows() {
    let g = ctx_geometry();
    let (r0, r1) = (ctx_rank(), ctx_rank());
    for r in [&r0, &r1] {
        r.split.begin(g.gamma, ROWS).unwrap();
    }
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    let input = r0.gpu.alloc(3 * 96 * 2).unwrap();
    let f0 = Frame {
        ctx_in: input,
        ..Frame::serial(&r0.scratch)
    };
    let f1 = Frame {
        ctx_in: r1.split.ctx_input,
        ..Frame::serial(&r1.scratch)
    };
    let plan = r0.split.plan();
    let run = |swap: Swap| {
        let bytes = plan.bytes(swap);
        let sent0 = r0.read(r0.split.ends(swap, 0, &f0).0, bytes);
        let sent1 = r1.read(r1.split.ends(swap, 1, &f1).0, bytes);
        p0.peer.borrow_mut().push_back(sent1.clone());
        p1.peer.borrow_mut().push_back(sent0.clone());
        r0.split.swap(swap, 0, &r0.gpu, &p0, &f0, 0).unwrap();
        r1.split.swap(swap, 1, &r1.gpu, &p1, &f1, 0).unwrap();
        (sent0, sent1)
    };

    // The head's accumulator rows reach the worker's landing rows.
    r0.fill(input, 3 * 96 * 2, 0x61);
    let (x, _) = run(Swap::CtxInput(0));
    assert_eq!(r1.read(r1.split.ctx_input, x.len()), x);

    // Both ranks hold the whole `fc` rows, rank 0's half first.
    let fc = plan.bytes(Swap::CtxHidden(0));
    r0.fill(r0.split.ctx_own, fc, 0x71);
    r1.fill(r1.split.ctx_own, fc, 0x72);
    let (a0, a1) = run(Swap::CtxHidden(0));
    let whole = joined(3, 64, &a0, &a1);
    assert_eq!(r0.read(r0.scratch.fc_proj, whole.len()), whole);
    assert_eq!(r1.read(r1.scratch.fc_proj, whole.len()), whole);

    // The head holds the whole fused K/V rows; the worker keeps its own.
    let kv = plan.bytes(Swap::CtxKv(0));
    r0.fill(r0.split.ctx_own, kv, 0x81);
    r1.fill(r1.split.ctx_own, kv, 0x82);
    r1.fill(r1.scratch.fused_kv_out, 3 * 80 * 2, 0x8f);
    let (k0, k1) = run(Swap::CtxKv(0));
    assert_eq!(
        r0.read(r0.scratch.fused_kv_out, 3 * 80 * 2),
        joined(3, 80, &k0, &k1)
    );
    let untouched: Vec<u8> = (0..3 * 80 * 2)
        .map(|i| 0x8f ^ (i as u8).wrapping_mul(31))
        .collect();
    assert_eq!(r1.read(r1.scratch.fused_kv_out, 3 * 80 * 2), untouched);
    assert_eq!(*p0.sizes.borrow(), *p1.sizes.borrow());
}

#[test]
fn the_head_walks_what_it_announced_and_drains_an_append_that_differs() {
    let g = ctx_geometry();
    let rank = ctx_rank();
    let pair = Pair::new(&rank.gpu, 0);

    // The announce is taken once, by the propose that follows it.
    rank.split.announced.store(ROWS.pack(), Ordering::Relaxed);
    rank.split.begin_announced(g.gamma).unwrap();
    assert_eq!(rank.split.plan().ctx, ROWS);
    let plan = rank.split.plan();
    let planned: Vec<usize> = plan.swaps().iter().map(|&s| plan.bytes(s)).collect();

    // Append 0 matches its announce: the head walks it, nothing is issued.
    assert_eq!(rank.split.ctx_turn(3, &pair, 0).unwrap(), Some(0));
    assert!(pair.sizes.borrow().is_empty());
    for &swap in &plan.swaps()[..3] {
        let frame = Frame::serial(&rank.scratch);
        rank.split
            .swap(swap, 0, &rank.gpu, &pair, &frame, 0)
            .unwrap();
    }
    // Append 1 was announced unsplit; append 2 appends 5 rows, not 2: its
    // swaps are drained in place and the append runs unsplit.
    assert_eq!(rank.split.ctx_turn(4, &pair, 0).unwrap(), None);
    assert_eq!(pair.sizes.borrow().len(), 3);
    assert_eq!(rank.split.ctx_turn(5, &pair, 0).unwrap(), None);
    assert_eq!(*pair.sizes.borrow(), planned[..6]);
    // Past the slots nothing splits.
    for _ in 0..3 {
        assert_eq!(rank.split.ctx_turn(1, &pair, 0).unwrap(), None);
    }
    // The propose stops: the layers' swaps drain after, the worker's plan.
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(*pair.sizes.borrow(), planned);

    // The next propose without an announce plans no context append.
    rank.split.begin_announced(g.gamma).unwrap();
    assert_eq!(rank.split.plan().ctx, CtxRows::default());
    assert_eq!(rank.split.ctx_turn(3, &pair, 0).unwrap(), None);
    // A split without the switch refuses announced context rows.
    let plain = Rank::new(geometry(ALL));
    assert!(plain.split.begin(g.gamma, ROWS).is_err());
}
