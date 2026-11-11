// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the one-shot layout, settings, stripes and flag words.

use super::*;

fn cfg(vars: &[(&str, &str)]) -> Option<Config> {
    Config::parse(|k| {
        vars.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    })
}

#[test]
fn off_unless_requested_and_defaults_cover_decode_payloads() {
    assert_eq!(cfg(&[]), None);
    assert_eq!(cfg(&[("ATLAS_RDMA_ONESHOT", "0")]), None);
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    // The capture contract promises >= 512 KiB (k5 route: 8 rows x 5 x 8 KiB).
    assert_eq!(c.max, 1 << 20);
    assert!(c.max >= 512 << 10);
    assert_eq!(c.stripe_min, SPLIT_MIN);
    assert_eq!(c.timeout_ns, 30_000_000_000);
    assert!(!c.stage_fence);
    assert_eq!(c.sm_stage, 0);
    assert!(c.pdl);
    let c = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "600000"),
        ("ATLAS_RDMA_ONESHOT_STRIPE_MIN", "16384"),
        ("ATLAS_RDMA_ONESHOT_TIMEOUT_MS", "0"),
        ("ATLAS_RDMA_ONESHOT_STAGE_FENCE", "1"),
    ])
    .unwrap();
    assert_eq!((c.max, c.stripe_min, c.timeout_ns), (600_000, 16384, 0));
    assert!(c.stage_fence);
    let sm = |v: &str| {
        cfg(&[
            ("ATLAS_RDMA_ONESHOT", "1"),
            ("ATLAS_RDMA_ONESHOT_SM_STAGE", v),
            ("ATLAS_RDMA_ONESHOT_PDL", "0"),
        ])
        .unwrap()
    };
    assert_eq!(
        (sm("1").sm_stage, sm("2").sm_stage, sm("7").sm_stage),
        (1, 2, 0)
    );
    assert_eq!(sm("x").sm_stage, 0);
    assert!(!sm("1").pdl);
    let big = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "999999999"),
    ]);
    assert_eq!(big.unwrap().max, MAX_LIMIT);
    // A chain keeps the size field's top bit for its op codes; a pair does not.
    assert_eq!(big.unwrap().for_chain().max, RS_BIT - 64);
    let tp4 = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "8388544"),
    ]);
    assert_eq!(tp4.unwrap().for_chain().max, 8_388_544);
    let small = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "4096"),
    ]);
    assert_eq!(small.unwrap().max, MIN_MAX);
}

#[test]
fn eligibility_depends_only_on_size() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    for rows in 1..=8 {
        assert!(c.eligible(rows * 4096 * 2), "decode rows {rows}");
        assert!(c.eligible(rows * 4096 * 2 * 5), "k5 rows {rows}");
    }
    assert!(c.eligible(16) && c.eligible(2) && c.eligible(c.max));
    assert!(!c.eligible(0) && !c.eligible(3) && !c.eligible(c.max + 2));
}

#[test]
fn wire_distinguishes_configs() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]);
    assert_eq!(Config::wire(None), [0; 3]);
    assert_ne!(Config::wire(c), Config::wire(None));
    let d = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_STRIPE_MIN", "1"),
    ]);
    assert_ne!(Config::wire(c), Config::wire(d));
    let e = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_SM_STAGE", "1"),
    ]);
    assert_ne!(Config::wire(c), Config::wire(e));
    // PDL and the stage fence are local choices.
    let f = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_PDL", "0"),
        ("ATLAS_RDMA_ONESHOT_STAGE_FENCE", "1"),
    ]);
    assert_eq!(Config::wire(c), Config::wire(f));
}

#[test]
fn region_layout_is_disjoint_and_aligned() {
    let max = 1 << 20;
    // A pair: `S`, the peer's two slots, the control page.
    assert_eq!(recv_off(max, 0, 0), max);
    assert_eq!(recv_off(max, 0, 1), 2 * max);
    assert_eq!(ctrl_off(max, 1), 3 * max);
    assert_eq!(region_bytes(max, 1), ctrl_off(max, 1) + 4096);
    assert_eq!(region_bytes(max, 3), ctrl_off(max, 3) + CTRL_PAGE);
    // The pair's words, as before the chain: wire-compatible with any pair.
    assert_eq!((FLAGS, PAIR_FLAG_SRC, PAIR_CTRL_PAGE), (128, 640, 4096));
    assert!(flag_line(0, MAX_RAILS - 1) + 64 <= PAIR_FLAG_SRC);
    const { assert!(PAIR_FLAG_SRC + 64 * MAX_RAILS <= PAIR_CTRL_PAGE) };
    // A chain of four: two slots per other rank, in rank order.
    assert_eq!(recv_off(max, 2, 1), 6 * max);
    assert_eq!(ctrl_off(max, 3), 7 * max);
    // Every control word on its own 64-byte line inside the page.
    let mut words = vec![STAGE, POISON];
    for r in 0..MAX_PIECES {
        words.push(FLAG_SRC + 64 * r);
        for o in 0..MAX_RANKS - 1 {
            words.extend([flag_line(o, r), FWD_SRC + ORIGIN_LINES * o + 64 * r]);
        }
    }
    words.sort_unstable();
    assert!(words.windows(2).all(|w| w[1] >= w[0] + 64));
    words.extend((0..MAX_PIECES).map(|p| READY + 64 * (p / 8)));
    words.dedup();
    assert!(words.iter().all(|w| w % 64 == 0 && w + 8 <= CTRL_PAGE));
    // Receive slots and the staging buffer are 16-byte aligned for the
    // kernel's vector loads whenever `max` is (it is a multiple of 64).
    assert!(recv_off(max, 2, 1).is_multiple_of(16));
}

#[test]
fn a_payload_keeps_its_origin_slot_along_the_chain() {
    // Slots are the other ranks in order.
    assert_eq!([0, 2, 3].map(|o| origin_slot(o, 1)), [0, 1, 2]);
    assert_eq!(origin_slot(1, 0), 0);
    // Rank 0's payload going up and rank 3's going down: the same slot at
    // every rank they pass, so a forward writes the slot it read.
    assert!((1..4).all(|at| origin_slot(0, at) == 0));
    assert!((0..3).all(|at| origin_slot(3, at) == 2));
    assert_eq!((origin_slot(1, 2), origin_slot(1, 3)), (1, 1));
    assert_eq!((origin_slot(2, 1), origin_slot(2, 0)), (1, 1));
}

#[test]
fn shapes_take_two_to_four_ranks_and_rails_where_links_are() {
    let chain = |rank, rails| Shape {
        ranks: 4,
        rank,
        rails,
    };
    assert!(Shape::pair(1, 2).check().is_ok());
    assert!(chain(0, [0, 2]).check().is_ok() && chain(3, [1, 0]).check().is_ok());
    assert!(chain(1, [2, 2]).check().is_ok());
    assert!(chain(1, [0, 2]).check().is_err());
    assert!(chain(2, [2, 9]).check().is_err());
    assert!(chain(4, [2, 2]).check().is_err());
    let five = Shape {
        ranks: 5,
        rank: 0,
        rails: [0, 1],
    };
    assert!(
        five.check()
            .unwrap_err()
            .to_string()
            .contains("2..=4 ranks")
    );
    // A pair waits on its link's stripes, a chain on the pieces, whatever
    // the rails.
    assert_eq!(stripes(1 << 20, 2, 1 << 10).len(), 2);
    assert_eq!(stripes(64 << 10, 2, 1 << 20).len(), 1);
    assert_eq!(pieces(1 << 20, 1 << 10).len(), 8);
    assert_eq!(pieces(32 << 10, 1 << 10).len(), 2);
    assert_eq!(pieces(64, 1 << 10).len(), 1);
}

#[test]
fn chain_pieces_follow_the_size_not_the_rails() {
    let k = 1 << 10;
    assert_eq!(pieces(8 * k, 16 * k), [(0, 8 * k)]);
    assert_eq!(pieces(16 * k, 16 * k), [(0, 8 * k), (8 * k, 8 * k)]);
    assert_eq!(pieces(64 * k, 16 * k).len(), 2);
    assert_eq!(pieces(512 * k, 16 * k).len(), 4);
    assert_eq!(pieces(1 << 20, 16 * k).len(), MAX_RAILS);
    // At most one per flag line, every one 64-byte aligned and the whole
    // payload covered.
    for bytes in [2, 100, 24 * k, 72 * k, 1 << 20, (1 << 20) + 6, 16 << 20] {
        let parts = pieces(bytes, 16 * k);
        assert!(!parts.is_empty() && parts.len() <= MAX_PIECES);
        let mut next = 0;
        for &(off, len) in &parts {
            assert!(off == next && off % 64 == 0 && len > 0);
            next += len;
        }
        assert_eq!(next, bytes);
    }
    assert_eq!(pieces(512 * k, 1 << 20).len(), 1);
    // A prefill-sized op: 256 KiB pieces, one per flag line.
    assert_eq!(pieces(8 << 20, 16 * k).len(), MAX_PIECES);
}

#[test]
fn reduce_scatter_pieces_divide_every_segment_alike() {
    let k = 1 << 10;
    // Small ops keep two pieces a segment; large ones cut finer, down to
    // 128 KiB, in whole 64-byte units of every segment.
    assert_eq!(rs_per_segment(4 * 8 * k), 2);
    assert_eq!(rs_per_segment(4 * 512), 2);
    assert_eq!(rs_per_segment(4 * 512 * k), 4);
    assert_eq!(rs_per_segment(4 * (2 << 20)), 8);
    assert_eq!(rs_per_segment(4 * ((2 << 20) + 128)), 2);
    for bytes in [512, 4 * 8 * k, 4 * 512 * k, 4 * (2 << 20), 8_388_096] {
        let k = rs_per_segment(bytes);
        let parts = rs_pieces(bytes);
        assert_eq!(parts.len(), 4 * k, "{bytes}");
        let seg = bytes / 4;
        for (p, &(off, len)) in parts.iter().enumerate() {
            assert_eq!(len, seg / k);
            assert_eq!(
                off,
                (p / k) * seg + (p % k) * len,
                "piece {p} stays in its segment"
            );
            assert!(len.is_multiple_of(64));
        }
    }
}

#[test]
fn fp8_reduce_scatter_pieces_are_the_bf16_pieces_wire_images() {
    // Up to a full slice of 8 KiB units below RS_BIT.
    for bytes in [4 * 8192, 4 * (512 << 10), 4 * ((2 << 20) - 8192)] {
        let code = bytes | RS_BIT | FP8_BIT;
        assert_eq!(decode_size(code), (bytes, OpKind::ReduceScatter));
        let (bf16, wire) = (rs_pieces(bytes), rs_wire_pieces(code));
        assert_eq!(wire.len(), bf16.len());
        for (&(o, l), &(wo, wl)) in bf16.iter().zip(&wire) {
            // E4M3 values then one f32 scale per 128: 33/64 of the bytes,
            // still 16-byte aligned for whole-KiB pieces.
            assert_eq!((wo, wl), (o * 33 / 64, l * 33 / 64));
            assert!(wo.is_multiple_of(16) && wl.is_multiple_of(16), "{bytes}");
        }
        // The BF16 op's and the all-to-all's pieces are unchanged.
        assert_eq!(rs_wire_pieces(bytes | RS_BIT), bf16);
        assert_eq!(rs_wire_pieces(bytes | RS_BIT | A2A_BIT), bf16);
    }
}

#[test]
fn stripes_cover_the_payload_in_aligned_parts() {
    assert_eq!(stripes(65536, 2, SPLIT_MIN), [(0, 65536)]);
    assert_eq!(stripes(65536, 2, 16384), [(0, 32768), (32768, 32768)]);
    assert_eq!(stripes(100, 2, 16), [(0, 64), (64, 36)]);
    // Too small to use every rail: fewer parts, and the kernel waits on
    // exactly as many flags as the proxy sends.
    assert_eq!(stripes(48, 4, 16), [(0, 48)]);
    for bytes in [2, 16, 8190, 65536, 327_680, 1 << 20] {
        for rails in 1..=MAX_RAILS {
            let parts = stripes(bytes, rails, 0);
            assert!(!parts.is_empty() && parts.len() <= rails);
            let mut next = 0;
            for &(off, len) in &parts {
                assert!(off == next && off % 64 == 0 && len > 0);
                next += len;
            }
            assert_eq!(next, bytes);
        }
    }
}

#[test]
fn flag_words_order_by_sequence_and_carry_the_size() {
    let (a, b) = (flag_word(7, 65536), flag_word(8, 16));
    assert!(a >> 24 == 7 && b >> 24 == 8 && a < b);
    assert_eq!(a & 0xff_ffff, 65536);
    assert_eq!(flag_word(1, MAX_LIMIT) & 0xff_ffff, MAX_LIMIT as u64);
    assert!(describe_poison(POISON_MISMATCH | 9).contains("op 9"));
    assert!(describe_poison(POISON_TIMEOUT | 3).contains("never arrived"));
    assert!(describe_poison(POISON_DESYNC | 5).contains("op 5: the peer's flag is more than one"));
    assert!(describe_poison(POISON_PROXY | 2).contains("op 2: the RDMA proxy failed"));
    assert!(describe_poison(POISON_LAUNCH).contains("launch failed"));
}

/// A zeroed stand-in for a pinned region of `bytes`, and its address.
fn region(bytes: usize) -> (Vec<u64>, usize) {
    let mut mem = vec![0u64; bytes / 8];
    let host = mem.as_mut_ptr() as usize;
    (mem, host)
}

#[test]
fn a_failed_proxy_poisons_the_channel_and_releases_the_stage_wait() {
    use super::super::{Peer, proxy::proxy_loop, region_bytes as legacy_bytes};
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    let ((_os, host), (_legacy, legacy)) =
        (region(region_bytes(c.max, 1)), region(legacy_bytes(64)));
    let ctrl = host + ctrl_off(c.max, 1);
    let stage = unsafe { &*((ctrl + STAGE) as *const AtomicU32) };
    // An ineligible staged size makes the proxy give up before it uses a rail.
    stage.store(3, Ordering::Release);
    let peer = Peer {
        base: 0,
        rkeys: Vec::new(),
    };
    let (jobs, stop) = Default::default();
    let ch = Channel::new(c, host, 0);
    let end = proxy_loop(
        Vec::new(),
        &[],
        &peer,
        legacy,
        64,
        &jobs,
        &stop,
        Some(ch),
        None,
    );
    assert!(end.unwrap_err().to_string().contains("not eligible"));
    // Nothing would clear `stage` again: the proxy leaves it clear, so the
    // stream reaches the kernel, and poisoned, so that kernel traps.
    assert_eq!(stage.load(Ordering::Acquire), 0);
    let why = poison_at(ctrl).expect("poisoned");
    assert!(why.contains("op 1: the RDMA proxy failed"), "{why}");
}

#[test]
fn host_poison_keeps_the_kernels_reason() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    let (_os, host) = region(region_bytes(c.max, 1));
    let ctrl = host + ctrl_off(c.max, 1);
    assert_eq!(poison_at(ctrl), None);
    poison_host(ctrl, POISON_LAUNCH);
    assert!(poison_at(ctrl).unwrap().contains("launch failed"));
    unsafe { ((ctrl + POISON) as *mut u64).write(POISON_TIMEOUT | 4) };
    poison_host(ctrl, POISON_PROXY | 9);
    assert!(
        poison_at(ctrl)
            .unwrap()
            .contains("op 4: the peer never arrived")
    );
}

#[test]
fn chain_completion_words_follow_payload_parity_without_changing_pair_layout() {
    for origin in 0..MAX_RANKS - 1 {
        for piece in 0..MAX_PIECES {
            let line = flag_line(origin, piece);
            assert_eq!(chain_flag_line(origin, piece, 2), line);
            assert_eq!(chain_flag_line(origin, piece, 3), line + 8);
            assert_eq!(chain_flag_line(origin, piece, 4), line);
            assert!(chain_flag_line(origin, piece, 3) + 8 <= line + 64);
            let mut flags = [0u64; 8];
            flags[1] = flag_word(3, 512);
            assert_eq!(flags[0], 0, "seq+1 cannot satisfy seq's wait");
            flags[0] = flag_word(2, 4096);
            assert_eq!(flags[1], flag_word(3, 512), "late seq cannot regress seq+1");
        }
    }
}

#[test]
fn a_fused_op_cuts_whole_producer_rows_or_declines() {
    let row = 4096 * 2;
    // 1356 rows a rank cut eight ways is 169.5 rows a piece: the producer
    // could not write it, so the seam reduces a block at a time instead.
    assert_eq!(fused::split(1356 * row, RS_FUSED_SEG, row), None);
    assert_eq!(
        fused::split(1352 * row, RS_FUSED_SEG, row),
        Some((8, 169 * row))
    );
    // A byte-granular producer (the copy) keeps every split.
    assert_eq!(
        fused::split(1356 * row, RS_FUSED_SEG, 16),
        Some((8, 1356 * row / 8))
    );
    assert_eq!(fused::split(RS_FUSED_SEG + row, RS_FUSED_SEG, row), None);
    assert_eq!(fused::split(0, RS_FUSED_SEG, row), None);
}
