// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of `glm_sp`: the rows each rank owns, and every SP split's
//! quantized gathers.

use super::*;

#[test]
fn every_sp_split_gathers_quantized() {
    // The chain and the pair both; a lone rank has no peers to gather.
    for ranks in [2, 4] {
        for rank in 0..ranks {
            assert!(SpRows::for_rank(8192, rank, ranks).quant_gathers());
        }
    }
    assert!(!SpRows::for_rank(8192, 0, 1).quant_gathers());
}

#[test]
fn a_pair_block_gather_swaps_the_two_parts_rows() {
    // The staged rows of a quantized gather are `[total, width]` tensors
    // in BF16 widths (2368 B a MoE row): the pair's block exchange sends
    // this rank's rows of the block and lands the peer's at the peer's.
    for rank in 0..2 {
        let sp = SpRows::for_rank(8196, rank, 2);
        let peer = SpRows::for_rank(8196, 1 - rank, 2);
        assert_eq!((sp.peer0, sp.peer_rows), (peer.row0, peer.rows));
        assert_eq!(sp.rows * (sp.ranks - 1), sp.peer_rows);
    }
}

#[test]
fn rank0_owns_the_upper_half() {
    assert_eq!(
        SpRows::for_rank(8196, 0, 2),
        SpRows {
            row0: 4098,
            rows: 4098,
            peer0: 0,
            peer_rows: 4098,
            full_shared: false,
            ranks: 2,
        }
    );
    assert_eq!(
        SpRows::for_rank(8196, 1, 2),
        SpRows {
            row0: 0,
            rows: 4098,
            peer0: 4098,
            peer_rows: 4098,
            full_shared: false,
            ranks: 2,
        }
    );
    let r = SpRows::for_rank(8, 0, 2);
    assert_eq!(
        r.local(DevicePtr(0x1000), 4096),
        DevicePtr(0x1000 + 4 * 4096 * 2)
    );
}

#[test]
fn chain_ranks_own_quarters_top_down() {
    let parts: Vec<(usize, usize)> = (0..4)
        .map(|r| SpRows::for_rank(8192, r, 4))
        .map(|sp| {
            assert_eq!((sp.total(), sp.peer0, sp.ranks), (8192, 0, 4));
            (sp.row0, sp.rows)
        })
        .collect();
    assert_eq!(parts, [(6144, 2048), (4096, 2048), (2048, 2048), (0, 2048)]);
}

#[test]
fn split_at_gives_rank0_the_lower_rows() {
    let lo = SpRows::split_at(16046, 8192, 0);
    let hi = SpRows::split_at(16046, 8192, 1);
    assert_eq!(
        (lo.row0, lo.rows, lo.peer0, lo.peer_rows),
        (0, 8192, 8192, 7854)
    );
    assert_eq!(
        (hi.row0, hi.rows, hi.peer0, hi.peer_rows),
        (8192, 7854, 0, 8192)
    );
    assert_eq!((lo.total(), hi.total()), (16046, 16046));
    assert!(lo.full_shared && hi.full_shared);
}

#[test]
fn scope_clears_on_drop() {
    assert_eq!(current(), None);
    {
        let _s = enter(SpRows::for_rank(16, 1, 2));
        assert_eq!(current().map(|r| r.rows), Some(8));
    }
    assert_eq!(current(), None);
}
