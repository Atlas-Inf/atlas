// SPDX-License-Identifier: AGPL-3.0-only
//! [`DecodeM16::pair`]: the slab family by row count, and the `_a16` twins
//! (`ATLAS_GLM_FP4L_DECODE_A16`) only for an A16 batch with them loaded.
use super::*;

fn k(v: u64) -> KernelHandle {
    KernelHandle(v)
}

/// A pair's handle values (`KernelHandle` has no `PartialEq`).
fn ids(p: [KernelHandle; 2]) -> [u64; 2] {
    p.map(|h| h.0)
}

fn twins(a16: Option<[[KernelHandle; 2]; 3]>) -> DecodeM16 {
    DecodeM16 {
        gate_up_silu: k(1),
        down: k(2),
        wide: [k(3), k(4)],
        wide4: [k(5), k(6)],
        a16,
        zskip: true,
        k128w: false,
        nosync: false,
        ctas: 48,
    }
}

const A16: [[KernelHandle; 2]; 3] = [
    [KernelHandle(11), KernelHandle(12)],
    [KernelHandle(13), KernelHandle(14)],
    [KernelHandle(15), KernelHandle(16)],
];

#[test]
fn a4_batches_take_the_slab_family_of_their_rows() {
    for a16 in [None, Some(A16)] {
        let d = twins(a16);
        for (rows, want) in [
            (1, [1, 2]),
            (16, [1, 2]),
            (17, [3, 4]),
            (32, [3, 4]),
            (33, [5, 6]),
            (64, [5, 6]),
        ] {
            assert_eq!(ids(d.pair(rows, false)), want, "{rows}");
        }
    }
}

#[test]
fn a16_batches_take_the_a16_twins_of_their_rows() {
    let d = twins(Some(A16));
    for (rows, want) in [(1, 0), (16, 0), (17, 1), (32, 1), (33, 2), (64, 2)] {
        assert_eq!(ids(d.pair(rows, true)), ids(A16[want]), "{rows}");
    }
    // Without the twins loaded an A16 batch never reaches here; the A4 pair
    // is what `pair` would hand back.
    assert_eq!(ids(twins(None).pair(8, true)), [1, 2]);
}
