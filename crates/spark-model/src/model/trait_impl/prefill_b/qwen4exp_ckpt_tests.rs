// SPDX-License-Identifier: AGPL-3.0-only

//! Where the qwen4_exp in-pass checkpoint lands and the QSA blob it keeps.

use super::{ckpt_row, qsa_blob_at, tail_ckpt_row};

#[test]
fn the_row_is_the_last_chunk_boundary_at_or_below_the_cut() {
    // Cold 16046-token prompt: cut 16016 (bs 16) -> 16000.
    assert_eq!(ckpt_row(0, 16046, 16016, 16), Some(16000));
    // Last chunk of a 28K prompt starting at 16384.
    assert_eq!(ckpt_row(16384, 11616, 27984, 16), Some(27968));
    // A warm pass from a block boundary that is not 64-aligned.
    assert_eq!(ckpt_row(1008, 900, 1888, 16), Some(1840));
    // No full chunk below the cut, or the cut outside the pass.
    assert_eq!(ckpt_row(1000, 100, 1050, 16), None);
    assert_eq!(ckpt_row(0, 500, 600, 16), None);
    assert_eq!(ckpt_row(0, 500, 0, 16), None);
    for start in (0..512).step_by(16) {
        for cut in start + 1..start + 700 {
            if let Some(cp) = ckpt_row(start, 1000, cut, 16) {
                assert!(cp > start && cp <= cut && cut - cp < 64 && (cp - start) % 64 == 0);
                assert_eq!(cp % 16, 0);
            }
        }
    }
}

/// The needle prompt of 2026-10-07: 77,463 tokens in 16,388-token idle
/// chunks. The last pass starts at 65,540, four tokens past a block
/// boundary, so no 64-row boundary of it is one: on the block grid there
/// is no tail checkpoint. Off it, `cp` is the last 64-row boundary under
/// the cut, a QSA pool-block boundary.
#[test]
fn a_pass_off_the_block_grid_gets_a_checkpoint_off_it() {
    let (start, total, bs, ratio) = (65_540usize, 77_463usize, 16usize, 4usize);
    let cut = 77_440; // tail_cut(77_463, 16)
    let count = total - start;
    assert_eq!(tail_ckpt_row(start, count, cut, bs, ratio, false), None);
    let cp = tail_ckpt_row(start, count, cut, bs, ratio, true);
    assert_eq!(cp, Some(77_380));
    // The follow-up turn (77,501 tokens, whole-block match 77,456)
    // replays 121 tokens instead of 11,961.
    assert_eq!(77_501 - cp.unwrap(), 121);
    // A pass on the grid lands where it did.
    for (start, count, cut) in [(0, 16_046, 16_016), (16_384, 11_616, 27_984)] {
        assert_eq!(
            tail_ckpt_row(start, count, cut, bs, ratio, true),
            tail_ckpt_row(start, count, cut, bs, ratio, false),
        );
    }
    // Every off-grid row: a 64-row boundary of the pass, a ratio
    // multiple, at most 63 under the cut.
    for start in (4..2_000).step_by(4) {
        for cut in start + 1..start + 300 {
            if let Some(cp) = tail_ckpt_row(start, 1_000, cut, bs, ratio, true) {
                assert!(cp > start && cp <= cut && cut - cp < 64);
                assert_eq!((cp - start) % 64, 0);
                assert_eq!(cp % ratio, 0);
            }
        }
    }
    // A pass that starts off the QSA grid gets none either way.
    assert_eq!(tail_ckpt_row(65_541, 11_922, cut, bs, ratio, true), None);
}

#[test]
fn the_qsa_blob_keeps_the_pooled_prefix() {
    let (ratio, row) = (4usize, 6usize);
    let mut blob = Vec::new();
    blob.extend_from_slice(&(1003u64).to_le_bytes());
    blob.extend_from_slice(&(250u64).to_le_bytes());
    let keys: Vec<u8> = (0..250 * row).map(|i| (i % 251) as u8).collect();
    blob.extend_from_slice(&keys);
    blob.extend_from_slice(&[9u8; 3 * 6]); // raw tail of 3 rows
    let got = qsa_blob_at(&blob, 960, ratio, row).unwrap();
    assert_eq!(u64::from_le_bytes(got[..8].try_into().unwrap()), 960);
    assert_eq!(u64::from_le_bytes(got[8..16].try_into().unwrap()), 240);
    assert_eq!(&got[16..], &keys[..240 * row]);
    assert!(
        qsa_blob_at(&blob, 962, ratio, row).is_none(),
        "not a block boundary"
    );
    assert!(
        qsa_blob_at(&blob, 1004, ratio, row).is_none(),
        "past ingested"
    );
}
