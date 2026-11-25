// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn patches_follow_their_tile() {
    // 2 tiles wide, 1 high (grid 2x4): tile 0's four patches, then tile 1's.
    assert_eq!(
        patch_order(2, 4, 2),
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3)
        ]
    );
}

#[test]
fn bilinear_taps_match_half_pixel_interpolation() {
    // An axis as long as the table reads it back exactly.
    for i in 0..32 {
        let [(a, wa), (b, wb)] = axis_taps(i, 32, 32);
        assert_eq!((a, wa), (i, 1.0));
        assert!(wb == 0.0 && b <= 31);
    }
    // 6 patches over 32 entries: src = (i + 0.5) * 32 / 6 - 0.5
    let [(a, wa), (b, wb)] = axis_taps(0, 6, 32);
    assert_eq!((a, b), (2, 3));
    assert!((wa - (1.0 - 1.0 / 6.0)).abs() < 1e-6 && (wb - 1.0 / 6.0).abs() < 1e-6);
    // 64 patches over 32: the first sits before entry 0 and clamps to it.
    let [(a, wa), (b, wb)] = axis_taps(0, 64, 32);
    assert_eq!((a, b), (0, 0));
    assert!((wa + wb - 1.0).abs() < 1e-6);
}

#[test]
fn positions_are_weighted_table_rows() {
    // side 2, hidden 1: table [[0, 1], [2, 3]]; a 2x2 grid reads it back.
    let table = [0.0, 1.0, 2.0, 3.0];
    let order = patch_order(2, 2, 2);
    assert_eq!(position_rows(&table, 2, 1, &order, 2, 2), table);
    // a 4x4 grid's first patch sits a quarter step before entry (0, 0)
    let order = patch_order(4, 4, 2);
    let rows = position_rows(&table, 2, 1, &order, 4, 4);
    assert_eq!(rows[0], 0.0);
    // patch (1, 1): src = 0.25 on both axes -> 0.75*0.75*0 + 0.75*0.25*1 + 0.25*0.75*2 + 0.0625*3
    assert!((rows[3] - 0.75).abs() < 1e-6, "{}", rows[3]);
}

#[test]
fn rope_tables_pair_halves_and_count_from_one() {
    let order = patch_order(2, 2, 2);
    let (cos, sin) = rope_rows(&order, 8);
    // CLS: identity
    assert_eq!(&cos[..8], &[1.0; 8]);
    assert_eq!(&sin[..8], &[0.0; 8]);
    // patch (row 0, col 1) is order[1], row 2 of the tables: col+1 = 2, row+1 = 1.
    // head_dim 8: 4 pairs, 2 per axis, freqs [1, theta^-0.5 = 0.01].
    let r = &cos[2 * 8..3 * 8];
    let want = [2.0f64, 0.02, 1.0, 0.01].map(|a| a.cos() as f32);
    assert_eq!(r, [want, want].concat());
    let s = &sin[2 * 8..3 * 8];
    assert_eq!(s[0], (2.0f64).sin() as f32);
    assert_eq!(s[4], s[0]);
}

#[test]
fn head_rows_move_from_pairs_to_halves() {
    // 2 heads of 4 channels, width 1: rows hold their own index.
    let rows: Vec<u8> = (0u16..8).flat_map(|v| v.to_le_bytes()).collect();
    let got: Vec<u16> = permute_head_rows(&rows, 2, 4, 1)
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    // head 0: [0, 2 | 1, 3]; head 1: [4, 6 | 5, 7]
    assert_eq!(got, [0, 2, 1, 3, 4, 6, 5, 7]);
}

#[test]
fn layer_scale_folds_into_rows() {
    let w = f32_to_bf16(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        bf16_to_f32(&scale_rows(&w, &[0.5, 2.0], 2)),
        [0.5, 1.0, 6.0, 8.0]
    );
    assert_eq!(
        bf16_to_f32(&scale_rows(&w, &[1.0, 0.0, 2.0, 0.25], 1)),
        [1.0, 0.0, 6.0, 1.0]
    );
}
