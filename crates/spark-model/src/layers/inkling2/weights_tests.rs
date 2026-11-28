// SPDX-License-Identifier: AGPL-3.0-only

use super::bias_lanes;

/// `proj[k][d]` as a distinct BF16 bit pattern.
fn tag(k: usize, d: usize) -> u16 {
    (k * 1000 + d) as u16
}

fn lane(out: &[u8], d: usize, q: usize, i: usize) -> u16 {
    let o = (d * 16 + q * 4 + i) * 2;
    u16::from_le_bytes([out[o], out[o + 1]])
}

#[test]
fn bias_lanes_hold_each_quads_fragment_then_a_zero_row() {
    let cols = 5;
    let raw: Vec<u8> = (0..16)
        .flat_map(|k| (0..cols).flat_map(move |d| tag(k, d).to_le_bytes()))
        .collect();
    let out = bias_lanes(&raw, 16, cols).unwrap();
    assert_eq!(out.len(), (cols + 1) * 16 * 2);
    for d in 0..cols {
        for q in 0..4 {
            let want = [
                tag(2 * q, d),
                tag(2 * q + 1, d),
                tag(2 * q + 8, d),
                tag(2 * q + 9, d),
            ];
            for (i, w) in want.into_iter().enumerate() {
                assert_eq!(lane(&out, d, q, i), w, "d {d} q {q} i {i}");
            }
        }
    }
    assert!(out[cols * 32..].iter().all(|&b| b == 0));
}

#[test]
fn bias_lanes_take_sixteen_features_only() {
    assert!(bias_lanes(&[0u8; 8 * 4 * 2], 8, 4).is_err());
}
