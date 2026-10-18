// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn spec() -> Nvfp4RowGeom {
    Nvfp4RowGeom {
        dim: 32,
        group: 16,
        global: 0.25,
    }
}

#[test]
fn decodes_rows_with_block_and_global_scales() {
    let s = spec();
    // Row 0: nibble 2 (=1.0) in block 0, nibble 7 (=6.0) in block 1.
    // Row 1: nibble 9 (=-0.5) everywhere.
    let packed = vec![
        [vec![0x22u8; 8], vec![0x77u8; 8]].concat(),
        vec![0x99u8; 16],
    ];
    // Block scales: E4M3 0x38 = 1.0, 0x40 = 2.0.
    let scales = vec![vec![0x38u8, 0x40], vec![0x40u8, 0x40]];
    let out = decode_rows_bf16(&s, &packed, &scales).unwrap();
    let at = |i: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([out[2 * i], out[2 * i + 1]])) << 16)
    };
    assert_eq!(out.len(), 2 * 32 * 2);
    assert!((0..16).all(|i| at(i) == 0.25)); // 1.0 * 1.0 * 0.25
    assert!((16..32).all(|i| at(i) == 3.0)); // 6.0 * 2.0 * 0.25
    assert!((32..64).all(|i| at(i) == -0.25)); // -0.5 * 2.0 * 0.25
}

#[test]
fn rejects_mismatched_rows_and_strides() {
    let s = spec();
    assert!(decode_rows_bf16(&s, &[vec![0; 16]], &[]).is_err());
    assert!(decode_rows_bf16(&s, &[vec![0; 15]], &[vec![0; 2]]).is_err());
    assert!(check_packed_stride(16, &s, false).is_ok());
    assert!(check_packed_stride(32, &s, false).is_err());
    assert!(check_packed_stride(16, &s, true).is_err());
}
