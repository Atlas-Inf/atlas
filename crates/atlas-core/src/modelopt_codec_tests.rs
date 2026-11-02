// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::numeric::{f32_to_bf16, nvfp4_dequant};

/// Deterministic xorshift so the tests need no RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte_in(&mut self, lo: u8, hi: u8) -> u8 {
        lo + (self.next() % u64::from(hi - lo)) as u8
    }
}

/// Reference CSF encoder: row base = row minimum capped at 240, any byte the
/// 4-bit window cannot reach becomes an exception. Mirrors the encoder in
/// vLLM's `test_csf_scales.py`, which is how the published layout is pinned.
fn encode(raw: &[u8], rows: usize, cols: usize) -> (Vec<u8>, usize, usize, Vec<u8>) {
    assert!(rows.is_multiple_of(CSF_SLAB_ROWS) && cols.is_multiple_of(2));
    let fixed_cols = CSF_SLAB_ROWS * (1 + cols / 2);
    let mut fixed = vec![0u8; rows / CSF_SLAB_ROWS * fixed_cols];
    let mut exceptions = Vec::new();
    for r in 0..rows {
        let row = &raw[r * cols..(r + 1) * cols];
        let base = (*row.iter().min().unwrap()).min(CSF_MAX_BASE);
        let (slab, i) = (r / CSF_SLAB_ROWS, r % CSF_SLAB_ROWS);
        let slab_bytes = &mut fixed[slab * fixed_cols..(slab + 1) * fixed_cols];
        slab_bytes[i] = base;
        for (c, &byte) in row.iter().enumerate() {
            let code = i32::from(byte) - i32::from(base);
            let code = if (0..=15).contains(&code) {
                code as u8
            } else {
                let pos = (r * cols + c) as u32;
                exceptions.extend_from_slice(&((u32::from(byte) << 24) | pos).to_le_bytes());
                0
            };
            let at = CSF_SLAB_ROWS + i * (cols / 2) + c / 2;
            slab_bytes[at] |= if c % 2 == 0 { code } else { code << 4 };
        }
    }
    (fixed, rows / CSF_SLAB_ROWS, fixed_cols, exceptions)
}

/// The two routed-expert scale shapes Flash-Next ships: gate/up `[640, 160]`
/// and down `[2560, 40]`.
#[test]
fn csf_round_trips_the_flash_next_expert_shapes() {
    for (rows, cols) in [(640, 160), (2560, 40)] {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (rows * cols) as u64);
        let mut raw: Vec<u8> = (0..rows * cols).map(|_| rng.byte_in(0x30, 0x40)).collect();
        for _ in 0..64 {
            let at = (rng.next() as usize) % raw.len();
            raw[at] = 0x7E; // far outside every row's window
        }
        let (fixed, fr, fc, exc) = encode(&raw, rows, cols);
        assert_eq!(csf_scale_shape(fr, fc).unwrap(), (rows, cols));
        assert!(exc.len() / 4 >= 64, "outliers must travel as exceptions");
        assert_eq!(decode_csf_scales(&fixed, fr, fc, &exc).unwrap(), raw);
    }
}

/// The real checkpoint's down_proj stream: `[160, 336]` fixed encodes
/// `[2560, 40]` scales.
#[test]
fn csf_shape_matches_the_checkpoint_header() {
    assert_eq!(csf_scale_shape(160, 336).unwrap(), (2560, 40));
    assert_eq!(csf_scale_shape(40, 1296).unwrap(), (640, 160));
    assert!(csf_scale_shape(40, 1295).is_err());
}

#[test]
fn csf_rejects_bad_streams() {
    let raw = vec![0x38u8; 16 * 4];
    let (mut fixed, fr, fc, _) = encode(&raw, 16, 4);
    // Exception past the end of the matrix.
    let bad = ((0x38u32 << 24) | 64).to_le_bytes();
    assert!(decode_csf_scales(&fixed, fr, fc, &bad).is_err());
    // A torn U32 word.
    assert!(decode_csf_scales(&fixed, fr, fc, &[1, 2, 3]).is_err());
    // Wrong byte count for the declared shape.
    assert!(decode_csf_scales(&fixed[1..], fr, fc, &[]).is_err());
    // A base no nibble can lift to 255 is not a legal stream.
    fixed[0] = 241;
    assert!(decode_csf_scales(&fixed, fr, fc, &[]).is_err());
}

/// MXFP8 -> BF16 must be EXACT: compare against an f64 reconstruction that
/// shares no code with the decoder.
#[test]
fn mxfp8_expansion_is_exact() {
    let (out, inp) = (8, 96);
    let mut rng = Rng(42);
    let weight: Vec<u8> = (0..out * inp)
        .map(|_| {
            let b = (rng.next() & 0xFF) as u8;
            if b & 0x7F == 0x7F { 0x38 } else { b } // skip NaN codes
        })
        .collect();
    let scale: Vec<u8> = (0..out * inp / 32).map(|_| rng.byte_in(110, 135)).collect();
    let got = mxfp8_to_bf16(&weight, &scale, out, inp).unwrap();
    for (i, &w) in weight.iter().enumerate() {
        let (sign, exp, man) = (w >> 7, (w >> 3) & 0xF, w & 7);
        let mag = if exp == 0 {
            f64::from(man) / 8.0 * 2f64.powi(-6)
        } else {
            (1.0 + f64::from(man) / 8.0) * 2f64.powi(i32::from(exp) - 7)
        };
        let signed = if sign == 1 { -mag } else { mag };
        let v = signed * 2f64.powi(i32::from(scale[i / 32]) - 127);
        let bits = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let back = f64::from(f32::from_bits(u32::from(bits) << 16));
        assert_eq!(
            back,
            v,
            "element {i}: byte {w:#04x} scale {}",
            scale[i / 32]
        );
    }
}

#[test]
fn mxfp8_rejects_misaligned_widths() {
    assert!(mxfp8_to_bf16(&[0; 48], &[127; 1], 1, 48).is_err());
    assert!(mxfp8_to_bf16(&[0; 64], &[127; 1], 1, 64).is_err());
}

/// NVFP4 rows decode to the same values as the existing whole-matrix
/// dequant, cast through the shared BF16 rounding.
#[test]
fn nvfp4_rows_match_the_reference_dequant() {
    let (rows, dim, group) = (5, 160, 16);
    let mut rng = Rng(7);
    let packed: Vec<u8> = (0..rows * dim / 2)
        .map(|_| (rng.next() & 0xFF) as u8)
        .collect();
    let scale: Vec<u8> = (0..rows * dim / group)
        .map(|_| rng.byte_in(0x20, 0x60))
        .collect();
    let global = 3.1e-4f32;
    let mut got = vec![0u8; rows * dim * 2];
    nvfp4_rows_to_bf16(&packed, &scale, global, dim, group, &mut got).unwrap();
    let want = nvfp4_dequant(&packed, &scale, global, rows, dim, group).unwrap();
    for (i, w) in want.iter().enumerate() {
        let bits = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        assert_eq!(bits, f32_to_bf16(*w), "element {i}");
    }
}
