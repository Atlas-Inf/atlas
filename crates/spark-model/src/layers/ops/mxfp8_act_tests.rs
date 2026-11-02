// SPDX-License-Identifier: AGPL-3.0-only

//! Host references for `mxfp8_act_quant.cu` (used by the GPU tests in
//! `mxfp8_act_test_gpu.rs`) and their own CPU checks.
//!
//! The quantization reference implements ModelOpt's rule from its source
//! (NVIDIA/Model-Optimizer @ c7966119,
//! modelopt/torch/kernels/quantization/gemm/tensor_quant_mx.cu,
//! `compute_scale` / `compute_scale_e8m0_NV` / `quantize`, Apache-2.0), with
//! an E4M3 encoder written independently of the device's `cvt`: the nearest
//! of the 126 finite magnitudes, ties to the even code, saturating at 448.

use super::{mx_lt_inner_pad, mx_lt_scale_bytes, mx_lt_sf_offset};

/// OCP E4M3 decode (0x7f / 0xff are NaN).
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0xf), f32::from(b & 7));
    if e == 15 && m == 7.0 {
        return f32::NAN;
    }
    sign * if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// E4M3 of `x`, round to nearest, ties to even, saturating (`__NV_SATFINITE`):
/// |x| > 448 (and +-inf) -> +-448, NaN -> 0x7f.
pub(crate) fn e4m3_rne_satfinite(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = f64::from(x.abs());
    // Codes 0..=0x7e are the finite magnitudes in increasing order.
    let mut best = 0u8;
    for c in 1..=0x7eu8 {
        let (dc, db) = (
            (f64::from(e4m3_to_f32(c)) - a).abs(),
            (f64::from(e4m3_to_f32(best)) - a).abs(),
        );
        if dc < db || (dc == db && c % 2 == 0) {
            best = c;
        }
    }
    if a > 448.0 {
        best = 0x7e;
    }
    sign | best
}

/// ModelOpt's NV E8M0 unscale exponent `e` (scale 2^e) for a block amax.
pub(crate) fn e8m0_exponent(amax: f32) -> i32 {
    if amax == 0.0 || amax.is_infinite() || amax.is_nan() {
        return 0;
    }
    let r = (amax / 448.0).to_bits();
    let (ef, sig) = ((r >> 23) & 0xff, r & 0x7f_ffff);
    if sig > 0 && ef != 0xfe && !(ef == 0 && sig <= 0x40_0000) {
        ef as i32 - 126
    } else {
        ef as i32 - 127
    }
}

/// BF16 bits -> f32.
pub(crate) fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One 32-value block: `(E8M0 byte, E4M3 bytes)`.
pub(crate) fn quant_block(v: &[f32]) -> (u8, Vec<u8>) {
    // ModelOpt clamps |x| to FLT_MAX before the max; fmaxf skips NaN.
    let amax = v
        .iter()
        .map(|x| x.abs().min(f32::MAX))
        .fold(0f32, |a, b| if b.is_nan() { a } else { a.max(b) });
    let e = e8m0_exponent(amax);
    let inv = 2f64.powi(-e) as f32;
    (
        (e + 127) as u8,
        v.iter().map(|&x| e4m3_rne_satfinite(x * inv)).collect(),
    )
}

/// `mxfp8_quant_act_bf16` of `m` BF16 rows of `k`: `(E4M3 [m, k], tiled
/// scales)`, scales of rows past `m` 0.
pub(crate) fn quant_rows(x: &[u16], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let pad = mx_lt_inner_pad(k);
    let mut q = vec![0u8; m * k];
    let mut sf = vec![0u8; mx_lt_scale_bytes(m, k)];
    for r in 0..m {
        for b in 0..k / 32 {
            let vals: Vec<f32> = x[r * k + b * 32..r * k + b * 32 + 32]
                .iter()
                .map(|&h| bf16(h))
                .collect();
            let (s, bytes) = quant_block(&vals);
            q[r * k + b * 32..r * k + b * 32 + 32].copy_from_slice(&bytes);
            sf[mx_lt_sf_offset(r, b, pad)] = s;
        }
    }
    (q, sf)
}

/// Row-major scales `[rows, inner]` (row stride `src_ld`) -> the tiled
/// layout, padding 0.
pub(crate) fn tile_scales(src: &[u8], rows: usize, inner: usize, src_ld: usize) -> Vec<u8> {
    let pad = inner.div_ceil(4) * 4;
    let mut dst = vec![0u8; mx_lt_scale_bytes(rows, inner * 32)];
    for r in 0..rows {
        for c in 0..inner {
            dst[mx_lt_sf_offset(r, c, pad)] = src[r * src_ld + c];
        }
    }
    dst
}

#[test]
fn tiled_offsets_follow_the_cublas_128x4_tile_layout() {
    // Inside a tile: rows 0..32 step 16 B, the four 32-row groups step 4 B,
    // the four columns are adjacent.
    assert_eq!(mx_lt_sf_offset(0, 0, 80), 0);
    assert_eq!(mx_lt_sf_offset(0, 3, 80), 3);
    assert_eq!(mx_lt_sf_offset(1, 0, 80), 16);
    assert_eq!(mx_lt_sf_offset(32, 0, 80), 4);
    assert_eq!(mx_lt_sf_offset(127, 3, 80), 31 * 16 + 3 * 4 + 3);
    // Tiles row-major over (outer / 128, inner_pad / 4).
    assert_eq!(mx_lt_sf_offset(0, 4, 80), 512);
    assert_eq!(mx_lt_sf_offset(128, 0, 80), 20 * 512);
    assert_eq!(mx_lt_inner_pad(320), 12);
    assert_eq!(mx_lt_inner_pad(2560), 80);
    assert_eq!(mx_lt_scale_bytes(1, 320), 128 * 12);
    assert_eq!(mx_lt_scale_bytes(129, 2560), 256 * 80);
    // A bijection onto the buffer.
    let (rows, k) = (256, 320);
    let pad = mx_lt_inner_pad(k);
    let mut seen = vec![false; mx_lt_scale_bytes(rows, k)];
    for r in 0..rows {
        for c in 0..pad {
            let o = mx_lt_sf_offset(r, c, pad);
            assert!(!seen[o], "({r}, {c}) collides at {o}");
            seen[o] = true;
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn e8m0_exponent_is_modelopt_ceil_with_its_edge_cases() {
    assert_eq!(e8m0_exponent(0.0), 0);
    assert_eq!(e8m0_exponent(f32::INFINITY), 0);
    assert_eq!(e8m0_exponent(f32::NAN), 0);
    // amax = 448 * 2^j exactly: ratio is a power of two, e = j.
    assert_eq!(e8m0_exponent(448.0), 0);
    assert_eq!(e8m0_exponent(448.0 * 8.0), 3);
    assert_eq!(e8m0_exponent(448.0 / 1024.0), -10);
    // Just above: the next power.
    assert_eq!(e8m0_exponent(f32::from_bits(448f32.to_bits() + 1)), 1);
    assert_eq!(e8m0_exponent(1.0), -8); // 1/448 = 2^-8.8 -> ceil -8
    // ModelOpt's documented RNE wrinkle (a quotient just above 2^j rounding
    // down to 2^j, giving the floor) cannot fire for E4M3: 448 * 2^j has
    // mantissa 1.75, so the next float up is 1.14 x 2^-24 above it relative,
    // past the half-ulp of the quotient. The rule above still encodes it.
    for j in -100..=100 {
        let a = 448.0 * 2f32.powi(j);
        assert_eq!(e8m0_exponent(f32::from_bits(a.to_bits() + 1)), j + 1);
    }
    // Ratio subnormal: ceil over the subnormal exponent field.
    assert_eq!(e8m0_exponent(f32::from_bits(1)), -127);
    assert_eq!(e8m0_exponent(448.0 * 2f32.powi(-127)), -127);
    assert_eq!(e8m0_exponent(448.0 * 2f32.powi(-127) * 1.5), -126);
    // FLT_MAX (an inf input's clamped magnitude): 2^128 / 448 -> 2^119.2.
    assert_eq!(e8m0_exponent(f32::MAX), 120);
}

#[test]
fn e4m3_encoder_rounds_to_nearest_even_and_saturates() {
    for c in 0..=0x7eu8 {
        for s in [0u8, 0x80] {
            let v = e4m3_to_f32(s | c);
            assert_eq!(e4m3_rne_satfinite(v), s | c, "round trip of {:#x}", s | c);
        }
    }
    // Ties: 1.0625 is halfway between 1.0 (0x38) and 1.125 (0x39) -> 0x38;
    // 1.1875 between 1.125 (0x39) and 1.25 (0x3a) -> 0x3a.
    assert_eq!(e4m3_rne_satfinite(1.0625), 0x38);
    assert_eq!(e4m3_rne_satfinite(1.1875), 0x3a);
    assert_eq!(e4m3_rne_satfinite(-1.1875), 0xba);
    // Saturation and the subnormal floor.
    assert_eq!(e4m3_rne_satfinite(448.0), 0x7e);
    assert_eq!(e4m3_rne_satfinite(470.0), 0x7e);
    assert_eq!(e4m3_rne_satfinite(1e30), 0x7e);
    assert_eq!(e4m3_rne_satfinite(f32::NEG_INFINITY), 0xfe);
    assert_eq!(e4m3_rne_satfinite(f32::NAN), 0x7f);
    assert_eq!(e4m3_rne_satfinite(2f32.powi(-10)), 0x00); // tie with 2^-9 -> even 0
    assert_eq!(e4m3_rne_satfinite(1.5 * 2f32.powi(-10)), 0x01);
    assert_eq!(e4m3_rne_satfinite(-0.0), 0x80);
}

#[test]
fn a_block_quantizes_by_its_amax_and_dequantizes_within_half_an_ulp() {
    let v: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.37).collect();
    let (s, q) = quant_block(&v);
    let e = i32::from(s) - 127;
    // amax 5.735 -> 5.735 / 448 = 2^-6.29 -> e = -6.
    assert_eq!(e, -6);
    for (x, b) in v.iter().zip(&q) {
        let d = e4m3_to_f32(*b) * 2f32.powi(e);
        assert!((d - x).abs() <= x.abs() / 16.0 + 1e-6, "{x} -> {d}");
    }
    // An all-zero block: scale 1, zero bytes.
    let (s, q) = quant_block(&[0.0; 32]);
    assert_eq!(s, 127);
    assert!(q.iter().all(|&b| b == 0));
    // An inf counts as FLT_MAX: e = 120, the inf saturates, the rest flush.
    let mut v = vec![1.0f32; 32];
    v[3] = f32::INFINITY;
    let (s, q) = quant_block(&v);
    assert_eq!(i32::from(s) - 127, 120);
    assert_eq!(q[3], 0x7e);
    assert_eq!(q[0], 0x00);
}
