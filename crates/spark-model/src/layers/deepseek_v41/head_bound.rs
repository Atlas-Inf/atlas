// SPDX-License-Identifier: AGPL-3.0-only

//! How far a token's logit through the FP8 copy of the head can lie from its
//! logit through the head itself: what lets a step that only wants the
//! largest logit read the copy, half the bytes, and the head for the few
//! tokens the copy cannot rule out (`head.rs`).
//!
//! For a head row `w`, its copy `c` and a hidden row `x`, with `|v|` the
//! 2-norm:
//!
//! - the two rows differ by `r = w - c`, so `|w.x - c.x| <= |r| |x|`;
//! - `dense_gemv_bf16` adds `w.x` in FP32, 86 additions deep (80 a lane, a
//!   warp's tree of 5, one more), each within half a unit in the last place
//!   of a sum that `|w| |x|` bounds, and its products are exact (two BF16
//!   values have 16 bits between them);
//! - the tensor cores add `c.x` in an order and with a rounding their maker
//!   does not state. Taken here as at most four units in the last place an
//!   element, of sums that `|c| |x|` bounds, which is far more than any
//!   rounding to FP32 gives.
//!
//! So `|exact - coarse| <= |x| * bound(row)`, with both sides as the kernels
//! leave them before the head's value is rounded to BF16. That rounding is
//! taken care of where the bound is used: it keeps order, so what is below
//! before it is not above after it.

use atlas_core::numeric::fp8_e4m3_to_f32;

use super::cache_recipe::bf16_to_f32;

/// Additions deep `dense_gemv_bf16` is, with room.
const EXACT_DEPTH: f64 = 128.0;
/// Half a unit in the last place of an FP32 sum, as a part of it.
const HALF_UNIT: f64 = 1.0 / (1u64 << 24) as f64;
/// What the bound is widened by, for the roundings of the kernels that
/// apply it: the norm of `x` and a product and a sum in FP32.
const WIDENED: f64 = 1.0 + 1.0 / 1024.0;

/// The bound of one head row: `w` its BF16 values, `codes` its copy's E4M3
/// bytes and `scales` their E8M0 exponents, one to 32 values.
pub fn row_bound(w: &[u16], codes: &[u8], scales: &[u8]) -> f32 {
    let (mut rr, mut ww, mut cc) = (0.0f64, 0.0f64, 0.0f64);
    for ((w, codes), &scale) in w.chunks(32).zip(codes.chunks(32)).zip(scales) {
        let scale = 2.0f64.powi(i32::from(scale) - 127);
        for (&w, &code) in w.iter().zip(codes) {
            let (w, c) = (
                f64::from(bf16_to_f32(w)),
                f64::from(fp8_e4m3_to_f32(code)) * scale,
            );
            rr += (w - c) * (w - c);
            ww += w * w;
            cc += c * c;
        }
    }
    let coarse_depth = 8.0 * (w.len() as f64 + 64.0);
    let bound = rr.sqrt() + HALF_UNIT * (EXACT_DEPTH * ww.sqrt() + coarse_depth * cc.sqrt());
    // Up to the next FP32 value: a cast rounds to the nearest.
    let bound = (bound * WIDENED) as f32;
    f32::from_bits(bound.to_bits() + 1)
}

/// [`row_bound`] of each of a head's rows of `k` values.
pub fn row_bounds(w: &[u16], codes: &[u8], scales: &[u8], k: usize) -> Vec<f32> {
    w.chunks_exact(k)
        .zip(codes.chunks_exact(k))
        .zip(scales.chunks_exact(k / 32))
        .map(|((w, codes), scales)| row_bound(w, codes, scales))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{row_bound, row_bounds};
    use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;
    use atlas_core::numeric::fp8_e4m3_to_f32;

    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// The E4M3 byte of a value E4M3 holds.
    fn e4m3(v: f32) -> u8 {
        (0..=255u8)
            .find(|&b| fp8_e4m3_to_f32(b).to_bits() == v.to_bits())
            .expect("an E4M3 value")
    }

    #[test]
    fn a_row_its_copy_holds_is_bounded_by_the_rounding_alone() {
        // 64 values of 1.5 under a scale of 2: the copy holds 0.75.
        let w = vec![bf16(1.5); 64];
        let codes = vec![e4m3(0.75); 64];
        let bound = row_bound(&w, &codes, &[128, 128]);
        // |w| = |c| = 12; 128 + 8 * 128 half units of it, widened.
        let rounding = 12.0 * (128.0 + 8.0 * 128.0) / 16_777_216.0;
        assert!(bound as f64 > rounding && (bound as f64) < rounding * 1.002);
    }

    #[test]
    fn what_the_copy_lost_is_in_the_bound() {
        // Each value is 1/128 above what its copy holds: |r| = 8/128 over
        // 64 values.
        let w = vec![bf16(1.0 + 1.0 / 128.0); 64];
        let codes = vec![e4m3(1.0); 64];
        let bound = f64::from(row_bound(&w, &codes, &[127, 127]));
        assert!(bound > 0.0625 && bound < 0.0625 + 0.001);
    }

    #[test]
    fn no_hidden_row_takes_a_logit_past_its_bound() {
        // Rows whose copies are off by varying amounts, under varying
        // scales, against hidden rows of both signs: the copy's logit is
        // within the bound of the head's, in exact arithmetic.
        let k = 64;
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let held = [0.0f32, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 3.5];
        for _ in 0..200 {
            let scales = [125 + (next() % 5) as u8, 125 + (next() % 5) as u8];
            let codes: Vec<u8> = (0..k)
                .map(|_| {
                    let v = held[(next() % 10) as usize];
                    e4m3(if next() % 2 == 0 { v } else { -v })
                })
                .collect();
            let w: Vec<u16> = codes
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let scale = 2.0f32.powi(i32::from(scales[i / 32]) - 127);
                    let off = 1.0 + ((next() % 9) as f32 - 4.0) / 64.0;
                    bf16(fp8_e4m3_to_f32(c) * scale * off)
                })
                .collect();
            let bound = f64::from(row_bound(&w, &codes, &scales));
            let x: Vec<f64> = (0..k)
                .map(|_| f64::from((next() % 2001) as f32 - 1000.0) / 64.0)
                .collect();
            let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let (mut exact, mut coarse) = (0.0f64, 0.0f64);
            for i in 0..k {
                let scale = 2.0f64.powi(i32::from(scales[i / 32]) - 127);
                exact += f64::from(bf16_to_f32(w[i])) * x[i];
                coarse += f64::from(fp8_e4m3_to_f32(codes[i])) * scale * x[i];
            }
            assert!((exact - coarse).abs() <= norm * bound);
        }
    }

    #[test]
    fn each_row_of_a_head_has_its_own_bound() {
        let w: Vec<u16> = [vec![bf16(1.5); 32], vec![bf16(1.0 + 1.0 / 128.0); 32]].concat();
        let codes: Vec<u8> = [vec![e4m3(0.75); 32], vec![e4m3(1.0); 32]].concat();
        let bounds = row_bounds(&w, &codes, &[128, 127], 32);
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds[0], row_bound(&w[..32], &codes[..32], &[128]));
        assert_eq!(bounds[1], row_bound(&w[32..], &codes[32..], &[127]));
        assert!(bounds[1] > 10.0 * bounds[0]);
    }
}
