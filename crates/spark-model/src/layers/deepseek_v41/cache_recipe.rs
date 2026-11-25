// SPDX-License-Identifier: AGPL-3.0-only

//! What each DeepSeek-V4.1 attention cache holds, on the host.
//!
//! These are the reference's rules (`inference/model.py` and `kernel.py` of
//! the 1017 release), written out once: the GPU kernels are tested against
//! them, and the loader takes its rotary tables from here. The compressed-KV
//! rule reproduces the reference's cache bit for bit
//! (`sparkseek/results/2026-11-15-reference-layers`).

use std::ops::Range;

use atlas_core::numeric::f32_to_bf16;

/// Positions a layer's sliding window holds.
pub const WINDOW: usize = 128;
/// Values per FP8 scale, and per E8M0 scale of an FP4 index key.
const BLOCK: usize = 32;
/// Values per E4M3 scale of an FP4 compressed key.
const COMPRESSED_GROUP: usize = 16;

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Rotary frequency of each value pair, in radians per position. With
/// `original_seq_len > 0` this is YaRN: pairs whose wavelength fits the
/// training context keep their frequency, those far beyond it are divided by
/// `factor`, and the `beta_fast`..`beta_slow` band between is a linear ramp.
pub fn rope_freqs(
    dim: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    let mut freqs: Vec<f32> = (0..dim / 2)
        .map(|pair| 1.0 / base.powf((2 * pair) as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        // The pair whose wavelength completes `rotations` turns over the
        // training context.
        let corrected = |rotations: f32| {
            dim as f64
                * (original_seq_len as f64 / (f64::from(rotations) * std::f64::consts::TAU)).ln()
                / (2.0 * f64::from(base).ln())
        };
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min(dim as f64 - 1.0);
        let span = (high - low).max(1e-3) as f32;
        for (pair, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((pair as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    freqs
}

/// Rotate adjacent pairs of BF16 `x` to `position`, in place.
pub fn rotate_pairs(x: &mut [u16], position: usize, freqs: &[f32]) {
    for (pair, freq) in x.chunks_exact_mut(2).zip(freqs) {
        let (sin, cos) = (position as f32 * freq).sin_cos();
        let (re, im) = (bf16_to_f32(pair[0]), bf16_to_f32(pair[1]));
        pair[0] = f32_to_bf16(re * cos - im * sin);
        pair[1] = f32_to_bf16(re * sin + im * cos);
    }
}

/// 2^ceil(log2(ratio)), read off the float's exponent as the reference does.
fn pow2_ceil(ratio: f32) -> f32 {
    let bits = ratio.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((exponent + 127) as u32) << 23)
}

fn exponent_of(magnitude: f32) -> i32 {
    ((magnitude.to_bits() >> 23) & 0xff) as i32 - 127
}

/// Nearest E4M3 value (3 mantissa bits, least exponent -6, at most 448).
fn round_e4m3(v: f32) -> f32 {
    let step = 2.0f32.powi(exponent_of(v.abs()).max(-6) - 3);
    ((v.abs() / step).round_ties_even() * step)
        .min(448.0)
        .copysign(v)
}

/// Nearest E2M1 value: 0, 0.5, 1, 1.5, 2, 3, 4 or 6.
fn round_fp4(v: f32) -> f32 {
    let step = 2.0f32.powi(exponent_of(v.abs()).clamp(0, 2) - 1);
    ((v.abs() / step).round_ties_even() * step).copysign(v)
}

fn amax(block: &[u16]) -> f32 {
    block
        .iter()
        .fold(0.0f32, |top, &bits| top.max(bf16_to_f32(bits).abs()))
}

/// A window key as the reference stores it: FP8 (E4M3) under one
/// power-of-two scale per 32 values, written back as BF16.
pub fn fp8_quant_dequant(x: &mut [u16]) {
    for block in x.chunks_mut(BLOCK) {
        let scale = pow2_ceil(amax(block).max(1e-4) * (1.0 / 448.0));
        for bits in block {
            let scaled = (bf16_to_f32(*bits) / scale).clamp(-448.0, 448.0);
            *bits = f32_to_bf16(round_e4m3(scaled) * scale);
        }
    }
}

fn fp4_quant_dequant(x: &mut [u16], group: usize, scale_of: impl Fn(f32) -> f32) {
    for block in x.chunks_mut(group) {
        let scale = scale_of(amax(block));
        for bits in block {
            let scaled = (bf16_to_f32(*bits) / scale).clamp(-6.0, 6.0);
            *bits = f32_to_bf16(round_fp4(scaled) * scale);
        }
    }
}

/// An index key: FP4 under one power-of-two scale per 32 values, as BF16.
pub fn fp4_quant_dequant_e8m0(x: &mut [u16]) {
    fp4_quant_dequant(x, BLOCK, |amax| {
        pow2_ceil(amax.max(6.0 * 2.0f32.powi(-126)) * (1.0 / 6.0))
    });
}

/// A compressed key: FP4 under one E4M3 scale per 16 values, as BF16.
pub fn fp4_quant_dequant_e4m3(x: &mut [u16]) {
    fp4_quant_dequant(x, COMPRESSED_GROUP, |amax| {
        round_e4m3(amax.max(6.0 * 2.0f32.powi(-9)) / 6.0)
    });
}

/// The compressed key a source layer caches for one group: its compressor
/// latent with the last `2 * freqs.len()` values rotated to the group's first
/// position, then quantized.
pub fn compressed_kv_from_latent(latent: &mut [u16], first_position: usize, freqs: &[f32]) {
    let rotated_from = latent.len() - 2 * freqs.len();
    rotate_pairs(&mut latent[rotated_from..], first_position, freqs);
    fp4_quant_dequant_e4m3(latent);
}

/// The ring slot that holds `position`'s window key.
pub fn window_slot(position: usize) -> usize {
    position % WINDOW
}

/// The positions a prefill query at `position` attends to in the window.
pub fn prefill_window(position: usize) -> Range<usize> {
    (position + 1).saturating_sub(WINDOW)..position + 1
}

/// The ring slots a query at `position` attends to: the positions of its
/// window in order, then -1 up to 128 entries. This is the reference's
/// prefill row, with ring slots for positions. The engine builds this row
/// for decode as well, so that a decode step and the same position inside a
/// prefill chunk give the attention kernel the same keys in the same places.
pub fn window_row(position: usize) -> Vec<i32> {
    let mut row: Vec<i32> = prefill_window(position)
        .map(|at| window_slot(at) as i32)
        .collect();
    row.resize(WINDOW, -1);
    row
}

/// The ring slots the reference's own decode step attends to, oldest first;
/// -1 for a slot the sequence has not reached. It differs from
/// [`window_row`] only in where the -1s sit, before position 127.
pub fn decode_window_slots(position: usize) -> Vec<i32> {
    let oldest = window_slot(position) + 1;
    (oldest..WINDOW)
        .chain(0..oldest)
        .map(|slot| if slot > position { -1 } else { slot as i32 })
        .collect()
}

/// The compression group that `position` completes, if it is the group's
/// last token.
pub fn completed_group(position: usize, ratio: usize) -> Option<usize> {
    ((position + 1).is_multiple_of(ratio)).then_some(position / ratio)
}

/// The position a group's latent is rotated to: its first token's.
pub fn group_position(group: usize, ratio: usize) -> usize {
    group * ratio
}

/// How many compressed groups a query at `position` may attend to.
pub fn visible_groups(position: usize, ratio: usize) -> usize {
    (position + 1) / ratio
}

#[cfg(test)]
#[path = "cache_recipe_tests.rs"]
mod tests;
