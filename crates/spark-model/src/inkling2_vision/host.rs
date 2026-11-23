// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side tables and weight rearrangement for Inkling 2's vision tower.

use super::ROPE_THETA;

pub(super) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

pub(super) fn f32_to_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
    }
    out
}

/// Bilinear taps into a `side`-long axis for `index` of `size`
/// (`align_corners = false`, edge-clamped): `[(tap, weight); 2]`.
pub(super) fn axis_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = (index as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = src.floor();
    let frac = src - floor;
    let clamp = |t: f32| (t.max(0.0) as usize).min(side - 1);
    [(clamp(floor), 1.0 - frac), (clamp(floor + 1.0), frac)]
}

/// `(row, col)` of each patch in tower order: tiles in raster order, a
/// tile's `merge²` patches together.
pub(super) fn patch_order(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(grid_h * grid_w);
    for tr in 0..grid_h / merge {
        for tc in 0..grid_w / merge {
            for mr in 0..merge {
                for mc in 0..merge {
                    order.push((tr * merge + mr, tc * merge + mc));
                }
            }
        }
    }
    order
}

/// The position table resampled to a `grid_h x grid_w` patch grid, `[n, hidden]`.
pub(super) fn position_rows(
    table: &[f32],
    side: usize,
    hidden: usize,
    order: &[(usize, usize)],
    grid_h: usize,
    grid_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; order.len() * hidden];
    for (p, &(row, col)) in order.iter().enumerate() {
        let dst = &mut out[p * hidden..(p + 1) * hidden];
        for (ht, hw) in axis_taps(row, grid_h, side) {
            for (wt, ww) in axis_taps(col, grid_w, side) {
                let w = hw * ww;
                let src = &table[(ht * side + wt) * hidden..][..hidden];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += w * s;
                }
            }
        }
    }
    out
}

/// `(cos, sin)`, each `[1 + n, head_dim]` f32, in the kernels' rotate-half
/// layout: channel `i` and `i + head_dim/2` carry pair `i`'s angle. Row 0 is
/// the CLS token at (0, 0).
pub(super) fn rope_rows(order: &[(usize, usize)], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let axis = half / 2;
    let freqs: Vec<f64> = (0..axis)
        .map(|j| ROPE_THETA.powf(-(j as f64) / axis as f64))
        .collect();
    let rows = order.len() + 1;
    let (mut cos, mut sin) = (vec![1.0f32; rows * head_dim], vec![0.0f32; rows * head_dim]);
    for (p, &(row, col)) in order.iter().enumerate() {
        let base = (p + 1) * head_dim;
        for i in 0..half {
            let angle = if i < axis {
                (col + 1) as f64 * freqs[i]
            } else {
                (row + 1) as f64 * freqs[i - axis]
            };
            let (s, c) = angle.sin_cos();
            for at in [base + i, base + half + i] {
                cos[at] = c as f32;
                sin[at] = s as f32;
            }
        }
    }
    (cos, sin)
}

/// `rows` of `width` BF16 values with the rows inside each head moved from
/// the reference's interleaved pairs to the kernels' halves.
pub(super) fn permute_head_rows(
    src: &[u8],
    heads: usize,
    head_dim: usize,
    width: usize,
) -> Vec<u8> {
    let (row, half) = (width * 2, head_dim / 2);
    let mut out = vec![0u8; src.len()];
    for h in 0..heads {
        for i in 0..half {
            for (dst, from) in [(i, 2 * i), (half + i, 2 * i + 1)] {
                let (d, s) = ((h * head_dim + dst) * row, (h * head_dim + from) * row);
                out[d..d + row].copy_from_slice(&src[s..s + row]);
            }
        }
    }
    out
}

/// `weight` `[n, width]` (or a bias with `width` 1) with row `r` times `gamma[r]`.
pub(super) fn scale_rows(weight: &[u8], gamma: &[f32], width: usize) -> Vec<u8> {
    let mut w = bf16_to_f32(weight);
    for (r, g) in gamma.iter().enumerate() {
        for v in &mut w[r * width..(r + 1) * width] {
            *v *= g;
        }
    }
    f32_to_bf16(&w)
}
