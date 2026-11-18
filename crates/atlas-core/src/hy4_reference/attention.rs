// SPDX-License-Identifier: AGPL-3.0-only

use super::{bf16, finite, sigmoid};
use crate::config::Hy4Rope;
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadPlan {
    pub start: usize,
    pub local: usize,
}
impl HeadPlan {
    /// Global (unsharded) head count, not ModelConfig's post-TP local count.
    pub fn new(global: usize, rank: usize, world: usize) -> Result<Self> {
        ensure!(
            global > 0 && matches!(world, 1 | 2) && rank < world && global.is_multiple_of(world),
            "Hy4 head partition"
        );
        Ok(Self {
            start: rank * (global / world),
            local: global / world,
        })
    }
}

pub fn window_indices(position: usize, window: usize) -> Result<Vec<usize>> {
    ensure!(window > 0, "Hy4 window must be positive");
    Ok((position.saturating_sub(window - 1)..=position).collect())
}

/// Main attention uses adjacent pairs. IndexPool converts the rotary suffix
/// to split-half storage AFTER rotation; both use the same pair frequencies.
pub fn rotate(x: &[f32], position: [i64; 3], rope: &Hy4Rope, split_half: bool) -> Result<Vec<f32>> {
    let pairs = rope.mrope_section.iter().sum::<usize>();
    ensure!(
        pairs > 0
            && pairs.checked_mul(2).is_some_and(|r| r <= x.len())
            && rope.rope_type == "default"
            && rope.rope_theta.is_finite()
            && rope.rope_theta > 0.
            && (rope.rope_theta as f32).is_finite()
            && rope.rope_theta as f32 > 0.
            && position.iter().all(|p| *p >= 0),
        "Hy4 rotary parameters/shape"
    );
    finite(x)?;
    let offset = x.len() - pairs * 2;
    let mut out = x.to_vec();
    for i in 0..pairs {
        let axis = if rope.mrope_interleaved {
            if i % 3 == 1 && i < rope.mrope_section[1] * 3 {
                1
            } else if i % 3 == 2 && i < rope.mrope_section[2] * 3 {
                2
            } else {
                0
            }
        } else if i < rope.mrope_section[0] {
            0
        } else if i < rope.mrope_section[0] + rope.mrope_section[1] {
            1
        } else {
            2
        };
        let angle =
            position[axis] as f32 * (rope.rope_theta as f32).powf(-(i as f32) / (pairs as f32));
        let (s, c) = angle.sin_cos();
        let (a, b) = (x[offset + 2 * i], x[offset + 2 * i + 1]);
        let (l, r) = if split_half {
            (offset + i, offset + pairs + i)
        } else {
            (offset + 2 * i, offset + 2 * i + 1)
        };
        out[l] = bf16(a * c - b * s);
        out[r] = bf16(a * s + b * c);
    }
    Ok(out)
}

/// Stable softmax with a virtual zero-valued sink. Empty real-key sets are
/// well-defined (all probability mass goes to the sink).
pub fn sink_softmax(scores: &[f32], sink: f32) -> Result<Vec<f32>> {
    finite(scores)?;
    finite(&[sink])?;
    let max = scores.iter().copied().fold(sink, f32::max);
    let exps: Vec<_> = scores.iter().map(|s| (*s - max).exp()).collect();
    let denom = exps.iter().sum::<f32>() + (sink - max).exp();
    Ok(exps.into_iter().map(|v| v / denom).collect())
}

/// Explicit per-head K/V reference. Indices are logical token IDs, not slots;
/// -1 padding and future keys contribute neither numerator nor denominator.
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    value_dim: usize,
    indices: &[i32],
    position: usize,
    window: Option<usize>,
    sink: f32,
    scale: f32,
) -> Result<Vec<f32>> {
    ensure!(
        !q.is_empty()
            && value_dim > 0
            && k.len().is_multiple_of(q.len())
            && k.len() / q.len() * value_dim == v.len()
            && scale.is_finite()
            && scale > 0.
            && window != Some(0),
        "Hy4 attention shape/scale"
    );
    finite(q)?;
    finite(k)?;
    finite(v)?;
    let tokens = k.len() / q.len();
    let mut rows = Vec::new();
    let mut scores = Vec::new();
    for &index in indices {
        ensure!(index >= -1, "Hy4 invalid sparse sentinel");
        if index < 0 {
            continue;
        }
        let j = index as usize;
        ensure!(j < tokens, "Hy4 sparse token exceeds KV");
        if j > position || window.is_some_and(|w| j < position.saturating_sub(w - 1)) {
            continue;
        }
        ensure!(!rows.contains(&j), "Hy4 duplicate sparse token");
        rows.push(j);
        scores.push(
            q.iter()
                .zip(&k[j * q.len()..(j + 1) * q.len()])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale,
        );
    }
    let probabilities = sink_softmax(&scores, sink)?;
    let mut out = vec![0.; value_dim];
    for (row, p) in rows.into_iter().zip(probabilities) {
        for d in 0..value_dim {
            out[d] += p * v[row * value_dim + d];
        }
    }
    Ok(out.into_iter().map(bf16).collect())
}

pub fn gate(attention: &[f32], logits: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        attention.len() == logits.len(),
        "Hy4 elementwise gate shape"
    );
    finite(attention)?;
    finite(logits)?;
    Ok(attention
        .iter()
        .zip(logits)
        .map(|(a, g)| bf16(a * sigmoid(*g)))
        .collect())
}

#[cfg(test)]
mod tests;
