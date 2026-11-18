// SPDX-License-Identifier: AGPL-3.0-only

use super::{bf16, finite};
use crate::numeric::fp8_e4m3_to_f32;
use anyhow::{Result, ensure};

/// Per-channel softmax over the four token members, followed by weighted RMS.
pub fn gated_pool(
    values: &[f32],
    gates: &[f32],
    ape: &[f32],
    norm: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    ensure!(
        values.len() == 512
            && gates.len() == 512
            && ape.len() == 512
            && norm.len() == 128
            && eps.is_finite()
            && eps > 0.,
        "Hy4 IndexPool shape/epsilon"
    );
    finite(values)?;
    finite(gates)?;
    finite(ape)?;
    finite(norm)?;
    let mut out = vec![0.; 128];
    for d in 0..128 {
        let max = (0..4)
            .map(|j| gates[j * 128 + d] + ape[j * 128 + d])
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = (0..4)
            .map(|j| (gates[j * 128 + d] + ape[j * 128 + d] - max).exp())
            .sum::<f32>();
        out[d] = (0..4)
            .map(|j| {
                values[j * 128 + d] * (gates[j * 128 + d] + ape[j * 128 + d] - max).exp()
                    / denominator
            })
            .sum();
    }
    let inv = (out.iter().map(|x| x * x).sum::<f32>() / 128. + eps)
        .sqrt()
        .recip();
    Ok(out
        .iter()
        .zip(norm)
        .map(|(x, w)| bf16(x * inv * w))
        .collect())
}

pub fn fwht_quantize(vector: &[f32]) -> Result<(Vec<u8>, f32)> {
    ensure!(vector.len() == 128, "Hy4 FWHT requires width 128");
    finite(vector)?;
    let mut x = vector.to_vec();
    for stride in [1, 2, 4, 8, 16, 32, 64] {
        for block in x.chunks_exact_mut(stride * 2) {
            for i in 0..stride {
                let (a, b) = (block[i], block[i + stride]);
                block[i] = a + b;
                block[i + stride] = a - b;
            }
        }
    }
    for v in &mut x {
        *v *= 128f32.sqrt().recip();
    }
    finite(&x)?;
    let maximum = x.iter().map(|v| v.abs()).fold(1e-4, f32::max);
    let scale = 2f32.powf((maximum / 448.).log2().ceil());
    let quant = x
        .iter()
        .map(|v| {
            let target = (v / scale).abs().min(448.);
            let mut best = 0u8;
            let mut error = f32::INFINITY;
            for byte in 0..=126u8 {
                let delta = (fp8_e4m3_to_f32(byte) - target).abs();
                if delta < error || (delta == error && byte % 2 == 0) {
                    best = byte;
                    error = delta;
                }
            }
            best | if v.is_sign_negative() { 128 } else { 0 }
        })
        .collect();
    Ok((quant, scale))
}

/// Quantized MQA score: sum_h weight_h * relu(dot(q_h,k)) / sqrt(heads*128).
pub fn score(query: &[(Vec<u8>, f32)], weights: &[f32], key: &(Vec<u8>, f32)) -> Result<f32> {
    ensure!(
        !query.is_empty()
            && query.len() <= 64
            && query.len() == weights.len()
            && key.0.len() == 128
            && key.1.is_finite()
            && key.1 > 0.,
        "Hy4 index score shape"
    );
    finite(weights)?;
    let mut score = 0.;
    for ((q, scale), weight) in query.iter().zip(weights) {
        ensure!(
            q.len() == 128 && scale.is_finite() && *scale > 0.,
            "Hy4 quantized query shape"
        );
        ensure!(
            q.iter().chain(&key.0).all(|b| b & 127 != 127),
            "Hy4 NaN FP8 code"
        );
        let dot = q
            .iter()
            .zip(&key.0)
            .map(|(a, b)| fp8_e4m3_to_f32(*a) * fp8_e4m3_to_f32(*b))
            .sum::<f32>();
        score += weight * scale * key.1 * dot.max(0.);
    }
    finite(&[score])?;
    Ok(score / (query.len() as f32 * 128.).sqrt())
}

/// Select by score, return selected historical groups sorted in logical order.
pub fn select_groups(scores: &[f32], upper: usize, budget: usize) -> Result<Vec<i32>> {
    ensure!(
        upper <= scores.len() && budget > 0 && scores.len() <= i32::MAX as usize,
        "Hy4 top-k bounds"
    );
    finite(&scores[..upper])?;
    let mut ids: Vec<_> = (0..upper).collect();
    ids.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    ids.truncate(budget);
    ids.sort_unstable();
    let mut out: Vec<_> = ids.into_iter().map(|i| i as i32).collect();
    out.resize(budget, -1);
    Ok(out)
}

pub fn expand_groups(groups: &[i32], position: usize, active: bool) -> Result<Vec<i32>> {
    ensure!(
        position <= i32::MAX as usize - 3,
        "Hy4 logical position overflow"
    );
    let upper = position / 4;
    let mut out = Vec::new();
    if active {
        for &group in groups {
            ensure!(group >= -1, "Hy4 invalid group sentinel");
            if group >= 0 && (group as usize) < upper {
                let start = group * 4;
                ensure!(!out.contains(&start), "Hy4 duplicate group");
                out.extend(start..start + 4);
            }
        }
        out.extend((upper * 4..=position).map(|v| v as i32));
    }
    out.resize(groups.len() * 4 + 4, -1);
    Ok(out)
}

#[cfg(test)]
mod tests;
