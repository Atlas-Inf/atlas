// SPDX-License-Identifier: AGPL-3.0-only

#[path = "engram_rng.rs"]
mod rng;
use super::{bf16, finite, rms, sigmoid};
use anyhow::{Context, Result, ensure};

pub fn multipliers(
    seed: u64,
    layer_id: usize,
    compressed_vocab: usize,
    order: usize,
) -> Result<Vec<u64>> {
    ensure!(
        compressed_vocab > 0 && compressed_vocab <= i32::MAX as usize && order >= 2,
        "Hy4 Engram multiplier geometry"
    );
    let seed = seed
        .checked_add(
            10007u64
                .checked_mul(layer_id as u64)
                .context("Engram seed overflow")?,
        )
        .context("Engram seed overflow")?;
    let bound = ((i64::MAX as u64 / compressed_vocab as u64) / 2).max(1);
    let mut g = rng::Pcg64::new(seed);
    Ok((0..order).map(|_| g.bounded(bound) * 2 + 1).collect())
}

/// Balance whole heads by row count (not by head count), as FlashEmbedding.
pub fn head_range(sizes: &[usize], rank: usize, world: usize) -> Result<std::ops::Range<usize>> {
    ensure!(
        matches!(world, 1 | 2)
            && rank < world
            && sizes.len() >= world
            && sizes.iter().all(|n| *n > 0),
        "Hy4 Engram head partition"
    );
    let total = sizes
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .context("Engram row overflow")?;
    ensure!(
        total <= i32::MAX as usize,
        "Hy4 Engram offsets exceed int32"
    );
    if world == 1 {
        return Ok(0..sizes.len());
    }
    let mut prefix = 0usize;
    let mut best = 1;
    let mut distance = usize::MAX;
    for (i, &size) in sizes.iter().enumerate().take(sizes.len() - 1) {
        prefix += size;
        let delta = (prefix * 2).abs_diff(total);
        if delta < distance {
            best = i + 1;
            distance = delta;
        }
    }
    Ok(if rank == 0 {
        0..best
    } else {
        best..sizes.len()
    })
}

#[derive(Clone)]
pub struct HashLayout {
    pub map: Vec<u32>,
    pub pad: u32,
    pub heads_per_order: usize,
    pub multipliers: Vec<u64>,
    pub sizes: Vec<usize>,
}
impl HashLayout {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.map.is_empty()
                && (self.pad as usize) < self.map.len()
                && self.multipliers.len() >= 2
                && self.heads_per_order > 0
                && self.sizes.len() == (self.multipliers.len() - 1) * self.heads_per_order
                && self.map.iter().all(|v| *v <= i32::MAX as u32),
            "Hy4 Engram hash geometry"
        );
        head_range(&self.sizes, 0, 1)?;
        ensure!(
            self.multipliers
                .iter()
                .all(|m| m % 2 == 1 && *m <= i64::MAX as u64),
            "Hy4 Engram multiplier overflow"
        );
        Ok(())
    }
    /// `history` is request-local, oldest first. Dead IDs block all further
    /// lookback; missing prefix tokens use mapped padding. Current padding
    /// masks injection but remains ordinary mapped context for later tokens.
    pub fn hash(&self, history: &[i64], token: i64) -> Result<Vec<i32>> {
        self.validate()?;
        if token < 0 || token as usize >= self.map.len() || token == self.pad as i64 {
            return Ok(vec![-1; self.sizes.len()]);
        }
        let mut rolling = 0u64;
        let mut blocked = false;
        let mut offset = 0usize;
        let mut result = Vec::new();
        for (shift, &multiplier) in self.multipliers.iter().enumerate() {
            let source = if shift == 0 {
                Some(token)
            } else {
                history.len().checked_sub(shift).map(|i| history[i])
            };
            let value = match source {
                None => {
                    blocked = true;
                    self.map[self.pad as usize]
                }
                Some(v) if v < 0 || v as usize >= self.map.len() => {
                    blocked = true;
                    self.map[self.pad as usize]
                }
                Some(v) if !blocked => self.map[v as usize],
                _ => self.map[self.pad as usize],
            };
            rolling ^= (value as u64).wrapping_mul(multiplier);
            if shift > 0 {
                let begin = (shift - 1) * self.heads_per_order;
                for &size in &self.sizes[begin..begin + self.heads_per_order] {
                    result
                        .push(((rolling as i64).rem_euclid(size as i64) as usize + offset) as i32);
                    offset += size;
                }
            }
        }
        Ok(result)
    }
}

pub fn fp8_row(bytes: &[u8], scale: f32, gamma: &[f32], valid: bool) -> Result<Vec<f32>> {
    ensure!(
        !bytes.is_empty() && bytes.len() == gamma.len() && scale.is_finite() && scale > 0.,
        "Hy4 Engram FP8 row geometry/scale"
    );
    finite(gamma)?;
    if !valid {
        return Ok(vec![0.; bytes.len()]);
    }
    ensure!(bytes.iter().all(|b| b & 127 != 127), "Hy4 Engram FP8 NaN");
    let x: Vec<_> = bytes
        .iter()
        .map(|&b| crate::numeric::fp8_e4m3_to_f32(b) * scale)
        .collect();
    finite(&x)?;
    rms(&x, gamma, f32::EPSILON)
}

/// Key/value projections and TP reduction precede this oracle. Key/query
/// normalization rounds to BF16 before the gate dot, as nn.RMSNorm does.
#[allow(clippy::too_many_arguments)]
pub fn inject(
    hidden: &[f32],
    keys: &[f32],
    value: &[f32],
    key_norm: &[f32],
    query_norm: &[f32],
    pre_gamma: &[f32],
    scales: &[f32],
    valid: bool,
) -> Result<Vec<f32>> {
    let h = value.len();
    ensure!(
        h > 0
            && hidden.len() == 4 * h
            && keys.len() == 4 * h
            && key_norm.len() == 4 * h
            && query_norm.len() == 4 * h
            && pre_gamma.len() == h
            && scales.len() == 4 * h,
        "Hy4 Engram injection shape"
    );
    for v in [hidden, keys, value, key_norm, query_norm, pre_gamma, scales] {
        finite(v)?;
    }
    if !valid {
        return Ok(hidden.to_vec());
    }
    let inv = (value.iter().map(|v| v * v).sum::<f32>() / h as f32 + f32::EPSILON)
        .sqrt()
        .recip();
    let mut output = vec![0.; 4 * h];
    for c in 0..4 {
        let range = c * h..(c + 1) * h;
        let k = rms(&keys[range.clone()], &key_norm[range.clone()], f32::EPSILON)?;
        let q = rms(
            &hidden[range.clone()],
            &query_norm[range.clone()],
            f32::EPSILON,
        )?;
        let z =
            bf16(bf16(k.iter().zip(q).map(|(k, q)| bf16(k * q)).sum::<f32>()) / (h as f32).sqrt());
        let signed = if z == 0. {
            0.
        } else {
            z.signum() * bf16(z.abs().max(bf16(1e-6)).sqrt())
        };
        let gate = bf16(sigmoid(signed));
        for d in 0..h {
            let signal = pre_gamma[d] * value[d] * inv;
            let update = bf16((gate + scales[c * h + d]) * signal);
            output[c * h + d] = bf16(hidden[c * h + d] + update);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
