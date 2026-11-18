// SPDX-License-Identifier: AGPL-3.0-only

use super::{bf16, finite, sigmoid};
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy)]
pub struct HcParams {
    pub hidden: usize,
    pub mult: usize,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub magnitude: f32,
}

impl HcParams {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.hidden > 0
                && self.hidden <= u32::MAX as usize / 4
                && self.mult == 4
                && self.hidden.checked_mul(self.mult).is_some(),
            "Hy4 iHC requires positive hidden width and hc_mult=4"
        );
        ensure!(
            [self.norm_eps, self.gate_eps, self.magnitude]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
            "Hy4 invalid iHC parameters"
        );
        Ok(())
    }
}

/// One token. Gates and weights FP32; input/output highway is BF16.
pub fn pre(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    base: &[f32],
    p: HcParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mix = gates(x, w, base, scale, p, false)?;
    let mut y = vec![0.0; p.hidden];
    for i in 0..p.mult {
        for d in 0..p.hidden {
            y[d] += mix[i] * bf16(x[i * p.hidden + d]);
        }
    }
    y.iter_mut().for_each(|v| *v = bf16(*v));
    Ok((y, mix[p.mult..].to_vec()))
}

pub fn head(x: &[f32], w: &[f32], scale: &[f32], base: &[f32], p: HcParams) -> Result<Vec<f32>> {
    let mix = gates(x, w, base, scale, p, true)?;
    Ok((0..p.hidden)
        .map(|d| {
            bf16(
                (0..p.mult)
                    .map(|i| mix[i] * bf16(x[i * p.hidden + d]))
                    .sum(),
            )
        })
        .collect())
}

fn gates(
    x: &[f32],
    w: &[f32],
    base: &[f32],
    scale: &[f32],
    p: HcParams,
    head: bool,
) -> Result<Vec<f32>> {
    p.validate()?;
    let dim = p.hidden * p.mult;
    let rows = if head { p.mult } else { 2 * p.mult };
    ensure!(
        x.len() == dim
            && w.len() == dim * rows
            && base.len() == rows
            && scale.len() == if head { 1 } else { 2 },
        "Hy4 iHC shape mismatch"
    );
    for v in [x, w, base, scale] {
        finite(v)?;
    }
    let x: Vec<_> = x.iter().map(|v| bf16(*v)).collect();
    let r = (x.iter().map(|v| v * v).sum::<f32>() / dim as f32 + p.norm_eps)
        .sqrt()
        .recip();
    Ok(w.chunks_exact(dim)
        .enumerate()
        .map(|(i, row)| {
            let z = row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() * r;
            let post = !head && i >= p.mult;
            let s = sigmoid(z * scale[usize::from(post)] + base[i]);
            s * if post { p.magnitude } else { 1.0 } + p.gate_eps
        })
        .collect())
}

pub fn post(x: &[f32], residual: &[f32], gate: &[f32], p: HcParams) -> Result<Vec<f32>> {
    p.validate()?;
    ensure!(
        x.len() == p.hidden && residual.len() == p.hidden * p.mult && gate.len() == p.mult,
        "Hy4 iHC post shape mismatch"
    );
    for v in [x, residual, gate] {
        finite(v)?;
    }
    Ok(residual
        .iter()
        .enumerate()
        .map(|(i, r)| bf16(bf16(*r) + gate[i / p.hidden] * bf16(x[i % p.hidden])))
        .collect())
}

#[cfg(test)]
mod tests;
