// SPDX-License-Identifier: AGPL-3.0-only

use super::topology::EpPlan;
use super::{bf16, finite, linear, rms, sigmoid};
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy)]
pub struct Routing {
    pub top_k: usize,
    pub renormalize: bool,
    pub scaling: f32,
}

/// Bias affects selection only, never the returned mixture weights.
/// Equal corrected scores use ascending expert ID as deterministic tie-break.
pub fn route(logits: &[f32], bias: &[f32], p: Routing) -> Result<Vec<(usize, f32)>> {
    ensure!(
        !logits.is_empty()
            && logits.len() <= 512
            && bias.len() == logits.len()
            && p.top_k > 0
            && p.top_k <= logits.len()
            && p.scaling.is_finite()
            && p.scaling > 0.0,
        "Hy4 invalid routing dimensions/scaling"
    );
    finite(logits)?;
    finite(bias)?;
    let score: Vec<_> = logits.iter().map(|v| sigmoid(*v)).collect();
    let mut ids: Vec<_> = (0..score.len()).collect();
    ids.sort_by(|&a, &b| {
        (score[b] + bias[b])
            .total_cmp(&(score[a] + bias[a]))
            .then(a.cmp(&b))
    });
    ids.truncate(p.top_k);
    let denom = if p.renormalize {
        ids.iter().map(|&i| score[i]).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    Ok(ids
        .iter()
        .map(|&i| (i, score[i] / denom * p.scaling))
        .collect())
}

/// Routed only: upper-bound the raw gate; symmetrically bound up.
pub fn swiglu(gate: f32, up: f32, limit: Option<f32>) -> Result<f32> {
    finite(&[gate, up])?;
    let (g, u) = if let Some(l) = limit {
        ensure!(l.is_finite() && l > 0.0, "Hy4 invalid SwiGLU limit");
        (gate.min(l), up.clamp(-l, l))
    } else {
        (gate, up)
    };
    Ok(bf16(g * sigmoid(g) * u))
}

pub struct Mlp<'a> {
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub intermediate: usize,
}
impl Mlp<'_> {
    pub fn forward(&self, x: &[f32], limit: Option<f32>) -> Result<Vec<f32>> {
        let g = linear(x, self.gate, self.intermediate)?;
        let u = linear(x, self.up, self.intermediate)?;
        let a = g
            .iter()
            .zip(u)
            .map(|(g, u)| swiglu(*g, u, limit))
            .collect::<Result<Vec<_>>>()?;
        linear(&a, self.down, x.len())
    }
}

pub struct LatentMoe<'a> {
    pub router: &'a [f32],
    pub bias: &'a [f32],
    pub latent_down: &'a [f32],
    pub latent_up: &'a [f32],
    pub latent: usize,
    pub experts: Vec<Mlp<'a>>,
    pub shared: Mlp<'a>,
    pub routing: Routing,
    pub limit: f32,
    pub norm: &'a [f32],
    pub eps: f32,
}
impl LatentMoe<'_> {
    /// Local routed contribution after the replicated latent-up transform.
    /// Router always sees full H, not the latent representation.
    pub fn routed(&self, x: &[f32], ep: EpPlan) -> Result<Vec<f32>> {
        ensure!(
            ep.experts == self.experts.len()
                && !x.is_empty()
                && self.router.len() == self.experts.len() * x.len(),
            "Hy4 router shape"
        );
        finite(self.router)?;
        finite(x)?;
        let logits: Vec<_> = self
            .router
            .chunks_exact(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect();
        let selected = route(&logits, self.bias, self.routing)?;
        let z = linear(x, self.latent_down, self.latent)?;
        let mut y = vec![0.0; self.latent];
        for (e, w) in selected {
            if ep.owns(e) {
                for (dst, v) in y
                    .iter_mut()
                    .zip(self.experts[e].forward(&z, Some(self.limit))?)
                {
                    *dst += w * v;
                }
            }
        }
        y.iter_mut().for_each(|v| *v = bf16(*v));
        linear(&y, self.latent_up, x.len())
    }

    /// Call once after routed contributions have been summed across EP.
    pub fn finish(&self, x: &[f32], routed: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.len() == routed.len(), "Hy4 shared/routed width mismatch");
        let shared = self.shared.forward(x, None)?;
        let sum: Vec<_> = shared
            .iter()
            .zip(routed)
            .map(|(a, b)| bf16(a + b))
            .collect();
        rms(&sum, self.norm, self.eps)
    }
}

#[cfg(test)]
mod tests;
