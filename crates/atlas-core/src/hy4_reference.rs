// SPDX-License-Identifier: AGPL-3.0-only

//! Scalar FP32 oracles for Hy4. Storage boundaries round to BF16, RNE.
//! Equations: vLLM #68296, af127016aa1e14f1e597b546e14f5fdfbd05db11,
//! HYV4HC{Pre,Post,Head}Layer, HYV4FlashMoE and Flash attention/IndexPool.
//! Not a serving backend.

use anyhow::{Result, ensure};

pub mod attention;
pub mod engram;
pub mod hc;
pub mod indexpool;
pub mod moe;
pub mod paged;
pub mod topology;

#[cfg(test)]
mod torch_tests;

pub fn bf16(x: f32) -> f32 {
    f32::from_bits(u32::from(crate::numeric::f32_to_bf16(x)) << 16)
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn finite(x: &[f32]) -> Result<()> {
    ensure!(x.iter().all(|v| v.is_finite()), "Hy4 non-finite tensor");
    Ok(())
}

/// Row-major [out, in], with an explicit BF16 output boundary.
pub fn linear(x: &[f32], w: &[f32], n: usize) -> Result<Vec<f32>> {
    ensure!(
        !x.is_empty() && n > 0 && n.checked_mul(x.len()) == Some(w.len()),
        "Hy4 linear shape mismatch"
    );
    finite(x)?;
    finite(w)?;
    Ok(w.chunks_exact(x.len())
        .map(|row| bf16(row.iter().zip(x).map(|(a, b)| a * b).sum()))
        .collect())
}

pub fn rms(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    ensure!(
        !x.is_empty() && x.len() == weight.len() && eps.is_finite() && eps > 0.0,
        "Hy4 RMS shape/epsilon"
    );
    finite(x)?;
    finite(weight)?;
    let r = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(x.iter().zip(weight).map(|(v, w)| bf16(v * r * w)).collect())
}
