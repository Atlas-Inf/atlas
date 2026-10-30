// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen static YaRN policy and frequency table. No I/O or checkpoint mutation.
//! Equations: Peng et al., YaRN (<https://arxiv.org/abs/2309.00071>), and Qwen's
//! published Qwen3.8-Flash-Next rope_parameters contract. Amplitude is applied
//! only to the rotated dimensions, including QSA indexer query/key rotations.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::config::ModelConfig;

pub const NATIVE_CONTEXT: usize = 262_144;

/// Fixed-size rank agreement covers effective values, whether from CLI or
/// checkpoint. The same setting names exist on every model/rank.
pub fn rank_settings(c: &ModelConfig) -> [(&'static str, u64); 7] {
    let enabled = c.model_type == "qwen4_exp" && c.yarn_factor > 0.0;
    let word = |v: u64| if enabled { v } else { 0 };
    [
        ("Qwen YaRN factor", word(u64::from(c.yarn_factor.to_bits()))),
        (
            "Qwen YaRN original context",
            word(c.yarn_original_max_position_embeddings as u64),
        ),
        (
            "Qwen YaRN beta fast",
            word(u64::from(c.yarn_beta_fast.to_bits())),
        ),
        (
            "Qwen YaRN beta slow",
            word(u64::from(c.yarn_beta_slow.to_bits())),
        ),
        (
            "Qwen YaRN amplitude",
            word(u64::from(c.yarn_attention_factor.to_bits())),
        ),
        ("Qwen YaRN theta", word(c.rope_theta.to_bits())),
        ("Qwen YaRN rotary width", word(c.rotary_dim() as u64)),
    ]
}

#[cfg(test)]
#[path = "qwen_yarn_tests.rs"]
mod tests;

/// Parse the official nested text_config.rope_parameters fields. Unknown
/// scaling must fail, never silently fall back to native RoPE.
pub fn parse_parameters(rp: &Value, config: &mut ModelConfig) -> Result<()> {
    let rope_type = rp.get("rope_type").map_or(Some("default"), Value::as_str);
    match rope_type {
        Some("default") => {
            ensure!(
                !rp.get("factor").is_some_and(|v| v.as_f64() != Some(1.0)),
                "Qwen RoPE factor requires rope_type=yarn"
            );
            config.yarn_factor = 0.0;
            config.yarn_attention_factor = 1.0;
            Ok(())
        }
        Some("yarn") => {
            ensure!(
                rp.get("truncate").is_none_or(|v| v.as_bool() == Some(true)),
                "Qwen YaRN supports only truncate=true (the default)"
            );
            config.yarn_factor = rp
                .get("factor")
                .and_then(Value::as_f64)
                .context("Qwen YaRN requires factor")? as f32;
            config.yarn_original_max_position_embeddings = rp
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .context("Qwen YaRN requires original_max_position_embeddings")?
                .try_into()
                .context("Qwen YaRN original context exceeds usize")?;
            config.yarn_beta_fast = number_or(rp, "beta_fast", 32.0)?;
            config.yarn_beta_slow = number_or(rp, "beta_slow", 1.0)?;
            config.yarn_attention_factor =
                number_or(rp, "attention_factor", 1.0 + 0.1 * config.yarn_factor.ln())?;
            validate(config)
        }
        other => anyhow::bail!("unsupported Qwen rope_type {other:?}; expected default or yarn"),
    }
}

fn number_or(rp: &Value, key: &str, fallback: f32) -> Result<f32> {
    match rp.get(key) {
        None => Ok(fallback),
        Some(v) => {
            Ok(v.as_f64()
                .with_context(|| format!("Qwen YaRN {key} must be numeric"))? as f32)
        }
    }
}

pub fn validate(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "qwen4_exp",
        "Qwen YaRN requires qwen4_exp"
    );
    ensure!(
        config.yarn_factor.is_finite() && config.yarn_factor > 1.0,
        "Qwen YaRN factor must be finite and greater than 1"
    );
    ensure!(
        config.yarn_original_max_position_embeddings > 0,
        "Qwen YaRN original context must be positive"
    );
    ensure!(
        config.rope_theta.is_finite() && config.rope_theta > 1.0,
        "Qwen YaRN rope_theta must be finite and greater than 1"
    );
    ensure!(
        config.rotary_dim() > 0 && config.rotary_dim().is_multiple_of(2),
        "Qwen YaRN rotary dimension must be positive and even"
    );
    ensure!(
        config.yarn_beta_slow.is_finite()
            && config.yarn_beta_slow > 0.0
            && config.yarn_beta_fast.is_finite()
            && config.yarn_beta_fast > config.yarn_beta_slow,
        "Qwen YaRN requires 0 < beta_slow < beta_fast"
    );
    ensure!(
        config.yarn_attention_factor.is_finite() && config.yarn_attention_factor > 0.0,
        "Qwen YaRN attention_factor must be finite and positive"
    );
    Ok(())
}

/// Apply the opt-in CLI policy before any derived buffers or weights exist.
/// A checkpoint that already specifies YaRN may only repeat the same factor.
pub fn configure_context(
    config: &mut ModelConfig,
    factor: Option<u8>,
    context: usize,
) -> Result<()> {
    if config.model_type != "qwen4_exp" {
        ensure!(
            factor.is_none(),
            "--qwen-yarn-factor requires Qwen3.8-Flash-Next"
        );
        return Ok(());
    }
    if let Some(factor) = factor {
        ensure!(matches!(factor, 2 | 4), "--qwen-yarn-factor must be 2 or 4");
        if config.yarn_factor > 0.0 {
            ensure!(
                config.yarn_factor == f32::from(factor),
                "--qwen-yarn-factor conflicts with checkpoint YaRN factor"
            );
        } else {
            ensure!(
                config.max_position_embeddings == NATIVE_CONTEXT,
                "Qwen YaRN override requires a native 262144-position checkpoint"
            );
            config.yarn_factor = f32::from(factor);
            config.yarn_original_max_position_embeddings = config.max_position_embeddings;
            config.yarn_beta_fast = 32.0;
            config.yarn_beta_slow = 1.0;
            config.yarn_attention_factor = 1.0 + 0.1 * config.yarn_factor.ln();
        }
    }
    let limit = if config.yarn_factor > 0.0 {
        validate(config)?;
        config.yarn_original_max_position_embeddings as f64 * f64::from(config.yarn_factor)
    } else {
        config.max_position_embeddings as f64
    };
    ensure!(
        context > 0 && context as f64 <= limit,
        "Qwen --max-seq-len={context} exceeds configured RoPE context {limit}; use explicit YaRN scaling for extended context"
    );
    Ok(())
}

/// Frequency ramp: retain high frequencies, interpolate low frequencies,
/// blending in the correction range. Return F32 as in the reference tables.
pub fn inverse_frequencies(config: &ModelConfig) -> Result<Vec<f32>> {
    validate(config)?;
    let dim = config.rotary_dim() as f64;
    let theta = config.rope_theta;
    let original = config.yarn_original_max_position_embeddings as f64;
    let correction = |rotations: f32| {
        dim * (original / (f64::from(rotations) * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * theta.ln())
    };
    let low = correction(config.yarn_beta_fast).floor().max(0.0);
    let high = correction(config.yarn_beta_slow).ceil().min(dim - 1.0);
    let span = if high == low { 0.001 } else { high - low };
    Ok((0..config.rotary_dim() / 2)
        .map(|j| {
            let ramp = ((j as f64 - low) / span).clamp(0.0, 1.0);
            let base = theta.powf(2.0 * j as f64 / dim);
            ((1.0 - ramp) / base + ramp / (f64::from(config.yarn_factor) * base)) as f32
        })
        .collect())
}
