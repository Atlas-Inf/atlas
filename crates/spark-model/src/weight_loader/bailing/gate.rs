// SPDX-License-Identifier: AGPL-3.0-only

//! Bailing router gating constants and the per-layer SwiGLU clamp limits,
//! parsed from the checkpoint `text_config`.
//!
//! Reference semantics (`modeling_bailing_moe_v3_vl.py`):
//!
//! - Routing (`BailingMoeV3Gate.forward`, ~line 480): FP32 logits →
//!   `scores = sigmoid(logits)`; selection uses `scores + expert_bias` under
//!   group-limited top-k (experts split into `n_group` groups of
//!   `num_experts / n_group`; the two highest scores per group are summed,
//!   the `topk_group` best groups survive, then top-`k` experts). The
//!   dispatched weights are the ORIGINAL (bias-free) sigmoid scores,
//!   renormalized over the selected set when `norm_topk_prob` and scaled by
//!   `routed_scaling_factor`.
//! - SwiGLU limits (`BailingMoeV3MLP.forward`, ~line 422): a positive limit
//!   clamps the activated gate at `max = limit` and the up branch at
//!   `[-limit, limit]`. A list entry of 0 (or an out-of-range index) means no
//!   clamp. Routed experts carry `expert_swiglu_limit_list`, the shared
//!   expert `share_expert_swiglu_limit_list`, both indexed by LAYER.

// Pure-parse library: reached through [`super::config::BailingConfig`]
// (q.v. the dead-code note there) and the colocated tests until the
// ling-3.2-air parser + kernel wiring land.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

/// Sigmoid group-limited top-k routing (`BailingMoeV3Gate`).
#[derive(Clone, Copy, Debug)]
pub(super) struct RouterGating {
    /// Total routed experts per MoE layer.
    pub num_experts: usize,
    /// `num_experts_per_tok` — experts dispatched per token.
    pub top_k: usize,
    /// `n_group` — number of routing groups over the expert axis.
    pub n_group: usize,
    /// `topk_group` — groups surviving group-limited top-k.
    pub topk_group: usize,
    /// Output scale applied after top-k normalization.
    pub routed_scaling_factor: f64,
    /// Renormalize selected scores over their sum.
    pub norm_topk_prob: bool,
    /// `moe_router_enable_expert_bias` — FP32 `gate.expert_bias` routing
    /// bias exists and participates in selection (never in the weights).
    pub expert_bias: bool,
    /// `router_dtype == "fp32"` — the gate runs FP32 logits.
    pub router_fp32: bool,
}

impl RouterGating {
    /// Parse from the checkpoint `text_config` object.
    pub(super) fn from_text_config(text_config: &Value) -> Result<Self> {
        let score_function = text_config
            .get("score_function")
            .and_then(Value::as_str)
            .unwrap_or_default();
        ensure!(
            score_function == "sigmoid",
            "bailing: only sigmoid routing is implemented \
             (modeling `BailingMoeV3Gate`), config says {score_function:?}"
        );
        let routed_scaling_factor = text_config
            .get("routed_scaling_factor")
            .and_then(Value::as_f64)
            .context("bailing: text_config missing routed_scaling_factor")?;
        let router_dtype = text_config
            .get("router_dtype")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let gating = Self {
            num_experts: required_usize(text_config, "num_experts")?,
            top_k: required_usize(text_config, "num_experts_per_tok")?,
            n_group: required_usize(text_config, "n_group")?,
            topk_group: required_usize(text_config, "topk_group")?,
            routed_scaling_factor,
            norm_topk_prob: bool_field(text_config, "norm_topk_prob", true),
            expert_bias: bool_field(text_config, "moe_router_enable_expert_bias", false),
            router_fp32: router_dtype == "fp32",
        };
        gating.validate()?;
        Ok(gating)
    }

    fn validate(&self) -> Result<()> {
        let Self {
            num_experts,
            top_k,
            n_group,
            topk_group,
            routed_scaling_factor,
            ..
        } = *self;
        ensure!(
            num_experts.is_multiple_of(n_group),
            "bailing: num_experts ({num_experts}) must be divisible by n_group ({n_group})"
        );
        ensure!(
            top_k > 0 && top_k <= num_experts,
            "bailing: num_experts_per_tok ({top_k}) out of range for {num_experts} experts"
        );
        ensure!(
            topk_group > 0 && topk_group <= n_group,
            "bailing: topk_group ({topk_group}) out of range for n_group ({n_group})"
        );
        ensure!(
            routed_scaling_factor > 0.0,
            "bailing: routed_scaling_factor must be positive"
        );
        ensure!(
            self.router_fp32,
            "bailing: router_dtype must be fp32 (the reference computes FP32 logits)"
        );
        Ok(())
    }

    /// Experts per routing group: 512 / 8 = 64 for Ling-3.2-air.
    pub(super) fn experts_per_group(&self) -> usize {
        self.num_experts / self.n_group
    }
}

/// Per-layer SwiGLU clamp limits (`expert_swiglu_limit_list` /
/// `share_expert_swiglu_limit_list`), indexed by LAYER index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct SwigluLimits {
    routed: Vec<usize>,
    shared: Vec<usize>,
}

impl SwigluLimits {
    /// Parse both limit lists from `text_config`. Ling-3.2-air ships both
    /// as `num_hidden_layers`-long arrays; entries of 0 mean "no clamp".
    pub(super) fn from_text_config(text_config: &Value) -> Result<Self> {
        let num_layers = required_usize(text_config, "num_hidden_layers")?;
        let routed = index_list(text_config, "expert_swiglu_limit_list", num_layers)?;
        let shared = index_list(text_config, "share_expert_swiglu_limit_list", num_layers)?;
        Ok(Self { routed, shared })
    }

    /// The modeling file's clamp for the routed experts of `layer`:
    /// `limit_list[layer]` when in range and positive, else no clamp.
    pub(super) fn routed_limit(&self, layer: usize) -> Option<usize> {
        positive_at(&self.routed, layer)
    }

    /// Same for the shared expert (`share_expert_swiglu_limit_list`).
    pub(super) fn shared_limit(&self, layer: usize) -> Option<usize> {
        positive_at(&self.shared, layer)
    }
}

/// A positive list entry means "clamp at this value"; 0 or out-of-range is
/// the reference's `None` (`BailingMoeV3SparseMoeBlock::_get_swiglu_limit`).
fn positive_at(list: &[usize], layer: usize) -> Option<usize> {
    list.get(layer).copied().filter(|&limit| limit > 0)
}

/// Parse an optional per-layer index list; when absent every layer is
/// clamp-free (`Vec` of zeros of `num_layers`, keeping the accessors total
/// and preserving the reference's `getattr(..., None)` behaviour).
fn index_list(text_config: &Value, key: &str, num_layers: usize) -> Result<Vec<usize>> {
    match text_config.get(key) {
        None => Ok(vec![0; num_layers]),
        Some(Value::Array(entries)) => {
            let list = entries
                .iter()
                .map(|v| {
                    let value = v.as_u64().or_else(|| v.as_i64().map(|i| i.max(0) as u64));
                    value.map(|u| u as usize)
                })
                .collect::<Option<Vec<usize>>>()
                .with_context(|| format!("bailing: {key} entries must be integers"))?;
            ensure!(
                list.len() == num_layers,
                "bailing: {key} has {} entries but num_hidden_layers is {num_layers}",
                list.len()
            );
            Ok(list)
        }
        Some(_) => bail!("bailing: {key} must be an array"),
    }
}

pub(super) fn required_usize(text_config: &Value, key: &str) -> Result<usize> {
    let value = text_config
        .get(key)
        .with_context(|| format!("bailing: text_config missing {key}"))?
        .as_u64()
        .with_context(|| format!("bailing: text_config field {key} must be an integer"))?;
    Ok(value as usize)
}

fn bool_field(text_config: &Value, key: &str, default: bool) -> bool {
    text_config
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

#[cfg(test)]
#[path = "gate_tests.rs"]
mod tests;
