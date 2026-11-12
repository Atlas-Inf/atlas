// SPDX-License-Identifier: AGPL-3.0-only

//! One startup warning for an FP4-Local checkpoint served with its
//! native-precision switches unset.
//!
//! `weight_loader/qwen4_exp/mode.rs` made the FP4-Local native-precision
//! paths opt-in on this engine (`ATLAS_QWEN4EXP_MXFP8_DECODE`,
//! `_PREFILL_W8A8`, `_MXFP8_SHARED`, `_PREFILL_W4A4`); SparkQwen's engine,
//! where most FP4-Local profiles were first written, turns them on by
//! default. A profile carried over without them serves the MXFP8 GDN/QSA
//! projections from BF16 expansions and prefills in BF16 — the same weights,
//! 15–25% lower decode throughput on a GB10 pair — and nothing in the boot
//! log said so. Unset is the hazard; an explicit `0` is a choice and stays
//! silent. A serving mode validates the same switches itself.

use spark_model::weight_map::modelopt_plan::ModeloptQuantPlan;

/// The switches the serving modes require at `1` for the checkpoint's own
/// formats, in the order the warning lists them.
pub const NATIVE_PRECISION_SWITCHES: [&str; 4] = [
    "ATLAS_QWEN4EXP_MXFP8_DECODE",
    "ATLAS_QWEN4EXP_PREFILL_W8A8",
    "ATLAS_QWEN4EXP_MXFP8_SHARED",
    "ATLAS_QWEN4EXP_PREFILL_W4A4",
];

/// The switches left unset for an FP4-Local checkpoint served outside a
/// serving mode; empty for every other case.
pub fn unset_native_precision_switches(
    is_fp4_local: bool,
    serving_mode: Option<&str>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<&'static str> {
    if !is_fp4_local || serving_mode.is_some() {
        return Vec::new();
    }
    NATIVE_PRECISION_SWITCHES
        .iter()
        .copied()
        .filter(|key| lookup(key).is_none())
        .collect()
}

/// The one WARN line, or `None` when there is nothing to say.
pub fn warning(unset: &[&str]) -> Option<String> {
    if unset.is_empty() {
        return None;
    }
    Some(format!(
        "FP4-Local checkpoint served with {} unset. Unset is off on this engine (SparkQwen's \
         engine defaults them on): the MXFP8 GDN/QSA projections decode from BF16 expansions \
         and prefill runs in BF16, which measured 15-25% lower decode throughput on a GB10 pair \
         at TP=EP=2. Set each explicitly: 1 for the checkpoint's own formats (what the \
         sparkrun-recipes FP4-Local recipes and ATLAS_QWEN4EXP_SERVING_MODE do), or 0 to keep \
         the BF16 path on purpose.",
        unset.join(", ")
    ))
}

/// Reads the model config and the process environment; logs at most one line.
pub fn warn_if_native_precision_unset(config: &atlas_core::config::ModelConfig) {
    let is_fp4_local =
        ModeloptQuantPlan::from_config(config).is_some_and(|plan| plan.is_fp4_local());
    let mode = std::env::var("ATLAS_QWEN4EXP_SERVING_MODE").ok();
    let unset = unset_native_precision_switches(is_fp4_local, mode.as_deref(), |key| {
        std::env::var(key).ok()
    });
    if let Some(line) = warning(&unset) {
        tracing::warn!("{line}");
    }
}

#[cfg(test)]
#[path = "fp4_local_precision_tests.rs"]
mod tests;
