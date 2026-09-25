// SPDX-License-Identifier: AGPL-3.0-only

//! ExLlamaV3 EXL3 weight format (trellis-coded linears).
//!
//! An EXL3 checkpoint quantizes each linear `<p>` into four tensors —
//! `<p>.trellis` (I16 packed 16x16 tiles), `<p>.suh` / `<p>.svh` (F16 input
//! / output scales) and `<p>.mul1` (I32 codebook tag) — a layout that has no
//! counterpart in the [`Nvfp4Variant`] dispatch every other quant format
//! maps onto. EXL3 linears therefore never go through `base_variant()` /
//! `variant_for()`; loaders must branch on `quant_method` before variant
//! dispatch. [`Exl3Format`] exists so `detect_quant_format` can name the
//! format instead of falling through to the BF16 heuristic (which would
//! treat the checkpoint as "BF16, runtime-quantize to NVFP4"), and
//! [`ensure_not_exl3`] is the single refusal every path that cannot handle
//! EXL3 calls.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// EXL3 checkpoint: recognised, named, and refused everywhere it cannot go.
/// qwen4_exp is supported (materialized to BF16 / requantized at load, see
/// `weight_map::exl3`); every other model type still refuses.
#[derive(Debug)]
pub struct Exl3Format {
    /// Module-path globs from the config's `ignore` list (unquantized
    /// linears, shipped dense). EXL3 writers emit none today; kept so an
    /// `ignore` block is honoured rather than silently dropped.
    pub ignore_modules: Vec<String>,
}

impl Exl3Format {
    pub fn new(ignore_modules: Vec<String>) -> Self {
        Self { ignore_modules }
    }
}

impl QuantFormat for Exl3Format {
    fn name(&self) -> &'static str {
        "exl3"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        unreachable!(
            "EXL3 linears have no Nvfp4Variant; EXL3 loaders must branch on \
             quant_method before variant dispatch"
        );
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}

/// Refuse an EXL3 checkpoint at a loader that cannot handle it.
///
/// `detect_nvfp4_variant` and `WeightFormat::detect` return non-`Result`
/// types, so they cannot carry this error themselves; every entry point
/// that can receive an exl3 checkpoint calls this first instead, so the
/// variant guess is never reached. `site` names the caller in the message.
///
/// * `quant_method != "exl3"` (or no `quantization_config`) -> `Ok(())`,
///   byte-for-byte the previous behaviour for every other checkpoint.
/// * `qwen4_exp`: supported — `build_model` materializes the dense linears
///   to BF16 and `quantized_any` requantizes the experts, so this only
///   validates the EXL3 block and passes. A malformed block still errors,
///   with the field message rather than the generic one.
/// * any other `model_type`: EXL3 packing is model-family agnostic but the
///   trellis loader is not written for the other architectures — refuse.
pub fn ensure_not_exl3(config: &ModelConfig, site: &str) -> Result<()> {
    let Some(qc) = &config.quantization_config else {
        return Ok(());
    };
    if !qc.is_exl3() {
        return Ok(());
    }
    if config.model_type == "qwen4_exp" {
        // A malformed block is still an exl3 checkpoint; refuse with what
        // was wrong rather than with the generic message.
        qc.exl3_config()
            .map_err(|err| anyhow::anyhow!("{err} (at {site})"))?;
        return Ok(());
    }
    bail!(
        "EXL3 checkpoints are only supported for qwen4_exp (Qwen3.8-Flash-Next); \
         got model_type {:?} (at {site})",
        config.model_type
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_format::detect_quant_format;
    use serde_json::json;
    use spark_runtime::weights::WeightStore;

    /// A config carrying an `exl3` `quantization_config` block. The base
    /// factory config's fields are irrelevant to these tests, but it must be
    /// a `qwen4_exp` one: `parse_config` re-parses `quantization_config` from
    /// the raw JSON and would otherwise overwrite it.
    fn exl3_config(model_type: &str) -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = model_type.to_string();
        config.quantization_config = atlas_core::config::parse_quantization_config(&json! {
            {"quantization_config": {
                "quant_method": "exl3", "version": "1.4.4", "bits": 4.05,
                "head_bits": 6, "codebook": "mul1", "out_scales": "always"
            }}
        });
        config
    }

    #[test]
    fn exl3_format_names_itself_and_is_not_ignored() {
        let fmt = Exl3Format::new(vec!["lm_head".to_string()]);
        assert_eq!(fmt.name(), "exl3");
        assert!(!fmt.is_ignored("model.layers.0.mlp.gate_proj"));
        assert!(fmt.is_ignored("lm_head"));
    }

    #[test]
    #[should_panic(expected = "EXL3")]
    fn exl3_format_has_no_base_variant() {
        let _ = Exl3Format::new(Vec::new()).base_variant();
    }

    #[test]
    fn detect_quant_format_picks_exl3_for_an_exl3_config() {
        let store = WeightStore::empty();
        assert_eq!(
            detect_quant_format(&exl3_config("qwen4_exp"), &store).name(),
            "exl3"
        );
    }

    #[test]
    fn ensure_not_exl3_passes_non_exl3_configs() {
        ensure_not_exl3(&ModelConfig::qwen3_next_80b_nvfp4(), "test").expect("modelopt must pass");
    }

    #[test]
    fn ensure_not_exl3_accepts_qwen4_exp() {
        ensure_not_exl3(&exl3_config("qwen4_exp"), "test").expect("qwen4_exp must load");
    }

    #[test]
    fn ensure_not_exl3_refuses_other_model_types_by_name() {
        let err = ensure_not_exl3(&exl3_config("qwen3_next"), "test")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("only supported for qwen4_exp") && err.contains("qwen3_next"),
            "{err}"
        );
    }

    #[test]
    fn ensure_not_exl3_surfaces_a_malformed_block() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        // The malformed-block error is reported on the qwen4_exp path; other
        // model types are refused by name before the block is consulted.
        config.model_type = "qwen4_exp".to_string();
        config.quantization_config = atlas_core::config::parse_quantization_config(&json! {
            {"quantization_config": {
                "quant_method": "exl3", "version": "1.4.4", "head_bits": 6,
                "out_scales": "always", "codebook": "mul1"
            }}
        });
        let err = ensure_not_exl3(&config, "test").unwrap_err().to_string();
        assert!(err.contains("quantization_config.bits"), "{err}");
    }
}
