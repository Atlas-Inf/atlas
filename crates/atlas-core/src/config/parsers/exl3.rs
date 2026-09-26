// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for the ExLlamaV3 `quant_method: "exl3"` block.
//!
//! An EXL3 `quantization_config` carries fields no other Atlas scheme uses
//! (`bits` as a float average bpw, `head_bits`, `codebook`, `out_scales`,
//! plus a potentially huge `tensor_storage` map). Only the small subset that
//! changes how the weights are interpreted is kept; `tensor_storage` is
//! deliberately dropped — it is a per-module inventory of the checkpoint
//! itself and can run to thousands of entries.
//!
//! [`parse_exl3_config`] is called from `parsers/quantization.rs`, which has
//! no error channel (it returns `Option`), so the outcome is carried as
//! `Option<Result<_, String>>` on [`QuantizationConfig::exl3`]: `None` for a
//! non-exl3 checkpoint, `Some(Err(msg))` for an exl3 block that is missing or
//! mistypes a required field, `Some(Ok(..))` otherwise.

use serde_json::Value;

use super::super::QuantizationConfig;

/// The EXL3-specific fields of a `quantization_config` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Exl3QuantConfig {
    /// `version` — the exllamav3 writer version, e.g. `"1.4.4"`.
    pub version: String,
    /// `bits` — average bits per weight (a float, e.g. `4.05`). The per-tensor
    /// codebook depth comes from the trellis dims, not from here.
    pub bits: f64,
    /// `head_bits` — bit width of the model's lm_head ("head") tensors.
    pub head_bits: u32,
    /// `mtp_bits` — bit width of the MTP block's quantized linears, when
    /// declared.
    pub mtp_bits: Option<u32>,
    /// `codebook` — which decoder is implemented. Only `"mul1"` is.
    pub codebook: String,
    /// `out_scales` — scale application mode, e.g. `"always"`.
    pub out_scales: String,
}

/// Parse an exl3 block. `Err` names the offending field.
pub(crate) fn parse_exl3_config(qc: &Value) -> Result<Exl3QuantConfig, String> {
    // Integer fields are read via `as_f64` so a JSON float like `4.0` (which
    // some writers emit for int knobs) still parses; a wrong type or a
    // negative value is reported as a bad value, naming it.
    let int_field = |key: &str| -> Result<u32, String> {
        match qc.get(key) {
            Some(v) => match v.as_f64() {
                Some(f) if f >= 0.0 && f == f.trunc() && f <= u32::MAX as f64 => Ok(f as u32),
                Some(f) => Err(format!(
                    "quantization_config.{key} must be a non-negative integer, got {f}"
                )),
                None => Err(format!(
                    "quantization_config.{key} must be a non-negative integer, got {v}"
                )),
            },
            None => Err(format!("missing quantization_config.{key}")),
        }
    };

    let version = match qc.get("version") {
        Some(v) => v
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("quantization_config.version must be a string, got {v}"))?,
        None => return Err("missing quantization_config.version".to_string()),
    };
    let bits = match qc.get("bits") {
        Some(v) => v
            .as_f64()
            .ok_or_else(|| format!("quantization_config.bits must be a number, got {v}"))?,
        None => return Err("missing quantization_config.bits".to_string()),
    };
    let head_bits = int_field("head_bits")?;
    let codebook = match qc.get("codebook") {
        Some(v) => v
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("quantization_config.codebook must be a string, got {v}"))?,
        None => return Err("missing quantization_config.codebook".to_string()),
    };
    // Optional: absent means the writer's default, which Atlas does not
    // consult (the trellis dims carry the truth), so no error.
    let mtp_bits = match qc.get("mtp_bits") {
        Some(serde_json::Value::Null) | None => None,
        Some(_) => Some(int_field("mtp_bits")?),
    };
    let out_scales = match qc.get("out_scales") {
        Some(v) => v.as_str().map(str::to_string).unwrap_or_default(),
        None => String::new(),
    };

    Ok(Exl3QuantConfig {
        version,
        bits,
        head_bits,
        mtp_bits,
        codebook,
        out_scales,
    })
}

impl QuantizationConfig {
    /// Does this checkpoint declare ExLlamaV3's EXL3 weight format?
    pub fn is_exl3(&self) -> bool {
        self.quant_method.eq_ignore_ascii_case("exl3")
    }

    /// The parsed EXL3 fields, or an error naming what was wrong.
    ///
    /// Also the single gate for the codebook: only `mul1` has a decoder in
    /// Atlas, so anything else is refused here rather than at first decode.
    pub fn exl3_config(&self) -> anyhow::Result<&Exl3QuantConfig> {
        anyhow::ensure!(
            self.is_exl3(),
            "quantization_config.quant_method is {:?}, not \"exl3\"",
            self.quant_method
        );
        let parsed = self.exl3.as_ref().ok_or_else(|| {
            anyhow::anyhow!("EXL3 quantization_config was not parsed (internal error)")
        })?;
        let cfg = match parsed {
            Ok(cfg) => cfg,
            Err(msg) => anyhow::bail!("EXL3 quantization_config is malformed: {msg}"),
        };
        anyhow::ensure!(
            cfg.codebook == "mul1",
            "EXL3 codebook {:?} is not supported; Atlas implements mul1",
            cfg.codebook
        );
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qc_of(block: Value) -> QuantizationConfig {
        crate::config::parse_quantization_config(&json! { {"quantization_config": block} })
            .expect("block must yield a QuantizationConfig")
    }

    #[test]
    fn parses_real_exl3_block() {
        let qc = qc_of(json!({
            "quant_method": "exl3", "version": "1.4.4", "bits": 4.05, "head_bits": 6,
            "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always",
            "codebook": "mul1", "mtp_bits": 4,
            "tensor_storage": {"model.layers.0.self_attn.q_proj": ["trellis", "suh"]}
        }));
        assert!(qc.is_exl3());
        let cfg = qc.exl3_config().expect("valid exl3 block");
        assert_eq!(cfg.version, "1.4.4");
        assert_eq!(cfg.bits, 4.05);
        assert_eq!(cfg.head_bits, 6);
        assert_eq!(cfg.mtp_bits, Some(4));
        assert_eq!(cfg.codebook, "mul1");
        assert_eq!(cfg.out_scales, "always");
    }

    #[test]
    fn missing_bits_errors_naming_bits() {
        let qc = qc_of(json!({
            "quant_method": "exl3", "version": "1.4.4", "head_bits": 6,
            "out_scales": "always", "codebook": "mul1"
        }));
        assert!(qc.is_exl3());
        let err = qc.exl3_config().unwrap_err().to_string();
        assert!(err.contains("bits"), "{err}");
    }

    #[test]
    fn mistyped_head_bits_errors_naming_head_bits() {
        let qc = qc_of(json!({
            "quant_method": "exl3", "version": "1.4.4", "bits": 4.0,
            "head_bits": "six", "out_scales": "always", "codebook": "mul1"
        }));
        let err = qc.exl3_config().unwrap_err().to_string();
        assert!(err.contains("head_bits"), "{err}");
    }

    #[test]
    fn non_mul1_codebook_is_refused_by_name() {
        let qc = qc_of(json!({
            "quant_method": "exl3", "version": "1.4.4", "bits": 4.0, "head_bits": 6,
            "out_scales": "always", "codebook": "lut8"
        }));
        let err = qc.exl3_config().unwrap_err().to_string();
        assert_eq!(
            err,
            "EXL3 codebook \"lut8\" is not supported; Atlas implements mul1"
        );
    }

    #[test]
    fn uppercase_method_parses_like_lowercase() {
        // `is_exl3` is case-insensitive, so the parse condition in
        // quantization.rs must be too, or "EXL3" would hit the
        // "not parsed (internal error)" branch.
        let qc = qc_of(json!({
            "quant_method": "EXL3", "version": "1.4.4", "bits": 4.0, "head_bits": 6,
            "out_scales": "always", "codebook": "mul1"
        }));
        assert!(qc.is_exl3());
        assert_eq!(qc.exl3_config().expect("parses").bits, 4.0);
    }

    #[test]
    fn non_exl3_method_has_no_exl3_field_and_errors() {
        let qc = qc_of(json!({"quant_method": "fp8", "bits": 8}));
        assert!(!qc.is_exl3());
        assert!(qc.exl3.is_none());
        assert!(qc.exl3_config().is_err());
    }
}
