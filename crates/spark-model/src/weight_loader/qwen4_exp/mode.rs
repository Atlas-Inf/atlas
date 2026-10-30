// SPDX-License-Identifier: AGPL-3.0-only
//! Qwen TP2 modes share native target precision and target-verified MTP.
//! Ported from Enntity/atlas `weight_loader/qwen4_exp/mode.rs` (AGPL-3.0-only),
//! adapted for this stack: `ATLAS_QWEN4EXP_MXFP8_DECODE` selects the
//! tensor-core MXFP8 tiers (`ops::mxfp8_decode_tc`) and is checked by startup
//! parity, not here; `ATLAS_QWEN4EXP_PREFILL_W8A8` and
//! `ATLAS_QWEN4EXP_MTP_BF16_ATTN` have no implementation here (their features
//! are not in this tree), so they are omitted from the flag contract; the corrected-RNE quantizer kernel
//! (`quantize_bf16_to_nvfp4_weight_rne`) likewise has no target here, so
//! `weight_quantizer()` always names the standard converter and Spark's draft
//! copy does not carry Enntity's corrected-rounding guarantee.
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServingMode {
    Strict,
    Spark,
}

pub const REQUIRED_FLAGS: &[(&str, &str)] = &[
    ("ATLAS_QWEN4EXP_NO_PLE", "0"),
    ("ATLAS_MTP_DRAFTER_QSA", "0"),
    ("ATLAS_QSA_DISABLE", "0"),
    ("ATLAS_QWEN4EXP_BF16_GDN", "1"),
    ("ATLAS_QWEN4EXP_FP8_GDN", "0"),
    ("ATLAS_QWEN4EXP_PREFILL_MOE", "1"),
    ("ATLAS_QWEN4EXP_PREFILL_MOE_BF16", "0"),
    ("ATLAS_QWEN4EXP_PREFILL_BF16_PROJ", "0"),
];

impl ServingMode {
    pub fn parse(value: Option<&str>) -> Result<Option<Self>> {
        match value {
            None => Ok(None),
            Some("strict") => Ok(Some(Self::Strict)),
            Some("spark") => Ok(Some(Self::Spark)),
            Some(value) => {
                bail!("ATLAS_QWEN4EXP_SERVING_MODE must be strict or spark, got {value:?}")
            }
        }
    }

    pub fn validate_flags(self, lookup: impl Fn(&str) -> Option<String>) -> Result<()> {
        for &(key, expected) in REQUIRED_FLAGS {
            ensure!(
                lookup(key).as_deref() == Some(expected),
                "{self:?} requires {key}={expected}"
            );
        }
        for (key, expected) in self.drafter_flags() {
            ensure!(
                lookup(key).as_deref() == Some(expected),
                "{self:?} requires {key}={expected}"
            );
        }
        Ok(())
    }

    /// Strict keeps the BF16 draft head (the only attention path in this
    /// stack — Enntity's separate `ATLAS_QWEN4EXP_MTP_BF16_ATTN` switch is
    /// absent here); Spark selects the NVFP4 draft copy.
    pub fn drafter_flags(self) -> [(&'static str, &'static str); 1] {
        match self {
            Self::Strict => [("ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4", "0")],
            Self::Spark => [("ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4", "1")],
        }
    }

    pub fn validate_config(self, config: &ModelConfig) -> Result<()> {
        ensure!(
            config.model_type == "qwen4_exp"
                && config.tp_world_size == 2
                && config.ep_world_size == 2
                && !config.expert_tp,
            "Qwen Strict/Spark require qwen4_exp with TP2/EP2"
        );
        ensure!(
            self != Self::Spark || !config.attn_proj_bf16,
            "Spark mode requires its reviewed NVFP4 draft attention policy"
        );
        ensure!(
            config.skip_lm_head_quantization() && !config.lm_head_fp8,
            "Qwen Strict/Spark require the checkpoint target LM head"
        );
        Ok(())
    }

    pub fn validate_speculation(self, mtp: bool, alternate: bool) -> Result<()> {
        ensure!(
            !alternate,
            "Qwen Strict/Spark do not support alternative speculative lanes"
        );
        ensure!(
            mtp,
            "Qwen modes require native MTP; use disable_mtp for serial requests"
        );
        Ok(())
    }
}

pub fn selected() -> Result<Option<ServingMode>> {
    static MODE: std::sync::OnceLock<std::result::Result<Option<ServingMode>, String>> =
        std::sync::OnceLock::new();
    MODE.get_or_init(|| {
        ServingMode::parse(std::env::var("ATLAS_QWEN4EXP_SERVING_MODE").ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Draft-weight quantizer name. Enntity selects the corrected-RNE converter
/// (`quantize_bf16_to_nvfp4_weight_rne`) under a mode; that kernel is not in
/// this stack, so both modes use the standard converter — Spark's draft copy
/// therefore does NOT carry their rounding guarantee.
pub fn weight_quantizer() -> Result<&'static str> {
    let _ = selected()?;
    Ok("quantize_bf16_to_nvfp4")
}

pub fn validate(config: &ModelConfig, mtp: bool, alternate: bool) -> Result<()> {
    if let Some(mode) = selected()? {
        mode.validate_flags(|key| std::env::var(key).ok())?;
        mode.validate_config(config)?;
        mode.validate_speculation(mtp, alternate)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mode_names_and_precision_contract_are_closed() {
        assert_eq!(ServingMode::parse(None).unwrap(), None);
        for value in ["", "fast", "STRICT"] {
            assert!(ServingMode::parse(Some(value)).is_err());
        }
        for (name, mode) in [
            ("strict", ServingMode::Strict),
            ("spark", ServingMode::Spark),
        ] {
            assert_eq!(ServingMode::parse(Some(name)).unwrap(), Some(mode));
            let flags: Vec<_> = REQUIRED_FLAGS
                .iter()
                .copied()
                .chain(mode.drafter_flags())
                .collect();
            let lookup = |key: &str| {
                flags
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            };
            mode.validate_flags(lookup).unwrap();
            for &(changed, _) in &flags {
                assert!(
                    mode.validate_flags(|key| if key == changed { None } else { lookup(key) })
                        .is_err()
                );
                assert!(
                    mode.validate_flags(|key| if key == changed {
                        Some("invalid".into())
                    } else {
                        lookup(key)
                    })
                    .is_err()
                );
            }
        }
    }
    #[test]
    fn both_modes_support_mtp() {
        for mode in [ServingMode::Strict, ServingMode::Spark] {
            for mtp in [false, true] {
                assert_eq!(mode.validate_speculation(mtp, false).is_ok(), mtp);
                assert!(mode.validate_speculation(mtp, true).is_err());
            }
        }
    }
}
