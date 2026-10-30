// SPDX-License-Identifier: AGPL-3.0-only
//! GLM TP2/TP4 serving modes (`ATLAS_GLM_SERVING_MODE=strict|spark`). Other
//! model launches, and GLM launches without the variable, keep their behavior.
//!
//! Ported from Enntity/atlas `layers/glm_native/mode.rs` and
//! `layers/glm_native/topology.rs` (f4d95a10, AGPL-3.0-only), with the
//! launcher's precision contract from SparkGLM 0.2.0 (`serve.py
//! serving_mode`) checked here, as `qwen4_exp::mode` does for Qwen:
//! - Strict decodes serially on the canonical one-row target. Requests never
//!   dispatch speculation; the drafter stays loaded on its BF16 weights.
//! - Spark speculates with its reviewed NVFP4 drafter and blends the shared
//!   expert's full output (no shared TP split).
//!
//! Both modes serve the FP4-Local native formats; Strict also decodes KDA on
//! MXFP8 and the routed experts at A16.
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServingMode {
    Strict,
    Spark,
}

/// Strict's drafter policy: BF16 drafter, every reduced-precision drafter
/// path and the drafter's pair split off (SparkGLM 0.2.0 `serve.py`).
const STRICT_FLAGS: &[(&str, &str)] = &[
    ("ATLAS_NO_DFLASH_DRAFTER_NVFP4", "1"),
    ("ATLAS_HC_PERSEQ_DECODE", "1"),
    ("ATLAS_GLM_CANONICAL_VERIFY", "1"),
    ("ATLAS_GLM_DFLASH", "1"),
    ("ATLAS_DFLASH_DRAFTER_FP8", "0"),
    ("ATLAS_DFLASH_NVFP4_TC", "0"),
    ("ATLAS_DFLASH_NVFP4_HEAD", "0"),
    ("ATLAS_DFLASH_MXFP8", "0"),
    ("ATLAS_DFLASH_MXFP8_HEAD", "0"),
    ("ATLAS_DFLASH_DROP_BF16", "0"),
    ("ATLAS_DFLASH_CTX_NVFP4", "0"),
    ("ATLAS_DFLASH_CONV_TWIN", "0"),
    ("ATLAS_GLM_DRAFT_TP", "0"),
    ("ATLAS_GLM_DRAFT_TP_BATCH", "0"),
    ("ATLAS_GLM_DRAFT_TP_CTX", "0"),
    ("ATLAS_GLM_DRAFT_TP_TOPK", "0"),
];

/// The FP4-Local native target both modes serve.
const NATIVE_FLAGS: &[(&str, &str)] = &[
    ("ATLAS_GLM_FP4L_NATIVE", "1"),
    ("ATLAS_GLM_FP4L_MLA_FP8", "1"),
    ("ATLAS_GLM_FP4L_DENSE_MXFP8", "1"),
    ("ATLAS_GLM_FP4L_KDA_W8A8_PREFILL", "1"),
    ("ATLAS_GLM_FP4L_INPUT_SCALES", "1"),
    ("ATLAS_GLM_FP4L_FP8_PREFILL", "w8a8"),
    ("ATLAS_GLM_LM_HEAD_MXFP8", "1"),
    ("ATLAS_GLM_FP4L_KDA_TILED_TWIN", "0"),
];

/// Strict's target: native shared expert, MXFP8 KDA decode, A16 routed decode.
const STRICT_NATIVE_FLAGS: &[(&str, &str)] = &[
    ("ATLAS_GLM_FP4L_SHARED_FP8", "1"),
    ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "1"),
    ("ATLAS_GLM_FP4L_DECODE_A16", "1"),
];

/// Splitting the shared expert rounds each partial BF16 output before the
/// routed/shared blend; Spark's matched arithmetic needs the full output.
/// Its shared expert format comes from the profile.
const SPARK_FLAGS: &[(&str, &str)] = &[
    ("ATLAS_GLM_SHARED_TP_SPLIT", "0"),
    ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "0"),
    ("ATLAS_GLM_FP4L_DECODE_A16", "0"),
];

impl ServingMode {
    pub fn parse(value: Option<&str>) -> Result<Option<Self>> {
        match value {
            None => Ok(None),
            Some("strict") => Ok(Some(Self::Strict)),
            Some("spark") => Ok(Some(Self::Spark)),
            Some(value) => bail!("ATLAS_GLM_SERVING_MODE must be strict or spark, got {value:?}"),
        }
    }

    /// The launcher's flag contract for this mode, read through `lookup`
    /// (an environment lookup). `native_mtp`: Strict's opt-in native
    /// predictor replaces the DFlash lane, so the drafter policy is moot;
    /// the target's still holds.
    pub fn validate_flags(
        self,
        native_mtp: bool,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<()> {
        ensure!(
            !native_mtp || self == Self::Strict,
            "ATLAS_GLM_STRICT_NATIVE_MTP requires ATLAS_GLM_SERVING_MODE=strict"
        );
        let flags: &[&[(&str, &str)]] = match self {
            Self::Strict if native_mtp => &[NATIVE_FLAGS, STRICT_NATIVE_FLAGS],
            Self::Strict => &[NATIVE_FLAGS, STRICT_NATIVE_FLAGS, STRICT_FLAGS],
            Self::Spark => {
                ensure!(
                    lookup("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_none(),
                    "Spark mode requires its reviewed NVFP4 drafter policy"
                );
                &[NATIVE_FLAGS, SPARK_FLAGS]
            }
        };
        for &(key, expected) in flags.iter().copied().flatten() {
            ensure!(
                lookup(key).as_deref() == Some(expected),
                "{self:?} mode requires {key}={expected}"
            );
        }
        Ok(())
    }

    /// Strict is serial target decoding: no alternative speculative lane.
    pub fn validate_speculation(self, alternate: bool) -> Result<()> {
        ensure!(
            self != Self::Strict || !alternate,
            "Strict mode requires serial target decoding"
        );
        Ok(())
    }
}

/// The target overlaps tensor and expert ranks on a pair or four Sparks.
pub fn supported_parallelism(world: usize, tp: usize, ep: usize) -> bool {
    matches!(world, 2 | 4) && tp == world && ep == world
}

pub fn selected() -> Result<Option<ServingMode>> {
    static MODE: std::sync::OnceLock<std::result::Result<Option<ServingMode>, String>> =
        std::sync::OnceLock::new();
    MODE.get_or_init(|| {
        ServingMode::parse(std::env::var("ATLAS_GLM_SERVING_MODE").ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// This includes Spark: unapproved fallback requantization is forbidden in
/// both modes.
pub fn require_format(present: bool, description: &str) -> Result<()> {
    require_format_for(selected()?, present, description)
}

fn require_format_for(mode: Option<ServingMode>, present: bool, description: &str) -> Result<()> {
    ensure!(
        mode.is_none() || present,
        "GLM serving mode requires {description}; refusing fallback weight conversion"
    );
    Ok(())
}

/// A calibrated expert is either local with all scales, or an empty EP
/// placeholder.
pub fn validate_calibration(required: bool, live: [bool; 3], scaled: [bool; 3]) -> Result<()> {
    ensure!(
        !required || live == [false; 3] || (live == [true; 3] && scaled == [true; 3]),
        "GLM serving mode requires gate/up/down input scales on every live expert"
    );
    Ok(())
}

/// Native MTP remains an explicit, bounded Strict option.
pub fn strict_native_mtp() -> bool {
    std::env::var("ATLAS_GLM_STRICT_NATIVE_MTP").as_deref() == Ok("1")
}

/// Strict without native MTP: every request decodes on the one-row target.
pub fn strict_serial() -> bool {
    strict_serial_for(selected().ok().flatten(), strict_native_mtp())
}

fn strict_serial_for(mode: Option<ServingMode>, native_mtp: bool) -> bool {
    mode == Some(ServingMode::Strict) && !native_mtp
}

/// Drafter weight conversions take the RNE weight entry under a mode;
/// activation quantizers keep their own.
pub fn weight_quantizer() -> Result<&'static str> {
    Ok(weight_quantizer_for(selected()?))
}

fn weight_quantizer_for(mode: Option<ServingMode>) -> &'static str {
    if mode.is_some() {
        "quantize_bf16_to_nvfp4_weight_rne"
    } else {
        "quantize_bf16_to_nvfp4"
    }
}

/// The whole contract for this process: a no-op without a mode.
/// `world`, `tp`, `ep`: the launch's parallel sizes; `alternate`: an
/// n-gram or self-speculative lane is configured.
pub fn validate(
    config: &ModelConfig,
    (world, tp, ep): (usize, usize, usize),
    alternate: bool,
) -> Result<()> {
    let Some(mode) = selected()? else {
        return Ok(());
    };
    ensure!(
        config.model_type == "glm5_next",
        "ATLAS_GLM_SERVING_MODE requires a glm5_next checkpoint, got {}",
        config.model_type
    );
    ensure!(
        supported_parallelism(world, tp, ep),
        "GLM serving modes require matching TP2/EP2 or TP4/EP4"
    );
    mode.validate_flags(strict_native_mtp(), |key| std::env::var(key).ok())?;
    mode.validate_speculation(alternate)
}

#[cfg(test)]
#[path = "mode_tests.rs"]
mod tests;
