// SPDX-License-Identifier: AGPL-3.0-only
//! Keep new CUDA selector capabilities out of the inherited native-HIP path.
use anyhow::{Result, ensure};

pub(super) fn validate_requested_features(draft_conf: bool) -> Result<()> {
    if cfg!(atlas_hip) {
        validate_confidence(
            draft_conf,
            std::env::var("ATLAS_DFLASH2_EDGE").ok().as_deref(),
        )?;
        for name in [
            "ATLAS_GLM_DRAFT_TP",
            "ATLAS_GLM_DRAFT_TP_BATCH",
            "ATLAS_GLM_DRAFT_TP_CTX",
            "ATLAS_GLM_DRAFT_TP_TOPK",
        ] {
            reject_split(name, std::env::var(name).ok().as_deref())?;
        }
    }
    Ok(())
}

fn validate_confidence(want: bool, edge: Option<&str>) -> Result<()> {
    ensure!(
        !want,
        "native HIP retains the upstream DFlash selector: ATLAS_DFLASH_CONF_WIDTH/LOG are CUDA-only"
    );
    ensure!(
        !matches!(edge.map(str::trim), Some("0.6" | "0.8")),
        "native HIP retains the upstream DFlash selector: ATLAS_DFLASH2_EDGE=0.6/0.8 is CUDA-only"
    );
    Ok(())
}

fn reject_split(name: &str, value: Option<&str>) -> Result<()> {
    ensure!(
        matches!(value.map(str::trim), None | Some("" | "0")),
        "native HIP retains the upstream DFlash selector: {name} is CUDA-only"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_hip_profile_is_allowed_but_new_confidence_is_explicitly_refused() {
        for edge in [None, Some(""), Some("1")] {
            validate_confidence(false, edge).unwrap();
        }
        assert!(validate_confidence(true, None).is_err());
        for edge in ["0.6", "0.8", " 0.6 "] {
            assert!(validate_confidence(false, Some(edge)).is_err());
        }
    }

    #[test]
    fn new_split_settings_are_refused_including_orphan_subsettings() {
        for name in [
            "ATLAS_GLM_DRAFT_TP",
            "ATLAS_GLM_DRAFT_TP_BATCH",
            "ATLAS_GLM_DRAFT_TP_CTX",
            "ATLAS_GLM_DRAFT_TP_TOPK",
        ] {
            for off in [None, Some(""), Some("0"), Some(" 0 ")] {
                reject_split(name, off).unwrap();
            }
            for on in ["1", "head", "mlp", "head,mlp", "typo"] {
                let error = reject_split(name, Some(on)).unwrap_err().to_string();
                assert!(error.contains(name));
                assert!(error.contains("CUDA-only"));
            }
        }
    }
}
