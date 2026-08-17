// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-static identity for the official Lightning DSpark product.
//!
//! The toggle reader is the only environment boundary in this module. Tests
//! construct [`LightningDsparkRuntimeToggles`] directly, so validation is pure
//! and never mutates or depends on process-global environment state.

use std::fmt;

use super::contract::{LightningDsparkContractError, LightningDsparkProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightningDsparkRuntimeToggles {
    pub option_b_enabled: bool,
    pub proposal_lane_count: usize,
    pub proposal_graph_eligible: bool,
    pub target_verify_graph_eligible: bool,
    pub batched_verify_enabled: bool,
    pub seam_serial_enabled: bool,
    pub draft_cap_override: Option<usize>,
    pub adaptive_enabled: bool,
}

impl LightningDsparkRuntimeToggles {
    /// Read the startup environment once for product admission.
    ///
    /// The value/presence rules mirror the existing DFlash sites: `=1` for
    /// positive opt-ins, presence for negative kill switches, and parsed
    /// defaults for lane count and draft-cap overrides.
    pub fn from_env() -> Self {
        let proposal_graph_eligible = !present("ATLAS_DFLASH_PROPOSE_NO_GRAPH")
            && !one("ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !one("ATLAS_DEBUG_NO_GRAPH")
            && !one("ATLAS_DIAG_GEMMA4")
            && !present_any(&[
                "ATLAS_DFLASH_DEBUG_DUMP_FULL",
                "ATLAS_DFLASH_OPTION_B_DIAG",
                "ATLAS_DFLASH_PRECOMPUTE_DUMP",
                "ATLAS_DFLASH_VERIFY_TRACE",
                "ATLAS_DFLASH_LOG_DRAFTS",
                "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
                "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
                "ATLAS_DFLASH_DEBUG_CTX_OFF",
                "ATLAS_DFLASH_DEBUG_CTX_USED",
                "ATLAS_DFLASH_BLOCK_DUMP",
            ]);
        let target_verify_graph_eligible = !one("ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !one("ATLAS_DEBUG_NO_GRAPH")
            && !one("ATLAS_DIAG_GEMMA4")
            && !one("ATLAS_DFLASH_VERIFY_COMPUTE_SERIAL");
        Self {
            option_b_enabled: one("ATLAS_DFLASH_OPTION_B"),
            proposal_lane_count: value("ATLAS_DFLASH_PROPOSE_LANES")
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(1)
                .max(1),
            proposal_graph_eligible,
            target_verify_graph_eligible,
            batched_verify_enabled: !present("ATLAS_NO_DFLASH_BATCH_VERIFY"),
            seam_serial_enabled: one("ATLAS_DFLASH_SEAM_SERIAL"),
            draft_cap_override: value("ATLAS_DFLASH_DRAFT_CAP").and_then(|raw| raw.parse().ok()),
            adaptive_enabled: one("ATLAS_DFLASH_ADAPTIVE"),
        }
    }

    pub fn validate(&self) -> Result<(), LightningDsparkPolicyError> {
        if !self.option_b_enabled {
            return Err(RuntimeToggleError::new(
                "option_b_enabled",
                "Option B is required",
            ));
        }
        if self.proposal_lane_count != 1 {
            return Err(RuntimeToggleError::new(
                "proposal_lane_count",
                format!(
                    "expected exactly one lane, found {}",
                    self.proposal_lane_count
                ),
            ));
        }
        if !self.proposal_graph_eligible {
            return Err(RuntimeToggleError::new(
                "proposal_graph_eligible",
                "proposal graph suppression or diagnostic is enabled",
            ));
        }
        if !self.target_verify_graph_eligible {
            return Err(RuntimeToggleError::new(
                "target_verify_graph_eligible",
                "target verify graph suppression or serial diagnostic is enabled",
            ));
        }
        if !self.batched_verify_enabled {
            return Err(RuntimeToggleError::new(
                "batched_verify_enabled",
                "batched verify is disabled",
            ));
        }
        if self.seam_serial_enabled {
            return Err(RuntimeToggleError::new(
                "seam_serial_enabled",
                "seam-serial diagnostic is enabled",
            ));
        }
        if let Some(cap) = self.draft_cap_override {
            return Err(RuntimeToggleError::new(
                "draft_cap_override",
                format!("arbitrary draft cap {cap} is set"),
            ));
        }
        if self.adaptive_enabled {
            return Err(RuntimeToggleError::new(
                "adaptive_enabled",
                "adaptive product behavior is enabled",
            ));
        }
        Ok(())
    }
}

fn value(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn present_any(names: &[&str]) -> bool {
    names.iter().any(|name| present(name))
}

fn one(name: &str) -> bool {
    value(name).as_deref() == Some("1")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightningDsparkProductPolicy {
    profile: LightningDsparkProfile,
    runtime_toggles: LightningDsparkRuntimeToggles,
}

impl LightningDsparkProductPolicy {
    pub fn try_new(
        profile: LightningDsparkProfile,
        runtime_toggles: LightningDsparkRuntimeToggles,
    ) -> Result<Self, LightningDsparkPolicyError> {
        profile
            .validate()
            .map_err(LightningDsparkPolicyError::Profile)?;
        runtime_toggles.validate()?;
        Ok(Self {
            profile,
            runtime_toggles,
        })
    }

    pub fn profile(&self) -> &LightningDsparkProfile {
        &self.profile
    }

    pub fn runtime_toggles(&self) -> LightningDsparkRuntimeToggles {
        self.runtime_toggles
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightningDsparkPolicyError {
    Profile(LightningDsparkContractError),
    RuntimeToggle { field: &'static str, detail: String },
}

type RuntimeToggleError = LightningDsparkPolicyError;

impl RuntimeToggleError {
    fn new(field: &'static str, detail: impl Into<String>) -> Self {
        Self::RuntimeToggle {
            field,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for LightningDsparkPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(error) => write!(formatter, "Lightning DSpark profile rejected: {error}"),
            Self::RuntimeToggle { field, detail } => {
                write!(
                    formatter,
                    "Lightning DSpark runtime policy rejected {field}: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for LightningDsparkPolicyError {}
