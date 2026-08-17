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
    pub fn from_env() -> Result<Self, LightningDsparkPolicyError> {
        Self::from_reader(|name| std::env::var(name).ok())
    }

    /// Parse presence-aware startup inputs without touching process-global env.
    pub fn from_reader<F>(mut read: F) -> Result<Self, LightningDsparkPolicyError>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let proposal_lane_count = match read("ATLAS_DFLASH_PROPOSE_LANES") {
            None => 1,
            Some(raw) => {
                let parsed = raw.parse::<usize>().map_err(|_| {
                    RuntimeToggleError::new(
                        "proposal_lane_count",
                        format!("invalid lane count {raw:?}"),
                    )
                })?;
                if parsed == 0 {
                    return Err(RuntimeToggleError::new(
                        "proposal_lane_count",
                        "lane count must be greater than zero",
                    ));
                }
                parsed
            }
        };
        let draft_cap_override = read("ATLAS_DFLASH_DRAFT_CAP")
            .map(|raw| {
                raw.parse::<usize>().map_err(|_| {
                    RuntimeToggleError::new(
                        "draft_cap_override",
                        format!("invalid draft cap {raw:?}"),
                    )
                })
            })
            .transpose()?;

        if present(&mut read, "ATLAS_DFLASH_OPTION_B_NO_CTX") {
            return Err(RuntimeToggleError::new(
                "option_b_no_ctx",
                "Option B no-context mode is not supported for the product",
            ));
        }
        let gemma4_diag = one_or_true(&mut read, "ATLAS_DIAG_GEMMA4");
        let proposal_graph_eligible = !present(&mut read, "ATLAS_DFLASH_PROPOSE_NO_GRAPH")
            && !one(&mut read, "ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !one(&mut read, "ATLAS_DEBUG_NO_GRAPH")
            && !gemma4_diag
            && !present_any(
                &mut read,
                &[
                    "ATLAS_DFLASH_DEBUG_DUMP_FULL",
                    "ATLAS_DFLASH_DEBUG_DUMP",
                    "ATLAS_DFLASH_OPTION_B_DIAG",
                    "ATLAS_DFLASH_PRECOMPUTE_DUMP",
                    "ATLAS_DFLASH_VERIFY_TRACE",
                    "ATLAS_DFLASH_LOG_DRAFTS",
                    "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
                    "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
                    "ATLAS_DFLASH_DEBUG_CTX_OFF",
                    "ATLAS_DFLASH_DEBUG_CTX_USED",
                    "ATLAS_DFLASH_BLOCK_DUMP",
                ],
            );
        let target_verify_graph_eligible = !one(&mut read, "ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !one(&mut read, "ATLAS_DEBUG_NO_GRAPH")
            && !gemma4_diag
            && !one(&mut read, "ATLAS_DFLASH_VERIFY_COMPUTE_SERIAL");

        Ok(Self {
            option_b_enabled: one(&mut read, "ATLAS_DFLASH_OPTION_B"),
            proposal_lane_count,
            proposal_graph_eligible,
            target_verify_graph_eligible,
            batched_verify_enabled: !present(&mut read, "ATLAS_NO_DFLASH_BATCH_VERIFY"),
            seam_serial_enabled: one(&mut read, "ATLAS_DFLASH_SEAM_SERIAL"),
            draft_cap_override,
            adaptive_enabled: one(&mut read, "ATLAS_DFLASH_ADAPTIVE"),
        })
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

fn present<F>(read: &mut F, name: &str) -> bool
where
    F: FnMut(&str) -> Option<String>,
{
    read(name).is_some()
}

fn present_any<F>(read: &mut F, names: &[&str]) -> bool
where
    F: FnMut(&str) -> Option<String>,
{
    names.iter().any(|name| present(read, name))
}

fn one<F>(read: &mut F, name: &str) -> bool
where
    F: FnMut(&str) -> Option<String>,
{
    read(name).as_deref() == Some("1")
}

fn one_or_true<F>(read: &mut F, name: &str) -> bool
where
    F: FnMut(&str) -> Option<String>,
{
    matches!(read(name).as_deref(), Some("1" | "true"))
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

/// Executable model-identity latch shared by TransformerModel setters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LightningDsparkIdentityLatch {
    policy: Option<LightningDsparkProductPolicy>,
}

impl LightningDsparkIdentityLatch {
    pub fn policy(&self) -> Option<&LightningDsparkProductPolicy> {
        self.policy.as_ref()
    }

    pub fn install_lightning(&mut self, policy: LightningDsparkProductPolicy) {
        self.policy = Some(policy);
    }

    pub fn install_generic(&mut self) {
        self.policy = None;
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
