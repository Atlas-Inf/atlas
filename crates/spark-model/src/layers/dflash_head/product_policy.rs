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
        // Presence semantics for the graph kill switches: the pre-freeze
        // runtime conditions suppressed graphs whenever the variable was
        // present (any value), so admission must treat a present-but-not-1
        // value as ineligible rather than silently graph-eligible.
        let proposal_graph_eligible = !present(&mut read, "ATLAS_DFLASH_PROPOSE_NO_GRAPH")
            && !present(&mut read, "ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !present(&mut read, "ATLAS_DEBUG_NO_GRAPH")
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
        let target_verify_graph_eligible = !present(&mut read, "ATLAS_DFLASH_DEBUG_NO_GRAPH")
            && !present(&mut read, "ATLAS_DEBUG_NO_GRAPH")
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

/// Startup-static execution values consumed by the draft head hot paths.
///
/// Resolved exactly once at head construction and frozen on the head:
/// `propose`, `forward_block`, and lane construction must read these fields
/// instead of re-reading the process environment. The official Lightning
/// product derives every field from the validated policy toggles
/// ([`DsparkStartupExecution::from_lightning`]); generic DFlash keeps the
/// legacy lenient environment semantics but still parses them once here
/// ([`DsparkStartupExecution::from_env_lenient`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkStartupExecution {
    /// Option B paged-context drafter path is active.
    pub option_b_enabled: bool,
    /// Number of total propose lanes (lane 0 is the default stream).
    pub proposal_lane_count: usize,
    /// Diagnostic draft-depth cap override; `None` keeps scheduler K.
    pub draft_cap_override: Option<usize>,
    /// Option B no-context ablation forces zero drafter context.
    pub option_b_no_ctx: bool,
    /// Per-forward debug dumps are active (graph-ineligible).
    pub debug_dump: bool,
    /// Any graph-ineligible diagnostic environment was set at startup.
    pub graph_ineligible_diags: bool,
    /// Frozen per-step diagnostic switches. Product heads carry the
    /// all-off set; generic heads keep legacy lenient semantics, parsed
    /// once. Hot paths must read these instead of the environment.
    pub diagnostics: DsparkDiagnostics,
}

/// Startup-static diagnostic switches for the draft head hot paths.
///
/// Every field defaults to off; each mirrors exactly one legacy
/// environment probe (value semantics preserved: `=1` booleans, parsed
/// integers with the legacy default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkDiagnostics {
    /// `ATLAS_DFLASH_DEBUG_CTX_OFF=1`.
    pub debug_ctx_off: bool,
    /// `ATLAS_DFLASH_DEBUG_CTX_USED=<usize>`.
    pub debug_ctx_used: Option<usize>,
    /// `ATLAS_DFLASH_PRECOMPUTE=1`.
    pub precompute_probe: bool,
    /// `ATLAS_DFLASH_PRECOMPUTE_COMMIT=1`.
    pub precompute_commit: bool,
    /// `ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1`.
    pub force_pattern: bool,
    /// `ATLAS_DFLASH_DEBUG_DUMP_FULL=1`.
    pub dump_full: bool,
    /// `ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1`.
    pub force_noise_pattern: bool,
    /// `ATLAS_DFLASH_PROPOSE_WARMUP_N=<usize>` (legacy default 2).
    pub propose_warmup_n: usize,
    /// `ATLAS_DFLASH_BLOCK_DUMP=1`.
    pub block_dump: bool,
    /// `ATLAS_DFLASH_BLOCK_DUMP_AT_POS=<usize>` (legacy default 0).
    pub block_dump_at_pos: usize,
    /// `ATLAS_DFLASH_LOG_DRAFTS=1`.
    pub log_drafts: bool,
    /// `ATLAS_DFLASH_CTX_PARITY_DUMP=1`.
    pub ctx_parity_dump: bool,
    /// `ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND=1`.
    pub no_decode_append: bool,
    /// `ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE=1`.
    pub full_precompute: bool,
    /// `ATLAS_DFLASH_CTXLEN_PROBE=1`.
    pub ctxlen_probe: bool,
    /// `ATLAS_DFLASH_VERIFY_TRACE=1`.
    pub verify_trace: bool,
    /// `ATLAS_DFLASH_PRECOMPUTE_DUMP=1`.
    pub precompute_dump: bool,
    /// `ATLAS_DFLASH_OPTION_B_DIAG=1`.
    pub option_b_diag: bool,
}

impl Default for DsparkDiagnostics {
    fn default() -> Self {
        Self {
            debug_ctx_off: false,
            debug_ctx_used: None,
            precompute_probe: false,
            precompute_commit: false,
            force_pattern: false,
            dump_full: false,
            force_noise_pattern: false,
            propose_warmup_n: 2,
            block_dump: false,
            block_dump_at_pos: 0,
            log_drafts: false,
            ctx_parity_dump: false,
            no_decode_append: false,
            full_precompute: false,
            ctxlen_probe: false,
            verify_trace: false,
            precompute_dump: false,
            option_b_diag: false,
        }
    }
}

impl DsparkDiagnostics {
    /// Legacy lenient parse of every hot-path diagnostic probe, executed
    /// exactly once. Malformed integers keep their legacy defaults.
    pub fn from_env_lenient() -> Self {
        fn one(name: &str) -> bool {
            std::env::var(name).ok().as_deref() == Some("1")
        }
        fn num(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(default)
        }
        Self {
            debug_ctx_off: one("ATLAS_DFLASH_DEBUG_CTX_OFF"),
            debug_ctx_used: std::env::var("ATLAS_DFLASH_DEBUG_CTX_USED")
                .ok()
                .and_then(|raw| raw.parse().ok()),
            precompute_probe: one("ATLAS_DFLASH_PRECOMPUTE"),
            precompute_commit: one("ATLAS_DFLASH_PRECOMPUTE_COMMIT"),
            force_pattern: one("ATLAS_DFLASH_DEBUG_FORCE_PATTERN"),
            dump_full: one("ATLAS_DFLASH_DEBUG_DUMP_FULL"),
            force_noise_pattern: one("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN"),
            propose_warmup_n: num("ATLAS_DFLASH_PROPOSE_WARMUP_N", 2),
            block_dump: one("ATLAS_DFLASH_BLOCK_DUMP"),
            block_dump_at_pos: num("ATLAS_DFLASH_BLOCK_DUMP_AT_POS", 0),
            log_drafts: one("ATLAS_DFLASH_LOG_DRAFTS"),
            ctx_parity_dump: one("ATLAS_DFLASH_CTX_PARITY_DUMP"),
            no_decode_append: one("ATLAS_DFLASH_DEBUG_NO_DECODE_APPEND"),
            full_precompute: one("ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE"),
            ctxlen_probe: one("ATLAS_DFLASH_CTXLEN_PROBE"),
            verify_trace: one("ATLAS_DFLASH_VERIFY_TRACE"),
            precompute_dump: one("ATLAS_DFLASH_PRECOMPUTE_DUMP"),
            option_b_diag: one("ATLAS_DFLASH_OPTION_B_DIAG"),
        }
    }
}

impl DsparkStartupExecution {
    /// Frozen product execution derived only from validated Lightning
    /// toggles. `validate()` guarantees Option B on, exactly one lane, no
    /// draft cap, and no diagnostic suppression, so the product never
    /// consults the environment again.
    pub fn from_lightning(toggles: LightningDsparkRuntimeToggles) -> Self {
        Self {
            option_b_enabled: toggles.option_b_enabled,
            proposal_lane_count: toggles.proposal_lane_count,
            draft_cap_override: toggles.draft_cap_override,
            option_b_no_ctx: false,
            debug_dump: false,
            graph_ineligible_diags: !(toggles.proposal_graph_eligible
                && toggles.target_verify_graph_eligible),
            diagnostics: DsparkDiagnostics::default(),
        }
    }

    /// Generic-DFlash execution: legacy lenient semantics (malformed values
    /// fall back to defaults), parsed exactly once instead of per step.
    pub fn from_env_lenient() -> Self {
        fn one(name: &str) -> bool {
            std::env::var(name).ok().as_deref() == Some("1")
        }
        fn present(name: &str) -> bool {
            std::env::var_os(name).is_some()
        }
        let graph_ineligible_diags = [
            "ATLAS_DFLASH_PROPOSE_NO_GRAPH",
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
        ]
        .iter()
        .any(|name| present(name));
        Self {
            option_b_enabled: one("ATLAS_DFLASH_OPTION_B"),
            proposal_lane_count: std::env::var("ATLAS_DFLASH_PROPOSE_LANES")
                .ok()
                .and_then(|raw| raw.parse().ok())
                .unwrap_or(1)
                .max(1),
            draft_cap_override: std::env::var("ATLAS_DFLASH_DRAFT_CAP")
                .ok()
                .and_then(|raw| raw.parse().ok()),
            option_b_no_ctx: one("ATLAS_DFLASH_OPTION_B_NO_CTX"),
            debug_dump: one("ATLAS_DFLASH_DEBUG_DUMP"),
            graph_ineligible_diags,
            diagnostics: DsparkDiagnostics::from_env_lenient(),
        }
    }
}

/// Structural (non-environment) preconditions the constructed target must
/// satisfy before an admitted Lightning policy may claim product graph
/// eligibility. Pure data so it is unit-testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightningStructuralGraphState {
    /// Target `suppress_graphs` is set (FP8 calibration or diagnostics).
    pub target_suppress_graphs: bool,
    /// A LoRA adapter set is installed (eager decode requirement).
    pub lora_installed: bool,
    /// A communicator/process-group topology is active beyond TP=1/EP=1.
    pub distributed_topology: bool,
}

impl LightningStructuralGraphState {
    /// An admitted policy may only claim graph eligibility when every
    /// structural condition allows CUDA-graph execution.
    pub fn graphs_allowed(&self) -> bool {
        !self.target_suppress_graphs && !self.lora_installed && !self.distributed_topology
    }
}
