// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    CompatibilityRule, GraphFingerprint, GraphMode, GraphPhase, GraphPrewarmProfile, ShapeBucket,
    SpeculativeAlgorithm,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhasePolicy {
    pub max_entries: usize,
    pub max_estimated_bytes: u64,
    pub capture_enabled: bool,
    pub replay_enabled: bool,
    pub prewarm_enabled: bool,
}

impl PhasePolicy {
    pub fn validate(self, phase: GraphPhase) -> Result<(), String> {
        if (self.capture_enabled || self.replay_enabled || self.prewarm_enabled)
            && (self.max_entries == 0 || self.max_estimated_bytes == 0)
        {
            return Err(format!(
                "{phase} graph policy enables execution with a zero cache quota"
            ));
        }
        if self.prewarm_enabled && !self.capture_enabled {
            return Err(format!(
                "{phase} graph policy enables prewarm while capture is disabled"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphPolicies {
    phases: [PhasePolicy; GraphPhase::COUNT],
}

impl GraphPolicies {
    pub fn new(
        prefill: PhasePolicy,
        decode: PhasePolicy,
        verify: PhasePolicy,
        propose: PhasePolicy,
        fused: PhasePolicy,
    ) -> Result<Self, String> {
        let phases = [prefill, decode, verify, propose, fused];
        for phase in [
            GraphPhase::Prefill,
            GraphPhase::Decode,
            GraphPhase::Verify,
            GraphPhase::Propose,
            GraphPhase::Fused,
        ] {
            phases[phase.index()].validate(phase)?;
        }
        Ok(Self { phases })
    }

    pub fn phase(&self, phase: GraphPhase) -> PhasePolicy {
        self.phases[phase.index()]
    }
}

#[derive(Clone, Debug)]
pub struct GraphRuntimeConfig {
    pub mode: GraphMode,
    pub speculative_algorithm: SpeculativeAlgorithm,
    pub shape_buckets: Vec<ShapeBucket>,
    pub policies: GraphPolicies,
    pub max_cache_entries: usize,
    pub max_cache_bytes: u64,
    pub fingerprint: GraphFingerprint,
    pub resource_generation: u64,
    pub compatibility_rules: Vec<CompatibilityRule>,
    pub export_dir: Option<std::path::PathBuf>,
    pub prewarm_profile: Option<GraphPrewarmProfile>,
}

impl GraphRuntimeConfig {
    pub fn disabled(fingerprint: GraphFingerprint) -> Self {
        let off = PhasePolicy {
            max_entries: 0,
            max_estimated_bytes: 0,
            capture_enabled: false,
            replay_enabled: false,
            prewarm_enabled: false,
        };
        Self {
            mode: GraphMode::Disabled,
            speculative_algorithm: SpeculativeAlgorithm::None,
            shape_buckets: Vec::new(),
            policies: GraphPolicies::new(off, off, off, off, off)
                .expect("fully-disabled graph policies are valid"),
            max_cache_entries: 0,
            max_cache_bytes: 0,
            fingerprint,
            resource_generation: 1,
            compatibility_rules: Vec::new(),
            export_dir: None,
            prewarm_profile: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.mode != GraphMode::Disabled
            && (self.shape_buckets.is_empty()
                || self.max_cache_entries == 0
                || self.max_cache_bytes == 0)
        {
            return Err(
                "an enabled graph runtime requires shape buckets and non-zero global cache limits"
                    .to_string(),
            );
        }
        for bucket in &self.shape_buckets {
            if bucket.token_limit == 0 || bucket.request_limit == 0 {
                return Err("CUDA graph bucket limits must be greater than zero".to_string());
            }
        }
        if self.resource_generation == 0 {
            return Err("CUDA graph resource generation must be greater than zero".to_string());
        }
        Ok(())
    }
}
