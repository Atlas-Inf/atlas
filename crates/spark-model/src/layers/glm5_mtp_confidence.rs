// SPDX-License-Identifier: AGPL-3.0-only

//! Cumulative-confidence stop: retain the current proposal and skip later
//! forwards once their accumulated probability falls below the threshold.
//! This affects proposal width only; target verification remains authoritative.
use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The stop threshold of strict native MTP (`ATLAS_GLM_STRICT_NATIVE_MTP`),
/// the one lane that builds this head in the shipped launches: the launcher
/// qualified it at 0.95. Any other `--speculative` GLM launch proposes its
/// whole width, as before.
const STRICT_STOP_PROB: f32 = 0.95;

pub(super) struct ConfidencePlan {
    pub argmax_batch_lp: Option<KernelHandle>,
    pub stop_ln_threshold: Option<f32>,
}
impl ConfidencePlan {
    /// The plan of a head on the strict native lane (`strict`) or another.
    pub(super) fn new(gpu: &dyn GpuBackend, strict: bool) -> Result<Self> {
        let threshold = strict.then_some(STRICT_STOP_PROB);
        Ok(Self {
            argmax_batch_lp: threshold
                .map(|_| gpu.kernel("argmax", "argmax_bf16_batch_lp"))
                .transpose()?,
            stop_ln_threshold: threshold.map(f32::ln),
        })
    }
}
#[derive(Default)]
pub(super) struct ConfidenceState {
    cumulative_lp: f32,
    stop_threshold: Option<f32>,
}
impl ConfidenceState {
    pub(super) fn begin(&mut self, threshold: Option<f32>) {
        self.cumulative_lp = 0.0;
        self.stop_threshold = threshold;
    }
    pub(super) fn record(&mut self, logprob: f32) -> Result<()> {
        ensure!(
            logprob.is_finite() && logprob <= 1e-4,
            "GLM-5 MTP confidence received invalid top-1 logprob {logprob:?}"
        );
        self.cumulative_lp += logprob;
        Ok(())
    }
    pub(super) fn should_stop(&self) -> bool {
        self.stop_threshold
            .is_some_and(|threshold| self.cumulative_lp < threshold)
    }
    pub(super) fn clear(&mut self) {
        self.begin(None);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_the_strict_native_lane_stops_at_its_threshold() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let strict = ConfidencePlan::new(&gpu, true).unwrap();
        assert_eq!(strict.stop_ln_threshold, Some(0.95f32.ln()));
        assert!(strict.argmax_batch_lp.is_some());
        let other = ConfidencePlan::new(&gpu, false).unwrap();
        assert_eq!(other.stop_ln_threshold, None);
        assert!(other.argmax_batch_lp.is_none());
    }
    #[test]
    fn stop_uses_cumulative_probability_and_resets_between_proposals() {
        let mut state = ConfidenceState::default();
        state.begin(Some(-3.0));
        for lp in [-0.5, -0.4, -0.3] {
            state.record(lp).unwrap();
        }
        assert!(!state.should_stop());
        state.record(-2.5).unwrap();
        assert!(state.should_stop());
        state.begin(Some(-3.0));
        assert!(!state.should_stop());
        state.record(-5.0).unwrap();
        assert!(state.should_stop());
        state.clear();
        assert!(!state.should_stop());
        assert!(state.record(f32::NAN).is_err());
        assert!(state.record(0.1).is_err());
    }
}
