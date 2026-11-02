// SPDX-License-Identifier: AGPL-3.0-only

//! Optional cumulative-confidence stop: retain the current proposal and skip
//! later forwards when their accumulated probability falls below the threshold.
//! This affects proposal width only; target verification remains authoritative.
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use std::ffi::OsStr;

const STOP_ENV: &str = "ATLAS_GLM_NATIVE_MTP_STOP_PROB";
fn parse_stop_threshold(value: Option<&OsStr>) -> Result<Option<f32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = value.to_str().context("MTP stop threshold is not UTF-8")?;
    let threshold: f32 = text
        .parse()
        .with_context(|| format!("{STOP_ENV} must be a finite decimal f32"))?;
    ensure!(
        threshold.is_finite() && threshold > 0.0 && threshold <= 1.0,
        "{STOP_ENV} must satisfy 0 < p <= 1"
    );
    Ok(Some(threshold))
}
pub(super) struct ConfidencePlan {
    pub argmax_batch_lp: Option<KernelHandle>,
    pub stop_ln_threshold: Option<f32>,
}
impl ConfidencePlan {
    pub(super) fn from_env(gpu: &dyn GpuBackend) -> Result<Self> {
        let threshold = parse_stop_threshold(std::env::var_os(STOP_ENV).as_deref())?;
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
    fn stop_threshold_accepts_only_finite_unit_interval() {
        assert_eq!(parse_stop_threshold(None).unwrap(), None);
        assert_eq!(
            parse_stop_threshold(Some(OsStr::new("0.95"))).unwrap(),
            Some(0.95)
        );
        assert_eq!(
            parse_stop_threshold(Some(OsStr::new("1"))).unwrap(),
            Some(1.0)
        );
        for bad in ["0", "-0.5", "1.5", "NaN", "inf", "-inf", "", "abc"] {
            assert!(parse_stop_threshold(Some(OsStr::new(bad))).is_err());
        }
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
    #[cfg(unix)]
    #[test]
    fn stop_threshold_rejects_non_utf8() {
        use std::os::unix::ffi::OsStringExt;
        let value = std::ffi::OsString::from_vec(vec![255]);
        assert!(parse_stop_threshold(Some(&value)).is_err());
    }
}
