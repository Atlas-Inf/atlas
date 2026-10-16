// SPDX-License-Identifier: AGPL-3.0-only

//! Draft-confidence calibrator for the qwen4_exp MTP proposer, a port of
//! exllamav3 `generator/draft_confidence.py`.
//!
//! Scores (the draft head's raw max logit) go into fixed-width bins of
//! exponentially decayed (tested, accepted) counts. Labels come only from
//! positions the verifier actually tested: every accepted draft, and the first
//! rejected one. `estimate` maps a score to the acceptance rate of the nearest
//! populated bin at or below it; the proposer multiplies the estimates along
//! the window and stops drafting once the product drops below `confidence`,
//! keeping that last draft so its outcome still gets labelled.
//!
//! `ATLAS_MTP_DRAFT_CONFIDENCE=<tau>` turns it on (exllamav3 default 0.4,
//! the S387T MTP-5 baseline ran 0.6). `ATLAS_MTP_DRAFT_CONF_BIN` sets the bin
//! width (default 1.0 logit).

use std::collections::BTreeMap;

pub(super) struct DraftConfCalibrator {
    pub(super) confidence: f64,
    bin_width: f64,
    decay: f64,
    min_count: f64,
    burn_in: f64,
    /// bin index -> [tested, accepted], decayed.
    bins: BTreeMap<i64, [f64; 2]>,
    total: f64,
}

impl DraftConfCalibrator {
    pub(super) fn from_env() -> Option<Self> {
        let tau: f64 = std::env::var("ATLAS_MTP_DRAFT_CONFIDENCE")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        if !(tau > 0.0 && tau < 1.0) {
            return None;
        }
        let bin_width = std::env::var("ATLAS_MTP_DRAFT_CONF_BIN")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|w: &f64| *w > 0.0)
            .unwrap_or(1.0);
        tracing::info!("qwen4_exp MTP draft confidence: tau {tau} bin {bin_width}");
        Some(Self::new(tau, bin_width))
    }

    pub(super) fn new(confidence: f64, bin_width: f64) -> Self {
        Self {
            confidence,
            bin_width,
            decay: 0.995,
            min_count: 8.0,
            burn_in: 64.0,
            bins: BTreeMap::new(),
            total: 0.0,
        }
    }

    fn bin(&self, score: f64) -> i64 {
        (score / self.bin_width).floor() as i64
    }

    pub(super) fn add_label(&mut self, score: f64, accepted: bool) {
        let b = self.bins.entry(self.bin(score)).or_insert([0.0, 0.0]);
        b[0] += 1.0;
        if accepted {
            b[1] += 1.0;
        }
        self.total += 1.0;
    }

    /// Age the statistics; once per verification round.
    pub(super) fn decay_step(&mut self) {
        for b in self.bins.values_mut() {
            b[0] *= self.decay;
            b[1] *= self.decay;
        }
        self.total *= self.decay;
    }

    /// Estimated acceptance probability for a draft with this score: the
    /// nearest populated bin at or below it, else the lowest populated bin.
    /// Optimistic 1.0 until burn-in, so early rounds draft full windows.
    pub(super) fn estimate(&self, score: f64) -> f64 {
        if self.total < self.burn_in {
            return 1.0;
        }
        let idx = self.bin(score);
        let mut below = None;
        let mut first = None;
        for (&k, v) in &self.bins {
            if v[0] < self.min_count {
                continue;
            }
            first.get_or_insert(k);
            if k <= idx {
                below = Some(k);
            }
        }
        match below.or(first) {
            Some(k) => {
                let v = self.bins[&k];
                v[1] / v[0]
            }
            None => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DraftConfCalibrator;

    #[test]
    fn optimistic_until_burn_in_then_binned() {
        let mut c = DraftConfCalibrator::new(0.6, 1.0);
        assert_eq!(c.estimate(3.0), 1.0);
        for _ in 0..40 {
            c.add_label(10.5, true);
            c.add_label(3.2, false);
        }
        assert_eq!(c.estimate(10.9), 1.0);
        assert_eq!(c.estimate(3.9), 0.0);
        // Below every populated bin: falls back to the lowest one.
        assert_eq!(c.estimate(-5.0), 0.0);
        // Between bins: nearest populated bin at or below.
        assert_eq!(c.estimate(7.0), 0.0);
    }
}
