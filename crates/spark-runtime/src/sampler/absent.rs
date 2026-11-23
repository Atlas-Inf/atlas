// SPDX-License-Identifier: AGPL-3.0-only
//
// Sampling a row that leaves tokens out (`sample_with_absent`): a
// vocabulary-split verify head's candidates. Split out of `sample_impl.rs`
// to keep it under the 500-line cap; the parent re-exports these.

use super::SamplingParams;
use super::sample_impl::draw;

/// The tokens a row leaves out: a vocabulary-split verify head gathers only
/// each rank's best logits, so the rest of a row is absent (NaN in the row).
/// Every absent token comes after `(bound, bound_id)` in the sampler's order
/// (logit descending, then the lower id), and `log_mass` is
/// `ln Σ_absent exp(logit / temperature)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Absent {
    pub bound: f32,
    pub bound_id: u32,
    pub log_mass: f32,
}

impl Absent {
    /// Whether token `id` at `logit` precedes every absent token.
    pub(super) fn beaten_by(&self, logit: f32, id: u32) -> bool {
        logit > self.bound || (logit == self.bound && id <= self.bound_id)
    }
}

/// [`super::sample_with_params_seeded`] over a row of which only `present`
/// (id, logit after the pipeline) are known: the pick is the one
/// the full row would give for the same draw, or `None` when the present
/// tokens do not decide it: a top-k, min-p or top-p cut that could reach an
/// absent token, or a draw over the whole vocabulary. The answer depends on
/// the row alone and comes before the draw, so a caller that then samples the
/// full row samples the same distribution. Penalties and bias must be
/// neutral (the caller's pipeline applied them).
pub fn sample_with_absent(
    present: &[(u32, f32)],
    params: &SamplingParams,
    seed: Option<u64>,
    absent: Absent,
) -> Option<u32> {
    let first = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if params.temperature <= 0.0 {
        let &(id, v) = present.iter().min_by(|a, b| first(a, b))?;
        return absent.beaten_by(v, id).then_some(id);
    }
    let logits: Vec<(u32, f32)> = present
        .iter()
        .copied()
        .filter(|(_, v)| v.is_finite())
        .collect();
    if logits.is_empty() {
        return None;
    }
    // Cantelli guarantees at least `count·k²/(1+k²)` survivors for k = top_n_sigma
    // and count = the full row's finite token count. Only the present finite
    // count is known here. The maximum always survives, even if the bound is < 1.
    let sigma_kept = (params.top_n_sigma > 0.0).then(|| {
        let k2 = f64::from(params.top_n_sigma).powi(2);
        ((logits.len() as f64 * k2 / (1.0 + k2)).floor() as usize).max(1)
    });
    draw(logits, params, seed, Some((absent, sigma_kept)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::sample_with_params_seeded;

    #[test]
    fn top_n_sigma_with_masked_logits_split_row() {
        let mut params = SamplingParams::greedy(1);
        params.temperature = 4.0;
        params.top_n_sigma = 0.1;
        let present = [(0, 3.0), (1, 0.0), (2, 0.0), (3, f32::NEG_INFINITY)];
        let absent = Absent {
            bound: 0.0,
            bound_id: 2,
            log_mass: -0.25,
        };
        let row = [3.0f32, 0.0, 0.0, f32::NEG_INFINITY, -1.0];
        let bytes: Vec<_> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let seed = Some(0);

        params.top_k = 2;
        assert_eq!(sample_with_absent(&present, &params, seed, absent), None);

        params.top_k = 1;
        let full = sample_with_params_seeded(&bytes, &params, &[], seed);
        assert_eq!(
            sample_with_absent(&present, &params, seed, absent),
            Some(full)
        );
    }
}
