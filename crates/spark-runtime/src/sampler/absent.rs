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
    /// Whether token `id` at `logit` precedes every absent token. On a scaled
    /// row (logit / temperature, BF16 logits keep their order) `bound` is
    /// scaled the same way.
    pub(super) fn beaten_by(&self, bound: f32, logit: f32, id: u32) -> bool {
        logit > bound || (logit == bound && id <= self.bound_id)
    }
}

/// [`super::sample_with_params_seeded`] over a row of `vocab` tokens of which only
/// `present` (id, logit after the pipeline) are known: the pick is the one
/// the full row would give for the same draw, or `None` when the present
/// tokens do not decide it: a top-k, min-p or top-p cut that could reach an
/// absent token, or a draw over the whole vocabulary. The answer depends on
/// the row alone and comes before the draw, so a caller that then samples the
/// full row samples the same distribution. Penalties and bias must be
/// neutral (the caller's pipeline applied them).
pub fn sample_with_absent(
    present: &[(u32, f32)],
    vocab: usize,
    params: &SamplingParams,
    seed: Option<u64>,
    absent: Absent,
) -> Option<u32> {
    let first = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if params.temperature <= 0.0 {
        let &(id, v) = present.iter().min_by(|a, b| first(a, b))?;
        return absent.beaten_by(absent.bound, v, id).then_some(id);
    }
    // The full row's top-n-sigma filter ran unless a mask made its sum -inf
    // (absent tokens are finite there); by Cantelli it keeps at least the
    // best `vocab·k²/(1+k²)` logits (k = top_n_sigma), so `draw` checks that
    // every token the pick depends on is among those.
    let sigma_kept = (params.top_n_sigma > 0.0
        && !present.iter().any(|&(_, v)| v == f32::NEG_INFINITY))
    .then(|| {
        let k2 = (params.top_n_sigma as f64).powi(2);
        (vocab as f64 * k2 / (1.0 + k2)).floor() as usize
    });
    let logits: Vec<(u32, f32)> = present
        .iter()
        .filter(|(_, v)| v.is_finite())
        .map(|&(i, v)| (i, v / params.temperature))
        .collect();
    if logits.is_empty() {
        return None;
    }
    draw(logits, params, seed, Some((absent, sigma_kept)))
}
