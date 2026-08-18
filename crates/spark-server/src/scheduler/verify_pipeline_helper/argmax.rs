// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time argmax with the sampler's historical last-index-wins tie-break.
//!
//! Split out of `verify_pipeline_helper.rs`, which is over the 500 LoC cap.

/// Argmax with the sampler's exact historical tie-break: the LAST vocabulary
/// index holding the maximum wins. Delegates to the runtime SSOT used by the
/// non-speculative temperature-zero sampler.
pub(super) fn argmax_last_wins(logits: &[f32]) -> u32 {
    spark_runtime::sampler::argmax_last_wins_f32(logits)
}

#[cfg(test)]
mod argmax_tests {
    use super::argmax_last_wins;

    fn reference(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn agree(v: &[f32]) {
        assert_eq!(argmax_last_wins(v), reference(v), "diverged on {v:?}");
    }

    #[test]
    fn matches_sampler_reference_on_edge_cases() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 2.0, 3.0]);
        agree(&[3.0, 2.0, 1.0]);
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0]);
        agree(&[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]);
        agree(&[-5.0, -1.0, -3.0]);
        agree(&[-0.0, 0.0]);
        agree(&[0.0, -0.0]);
        agree(&[-1.0, -0.0, 0.0, -1.0]);
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, f32::NAN, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN, f32::NAN]);
        agree(&[f32::NEG_INFINITY, -1.0]);
        agree(&[f32::INFINITY, 1.0]);
        agree(&[1.0, f32::INFINITY, f32::INFINITY]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn matches_reference_on_vocab_sized_input() {
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[123_457] = 999.0;
        v[200_003] = 999.0;
        assert_eq!(argmax_last_wins(&v), reference(&v));
        assert_eq!(argmax_last_wins(&v), 200_003);
    }
}
