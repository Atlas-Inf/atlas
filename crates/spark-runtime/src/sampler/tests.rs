// SPDX-License-Identifier: AGPL-3.0-only

// Module is gated by parent's `#[cfg(test)] mod tests;` declaration —
// no inner `#![cfg(test)]` needed (and nesting them is a duplicated
// attribute under recent rustc).

use super::*;

#[test]
fn test_bf16_to_f32() {
    // BF16 for 1.0: 0x3F80 → f32 bits 0x3F800000 = 1.0
    assert_eq!(bf16_to_f32(0x80, 0x3F), 1.0);
    // BF16 for -1.0: 0xBF80 → f32 bits 0xBF800000 = -1.0
    assert_eq!(bf16_to_f32(0x80, 0xBF), -1.0);
    // BF16 for 0.0: 0x0000
    assert_eq!(bf16_to_f32(0x00, 0x00), 0.0);
}

#[test]
fn test_argmax_bf16() {
    // 3 values: 1.0, 2.0, 0.5
    // BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000, BF16(0.5) = 0x3F00
    let data: Vec<u8> = vec![
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
        0x00, 0x3F, // 0.5
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_argmax_negative() {
    // Values: -1.0, -0.5, -2.0 → argmax should be index 1 (-0.5)
    let data: Vec<u8> = vec![
        0x80, 0xBF, // -1.0
        0x00, 0xBF, // -0.5
        0x00, 0xC0, // -2.0
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_greedy_params() {
    let params = SamplingParams::greedy(100);
    assert!(params.is_greedy());
    assert_eq!(params.max_tokens, 100);
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn test_argmax_f32() {
    // 3 values: 1.0, 2.0, 0.5 as FP32 little-endian
    let data: Vec<u8> = [1.0f32, 2.0f32, 0.5f32]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(argmax_f32(&data), 1);
}

#[test]
fn test_sampler_with_mock() {
    use crate::gpu::mock::MockGpuBackend;

    let gpu = MockGpuBackend::new();
    let vocab_size = 4;
    let mut sampler = Sampler::new(vocab_size);

    // Sampler reads BF16 from device (2 bytes/element), not FP32.
    // BF16 encoding: upper 2 bytes of the IEEE 754 FP32 representation.
    // BF16(0.5) = 0x3F00, BF16(3.0) = 0x4040, BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000
    let ptr = gpu.alloc(vocab_size * 2).unwrap();
    let logits: Vec<u8> = vec![
        0x00, 0x3F, // 0.5
        0x40, 0x40, // 3.0
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
    ];
    gpu.copy_h2d(&logits, ptr).unwrap();

    let params = SamplingParams::greedy(10);
    let token = sampler.sample(ptr, &params, &gpu).unwrap();
    assert_eq!(token, 1); // index 1 = BF16(3.0) = max
}

#[test]
fn test_top_n_sigma_keeps_high_logits() {
    // 5 tokens with moderate spread: [2.0, 1.0, 1.0, 1.0, 1.0]
    // mean = 1.2, sigma ≈ 0.4
    // Correct threshold (mean - 1*sigma) = 0.8 → keeps ALL tokens (all >= 0.8)
    // Bug threshold (mean + 1*sigma) = 1.6 → kills tokens 1-4 (1.0 < 1.6)
    let logits_f32 = [2.0f32, 1.0, 1.0, 1.0, 1.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 1.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // With correct threshold (mean - sigma = 0.8), all 5 tokens survive.
    // After softmax at temp=1: P(0)=exp(2)/Z≈0.42, P(1-4)=exp(1)/Z≈0.145 each.
    // With 500 samples, P(never see non-zero) ≈ 0.42^500 ≈ 0. Very reliable.
    let mut saw_non_zero = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token != 0 {
            saw_non_zero = true;
            break;
        }
    }
    assert!(
        saw_non_zero,
        "top_n_sigma=1.0 should not filter tokens above mean-sigma"
    );
}

#[test]
fn test_top_n_sigma_disabled_at_zero() {
    // With top_n_sigma=0.0, no filtering should occur.
    // Use moderate logits so softmax gives reasonable probabilities.
    let logits_f32 = [1.0f32, 1.0, 1.0, 1.0, 1.5];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0, // disabled
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // P(token 0-3) = exp(1.0)/Z ≈ 0.19 each, P(token 4) = exp(1.5)/Z ≈ 0.24
    // With 500 samples, P(never see token < 4) ≈ 0.24^500 ≈ 0.
    let mut saw_low = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token < 4 {
            saw_low = true;
            break;
        }
    }
    assert!(saw_low, "top_n_sigma=0.0 should not filter any tokens");
}

#[test]
fn greedy_breaks_exact_ties_to_the_lowest_index_like_the_gpu_argmax() {
    // Was LAST-index-wins (`max_by`), the opposite of the GPU kernels and the
    // verify pick, so an exact top-1 tie made plain decode and a speculative
    // verify commit different tokens.
    let mut v = vec![0.0f32; 300_000];
    v[7] = 3.5;
    v[200_000] = 3.5;
    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams::greedy(10);
    assert_eq!(sample_with_params_seeded(&bytes, &params, &[], None), 7);
    assert_eq!(argmax_first_wins_f32(&v), 7);
}

#[test]
fn test_sample_with_params_seeded_temperature_zero_returns_argmax() {
    // Direct call with temperature=0.0 must NOT divide-by-zero.
    // Should return the argmax of raw logits.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0; // explicit
    for _ in 0..10 {
        assert_eq!(sample_with_params_seeded(&logits, &params, &[], None), 1);
    }
}

#[test]
fn test_greedy_applies_repetition_penalty_before_argmax() {
    // Regression for 2026-05-01 Gemma-4-31B greedy creative collapse.
    // At temperature=0, repetition_penalty MUST shift argmax when the
    // previous-argmax token is in history. Before the fix, greedy
    // bypassed all penalty processing → infinite repetition loops on
    // models with `repetition_penalty=1.1` configured in MODEL.toml.
    //
    // Setup: token 1 has highest raw logit (1.7). With rep_penalty=1.5
    // applied to history [1, 1] (token 1 twice), its logit becomes
    // 1.7 / 1.5 = 1.133, which drops below token 3's 1.2. Argmax
    // should flip from 1 → 3.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.repetition_penalty = 1.5;
    let history = vec![1u32, 1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token, 3,
        "rep_penalty must shift greedy argmax away from history-repeated token"
    );

    // Sanity: same prompt without history → original argmax (token 1).
    let token_no_hist = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token_no_hist, 1, "no history → no penalty → raw argmax");

    // Sanity: rep_penalty=1.0 (default) → original argmax even with history.
    params.repetition_penalty = 1.0;
    let token_no_pen = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token_no_pen, 1,
        "rep_penalty=1.0 is no-op even with history"
    );
}

#[test]
fn test_greedy_applies_logit_bias_before_argmax() {
    // logit_bias must shift greedy argmax (e.g. for tool-call grammar).
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    // Bias token 0 by +5.0 → it should become argmax (0.5+5.0 = 5.5 > 1.7)
    params.logit_bias = vec![(0, 5.0)];
    let token = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token, 0, "logit_bias must shift greedy argmax");
}

#[test]
fn test_sample_with_params_seeded_repetition_penalty_zero_doesnt_div_by_zero() {
    // repetition_penalty=0.0 used to produce inf/0 logits. Now skipped.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.repetition_penalty = 0.0; // pathological
    // Token 1 was "seen" — under broken code its logit would have been
    // divided by 0, producing inf. With the guard, no penalty applies.
    let history = vec![1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, Some(42));
    // Just assert we got a valid token (no panic, no infinite loop).
    assert!(token < 4);
}

#[test]
fn test_top_n_sigma_filters_extreme_outliers() {
    // Logits: [100.0, -100.0, -100.0, -100.0, -100.0]
    // mean = -60.0, sigma ≈ 80.0
    // threshold at n=1: mean - sigma = -140 → keeps everything
    // threshold at n=0.5: mean - 0.5*sigma = -100 → keeps token 0 only
    // With very tight sigma (n=0.1): mean - 0.1*sigma = -68 → kills tokens 1-4
    let logits_f32 = [100.0f32, -100.0, -100.0, -100.0, -100.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.1, // tight filter
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // Token 0 (100.0) should always be selected since others are far below threshold
    for _ in 0..50 {
        let token = sample_with_params(&logits, &params);
        assert_eq!(
            token, 0,
            "extreme low-logit tokens should be filtered at tight sigma"
        );
    }
}

/// A BF16-valued row (ties are common, as in a real head) of `n` logits.
fn bf16_row(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: f32 = rng.r#gen::<f32>() * spread - spread / 2.0;
            f32::from_bits(v.to_bits() & 0xFFFF_0000)
        })
        .collect()
}

/// `row` as a vocabulary split over `ranks` would leave it: each rank's best
/// `c` logits (value descending, then the lower index) present, the rest NaN,
/// with the absent tokens' bound and log-mass at `temperature`.
fn split_view(row: &[f32], ranks: usize, c: usize, temperature: f32) -> (Vec<f32>, Absent) {
    let shard = row.len() / ranks;
    let mut view = vec![f32::NAN; row.len()];
    let mut bound = (f32::NEG_INFINITY, u32::MAX);
    let mut mass = 0.0f64;
    for r in 0..ranks {
        let mut ids: Vec<usize> = (r * shard..(r + 1) * shard).collect();
        ids.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        for &i in &ids[..c] {
            view[i] = row[i];
        }
        let last = (row[ids[c - 1]], ids[c - 1] as u32);
        if last.0 > bound.0 || (last.0 == bound.0 && last.1 < bound.1) {
            bound = last;
        }
        mass += ids[c..]
            .iter()
            .map(|&i| ((row[i] / temperature) as f64).exp())
            .sum::<f64>();
    }
    (
        view,
        Absent {
            bound: bound.0,
            bound_id: bound.1,
            log_mass: mass.ln() as f32,
        },
    )
}

/// The known entries of a split view.
fn present(view: &[f32]) -> Vec<(u32, f32)> {
    (0..view.len() as u32)
        .filter(|&i| !view[i as usize].is_nan())
        .map(|i| (i, view[i as usize]))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The split verify head's exactness claim: whenever the candidates decide
/// the pick, it is the full row's pick for the same seed; and the common
/// filters (top-k, a min-p floor, a top-p nucleus inside the candidates)
/// decide it. Covers top-n-sigma with and without a pipeline mask.
#[test]
fn a_split_rows_candidates_pick_what_the_full_row_picks() {
    let (n, ranks, c) = (8192, 4, 64);
    // (temperature, top_k, top_p, min_p, top_n_sigma, spread, expect decided)
    let configs: [(f32, u32, f32, f32, f32, f32, bool); 7] = [
        (0.7, 20, 0.95, 0.08, 1.0, 24.0, true),
        (1.0, 20, 0.95, 0.0, 0.0, 24.0, true),
        (1.0, 0, 1.0, 0.1, 0.0, 400.0, true),
        (0.5, 0, 0.95, 0.0, 0.0, 400.0, true),
        (0.7, 64 * 4, 0.95, 0.0, 0.0, 24.0, false),
        (1.0, 0, 1.0, 0.0, 0.0, 24.0, false),
        (1.0, 0, 0.95, 0.0, 1.0, 40.0, false),
    ];
    for (cfg, &(t, top_k, top_p, min_p, sigma, spread, decided)) in configs.iter().enumerate() {
        let mut params = SamplingParams::greedy(1);
        (params.temperature, params.top_k, params.top_p) = (t, top_k, top_p);
        (params.min_p, params.top_n_sigma) = (min_p, sigma);
        let mut some = 0;
        for trial in 0..200u64 {
            let mut row = bf16_row(n, 1000 * cfg as u64 + trial, spread);
            if trial % 2 == 1 {
                row[5] = f32::NEG_INFINITY; // a pipeline mask
            }
            let (view, absent) = split_view(&row, ranks, c, t);
            let seed = Some(77 + trial);
            let full = sample_with_params_seeded(&f32_bytes(&row), &params, &[], seed);
            if let Some(pick) = sample_with_absent(&present(&view), n, &params, seed, absent) {
                assert_eq!(pick, full, "config {cfg} trial {trial}");
                some += 1;
            }
        }
        if decided {
            assert_eq!(some, 200, "config {cfg} should always be decided");
        } else {
            assert!(some < 200, "config {cfg} cannot always be decided");
        }
    }
}

/// With nothing absent the split view is the row itself.
#[test]
fn a_row_without_absent_tokens_samples_as_before() {
    let row = bf16_row(4096, 5, 20.0);
    let mut params = SamplingParams::greedy(1);
    (params.temperature, params.top_p, params.top_k) = (0.8, 0.9, 0);
    let none = Absent {
        bound: f32::NEG_INFINITY,
        bound_id: 0,
        log_mass: f32::NEG_INFINITY,
    };
    for s in 0..100 {
        assert_eq!(
            sample_with_absent(&present(&row), row.len(), &params, Some(s), none),
            Some(sample_with_params_seeded(
                &f32_bytes(&row),
                &params,
                &[],
                Some(s)
            ))
        );
    }
}

/// Exact ties fall to the lower index wherever they sit in the row, so the
/// same seed picks the same token from any arrangement of the candidates.
#[test]
fn tied_logits_are_drawn_in_index_order() {
    let mut row = vec![-20.0f32; 1000];
    for i in [900, 3, 500] {
        row[i] = 4.0;
    }
    let mut params = SamplingParams::greedy(1);
    (params.temperature, params.top_k) = (1.0, 3);
    let picks: Vec<u32> = (0..3u64)
        .map(|s| sample_with_params_seeded(&f32_bytes(&row), &params, &[], Some(s)))
        .collect();
    // StdRng's first f32 for seeds 0..3 lands in each third in turn.
    let thirds: Vec<u32> = (0..3u64)
        .map(|s| {
            use rand::{Rng, SeedableRng};
            let u: f32 = rand::rngs::StdRng::seed_from_u64(s).r#gen();
            [3, 500, 900][((u * 3.0) as usize).min(2)]
        })
        .collect();
    assert_eq!(picks, thirds);
}
