// SPDX-License-Identifier: AGPL-3.0-only

//! Small-vocabulary fixtures outside GLM's shape. Only the last codebook
//! dimension is active, so the transition score has an exact scalar oracle;
//! this tests rank tails, staged-row boundaries and greedy predecessor updates
//! without depending on a particular CUDA reduction's rounding order.
use super::*;

#[path = "batched.rs"]
mod batched;

#[derive(Clone, Copy, Debug)]
struct Shape {
    gamma: usize,
    vocab: usize,
    rank: usize,
    top_k: usize,
}

fn reference(
    shape: Shape,
    logits: &[u16],
    anchor: u32,
    ban_depth: u32,
    end_ids: [u32; 4],
) -> Vec<(u32, f32)> {
    let mut prev = anchor;
    (0..shape.gamma)
        .map(|row| {
            let mut candidates: Vec<(u32, f32)> = logits[row * shape.vocab..][..shape.vocab]
                .iter()
                .enumerate()
                .map(|(id, &bits)| (id as u32, bf16_bits_to_f32(bits)))
                .filter(|(_, value)| !value.is_nan())
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            candidates.truncate(shape.top_k);
            candidates.resize(shape.top_k, (u32::MAX, f32::NEG_INFINITY));
            if row == 0 {
                return (candidates[0].0, 0.0);
            }
            let predecessor = if prev < shape.vocab as u32 { prev } else { 0 };
            let scores: Vec<f32> = candidates
                .iter()
                .map(|&(id, unary)| {
                    if id >= shape.vocab as u32
                        || unary == f32::NEG_INFINITY
                        || (row as u32 <= ban_depth && end_ids.contains(&id))
                    {
                        f32::NEG_INFINITY
                    } else {
                        // All factors and finite scores are exact binary fractions.
                        unary + 0.5 * (1 + predecessor % 2) as f32 * (id % 3) as f32 * 0.125
                    }
                })
                .collect();
            let best = (1..scores.len()).fold(
                0,
                |best, i| {
                    if scores[i] > scores[best] { i } else { best }
                },
            );
            prev = candidates[best].0;
            let mass: f32 = scores
                .iter()
                .map(|score| (score - scores[best]).exp())
                .sum();
            (prev, -mass.ln())
        })
        .collect()
}

fn compare(words: &[u32], expected: &[(u32, f32)], confidence: bool) -> Result<()> {
    let gamma = expected.len();
    for (row, &(token, want)) in expected.iter().enumerate() {
        ensure!(
            words[row] == token,
            "row {row}: token {} != {token}",
            words[row]
        );
        if confidence {
            let got = f32::from_bits(words[gamma + row]);
            if want.is_nan() {
                // No scored candidate (or an infinite best score) has no
                // finite log-softmax confidence. Do not compare NaN payloads.
                ensure!(
                    got.is_nan(),
                    "row {row}: expected unavailable confidence, got {got}"
                );
            } else {
                ensure!(
                    got.is_finite() && got <= 0.0 && (got - want).abs() <= 2e-6,
                    "row {row}: confidence {got} != {want}"
                );
            }
        }
    }
    Ok(())
}

fn run(shape: Shape, logits: &[u16], ban_depth: u32, end_ids: [u32; 4]) -> Result<()> {
    let (gpu, stream) = setup()?;
    let (prod, conf) = kernels(&gpu)?;
    let anchor = 5;
    let expected = reference(shape, logits, anchor, ban_depth, end_ids);
    let mut pred = vec![0u16; shape.vocab * shape.rank];
    let mut succ = pred.clone();
    let mut projected = vec![0u16; shape.gamma * shape.rank];
    for id in 0..shape.vocab {
        pred[(id + 1) * shape.rank - 1] = f32_to_bf16_bits((1 + id % 2) as f32);
        succ[(id + 1) * shape.rank - 1] = f32_to_bf16_bits((id % 3) as f32 * 0.125);
    }
    for row in 0..shape.gamma {
        projected[(row + 1) * shape.rank - 1] = f32_to_bf16_bits(0.5);
    }
    let d_logits = up_u16(&gpu, logits)?;
    let d_pred = up_u16(&gpu, &pred)?;
    let d_succ = up_u16(&gpu, &succ)?;
    let d_proj = up_u16(&gpu, &projected)?;
    let d_anchor = up_u32(&gpu, &[anchor])?;
    let d_ban = up_u32(&gpu, &[ban_depth])?;
    let scratch_bytes = ops::dflash2_selector_scratch_bytes(shape.gamma);
    let scratch = gpu.alloc(scratch_bytes)?;
    gpu.memset(scratch, 0, scratch_bytes)?;
    let output = gpu.alloc(shape.gamma * 8)?;
    for (kernel, confidence) in [(prod, false), (conf, true)] {
        // Reuse the scratch across widths of output and repeated launches.
        for _ in 0..2 {
            gpu.memset(output, 0xff, shape.gamma * 8)?;
            ops::dflash2_candidate_selector(
                &gpu,
                kernel,
                d_logits,
                d_proj,
                d_pred,
                d_succ,
                output,
                d_anchor,
                d_ban,
                end_ids,
                shape.gamma as u32,
                shape.vocab as u32,
                shape.rank as u32,
                shape.top_k as u32,
                scratch,
                stream,
            )?;
            gpu.synchronize(stream)?;
            compare(
                &rd_u32(&gpu, output, shape.gamma * 2)?,
                &expected,
                confidence,
            )?;
        }
    }
    for ptr in [
        d_logits, d_pred, d_succ, d_proj, d_anchor, d_ban, scratch, output,
    ] {
        gpu.free(ptr)?;
    }
    gpu.destroy_stream(stream)?;
    Ok(())
}

#[test]
#[ignore = "requires GB10 and the non-stub glm-5.3-flash/nvfp4 kernel target"]
fn selector_matches_scalar_reference_outside_glm_geometry() -> Result<()> {
    for (gamma, vocab, rank, top_k) in [
        (3, 73, 13, 3),
        (17, 1003, 64, 7),
        (19, 257, 8, 1),
        (2, 31, 1, 16),
    ] {
        let shape = Shape {
            gamma,
            vocab,
            rank,
            top_k,
        };
        let logits: Vec<_> = (0..gamma * vocab)
            .map(|i| f32_to_bf16_bits(((i * 17 + i / vocab * 7) % 19) as f32 - 9.0))
            .collect();
        run(shape, &logits, gamma as u32, [0, 1, u32::MAX, u32::MAX])?;
    }
    Ok(())
}

#[test]
#[ignore = "requires GB10 and the non-stub glm-5.3-flash/nvfp4 kernel target"]
fn selector_preserves_unavailable_confidence_for_nonfinite_or_banned_rows() -> Result<()> {
    let shape = Shape {
        gamma: 18,
        vocab: 97,
        rank: 13,
        top_k: 4,
    };
    let logits: Vec<_> = (0..shape.gamma * shape.vocab)
        .map(|i| {
            let (row, id) = (i / shape.vocab, i % shape.vocab);
            f32_to_bf16_bits(match row % 5 {
                1 => f32::NEG_INFINITY,
                2 => f32::NAN,
                3 if id < 4 => 20.0,
                4 if (10..12).contains(&id) => f32::INFINITY,
                _ => (id % 13) as f32 - 6.0,
            })
        })
        .collect();
    run(shape, &logits, shape.gamma as u32, [0, 1, 2, 3])
}

#[test]
fn confidence_gate_rejects_nan_for_finite_reference_and_accepts_explicit_nan() {
    assert!(compare(&[2, f32::NAN.to_bits()], &[(2, -0.5)], true).is_err());
    assert!(compare(&[2, f32::INFINITY.to_bits()], &[(2, -0.5)], true).is_err());
    assert!(compare(&[2, (-0.5f32).to_bits()], &[(2, -0.5)], true).is_ok());
    assert!(compare(&[2, f32::NAN.to_bits()], &[(2, f32::NAN)], true).is_ok());
    assert!(compare(&[2, 0.0f32.to_bits()], &[(2, f32::NAN)], true).is_err());
}

#[test]
fn scalar_reference_uses_anchor_then_each_selected_predecessor() {
    let shape = Shape {
        gamma: 3,
        vocab: 5,
        rank: 1,
        top_k: 2,
    };
    let logits: Vec<_> = [
        0.0, 3.0, 3.0, 0.0, -1.0, 1.0, 4.0, 4.0, 0.0, -1.0, 3.0, 0.0, 3.0, 1.0, -1.0,
    ]
    .into_iter()
    .map(f32_to_bf16_bits)
    .collect();
    let got = reference(shape, &logits, 0, 0, [u32::MAX; 4]);
    assert_eq!(got.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 2, 2]);
    assert!((got[1].1 + (1.0 + (-0.0625f32).exp()).ln()).abs() < 1e-6);
    assert!((got[2].1 + (1.0 + (-0.125f32).exp()).ln()).abs() < 1e-6);
}
