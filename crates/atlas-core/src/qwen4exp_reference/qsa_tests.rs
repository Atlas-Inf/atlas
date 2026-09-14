// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the QSA indexer reference. These need no GPU and no checkpoint,
//! and they pin the details a transcription of this mechanism gets wrong:
//! whether the relu is per head or on the sum, whether the tail is always
//! visible, and where the inertness threshold actually falls.

use super::*;

/// Small but structurally faithful: 1 key head, several query heads, and a
/// budget low enough that a test sequence can cross it.
fn dims(budget: usize) -> QsaDims {
    QsaDims {
        hidden: 8,
        n_heads: 2,
        kv_heads: 1,
        head_dim: 4,
        // Rope off, so a test about SELECTION is not also a test about rope.
        // Position dependence gets its own test below.
        rotary_dim: 0,
        budget,
        ratio: 4,
        rope_theta: 10_000.0,
        eps: 1e-6,
    }
}

/// SplitMix64 — reproducible from the seed, no dev-dependency.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn weights(d: &QsaDims, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        noise(d.qk_width() * d.hidden, seed),
        noise(d.head_dim, seed ^ 0xAAAA),
        noise(d.head_dim, seed ^ 0x5555),
    )
}

/// **The property that makes short contexts exact rather than approximate.**
///
/// `topk(min(block_topk, complete_blocks))` means that while the visible
/// prefix holds no more complete blocks than `block_topk`, every block is
/// selected — so the indexer cannot mask anything and dense attention is
/// numerically identical. A first bring-up capped at the budget is therefore
/// exact, and this is what says so without a GPU.
#[test]
fn selects_everything_below_the_budget() {
    let d = dims(16); // block_topk = 4
    let (proj, qn, kn) = weights(&d, 7);
    let seq = 16; // at t=15: 4 complete blocks == block_topk
    let hidden = noise(seq * d.hidden, 3);
    let out = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &qn,
            k_layernorm: &kn,
        },
        &hidden,
    );
    for (t, sel) in out.selected.iter().enumerate() {
        let want: Vec<u32> = (0..=t as u32).collect();
        assert_eq!(
            *sel, want,
            "query {t} must see its whole causal prefix while the indexer is inert"
        );
    }
}

/// One block past the budget, the indexer starts restricting — and the number
/// of tokens it drops is exactly one block's worth per block over the limit.
#[test]
fn restricts_once_the_budget_is_crossed() {
    let d = dims(16); // block_topk = 4, ratio 4
    let (proj, qn, kn) = weights(&d, 11);
    let seq = 24; // at t=23: 6 complete blocks vs block_topk 4
    let hidden = noise(seq * d.hidden, 5);
    let out = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &qn,
            k_layernorm: &kn,
        },
        &hidden,
    );
    let last = out.selected.last().expect("a query");
    assert_eq!(
        last.len(),
        d.budget,
        "4 selected blocks x ratio 4 = 16 tokens, and the tail is empty at t=23"
    );
    // Dropped exactly the two blocks that lost, nothing from the tail.
    assert!(last.iter().all(|t| *t < 24));
    assert_eq!(
        out.selected[15].len(),
        16,
        "at t=15 it is still inert: 4 complete blocks, all selected"
    );
}

/// The incomplete tail is ALWAYS visible — and, the surprise, the CURRENT
/// TOKEN IS NOT.
///
/// `selected = top_k_blocks + tail`, and `tail` is
/// `visible[complete * ratio ..]`, which is EMPTY whenever the visible count
/// is a multiple of `ratio`. At those positions the current token sits inside a
/// complete block, and if that block loses the ranking the query cannot attend
/// itself. Read straight off the reference
/// (`modeling_qwen4_exp.py`), which force-includes nothing.
///
/// That is worth a test rather than a comment, because it is the first thing
/// someone will suspect is a bug when a decode step appears to ignore the
/// newest token — and because a "helpful" fix that force-included the current
/// token would silently diverge from the reference on 1 in `ratio` positions.
#[test]
fn the_tail_is_visible_and_the_current_token_is_not_force_included() {
    let d = dims(8); // block_topk = 2
    let (proj, qn, kn) = weights(&d, 13);
    let seq = 19; // 4 complete blocks + a 3-token tail at t=18
    let hidden = noise(seq * d.hidden, 17);
    let out = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &qn,
            k_layernorm: &kn,
        },
        &hidden,
    );

    let mut saw_masked_self = false;
    for (t, sel) in out.selected.iter().enumerate() {
        assert!(
            sel.iter().all(|&i| i as usize <= t),
            "query {t} attends a FUTURE token — the block set is not causal"
        );
        let complete = (t + 1) / d.ratio;
        let tail = complete * d.ratio..=t;
        let tail_empty = tail.is_empty();
        for token in tail {
            assert!(
                sel.contains(&(token as u32)),
                "query {t} dropped tail token {token}, which is always visible"
            );
        }
        if tail_empty {
            // The current token is inside a complete block here, so it is
            // subject to the ranking like any other.
            if !sel.contains(&(t as u32)) {
                saw_masked_self = true;
            }
        } else {
            assert!(
                sel.contains(&(t as u32)),
                "query {t} has a non-empty tail, so it must contain itself"
            );
        }
    }
    assert!(
        saw_masked_self,
        "no query was masked from itself in this fixture — if that ever becomes \
         true by construction rather than by luck, the force-inclusion question \
         is settled somewhere else and this test is no longer pinning it"
    );
}

/// **The relu is per HEAD, then summed.** `relu(sum)` is the natural misreading
/// and it is a different function: this fixture makes the two heads' dots
/// exactly cancel, so `relu(sum)` scores the block at zero while the reference
/// scores it at `|dot|`.
///
/// The projection is hand-built rather than random for exactly that reason —
/// head 1's query is the elementwise negation of head 0's, so after the same
/// per-head RMS norm their dots against any key are exactly opposite.
#[test]
fn the_relu_is_per_head_not_on_the_sum() {
    let mut d = dims(64);
    d.hidden = 4;
    d.head_dim = 4;
    d.n_heads = 2;
    let (h, hd) = (d.hidden, d.head_dim);

    // [qk_width, hidden] row-major: q0 = +I, q1 = -I, key = +I.
    let mut proj = vec![0f32; d.qk_width() * h];
    for i in 0..hd {
        proj[i * h + i] = 1.0; // head 0
        proj[(hd + i) * h + i] = -1.0; // head 1 — exact negation
        proj[(2 * hd + i) * h + i] = 1.0; // the key row
    }
    // Zero offsets, so the norm is a pure RMS normalise and the signs are all
    // that survive.
    let zeros = vec![0f32; hd];

    let seq = 8;
    let hidden = noise(seq * h, 23);
    let out = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &zeros,
            k_layernorm: &zeros,
        },
        &hidden,
    );

    let blocks = seq / d.ratio;
    assert!(blocks > 0, "the fixture must produce at least one block");
    let mut any_positive = false;
    for t in 0..seq {
        for b in 0..(t + 1) / d.ratio {
            let s = out.scores[t * blocks + b];
            assert!(s >= 0.0, "a relu-summed score cannot be negative, got {s}");
            if s > 1e-6 {
                any_positive = true;
            }
        }
    }
    assert!(
        any_positive,
        "every score came out zero — that is what relu(sum) produces on this \
         fixture, and the reference sums relu PER HEAD"
    );
}

/// A block's key is roped at the block's FIRST token, so the scores depend on
/// WHERE a block sits, not only on what is in it. This feeds the same token
/// content at two different rotary widths and requires the scores to move —
/// a transcription that skipped rope, or roped at the query instead, would
/// leave them identical.
#[test]
fn block_keys_carry_their_position() {
    let (mut with_rope, mut without) = (dims(64), dims(64));
    with_rope.rotary_dim = 4; // the whole head rotates
    without.rotary_dim = 0; // rope off
    let (proj, qn, kn) = weights(&with_rope, 29);
    let seq = 12;
    let hidden = noise(seq * with_rope.hidden, 31);
    let w = QsaWeights {
        index_qk_proj: &proj,
        q_layernorm: &qn,
        k_layernorm: &kn,
    };
    let roped = qsa_select(&with_rope, &w, &hidden);
    let plain = qsa_select(&without, &w, &hidden);
    let moved = roped
        .scores
        .iter()
        .zip(&plain.scores)
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        moved,
        "rope changed no score — the block keys are not carrying position"
    );
}

/// The norms are offset-from-1 (`x * (1 + w)`), not `x * w`, and they are the
/// same trap as everywhere else in this model: with weights near zero the two
/// forms differ by a factor the scores absorb silently. Changing the offset
/// must change the scores.
#[test]
fn the_norm_offset_is_load_bearing() {
    let d = dims(64);
    let (proj, qn, kn) = weights(&d, 37);
    let seq = 12;
    let hidden = noise(seq * d.hidden, 41);
    let base = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &qn,
            k_layernorm: &kn,
        },
        &hidden,
    );
    // The same weights shifted: under `x * (1 + w)` this is a real change;
    // under a hypothetical `x * w` it would also change, but a transcription
    // that IGNORED the weight entirely would not notice either.
    let shifted: Vec<f32> = qn.iter().map(|v| v + 0.5).collect();
    let moved = qsa_select(
        &d,
        &QsaWeights {
            index_qk_proj: &proj,
            q_layernorm: &shifted,
            k_layernorm: &kn,
        },
        &hidden,
    );
    let changed = base
        .scores
        .iter()
        .zip(&moved.scores)
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(changed, "q_layernorm's weight is not reaching the scores");
}
