// SPDX-License-Identifier: AGPL-3.0-only
//! GPU parity test for the DFlash2 on-device candidate selector
//! (`dflash2_candidate_selector`) against an f32 host reference
//! implementing the same total order and greedy chain. Requires a CUDA GPU
//! and a real (non-stub) kernel build — all tests are `#[ignore]`d; run on
//! the GB10 with:
//!   cargo test -p spark-model --features spark-runtime/cuda \
//!     --test dflash2_selector_parity -- --ignored --nocapture
//!
//! Host reference (identical to the kernel semantics):
//!   - anchor row 0: unary argmax, first index wins ties
//!   - rows 1..gamma: top-k by (unary value desc, index asc);
//!     context[r] = pred[prev][r] * H[row][r];
//!     score(c) = unary(c) + sum_r context[r] * succ[c][r];
//!     best by strict `>` in list order; pick becomes `prev`.
#[path = "arm2_common/support.rs"]
mod support;

use anyhow::Result;
use spark_model::layers::ops;
use spark_runtime::gpu::GpuBackend;
use std::collections::HashSet;
use support::*;

/// bf16 bit patterns for the unary-logit grid. Patterns are monotonic in
/// value within each sign side: `0x3F80 + i` steps up through the positive
/// finite normals (from 1.0 toward bf16-max), then `0x8000 + j` continues
/// down the negative side — every pattern distinct, no duplicates, no
/// NaN/Inf. `n` past the ~49k available patterns wraps and repeats; those
/// duplicates are INTENTIONAL tie coverage — the (value desc, index asc)
/// order the kernel is contracted to match stays deterministic.
fn logit_grid(n: usize) -> Vec<u16> {
    const POS_BASE: u32 = 0x3F80; // bf16 1.0
    const POS_CAP: u32 = 0x7F7F - POS_BASE; // 0x3F80..=0x7F7E finite positive
    const NEG_CAP: u32 = 0x7F7E; // 0x8000..=0xFF7D finite negative
    const AVAIL: u64 = (POS_CAP + NEG_CAP) as u64;
    let grid: Vec<u16> = (0..n as u64)
        .map(|i| {
            let j = (i % AVAIL) as u32;
            if j < POS_CAP {
                (POS_BASE + j) as u16
            } else {
                (0x8000 + (j - POS_CAP)) as u16
            }
        })
        .collect();
    let distinct: HashSet<u16> = grid.iter().copied().collect();
    if n as u64 <= AVAIL {
        assert_eq!(
            distinct.len(),
            n,
            "bf16 grid must be duplicate-free for n={n}"
        );
    } else {
        eprintln!(
            "note: n={n} exceeds the {AVAIL} distinct grid patterns; \
             {} repeats are intentional tie coverage under the idx-asc rule",
            n as u64 - AVAIL
        );
    }
    grid
}

/// Fisher-Yates shuffle of 0..n with the support PRNG.
fn permutation(rng: &mut Rng, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        p.swap(i, j);
    }
    p
}

struct Case {
    gamma: usize,
    vocab: usize,
    rank: usize,
    top_k: usize,
    seed: u64,
}

fn run_case(c: &Case) -> Result<()> {
    let (gpu, stream) = setup()?;
    let kernel = gpu.kernel("dflash2_candidate_selector", "dflash2_candidate_selector")?;
    let mut rng = Rng(c.seed);

    // ── inputs ──────────────────────────────────────────────────────────
    // Per-row logits: a random permutation of a strictly increasing grid.
    let grid = logit_grid(c.vocab);
    let mut logits = vec![0u16; c.gamma * c.vocab];
    for row in 0..c.gamma {
        let perm = permutation(&mut rng, c.vocab);
        for (tok, &pos) in perm.iter().enumerate() {
            logits[row * c.vocab + tok] = grid[pos as usize];
        }
    }

    // Codebooks: seeded uniform [-0.05, 0.05] -> bf16.
    let pred: Vec<u16> = (0..c.vocab * c.rank)
        .map(|_| f32_to_bf16_bits(-0.05 + 0.1 * rng.unit()))
        .collect();
    let succ: Vec<u16> = (0..c.vocab * c.rank)
        .map(|_| f32_to_bf16_bits(-0.05 + 0.1 * rng.unit()))
        .collect();

    // Projected hidden: seeded uniform [-1, 1] -> bf16.
    let proj: Vec<u16> = (0..c.gamma * c.rank)
        .map(|_| f32_to_bf16_bits(-1.0 + 2.0 * rng.unit()))
        .collect();

    let last_token = (rng.next_u64() % c.vocab as u64) as u32;

    // ── device run ──────────────────────────────────────────────────────
    let d_logits = up_u16(&gpu, &logits)?;
    let d_proj = up_u16(&gpu, &proj)?;
    let d_pred = up_u16(&gpu, &pred)?;
    let d_succ = up_u16(&gpu, &succ)?;
    let d_out = gpu.alloc(c.gamma * 4)?;

    ops::dflash2_candidate_selector(
        &gpu,
        kernel,
        d_logits,
        d_proj,
        d_pred,
        d_succ,
        d_out,
        last_token,
        c.gamma as u32,
        c.vocab as u32,
        c.rank as u32,
        c.top_k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let kern_out = rd_u32(&gpu, d_out, c.gamma)?;

    // ── f32 host reference ──────────────────────────────────────────────
    let logits_f32 = |row: usize, tok: usize| bf16_bits_to_f32(logits[row * c.vocab + tok]);
    let pred_f32 = |t: usize, r: usize| bf16_bits_to_f32(pred[t * c.rank + r]);
    let succ_f32 = |t: usize, r: usize| bf16_bits_to_f32(succ[t * c.rank + r]);
    let h_f32 = |row: usize, r: usize| bf16_bits_to_f32(proj[row * c.rank + r]);

    let mut host_out = vec![0u32; c.gamma];

    // Anchor: unary argmax, first index wins ties.
    let mut anchor = 0usize;
    for t in 1..c.vocab {
        if logits_f32(0, t) > logits_f32(0, anchor) {
            anchor = t;
        }
    }
    host_out[0] = anchor as u32;
    assert_eq!(
        kern_out[0], host_out[0],
        "anchor mismatch: kernel {} vs host {}",
        kern_out[0], host_out[0]
    );

    // prev clamps to < vocab_size exactly like the kernel (`>= V` -> 0).
    let mut prev = if (last_token as usize) < c.vocab {
        last_token as usize
    } else {
        0
    };

    for row in 1..c.gamma {
        // top-k by (value desc, index asc)
        let mut cand: Vec<(f32, usize)> = (0..c.vocab).map(|t| (logits_f32(row, t), t)).collect();
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        cand.truncate(c.top_k);

        let context: Vec<f32> = (0..c.rank)
            .map(|r| pred_f32(prev, r) * h_f32(row, r))
            .collect();
        let score = |cand_id: usize| -> f32 {
            let unary = logits_f32(row, cand_id);
            let mut dot = 0.0f32;
            for r in 0..c.rank {
                dot += context[r] * succ_f32(cand_id, r);
            }
            unary + dot
        };

        let mut host_pick = cand[0].1;
        let mut host_score = score(host_pick);
        for &(_, cand_id) in cand.iter().skip(1) {
            let s = score(cand_id);
            if s > host_score {
                host_score = s;
                host_pick = cand_id;
            }
        }
        host_out[row] = host_pick as u32;
        prev = host_pick;

        let kern_pick = kern_out[row] as usize;
        if kern_pick != host_pick {
            let hs = host_score;
            let ks = score(kern_pick.min(c.vocab - 1));
            let tol = 1e-3 * hs.abs().max(1.0);
            assert!(
                (hs - ks).abs() <= tol,
                "row {row}: kernel pick {kern_pick} vs host pick {host_pick} \
                 (scores {ks} vs {hs}, tol {tol})"
            );
            eprintln!(
                "row {row}: tolerated near-tie — kernel pick {kern_pick} \
                 (score {ks}) vs host pick {host_pick} (score {hs})"
            );
        }
    }

    gpu.free(d_logits)?;
    gpu.free(d_proj)?;
    gpu.free(d_pred)?;
    gpu.free(d_succ)?;
    gpu.free(d_out)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn parity_gamma8_v4096_rank256_topk16() -> Result<()> {
    run_case(&Case {
        gamma: 8,
        vocab: 4096,
        rank: 256,
        top_k: 16,
        seed: 0xDF2A_0001,
    })
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn parity_gamma8_v100003_rank256_topk16() -> Result<()> {
    run_case(&Case {
        gamma: 8,
        vocab: 100_003,
        rank: 256,
        top_k: 16,
        seed: 0xDF2A_0002,
    })
}

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn parity_gamma4_v4096_rank64_topk4() -> Result<()> {
    run_case(&Case {
        gamma: 4,
        vocab: 4096,
        rank: 64,
        top_k: 4,
        seed: 0xDF2A_0003,
    })
}
