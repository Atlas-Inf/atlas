// SPDX-License-Identifier: AGPL-3.0-only

//! QSA half of the qwen4_exp oracle parity gate. `super` is
//! `qwen4exp_oracle_tests`, so the fixture RNG, the BF16 rounding, the device
//! helpers and the `check` / `check_control` pair come from there.
//!
//! This is the piece with NO committed golden: `bench/qwen4_exp/qsa_golden.py`
//! generates one from the real reference module, but the `.npz` is gitignored,
//! so on a fresh checkout the QSA kernels are the least-checked thing in the
//! port — and QSA is what makes context above `indexer_budget` correct rather
//! than approximate.
//!
//! `atlas_core::qwen4exp_reference::qsa_*` closes that: it is transcribed from
//! the vendored reference module rather than from the kernel, and the per-stage
//! helpers it exposes are the SAME code `qsa_select` runs, so this compares
//! against the oracle rather than against a second copy of the formula.

use spark_runtime::gpu::GpuBackend;

use super::*;
use crate::layers::ops;

/// The published indexer geometry: 4 query heads, 1 key head, 128 dims, budget
/// 2048, ratio 4, and 64 of 128 dims rotated (`partial_rotary_factor` 0.25 of
/// the attention head_dim 256).
fn qsa_dims() -> atlas_core::qwen4exp_reference::QsaDims {
    atlas_core::qwen4exp_reference::QsaDims {
        hidden: HIDDEN,
        n_heads: 4,
        kv_heads: 1,
        head_dim: 128,
        rotary_dim: 64,
        budget: 2048,
        ratio: 4,
        rope_theta: 10_000_000.0,
        eps: EPS,
    }
}

/// `qsa_block_pool` / `qsa_qprep` / `qsa_score` against the CPU oracle.
///
/// Three details are pinned, and each one leaves plausible numbers when wrong:
///
///   * a block's key is roped at the block's FIRST token, not at the query and
///     not at the block's centre;
///   * the norms are offset-from-1 and per HEAD;
///   * the relu is per head, then summed.
#[test]
#[ignore]
fn qwen4exp_oracle_qsa_matches_the_cpu_reference() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_pool = g.kernel("qsa_indexer", "qsa_block_pool").unwrap();
    let k_qprep = g.kernel("qsa_indexer", "qsa_qprep").unwrap();
    let k_score = g.kernel("qsa_indexer", "qsa_score").unwrap();
    for (name, k) in [
        ("qsa_block_pool", k_pool),
        ("qsa_qprep", k_qprep),
        ("qsa_score", k_score),
    ] {
        assert!(k.0 != 0, "{name} resolved to handle 0");
    }

    let d = qsa_dims();
    let hd = d.head_dim;
    let mut rng = Rng(0xB5AD_4ECE_DA08_1AFF);

    // Enough tokens for selection to be interesting without needing 2048 of
    // them: the kernels do not care how many blocks there are, and the
    // per-stage math is what this test is about.
    let blocks = 24usize;
    let seq = blocks * d.ratio;

    // The kernel reads BF16 raw keys and BF16 norm weights, so the oracle is
    // fed exactly those values — otherwise the tolerance would be absorbing a
    // rounding step instead of measuring the kernel.
    let raw_keys = bf16_round(&rng.vec(seq * hd, 1.0));
    // Offset-from-1 norms, centred off zero like the real tensors.
    let k_norm = bf16_round(
        &rng.vec(hd, 0.3)
            .iter()
            .map(|v| v - 0.08)
            .collect::<Vec<_>>(),
    );
    let q_norm = bf16_round(
        &rng.vec(hd, 0.3)
            .iter()
            .map(|v| v - 0.05)
            .collect::<Vec<_>>(),
    );

    let d_raw = upload(g, &bf16_bytes(&raw_keys));
    let d_knorm = upload(g, &bf16_bytes(&k_norm));
    let d_qnorm = upload(g, &bf16_bytes(&q_norm));
    let d_blocks = g.alloc(blocks * hd * 2).unwrap();

    // ── qsa_block_pool: mean over the block, norm, rope at the block start.
    ops::qsa_block_pool(
        g,
        k_pool,
        d_raw,
        d_knorm,
        d_blocks,
        0,
        blocks as u32,
        d.ratio as u32,
        hd as u32,
        d.rotary_dim as u32,
        d.rope_theta,
        d.eps,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let mut want_blocks = Vec::with_capacity(blocks * hd);
    for b in 0..blocks {
        let first = b * d.ratio;
        want_blocks.extend(atlas_core::qwen4exp_reference::qsa_block_key(
            &d,
            &k_norm,
            &raw_keys[first * hd..(first + d.ratio) * hd],
            first,
        ));
    }
    println!("qsa_block_pool:");
    let got_blocks = download_bf16(g, d_blocks, blocks * hd);
    check("block_keys", &got_blocks, &want_blocks);

    // CONTROL. Rope at the block's LAST token instead of its first. Both are
    // plausible readings of "the block's position" and they are different
    // functions; without this the check above would pass on either.
    let mut wrong_pos = Vec::with_capacity(blocks * hd);
    for b in 0..blocks {
        let first = b * d.ratio;
        wrong_pos.extend(atlas_core::qwen4exp_reference::qsa_block_key(
            &d,
            &k_norm,
            &raw_keys[first * hd..(first + d.ratio) * hd],
            first + d.ratio - 1,
        ));
    }
    check_control("control: rope@last", &got_blocks, &wrong_pos);

    // ── qsa_qprep: one decode query, per head norm + rope at its position.
    let pos = seq - 1;
    let q_raw = bf16_round(&rng.vec(d.n_heads * hd, 1.0));
    let d_q_in = upload(g, &bf16_bytes(&q_raw));
    let d_q_out = g.alloc(d.n_heads * hd * 4).unwrap();
    ops::qsa_qprep(
        g,
        k_qprep,
        d_q_in,
        d_qnorm,
        d_q_out,
        d.n_heads as u32,
        hd as u32,
        d.rotary_dim as u32,
        pos as u32,
        d.rope_theta,
        d.eps,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let mut want_q = Vec::with_capacity(d.n_heads * hd);
    for head in 0..d.n_heads {
        want_q.extend(atlas_core::qwen4exp_reference::qsa_query_head(
            &d,
            &q_norm,
            &q_raw[head * hd..(head + 1) * hd],
            pos,
        ));
    }
    println!("qsa_qprep:");
    let got_q = download_f32(g, d_q_out, d.n_heads * hd);
    check("q_post", &got_q, &want_q);

    // ── qsa_score: sum over heads of relu(q_h . k_b), over sqrt(hd).
    let d_scores = g.alloc(blocks * 4).unwrap();
    ops::qsa_score(
        g,
        k_score,
        d_q_out,
        d_blocks,
        d_scores,
        blocks as u32,
        d.n_heads as u32,
        hd as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let want_scores: Vec<f32> = (0..blocks)
        .map(|b| {
            atlas_core::qwen4exp_reference::qsa_block_score(
                &d,
                &want_q,
                &want_blocks[b * hd..(b + 1) * hd],
            )
        })
        .collect();
    println!("qsa_score:");
    let got_scores = download_f32(g, d_scores, blocks);
    check("scores", &got_scores, &want_scores);
    assert!(
        got_scores.iter().all(|s| *s >= 0.0),
        "a relu-summed score cannot be negative"
    );

    // CONTROL. Drop the per-head relu and sum the raw dots. With 4 heads on
    // random data some dots are negative, so this must differ — and a kernel
    // that took relu of the SUM would land here, not on the reference.
    let no_relu: Vec<f32> = (0..blocks)
        .map(|b| {
            let k = &want_blocks[b * hd..(b + 1) * hd];
            let dots: f32 = (0..d.n_heads)
                .map(|head| {
                    want_q[head * hd..(head + 1) * hd]
                        .iter()
                        .zip(k)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .sum();
            dots / (hd as f32).sqrt()
        })
        .collect();
    check_control("control: no per-head relu", &got_scores, &no_relu);
}
