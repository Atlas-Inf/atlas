// SPDX-License-Identifier: AGPL-3.0-only

//! The hyper-connection sandwich: collapse, block, scatter, composed.

use super::*;

/// The two hyper-connection kernels are each checked alone. This checks them
/// COMPOSED, in the order `Qwen4ExpLayer::decode` will run them, because the
/// errors that survive per-kernel checks are the ones between kernels: a
/// scatter that accumulates onto the NORMALISED copy instead of the raw
/// residual, an injection indexed per-hidden rather than per-stream, or a
/// collapse whose scratch aliases the residual it is still reading.
///
/// The block itself is a fixed vector rather than a real mixer -- what is under
/// test is the plumbing around it, and a known block output makes the expected
/// residual computable in closed form.
pub(super) fn hc_sandwich_roundtrip(g: &dyn GpuBackend) -> Result<()> {
    use atlas_core::qwen4exp_reference::{
        HyperConnectionWeights as OracleHcW, PleDims, broadcast_inject, hyper_connection_forward,
    };
    use spark_model::layers::ops::qwen4exp as q4e;
    use spark_model::weight_map::DenseWeight;

    const HIDDEN: usize = 256;
    const HC: usize = 4;
    const LOWRANK: usize = 64;
    const EPS: f32 = 1e-6;
    let wide = HC * HIDDEN;

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    let residual: Vec<f32> = (0..wide).map(|_| next()).collect();
    // hc_norm is `wide`, not `hidden`: the grouped norm carries one weight per
    // element of the concatenated streams.
    let hc_norm: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();
    let mix_down: Vec<f32> = (0..LOWRANK * wide).map(|_| next() * 0.05).collect();
    let mix_up: Vec<f32> = (0..wide * LOWRANK).map(|_| next() * 0.05).collect();
    // The injection scale has to thread a needle, and the first attempt here
    // failed the control rather than the kernel.
    //
    // Too large and the sigmoid saturates at 2.0, where a gain agrees
    // regardless of sign or the /hc_count divisor. Too small -- 0.5/sqrt(wide),
    // which is what the STANDALONE check uses -- and every gain sits at
    // 2*sigmoid(0) = 1.0, the four streams receive near-identical updates, and
    // the per-stream control below cannot distinguish a correct scatter from
    // one that broadcast a single gain. That is the failure this value fixes:
    // agreement was 4.5e-3, but the spread was only 5x the error.
    //
    // 5/sqrt(wide) puts the gains in roughly 0.85..1.15 -- off both rails. The
    // assertion below pins that rather than trusting the constant.
    let inject_scale = 5.0 / (wide as f32).sqrt();
    let inject_w: Vec<f32> = (0..HC * wide).map(|_| next() * inject_scale).collect();
    // A distinctive block output: if the scatter ever dropped it or scaled it
    // uniformly, the per-stream differences would vanish.
    let block_out: Vec<f32> = (0..HIDDEN).map(|_| next() * 2.0).collect();

    let d_residual = up_bf16(g, &residual)?;
    let d_hc_norm = up_bf16(g, &hc_norm)?;
    let d_block_out = up_bf16(g, &block_out)?;
    let mix_down_w = DenseWeight {
        weight: up_bf16(g, &mix_down)?,
    };
    let mix_up_w = DenseWeight {
        weight: up_bf16(g, &mix_up)?,
    };
    let inject_ww = DenseWeight {
        weight: up_bf16(g, &inject_w)?,
    };

    let kernels = q4e::Qwen4ExpKernels::resolve(g)?;
    let dims = q4e::Qwen4ExpDims {
        hidden: HIDDEN,
        hc_count: HC,
        hc_lowrank: LOWRANK,
        eps: EPS,
    };
    let scratch = q4e::HyperConnectionScratch {
        normed: g.alloc(wide * 2)?,
        lowrank: g.alloc(LOWRANK * 2)?,
        gate: g.alloc(wide * 2)?,
        mixed: g.alloc(HIDDEN * 2)?,
        raw_injection: g.alloc(HC * 2)?,
        injection: g.alloc(HC * 2)?,
    };

    q4e::hyper_connection_collapse(
        g,
        &kernels,
        &dims,
        &q4e::HyperConnectionWeights {
            hc_norm: d_hc_norm,
            mix_down: &mix_down_w,
            mix_up: &mix_up_w,
            block_inject: Some(&inject_ww),
        },
        d_residual,
        &scratch,
        0,
    )?;
    // The block runs here in the real layer. Scatter its output back.
    q4e::scatter_add(
        g,
        &kernels,
        &dims,
        d_block_out,
        scratch.injection,
        d_residual,
        1,
        0,
    )?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_residual, wide)?;

    // Oracle: the same two steps, on BF16-rounded inputs.
    let pd = PleDims {
        hidden: HIDDEN,
        hc_count: HC,
        ple_embed_dim: 0,
        kernel: 0,
        dilation: 0,
        eps: EPS,
    };
    let r: Vec<f32> = residual.iter().map(round).collect();
    let out = hyper_connection_forward(
        &pd,
        &OracleHcW {
            hc_norm: &hc_norm.iter().map(round).collect::<Vec<_>>(),
            mix_down: &mix_down.iter().map(round).collect::<Vec<_>>(),
            mix_up: &mix_up.iter().map(round).collect::<Vec<_>>(),
            block_inject: Some(&inject_w.iter().map(round).collect::<Vec<_>>()),
        },
        LOWRANK,
        &r,
    );
    let scattered = broadcast_inject(
        &block_out.iter().map(round).collect::<Vec<_>>(),
        &out.injection,
        HIDDEN,
    );
    let want: Vec<f32> = r.iter().zip(&scattered).map(|(a, b)| a + b).collect();

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "hc sandwich: max|diff| {worst:.3e} over {} values (up to {scale:.3e}), relative {:.3e}",
        got.len(),
        worst / scale.max(1e-9)
    );

    // CONTROL: the streams must have DIVERGED. Each stream got the same block
    // output scaled by its own injection gain, so if the scatter had used one
    // gain for all four -- or indexed the injection wrongly -- the deltas would
    // coincide. Compare the per-stream deltas from the original residual.
    let delta = |sidx: usize| -> Vec<f32> {
        (0..HIDDEN)
            .map(|d| got[sidx * HIDDEN + d] - r[sidx * HIDDEN + d])
            .collect()
    };
    let (d0, d1) = (delta(0), delta(1));
    let spread = d0
        .iter()
        .zip(&d1)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "injection gains {:?} (must differ, and be off 0 and 2)",
        out.injection
            .iter()
            .map(|v| (v * 1e3).round() / 1e3)
            .collect::<Vec<_>>()
    );
    // The control is only meaningful if the gains actually differ. Pin that
    // here rather than trusting the scale constant above: a degenerate set
    // would make the spread assertion vacuous instead of failing loudly.
    let gmin = out.injection.iter().copied().fold(f32::MAX, f32::min);
    let gmax = out.injection.iter().copied().fold(f32::MIN, f32::max);
    anyhow::ensure!(
        gmax - gmin > 0.05 && gmin > 0.05 && gmax < 1.95,
        "injection gains are degenerate or saturated (min {gmin}, max {gmax}) -- \
         the per-stream control below would pass on a broadcast scatter"
    );

    println!("per-stream delta spread: {spread:.3e} (must be large)");
    anyhow::ensure!(
        spread > worst * 10.0,
        "streams received identical updates -- the injection is not per-stream"
    );

    anyhow::ensure!(
        worst / scale.max(1e-9) < 2e-2,
        "the composed collapse->scatter disagrees with the oracle"
    );
    println!("HC SANDWICH ROUND-TRIP MATCHES THE ORACLE");
    Ok(())
}
