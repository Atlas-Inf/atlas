// SPDX-License-Identifier: AGPL-3.0-only

//! The hyper-connection block, composed, against the CPU oracle.

use super::*;

/// Run Qwen4ExpTextGatedResidual end to end on the device and diff it against
/// the CPU oracle, which is itself checked against HuggingFace at 1.6e-7.
///
/// This is the first of the two novel blocks to exist as GPU work rather than
/// as a specification.
pub(super) fn hyper_connection_block(g: &dyn GpuBackend) -> Result<()> {
    use spark_model::layers::ops;
    use spark_model::weight_map::DenseWeight;

    const LOWRANK: usize = 320; // published hc_lowrank
    let wide = HIDDEN * GROUPS;

    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    let hyper: Vec<f32> = (0..wide).map(|_| next()).collect();
    let hc_norm: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();
    let mix_down: Vec<f32> = (0..LOWRANK * wide).map(|_| next() * 0.05).collect();
    let mix_up: Vec<f32> = (0..wide * LOWRANK).map(|_| next() * 0.05).collect();
    // Scaled by 1/sqrt(wide): a dot product over 10240 normalised terms
    // otherwise saturates the injection sigmoid at 2.0 on both sides, and a
    // saturated sigmoid agrees no matter what the sign or the /hc_count
    // divisor did.
    let inject_scale = 0.5 / (wide as f32).sqrt();
    let inject_w: Vec<f32> = (0..GROUPS * wide).map(|_| next() * inject_scale).collect();

    let d_hyper = up_bf16(g, &hyper)?;
    let d_norm_w = up_bf16(g, &hc_norm)?;
    let d_down = DenseWeight {
        weight: up_bf16(g, &mix_down)?,
    };
    let d_up = DenseWeight {
        weight: up_bf16(g, &mix_up)?,
    };
    let d_inject = DenseWeight {
        weight: up_bf16(g, &inject_w)?,
    };

    let d_normed = g.alloc(wide * 2)?;
    let d_lowrank = g.alloc(LOWRANK * 2)?;
    let d_gate = g.alloc(wide * 2)?;
    let d_mixed = g.alloc(HIDDEN * 2)?;
    let d_raw_inject = g.alloc(GROUPS * 2)?;
    let d_injection = g.alloc(GROUPS * 2)?;

    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let norm_k = g.kernel("norm", "rms_norm_grouped")?;
    let act_k = g.kernel("qwen4exp_hc", "q4e_hc_lowrank_act")?;
    let mix_k = g.kernel("qwen4exp_hc", "q4e_hc_stream_mix")?;
    let inj_k = g.kernel("qwen4exp_hc", "q4e_hc_injection")?;

    KernelLaunch::new(g, norm_k)
        .grid([1, GROUPS as u32, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_hyper)
        .arg_ptr(d_norm_w)
        .arg_ptr(d_normed)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .arg_f32(EPS)
        .launch(0)?;

    ops::dense_gemv(
        g,
        gemv,
        d_normed,
        &d_down,
        d_lowrank,
        LOWRANK as u32,
        wide as u32,
        0,
    )?;
    KernelLaunch::new(g, act_k)
        .grid([1, 1, 1])
        .block([LOWRANK.min(1024) as u32, 1, 1])
        .arg_ptr(d_lowrank)
        .arg_u32(LOWRANK as u32)
        .arg_u32(GROUPS as u32)
        .launch(0)?;
    ops::dense_gemv(
        g,
        gemv,
        d_lowrank,
        &d_up,
        d_gate,
        wide as u32,
        LOWRANK as u32,
        0,
    )?;

    KernelLaunch::new(g, mix_k)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_gate)
        .arg_ptr(d_normed)
        .arg_ptr(d_mixed)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .launch(0)?;

    ops::dense_gemv(
        g,
        gemv,
        d_normed,
        &d_inject,
        d_raw_inject,
        GROUPS as u32,
        wide as u32,
        0,
    )?;
    KernelLaunch::new(g, inj_k)
        .grid([1, 1, 1])
        .block([GROUPS as u32, 1, 1])
        .arg_ptr(d_raw_inject)
        .arg_ptr(d_injection)
        .arg_u32(GROUPS as u32)
        .launch(0)?;
    g.synchronize(0)?;

    let got_mixed = down_bf16(g, d_mixed, HIDDEN)?;
    let got_inject = down_bf16(g, d_injection, GROUPS)?;

    // Oracle on the same BF16-rounded inputs.
    let dims = atlas_core::qwen4exp_reference::PleDims {
        hidden: HIDDEN,
        hc_count: GROUPS,
        ple_embed_dim: HIDDEN,
        kernel: 0,
        dilation: 0,
        eps: EPS,
    };
    let (hn, md, mu, iw) = (
        hc_norm.iter().map(round).collect::<Vec<_>>(),
        mix_down.iter().map(round).collect::<Vec<_>>(),
        mix_up.iter().map(round).collect::<Vec<_>>(),
        inject_w.iter().map(round).collect::<Vec<_>>(),
    );
    let want = atlas_core::qwen4exp_reference::hyper_connection_forward(
        &dims,
        &atlas_core::qwen4exp_reference::HyperConnectionWeights {
            hc_norm: &hn,
            mix_down: &md,
            mix_up: &mu,
            block_inject: Some(&iw),
        },
        LOWRANK,
        &hyper.iter().map(round).collect::<Vec<_>>(),
    );

    let mixed_gap = got_mixed
        .iter()
        .zip(&want.mixed)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mixed_scale = want.mixed.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let inject_gap = got_inject
        .iter()
        .zip(&want.injection)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    println!(
        "hyper-connection: mixed max|diff| {mixed_gap:.3e} (up to {mixed_scale:.3e}, relative {:.3e})",
        mixed_gap / mixed_scale.max(1e-9)
    );
    println!(
        "hyper-connection: injection {:?} vs oracle {:?}  max|diff| {inject_gap:.3e}",
        got_inject
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        want.injection
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    // Several GEMV + BF16 rounds accumulate; a few ulps is expected.
    anyhow::ensure!(
        mixed_gap / mixed_scale.max(1e-9) < 2e-2,
        "hyper-connection mixed output disagrees with the oracle"
    );
    anyhow::ensure!(
        inject_gap < 2e-2,
        "injection gains disagree with the oracle"
    );
    // Guard against agreeing only because the sigmoid saturated.
    anyhow::ensure!(
        want.injection
            .iter()
            .any(|v| (*v - 2.0).abs() > 0.05 && *v > 0.05),
        "injection gains are saturated -- this test would pass with the wrong sign"
    );
    println!("HYPER-CONNECTION BLOCK MATCHES THE ORACLE\n");
    ple_block(g)
}

// ── The whole PLE tower on GPU ──────────────────────────────────────────────
