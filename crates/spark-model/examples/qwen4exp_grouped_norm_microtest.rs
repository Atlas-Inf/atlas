// SPDX-License-Identifier: AGPL-3.0-only

//! Check the `rms_norm_grouped` kernel against the CPU oracle.
//!
//! ```text
//! cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest
//! ```
//!
//! The oracle (`atlas_core::qwen4exp_reference::grouped_rms_norm`) is itself
//! checked against HuggingFace at real weights, so agreement here chains the
//! kernel to the reference implementation rather than to another guess.
//!
//! Grouping is the whole point: qwen4_exp's residual stream is `hc_count`
//! streams of `hidden_size` concatenated and each normalises over ITSELF. A
//! single reduction over the full row is a different function that still
//! produces fluent output, so this also asserts the two disagree — a kernel
//! that silently ignored `group_size` would otherwise look correct.

use anyhow::Result;
use atlas_core::qwen4exp_reference::grouped_rms_norm;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HIDDEN: usize = 2560; // published hidden_size
const GROUPS: usize = 4; // hc_count
const TOKENS: usize = 3;
const EPS: f32 = 1e-6;

fn up_bf16(g: &dyn GpuBackend, values: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let ptr = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn down_bf16(g: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; len * 2];
    g.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let wide = HIDDEN * GROUPS;

    // Deterministic inputs; the streams are given different scales so a
    // whole-row reduction cannot coincidentally agree with a per-group one.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 8192.0 - 1.0
    };
    let input: Vec<f32> = (0..TOKENS * wide)
        .map(|i| next() * (1.0 + (i / HIDDEN % GROUPS) as f32))
        .collect();
    let weight: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();

    // The kernel reads BF16. Round the oracle's inputs the same way, so the
    // comparison measures the kernel rather than the upload -- otherwise BF16's
    // ~2^-8 input rounding dominates and the test says nothing.
    let round = |v: &f32| bf16::from_f32(*v).to_f32();
    let input_bf16: Vec<f32> = input.iter().map(round).collect();
    let weight_bf16: Vec<f32> = weight.iter().map(round).collect();

    let d_in = up_bf16(g, &input)?;
    let d_w = up_bf16(g, &weight)?;
    let d_out = g.alloc(TOKENS * wide * 2)?;

    let kernel = g.kernel("norm", "rms_norm_grouped")?;
    KernelLaunch::new(g, kernel)
        .grid([TOKENS as u32, GROUPS as u32, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_in)
        .arg_ptr(d_w)
        .arg_ptr(d_out)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .arg_f32(EPS)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, TOKENS * wide)?;

    // Oracle, per token.
    let mut want = Vec::with_capacity(TOKENS * wide);
    for t in 0..TOKENS {
        want.extend(grouped_rms_norm(
            &input_bf16[t * wide..(t + 1) * wide],
            HIDDEN,
            &weight_bf16,
            EPS,
        ));
    }

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "grouped: max|diff| {worst:.3e} over {} values (up to {scale:.3e}), relative {:.3e}",
        got.len(),
        worst / scale.max(1e-9)
    );

    // A whole-row normalisation must NOT match — otherwise the kernel could be
    // ignoring group_size and this test would pass for the wrong reason.
    let mut ungrouped = Vec::with_capacity(TOKENS * wide);
    for t in 0..TOKENS {
        ungrouped.extend(grouped_rms_norm(
            &input_bf16[t * wide..(t + 1) * wide],
            wide,
            &weight_bf16,
            EPS,
        ));
    }
    let ungrouped_gap = got
        .iter()
        .zip(&ungrouped)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("ungrouped control: max|diff| {ungrouped_gap:.3e} (must be large)");

    // Inputs now match; only the BF16 OUTPUT store separates the two, so this
    // should sit near a single ulp (~2^-8 relative) rather than a few.
    anyhow::ensure!(
        worst / scale.max(1e-9) < 5e-3,
        "rms_norm_grouped disagrees with the oracle"
    );
    anyhow::ensure!(
        ungrouped_gap > worst * 100.0,
        "grouping made no difference — the kernel may be ignoring group_size"
    );
    println!("GROUPED RMS NORM MATCHES THE ORACLE\n");

    hyper_connection_block(g)
}


// ── The whole hyper-connection block on GPU ─────────────────────────────────

/// Run Qwen4ExpTextGatedResidual end to end on the device and diff it against
/// the CPU oracle, which is itself checked against HuggingFace at 1.6e-7.
///
/// This is the first of the two novel blocks to exist as GPU work rather than
/// as a specification.
fn hyper_connection_block(g: &dyn GpuBackend) -> Result<()> {
    use spark_model::layers::ops;
    use spark_model::weight_map::DenseWeight;

    const LOWRANK: usize = 320; // published hc_lowrank
    let wide = HIDDEN * GROUPS;

    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16384.0 - 0.5
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
    let d_down = DenseWeight { weight: up_bf16(g, &mix_down)? };
    let d_up = DenseWeight { weight: up_bf16(g, &mix_up)? };
    let d_inject = DenseWeight { weight: up_bf16(g, &inject_w)? };

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

    ops::dense_gemv(g, gemv, d_normed, &d_down, d_lowrank, LOWRANK as u32, wide as u32, 0)?;
    KernelLaunch::new(g, act_k)
        .grid([1, 1, 1])
        .block([LOWRANK.min(1024) as u32, 1, 1])
        .arg_ptr(d_lowrank)
        .arg_u32(LOWRANK as u32)
        .arg_u32(GROUPS as u32)
        .launch(0)?;
    ops::dense_gemv(g, gemv, d_lowrank, &d_up, d_gate, wide as u32, LOWRANK as u32, 0)?;

    KernelLaunch::new(g, mix_k)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_gate)
        .arg_ptr(d_normed)
        .arg_ptr(d_mixed)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .launch(0)?;

    ops::dense_gemv(g, gemv, d_normed, &d_inject, d_raw_inject, GROUPS as u32, wide as u32, 0)?;
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
    anyhow::ensure!(inject_gap < 2e-2, "injection gains disagree with the oracle");
    // Guard against agreeing only because the sigmoid saturated.
    anyhow::ensure!(
        want.injection.iter().any(|v| (*v - 2.0).abs() > 0.05 && *v > 0.05),
        "injection gains are saturated -- this test would pass with the wrong sign"
    );
    println!("HYPER-CONNECTION BLOCK MATCHES THE ORACLE");
    Ok(())
}
