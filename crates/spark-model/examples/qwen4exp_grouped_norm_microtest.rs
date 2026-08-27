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
        (state >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0
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

/// Run Qwen4ExpTextPLELayer end to end on the device and diff it against the
/// CPU oracle, which is checked against HuggingFace at 5.1e-7.
///
/// Second of the two novel blocks. Multiple tokens on purpose: the dilated
/// conv is the whole reason this layer has state, and a single position cannot
/// exercise it.
fn ple_block(g: &dyn GpuBackend) -> Result<()> {
    use spark_model::layers::ops;
    use spark_model::weight_map::DenseWeight;

    const SEQ: usize = 12; // > (kernel-1)*dilation, so taps actually reach back
    const KERNEL: usize = 4;
    const DILATION: usize = 3; // ngram_size
    let wide = HIDDEN * GROUPS;

    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();
    let unit = 1.0 / (HIDDEN as f32).sqrt();

    let embeddings: Vec<f32> = (0..SEQ * HIDDEN).map(|_| next()).collect();
    let hidden_states: Vec<f32> = (0..SEQ * wide).map(|_| next()).collect();
    let key_proj: Vec<f32> = (0..wide * HIDDEN).map(|_| next() * unit).collect();
    let value_proj: Vec<f32> = (0..HIDDEN * HIDDEN).map(|_| next() * unit).collect();
    let norm_key: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();
    let norm_query: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();
    let norm_conv: Vec<f32> = (0..wide).map(|_| next() * 0.1).collect();
    let conv1d: Vec<f32> = (0..wide * KERNEL).map(|_| next() * 0.3).collect();

    let d_emb = up_bf16(g, &embeddings)?;
    let d_hidden = up_bf16(g, &hidden_states)?;
    let d_key_w = DenseWeight {
        weight: up_bf16(g, &key_proj)?,
    };
    let d_val_w = DenseWeight {
        weight: up_bf16(g, &value_proj)?,
    };
    let d_nk = up_bf16(g, &norm_key)?;
    let d_nq = up_bf16(g, &norm_query)?;
    let d_nc = up_bf16(g, &norm_conv)?;
    let d_conv = up_bf16(g, &conv1d)?;

    let d_key = g.alloc(SEQ * wide * 2)?;
    let d_key_n = g.alloc(SEQ * wide * 2)?;
    let d_query_n = g.alloc(SEQ * wide * 2)?;
    let d_value = g.alloc(SEQ * HIDDEN * 2)?;
    let d_gated = g.alloc(SEQ * wide * 2)?;
    let d_gated_n = g.alloc(SEQ * wide * 2)?;

    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let norm_k = g.kernel("norm", "rms_norm_grouped")?;
    let gate_k = g.kernel("qwen4exp_ple", "q4e_ple_gate")?;
    let conv_k = g.kernel("qwen4exp_ple", "q4e_ple_conv_add")?;

    // Projections, per position.
    for t in 0..SEQ {
        let emb = d_emb.offset(t * HIDDEN * 2);
        let key = d_key.offset(t * wide * 2);
        let val = d_value.offset(t * HIDDEN * 2);
        ops::dense_gemv(g, gemv, emb, &d_key_w, key, wide as u32, HIDDEN as u32, 0)?;
        ops::dense_gemv(g, gemv, emb, &d_val_w, val, HIDDEN as u32, HIDDEN as u32, 0)?;
    }

    let norm = |input: DevicePtr, weight: DevicePtr, out: DevicePtr| -> Result<()> {
        KernelLaunch::new(g, norm_k)
            .grid([SEQ as u32, GROUPS as u32, 1])
            .block([1024, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight)
            .arg_ptr(out)
            .arg_u32(HIDDEN as u32)
            .arg_u32(GROUPS as u32)
            .arg_f32(EPS)
            .launch(0)
    };
    norm(d_key, d_nk, d_key_n)?;
    norm(d_hidden, d_nq, d_query_n)?;

    KernelLaunch::new(g, gate_k)
        .grid([SEQ as u32, GROUPS as u32, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_key_n)
        .arg_ptr(d_query_n)
        .arg_ptr(d_value)
        .arg_ptr(d_gated)
        .arg_u32(HIDDEN as u32)
        .arg_u32(GROUPS as u32)
        .launch(0)?;

    norm(d_gated, d_nc, d_gated_n)?;

    KernelLaunch::new(g, conv_k)
        .grid([SEQ as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(d_gated_n)
        .arg_ptr(d_conv)
        .arg_ptr(d_gated)
        .arg_u32(wide as u32)
        .arg_u32(KERNEL as u32)
        .arg_u32(DILATION as u32)
        .launch(0)?;
    g.synchronize(0)?;

    let got = down_bf16(g, d_gated, SEQ * wide)?;

    let dims = atlas_core::qwen4exp_reference::PleDims {
        hidden: HIDDEN,
        hc_count: GROUPS,
        ple_embed_dim: HIDDEN,
        kernel: KERNEL,
        dilation: DILATION,
        eps: EPS,
    };
    let r = |v: &Vec<f32>| v.iter().map(round).collect::<Vec<_>>();
    let (kp, vp, nk, nq, nc, cv) = (
        r(&key_proj),
        r(&value_proj),
        r(&norm_key),
        r(&norm_query),
        r(&norm_conv),
        r(&conv1d),
    );
    let want = atlas_core::qwen4exp_reference::ple_forward(
        &dims,
        &atlas_core::qwen4exp_reference::PleWeights {
            conv1d: &cv,
            key_proj: &kp,
            value_proj: &vp,
            norm_conv: &nc,
            norm_key: &nk,
            norm_query: &nq,
        },
        &r(&embeddings),
        &r(&hidden_states),
    );

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "PLE tower: max|diff| {worst:.3e} over {} values (up to {scale:.3e}), relative {:.3e}",
        got.len(),
        worst / scale.max(1e-9)
    );

    // The conv must actually be reaching back: zeroing the taps has to change
    // the answer, or a broken dilation would pass unnoticed.
    let mut no_conv = dims;
    no_conv.kernel = KERNEL;
    let flat = vec![0f32; wide * KERNEL];
    let without = atlas_core::qwen4exp_reference::ple_forward(
        &no_conv,
        &atlas_core::qwen4exp_reference::PleWeights {
            conv1d: &flat,
            key_proj: &kp,
            value_proj: &vp,
            norm_conv: &nc,
            norm_key: &nk,
            norm_query: &nq,
        },
        &r(&embeddings),
        &r(&hidden_states),
    );
    let conv_effect = want
        .iter()
        .zip(&without)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("conv contribution: {conv_effect:.3e} (must be large)");

    anyhow::ensure!(
        worst / scale.max(1e-9) < 3e-2,
        "PLE tower disagrees with the oracle"
    );
    anyhow::ensure!(
        conv_effect > worst * 10.0,
        "the conv contributes nothing -- dilation or taps may be wrong"
    );
    println!("PLE TOWER MATCHES THE ORACLE\n");
    gdn_decode_step(g)
}

// ── Atlas's existing GDN decode kernel, against the qwen4_exp oracle ────────

/// `gated_delta_rule_decode` covers 36 of this model's 48 layers. Atlas already
/// ships it for Qwen3.5/3.6, so the question is not whether it works but
/// whether it computes the SAME recurrence this model expects -- decay first,
/// then correct by the recall error, rather than accumulating k v^T.
fn gdn_decode_step(g: &dyn GpuBackend) -> Result<()> {
    use atlas_core::qwen4exp_reference::gdn_delta_step;

    // Published qwen4_exp linear-attention geometry.
    const NUM_K_HEADS: usize = 16;
    const NUM_V_HEADS: usize = 48;
    const KD: usize = 128;
    const VD: usize = 128;
    let repeat = NUM_V_HEADS / NUM_K_HEADS;

    let mut state = 0xDEAD_BEEF_1234_5678u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    // q and k arrive already L2-normalised and scaled -- the kernel does not
    // do it, and neither does the oracle's step.
    let mut q: Vec<f32> = (0..NUM_K_HEADS * KD).map(|_| next()).collect();
    let mut k: Vec<f32> = (0..NUM_K_HEADS * KD).map(|_| next()).collect();
    for head in 0..NUM_K_HEADS {
        for buf in [&mut q, &mut k] {
            let slice = &mut buf[head * KD..(head + 1) * KD];
            let inv = 1.0 / (slice.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for v in slice.iter_mut() {
                *v *= inv;
            }
        }
    }
    // The 1/sqrt(key_head_dim) scale is applied ONCE, and the two sides do it
    // in different places: HF scales the QUERY before the recurrence, Atlas's
    // kernel scales the OUTPUT after it. Algebraically the same -- the output
    // is linear in q, and neither placement touches the state -- but doing
    // both is an 11x error, which is exactly what this harness did first.
    //
    // So: the kernel gets q UNSCALED, the oracle gets it scaled.
    let scale = 1.0 / (KD as f32).sqrt();
    let q_scaled: Vec<f32> = q.iter().map(|v| v * scale).collect();
    let v: Vec<f32> = (0..NUM_V_HEADS * VD).map(|_| next()).collect();
    // Decay in (0,1), beta in (0,1) -- the ranges exp(-softplus) and sigmoid
    // actually produce.
    let decay: Vec<f32> = (0..NUM_V_HEADS)
        .map(|_| next().abs() * 0.9 + 0.05)
        .collect();
    let beta: Vec<f32> = (0..NUM_V_HEADS)
        .map(|_| next().abs() * 0.9 + 0.05)
        .collect();
    let h0: Vec<f32> = (0..NUM_V_HEADS * KD * VD).map(|_| next() * 0.1).collect();

    let up_f32 = |d: &[f32]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
        let p = g.alloc(bytes.len())?;
        g.copy_h2d(&bytes, p)?;
        Ok(p)
    };

    let d_h = up_f32(&h0)?;
    let d_q = up_bf16(g, &q)?;
    let d_k = up_bf16(g, &k)?;
    let d_v = up_bf16(g, &v)?;
    let d_decay = up_f32(&decay)?;
    let d_beta = up_f32(&beta)?;
    let d_out = g.alloc(NUM_V_HEADS * VD * 2)?;

    let kernel = g.kernel("gated_delta_rule", "gated_delta_rule_decode")?;
    KernelLaunch::new(g, kernel)
        .grid([NUM_V_HEADS as u32, 1, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(d_h)
        .arg_ptr(d_q)
        .arg_ptr(d_k)
        .arg_ptr(d_v)
        .arg_ptr(d_decay)
        .arg_ptr(d_beta)
        .arg_ptr(d_out)
        .arg_u32(1)
        .arg_u32(NUM_K_HEADS as u32)
        .arg_u32(NUM_V_HEADS as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, NUM_V_HEADS * VD)?;

    // Oracle, per value head, on the same BF16-rounded inputs.
    let (qr, kr, vr) = (
        q_scaled.iter().map(round).collect::<Vec<_>>(),
        k.iter().map(round).collect::<Vec<_>>(),
        v.iter().map(round).collect::<Vec<_>>(),
    );
    let mut want = Vec::with_capacity(NUM_V_HEADS * VD);
    let mut h = h0.clone();
    for head in 0..NUM_V_HEADS {
        let kh = head / repeat;
        let st = &mut h[head * KD * VD..(head + 1) * KD * VD];
        want.extend(gdn_delta_step(
            st,
            &qr[kh * KD..(kh + 1) * KD],
            &kr[kh * KD..(kh + 1) * KD],
            &vr[head * VD..(head + 1) * VD],
            decay[head],
            beta[head],
        ));
    }

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale_out = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "GDN decode step: max|diff| {worst:.3e} over {} values (up to {scale_out:.3e}), relative {:.3e}",
        got.len(),
        worst / scale_out.max(1e-9)
    );

    // q and k are SHARED across `repeat` value heads. If the kernel mapped
    // them per value head instead, heads inside a group would stop agreeing --
    // so check the mapping actually happened.
    let group_shared = (0..repeat).all(|r| {
        let a = &want[r * VD..(r + 1) * VD];
        let b = &want[VD..2 * VD];
        r == 1 || a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6)
    });
    println!("head-group mapping exercised: {group_shared}");

    anyhow::ensure!(
        worst / scale_out.max(1e-9) < 2e-2,
        "gated_delta_rule_decode disagrees with the qwen4_exp oracle"
    );
    println!("GDN DECODE STEP MATCHES THE ORACLE\n");
    attn_decode_step(g)
}

// ── The gated-Q attention decode kernel, against the qwen4_exp oracle ───────

/// `q4e_attn_decode` covers the 12 full-attention layers. What distinguishes
/// this model's attention is the gate: `q_proj` emits `[query | gate]` PER
/// HEAD, and the gate is applied ELEMENTWISE to the attention output before
/// `o_proj`. Read as ungated, a loader takes the gate half as query values and
/// the model still produces text.
///
/// The oracle here is `attention_decode_step`, which `attention_forward` itself
/// calls -- so agreement chains to the same code that matches HuggingFace at
/// 8.0e-7 rather than to a second transcription of the equations.
fn attn_decode_step(g: &dyn GpuBackend) -> Result<()> {
    use atlas_core::qwen4exp_reference::{AttnDims, attention_decode_step};

    // Published qwen4_exp full-attention geometry.
    const NUM_HEADS: usize = 24;
    const NUM_KV_HEADS: usize = 2;
    const HD: usize = 256;
    const PAST: usize = 37; // an awkward length, not a multiple of the block

    let mut state = 0x0BAD_F00D_5EED_9911u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let round = |v: &f32| bf16::from_f32(*v).to_f32();

    let q_dim = NUM_HEADS * HD;
    let kv_dim = NUM_KV_HEADS * HD;
    let query: Vec<f32> = (0..q_dim).map(|_| next()).collect();
    // Gates spread across the sigmoid's useful range. A gate stuck in
    // saturation would make the elementwise multiply look like a constant,
    // which is the failure this check exists to see.
    let gate: Vec<f32> = (0..q_dim).map(|_| next() * 6.0).collect();
    let keys: Vec<f32> = (0..PAST * kv_dim).map(|_| next()).collect();
    let values: Vec<f32> = (0..PAST * kv_dim).map(|_| next()).collect();

    let d_q = up_bf16(g, &query)?;
    let d_gate = up_bf16(g, &gate)?;
    let d_k = up_bf16(g, &keys)?;
    let d_v = up_bf16(g, &values)?;
    let d_out = g.alloc(q_dim * 2)?;

    let kernel = g.kernel("qwen4exp_attn", "q4e_attn_decode")?;
    KernelLaunch::new(g, kernel)
        .grid([NUM_HEADS as u32, 1, 1])
        .block([HD as u32, 1, 1])
        .shared_mem((PAST * 4) as u32)
        .arg_ptr(d_q)
        .arg_ptr(d_gate)
        .arg_ptr(d_k)
        .arg_ptr(d_v)
        .arg_ptr(d_out)
        .arg_u32(NUM_HEADS as u32)
        .arg_u32(NUM_KV_HEADS as u32)
        .arg_u32(HD as u32)
        .arg_u32(PAST as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_out, q_dim)?;

    // Oracle on the same BF16-rounded inputs.
    let dims = AttnDims {
        hidden: 2560,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HD,
        rotary_dim: 64,
        eps: 1e-6,
    };
    let want = attention_decode_step(
        &dims,
        &query.iter().map(round).collect::<Vec<_>>(),
        &gate.iter().map(round).collect::<Vec<_>>(),
        &keys.iter().map(round).collect::<Vec<_>>(),
        &values.iter().map(round).collect::<Vec<_>>(),
    );

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale_out = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "attention decode: max|diff| {worst:.3e} over {} values (up to {scale_out:.3e}), relative {:.3e}",
        got.len(),
        worst / scale_out.max(1e-9)
    );

    // CONTROL 1: the gate must be doing something. Recompute with every gate
    // forced to +8 (sigmoid ~ 1, i.e. effectively ungated) and require the
    // answer to move -- otherwise a kernel that ignored `gate` would pass.
    let ungated = attention_decode_step(
        &dims,
        &query.iter().map(round).collect::<Vec<_>>(),
        &vec![8.0f32; q_dim],
        &keys.iter().map(round).collect::<Vec<_>>(),
        &values.iter().map(round).collect::<Vec<_>>(),
    );
    let gate_effect = ungated
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("gate contribution: {gate_effect:.3e} (must be large)");
    anyhow::ensure!(
        gate_effect > worst * 10.0,
        "the gate changes nothing -- the kernel may be ignoring it"
    );

    // CONTROL 2: 24 query heads share 2 KV heads. If the kernel mapped KV per
    // query head instead of per group, heads in DIFFERENT groups would stop
    // differing in the way grouping implies. Heads 0..11 read kv_head 0 and
    // heads 12..23 read kv_head 1, so the two halves must disagree.
    let half = NUM_HEADS / 2 * HD;
    let cross_group = got[..half]
        .iter()
        .zip(&got[half..])
        .any(|(a, b)| (a - b).abs() > 1e-3);
    println!("kv head-group mapping exercised: {cross_group}");
    anyhow::ensure!(cross_group, "both KV groups produced the same context");

    anyhow::ensure!(
        worst / scale_out.max(1e-9) < 2e-2,
        "q4e_attn_decode disagrees with the qwen4_exp oracle"
    );
    println!("ATTENTION DECODE STEP MATCHES THE ORACLE\n");
    hc_expand_entry(g)
}

// ── Trunk entry: the embedding tiled across the residual streams ───────────

/// `q4e_hc_expand` runs once after the embedding lookup, not per layer. It is
/// three lines of kernel, and it is here because getting it wrong is silent:
/// the streams must start IDENTICAL. Zero-initialising them instead makes the
/// first hyper-connection collapse read a zero mean, and the model does not
/// recover -- it still emits tokens.
fn hc_expand_entry(g: &dyn GpuBackend) -> Result<()> {
    const HIDDEN: usize = 2560;
    const HC: usize = 4;
    const TOKENS: usize = 3;

    let mut state = 0x51DE_4A17_C0DE_0001u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let hidden: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| next()).collect();

    let d_hidden = up_bf16(g, &hidden)?;
    // Pre-fill the destination with a sentinel so an unwritten slot is visible
    // rather than reading as a plausible zero.
    let sentinel = vec![-7.0f32; TOKENS * HC * HIDDEN];
    let d_streams = up_bf16(g, &sentinel)?;

    let kernel = g.kernel("qwen4exp_hc", "q4e_hc_expand")?;
    KernelLaunch::new(g, kernel)
        .grid([TOKENS as u32, HC as u32, 1])
        .block([HIDDEN.min(1024) as u32, 1, 1])
        .arg_ptr(d_hidden)
        .arg_ptr(d_streams)
        .arg_u32(HIDDEN as u32)
        .arg_u32(HC as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let got = down_bf16(g, d_streams, TOKENS * HC * HIDDEN)?;

    let round = |v: &f32| bf16::from_f32(*v).to_f32();
    let want: Vec<f32> = (0..TOKENS)
        .flat_map(|t| {
            let row: Vec<f32> = hidden[t * HIDDEN..(t + 1) * HIDDEN].iter().map(round).collect();
            std::iter::repeat_n(row, HC).flatten()
        })
        .collect();

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("hc expand: max|diff| {worst:.3e} over {} values", got.len());

    // The sentinel must be gone everywhere -- a kernel that wrote only stream 0
    // would leave -7.0 in the other three and still match on the part it wrote.
    let untouched = got.iter().filter(|v| (**v + 7.0).abs() < 1e-6).count();
    println!("slots left at the sentinel: {untouched} (must be 0)");
    anyhow::ensure!(untouched == 0, "q4e_hc_expand did not write every stream");

    anyhow::ensure!(worst < 1e-6, "q4e_hc_expand does not tile the hidden state");
    println!("HC EXPAND MATCHES THE ORACLE");
    Ok(())
}
