// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE tower on GPU against the oracle, with its dilation control.

use super::*;

/// Run Qwen4ExpTextPLELayer end to end on the device and diff it against the
/// CPU oracle, which is checked against HuggingFace at 5.1e-7.
///
/// Second of the two novel blocks. Multiple tokens on purpose: the dilated
/// conv is the whole reason this layer has state, and a single position cannot
/// exercise it.
pub(super) fn ple_block(g: &dyn GpuBackend) -> Result<()> {
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
