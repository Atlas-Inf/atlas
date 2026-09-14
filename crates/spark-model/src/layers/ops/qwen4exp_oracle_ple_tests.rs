// SPDX-License-Identifier: AGPL-3.0-only

//! PLE half of the qwen4_exp oracle parity gate. Split out for the 500-LoC
//! cap; `super` is `qwen4exp_oracle_tests`, so the fixture RNG, the BF16
//! rounding, the device helpers and the `check` / `check_control` pair all
//! come from there and cannot drift between the two halves.

use spark_runtime::gpu::GpuBackend;

use super::*;
use crate::layers::ops;

/// The PLE tower — `#753` called this the port's top correctness risk, and the
/// three details that make it one are all invisible when wrong:
///
///   1. all three norms are offset-from-1 (`normed * (1 + w)`), initialised to
///      zeros, and the checkpoint's `ple.norm_key` centres at -0.1067;
///   2. they are GROUPED at `group_size = hidden`, so the four streams
///      normalise independently inside the 10240-wide vector;
///   3. the gate takes a SIGNED SQUARE ROOT of the dot product.
///
/// Each one leaves a finite, plausible gate when dropped. This runs the whole
/// kernel chain — `ple_gate` -> `ple_conv` -> `ple_add_highway` — against
/// `ple_forward`, which is the oracle's own composite rather than a second
/// transcription of the formula in this file.
#[test]
#[ignore]
fn qwen4exp_oracle_ple_matches_the_cpu_reference() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_gate = g.kernel("ple", "ple_gate").unwrap();
    let k_conv = g.kernel("ple", "ple_conv").unwrap();
    let k_add = g.kernel("ple", "ple_add_highway").unwrap();
    for (name, k) in [
        ("ple_gate", k_gate),
        ("ple_conv", k_conv),
        ("ple_add_highway", k_add),
    ] {
        assert!(k.0 != 0, "{name} resolved to handle 0");
    }

    let wide = HC * HIDDEN;
    // The conv is dilated by `ngram_size`, so its state is (K-1)*D = 9 steps,
    // not K-1 = 3. Sizing it as an undilated conv reads the wrong four
    // timesteps and still produces numbers.
    let kernel_size = 4usize;
    let dilation = 3usize;
    let state_len = (kernel_size - 1) * dilation;
    // `ple_embed_dim` only enters the two projections, which this test does on
    // the host because the kernels take their OUTPUTS. Keeping it at 512
    // instead of the checkpoint's 2560 shrinks a [10240, e] host matmul
    // without changing a single kernel path.
    let embed_dim = 512usize;
    let tokens = 24usize;

    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let dims = atlas_core::qwen4exp_reference::PleDims {
        hidden: HIDDEN,
        hc_count: HC,
        ple_embed_dim: embed_dim,
        kernel: kernel_size,
        dilation,
        eps: EPS,
    };

    // Offset-from-1 norms, centred where the real tensors are.
    let norm_key: Vec<f32> = bf16_round(
        &rng.vec(wide, 0.3)
            .iter()
            .map(|v| v - 0.1067)
            .collect::<Vec<_>>(),
    );
    let norm_query: Vec<f32> = bf16_round(
        &rng.vec(wide, 0.3)
            .iter()
            .map(|v| v - 0.05)
            .collect::<Vec<_>>(),
    );
    let norm_conv: Vec<f32> = bf16_round(
        &rng.vec(wide, 0.3)
            .iter()
            .map(|v| v - 0.05)
            .collect::<Vec<_>>(),
    );
    let key_proj = rng.vec(wide * embed_dim, (embed_dim as f32).sqrt().recip());
    let value_proj = rng.vec(HIDDEN * embed_dim, (embed_dim as f32).sqrt().recip());
    let conv1d = bf16_round(&rng.vec(wide * kernel_size, 0.5));

    let embeddings = rng.vec(tokens * embed_dim, 1.0);
    // The highway is FP32.
    let hidden_host = rng.vec(tokens * wide, 1.0);

    // The kernels take the PROJECTION OUTPUTS, in BF16 — `ple.cu` keeps only
    // key/value in BF16 (they are read once and not cancelled against
    // anything) and runs everything else in FP32. So project on the host with
    // the oracle's own `linear`, and round exactly what the kernel will read.
    let mut key_host = Vec::with_capacity(tokens * wide);
    let mut value_host = Vec::with_capacity(tokens * HIDDEN);
    for t in 0..tokens {
        let emb = &embeddings[t * embed_dim..(t + 1) * embed_dim];
        key_host.extend(atlas_core::qwen4exp_reference::linear(
            emb, &key_proj, wide, embed_dim,
        ));
        value_host.extend(atlas_core::qwen4exp_reference::linear(
            emb,
            &value_proj,
            HIDDEN,
            embed_dim,
        ));
    }
    let key_bf16 = bf16_round(&key_host);
    let value_bf16 = bf16_round(&value_host);

    let d_hidden = upload(g, &f32_bytes(&hidden_host));
    let d_key = upload(g, &bf16_bytes(&key_bf16));
    let d_value = upload(g, &bf16_bytes(&value_bf16));
    let d_nq = upload(g, &bf16_bytes(&norm_query));
    let d_nk = upload(g, &bf16_bytes(&norm_key));
    let d_nc = upload(g, &bf16_bytes(&norm_conv));
    let d_conv_w = upload(g, &bf16_bytes(&conv1d));
    let gated = g.alloc(tokens * wide * 4).unwrap();
    let gated_normed = g.alloc(tokens * wide * 4).unwrap();
    let ple_out = g.alloc(tokens * wide * 4).unwrap();
    // Fresh sequence: a ZERO state is what makes the kernel's left context
    // agree with the oracle's zero padding.
    let state = g.alloc(state_len * wide * 4).unwrap();
    g.copy_h2d_async(&f32_bytes(&vec![0f32; state_len * wide]), state, stream)
        .unwrap();

    ops::ple_gate(
        g,
        k_gate,
        d_hidden,
        d_key,
        d_value,
        d_nq,
        d_nk,
        d_nc,
        gated,
        gated_normed,
        tokens as u32,
        HIDDEN as u32,
        HC as u32,
        EPS,
        stream,
    )
    .unwrap();
    ops::ple_conv(
        g,
        k_conv,
        gated_normed,
        gated,
        d_conv_w,
        state,
        ple_out,
        tokens as u32,
        wide as u32,
        kernel_size as u32,
        dilation as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    // The oracle projects internally in FP32, so its key/value are a BF16 ulp
    // sharper than what the kernel read. That is ~0.4% relative on one input
    // to a 2560-term dot product, well inside a tolerance set at 5% of the
    // reference's RMS — and it is the honest direction to err, since the
    // kernel is being held to the sharper number.
    let want = atlas_core::qwen4exp_reference::ple_forward(
        &dims,
        &atlas_core::qwen4exp_reference::PleWeights {
            key_proj: &key_proj,
            value_proj: &value_proj,
            norm_key: &norm_key,
            norm_query: &norm_query,
            norm_conv: &norm_conv,
            conv1d: &conv1d,
        },
        &embeddings,
        &hidden_host,
    );
    let got = download_f32(g, ple_out, tokens * wide);
    println!("PLE tower (gate + dilated conv):");
    check("ple_out", &got, &want);

    // CONTROL 1: the conv must actually contribute. Compare against the gated
    // value alone — that is what the output would be if every tap were zero.
    let gated_only = download_f32(g, gated, tokens * wide);
    check_control("control: conv≠0", &gated_only, &want);

    // CONTROL 2: dilation. Re-run the conv with dilation 1 and require a wide
    // miss — a kernel that silently treated the dilation as 1 would read
    // timesteps t-3..t instead of t-9, t-6, t-3, t, and without this arm the
    // check above would still pass.
    g.copy_h2d_async(&f32_bytes(&vec![0f32; state_len * wide]), state, stream)
        .unwrap();
    let undilated = g.alloc(tokens * wide * 4).unwrap();
    ops::ple_conv(
        g,
        k_conv,
        gated_normed,
        gated,
        d_conv_w,
        state,
        undilated,
        tokens as u32,
        wide as u32,
        kernel_size as u32,
        1,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    check_control(
        "control: dilation=1",
        &download_f32(g, undilated, tokens * wide),
        &want,
    );

    // ── ple_add_highway: FP32 accumulate onto the highway, in place.
    ops::ple_add_highway(g, k_add, ple_out, d_hidden, (tokens * wide) as u32, stream).unwrap();
    g.synchronize(stream).unwrap();
    let want_highway: Vec<f32> = hidden_host.iter().zip(&want).map(|(h, p)| h + p).collect();
    println!("ple_add_highway:");
    check(
        "highway",
        &download_f32(g, d_hidden, tokens * wide),
        &want_highway,
    );
}
