// SPDX-License-Identifier: AGPL-3.0-only

//! The dense, fused KDA and W8A8 halves of `native_test_gpu.rs` (split for
//! the file-size cap): the same helpers, the same real checkpoint.

use super::*;

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn dense_layer_0_mxfp8_twins_expand_and_project_exactly() -> Result<()> {
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let p = l(0, "mlp");
    let stems: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|n| format!("{p}.{n}"))
        .collect();
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    // Decode twins only (the W8A8 operands need a tuned cuBLASLt; m_max 0).
    let native =
        crate::weight_loader::glm5::dense::native_dense(NATIVE, &store, &p, [4096, 12288], gpu)?
            .context("FP4-Local ships the dense MLP MXFP8")?;
    for (stem, twin) in stems.iter().zip([native.gate, native.up, native.down]) {
        let (_, ws, w) = &host[&format!("{stem}.weight")];
        let (_, _, s) = &host[&format!("{stem}.weight_scale")];
        let (n, k) = (ws[0], ws[1]);
        ensure!(
            bytes(gpu, twin.data, n * k)? == *w && bytes(gpu, twin.scales, n * k / 32)? == *s,
            "{stem}: twin != checkpoint"
        );
        // The exact transient expansion == the host's BF16, bit for bit.
        let out = gpu.alloc(n * k * 2)?;
        ops::mxfp8_expand_bf16(gpu, nk.mx_expand, &twin, out, n, k, stream)?;
        gpu.synchronize(stream)?;
        let want = mxfp8_to_bf16_host(w, s, n, k)?;
        ensure!(bf16s(gpu, out, n * k)? == want, "{stem}: expansion != host");
        gpu.free(out)?;
        // GEMV rows invariant up to 64; row 0 against the f64 reference.
        let (x, input) = activations(gpu, 64, k, 11)?;
        let y = gpu.alloc(64 * n * 2)?;
        let got = rows_invariant(gpu, y, n, &[1, 3, 8, 16, 32, 33, 64], &|m| {
            ops::mxfp8_gemv_rows(gpu, &nk.mx.row_major, &twin, input, y, m, n, k, n, stream)
        })?;
        let e = worst_rel(&got, &x, 0, n.min(512), k, |c, j| f(want[c * k + j]) as f64);
        ensure!(e < 1e-2, "{stem}: row 0 relative error {e}");
        gpu.free(input)?;
        gpu.free(y)?;
        eprintln!("{stem}: twin == checkpoint, expansion exact, rows invariant, err {e:.2e}");
    }
    native.free(gpu)
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn kda_layer_0_fused_qkv_equals_three_launches() -> Result<()> {
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let stems = ["q_proj", "k_proj", "v_proj"].map(|n| l(0, &format!("self_attn.{n}")));
    let (store, _) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    let policy = KdaMxPolicy {
        decode: true,
        w8a8_prefill: false,
    };
    for rank in 0..2 {
        let twins: Vec<Mxfp8Weight> = stems
            .iter()
            .map(|stem| {
                kda_twin(
                    policy,
                    &store,
                    stem,
                    [8192, 4096],
                    TpShardKind::ColumnParallel,
                    rank,
                    2,
                    gpu,
                )?
                .and_then(|m| m.decode)
                .context("twin")
            })
            .collect::<Result<_>>()?;
        let (_, x) = activations(gpu, 32, 4096, rank as u64)?;
        let outs = [0, 1, 2, 3].map(|_| gpu.alloc(32 * 4096 * 2).unwrap());
        for m in [1usize, 2, 5, 8, 9, 16, 17, 32] {
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::mx(&twins[i], outs[i], 4096))
                .collect();
            ops::mx_multi_rows(
                gpu,
                &nk.mx_multi,
                twins[0].tiled,
                x,
                &parts,
                m,
                4096,
                stream,
            )?;
            gpu.synchronize(stream)?;
            for i in 0..3 {
                let fused = bf16s(gpu, outs[i], m * 4096)?;
                ops::mxfp8_gemv_rows(
                    gpu,
                    &nk.mx.row_major,
                    &twins[i],
                    x,
                    outs[3],
                    m,
                    4096,
                    4096,
                    4096,
                    stream,
                )?;
                gpu.synchronize(stream)?;
                ensure!(
                    fused == bf16s(gpu, outs[3], m * 4096)?,
                    "rank {rank} {} at {m} rows",
                    stems[i]
                );
            }
        }
        eprintln!("rank {rank}: fused q/k/v == three launches at 1..32 rows");
        for p in outs.into_iter().chain([x]) {
            gpu.free(p)?;
        }
        for t in twins {
            gpu.free(t.data)?;
            gpu.free(t.scales)?;
        }
    }
    Ok(())
}

/// Every native W8A8 prefill path gives a row the same bytes at every pass
/// height and position: the pinned cuBLASLt MXFP8 kernel (KDA layer 0 q,
/// dense layer 0 gate and down) and the block-FP8 GEMM (MLA layer 3 o, shared
/// layer 3 gate). The first `h` rows at each height and a 30-row window
/// ending at row `h` computed alone equal the same rows of a 16384-row pass.
#[test]
#[ignore = "needs a GB10, cuBLASLt with MXFP8, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn native_w8a8_prefill_rows_are_height_invariant() -> Result<()> {
    use crate::layers::ops::{MxAct, mx_lt_scale_bytes, mx_quant_rows};
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    const HEIGHTS: [usize; 6] = [1, 30, 64, 2048, 8192, 16384];
    const TALL: usize = 16384;
    let gpu = native_backend()?;
    let stream = gpu.default_stream();
    let nk = native_kernels(gpu);
    let kda = l(0, "self_attn.q_proj");
    let dense = l(0, "mlp");
    let mla_o = l(3, "self_attn.o_proj");
    let shared = l(3, "mlp.shared_experts");
    let mut stems = vec![kda.clone(), mla_o.clone(), format!("{shared}.gate_proj")];
    stems.extend(["gate_proj", "up_proj", "down_proj"].map(|p| format!("{dense}.{p}")));
    let (store, _) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;

    // (label, n, k, run(x, out, rows)) for every path under test.
    let kda_w8a8 = kda_twin(
        KdaMxPolicy {
            decode: false,
            w8a8_prefill: true,
        },
        &store,
        &kda,
        [8192, 4096],
        TpShardKind::ColumnParallel,
        0,
        2,
        gpu,
    )?
    .and_then(|m| m.w8a8)
    .context("KDA W8A8 operand")?;
    let dense_native = crate::weight_loader::glm5::dense::native_dense(
        NATIVE,
        &store,
        &dense,
        [4096, 12288],
        gpu,
    )?
    .context("dense twins")?;
    let dense_ops = dense_native
        .w8a8
        .context("dense W8A8 operands (ATLAS_GLM_FP4L_FP8_PREFILL unset)")?;
    // MLA o: this rank's column slice of the checkpoint's FP8 block bytes.
    let raw = crate::weight_map::raw_fp8_block(&store, &mla_o)?.context("MLA o FP8 block")?;
    let (o_twin, on, ok) =
        crate::weight_map::fp8_block_twin(gpu, raw, TpShardKind::RowParallel, 0, 2, stream)?;
    let raw = crate::weight_map::raw_fp8_block(&store, &format!("{shared}.gate_proj"))?
        .context("shared gate")?;
    let (g_twin, gn, gk) =
        crate::weight_map::fp8_block_twin(gpu, raw, TpShardKind::Replicated, 0, 1, stream)?;

    let kmax = 12288;
    let quant = nk.mx_quant;
    let act = MxAct {
        q: gpu.alloc(TALL * kmax)?,
        sf: gpu.alloc(mx_lt_scale_bytes(TALL, kmax))?,
    };
    let scratch = ops::Fp8ActScratch {
        act: act.q,
        act_bytes: TALL * kmax,
        scale: act.sf,
        scale_bytes: TALL * (kmax / 128) * 4,
    };
    let mx = |w: crate::layers::ops::MxW8a8| {
        move |x: DevicePtr, out: DevicePtr, rows: usize| -> Result<()> {
            mx_quant_rows(gpu, quant, x, act, 0..rows, w.k, stream)?;
            w.gemm(act, out, rows, stream)
        }
    };
    let fb = |w: crate::weight_map::Fp8BlockWeight, n: usize, k: usize| {
        move |x: DevicePtr, out: DevicePtr, rows: usize| -> Result<()> {
            ops::fp8b_prefill_gemm(
                gpu,
                &nk,
                scratch,
                Fp8bPrefill::W8a8,
                &w,
                x,
                out,
                rows,
                n,
                k,
                stream,
            )
        }
    };
    type Run<'a> = Box<dyn Fn(DevicePtr, DevicePtr, usize) -> Result<()> + 'a>;
    let paths: Vec<(&str, usize, usize, Run)> = vec![
        (
            "KDA q MXFP8 W8A8 (pinned)",
            kda_w8a8.n,
            kda_w8a8.k,
            Box::new(mx(kda_w8a8)),
        ),
        (
            "dense gate MXFP8 W8A8 (pinned)",
            dense_ops[0].n,
            dense_ops[0].k,
            Box::new(mx(dense_ops[0])),
        ),
        (
            "dense down MXFP8 W8A8 (pinned)",
            dense_ops[2].n,
            dense_ops[2].k,
            Box::new(mx(dense_ops[2])),
        ),
        ("MLA o FP8 block W8A8", on, ok, Box::new(fb(o_twin, on, ok))),
        (
            "shared gate FP8 block W8A8",
            gn,
            gk,
            Box::new(fb(g_twin, gn, gk)),
        ),
    ];
    for (label, n, k, run) in &paths {
        let (_, x) = activations(gpu, TALL, *k, *n as u64)?;
        let (full, part) = (gpu.alloc(TALL * n * 2)?, gpu.alloc(TALL * n * 2)?);
        run(x, full, TALL)?;
        gpu.synchronize(stream)?;
        let want = bf16s(gpu, full, TALL * n)?;
        for h in HEIGHTS {
            run(x, part, h)?;
            gpu.synchronize(stream)?;
            ensure!(
                bf16s(gpu, part, h * n)? == want[..h * n],
                "{label}: the first {h} rows differ from the {TALL}-row pass"
            );
            let m = h.min(30);
            let r0 = h - m;
            run(x.offset(r0 * k * 2), part, m)?;
            gpu.synchronize(stream)?;
            ensure!(
                bf16s(gpu, part, m * n)? == want[r0 * n..h * n],
                "{label}: rows {r0}..{h} alone differ from the same rows of the {TALL}-row pass"
            );
        }
        eprintln!("{label}: rows invariant at heights {HEIGHTS:?} and alone");
        for p in [x, full, part] {
            gpu.free(p)?;
        }
    }
    drop(paths);
    gpu.free(act.q)?;
    gpu.free(act.sf)?;
    o_twin.free(gpu)?;
    g_twin.free(gpu)?;
    kda_w8a8.free(gpu)?;
    gpu.free(kda_w8a8.data)?;
    dense_native.free(gpu)
}
