// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`) of the KDA MXFP8 decode twins and W8A8 prefill
//! operands on the real checkpoint (`ATLAS_GLM_FP4L_CKPT`), split from
//! `fp4l_test_gpu.rs` for the file-size cap.

use super::*;

/// `ATLAS_GLM_FP4L_KDA_MXFP8_DECODE`: each rank's KDA q (rows) and o
/// (columns) twin is the checkpoint's bytes, and its GEMV rows agree bit for
/// bit at every row count 1..=32 (what keeps a DFlash verify row equal to the
/// same row decoded), within MXFP8's BF16-activation error of an f64
/// reference.
#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn kda_mxfp8_decode_twins_are_checkpoint_bytes_and_row_invariant() -> Result<()> {
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let tiers = ["mxfp8_gemv_tc8", "mxfp8_gemv_tc16", "mxfp8_gemv_tc32"]
        .map(|name| gpu.kernel("mxfp8_gemv", name).unwrap());
    let stems = [l(0, "self_attn.q_proj"), l(0, "self_attn.o_proj")];
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    let policy = KdaMxPolicy {
        decode: true,
        w8a8_prefill: false,
    };
    for rank in 0..2 {
        for (stem, kind) in [
            (&stems[0], crate::tp_shard::TpShardKind::ColumnParallel),
            (&stems[1], crate::tp_shard::TpShardKind::RowParallel),
        ] {
            let shape = store.get(&format!("{stem}.weight"))?.shape.clone();
            let twin = kda_twin(
                policy,
                &store,
                stem,
                [shape[0], shape[1]],
                kind,
                rank,
                2,
                gpu,
            )?
            .and_then(|m| m.decode)
            .context("twin")?;
            ensure!(
                !twin.tiled,
                "ATLAS_GLM_MXFP8_TILED must be unset for the byte check"
            );
            let (_, ws, w) = &host[&format!("{stem}.weight")];
            let (_, _, s) = &host[&format!("{stem}.weight_scale")];
            let (n, k) = (ws[0], ws[1]);
            let column = kind == crate::tp_shard::TpShardKind::ColumnParallel;
            let (ln, lk) = if column { (n / 2, k) } else { (n, k / 2) };
            let (r0, c0) = if column {
                (rank * ln, 0)
            } else {
                (0, rank * lk)
            };
            let want_w: Vec<u8> = (0..ln)
                .flat_map(|r| w[(r0 + r) * k + c0..(r0 + r) * k + c0 + lk].to_vec())
                .collect();
            let want_s: Vec<u8> = (0..ln)
                .flat_map(|r| s[((r0 + r) * k + c0) / 32..((r0 + r) * k + c0 + lk) / 32].to_vec())
                .collect();
            let (mut gw, mut gs) = (vec![0u8; ln * lk], vec![0u8; ln * lk / 32]);
            gpu.copy_d2h(twin.data, &mut gw)?;
            gpu.copy_d2h(twin.scales, &mut gs)?;
            ensure!(
                gw == want_w && gs == want_s,
                "rank {rank} {stem}: twin != checkpoint slice"
            );
            // Row invariance: rows of every width equal the same rows at 32.
            let mut x = Vec::with_capacity(32 * lk);
            let mut seed = 0x9e37_79b9_u64 ^ (rank as u64);
            for _ in 0..32 * lk {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                x.push(f32_to_bf16(
                    ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5,
                ));
            }
            let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let input = gpu.alloc(xb.len())?;
            gpu.copy_h2d(&xb, input)?;
            let out = gpu.alloc(32 * ln * 2)?;
            crate::layers::ops::mxfp8_gemv_rows(
                gpu, &tiers, &twin, input, out, 32, ln, lk, ln, stream,
            )?;
            gpu.synchronize(stream)?;
            let full = bf16s(gpu, out, 32 * ln)?;
            for m in [1usize, 2, 5, 8, 9, 17, 31] {
                crate::layers::ops::mxfp8_gemv_rows(
                    gpu, &tiers, &twin, input, out, m, ln, lk, ln, stream,
                )?;
                gpu.synchronize(stream)?;
                ensure!(
                    bf16s(gpu, out, m * ln)? == full[..m * ln],
                    "{stem}: {m} rows differ"
                );
            }
            // Row 0 against an f64 reference on the exact weights.
            let wf = mxfp8_to_bf16_host(&want_w, &want_s, ln, lk)?;
            let mut worst = 0f64;
            for c in 0..ln {
                let r: f64 = (0..lk)
                    .map(|j| f(x[j]) as f64 * f(wf[c * lk + j]) as f64)
                    .sum();
                let scale: f64 = (0..lk)
                    .map(|j| (f(x[j]) as f64 * f(wf[c * lk + j]) as f64).abs())
                    .sum();
                worst = worst.max((f(full[c]) as f64 - r).abs() / scale.max(1e-30));
            }
            ensure!(worst < 1e-2, "{stem}: row 0 relative error {worst}");
            for p in [input, out, twin.data, twin.scales] {
                gpu.free(p)?;
            }
            eprintln!(
                "rank {rank} {stem}: twin == checkpoint, rows 1..32 invariant, err {worst:.2e}"
            );
        }
    }
    Ok(())
}

/// `ATLAS_GLM_FP4L_KDA_W8A8_PREFILL`: the W8A8 prefill GEMM on layer 0's
/// rank-0 q twin gives each row the same bytes at every row count (1 ..
/// 2048), within the MXFP8-activation error (~3e-2) of an f64 reference on
/// the exact weights.
#[test]
#[ignore = "needs a GB10, cuBLASLt with MXFP8, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn kda_w8a8_prefill_is_row_invariant_on_the_checkpoint_twin() -> Result<()> {
    use crate::layers::ops::{
        MX_QUANT_KERNEL, MXFP8_ACT_MODULE, MxAct, mx_lt_scale_bytes, mx_quant_rows,
    };
    use crate::weight_loader::glm5::kda_mx::{KdaMxPolicy, kda_twin};
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let stem = l(0, "self_attn.q_proj");
    let (store, host) = upload(
        gpu,
        &with_parts(std::slice::from_ref(&stem), &["weight", "weight_scale"]),
    )?;
    const M: usize = 2048;
    let policy = KdaMxPolicy {
        decode: false,
        w8a8_prefill: true,
    };
    let mx = kda_twin(
        policy,
        &store,
        &stem,
        [8192, 4096],
        crate::tp_shard::TpShardKind::ColumnParallel,
        0,
        2,
        gpu,
    )?
    .context("twin")?;
    ensure!(mx.decode.is_none(), "W8A8 alone keeps no decode twin");
    let w = mx.w8a8.context("W8A8 operand")?;
    let (n, k) = (w.n, w.k);
    let mut seed = 0x51_7cc1_b727_220a_u64;
    let x: Vec<u16> = (0..M * k)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            f32_to_bf16(((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = gpu.alloc(xb.len())?;
    gpu.copy_h2d(&xb, input)?;
    let act = MxAct {
        q: gpu.alloc(M * k)?,
        sf: gpu.alloc(mx_lt_scale_bytes(M, k))?,
    };
    let quant = gpu.kernel(MXFP8_ACT_MODULE, MX_QUANT_KERNEL)?;
    let out = gpu.alloc(M * n * 2)?;
    let run = |m: usize| -> Result<Vec<u16>> {
        gpu.memset(act.sf, 0, mx_lt_scale_bytes(M, k))?;
        mx_quant_rows(gpu, quant, input, act, 0..m, k, stream)?;
        w.gemm(act, out, m, stream)?;
        gpu.synchronize(stream)?;
        bf16s(gpu, out, m * n)
    };
    let full = run(M)?;
    for m in [1usize, 13, 64, 100, 777] {
        ensure!(
            run(m)? == full[..m * n],
            "{m} rows differ from the same rows at {M}"
        );
    }
    let (_, _, wb) = &host[&format!("{stem}.weight")];
    let (_, _, sb) = &host[&format!("{stem}.weight_scale")];
    let wf = mxfp8_to_bf16_host(&wb[..n * k], &sb[..n * k / 32], n, k)?;
    let mut worst = 0f64;
    for c in (0..n).step_by(97) {
        let terms = (0..k).map(|j| f(x[j]) as f64 * f(wf[c * k + j]) as f64);
        let (r, scale) = terms.fold((0f64, 0f64), |(s, a), t| (s + t, a + t.abs()));
        worst = worst.max((f(full[c]) as f64 - r).abs() / scale.max(1e-30));
    }
    ensure!(worst < 5e-2, "row 0 relative error {worst}");
    eprintln!("W8A8 [{n}, {k}]: rows invariant 1..{M}, row 0 error {worst:.2e}");
    for p in [input, act.q, act.sf, out, w.data, w.sf] {
        gpu.free(p)?;
    }
    Ok(())
}
