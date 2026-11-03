// SPDX-License-Identifier: AGPL-3.0-only
//! The correctness gate of `glm_fp4l_native_bench`.

use super::*;

/// Every check of the module docs; true when one failed.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    g: &dyn GpuBackend,
    nk: ops::NativeKernels,
    x: DevicePtr,
    x_host: &[f32],
    outs: &[DevicePtr],
    [kda, q_a, kv_a, q_b, mla_o, sh_down]: [&Weight; 6],
) -> Result<bool> {
    let mut fail = false;
    println!("== gate");
    // 1. FP8 block row invariance: rows at M == the same rows at 32.
    for (label, w) in [
        ("q_a", &q_a),
        ("q_b", &q_b),
        ("o", &mla_o),
        ("shared down", &sh_down),
    ] {
        let (n, k) = (w.n, w.k);
        ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[3], 32, n, k, n, 0)?;
        g.synchronize(0)?;
        let full = fetch(g, outs[3], 32 * n * 2)?;
        for m in ROWS {
            ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[2], m, n, k, n, 0)?;
            g.synchronize(0)?;
            let got = fetch(g, outs[2], m * n * 2)?;
            if got != full[..m * n * 2] {
                println!("  FAIL {label}: rows at M={m} differ from the same rows at 32");
                fail = true;
            }
        }
    }
    // 2. Multi == singles (KDA q/k/v MXFP8, MLA q_a + kv_a FP8 block).
    for m in [1usize, 9, 32] {
        let parts: Vec<MxPart> = (0..3)
            .map(|i| MxPart::mx(&kda.mx[i], outs[i], 4096))
            .collect();
        ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)?;
        g.synchronize(0)?;
        for i in 0..3 {
            let fused = fetch(g, outs[i], m * 4096 * 2)?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kda.mx[i],
                x,
                outs[3],
                m,
                4096,
                4096,
                4096,
                0,
            )?;
            g.synchronize(0)?;
            if fused != fetch(g, outs[3], m * 4096 * 2)? {
                println!("  FAIL KDA fused projection {i} at M={m} differs from its own launch");
                fail = true;
            }
        }
        let parts = [
            MxPart::fp8b(&q_a.fp8b[0], outs[0], 1536),
            MxPart::fp8b(&kv_a.fp8b[0], outs[1], 512),
        ];
        ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)?;
        g.synchronize(0)?;
        for (i, w) in [&q_a, &kv_a].into_iter().enumerate() {
            let fused = fetch(g, outs[i], m * w.n * 2)?;
            ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[3], m, w.n, 4096, w.n, 0)?;
            g.synchronize(0)?;
            if fused != fetch(g, outs[3], m * w.n * 2)? {
                println!("  FAIL MLA fused projection {i} at M={m} differs from its own launch");
                fail = true;
            }
        }
    }
    // 3. FP8 block vs the f64 host reference (q_b: K = 1536, first 256 rows).
    {
        let (m, w) = (4usize, &q_b);
        ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[0], x, outs[0], m, w.n, w.k, w.n, 0)?;
        g.synchronize(0)?;
        let got = bf16_to_f32(&fetch(g, outs[0], m * w.n * 2)?);
        let xb = bf16_to_f32(&bf16_bytes(&x_host[..m * w.k]));
        let (values, grid) = &w.host;
        let mut worst = 0f64;
        for i in 0..m {
            let mut ymax = 1e-30f64;
            let mut ys = Vec::with_capacity(256);
            for r in 0..256 {
                let mut acc = 0f64;
                for c in 0..w.k {
                    let s = grid[(r / 128) * (w.k / 128) + c / 128] as f64;
                    acc += xb[i * w.k + c] as f64 * e4m3(values[r * w.k + c]) * s;
                }
                ymax = ymax.max(acc.abs());
                ys.push(acc);
            }
            for (r, y) in ys.iter().enumerate() {
                worst = worst.max((got[i * w.n + r] as f64 - y).abs() / ymax);
            }
        }
        println!("  fp8b vs f64 reference (q_b, M=4, 256 rows): max |err| / |y|max = {worst:.2e}");
        if worst > 1e-2 {
            println!("  FAIL fp8b reference");
            fail = true;
        }
    }
    Ok(fail)
}
