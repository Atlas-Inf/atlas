// SPDX-License-Identifier: AGPL-3.0-only

//! Dependent launches through the new canonical BF16 grouped/planes tiers.
//! Cover every tier and the 26-row prefill shape; only the reference inserts
//! host barriers.

use super::*;

const STEPS: usize = 48;

fn compare(mut run: impl FnMut(bool) -> Result<Vec<u16>>, label: &str) -> Result<()> {
    let reference = run(true)?;
    for trial in 0..8 {
        ensure!(
            run(false)? == reference,
            "{label} differs without host barriers, trial={trial}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires GB10 GLM kernels, ATLAS_PDL=1 and ATLAS_W4A16_TC=1"]
fn canonical_dense_chains_need_no_host_barriers() -> Result<()> {
    ensure!(enabled(), "requires canonical verify and ATLAS_W4A16_TC=1");
    let gpu = backend()?;
    ensure!(
        spark_runtime::cuda_backend::pdl_enabled(),
        "requires ATLAS_PDL=1"
    );
    // Resolve every new BF16 entry up front; loading a module or resolving
    // a function must not insert a hidden barrier into the async chains.
    for rows in [1, 8, 13, 26] {
        for suffix in ["", "_grouped", "_planes"] {
            ensure!(
                dense_tier(gpu, rows, suffix).0 != 0,
                "missing dense tier {suffix}"
            );
        }
        grouped_chain(gpu, rows)?;
        planes_chain(gpu, rows)?;
    }
    Ok(())
}

fn grouped_chain(gpu: &dyn GpuBackend, rows: u32) -> Result<()> {
    let (g, k, n) = (32u32, 256u32, 512u32);
    let h = g * k;
    let mut rng = Rng(0x26ab_0123_8899_4455);
    let (uk, _) = bf16_weight(gpu, &mut rng, g * n, k, 0.05)?;
    let (uv, _) = bf16_weight(gpu, &mut rng, g * k, n, 0.05)?;
    let norm_w = DenseWeight {
        weight: upload(gpu, &rows_bytes(&[vec![bf16(1.0); h as usize]]))?,
    };
    let input = rows_bytes(
        &(0..rows)
            .map(|_| random_row(&mut rng, h))
            .collect::<Vec<_>>(),
    );
    let bytes = (rows * h) as usize * 2;
    let hidden = gpu.alloc(bytes)?;
    let normed = gpu.alloc(bytes)?;
    let absorbed = gpu.alloc((rows * g * n) as usize * 2)?;
    let extracted = gpu.alloc(bytes)?;
    let norm = gpu.kernel("norm", "rms_norm")?;
    let add = gpu.kernel("residual_add", "bf16_residual_add")?;
    let stream = gpu.default_stream();
    compare(
        |barriers| {
            for (p, size) in [
                (normed, bytes),
                (absorbed, (rows * g * n) as usize * 2),
                (extracted, bytes),
            ] {
                gpu.memset(p, 0xff, size)?;
            }
            gpu.copy_h2d(&input, hidden)?;
            let barrier = || {
                if barriers {
                    gpu.synchronize(stream)
                } else {
                    Ok(())
                }
            };
            for _ in 0..STEPS {
                ops::rms_norm(gpu, norm, hidden, &norm_w, normed, rows, h, 1e-5, stream)?;
                barrier()?;
                dense_grouped(
                    gpu,
                    normed,
                    uk.weight,
                    absorbed,
                    [rows, g, k, n, h, g * n],
                    stream,
                )?;
                barrier()?;
                dense_grouped(
                    gpu,
                    absorbed,
                    uv.weight,
                    extracted,
                    [rows, g, n, k, g * n, h],
                    stream,
                )?;
                barrier()?;
                ops::residual_add(gpu, add, hidden, extracted, rows * h, stream)?;
                barrier()?;
            }
            gpu.synchronize(stream)?;
            download(gpu, hidden, (rows * h) as usize)
        },
        &format!("MLA grouped absorb/extract rows={rows}"),
    )?;
    for p in [
        hidden,
        normed,
        absorbed,
        extracted,
        uk.weight,
        uv.weight,
        norm_w.weight,
    ] {
        gpu.free(p)?;
    }
    Ok(())
}

fn planes_chain(gpu: &dyn GpuBackend, rows: u32) -> Result<()> {
    let (h, heads, dim) = (4096u32, 32u32, 128u32);
    let mut rng = Rng(0x26ab_0199_7788_4422);
    let (beta_w, _) = bf16_weight(gpu, &mut rng, heads, h, 0.05)?;
    let (fa_w, _) = bf16_weight(gpu, &mut rng, dim, h, 0.05)?;
    let (ga_w, _) = bf16_weight(gpu, &mut rng, dim, h, 0.05)?;
    let (fb_w, _) = bf16_weight(gpu, &mut rng, h, dim, 0.05)?;
    let (gb_w, _) = bf16_weight(gpu, &mut rng, h, dim, 0.05)?;
    let norm_w = DenseWeight {
        weight: upload(gpu, &rows_bytes(&[vec![bf16(1.0); h as usize]]))?,
    };
    let input = rows_bytes(
        &(0..rows)
            .map(|_| random_row(&mut rng, h))
            .collect::<Vec<_>>(),
    );
    let bytes = (rows * h) as usize * 2;
    let hidden = gpu.alloc(bytes)?;
    let normed = gpu.alloc(bytes)?;
    // The contiguous beta/f_a/g_a layout used by KDA.
    let side = gpu.alloc((rows * (heads + 2 * dim)) as usize * 2)?;
    let fa = side.offset((rows * heads) as usize * 2);
    let ga = fa.offset((rows * dim) as usize * 2);
    let fb = gpu.alloc(bytes)?;
    let gb = gpu.alloc(bytes)?;
    let norm = gpu.kernel("norm", "rms_norm")?;
    let add = gpu.kernel("residual_add", "bf16_residual_add")?;
    let stream = gpu.default_stream();
    compare(
        |barriers| {
            for (p, size) in [
                (normed, bytes),
                (side, (rows * (heads + 2 * dim)) as usize * 2),
                (fb, bytes),
                (gb, bytes),
            ] {
                gpu.memset(p, 0xff, size)?;
            }
            gpu.copy_h2d(&input, hidden)?;
            let barrier = || {
                if barriers {
                    gpu.synchronize(stream)
                } else {
                    Ok(())
                }
            };
            for _ in 0..STEPS {
                ops::rms_norm(gpu, norm, hidden, &norm_w, normed, rows, h, 1e-5, stream)?;
                barrier()?;
                dense(gpu, normed, &fa_w, fa, rows, dim, h, dim, stream)?;
                barrier()?;
                dense(gpu, fa, &fb_w, fb, rows, h, dim, h, stream)?;
                barrier()?;
                ops::residual_add(gpu, add, hidden, fb, rows * h, stream)?;
                barrier()?;
                dense_planes(
                    gpu,
                    &[
                        (normed, &beta_w, side, heads, h),
                        (normed, &fa_w, fa, dim, h),
                        (normed, &ga_w, ga, dim, h),
                    ],
                    rows,
                    stream,
                )?;
                barrier()?;
                dense_planes(
                    gpu,
                    &[(fa, &fb_w, fb, h, dim), (ga, &gb_w, gb, h, dim)],
                    rows,
                    stream,
                )?;
                barrier()?;
                ops::residual_add(gpu, add, hidden, fb, rows * h, stream)?;
                barrier()?;
                ops::residual_add(gpu, add, hidden, gb, rows * h, stream)?;
                barrier()?;
            }
            gpu.synchronize(stream)?;
            let mut out = download(gpu, hidden, (rows * h) as usize)?;
            out.extend(download(gpu, side, (rows * (heads + 2 * dim)) as usize)?);
            Ok(out)
        },
        &format!("KDA side projection planes rows={rows}"),
    )?;
    for p in [
        hidden,
        normed,
        side,
        fb,
        gb,
        norm_w.weight,
        beta_w.weight,
        fa_w.weight,
        ga_w.weight,
        fb_w.weight,
        gb_w.weight,
    ] {
        gpu.free(p)?;
    }
    Ok(())
}
