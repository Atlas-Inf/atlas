// SPDX-License-Identifier: AGPL-3.0-only

//! The BF16 half of `canonical_verify_gpu_tests.rs` (GPU, ignored): router,
//! KDA side projections and their planes launch, MLA W_uk / W_uv grouped.
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib canonical_dense -- --ignored --nocapture

use super::*;

/// The width-tuned BF16 GEMV the serial (and E9) router used for `m` rows,
/// as `(module, function)`: the C1 strict-order kernel at one row, the C3
/// rows kernel at three, M5 at five, `batchm` lane splits otherwise, tensor
/// cores above eight.
fn legacy_dense(m: u32) -> (&'static str, &'static str) {
    match m {
        1 => ("gemm", "dense_gemm_bf16_router"),
        3 => ("gemm", "dense_gemm_bf16_router_rows"),
        5 => ("gemm", "dense_gemm_bf16_router_m5"),
        2..=8 => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        9..=16 => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc16"),
        _ => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc32"),
    }
}

#[test]
#[ignore = "requires a GB10, a glm-5.3-flash kernel build and ATLAS_W4A16_TC=1"]
fn canonical_dense_rows_do_not_depend_on_the_width() -> Result<()> {
    ensure!(enabled(), "run with ATLAS_W4A16_TC=1 and the switch on");
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    let mut rng = Rng(0x5eed_1234_abcd_0002);
    let (mut failed, mut legacy_varies) = (0, 0);
    // Router (288 x 4096), KDA beta (32 x 4096), f_a/g_a (128 x 4096),
    // f_b/g_b (4096 x 128).
    for (n, k) in [(288, 4096), (32, 4096), (128, 4096), (4096, 128)] {
        let (w, _) = bf16_weight(gpu, &mut rng, n, k, 0.05)?;
        let probe = random_row(&mut rng, k);
        let canon = probe_bits(&mut rng, &probe, k, &WIDTHS, |rows| {
            run_rows(gpu, rows, n, |a, c, m| {
                dense(gpu, a, &w, c, m, n, k, n, gpu.default_stream())
            })
        })?;
        failed += differing(&format!("canonical dense {n}x{k}"), &canon);
        if n == 288 {
            let legacy = probe_bits(&mut rng, &probe, k, &WIDTHS, |rows| {
                let m = rows.len() as u32;
                run_rows(gpu, rows, n, |a, c, m2| {
                    launch_dense(gpu, legacy_dense(m), &w, a, c, m2, n, k)
                })
            })?;
            legacy_varies += differing("legacy router 288x4096", &legacy);
        }
    }
    // KDA beta | f_a | g_a and f_b | g_b planes: each plane as its own launch.
    let (beta, _) = bf16_weight(gpu, &mut rng, 32, 4096, 0.05)?;
    let (fa, _) = bf16_weight(gpu, &mut rng, 128, 4096, 0.05)?;
    let (fb, _) = bf16_weight(gpu, &mut rng, 4096, 128, 0.05)?;
    for m in WIDTHS {
        let x = upload(
            gpu,
            &rows_bytes(
                &(0..m)
                    .map(|_| random_row(&mut rng, 4096))
                    .collect::<Vec<_>>(),
            ),
        )?;
        let y = upload(
            gpu,
            &rows_bytes(
                &(0..m)
                    .map(|_| random_row(&mut rng, 128))
                    .collect::<Vec<_>>(),
            ),
        )?;
        let outs: Vec<DevicePtr> = (0..5)
            .map(|_| gpu.alloc(m as usize * 4096 * 2))
            .collect::<Result<_>>()?;
        let s = gpu.default_stream();
        let planes3 = [
            (x, &beta, outs[0], 32, 4096),
            (x, &fa, outs[1], 128, 4096),
            (y, &fb, outs[2], 4096, 128),
        ];
        dense_planes(gpu, &planes3, m, s)?;
        dense_planes(gpu, &planes3[2..], m, s)?;
        gpu.synchronize(s)?;
        for (i, &(a, w, c, n, k)) in planes3.iter().enumerate() {
            let fused = download(gpu, c, (m * n) as usize)?;
            dense(gpu, a, w, outs[3], m, n, k, n, s)?;
            gpu.synchronize(s)?;
            let alone = download(gpu, outs[3], (m * n) as usize)?;
            if fused != alone {
                println!("DIFFERS planes m={m} plane {i}");
                failed += 1;
            }
        }
        for p in outs.into_iter().chain([x, y]) {
            gpu.free(p)?;
        }
    }
    println!("planes vs plain: checked widths {WIDTHS:?}");
    // MLA W_uk absorb (heads x [512, 256]) and W_uv extract (heads x
    // [256, 512]), 8 heads, rows `heads * k` apart.
    let g = 8u32;
    for (n, k) in [(512u32, 256u32), (256, 512)] {
        let (w, _) = bf16_weight(gpu, &mut rng, g * n, k, 0.05)?;
        let probe = random_row(&mut rng, g * k);
        let canon = probe_bits(&mut rng, &probe, g * k, &WIDTHS, |rows| {
            run_rows(gpu, rows, g * n, |a, c, m| {
                dense_grouped(
                    gpu,
                    a,
                    w.weight,
                    c,
                    [m, g, k, n, g * k, g * n],
                    gpu.default_stream(),
                )
            })
        })?;
        failed += differing(&format!("canonical grouped {g}x{n}x{k}"), &canon);
    }
    ensure!(
        legacy_varies > 0,
        "the width-tuned router showed no variation: test is blind"
    );
    ensure!(failed == 0, "{failed} canonical BF16 cases differ");
    Ok(())
}
