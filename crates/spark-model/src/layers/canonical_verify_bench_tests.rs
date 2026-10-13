// SPDX-License-Identifier: AGPL-3.0-only

//! Microbench (GPU, ignored): the canonical verify kernels against the
//! width-tuned ones they replace, per verify width, on one rank's GLM shapes
//! with cold weights (eight copies of each weight in rotation, > L2). Prints
//! microseconds per launch and per layer group:
//! - KDA projections: q, k, v, o (W4A16 4096 x 4096 each) plus beta (32 x
//!   4096), f_a / g_a (128 x 4096), f_b / g_b (4096 x 128). Legacy rows: the
//!   scalar / batch2 / batch3 / tc8 W4A16 tiers and the fused batch-M
//!   triple / dual below nine rows, tensor cores above.
//! - MoE router (288 x 4096 BF16) and the TP-split shared expert (gate and up
//!   1024 x 4096, down 4096 x 1024).
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib canonical_verify_microbench -- --ignored --nocapture

use super::*;

const COPIES: usize = 8;
const ITERS: usize = 64;

/// Mean microseconds of `launch(copy)` over `ITERS` launches.
fn time(gpu: &dyn GpuBackend, launch: impl Fn(usize) -> Result<()>) -> Result<f64> {
    let s = gpu.default_stream();
    for i in 0..COPIES {
        launch(i)?;
    }
    gpu.synchronize(s)?;
    let t = std::time::Instant::now();
    for i in 0..ITERS {
        launch(i % COPIES)?;
    }
    gpu.synchronize(s)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn legacy_dense_name(m: u32) -> (&'static str, &'static str) {
    match m {
        1..=8 => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        9..=16 => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc16"),
        _ => ("dense_gemv_bf16_batchm", "dense_gemv_bf16_tc32"),
    }
}

/// The serial verify's router for `m` rows: the strict-order C1 kernel at
/// one row, `independent_router_logits` (batch-M) at 2..8, tensor cores above.
fn legacy_router_name(m: u32) -> (&'static str, &'static str) {
    match m {
        1 => ("gemm", "dense_gemm_bf16_router"),
        _ => legacy_dense_name(m),
    }
}

#[test]
#[ignore = "requires a GB10, a glm-5.3-flash kernel build and ATLAS_W4A16_TC=1"]
fn canonical_verify_microbench() -> Result<()> {
    ensure!(enabled(), "run with ATLAS_W4A16_TC=1 and the switch on");
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu);
    let mut rng = Rng(0x5eed_1234_abcd_0004);
    let w4s = |rng: &mut Rng, n, k| {
        (0..COPIES)
            .map(|_| w4(gpu, rng, n, k))
            .collect::<Result<Vec<_>>>()
    };
    let bf = |rng: &mut Rng, n, k| {
        (0..COPIES)
            .map(|_| bf16_weight(gpu, rng, n, k, 0.05).map(|w| w.0))
            .collect::<Result<Vec<_>>>()
    };
    let qkvo = w4s(&mut rng, 4096, 4096)?;
    let shared_gu = w4s(&mut rng, 1024, 4096)?;
    let shared_down = w4s(&mut rng, 4096, 1024)?;
    let router = bf(&mut rng, 288, 4096)?;
    let beta = bf(&mut rng, 32, 4096)?;
    let fa = bf(&mut rng, 128, 4096)?;
    let fb = bf(&mut rng, 4096, 128)?;
    let a = gpu.alloc(32 * 4096 * 2)?;
    gpu.memset(a, 0x3c, 32 * 4096 * 2)?;
    let c = gpu.alloc(3 * 32 * 4096 * 2)?;
    let s = gpu.default_stream();
    let canon_w4 = |w: &W4, m| w4a16(gpu, a, &w.q, c, m, w.n, w.k, s);
    let old_w4 = |w: &W4, m| launch_w4(gpu, legacy_w4_name(m), w, a, c, m);
    let canon_bf = |w: &DenseWeight, m, n, k| dense(gpu, a, w, c, m, n, k, n, s);
    let old_bf = |name, w: &DenseWeight, m, n, k| launch_dense(gpu, name, w, a, c, m, n, k);
    println!("MICROBENCH us/launch (cold weights)  [legacy -> canonical]");
    let (mut kda_delta, mut moe_delta) = (vec![], vec![]);
    for m in WIDTHS {
        let q = [
            time(gpu, |i| old_w4(&qkvo[i], m))?,
            time(gpu, |i| canon_w4(&qkvo[i], m))?,
        ];
        // KDA beta | f_a | g_a then f_b | g_b: production fuses each set into
        // one batch-M launch up to 8 rows (tensor cores per projection above);
        // canonical runs one planes launch each.
        let triple = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_triple_n")?;
        let dual = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_dual")?;
        let (c1, c2) = (c.offset(32 * 4096 * 2), c.offset(2 * 32 * 4096 * 2));
        let side = [
            time(gpu, |i| {
                if m <= 8 {
                    ops::dense_gemv_batchm_triple_n(
                        gpu,
                        triple,
                        a,
                        [&beta[i], &fa[i], &fa[(i + 1) % COPIES]],
                        [c, c1, c2],
                        m,
                        [32, 128],
                        4096,
                        s,
                    )?;
                    ops::dense_gemv_batchm_dual(
                        gpu,
                        dual,
                        [a, a],
                        [&fb[i], &fb[(i + 1) % COPIES]],
                        [c1, c2],
                        m,
                        4096,
                        128,
                        s,
                    )
                } else {
                    for (w, n, k) in [
                        (&beta[i], 32, 4096),
                        (&fa[i], 128, 4096),
                        (&fa[(i + 1) % COPIES], 128, 4096),
                        (&fb[i], 4096, 128),
                        (&fb[(i + 1) % COPIES], 4096, 128),
                    ] {
                        old_bf(legacy_dense_name(m), w, m, n, k)?;
                    }
                    Ok(())
                }
            })?,
            time(gpu, |i| {
                dense_planes(
                    gpu,
                    &[
                        (a, &beta[i], c, 32, 4096),
                        (a, &fa[i], c1, 128, 4096),
                        (a, &fa[(i + 1) % COPIES], c2, 128, 4096),
                    ],
                    m,
                    s,
                )?;
                dense_planes(
                    gpu,
                    &[
                        (a, &fb[i], c1, 4096, 128),
                        (a, &fb[(i + 1) % COPIES], c2, 4096, 128),
                    ],
                    m,
                    s,
                )
            })?,
        ];
        let r = [
            time(gpu, |i| {
                old_bf(legacy_router_name(m), &router[i], m, 288, 4096)
            })?,
            time(gpu, |i| canon_bf(&router[i], m, 288, 4096))?,
        ];
        let sh = |down: bool| -> [f64; 2] {
            let ws = if down { &shared_down } else { &shared_gu };
            [
                time(gpu, |i| old_w4(&ws[i], m)).unwrap_or(f64::NAN),
                time(gpu, |i| canon_w4(&ws[i], m)).unwrap_or(f64::NAN),
            ]
        };
        let (gu, dn) = (sh(false), sh(true));
        // Per layer: KDA = 4 W4A16 + the side projections; MoE = router +
        // shared gate + up + down.
        let kda = |j: usize| 4.0 * q[j] + side[j];
        let moe = |j: usize| r[j] + 2.0 * gu[j] + dn[j];
        println!(
            "MICROBENCH m={m:<2} w4a16 4096x4096 {:7.1} -> {:7.1} | KDA side 5 proj {:5.1} -> {:5.1} | router {:5.1} -> {:5.1} | shared g/u {:5.1} -> {:5.1} down {:5.1} -> {:5.1} | KDA layer {:7.1} -> {:7.1} | MoE layer {:6.1} -> {:6.1}",
            q[0],
            q[1],
            side[0],
            side[1],
            r[0],
            r[1],
            gu[0],
            gu[1],
            dn[0],
            dn[1],
            kda(0),
            kda(1),
            moe(0),
            moe(1)
        );
        kda_delta.push((m, kda(1) - kda(0)));
        moe_delta.push((m, moe(1) - moe(0)));
    }
    // GLM-5.3-Flash: 34 KDA layers, 44 MoE layers per rank.
    for ((m, k), (_, mo)) in kda_delta.iter().zip(&moe_delta) {
        println!(
            "MICROBENCH m={m:<2} step delta (34 KDA x {k:+.1} us + 44 MoE x {mo:+.1} us) = {:+.3} ms",
            (34.0 * k + 44.0 * mo) / 1000.0
        );
    }
    Ok(())
}
