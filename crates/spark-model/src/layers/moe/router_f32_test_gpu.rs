// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 and a glm-5.3-flash kernel build): GLM's
//! FP32 router logits (`moe_router_dtype: float32`) against a CPU reference
//! of the reference engine's routing -- BF16 inputs, an unrounded dot
//! product (f64 here), sigmoid, `+ e_score_correction_bias` for the top-k
//! with the lower index on a tie, the pre-bias scores of the winners
//! normalized and times `routed_scaling_factor`. Every FP32 router GEMM
//! (decode GEMV, batch-M, the canonical tensor-core tiers, the scalar,
//! register-blocked, rows, M5, BN4 and BN32 kernels, CUTLASS and cuBLASLt)
//! feeds the sigmoid top-k (single-token and batched) and must give the
//! reference's experts exactly and its weights to 2e-6; so must the fused
//! routing launches at every batch of 1..=32 rows. One row's two contending
//! experts round to the same BF16 logit: the BF16 path picks the lower
//! index, the FP32 path (and the reference) the larger logit.
//!
//!   cargo test --release -p spark-model --lib router_f32 -- --ignored --nocapture

use super::*;

#[path = "router_f32_gemms_test_gpu.rs"]
mod gemm_paths;

use gemm_paths::gemms;

/// The flip row's contenders: `A` (lower index, logit 2 + 2^-10) and `B`
/// (logit 2 + 2^-8), one BF16 word (2.0) apart from nothing; seven experts
/// lead both, so they contend for the eighth route.
const A: usize = 20;
const B: usize = 21;

/// A layer's router weight and bias, on the host and on the device.
struct Router {
    w: Vec<u16>,
    bias: Vec<f32>,
    gate: DevicePtr,
    bias_dev: DevicePtr,
}

impl Router {
    fn upload(gpu: &dyn GpuBackend, w: Vec<u16>, bias: Vec<f32>) -> Result<Self> {
        Ok(Self {
            gate: upload(gpu, &bytes_of(&w))?,
            bias_dev: upload(gpu, &bytes_of(&bias))?,
            w,
            bias,
        })
    }

    fn random(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        let w = (0..E * H).map(|_| bf16(rng.unit() * 0.03)).collect();
        let bias = (0..E).map(|_| rng.unit() * 0.01).collect();
        Self::upload(gpu, w, bias)
    }

    /// Row 0 of `x` (columns 0 and 1 = 1.0) gives experts 0..7 logit 4, `A`
    /// and `B` their contending logits and every other expert a distinct
    /// negative one; every sum is exact in FP32 whatever the order.
    fn flip(gpu: &dyn GpuBackend) -> Result<Self> {
        let mut w = vec![0u16; (E * H) as usize];
        for e in 0..E as usize {
            let (c0, c1) = match e {
                0..=6 => (4.0, 0.0),
                A => (2.0, 1.0 / 1024.0),
                B => (2.0, 1.0 / 256.0),
                _ => (-1.0 - e as f32 / 64.0, 0.0),
            };
            w[e * H as usize] = bf16(c0);
            w[e * H as usize + 1] = bf16(c1);
        }
        Self::upload(gpu, w, vec![0.0; E as usize])
    }

    /// The reference's logits (f64), experts and weights of each row of `x`.
    fn reference(&self, x: &[u16]) -> (Vec<f64>, Vec<u32>, Vec<f32>) {
        let f = |b: u16| f32::from_bits(u32::from(b) << 16) as f64;
        let rows = x.len() / H as usize;
        let (mut logits, mut ids, mut weights) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let xr = &x[r * H as usize..(r + 1) * H as usize];
            let row: Vec<f64> = (0..E as usize)
                .map(|e| {
                    let we = &self.w[e * H as usize..(e + 1) * H as usize];
                    xr.iter().zip(we).map(|(&a, &b)| f(a) * f(b)).sum()
                })
                .collect();
            let (top, w) = route(&row, &self.bias);
            logits.extend(row);
            ids.extend(top);
            weights.extend(w);
        }
        (logits, ids, weights)
    }
}

/// The reference's top-k of one row of logits: indices by descending
/// `sigmoid + bias`, the lower index first on a tie, and their normalized,
/// scaled pre-bias sigmoids.
fn route(logits: &[f64], bias: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let sig: Vec<f64> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    let sel = |e: usize| sig[e] + bias[e] as f64;
    order.sort_by(|&a, &b| sel(b).total_cmp(&sel(a)).then(a.cmp(&b)));
    let top = &order[..TOPK as usize];
    let sum: f64 = top.iter().map(|&e| sig[e]).sum();
    (
        top.iter().map(|&e| e as u32).collect(),
        top.iter()
            .map(|&e| (sig[e] / sum * SCALING as f64) as f32)
            .collect(),
    )
}

/// Whether every row's top `TOPK + 1` selection scores are 1e-4 apart, so
/// FP32 accumulation order cannot reorder them (the flip row's are exact).
fn separated(router: &Router, x: &[u16]) -> bool {
    let (logits, _, _) = router.reference(x);
    logits.chunks(E as usize).all(|row| {
        let mut sel: Vec<f64> = row
            .iter()
            .zip(&router.bias)
            .map(|(&l, &b)| 1.0 / (1.0 + (-l).exp()) + b as f64)
            .collect();
        sel.sort_by(|a, b| b.total_cmp(a));
        sel[..=TOPK as usize].windows(2).all(|p| p[0] - p[1] > 1e-4)
    })
}

/// `rows` activation rows whose routes are separated under `router`.
fn separated_rows(router: &Router, rng: &mut Rng, rows: u32) -> Vec<u16> {
    let mut x = Vec::new();
    while x.len() < (rows * H) as usize {
        let row: Vec<u16> = (0..H).map(|_| bf16(rng.unit() * 2.0)).collect();
        if separated(router, &row) {
            x.extend(row);
        }
    }
    x
}

/// The flip row: columns 0 and 1 are 1.0.
fn flip_row() -> Vec<u16> {
    let mut x = vec![0u16; H as usize];
    x[0] = bf16(1.0);
    x[1] = bf16(1.0);
    x
}

fn floats(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(words(gpu, p, n)?.into_iter().map(f32::from_bits).collect())
}

/// `got` routes against the reference's: the same experts in the same
/// order, weights to 2e-6 (FP32 accumulation order and `__expf` against an
/// exact reference).
fn check_routes(
    what: &str,
    ids: &[u32],
    weights: &[f32],
    want: &(Vec<u32>, Vec<f32>),
) -> Result<()> {
    ensure!(
        ids == want.0.as_slice(),
        "{what}: experts {ids:?}, want {:?}",
        want.0
    );
    for (i, (&g, &w)) in weights.iter().zip(&want.1).enumerate() {
        ensure!((g - w).abs() <= 2e-6, "{what}: weight {i} is {g}, want {w}");
    }
    Ok(())
}

/// FP32 `logits` (`[rows, E]` on the device) against the reference's (to
/// 1e-4 absolute: accumulation order; the routes are the exact check), then
/// both sigmoid top-k kernels on them.
fn check_logits(
    gpu: &dyn GpuBackend,
    what: &str,
    router: &Router,
    logits: DevicePtr,
    rows: u32,
    want: &(Vec<f64>, Vec<u32>, Vec<f32>),
) -> Result<()> {
    let s = gpu.default_stream();
    gpu.synchronize(s)?;
    let got = floats(gpu, logits, (rows * E) as usize)?;
    for (i, (&g, &w)) in got.iter().zip(&want.0).enumerate() {
        ensure!(
            (g as f64 - w).abs() <= 1e-4,
            "{what}: logit {i} is {g}, want {w}"
        );
    }
    let te = (rows * TOPK) as usize;
    let (ids, weights) = (gpu.alloc(te * 4)?, gpu.alloc(te * 4)?);
    let batched = gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched_f32")?;
    ops::moe_topk_sigmoid_batched(
        gpu,
        batched,
        logits,
        router.bias_dev,
        ids,
        weights,
        E,
        TOPK,
        true,
        SCALING,
        rows,
        s,
    )?;
    gpu.synchronize(s)?;
    let want_routes = (want.1.clone(), want.2.clone());
    check_routes(
        &format!("{what} + batched top-k"),
        &words(gpu, ids, te)?,
        &floats(gpu, weights, te)?,
        &want_routes,
    )?;
    let single = gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_f32")?;
    let k = TOPK as usize;
    for r in 0..rows as usize {
        ops::moe_topk_sigmoid(
            gpu,
            single,
            logits.offset(r * E as usize * 4),
            router.bias_dev,
            ids,
            weights,
            E,
            TOPK,
            true,
            SCALING,
            s,
        )?;
        gpu.synchronize(s)?;
        check_routes(
            &format!("{what} + single-token top-k, row {r}"),
            &words(gpu, ids, k)?,
            &floats(gpu, weights, k)?,
            &(
                want.1[r * k..(r + 1) * k].to_vec(),
                want.2[r * k..(r + 1) * k].to_vec(),
            ),
        )?;
    }
    gpu.free(ids)?;
    gpu.free(weights)?;
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fp32_router_paths_match_the_reference() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = &gpu;
    let mut rng = Rng(0x0f32_2026_1019);
    let random = Router::random(gpu, &mut rng)?;
    let flip = Router::flip(gpu)?;
    let mut checked = 0;
    for rows in [1u32, 2, 5, 8, 9, 16, 31, 32, 64, 300] {
        for (label, router) in [("random", &random), ("flip", &flip)] {
            let x = if label == "flip" {
                // The flip row in every slot of the batch.
                flip_row().repeat(rows as usize)
            } else {
                separated_rows(router, &mut rng, rows)
            };
            let want = router.reference(&x);
            if label == "flip" {
                ensure!(want.1[7] == B as u32, "flip reference does not pick B");
            }
            let x_dev = upload(gpu, &bytes_of(&x))?;
            gemms(gpu, router, x_dev, rows, |name, logits| {
                checked += 1;
                check_logits(
                    gpu,
                    &format!("{label} {rows} rows {name}"),
                    router,
                    logits,
                    rows,
                    &want,
                )
            })?;
            gpu.free(x_dev)?;
        }
    }
    println!("{checked} FP32 router GEMM x top-k cases == reference (experts exact, weights 2e-6)");
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fp32_fused_routing_matches_the_reference() -> Result<()> {
    let gpu = backend()?;
    let fuse = RouteFuse::load(&gpu);
    ensure!(fuse.loaded(), "glm_moe_route kernels not in this build");
    let s = gpu.default_stream();
    let mut rng = Rng(0x0f32_f05e);
    let random = Router::random(&gpu, &mut rng)?;
    let flip = Router::flip(&gpu)?;
    let ptrs: Vec<u64> = (0..E as u64).map(|e| 0x1000 + e * 64).collect();
    let ptrs = upload(&gpu, &bytes_of(&ptrs))?;
    for rows in 1..=32u32 {
        for router in [&random, &flip] {
            // Under the flip router: the flip row at the batch's middle slot,
            // zero rows (every logit 0: an eight-way tie by index) around it.
            let x = if std::ptr::eq(router, &flip) {
                let mut x = vec![0u16; (rows * H) as usize];
                let at = (rows / 2 * H) as usize;
                x[at..at + H as usize].copy_from_slice(&flip_row());
                x
            } else {
                separated_rows(router, &mut rng, rows)
            };
            let want = router.reference(&x);
            let layer = Layer {
                gate: router.gate,
                bias: router.bias_dev,
                ptrs,
            };
            let b = Bufs::new(&gpu, &layer, upload(&gpu, &bytes_of(&x))?, rows)?;
            fuse.launch(&gpu, &b.args, s)?;
            let got = b.read(&gpu, s)?;
            let weights: Vec<f32> = got.weights.iter().map(|&w| f32::from_bits(w)).collect();
            check_routes(
                &format!("fused, {rows} rows"),
                &got.ids,
                &weights,
                &(want.1.clone(), want.2.clone()),
            )?;
            if std::ptr::eq(router, &flip) {
                ensure!(
                    got.ids[(rows / 2 * TOPK) as usize + 7] == B as u32,
                    "fused, {rows} rows: the flip row's eighth expert is not B"
                );
            }
        }
    }
    println!("fused FP32 routing == reference at 1..=32 rows, flip row included");
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn bf16_router_logits_flip_the_contended_route() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = &gpu;
    let s = gpu.default_stream();
    let flip = Router::flip(gpu)?;
    let x = upload(gpu, &bytes_of(&flip_row()))?;
    let gate = DenseWeight { weight: flip.gate };
    let logits = gpu.alloc(E as usize * 4)?;
    let (ids, weights) = (gpu.alloc(TOPK as usize * 4)?, gpu.alloc(TOPK as usize * 4)?);
    let eighth = |tier: &str, topk: &str| -> Result<u32> {
        let tier = gpu.kernel("dense_gemv_bf16_batchm", tier)?;
        ops::dense_gemv_bf16_tc(gpu, tier, x, &gate, logits, 1, E, H, E, s)?;
        let topk = gpu.kernel("moe_topk_sig", topk)?;
        ops::moe_topk_sigmoid_batched(
            gpu,
            topk,
            logits,
            flip.bias_dev,
            ids,
            weights,
            E,
            TOPK,
            true,
            SCALING,
            1,
            s,
        )?;
        gpu.synchronize(s)?;
        Ok(words(gpu, ids, TOPK as usize)?[7])
    };
    let bf16_pick = eighth("dense_gemv_bf16_tc8", "moe_topk_sigmoid_batched")?;
    let f32_pick = eighth("dense_gemv_bf16_tc8_f32", "moe_topk_sigmoid_batched_f32")?;
    ensure!(
        bf16_pick == A as u32 && f32_pick == B as u32,
        "eighth route: BF16 logits {bf16_pick} (want {A}), FP32 {f32_pick} (want {B})"
    );
    println!("contended route: BF16 logits pick {A} (tie), FP32 logits pick {B} (reference)");
    Ok(())
}

/// What FP32 logits cost: the canonical router tier and the batched top-k at
/// a verify block of 8 rows, BF16 against FP32 logits (us a launch, warm).
#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fp32_router_logits_cost() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = &gpu;
    let s = gpu.default_stream();
    let mut rng = Rng(0x0f32_c057);
    let router = Router::random(gpu, &mut rng)?;
    let rows = 8u32;
    let x = upload(gpu, &bytes_of(&separated_rows(&router, &mut rng, rows)))?;
    let gate = DenseWeight {
        weight: router.gate,
    };
    let logits = gpu.alloc((rows * E) as usize * 4)?;
    let te = (rows * TOPK) as usize;
    let (ids, weights) = (gpu.alloc(te * 4)?, gpu.alloc(te * 4)?);
    let time = |launch: &dyn Fn() -> Result<()>| -> Result<f64> {
        for _ in 0..50 {
            launch()?;
        }
        gpu.synchronize(s)?;
        let t = std::time::Instant::now();
        for _ in 0..2000 {
            launch()?;
        }
        gpu.synchronize(s)?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / 2000.0)
    };
    for suffix in ["", "_f32"] {
        let tier = gpu.kernel(
            "dense_gemv_bf16_batchm",
            &format!("dense_gemv_bf16_tc8{suffix}"),
        )?;
        let topk = gpu.kernel("moe_topk_sig", &format!("moe_topk_sigmoid_batched{suffix}"))?;
        let gemv =
            time(&|| ops::dense_gemv_bf16_tc(gpu, tier, x, &gate, logits, rows, E, H, E, s))?;
        let route = time(&|| {
            ops::moe_topk_sigmoid_batched(
                gpu,
                topk,
                logits,
                router.bias_dev,
                ids,
                weights,
                E,
                TOPK,
                true,
                SCALING,
                rows,
                s,
            )
        })?;
        println!(
            "logits{}: router tc8 {gemv:.2} us, top-k {route:.2} us ({rows} rows)",
            if suffix.is_empty() { " BF16" } else { " FP32" }
        );
    }
    Ok(())
}
