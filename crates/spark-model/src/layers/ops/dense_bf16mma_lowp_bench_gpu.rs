// SPDX-License-Identifier: AGPL-3.0-only

//! Timing of the low-precision kernels at d3-ultra's shapes (`#[ignore]`; a
//! GB10 and a qwen3.8-27b kernel build).

use super::*;

/// Timing at d3-ultra's shapes (Qwen3.6/3.8-27B: hidden 5120, MLP 17408, GDN
/// qkvz 16384 and out 6144 -> 5120) on random weights, quantizer included:
/// `DENSE_BENCH_M` rows (default 685, one image question at 1024x576).
///
///   DENSE_BENCH_M=685 cargo test --release -p spark-model --lib dense_lowp_bench -- --ignored --nocapture
#[test]
#[ignore = "needs a GB10 and a qwen3.8-27b kernel build"]
fn dense_lowp_bench_d3_shapes() -> Result<()> {
    const HD: usize = 5120;
    const ID: usize = 17408;
    let m: usize = std::env::var("DENSE_BENCH_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(685);
    let reps: usize = 20;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = DenseBf16mma::resolve_lowp(&gpu, LowpMix::parse("gu,dn,qkv,o,qkvz,gout"));
    let stream = gpu.default_stream();
    let mut rng = Rng(0xbe9c_d3d3);
    let (gate, up, down) = (
        Nvfp4::new(&mut rng, HD, ID, 0.02),
        Nvfp4::new(&mut rng, HD, ID, 0.03),
        Nvfp4::new(&mut rng, ID, HD, 0.05),
    );
    let (g_d, u_d, d_d) = (gate.upload(&gpu)?, up.upload(&gpu)?, down.upload(&gpu)?);
    drop((gate, up, down));
    // Transposed W8A8 copies: qkvz [5120 -> 16384], out [6144 -> 5120].
    let f8t = |rng: &mut Rng, n: usize, k: usize| -> Result<(Fp8Weight, Fp8WeightTransposed)> {
        let grid = vec![0.001f32; (n / 128) * (k / 128)];
        let (w8, w) = fp8_weight(rng, &gpu, n, k, &grid)?;
        let w8t: Vec<u8> = (0..n * k).map(|i| w8[i % k * n + i / k]).collect();
        let t = Fp8WeightTransposed {
            weight_t: upload(&gpu, &w8t)?,
            scale_t: upload(&gpu, &bytes_of(&grid))?,
            n: n as u32,
            k: k as u32,
        };
        Ok((w, t))
    };
    let (qkvz, qkvz_t) = f8t(&mut rng, 16384, HD)?;
    let (gout, gout_t) = f8t(&mut rng, HD, 6144)?;
    let x: Vec<u16> = (0..m * ID).map(|_| bf16(rng.unit() * 2.0)).collect();
    let (x_d, act, out) = (
        upload(&gpu, &bytes_of(&x))?,
        gpu.alloc((m + SPARE_ROWS) * ID * 2)?,
        gpu.alloc((m + SPARE_ROWS) * 16384 * 2)?,
    );
    let scratch = [gpu.alloc(m * ID)?, gpu.alloc(m * ID / 16 * 4)?];
    let lowp = |group| LowpArgs {
        group,
        codes: scratch[0],
        scales: scratch[1],
    };
    let mu = m as u32;
    let time = |name: &str, flops: f64, f: &dyn Fn() -> Result<()>| -> Result<()> {
        f()?;
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f()?;
        }
        gpu.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!(
            "M={m} {name:10} {us:8.1} us  {:6.1} TFLOPS",
            flops / us * 1e-6
        );
        Ok(())
    };
    let fl = |n: usize, k: usize| 2.0 * (m * n * k) as f64;
    time("gate_up", fl(2 * ID, HD), &|| {
        kernels.gate_up_silu(
            &gpu,
            x_d,
            [&g_d, &u_d],
            act,
            [mu, ID as u32, HD as u32],
            lowp(LowpMix::GATE_UP),
            stream,
        )
    })?;
    time("down", fl(HD, ID), &|| {
        kernels.nvfp4(
            &gpu,
            x_d,
            &d_d,
            out,
            [mu, HD as u32, ID as u32],
            lowp(LowpMix::DOWN),
            stream,
        )
    })?;
    time("gdn_qkvz", fl(16384, HD), &|| {
        kernels.fp8(
            &gpu,
            x_d,
            &qkvz,
            Some(&qkvz_t),
            out,
            mu,
            lowp(LowpMix::GDN_QKVZ),
            stream,
        )
    })?;
    time("gdn_out", fl(HD, 6144), &|| {
        kernels.fp8(
            &gpu,
            x_d,
            &gout,
            Some(&gout_t),
            out,
            mu,
            lowp(LowpMix::GDN_OUT),
            stream,
        )
    })?;
    Ok(())
}
