// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests of the low-precision kernels (`#[ignore]`; a GB10 and a
//! qwen3.8-27b kernel build): W4A4, W4A8 and W8A8 (also on the transposed
//! FP8 copy) against an FP64 reference on the activations read back from the
//! quantizer's scratch, which is checked on the way.
//!
//!   cargo test --release -p spark-model --lib dense_lowp -- --ignored --nocapture

use super::gpu_tests::{H, I, Nvfp4, POISON, ROWS, SPARE_ROWS, fp8_weight};
use super::*;
use crate::layers::bf16mma_test_ref::*;
use anyhow::ensure;
use spark_runtime::cuda_backend::AtlasCudaBackend;

/// The activations a low-precision kernel multiplied, read back from the
/// scratch: NVFP4 (`k / 16` block scales per row) or E4M3 codes, each times
/// its row's power-of-two token scale. Checks the quantizer on the way: the
/// token scale is the smallest power of two that fits the row, each block
/// scale is within E4M3 rounding of the block's amax / 6, and every code is
/// a nearest grid point.
fn lowp_activations(
    gpu: &dyn GpuBackend,
    nvfp4: bool,
    x: &[u16],
    [m, k]: [usize; 2],
    scratch: [DevicePtr; 2],
) -> Result<Vec<u16>> {
    let mut codes = vec![0u8; if nvfp4 { m * k / 2 } else { m * k }];
    gpu.copy_d2h(scratch[0], &mut codes)?;
    let nb = if nvfp4 {
        (m * k / 16).next_multiple_of(16)
    } else {
        0
    };
    let mut sc = vec![0u8; nb + m * 4];
    gpu.copy_d2h(scratch[1], &mut sc)?;
    let ts = |r: usize| f32::from_le_bytes(sc[nb + 4 * r..][..4].try_into().unwrap()) as f64;
    let lim = if nvfp4 { 6.0 * 448.0 } else { 448.0 };
    let mut out = vec![0u16; m * k];
    for r in 0..m {
        let xr: Vec<f64> = x[r * k..][..k].iter().map(|&h| f32_of(h) as f64).collect();
        let amax = xr.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let t = ts(r);
        ensure!(
            t.log2().fract() == 0.0,
            "row {r}: token scale {t} not a power of two"
        );
        ensure!(
            amax == 0.0 || (amax / t <= lim && amax / t > lim / 2.0),
            "row {r}: token scale {t} for amax {amax}"
        );
        for b in 0..k / 16 {
            let xb = &xr[16 * b..][..16];
            let (grid, step): (Vec<f64>, f64) = if nvfp4 {
                let s = e4m3(sc[r * (k / 16) + b]) * t;
                let bmax = xb.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                ensure!(
                    (s * 6.0 - bmax).abs() <= bmax / 16.0 + 2f64.powi(-9) * 6.0 * t,
                    "row {r} block {b}: scale {s} for amax {bmax}"
                );
                ((0..16u8).map(|c| e2m1(c) * s).collect(), s)
            } else {
                (
                    (0..=255u8)
                        .filter(|c| c & 0x7f != 0x7f)
                        .map(|c| e4m3(c) * t)
                        .collect(),
                    0.0,
                )
            };
            for (j, &v) in xb.iter().enumerate() {
                let i = 16 * b + j;
                let q = if nvfp4 {
                    let byte = codes[r * (k / 2) + i / 2];
                    e2m1(if i & 1 == 0 { byte & 15 } else { byte >> 4 }) * step
                } else {
                    e4m3(codes[r * k + i]) * t
                };
                let best = grid.iter().fold(f64::MAX, |a, g| a.min((g - v).abs()));
                ensure!(
                    (q - v).abs() <= best * (1.0 + 1e-6) + 1e-30,
                    "row {r} k {i}: {v} -> {q}, nearest is {best} away"
                );
                out[r * k + i] = bf16(q as f32);
                ensure!(
                    f32_of(out[r * k + i]) as f64 == q,
                    "row {r} k {i}: {q} not exact in BF16"
                );
            }
        }
    }
    Ok(out)
}

#[test]
#[ignore = "needs a GB10 and a qwen3.8-27b kernel build"]
fn dense_lowp_matches_exact_reference_on_quantized_activations() -> Result<()> {
    // W4A4 MLP, then W4A8; the FP8 projection is W8A8 in both.
    lowp_case("all", false)?;
    lowp_case("gu8,dn8,gout", true)
}

fn lowp_case(mix: &str, a8: bool) -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = DenseBf16mma::resolve_lowp(&gpu, LowpMix::parse(mix));
    ensure!(
        [
            kernels.w4a4_gate_up,
            kernels.w4a4,
            kernels.w4a8_gate_up,
            kernels.w4a8,
            kernels.w8a8,
            kernels.quant_nvfp4,
            kernels.quant_fp8
        ]
        .iter()
        .all(|h| h.0 != 0),
        "dense lowp kernels not in this build"
    );
    let tag = if a8 { "w4a8" } else { "w4a4" };
    let stream = gpu.default_stream();
    let mut rng = Rng(0x10e5_a4a4);
    let (gate, up, down) = (
        Nvfp4::new(&mut rng, H, I, 0.02),
        Nvfp4::new(&mut rng, H, I, 0.03),
        Nvfp4::new(&mut rng, I, H, 0.05),
    );
    let (g_d, u_d, d_d) = (gate.upload(&gpu)?, up.upload(&gpu)?, down.upload(&gpu)?);
    let grid: Vec<f32> = (0..H / 128)
        .flat_map(|nb| [0.001 * (1 + nb) as f32; I / 128])
        .collect();
    let (w8, w8_d) = fp8_weight(&mut rng, &gpu, H, I, &grid)?;
    // The transposed [K, N] prefill copy and its [K/128, N/128] grid.
    let w8t: Vec<u8> = (0..I * H).map(|i| w8[i % H * I + i / H]).collect();
    let grid_t: Vec<f32> = (0..I / 128)
        .flat_map(|kb| (0..H / 128).map(move |nb| (kb, nb)))
        .map(|(kb, nb)| grid[nb * (I / 128) + kb])
        .collect();
    let w8t_d = Fp8WeightTransposed {
        weight_t: upload(&gpu, &w8t)?,
        scale_t: upload(&gpu, &bytes_of(&grid_t))?,
        n: H as u32,
        k: I as u32,
    };
    let kmax = H.max(I);
    for m in ROWS {
        // Rows of every magnitude, one all zero.
        let x: Vec<u16> = (0..m * H)
            .map(|i| match (i / H, i / H % 4) {
                (7, _) => 0,
                (_, 1) => bf16(rng.unit() * 300.0),
                (_, 2) => bf16(rng.unit() * 1e-3),
                _ => bf16(rng.unit() * 2.0),
            })
            .collect();
        let poison = vec![POISON; (m + SPARE_ROWS) * kmax];
        let (x_d, act, out, out8) = (
            upload(&gpu, &bytes_of(&x))?,
            upload(&gpu, &bytes_of(&poison))?,
            upload(&gpu, &bytes_of(&poison))?,
            upload(&gpu, &bytes_of(&poison))?,
        );
        let scratch = [gpu.alloc(m * kmax)?, gpu.alloc(m * kmax / 16 * 4)?];
        let lowp = |group| LowpArgs {
            group,
            codes: scratch[0],
            scales: scratch[1],
        };
        let mu = m as u32;
        kernels.gate_up_silu(
            &gpu,
            x_d,
            [&g_d, &u_d],
            act,
            [mu, I as u32, H as u32],
            lowp(LowpMix::GATE_UP),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let xq = lowp_activations(&gpu, !a8, &x, [m, H], scratch)?;
        let act_h = read_u16(&gpu, act, (m + SPARE_ROWS) * I)?;
        kernels.nvfp4(
            &gpu,
            act,
            &d_d,
            out,
            [mu, H as u32, I as u32],
            lowp(LowpMix::DOWN),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let aq = lowp_activations(&gpu, !a8, &act_h, [m, I], scratch)?;
        kernels.fp8(
            &gpu,
            act,
            &w8_d,
            None,
            out8,
            mu,
            lowp(LowpMix::GDN_OUT),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let act8 = lowp_activations(&gpu, false, &act_h, [m, I], scratch)?;
        // W8A8 on the transposed copy: the same MMA operands, bit for bit.
        let out8t = upload(&gpu, &bytes_of(&poison))?;
        kernels.fp8(
            &gpu,
            act,
            &w8_d,
            Some(&w8t_d),
            out8t,
            mu,
            lowp(LowpMix::GDN_OUT),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let out_h = read_u16(&gpu, out, (m + SPARE_ROWS) * H)?;
        let out8_h = read_u16(&gpu, out8, (m + SPARE_ROWS) * H)?;
        ensure!(
            read_u16(&gpu, out8t, (m + SPARE_ROWS) * H)? == out8_h,
            "M={m}: W8A8 on the transposed copy differs"
        );

        let (mut w_gu, mut w_dn, mut w_f8) = (0.0f64, 0.0f64, 0.0f64);
        for r in 0..m {
            let xr = &xq[r * H..][..H];
            let want: Vec<f64> = (0..I)
                .map(|n| {
                    let (g, u) = if a8 {
                        (gate.dot_a8(xr, n), up.dot_a8(xr, n))
                    } else {
                        (gate.dot(xr, n), up.dot(xr, n))
                    };
                    let (g, u) = (f32_of(bf16(g as f32)) as f64, f32_of(bf16(u as f32)) as f64);
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            w_gu = w_gu.max(worst(&act_h[r * I..][..I], &want, 2e-2, 2e-3));
            let ar = &aq[r * I..][..I];
            let want: Vec<f64> = (0..H)
                .map(|n| {
                    if a8 {
                        down.dot_a8(ar, n)
                    } else {
                        down.dot(ar, n)
                    }
                })
                .collect();
            w_dn = w_dn.max(worst(&out_h[r * H..][..H], &want, 1e-2, 1e-3));
            let ar = &act8[r * I..][..I];
            let want: Vec<f64> = (0..H)
                .map(|n| {
                    let s: f64 = (0..I)
                        .map(|k| f32_of(ar[k]) as f64 * e4m3(w8[n * I + k]))
                        .sum();
                    s * grid[n / 128 * (I / 128)] as f64
                })
                .collect();
            w_f8 = w_f8.max(worst(&out8_h[r * H..][..H], &want, 1e-2, 1e-3));
        }
        println!(
            "M={m}: worst error / tolerance: {tag} gate_up+silu {w_gu:.3}, {tag} down {w_dn:.3}, w8a8 {w_f8:.3}"
        );
        ensure!(
            w_gu <= 1.0 && w_dn <= 1.0 && w_f8 <= 1.0,
            "M={m}: off the reference"
        );
        ensure!(
            act_h[m * I..]
                .iter()
                .chain(&out_h[m * H..])
                .chain(&out8_h[m * H..])
                .all(|&v| v == POISON),
            "M={m}: rows past M were written"
        );
    }
    Ok(())
}

/// The MX8 handoff (`gu8,dnmx`): the W4A8 gate+up writes E4M3 codes and
/// UE8M0 block scales (block: 64-K stage st, half h = k % 16 / 8), each a
/// block's smallest power of two that fits its amax under 448; the down reads
/// them. The codes stay within E4M3 rounding of the SiLU values, and the down
/// matches an FP64 reference on the tokens read back.
#[test]
#[ignore = "needs a GB10 and a qwen3.8-27b kernel build"]
fn dense_mx_handoff_matches_reference() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = DenseBf16mma::resolve_lowp(&gpu, LowpMix::parse("gu8,dnmx"));
    let stream = gpu.default_stream();
    let mut rng = Rng(0x3c8_e8e0);
    // The handoff stages its scales 512 K at a time.
    const IM: usize = 512;
    let (gate, up, down) = (
        Nvfp4::new(&mut rng, H, IM, 0.02),
        Nvfp4::new(&mut rng, H, IM, 0.03),
        Nvfp4::new(&mut rng, IM, H, 0.05),
    );
    let (g_d, u_d, d_d) = (gate.upload(&gpu)?, up.upload(&gpu)?, down.upload(&gpu)?);
    for m in ROWS {
        let x: Vec<u16> = (0..m * H)
            .map(|i| bf16(rng.unit() * if i / H % 3 == 1 { 50.0 } else { 2.0 }))
            .collect();
        let poison = vec![POISON; (m + SPARE_ROWS) * IM];
        let (x_d, act, out) = (
            upload(&gpu, &bytes_of(&x))?,
            upload(&gpu, &bytes_of(&poison))?,
            upload(&gpu, &bytes_of(&poison))?,
        );
        let scratch = [gpu.alloc(m * IM)?, gpu.alloc(m * IM / 16 * 4)?];
        let lowp = |group| LowpArgs {
            group,
            codes: scratch[0],
            scales: scratch[1],
        };
        ensure!(
            kernels.mx_on(&lowp(LowpMix::GATE_UP), IM as u32),
            "MX handoff kernels not in this build"
        );
        let mu = m as u32;
        kernels.gate_up_silu(
            &gpu,
            x_d,
            [&g_d, &u_d],
            act,
            [mu, IM as u32, H as u32],
            lowp(LowpMix::GATE_UP),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let xq = lowp_activations(&gpu, false, &x, [m, H], scratch)?;
        kernels.nvfp4(
            &gpu,
            act,
            &d_d,
            out,
            [mu, H as u32, IM as u32],
            lowp(LowpMix::DOWN),
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut mx = vec![0u8; m * IM + m * IM / 32];
        gpu.copy_d2h(act, &mut mx)?;
        let (codes, scales) = mx.split_at(m * IM);
        let out_h = read_u16(&gpu, out, (m + SPARE_ROWS) * H)?;
        let (mut w_gu, mut w_dn) = (0.0f64, 0.0f64);
        for r in 0..m {
            let deq: Vec<f64> = (0..IM)
                .map(|k| {
                    let s = scales[r * (IM / 32) + k / 64 * 2 + k % 16 / 8] as i32 - 127;
                    e4m3(codes[r * IM + k]) * 2f64.powi(s)
                })
                .collect();
            for b in 0..IM / 32 {
                let p = 2f64.powi(scales[r * (IM / 32) + b] as i32 - 127);
                let amax = (0..IM)
                    .filter(|k| k / 64 * 2 + k % 16 / 8 == b)
                    .fold(0.0f64, |a, k| a.max(deq[k].abs()));
                ensure!(
                    amax <= 448.0 * p
                        && (amax == 0.0 || amax > 448.0 * p / 2.0 * (1.0 - 1.0 / 16.0)),
                    "M={m} row {r} block {b}: amax {amax} under scale {p}"
                );
            }
            let xr = &xq[r * H..][..H];
            let want: Vec<f64> = (0..IM)
                .map(|n| {
                    let g = f32_of(bf16(gate.dot_a8(xr, n) as f32)) as f64;
                    let u = f32_of(bf16(up.dot_a8(xr, n) as f32)) as f64;
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            let deq_h: Vec<u16> = deq.iter().map(|&v| bf16(v as f32)).collect();
            w_gu = w_gu.max(worst(&deq_h, &want, 2e-2 + 1.0 / 16.0, 2e-3));
            let want: Vec<f64> = (0..H).map(|n| down.dot_a8(&deq_h, n)).collect();
            w_dn = w_dn.max(worst(&out_h[r * H..][..H], &want, 1e-2, 1e-3));
        }
        println!("M={m}: worst error / tolerance: mx gate_up+silu {w_gu:.3}, mx down {w_dn:.3}");
        ensure!(w_gu <= 1.0 && w_dn <= 1.0, "M={m}: off the reference");
        ensure!(
            out_h[m * H..].iter().all(|&v| v == POISON),
            "M={m}: rows past M were written"
        );
    }
    Ok(())
}
