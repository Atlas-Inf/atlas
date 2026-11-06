// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a qwen3.8-27b
//! kernel build): `dense_bf16mma_prefill.cu` against an FP64 CPU reference of
//! the exact NVFP4 / FP8 dequant, at 33, 135 and 1100 rows (a partial token
//! tile, balanced chunks, and more chunks than one raster group), with zero
//! and subnormal block scales and every e2m1 / e4m3 code. Rows past M stay
//! untouched, and the FP8 grid check rejects a grid that varies along K.
//!
//!   cargo test --release -p spark-model --lib dense_bf16mma -- --ignored --nocapture

use super::*;
use crate::layers::bf16mma_test_ref::*;
use anyhow::ensure;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: usize = 256;
const I: usize = 256;
const ROWS: [usize; 3] = [33, 135, 1100];
const SPARE_ROWS: usize = 3;
const POISON: u16 = 0xa5a5;
const NO_LOWP: LowpArgs = LowpArgs {
    group: 0,
    codes: DevicePtr::NULL,
    scales: DevicePtr::NULL,
};

/// A transposed NVFP4 table: `[K/2, N]` packed (low nibble = even k),
/// `[K/16, N]` scales, scale2.
struct Nvfp4 {
    k: usize,
    n: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale2: f32,
}

impl Nvfp4 {
    fn new(rng: &mut Rng, k: usize, n: usize, scale2: f32) -> Self {
        Self {
            k,
            n,
            packed: (0..k / 2 * n).map(|_| rng.next() as u8).collect(),
            scales: (0..k / 16 * n).map(|_| rng.scale()).collect(),
            scale2,
        }
    }
    fn dot(&self, x: &[u16], n: usize) -> f64 {
        let s: f64 = (0..self.k)
            .map(|k| {
                let b = self.packed[k / 2 * self.n + n];
                let code = if k & 1 == 0 { b & 15 } else { b >> 4 };
                f32_of(x[k]) as f64 * e2m1(code) * e4m3(self.scales[k / 16 * self.n + n])
            })
            .sum();
        s * self.scale2 as f64
    }
    /// The W4A8 kernels' weights: per feature and 64-k stage, E4M3 of
    /// value * scale / 2^E (round to nearest even), times 2^E, where 2^E is
    /// the smallest power of two with 6 * (the stage's largest scale) / 2^E
    /// <= 448.
    fn dot_a8(&self, x: &[u16], n: usize) -> f64 {
        let s: f64 = (0..self.k)
            .map(|k| {
                let smax = (0..4)
                    .map(|j| e4m3(self.scales[(k / 64 * 4 + j) * self.n + n]))
                    .fold(0.0, f64::max);
                let e = if smax > 0.0 {
                    (smax * 6.0 / 448.0).log2().ceil()
                } else {
                    0.0
                };
                let b = self.packed[k / 2 * self.n + n];
                let code = if k & 1 == 0 { b & 15 } else { b >> 4 };
                let w = e2m1(code) * e4m3(self.scales[k / 16 * self.n + n]) / 2f64.powf(e);
                f32_of(x[k]) as f64 * e4m3_rn(w) * 2f64.powf(e)
            })
            .sum();
        s * self.scale2 as f64
    }
    fn upload(&self, gpu: &dyn GpuBackend) -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: upload(gpu, &self.packed)?,
            weight_scale: upload(gpu, &self.scales)?,
            weight_scale_2: self.scale2,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    }
}

/// `[N, K]` e4m3 (no NaN codes) and an FP32 `[N/128, K/128]` grid.
fn fp8_weight(
    rng: &mut Rng,
    gpu: &dyn GpuBackend,
    n: usize,
    k: usize,
    grid: &[f32],
) -> Result<(Vec<u8>, Fp8Weight)> {
    let bytes: Vec<u8> = (0..n * k)
        .map(|_| match rng.next() as u8 {
            b if b & 0x7f == 0x7f => b ^ 1,
            b => b,
        })
        .collect();
    let w = Fp8Weight {
        weight: upload(gpu, &bytes)?,
        row_scale: upload(gpu, &bytes_of(grid))?,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    Ok((bytes, w))
}

#[test]
#[ignore = "needs a GB10 and a qwen3.8-27b kernel build"]
fn dense_bf16mma_matches_exact_dequant_reference() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kernels = DenseBf16mma::resolve_lowp(&gpu, LowpMix(0));
    ensure!(
        kernels.gate_up.0 != 0 && kernels.nvfp4.0 != 0 && kernels.fp8.0 != 0,
        "dense_bf16mma_prefill not in this build"
    );
    let stream = gpu.default_stream();
    let mut rng = Rng(0xd3e5_bf16);
    let (gate, up, down) = (
        Nvfp4::new(&mut rng, H, I, 0.02),
        Nvfp4::new(&mut rng, H, I, 0.03),
        Nvfp4::new(&mut rng, I, H, 0.05),
    );
    let (g_d, u_d, d_d) = (gate.upload(&gpu)?, up.upload(&gpu)?, down.upload(&gpu)?);
    ensure!(kernels.serves_mlp([&g_d, &u_d, &d_d], [64, H as u32, I as u32]));
    let grid: Vec<f32> = (0..H / 128)
        .flat_map(|nb| [0.001 * (1 + nb) as f32; I / 128])
        .collect();
    let (w8, w8_d) = fp8_weight(&mut rng, &gpu, H, I, &grid)?;
    ensure!(fp8_scale_k_uniform(&gpu, &w8_d)? && kernels.serves_fp8(&w8_d, true, 64));
    let mut varying = grid.clone();
    varying[1] *= 2.0;
    let (_, w8_var) = fp8_weight(&mut rng, &gpu, H, I, &varying)?;
    ensure!(
        !fp8_scale_k_uniform(&gpu, &w8_var)?,
        "a K-varying grid passed"
    );

    for m in ROWS {
        let x: Vec<u16> = (0..m * H).map(|_| bf16(rng.unit() * 2.0)).collect();
        let poison = vec![POISON; (m + SPARE_ROWS) * H.max(I)];
        let (x_d, act, out, out8) = (
            upload(&gpu, &bytes_of(&x))?,
            upload(&gpu, &bytes_of(&poison))?,
            upload(&gpu, &bytes_of(&poison))?,
            upload(&gpu, &bytes_of(&poison))?,
        );
        let mu = m as u32;
        kernels.gate_up_silu(
            &gpu,
            x_d,
            [&g_d, &u_d],
            act,
            [mu, I as u32, H as u32],
            NO_LOWP,
            stream,
        )?;
        kernels.nvfp4(
            &gpu,
            act,
            &d_d,
            out,
            [mu, H as u32, I as u32],
            NO_LOWP,
            stream,
        )?;
        // FP8 [H, I] weight on the [m, I] SiLU output.
        kernels.fp8(&gpu, act, &w8_d, out8, mu, NO_LOWP, stream)?;
        gpu.synchronize(stream)?;
        let act_h = read_u16(&gpu, act, (m + SPARE_ROWS) * I)?;
        let out_h = read_u16(&gpu, out, (m + SPARE_ROWS) * H)?;
        let out8_h = read_u16(&gpu, out8, (m + SPARE_ROWS) * H)?;

        let (mut w_gu, mut w_dn, mut w_f8) = (0.0f64, 0.0f64, 0.0f64);
        for r in 0..m {
            let xr = &x[r * H..][..H];
            let want: Vec<f64> = (0..I)
                .map(|n| {
                    // The SiLU reads BF16 gate/up, as the unfused chain does.
                    let g = f32_of(bf16(gate.dot(xr, n) as f32)) as f64;
                    let u = f32_of(bf16(up.dot(xr, n) as f32)) as f64;
                    g / (1.0 + (-g).exp()) * u
                })
                .collect();
            w_gu = w_gu.max(worst(&act_h[r * I..][..I], &want, 2e-2, 2e-3));
            let ar = &act_h[r * I..][..I];
            let want: Vec<f64> = (0..H).map(|n| down.dot(ar, n)).collect();
            w_dn = w_dn.max(worst(&out_h[r * H..][..H], &want, 1e-2, 1e-3));
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
            "M={m}: worst error / tolerance: gate_up+silu {w_gu:.3}, down {w_dn:.3}, fp8 {w_f8:.3}"
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

/// `v` rounded to the nearest E4M3 value, ties to the even code (|v| <= 448).
fn e4m3_rn(v: f64) -> f64 {
    let mut best = (f64::MAX, 0.0, 0u8);
    for c in 0..=0x7eu8 {
        let d = (e4m3(c) - v.abs()).abs();
        if d < best.0 || (d == best.0 && c & 1 == 0 && best.2 & 1 == 1) {
            best = (d, e4m3(c), c);
        }
    }
    best.1.copysign(v)
}

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
        kernels.fp8(&gpu, act, &w8_d, out8, mu, lowp(LowpMix::GDN_OUT), stream)?;
        gpu.synchronize(stream)?;
        let act8 = lowp_activations(&gpu, false, &act_h, [m, I], scratch)?;
        let out_h = read_u16(&gpu, out, (m + SPARE_ROWS) * H)?;
        let out8_h = read_u16(&gpu, out8, (m + SPARE_ROWS) * H)?;

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
