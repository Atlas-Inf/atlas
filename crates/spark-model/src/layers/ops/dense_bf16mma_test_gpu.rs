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
    let kernels = DenseBf16mma::resolve(&gpu);
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
            stream,
        )?;
        kernels.nvfp4(&gpu, act, &d_d, out, [mu, H as u32, I as u32], stream)?;
        // FP8 [H, I] weight on the [m, I] SiLU output.
        kernels.fp8(&gpu, act, &w8_d, out8, mu, stream)?;
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
