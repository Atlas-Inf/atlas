// SPDX-License-Identifier: AGPL-3.0-only

//! `dflash2_grouped_dynamic_causal_conv` GPU-vs-CPU reference for the real
//! delta layout `[io][tap][group]` (2·k·groups per row), k=2 and k=4, both
//! io sides. CPU formula (z-lab dflash):
//!   out[i,c] = Σ_{t<k} (base[io][t][c] + delta[i][io][t][g(c)]) · x[i−t,c]
//!
//!   cargo test --release -p spark-model --lib conv_kernel -- --ignored --nocapture

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn f16(x: u16) -> f32 {
    half::bf16::from_bits(x).to_f32()
}

fn upload(gpu: &dyn GpuBackend, h: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn download(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// CPU reference for one io side of one block.
fn cpu_conv(
    x: &[f32],
    base: &[f32],
    delta: &[f32], // [K][io][tap][group] — delta[i][io][t][g]
    io: usize,
    k: usize,
    rows: usize,
    hidden: usize,
    group_size: usize,
    groups: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * hidden];
    for i in 0..rows {
        for c in 0..hidden {
            let g = c / group_size;
            let mut acc = 0.0f32;
            let taps = (i + 1).min(k);
            for t in 0..taps {
                let b = base[t * hidden + c];
                let d = delta[((i * 2 + io) * k + t) * groups + g];
                acc += (b + d) * x[(i - t) * hidden + c];
            }
            y[i * hidden + c] = acc;
        }
    }
    y
}

#[test]
#[ignore = "GPU test"]
fn conv_kernel_matches_cpu_for_k2_and_k4() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0)?;
    let kernel = gpu.kernel("dflash2_conv", "dflash2_grouped_dynamic_causal_conv")?;
    for &kernel_size in &[2usize, 4] {
        let (rows, hidden, group_size) = (16usize, 512usize, 16usize);
        let groups = hidden / group_size;
        let mut rng = Rng(0x9E37 ^ kernel_size as u64);
        let x: Vec<f32> = (0..rows * hidden).map(|_| rng.unit() * 2.0).collect();
        let base: Vec<f32> = (0..2 * kernel_size * hidden)
            .map(|_| rng.unit())
            .collect();
        let delta: Vec<f32> = (0..rows * 2 * kernel_size * groups)
            .map(|_| rng.unit() * 0.5)
            .collect();
        for io in 0..2 {
            // io-side slices: base[io*k*hidden..], delta row offset io*k*groups.
            let x_dev = upload(&gpu, &x.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>())?;
            let base_dev = upload(
                &gpu,
                &base[io * kernel_size * hidden..(io + 1) * kernel_size * hidden]
                    .iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(),
            )?;
            let mut row_delta = Vec::with_capacity(rows * kernel_size * groups);
            for i in 0..rows {
                row_delta.extend_from_slice(
                    &delta[(i * 2 + io) * kernel_size * groups
                        ..(i * 2 + io + 1) * kernel_size * groups],
                );
            }
            let delta_dev = upload(&gpu, &row_delta.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>())?;
            let y_dev = gpu.alloc(rows * hidden * 2)?;
            KernelLaunch::new(&gpu, kernel)
                .grid([((rows * hidden) as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(y_dev)
                .arg_ptr(x_dev)
                .arg_ptr(delta_dev)
                .arg_ptr(base_dev)
                .arg_u32(rows as u32)
                .arg_u32(hidden as u32)
                .arg_u32(group_size as u32)
                .arg_u32(groups as u32)
                .arg_u32(kernel_size as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let got = download(&gpu, y_dev, rows * hidden)?;
            let want = cpu_conv(&x, &base, &delta, io, kernel_size, rows, hidden, group_size, groups);
            let (mut e2, mut r2) = (0f64, 0f64);
            for (g, w) in got.iter().zip(&want) {
                e2 += (f16(*g) as f64 - *w as f64).powi(2);
                r2 += (*w as f64).powi(2);
            }
            let rel = (e2 / r2.max(1e-30)).sqrt();
            assert!(
                rel < 0.02,
                "k={kernel_size} io={io}: rel RMS {rel} > 0.02"
            );
        }
    }
    Ok(())
}
