// SPDX-License-Identifier: AGPL-3.0-only
//! Microbench: `mxfp8_proj_batchm` 32-row single launch vs the old 8-row
//! chunked loop, at the FP4-Local TP=2 projection shapes.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example mxfp8_proj32_bench

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const SHAPES: &[(u32, u32, &str)] = &[
    (8192, 2560, "gdn-qkvz"),
    (2560, 3072, "out/o_proj"),
    (1280, 2560, "shared gate|up"),
    (2560, 640, "shared down"),
];
const MS: &[usize] = &[1, 2, 4, 8, 12, 16, 24, 32];
const WARMUP: usize = 20;
const ITERS: usize = 100;

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let kb = gpu.kernel("mxfp8_proj", "mxfp8_proj_batchm")?;
    let kr = gpu.kernel("mxfp8_proj", "mxfp8_proj_rows32")?;
    let mut rng: u64 = 0x5EED;
    let mut next = move || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as u8
    };
    println!("{:>16} {:>4} {:>10} {:>10} {:>10}", "shape", "m", "old us", "new us", "GB/s new");
    for &(n, kk, label) in SHAPES {
        let (wd, sd) = (
            up(g, &(0..n as usize * kk as usize).map(|_| next()).collect::<Vec<u8>>())?,
            up(g, &(0..n as usize * (kk / 32) as usize).map(|_| next()).collect::<Vec<u8>>())?,
        );
        let a: Vec<bf16> = (0..32 * kk as usize)
            .map(|_| bf16::from_f32(next() as f32 / 128.0 - 1.0))
            .collect();
        let ab: Vec<u8> = a.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        let ad = up(g, &ab)?;
        let cd = g.alloc(32 * n as usize * 2)?;
        for &m in MS {
            // old = the 8-row chunked batchm loop; new = one rows32 launch.
            let launch = |rows32: bool| -> Result<()> {
                if rows32 {
                    return KernelLaunch::new(g, kr)
                        .grid([div_ceil(n, 8), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(ad)
                        .arg_ptr(wd)
                        .arg_ptr(sd)
                        .arg_ptr(cd)
                        .arg_u32(m as u32)
                        .arg_u32(n)
                        .arg_u32(kk)
                        .launch(stream);
                }
                for first in (0..m).step_by(8) {
                    let rows = (m - first).min(8);
                    KernelLaunch::new(g, kb)
                        .grid([div_ceil(n, 4), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(ad.offset(first * kk as usize * 2))
                        .arg_ptr(wd)
                        .arg_ptr(sd)
                        .arg_ptr(cd.offset(first * n as usize * 2))
                        .arg_u32(rows as u32)
                        .arg_u32(n)
                        .arg_u32(kk)
                        .launch(stream)?;
                }
                Ok(())
            };
            let time = |rows32: bool| -> Result<f64> {
                for _ in 0..WARMUP { launch(rows32)?; }
                g.synchronize(stream)?;
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS { launch(rows32)?; }
                g.synchronize(stream)?;
                Ok(t0.elapsed().as_micros() as f64 / ITERS as f64)
            };
            let (old, new) = (time(false)?, time(true)?);
            let gbps = (n as u64 * kk as u64 + n as u64 * (kk / 32) as u64) as f64 / (new * 1e3);
            println!("{label:>16} {m:>4} {old:>10.1} {new:>10.1} {gbps:>10.0}");
        }
    }
    Ok(())
}
