// SPDX-License-Identifier: AGPL-3.0-only

//! Microbench for the MXFP8 projection tiers on the four real FP4-Local
//! shapes, COLD-L2: each iteration reads a different weight copy (copies
//! total > 2x L2), so reported times are DRAM-bound, not L2-bound.
//! Tiers: `batchm` (8-row chunked loop), `rows32` (register-tiled GEMV,
//! one launch), `tc` (mxfp8_proj_tc, the MMA family).

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
const MS: &[usize] = &[1, 2, 4, 8, 16, 24, 32];
const WARMUP: usize = 12;
const ITERS: usize = 48;
/// Rotate this many distinct weight copies; sized so the working set clears
/// L2 twice over (GB10 L2 ~60 MiB; gdn-qkvz is 21 MiB -> 8 copies = 168 MiB).
const COPIES: usize = 8;

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
    let kt = gpu.kernel("mxfp8_proj", "mxfp8_proj_tc")?;
    let mut rng: u64 = 0x5EED;
    let mut next = move || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as u8
    };
    println!(
        "{:>16} {:>4} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}",
        "shape", "m", "batchm us", "rows32 us", "tc us", "bm GB/s", "r32 GB/s", "tc GB/s"
    );
    for &(n, kk, label) in SHAPES {
        let wbytes = n as usize * kk as usize;
        let sbytes = n as usize * (kk / 32) as usize;
        let mut wds = Vec::new();
        let mut sds = Vec::new();
        for _ in 0..COPIES {
            wds.push(up(g, &(0..wbytes).map(|_| next()).collect::<Vec<u8>>())?);
            sds.push(up(g, &(0..sbytes).map(|_| next()).collect::<Vec<u8>>())?);
        }
        let a: Vec<bf16> = (0..32 * kk as usize)
            .map(|_| bf16::from_f32(next() as f32 / 128.0 - 1.0))
            .collect();
        let ab: Vec<u8> = a.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        let ad = up(g, &ab)?;
        let cd = g.alloc(32 * n as usize * 2)?;
        let bytes = (wbytes + sbytes) as f64;
        for &m in MS {
            // which: 0 = batchm 8-row chunks, 1 = rows32, 2 = tc.
            let launch = |which: usize, i: usize| -> Result<()> {
                let (wd, sd) = (wds[i % COPIES], sds[i % COPIES]);
                match which {
                    1 => KernelLaunch::new(g, kr)
                        .grid([div_ceil(n, 8), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(ad)
                        .arg_ptr(wd)
                        .arg_ptr(sd)
                        .arg_ptr(cd)
                        .arg_u32(m as u32)
                        .arg_u32(n)
                        .arg_u32(kk)
                        .launch(stream),
                    2 => KernelLaunch::new(g, kt)
                        .grid([div_ceil(n, 64), div_ceil(m as u32, 16), 1])
                        .block([128, 1, 1])
                        .arg_ptr(ad)
                        .arg_ptr(wd)
                        .arg_ptr(sd)
                        .arg_ptr(cd)
                        .arg_u32(m as u32)
                        .arg_u32(n)
                        .arg_u32(kk)
                        .launch(stream),
                    _ => {
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
                    }
                }
            };
            let time = |which: usize| -> Result<f64> {
                for i in 0..WARMUP {
                    launch(which, i)?;
                }
                g.synchronize(stream)?;
                let t0 = std::time::Instant::now();
                for i in 0..ITERS {
                    launch(which, WARMUP + i)?;
                }
                g.synchronize(stream)?;
                Ok(t0.elapsed().as_micros() as f64 / ITERS as f64)
            };
            let (t0, t1, t2) = (time(0)?, time(1)?, time(2)?);
            println!(
                "{label:>16} {m:>4} {t0:>10.1} {t1:>10.1} {t2:>10.1} {:>9.0} {:>9.0} {:>9.0}",
                bytes / (t0 * 1e3),
                bytes / (t1 * 1e3),
                bytes / (t2 * 1e3)
            );
        }
    }
    Ok(())
}
