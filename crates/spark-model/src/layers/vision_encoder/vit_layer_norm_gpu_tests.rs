// SPDX-License-Identifier: AGPL-3.0-only

//! Ignored, CUDA-only check that `vision_layer_norm_to` (256 threads, src to
//! dst) gives `vision_layer_norm`'s bits (one 1024-thread CTA a row, in
//! place), in place and out of place, for the kernel targets whose ViT uses it.
//!
//!   cargo test --release -p spark-model --lib vision_layer_norm_to -- --ignored --nocapture

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

/// Deterministic BF16 bits around `centre` (a row mean far from 0 stresses
/// the variance pass), roughly `centre +- 4`.
fn values(n: usize, seed: u32, centre: f32) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = centre + (state >> 8) as f32 / (1u32 << 21) as f32 - 4.0;
            let bits = v.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
        })
        .collect()
}

fn upload(gpu: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut out)?;
    Ok(out)
}

#[test]
#[ignore = "requires a CUDA host"]
fn vision_layer_norm_to_gives_vision_layer_norm_bits() -> Result<()> {
    let mut checked = 0;
    for target in ["qwen3.8-27b", "qwen3.8-flash-next"] {
        let Some(ptx) = atlas_kernels::ptx_for_exact_target(target, "nvfp4") else {
            eprintln!("{target}: not in this kernel build, skipped");
            continue;
        };
        checked += 1;
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules)
            .context("create CUDA backend")?;
        let gpu: &dyn GpuBackend = &gpu;
        let norm = gpu.kernel("vision_encoder", "vision_layer_norm")?;
        let norm_to = gpu.kernel("vision_encoder", "vision_layer_norm_to")?;
        let stream = gpu.default_stream();
        // The Qwen ViT width (1152) and other widths the kernel accepts.
        for (d, rows) in [(1152usize, 300usize), (1024, 17), (2048, 9), (5120, 5)] {
            for centre in [0.0f32, 37.0] {
                let seed = (d * 31 + rows) as u32 + centre as u32;
                let x = values(rows * d, seed, centre);
                let w = values(d, seed + 1, 1.0);
                let b = values(d, seed + 2, 0.0);
                let (dw, db) = (upload(gpu, &w)?, upload(gpu, &b)?);
                let (a, src, dst) = (upload(gpu, &x)?, upload(gpu, &x)?, gpu.alloc(rows * d * 2)?);
                let inplace = upload(gpu, &x)?;
                let (n, dd) = (rows as u32, d as u32);
                KernelLaunch::new(gpu, norm)
                    .grid([n, 1, 1])
                    .block([dd.min(1024), 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(dw)
                    .arg_ptr(db)
                    .arg_u32(n)
                    .arg_u32(dd)
                    .arg_f32(1e-6)
                    .launch(stream)?;
                for (from, to) in [(src, dst), (inplace, inplace)] {
                    KernelLaunch::new(gpu, norm_to)
                        .grid([n, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(from)
                        .arg_ptr(to)
                        .arg_ptr(dw)
                        .arg_ptr(db)
                        .arg_u32(n)
                        .arg_u32(dd)
                        .arg_f32(1e-6)
                        .launch(stream)?;
                }
                gpu.synchronize(stream)?;
                let want = download(gpu, a, rows * d)?;
                let out = download(gpu, dst, rows * d)?;
                let same = download(gpu, inplace, rows * d)?;
                let unchanged = download(gpu, src, rows * d)?;
                for p in [dw, db, a, src, dst, inplace] {
                    gpu.free(p)?;
                }
                let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                ensure!(unchanged == x_bytes, "{target} D {d}: src was written");
                for (name, got) in [("src -> dst", &out), ("in place", &same)] {
                    let diff = want
                        .chunks(2)
                        .zip(got.chunks(2))
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!("{target} D {d} rows {rows} centre {centre} {name}: {diff} differ");
                    ensure!(
                        diff == 0,
                        "{target} D {d} {name}: {diff} of {} differ",
                        rows * d
                    );
                }
            }
        }
    }
    ensure!(checked > 0, "neither ViT kernel target is in this build");
    Ok(())
}
