// SPDX-License-Identifier: AGPL-3.0-only
//! The DFlash drafter through the fast weight loader against the standard
//! loader serving uses today: the same tensor names, shapes, dtypes and
//! bytes, and both load times.
//!
//! Exit: 0 identical, 1 any difference.
//!
//! Run:
//!   DFLASH_DIR=~/.lloom/models/incoai--GLM-5.3-Flash-DFlash2 ATLAS_TARGET_HW=gb10 \
//!   ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model \
//!     --release --features cuda,gpu-examples --example dflash_fast_load_check

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};

fn bytes(g: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<Vec<u8>> {
    let t = store.get(name)?;
    let mut v = vec![0u8; t.num_elements() * t.dtype.byte_size()];
    g.copy_d2h(t.ptr, &mut v)?;
    Ok(v)
}

fn main() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("DFLASH_DIR").context("DFLASH_DIR")?);
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let t0 = std::time::Instant::now();
    let mut standard = SafetensorsLoader::new();
    standard.peak_memory_multiplier = None;
    let a = standard.load(&dir, g, 0)?;
    let t_std = t0.elapsed();
    let t1 = std::time::Instant::now();
    let b = spark_runtime::fast_weights::FastSafetensorsLoader::new().load(&dir, g, 0)?;
    let t_fast = t1.elapsed();
    println!(
        "standard {:.2} s, fast {:.2} s; {} vs {} tensors, {} vs {} bytes",
        t_std.as_secs_f64(),
        t_fast.as_secs_f64(),
        a.len(),
        b.len(),
        a.total_bytes(),
        b.total_bytes()
    );
    let mut names: Vec<&str> = a.names().collect();
    names.sort_unstable();
    let mut other: Vec<&str> = b.names().collect();
    other.sort_unstable();
    let mut diff = usize::from(names != other);
    for name in names {
        let (x, y) = (a.get(name)?, b.get(name)?);
        if x.shape != y.shape || x.dtype != y.dtype || bytes(g, &a, name)? != bytes(g, &b, name)? {
            println!(
                "  {name}: differs ({:?} {:?} vs {:?} {:?})",
                x.shape, x.dtype, y.shape, y.dtype
            );
            diff += 1;
        }
    }
    println!("{}", if diff == 0 { "identical" } else { "DIFFERENT" });
    std::process::exit(i32::from(diff != 0));
}
