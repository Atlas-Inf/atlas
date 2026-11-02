// SPDX-License-Identifier: AGPL-3.0-only
//! Times the fast weight loader on a checkpoint as one EP / expert-TP rank
//! loads it, and prints a digest of every resident tensor (name, dtype,
//! shape, bytes) and deferred tensor, so two builds of the loader can be
//! checked to store the same bytes.
//!
//! Run (`EP_RANK`/`EP_SIZE`/`NUM_EXPERTS` default 0/4/288, `EXPERT_TP` 1):
//!   MODEL_DIR=~/.lloom/models/nvidia--GLM-5.3-Flash-FP4-Local ATLAS_TARGET_HW=gb10 \
//!   ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model \
//!     --release --features cuda,gpu-examples --example fast_load_bench

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightLoader;
use std::hash::{DefaultHasher, Hash, Hasher};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("MODEL_DIR").context("MODEL_DIR")?);
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut loader = spark_runtime::fast_weights::FastSafetensorsLoader::with_ep(
        env("EP_RANK", 0),
        env("EP_SIZE", 4),
        env("NUM_EXPERTS", 288),
    );
    loader.expert_tp = env("EXPERT_TP", 1) == 1;
    let t0 = std::time::Instant::now();
    let store = loader.load(&dir, g, 4 << 30)?;
    let load = t0.elapsed().as_secs_f64();
    let mut names: Vec<&str> = store.names().collect();
    names.sort_unstable();
    let mut h = DefaultHasher::new();
    let mut buf = Vec::new();
    for name in &names {
        let t = store.get(name)?;
        buf.resize(t.num_elements() * t.dtype.byte_size(), 0);
        g.copy_d2h(t.ptr, &mut buf)?;
        (name, &t.shape, format!("{:?}", t.dtype), &buf).hash(&mut h);
    }
    let deferred = store.deferred_sorted();
    for (name, d) in &deferred {
        (name, &d.path, d.offset, &d.shape).hash(&mut h);
    }
    println!(
        "fast load {load:.2} s: {} tensors, {:.2} GB, {} deferred, digest {:016x}",
        store.len(),
        store.total_bytes() as f64 / 1e9,
        deferred.len(),
        h.finish()
    );
    Ok(())
}
