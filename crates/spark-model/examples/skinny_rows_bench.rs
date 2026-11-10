// SPDX-License-Identifier: AGPL-3.0-only
//! Skinny-kernel cost against the row count, on the Qwen3.8-27B projection
//! shapes: how the `ATLAS_SKINNY_MMA` call (`layers/ops/skinny_mma.rs`)
//! scales from serial decode's 1 row to the 128 rows a batched
//! multi-sequence verify can carry under the batch lever
//! (`model/exact_verify.rs`). Prints us a call, us a row, and weight GB/s.
//!
//! COLD WEIGHTS: calls cycle over enough weight copies (512 MB) that each
//! streams its weight from DRAM, as a decode or verify step does.
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example skinny_rows_bench
//!
//! Env: SKINNY_BENCH_ROWS (default 1,8,16,24,32,48,64,72,96,128),
//! SKINNY_BENCH_ITERS (default 20), ATLAS_PEAK_GBPS (default 273).

use anyhow::Result;
use spark_model::layers::ops::skinny_mma::{self, Family, MAX_ROWS, SkinnyKernels};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

const COLD_CYCLE_BYTES: usize = 512 << 20;
const WARMUP: usize = 3;

/// `(label, family, N, K, heads, head_dim)`.
const SHAPES: &[(&str, Family, u32, u32, u32, u32)] = &[
    (
        "nvfp4 gate|up 2x17408x5120",
        Family::W4a16,
        17408,
        5120,
        0,
        0,
    ),
    (
        "nvfp4 down     5120x17408",
        Family::W4a16,
        5120,
        17408,
        0,
        0,
    ),
    ("nvfp4 o_proj   5120x6144 ", Family::W4a16, 5120, 6144, 0, 0),
    (
        "q+gate        12288x5120 ",
        Family::Qg,
        12288,
        5120,
        24,
        256,
    ),
    (
        "fp8 gdn qkvz  16384x5120 ",
        Family::W8a16,
        16384,
        5120,
        0,
        0,
    ),
    ("fp8 gdn out    5120x6144 ", Family::W8a16, 5120, 6144, 0, 0),
];

fn env_list(name: &str, default: &[u32]) -> Vec<u32> {
    std::env::var(name).ok().map_or_else(
        || default.to_vec(),
        |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
    )
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b ptx set: build with ATLAS_TARGET_MODEL=qwen3.8-27b");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let ks = SkinnyKernels::resolve(g);
    if !ks.complete() {
        println!("skinny_mma kernels absent from this target set — SKIP");
        std::process::exit(2);
    }
    let rows = env_list(
        "SKINNY_BENCH_ROWS",
        &[1, 8, 16, 24, 32, 48, 64, 72, 96, 128],
    );
    let iters = env_list("SKINNY_BENCH_ITERS", &[20])[0] as usize;
    let peak: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);
    let max_m = MAX_ROWS as usize;
    for &(label, family, n, k, heads, hd) in SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let (w_bytes, s_bytes) = match family {
            Family::W8a16 => (nu * ku, nu.div_ceil(128) * ku.div_ceil(128) * 4),
            _ => (nu * ku / 2, nu * ku / 16),
        };
        // gate|up: the dual launch serial decode's FFN runs (two weights).
        let parts = if label.contains("gate|up") { 2 } else { 1 };
        let copies = (COLD_CYCLE_BYTES / (parts * (w_bytes + s_bytes))).max(2);
        let mut weights = Vec::with_capacity(copies);
        for _ in 0..copies {
            let w = g.alloc(w_bytes)?;
            let s = g.alloc(s_bytes)?;
            // Scales 0x38 (E4M3 1.0) / FP32 1e-3; weight bytes 0x11: finite everywhere.
            g.memset(w, 0x11, w_bytes)?;
            g.memset(s, 0x38, s_bytes)?;
            weights.push((w, s));
        }
        let a = g.alloc(max_m * ku * 2)?;
        g.memset(a, 0x3c, max_m * ku * 2)?;
        let c = g.alloc(parts * max_m * nu * 2)?;
        let weight_bytes = parts * (w_bytes + s_bytes);
        println!(
            "── {label}  weights {:.1} MB/call ──",
            weight_bytes as f64 / 1e6
        );
        let call = |m: u32, (w, s): (DevicePtr, DevicePtr)| -> Result<()> {
            let q = QuantizedWeight {
                weight: w,
                weight_scale: s,
                weight_scale_2: 0.01,
                ..QuantizedWeight::null()
            };
            let c1 = c.offset(max_m * nu * 2);
            match family {
                Family::W4a16 => {
                    let all = [(&q, c, n), (&q, c1, n)];
                    skinny_mma::launch_w4a16(g, &ks, a, &all[..parts], (m, k), n, 0)
                }
                Family::Qg => skinny_mma::launch_qg(g, &ks, a, &q, c, (m, n, k), (heads, hd), 0),
                Family::W8a16 => {
                    skinny_mma::launch_w8a16(g, &ks, a, &[((w, s), c, n)], (m, k), n, 0)
                }
            }
        };
        let mut base = None;
        for &m in &rows {
            if !skinny_mma::serves(family, m, k) {
                continue;
            }
            for i in 0..WARMUP {
                call(m, weights[i % copies])?;
            }
            g.synchronize(0)?;
            let t0 = Instant::now();
            for i in 0..iters {
                call(m, weights[(WARMUP + i) % copies])?;
            }
            g.synchronize(0)?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let b = *base.get_or_insert(us);
            let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
            println!(
                "  M={m:<3} {us:>9.1} us  {:>7.2} us/row  x{:.2} of M=1  {gbps:>6.1} GB/s {:>5.1}% peak",
                us / m as f64,
                us / b,
                100.0 * gbps / peak
            );
        }
        for (w, s) in weights {
            g.free(w).ok();
            g.free(s).ok();
        }
        for p in [a, c] {
            g.free(p).ok();
        }
    }
    Ok(())
}
