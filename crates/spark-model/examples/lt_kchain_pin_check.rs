// SPDX-License-Identifier: AGPL-3.0-only
//! End-to-end check of the cuBLASLt k-chain pin (`ATLAS_LT_KCHAIN_PIN`,
//! `spark_runtime::cublaslt::kchain_pin`) through the production BF16 GEMM
//! (`cublaslt::bf16_gemm_act_weight_t`), at the qwen4_exp TP2 prefill
//! projection shapes over several row counts. Prints an FNV-1a hash of every
//! output and its time; a run with the switch and one without must print the
//! SAME hashes.
//!
//! With the switch it also checks `cublaslt::bf16_gemm_act_weight_t_kchain`
//! (a forced k-chain kernel) against the Atlas tile kernel
//! `dense_gemm_bf16_pipelined` it replaces (the mHC down GEMM of a
//! 1921..2048-row slab, 320 x 10240; the GDN out_proj, 2560 x 3072): every
//! byte must match.
//!
//! Run inside atlas-release-builder (the runtime image's cuBLASLt):
//!   cargo build -p spark-model --release --features cuda,gpu-examples \
//!     --example lt_kchain_pin_check
//!   ./lt_kchain_pin_check > off.txt
//!   ATLAS_LT_KCHAIN_PIN=1 ./lt_kchain_pin_check > on.txt
//!   diff <(grep hash off.txt) <(grep hash on.txt)

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

fn bf16s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            (((f * scale).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut seed = 0x5eed_u64;
    // (N, K): GDN in_proj, attention q+gate, a 4096-wide and a 2560-wide one.
    for (n, k) in [
        (8192usize, 2560usize),
        (6144, 2560),
        (4096, 2560),
        (2560, 3072),
    ] {
        let w = g.alloc(n * k * 2)?;
        g.copy_h2d(&bf16s(&mut seed, n * k, 0.02), w)?;
        for m in [1000usize, 2048, 5003, 9999, 16016] {
            let a = g.alloc(m * k * 2)?;
            g.copy_h2d(&bf16s(&mut seed, m * k, 1.0), a)?;
            let out = g.alloc(m * n * 2)?;
            let run = || {
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    a.0, w.0, out.0, m as u32, n as u32, k as u32, stream,
                )
            };
            run()?;
            g.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                run()?;
            }
            g.synchronize(stream)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / 5.0;
            let mut v = vec![0u8; m * n * 2];
            g.copy_d2h(out, &mut v)?;
            let h = v.iter().fold(0xcbf29ce484222325u64, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x100000001b3)
            });
            println!("M={m} N={n} K={k} hash {h:016x}");
            println!("M={m} N={n} K={k} time {ms:.3} ms");
            g.free(a)?;
            g.free(out)?;
        }
        g.free(w)?;
    }
    if std::env::var("ATLAS_LT_KCHAIN_PIN").as_deref() == Ok("1") {
        tile_vs_forced(g, stream, &mut seed)?;
    }
    Ok(())
}

/// The tile-kernel GEMMs the forced k-chain replaces: the mHC down GEMM at
/// the slab heights that run on the tile kernel, and the qwen4_exp TP2 GDN
/// out_proj at prefill row counts.
fn tile_vs_forced(g: &dyn GpuBackend, stream: u64, seed: &mut u64) -> Result<()> {
    let tile = g.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let mut bad = 0;
    let down: Vec<usize> = (1921usize..=2048).step_by(9).chain([2048]).collect();
    let out_proj = vec![2048usize, 2049, 5003, 8192, 9999, 16016];
    for (n, k, ms) in [(320usize, 10240usize, down), (2560, 3072, out_proj)] {
        let w = g.alloc(n * k * 2)?;
        g.copy_h2d(&bf16s(seed, n * k, 0.02), w)?;
        for m in ms {
            let a = g.alloc(m * k * 2)?;
            g.copy_h2d(&bf16s(seed, m * k, 1.0), a)?;
            let (o1, o2) = (g.alloc(m * n * 2)?, g.alloc(m * n * 2)?);
            let run_tile = || {
                KernelLaunch::new(g, tile)
                    .grid([(n as u32).div_ceil(128), (m as u32).div_ceil(128), 1])
                    .block([256, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(w)
                    .arg_ptr(o1)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(stream)
            };
            let run_forced = || {
                spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain(
                    a.0, w.0, o2.0, m as u32, n as u32, k as u32, stream,
                )
            };
            run_tile()?;
            let ran = run_forced()?;
            g.synchronize(stream)?;
            let (mut v1, mut v2) = (vec![0u8; m * n * 2], vec![0u8; m * n * 2]);
            g.copy_d2h(o1, &mut v1)?;
            g.copy_d2h(o2, &mut v2)?;
            let diff = v1.iter().zip(&v2).filter(|(x, y)| x != y).count();
            bad += usize::from(!ran || diff != 0);
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                g.synchronize(stream)?;
                let t0 = std::time::Instant::now();
                for _ in 0..5 {
                    f()?;
                }
                g.synchronize(stream)?;
                Ok(t0.elapsed().as_secs_f64() * 1e3 / 5.0)
            };
            let t_tile = time(&|| run_tile())?;
            let t_forced = time(&|| run_forced().map(|_| ()))?;
            println!(
                "forced k-chain vs tile kernel M={m} N={n} K={k}: ran {ran}, {diff} bytes differ, {t_tile:.3} -> {t_forced:.3} ms"
            );
            for p in [a, o1, o2] {
                g.free(p)?;
            }
        }
        g.free(w)?;
    }
    anyhow::ensure!(bad == 0, "{bad} shapes differ");
    println!("forced k-chain == tile kernel at every shape");
    Ok(())
}
