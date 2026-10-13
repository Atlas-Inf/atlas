// SPDX-License-Identifier: AGPL-3.0-only
//! BF16 prefill GEMM throughput and bit-identity at GLM-5.3 Flash TP2 shapes:
//! cuBLASLt's heuristic vs the CUTLASS configs of
//! `spark_runtime::cutlass::bf16_gemm_tuned` vs `dense_gemm_bf16_pipelined`
//! (KDA's small projections, and their beta|f_a|g_a triple), plus the
//! head-batched MLA absorb / V-up GEMMs (cuBLASLt vs CUTLASS batched).
//! Timings are cold-L2: every rep follows its own 64 MiB memset and is
//! bracketed by CUDA events; the median is reported. `=N` counts BF16 outputs
//! that differ from the reference (cuBLASLt for CUTLASS, cfg9 for the
//! pipelined kernel, three pipelined launches for the triple); 0 means
//! bit-identical.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example bf16_gemm_bench

use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[allow(dead_code)]
#[path = "moe_verify_bench/timing.rs"]
mod timing;

/// Rows (`BF16_BENCH_M` overrides, e.g. 8196 for an 8K prefill chunk).
fn rows() -> u32 {
    std::env::var("BF16_BENCH_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}
const CONFIGS: [u32; 4] = [0, 4, 5, 9];
const SHAPES: [(&str, u32, u32); 10] = [
    ("q_a      [1536 x 4096]", 1536, 4096),
    ("q_b      [8192 x 1536]", 8192, 1536),
    ("o        [4096 x 8192]", 4096, 8192),
    ("kv_a     [ 512 x 4096]", 512, 4096),
    ("router   [ 288 x 4096]", 288, 4096),
    ("index wk [ 128 x 4096]", 128, 4096),
    ("weights  [  32 x 4096]", 32, 4096),
    ("kda f_b  [4096 x  128]", 4096, 128),
    ("shared g [2048 x 4096]", 2048, 4096),
    ("shared d [4096 x 2048]", 4096, 2048),
];
const FLUSH_BYTES: usize = 64 << 20;

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let pipelined = g.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let triple = g.kernel("gemm", "dense_gemm_bf16_pipelined_triple_n")?;
    let mut rng = 0x1234_5678u64;
    let mut fill = |p: DevicePtr, count: usize| -> Result<()> {
        let bytes: Vec<u8> = (0..count)
            .flat_map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                bf16::from_f32(v).to_bits().to_le_bytes()
            })
            .collect();
        g.copy_h2d(&bytes, p)
    };
    let timer = timing::Timer {
        stream: 0,
        scratch: g.alloc(FLUSH_BYTES)?,
        reps: 15,
    };
    // Cold-L2 seconds per call.
    let time =
        |f: &dyn Fn() -> Result<()>| -> Result<f64> { Ok(timer.time(g, true, &|_| f())? * 1e-6) };
    let bits = |p: DevicePtr, count: usize| -> Result<Vec<u16>> {
        let mut b = vec![0u8; count * 2];
        g.copy_d2h(p, &mut b)?;
        Ok(b.chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect())
    };
    let diff = |a: DevicePtr, b: DevicePtr, count: usize| -> Result<usize> {
        let (x, y) = (bits(a, count)?, bits(b, count)?);
        Ok(x.iter().zip(&y).filter(|(p, q)| p != q).count())
    };

    #[allow(non_snake_case)]
    let M = rows();
    for (name, n, k) in SHAPES {
        let (mu, nu, ku) = (M as usize, n as usize, k as usize);
        let (a, w) = (g.alloc(mu * ku * 2)?, g.alloc(nu * ku * 2)?);
        let (c, c2) = (g.alloc(mu * nu * 2)?, g.alloc(mu * nu * 2)?);
        fill(a, mu * ku)?;
        fill(w, nu * ku)?;
        let tf = |s: f64| 2.0 * M as f64 * n as f64 * k as f64 / s / 1e12;
        let lt =
            time(&|| spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, w.0, c.0, M, n, k, 0))?;
        let mut line = format!("{name}: cuBLASLt {:5.1}TF | CUTLASS", tf(lt));
        for config in CONFIGS {
            let run = || {
                spark_runtime::cutlass::bf16_gemm_tuned(a.0, w.0, c2.0, M, n, k, k, n, config, 0)
            };
            match run() {
                Ok(()) => {
                    let t = time(&run)?;
                    line += &format!(" {config}:{:5.1}={}", tf(t), diff(c, c2, mu * nu)?);
                }
                Err(_) => line += &format!(" {config}:  err"),
            }
        }
        if n <= 128 || k == 128 {
            // KDA's pipelined kernel (into c) against CUTLASS cfg9 (into c2).
            spark_runtime::cutlass::bf16_gemm_tuned(a.0, w.0, c2.0, M, n, k, k, n, 9, 0)?;
            let weight = DenseWeight { weight: w };
            let run = || ops::dense_gemm_bf16_pipelined(g, pipelined, a, &weight, c, M, n, k, 0);
            let t = time(&run)?;
            line += &format!(" | pipelined {:5.1}={}", tf(t), diff(c, c2, mu * nu)?);
        }
        println!("{line}");
        for p in [a, w, c, c2] {
            g.free(p)?;
        }
    }

    // KDA beta | f_a | g_a (N = 32/128/128 over the same activation):
    // three pipelined launches vs the one-grid triple.
    let (h, heads, dim) = (4096u32, 32u32, 128u32);
    let out = (M * (heads + 2 * dim)) as usize;
    let a = g.alloc(M as usize * h as usize * 2)?;
    let w: Vec<DenseWeight> = [heads, dim, dim]
        .iter()
        .map(|&n| {
            let p = g.alloc((n * h) as usize * 2)?;
            fill(p, (n * h) as usize).map(|()| DenseWeight { weight: p })
        })
        .collect::<Result<_>>()?;
    let (c, c2) = (g.alloc(out * 2)?, g.alloc(out * 2)?);
    fill(a, M as usize * h as usize)?;
    let planes = |base: DevicePtr| {
        let fa = base.offset(M as usize * heads as usize * 2);
        [base, fa, fa.offset(M as usize * dim as usize * 2)]
    };
    let three = time(&|| {
        for (i, (wi, ci)) in w.iter().zip(planes(c)).enumerate() {
            let n = if i == 0 { heads } else { dim };
            ops::dense_gemm_bf16_pipelined(g, pipelined, a, wi, ci, M, n, h, 0)?;
        }
        Ok(())
    })?;
    let one = time(&|| {
        ops::dense_gemm_pipelined_triple_n(
            g,
            triple,
            a,
            [&w[0], &w[1], &w[2]],
            planes(c2),
            M,
            [heads, dim],
            h,
            0,
        )
    })?;
    println!(
        "kda beta|f_a|g_a: 3x pipelined {:7.1}us | triple {:7.1}us ={}",
        three * 1e6,
        one * 1e6,
        diff(c, c2, out)?
    );
    for p in [a, c, c2].into_iter().chain(w.iter().map(|w| w.weight)) {
        g.free(p)?;
    }

    // MLA absorb (q_nope -> latent) and V-up (latent -> v): 32 TP-local heads,
    // token rows interleaved across heads.
    for (name, gk, gn) in [
        ("absorb 32x[256->512]", 256u32, 512u32),
        ("v-up   32x[512->256]", 512, 256),
    ] {
        let (a_stride, c_stride) = (heads * gk, heads * gn);
        let rows = M as usize * c_stride as usize;
        let a = g.alloc(M as usize * a_stride as usize * 2)?;
        let w = g.alloc((heads * gk * gn) as usize * 2)?;
        let (c, c2) = (g.alloc(rows * 2)?, g.alloc(rows * 2)?);
        fill(a, M as usize * a_stride as usize)?;
        fill(w, (heads * gk * gn) as usize)?;
        let bytes = 2.0 * M as f64 * (a_stride + c_stride) as f64;
        let lt = time(&|| {
            spark_runtime::cublaslt::bf16_grouped_gemm_act_weight_t(
                a.0, w.0, c.0, M, heads, gn, gk, a_stride, c_stride, 0,
            )
        })?;
        let cu = time(&|| {
            spark_runtime::cutlass::bf16_grouped_gemm_act_weight_t(
                a.0, w.0, c2.0, M, heads, gn, gk, a_stride, c_stride, 0,
            )
        })?;
        println!(
            "{name}: cuBLASLt {:7.1}us {:4.0}GB/s | CUTLASS {:7.1}us {:4.0}GB/s ={}",
            lt * 1e6,
            bytes / lt / 1e9,
            cu * 1e6,
            bytes / cu / 1e9,
            diff(c, c2, rows)?
        );
        for p in [a, w, c, c2] {
            g.free(p)?;
        }
    }
    Ok(())
}
