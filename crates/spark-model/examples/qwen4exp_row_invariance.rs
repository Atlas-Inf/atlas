// SPDX-License-Identifier: AGPL-3.0-only
//! Is a qwen4_exp prefill projection ROW-INVARIANT: does a sequence's row get
//! the same bytes when its rows share the launch with other sequences' rows?
//! That is the precondition for an exact multi-sequence prefill (one forward
//! over several short prompts, each prompt's logits = its solo prefill's).
//!
//! For each production launcher and qwen4_exp TP2 shape, a "solo" run over
//! `m` rows is compared with the same rows placed at row offset `off` inside a
//! "batch" run of `big` rows (the other rows random): any differing byte means
//! the projection is not row-invariant between those row counts.
//!
//! Families: cuBLASLt BF16 opT (`bf16_gemm_act_weight_t`, the small-M path of
//! `ops::bf16_gemm` and of the mHC skinny projections), cuBLASLt BF16 opN (the
//! mHC up projection), and the in-order tile kernel `dense_gemm_bf16_pipelined`.
//!
//! Run inside atlas-release-builder (the runtime image's cuBLASLt) on a GB10:
//!   cargo build -p spark-model --release --features cuda,gpu-examples \
//!     --example qwen4exp_row_invariance
//!   ./qwen4exp_row_invariance            # summary per (family, shape, m, big)

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
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

#[derive(Clone, Copy, PartialEq)]
enum Family {
    LtOpT,
    LtOpN,
    Tile,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::LtOpT => "cublaslt_opT",
            Family::LtOpN => "cublaslt_opN",
            Family::Tile => "tile_pipelined",
        }
    }
}

/// `out[m, n] = a[m, k] x w`, `w` as the family lays it out.
#[allow(clippy::too_many_arguments)]
fn gemm(
    g: &dyn GpuBackend,
    f: Family,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let (m, n, k) = (m as u32, n as u32, k as u32);
    match f {
        Family::LtOpT => {
            spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, w.0, out.0, m, n, k, stream)
        }
        Family::LtOpN => {
            spark_runtime::cublaslt::bf16_gemm_act_weight_n(a.0, w.0, out.0, m, n, k, stream)
        }
        Family::Tile => KernelLaunch::new(g, g.kernel("gemm", "dense_gemm_bf16_pipelined")?)
            .grid([n.div_ceil(128), m.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream),
    }
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut seed = 0x5eed_u64;
    // qwen4_exp TP2 prefill projections that run on these launchers at short
    // prompt row counts: GDN in_proj and out_proj, attention o_proj, the mHC
    // collapse's down (N=320) and injection (N=4) on cuBLASLt (`hc_gemm`'s
    // machine-fill rule below 2048 rows) and its up projection (opN).
    let shapes: [(Family, &str, usize, usize); 9] = [
        (Family::LtOpT, "gdn_in_proj", 8192, 2560),
        (Family::LtOpT, "gdn_out_proj/o_proj", 2560, 3072),
        (Family::LtOpT, "hc_down", 320, 10240),
        (Family::LtOpT, "hc_inject", 4, 10240),
        (Family::LtOpN, "hc_up", 10240, 320),
        (Family::Tile, "gdn_in_proj", 8192, 2560),
        (Family::Tile, "hc_down", 320, 10240),
        (Family::Tile, "gdn_out_proj/o_proj", 2560, 3072),
        (Family::Tile, "hc_up_nt", 10240, 320),
    ];
    // Solo row counts of short chat prompts, and the admission waves they
    // would join (8 prompts of ~46 and ~54 rows, and 512).
    let solos = [1usize, 7, 16, 22, 30, 32, 39, 46, 54, 64, 96, 128, 200];
    let bigs = [368usize, 432, 512];
    let mut summary: Vec<String> = Vec::new();
    for (f, name, n, k) in shapes {
        let w = g.alloc(n * k * 2)?;
        g.copy_h2d(&bf16s(&mut seed, n * k, 0.02), w)?;
        for &big in &bigs {
            let a = g.alloc(big * k * 2)?;
            g.copy_h2d(&bf16s(&mut seed, big * k, 1.0), a)?;
            let ob = g.alloc(big * n * 2)?;
            gemm(g, f, a, w, ob, big, n, k, stream)?;
            g.synchronize(stream)?;
            let mut vb = vec![0u8; big * n * 2];
            g.copy_d2h(ob, &mut vb)?;
            let mut bad_ms = Vec::new();
            for &m in &solos {
                for off in [0usize, 46.min(big - m)] {
                    let os = g.alloc(m * n * 2)?;
                    gemm(g, f, a.offset(off * k * 2), w, os, m, n, k, stream)?;
                    g.synchronize(stream)?;
                    let mut vs = vec![0u8; m * n * 2];
                    g.copy_d2h(os, &mut vs)?;
                    g.free(os)?;
                    let rows = &vb[off * n * 2..(off + m) * n * 2];
                    let diff = rows.iter().zip(&vs).filter(|(x, y)| x != y).count();
                    println!(
                        "{} {name} N={n} K={k} solo={m} off={off} batch={big}: {diff} of {} bytes differ",
                        f.name(),
                        m * n * 2
                    );
                    if diff > 0 && !bad_ms.contains(&m) {
                        bad_ms.push(m);
                    }
                }
            }
            summary.push(format!(
                "{:<15} {name:<20} N={n:<5} K={k:<5} batch={big}: {}",
                f.name(),
                if bad_ms.is_empty() {
                    "row-invariant at every solo m".to_string()
                } else {
                    format!("DIFFERS at solo m = {bad_ms:?}")
                }
            ));
            g.free(a)?;
            g.free(ob)?;
        }
        g.free(w)?;
    }
    println!("\n== summary");
    for s in summary {
        println!("{s}");
    }
    Ok(())
}
