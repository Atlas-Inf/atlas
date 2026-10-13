// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-shaped benchmark + bitwise gate for the native-FP4 routed MoE
//! grouped GEMM (`moe_w4a4_grouped_gemm_prequant_t_k64_vecscale`, transposed
//! `[K/2, N]` expert weights, prequantized NVFP4 activations).
//!
//! Synthetic EP2 rank at a 4K chunk, launched as in serving: a grid over all
//! 288 experts, of which the second 144 are remote (NULL weights) but still
//! own sorted rows; 16384 routed rows on the 144 local experts with a
//! hot-expert skew, gate (N=2048, K=4096) and down (N=4096, K=2048).
//! Every candidate kernel in `CANDIDATES` must reproduce the reference
//! output bit for bit (same per-element K accumulation order).
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example moe_fp4_prefill_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[path = "moe_fp4_prefill_bench/gates.rs"]
mod gates;
use gates::{gate_up_silu, tile_worklist};

const MODULE: &str = "moe_w4a16";
const REFERENCE: &str = "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale";
/// (kernel, CTA N tile, threads per CTA, reads K-major `[N, K/2]` weights,
/// compact row-tile grid over `moe_mtile_prefix`).
const CANDIDATES: &[(&str, u32, u32, bool, bool)] = &[
    (
        "moe_w4a4_grouped_gemm_prequant_t_k128",
        128,
        256,
        false,
        false,
    ),
    (
        "moe_w4a4_grouped_gemm_prequant_t_k128w_compact",
        256,
        256,
        false,
        true,
    ),
    (
        "moe_w4a4_grouped_gemm_prequant_nk_k128",
        128,
        256,
        true,
        false,
    ),
];
/// Local experts; the grid covers twice as many (the remote rank's half).
const EXPERTS: usize = 144;
const GRID_EXPERTS: usize = 2 * EXPERTS;
/// Routed rows on one EP2 rank: 16384 at a 4K chunk (top-8 over two ranks);
/// `MOE_BENCH_ROWS` overrides (32768 = an 8K chunk).
fn rows_total() -> usize {
    std::env::var("MOE_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Expert row counts: a Zipf-like skew (hottest ~3K rows), summing to rows_total().
fn expert_rows() -> Vec<usize> {
    let weights: Vec<f64> = (0..EXPERTS)
        .map(|e| 1.0 / (1.0 + e as f64).powf(0.9))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut rows: Vec<usize> = weights
        .iter()
        .map(|w| (w / total * rows_total() as f64) as usize)
        .collect();
    let short = rows_total() - rows.iter().sum::<usize>();
    rows[EXPERTS - 1] += short;
    rows
}

/// Per-expert weight tables for `EXPERTS` local experts (the remote half
/// NULL): Atlas transposed `[K/2, N]` + `[K/16, N]`, and the same bytes
/// K-major `[N, K/2]` + `[N, K/16]` (each byte keeps its k-pair). Returns
/// (packed, scales, packed K-major, scales K-major) pointer tables and the
/// allocations to free.
fn expert_tables(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    n: usize,
    k: usize,
) -> Result<([DevicePtr; 4], Vec<DevicePtr>)> {
    let mut tables = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut owned = Vec::new();
    for _ in 0..EXPERTS {
        let w: Vec<u8> = (0..n * k / 2).map(|_| rng.next() as u8).collect();
        let s: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let mut w_nk = vec![0u8; w.len()];
        for kp in 0..k / 2 {
            for c in 0..n {
                w_nk[c * (k / 2) + kp] = w[kp * n + c];
            }
        }
        let mut s_nk = vec![0u8; s.len()];
        for gi in 0..k / 16 {
            for c in 0..n {
                s_nk[c * (k / 16) + gi] = s[gi * n + c];
            }
        }
        for (table, bytes) in tables.iter_mut().zip([&w, &s, &w_nk, &s_nk]) {
            let d = up(g, bytes)?;
            table.extend_from_slice(&d.0.to_le_bytes());
            owned.push(d);
        }
    }
    let mut ptrs = [DevicePtr(0); 4];
    for (p, mut table) in ptrs.iter_mut().zip(tables) {
        table.resize(GRID_EXPERTS * 8, 0);
        *p = up(g, &table)?;
        owned.push(*p);
    }
    Ok((ptrs, owned))
}

fn time(g: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() / 10.0)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    threads: u32,
    args: &[DevicePtr; 8],
    n: u32,
    k: u32,
    max_m_tiles: u32,
    prefix: Option<DevicePtr>,
) -> Result<()> {
    let z = if prefix.is_some() {
        1
    } else {
        GRID_EXPERTS as u32
    };
    let mut l = KernelLaunch::new(g, kernel)
        .grid([div_ceil(n, n_tile), max_m_tiles, z])
        .block([threads, 1, 1]);
    for p in args {
        l = l.arg_ptr(*p);
    }
    l = l.arg_u32(GRID_EXPERTS as u32).arg_u32(n).arg_u32(k);
    if let Some(p) = prefix {
        l = l.arg_ptr(p);
    }
    l.launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let reference = g.kernel(MODULE, REFERENCE)?;
    let rows = expert_rows();
    let max_m_tiles = rows.iter().map(|r| r.div_ceil(64)).max().unwrap_or(1) as u32;
    // Remote experts mirror the local histogram; their rows follow the local ones.
    let mut offsets = vec![0i32];
    for r in rows.iter().chain(rows.iter()) {
        offsets.push(offsets.last().unwrap() + *r as i32);
    }
    let all_rows = 2 * rows_total();
    // Compact grid bound as served without an offsets download: every local
    // row in M64 tiles plus one partial tile per local expert.
    let tile_bound = (rows_total().div_ceil(64) + EXPERTS) as u32;
    let prefix_k = g.kernel(MODULE, "moe_mtile_prefix")?;
    let d_prefix = g.alloc((GRID_EXPERTS + 1) * 4)?;
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let d_off = up(g, &off_bytes)?;
    let mut rng = Lcg(0xF4F4);
    let mut fail = false;
    for (name, n, k) in [
        ("gate [2048 x 4096]", 2048u32, 4096u32),
        ("down [4096 x 2048]", 4096, 2048),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        // A: packed E2M1 rows + UE4M3 group scales in a sane exponent range.
        let a: Vec<u8> = (0..all_rows * ku / 2).map(|_| rng.next() as u8).collect();
        let a_s: Vec<u8> = (0..all_rows * ku / 16)
            .map(|_| 0x30 + (rng.next() % 16) as u8)
            .collect();
        let (d_a, d_as) = (up(g, &a)?, up(g, &a_s)?);
        let ([d_pp, d_sp, d_ppk, d_spk], owned) = expert_tables(g, &mut rng, nu, ku)?;
        let scale2: Vec<u8> = (0..GRID_EXPERTS)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect();
        let d_s2 = up(g, &scale2)?;
        let c_bytes = all_rows * nu * 2;
        let (c_ref, c_new) = (g.alloc(c_bytes)?, g.alloc(c_bytes)?);
        g.memset(c_ref, 0, c_bytes)?;
        let args = |c| [d_a, d_as, d_pp, d_sp, d_s2, c, d_off, DevicePtr(0)];
        let args_nk = |c| [d_a, d_as, d_ppk, d_spk, d_s2, c, d_off, DevicePtr(0)];
        let flop = 2.0 * rows_total() as f64 * n as f64 * k as f64;
        let run_ref = || {
            launch(
                g,
                reference,
                128,
                128,
                &args(c_ref),
                n,
                k,
                max_m_tiles,
                None,
            )
        };
        let t_ref = time(g, &run_ref)?;
        println!(
            "{name}: reference {:7.1}us {:5.1} TFLOPS (max_m_tiles {max_m_tiles})",
            t_ref * 1e6,
            flop / t_ref / 1e12
        );
        let mut want = vec![0u8; c_bytes];
        g.copy_d2h(c_ref, &mut want)?;
        for &(kname, n_tile, threads, k_major, compact) in CANDIDATES {
            let Ok(kernel) = g.kernel(MODULE, kname) else {
                println!("  {kname}: absent");
                continue;
            };
            g.memset(c_new, 0, c_bytes)?;
            let a = if k_major { args_nk(c_new) } else { args(c_new) };
            let run = || {
                if !compact {
                    return launch(g, kernel, n_tile, threads, &a, n, k, max_m_tiles, None);
                }
                KernelLaunch::new(g, prefix_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(d_off)
                    .arg_ptr(a[2])
                    .arg_ptr(d_prefix)
                    .arg_u32(GRID_EXPERTS as u32)
                    .launch(0)?;
                launch(
                    g,
                    kernel,
                    n_tile,
                    threads,
                    &a,
                    n,
                    k,
                    tile_bound,
                    Some(d_prefix),
                )
            };
            let t = time(g, &run)?;
            let mut got = vec![0u8; c_bytes];
            g.copy_d2h(c_new, &mut got)?;
            let diff = want
                .chunks_exact(2)
                .zip(got.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            fail |= diff != 0;
            println!(
                "  {kname}: {:7.1}us {:5.1} TFLOPS  {}",
                t * 1e6,
                flop / t / 1e12,
                if diff == 0 {
                    "bitwise".to_string()
                } else {
                    format!("MISMATCH {diff} values")
                }
            );
        }
        for p in owned.into_iter().chain([d_a, d_as, d_s2, c_ref, c_new]) {
            g.free(p)?;
        }
    }
    fail |= gate_up_silu(g, &mut rng, d_off, d_prefix, tile_bound, all_rows)?;
    fail |= tile_worklist(g, &offsets)?;
    std::process::exit(if fail { 1 } else { 0 });
}
