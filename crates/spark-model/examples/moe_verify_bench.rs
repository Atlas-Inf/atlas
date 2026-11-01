// SPDX-License-Identifier: AGPL-3.0-only
//! Cold-weight DRAM bandwidth of the routed-MoE native-FP4 grouped GEMMs at
//! GLM-5.3 Flash decode/verify shapes under expert TP: 288 experts, all
//! local, H=4096, per-rank expert intermediate I=1024, top-8 routing of T
//! tokens over U distinct experts. Times the kernels production launches,
//! through the production `ops` wrappers:
//!   * compact gate+up  `moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up`
//!     over a `moe_build_tile_worklist` M64 x N128 worklist;
//!   * dense down       `moe_w4a4_grouped_gemm_prequant_t_k128` (N/128, m_tiles, E);
//!   * K128W gate+up+SiLU `moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w`
//!     and K128W down  `moe_w4a4_grouped_gemm_prequant_t_k128w_compact`, both
//!     over the `moe_mtile_prefix` row-tile grid.
//!
//! Each rep is bracketed by CUDA events on one stream (median of the reps).
//! Two cold regimes, both over a ring of routings whose active expert sets
//! are disjoint (consecutive reps never share weights):
//!   * flush:  a 256 MB scratch memset before every rep (evicts L2, but
//!     leaves it dirty, so the kernel pays some write-back);
//!   * rotate: no memset, the disjoint expert ring alone keeps weights cold.
//! GB/s = weight + block-scale bytes of the U active experts / time.
//!
//! Exit: 0 ok, 1 non-finite output, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example moe_verify_bench -- [T:U ...]
//! Default workloads 8:41 16:62 32:90; `MOE_VERIFY_REPS` (default 30).

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[path = "moe_verify_bench/data.rs"]
mod data;
#[path = "moe_verify_bench/timing.rs"]
mod timing;
use data::{Proj, Rng, le, routing, up};
use timing::{Timer, ctas_per_sm};

const MODULE: &str = "moe_w4a16";
const EXPERTS: usize = 288;
const H: usize = 4096;
const INTER: usize = 1024;
const TOP_K: usize = 8;
const FLUSH_BYTES: usize = 256 << 20;
/// Packed E2M1 + E4M3 block scales of one [1024 x 4096] or [4096 x 1024] matrix.
const PROJ_BYTES: usize = INTER * H / 2 + INTER * H / 16;

/// One routing of the ring, with its device offsets, sorted ids, compact
/// gate/up worklist (`[total, pad, pad, pad, items...]`) and M64 prefix.
struct Route {
    offsets: DevicePtr,
    sorted: DevicePtr,
    work: DevicePtr,
    prefix: DevicePtr,
    distinct: usize,
    /// M64 row tiles over all experts.
    mtiles: usize,
    max_rows: i32,
}

struct Kernels {
    gate_up_compact: KernelHandle,
    k128: KernelHandle,
    gate_up_silu: KernelHandle,
    k128w: KernelHandle,
    prefix: KernelHandle,
    worklist: KernelHandle,
}

fn kernels(g: &dyn GpuBackend) -> Option<Kernels> {
    Some(Kernels {
        gate_up_compact: g
            .kernel(
                MODULE,
                "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up",
            )
            .ok()?,
        k128: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128")
            .ok()?,
        gate_up_silu: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w")
            .ok()?,
        k128w: g
            .kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128w_compact")
            .ok()?,
        prefix: g.kernel(MODULE, "moe_mtile_prefix").ok()?,
        worklist: g.kernel("moe", "moe_build_tile_worklist").ok()?,
    })
}

/// Count of BF16 values that are Inf/NaN.
fn non_finite_bf16(g: &dyn GpuBackend, p: DevicePtr, values: usize) -> Result<usize> {
    let mut v = vec![0u8; values * 2];
    g.copy_d2h(p, &mut v)?;
    Ok(v.chunks_exact(2)
        .filter(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7F80 == 0x7F80)
        .count())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let Some(k) = kernels(g) else {
        eprintln!("moe_verify_bench: prequant FP4 grouped kernels absent from this target");
        std::process::exit(2);
    };
    let workloads: Vec<(usize, usize)> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let args = if args.is_empty() {
            vec!["8:41".into(), "16:62".into(), "32:90".into()]
        } else {
            args
        };
        args.iter()
            .map(|a| {
                let (t, u) = a
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("workload {a}: want T:U"))?;
                Ok((t.parse()?, u.parse()?))
            })
            .collect::<Result<_>>()?
    };
    let reps = std::env::var("MOE_VERIFY_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30usize)
        .max(1);
    let stream = g.create_stream()?;
    let timer = Timer {
        stream,
        scratch: g.alloc(FLUSH_BYTES)?,
        reps,
    };
    let sms = g.sm_count()?;

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // Output scales that keep the SiLU inputs in range (as moe_fp4_prefill_bench).
    let gate = Proj::new(g, &mut rng, INTER, H, 1.0 / 256.0)?;
    let upp = Proj::new(g, &mut rng, INTER, H, 1.0 / 128.0)?;
    let down = Proj::new(g, &mut rng, H, INTER, 1.0 / 256.0)?;
    let (e, h, inter) = (EXPERTS as u32, H as u32, INTER as u32);
    let n_tiles = inter / 128;
    println!(
        "moe_verify_bench: E={EXPERTS} (all local) H={H} I={INTER} top-{TOP_K}, {sms} SMs, median of {reps} reps, \
         cold via disjoint-expert ring (+{} MB memset for flush)",
        FLUSH_BYTES >> 20
    );
    println!(
        "GB/s = weight+scale bytes of the U active experts / time (streaming-read ceiling ~235 GB/s)"
    );
    println!(
        "{:>3} {:>4} {:>5} {:<22} {:>16} {:>6} {:>7} {:>6} {:>8} | {:>9} {:>6} | {:>9} {:>6}",
        "T",
        "U",
        "rows",
        "kernel",
        "grid",
        "live",
        "CTA/SM",
        "waves",
        "MB",
        "flush us",
        "GB/s",
        "rotate us",
        "GB/s"
    );
    let mut fail = false;
    for (t, u) in workloads {
        if !(TOP_K..=EXPERTS).contains(&u) || t * TOP_K < u {
            bail!("workload T={t} U={u}: need 8 <= U <= {EXPERTS} and T*8 >= U");
        }
        let rows = t * TOP_K;
        // Ring of disjoint active expert sets (one set when U > E/2).
        let mut perm: Vec<usize> = (0..EXPERTS).collect();
        rng.shuffle(&mut perm);
        let ring = (EXPERTS / u).max(1);
        let mut routes = Vec::with_capacity(ring);
        for r in 0..ring {
            let mut active = perm[r * u..(r + 1) * u].to_vec();
            active.sort_unstable();
            let (offsets, sorted) = routing(&mut rng, t, &active);
            let distinct = offsets.windows(2).filter(|w| w[1] > w[0]).count();
            let max_rows = offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
            let mtiles = offsets
                .windows(2)
                .map(|w| ((w[1] - w[0]) as usize).div_ceil(64))
                .sum();
            let offsets = up(g, &le(&offsets, i32::to_le_bytes))?;
            let sorted = up(g, &le(&sorted, i32::to_le_bytes))?;
            // As prequant_fp4::compact_gate_up_worklist_bytes: 16-byte total + items.
            let work = g.alloc(16 + rows * n_tiles as usize * 8)?;
            let prefix = g.alloc((EXPERTS + 1) * 4)?;
            ops::moe_build_tile_worklist(
                g,
                k.worklist,
                offsets,
                gate.packed,
                work.offset(16),
                work,
                e,
                n_tiles,
                64,
                stream,
            )?;
            ops::moe_mtile_prefix(g, k.prefix, offsets, gate.packed, prefix, e, stream)?;
            routes.push(Route {
                offsets,
                sorted,
                work,
                prefix,
                distinct,
                mtiles,
                max_rows,
            });
        }
        g.synchronize(stream)?;
        let (distinct, mtiles) = (routes[0].distinct, routes[0].mtiles);
        if routes.iter().any(|r| r.distinct != u) {
            println!(
                "  warning: routed distinct experts {:?} != U={u}",
                routes.iter().map(|r| r.distinct).collect::<Vec<_>>()
            );
        }
        let max_m_tiles =
            (routes.iter().map(|r| r.max_rows).max().unwrap_or(1).max(1) as u32).div_ceil(64);
        // Activations: token-major gate/up A (T rows x H) and sorted down A (rows x I).
        let a_gu = up(g, &rng.packed(t * H / 2))?;
        let a_gu_s = up(g, &rng.scales(t * H / 16))?;
        let a_dn = up(g, &rng.packed(rows * INTER / 2))?;
        let a_dn_s = up(g, &rng.scales(rows * INTER / 16))?;
        let c_gate = g.alloc(rows * INTER * 2)?;
        let c_up = g.alloc(rows * INTER * 2)?;
        let c_down = g.alloc(rows * H * 2)?;
        let (q_bytes, qs_bytes) = (rows * INTER / 2, rows * INTER / 16);
        let (silu_q, silu_s) = (g.alloc(q_bytes)?, g.alloc(qs_bytes)?);
        for (p, b) in [
            (c_gate, rows * INTER * 2),
            (c_up, rows * INTER * 2),
            (c_down, rows * H * 2),
            (silu_q, q_bytes),
            (silu_s, qs_bytes),
        ] {
            g.memset(p, 0, b)?;
        }
        let bound = (rows as u32).div_ceil(64) + e;
        let max_tiles = rows as u32 * n_tiles;
        let route = |i: usize| &routes[i % ring];
        let gu_bytes = (distinct * 2 * PROJ_BYTES) as f64;
        let dn_bytes = (distinct * PROJ_BYTES) as f64;
        // (label, kernel, threads, grid, live CTAs, bytes, launch)
        let cases: [(
            &str,
            KernelHandle,
            u32,
            [u32; 3],
            usize,
            f64,
            &dyn Fn(usize) -> Result<()>,
        ); 4] = [
            (
                "gate_up k64 compact",
                k.gate_up_compact,
                128,
                [max_tiles, 2, 1],
                mtiles * n_tiles as usize * 2,
                gu_bytes,
                &|i| {
                    let r = route(i);
                    ops::moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
                        g,
                        k.gate_up_compact,
                        a_gu,
                        a_gu_s,
                        gate.packed,
                        gate.scales,
                        gate.scale2,
                        c_gate,
                        upp.packed,
                        upp.scales,
                        upp.scale2,
                        c_up,
                        r.offsets,
                        r.sorted,
                        e,
                        inter,
                        h,
                        r.work.offset(16),
                        r.work,
                        max_tiles,
                        stream,
                    )
                },
            ),
            (
                "down k128 dense",
                k.k128,
                256,
                [h / 128, max_m_tiles, e],
                mtiles * (h / 128) as usize,
                dn_bytes,
                &|i| {
                    let r = route(i);
                    let [pp, sp, s2] = down.table();
                    ops::moe_w4a4_grouped_gemm_prequant_n128(
                        g,
                        k.k128,
                        a_dn,
                        a_dn_s,
                        pp,
                        sp,
                        s2,
                        c_down,
                        r.offsets,
                        DevicePtr::NULL,
                        e,
                        h,
                        inter,
                        max_m_tiles,
                        256,
                        stream,
                    )
                },
            ),
            (
                "gate_up_silu k128w",
                k.gate_up_silu,
                256,
                [inter / 128, bound, 1],
                mtiles * (inter / 128) as usize,
                gu_bytes,
                &|i| {
                    let r = route(i);
                    ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
                        g,
                        ops::K128wKernel {
                            grid: k.gate_up_silu,
                            persist: KernelHandle(0),
                        },
                        a_gu,
                        a_gu_s,
                        gate.table(),
                        upp.table(),
                        silu_q,
                        silu_s,
                        r.offsets,
                        r.sorted,
                        e,
                        inter,
                        h,
                        r.prefix,
                        ops::K128wSchedule::Grid { bound },
                        stream,
                    )
                },
            ),
            (
                "down k128w compact",
                k.k128w,
                256,
                [h / 256, bound, 1],
                mtiles * (h / 256) as usize,
                dn_bytes,
                &|i| {
                    let r = route(i);
                    let [pp, sp, s2] = down.table();
                    ops::moe_w4a4_grouped_gemm_prequant_k128w(
                        g,
                        ops::K128wKernel {
                            grid: k.k128w,
                            persist: KernelHandle(0),
                        },
                        a_dn,
                        a_dn_s,
                        pp,
                        sp,
                        s2,
                        c_down,
                        r.offsets,
                        DevicePtr::NULL,
                        e,
                        h,
                        inter,
                        r.prefix,
                        ops::K128wSchedule::Grid { bound },
                        stream,
                    )
                },
            ),
        ];
        for (label, kernel, threads, grid, live, bytes, f) in cases {
            let occ = ctas_per_sm(kernel, threads)?;
            let waves = live as f64 / (occ * sms).max(1) as f64;
            let flush = timer.time(g, true, f)?;
            let rotate = timer.time(g, false, f)?;
            println!(
                "{t:>3} {distinct:>4} {rows:>5} {label:<22} {:>16} {live:>6} {occ:>7} {waves:>6.2} {:>8.1} | {flush:>9.1} {:>6.1} | {rotate:>9.1} {:>6.1}",
                format!("{}x{}x{}", grid[0], grid[1], grid[2]),
                bytes / 1e6,
                bytes / flush / 1e3,
                bytes / rotate / 1e3,
            );
        }
        g.synchronize(stream)?;
        let bad = non_finite_bf16(g, c_gate, rows * INTER)?
            + non_finite_bf16(g, c_up, rows * INTER)?
            + non_finite_bf16(g, c_down, rows * H)?;
        let mut s = vec![0u8; qs_bytes];
        g.copy_d2h(silu_s, &mut s)?;
        let bad_s = s.iter().filter(|&&b| b & 0x7F == 0x7F).count();
        if bad + bad_s != 0 {
            println!("  NON-FINITE: {bad} BF16 outputs, {bad_s} SiLU NVFP4 scales");
            fail = true;
        }
        for p in [
            a_gu, a_gu_s, a_dn, a_dn_s, c_gate, c_up, c_down, silu_q, silu_s,
        ]
        .into_iter()
        .chain(
            routes
                .iter()
                .flat_map(|r| [r.offsets, r.sorted, r.work, r.prefix]),
        ) {
            g.free(p)?;
        }
    }
    for p in gate
        .owned
        .into_iter()
        .chain(upp.owned)
        .chain(down.owned)
        .chain([timer.scratch])
    {
        g.free(p)?;
    }
    std::process::exit(if fail { 1 } else { 0 });
}
