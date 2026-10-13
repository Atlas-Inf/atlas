// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-level checks behind `ATLAS_QWEN4EXP_PREFILL_ROWINV=1`, at the
//! qwen4_exp TP2 rank shapes, on a GB10 (run inside atlas-release-builder for
//! the runtime image's cuBLASLt):
//!
//! 1. `fixed`: the BF16 projections ROWINV pins (`qwen4exp_rowinv::
//!    try_bf16_gemm`: N >= 2048 on the in-order k-chain, narrower shapes on
//!    one cuBLASLt configuration, the heuristic's pick at a reference row
//!    count -- 512 for the mHC down / injection, 2048 otherwise;
//!    `PROBE_REF` forces one reference for every shape). Solo passes over 1..513 rows
//!    against the same rows inside 368/1920/2048-row passes at two offsets,
//!    run-to-run repeats, and microseconds a call against the heuristic
//!    (`PROBE_MS`, `PROBE_SHAPE`).
//! 2. `hc`: for `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ`, the prefill mHC collapse
//!    (`hc_pre_stage_vec` + `hc_mma_down_rows` + `hc_mma_finish_rows` over
//!    a 1920-row slab) against the DECODE collapse (`hc_mma_down` +
//!    `hc_mma_finish`, 1..32 rows) of the same rows: differing `y` / `inj`
//!    bytes. Zero means a prefill row's collapse is its decode collapse,
//!    byte for byte.
//!
//!   cargo build -p spark-model --release --features cuda,gpu-examples \
//!     --example qwen4exp_rowinv_probe

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

fn bf16s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    f32s(seed, n, scale)
        .chunks(4)
        .flat_map(|b| {
            let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (((f.to_bits()) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn f32s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            (f * scale).to_le_bytes()
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

const H: usize = 2560;
const HC: usize = 4;
const RANK: usize = 320;
const D: usize = HC * H;

struct HcW {
    norm: DevicePtr,
    down: DevicePtr,
    inject: DevicePtr,
    up: DevicePtr,
}

/// stage rows of `streams` into `normed` (FP32).
fn stage(g: &dyn GpuBackend, w: &HcW, streams: DevicePtr, normed: DevicePtr, t: u32) -> Result<()> {
    KernelLaunch::new(g, g.kernel("hyper_connection", "hc_pre_stage_vec")?)
        .grid([t, if t >= 48 { 1 } else { 8 }, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm)
        .arg_ptr(normed)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_f32(1e-6)
        .launch(g.default_stream())
}

/// down + finish; `rows` picks the prefill twins (groups on blockIdx.z).
fn collapse(
    g: &dyn GpuBackend,
    w: &HcW,
    rows: bool,
    [normed, low, y, inj]: [DevicePtr; 4],
    t: u32,
) -> Result<()> {
    let s = g.default_stream();
    let (dn, fin, wm, gd, gf) = if rows {
        ("hc_mma_down_rows", "hc_mma_finish_rows", 7u32, 96u32, 64u32)
    } else {
        ("hc_mma_down", "hc_mma_finish", 3, 32, 32)
    };
    let out_rows = (RANK + HC) as u32;
    KernelLaunch::new(g, g.kernel("qwen4exp_hc_mma", dn)?)
        .grid([out_rows.div_ceil(16).div_ceil(wm), 8, t.div_ceil(gd)])
        .block([32 * wm, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down)
        .arg_ptr(w.inject)
        .arg_ptr(low)
        .arg_ptr(inj)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_u32(RANK as u32)
        .arg_u32(t)
        .launch(s)?;
    let nt = t.min(gf).div_ceil(8);
    let smem = if rows {
        4 * nt * 8 * 32 * 4
    } else {
        // hc_mma_finish_smem: staged fragments or the mean tile.
        (RANK as u32 / 16 * nt * 32 * 16).max(4 * nt * 8 * 32 * 4)
    };
    KernelLaunch::new(g, g.kernel("qwen4exp_hc_mma", fin)?)
        .grid([H as u32 / 32, 1, t.div_ceil(gf)])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up)
        .arg_ptr(y)
        .arg_u32(H as u32)
        .arg_u32(RANK as u32)
        .arg_u32(t)
        .launch(s)
}

fn check_hc(g: &dyn GpuBackend, seed: &mut u64) -> Result<()> {
    let w = HcW {
        norm: upload(g, &bf16s(seed, D, 0.5))?,
        down: upload(g, &bf16s(seed, RANK * D, 0.02))?,
        inject: upload(g, &bf16s(seed, HC * D, 0.02))?,
        up: upload(g, &bf16s(seed, RANK * D, 0.02))?,
    };
    let big = 1920usize;
    let streams = upload(g, &f32s(seed, big * D, 1.0))?;
    let buf = |n: usize| g.alloc(n);
    let (nb, lb, yb, ib) = (
        buf(big * D * 4)?,
        buf(big * RANK * 4)?,
        buf(big * H * 2)?,
        buf(big * HC * 4)?,
    );
    stage(g, &w, streams, nb, big as u32)?;
    collapse(g, &w, true, [nb, lb, yb, ib], big as u32)?;
    g.synchronize(g.default_stream())?;
    let (yv, iv) = (read(g, yb, big * H * 2)?, read(g, ib, big * HC * 4)?);
    let (ns, ls, ys, is) = (
        buf(32 * D * 4)?,
        buf(32 * RANK * 4)?,
        buf(32 * H * 2)?,
        buf(32 * HC * 4)?,
    );
    let mut worst = (0usize, 0usize);
    let mut cases = 0;
    for t in [1usize, 2, 5, 8, 9, 17, 31, 32] {
        for off in [0usize, 95, 96, 1000, big - t] {
            stage(g, &w, streams.offset(off * D * 4), ns, t as u32)?;
            collapse(g, &w, false, [ns, ls, ys, is], t as u32)?;
            g.synchronize(g.default_stream())?;
            let dy = diff(
                &yv[off * H * 2..(off + t) * H * 2],
                &read(g, ys, t * H * 2)?,
            );
            let di = diff(
                &iv[off * HC * 4..(off + t) * HC * 4],
                &read(g, is, t * HC * 4)?,
            );
            worst = (worst.0.max(dy), worst.1.max(di));
            cases += 1;
        }
    }
    println!(
        "hc   prefill rows twins (1920-row slab) vs decode collapse (1..32 rows), {cases} cases: \
         worst y {} B, inj {} B differ",
        worst.0, worst.1
    );
    Ok(())
}

/// ROWINV's BF16 projections (see the module docs): row invariance (solo
/// 1..513 rows inside 368/1920/2048-row passes), run-to-run determinism, and
/// microseconds a call against the heuristic.
fn check_fixed(g: &dyn GpuBackend, seed: &mut u64) -> Result<()> {
    let s = g.default_stream();
    // `qwen4exp_rowinv::reference_rows`; PROBE_REF forces one for every shape.
    let ref_of = |n: usize, k: usize| if n <= 512 && k >= 8192 { 512 } else { 2048 };
    let ref_env: Option<u32> = std::env::var("PROBE_REF").ok().and_then(|v| v.parse().ok());
    let only = std::env::var("PROBE_SHAPE").ok();
    let shapes = [
        ("hc_down", 320, 10240),
        ("hc_inject", 4, 10240),
        ("qsa_qk", 640, 2560),
        ("gdn_in_proj", 8192, 2560),
        ("gdn_in_proj_tp1", 16384, 2560),
    ];
    let shapes = shapes
        .into_iter()
        .filter(|s| only.as_deref().is_none_or(|o| o == s.0));
    // `qwen4exp_rowinv::try_bf16_gemm`: wide shapes on the k-chain (any
    // offered algo-21 kernel, else the tile kernel), narrow ones on one
    // configuration.
    let fixed = |a: DevicePtr, w: DevicePtr, o: DevicePtr, mnk: [u32; 3]| -> Result<()> {
        if mnk[1] >= 2048 && ref_env.is_none() {
            let lt =
                spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(a.0, w.0, o.0, mnk, s)?;
            if !lt {
                KernelLaunch::new(g, g.kernel("gemm", "dense_gemm_bf16_pipelined")?)
                    .grid([mnk[1].div_ceil(128), mnk[0].div_ceil(128), 1])
                    .block([256, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(w)
                    .arg_ptr(o)
                    .arg_u32(mnk[0])
                    .arg_u32(mnk[1])
                    .arg_u32(mnk[2])
                    .launch(s)?;
            }
            return Ok(());
        }
        let ref_m = ref_env.unwrap_or(ref_of(mnk[1] as usize, mnk[2] as usize));
        anyhow::ensure!(
            spark_runtime::cublaslt::bf16_gemm_act_weight_t_fixed(a.0, w.0, o.0, mnk, ref_m, s)?,
            "fixed config refused {mnk:?}"
        );
        Ok(())
    };
    for (name, n, k) in shapes {
        let w = upload(g, &bf16s(seed, n * k, 0.02))?;
        let mut bad = Vec::new();
        for big in [368usize, 1920, 2048] {
            let a = upload(g, &bf16s(seed, big * k, 1.0))?;
            let (ob, ob2) = (g.alloc(big * n * 2)?, g.alloc(big * n * 2)?);
            fixed(a, w, ob, [big as u32, n as u32, k as u32])?;
            fixed(a, w, ob2, [big as u32, n as u32, k as u32])?;
            g.synchronize(s)?;
            let vb = read(g, ob, big * n * 2)?;
            let d = diff(&vb, &read(g, ob2, big * n * 2)?);
            if d > 0 {
                bad.push(format!("run-to-run at {big}: {d} B"));
            }
            for &m in [1usize, 7, 22, 46, 64, 100, 200, 513]
                .iter()
                .filter(|&&m| m < big)
            {
                for off in [0usize, 46.min(big - m)] {
                    let os = g.alloc(m * n * 2)?;
                    fixed(a.offset(off * k * 2), w, os, [m as u32, n as u32, k as u32])?;
                    g.synchronize(s)?;
                    let d = diff(
                        &vb[off * n * 2..(off + m) * n * 2],
                        &read(g, os, m * n * 2)?,
                    );
                    if d > 0 {
                        bad.push(format!("solo {m}@{off} in {big}: {d} B"));
                    }
                    g.free(os)?;
                }
            }
            g.free(a)?;
            g.free(ob)?;
            g.free(ob2)?;
        }
        let mut times = Vec::new();
        // PROBE_MS: the timed row counts (default 64,512,2048,8192).
        let ms: Vec<u32> = std::env::var("PROBE_MS")
            .unwrap_or_else(|_| "64,512,2048,8192".into())
            .split(',')
            .filter_map(|v| v.parse().ok())
            .collect();
        for m in ms {
            let a = upload(g, &bf16s(seed, m as usize * k, 1.0))?;
            let o = g.alloc(m as usize * n * 2)?;
            let mnk = [m, n as u32, k as u32];
            let us = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                f()?;
                g.synchronize(s)?;
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    f()?;
                }
                g.synchronize(s)?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
            };
            let th = us(&mut || {
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    a.0, w.0, o.0, m, n as u32, k as u32, s,
                )
            })?;
            let tf = us(&mut || fixed(a, w, o, mnk))?;
            times.push(format!("m={m}: heuristic {th:.0} rowinv {tf:.0} us"));
            g.free(a)?;
            g.free(o)?;
        }
        g.free(w)?;
        let ref_m = ref_env.unwrap_or(ref_of(n, k));
        let algo = if n >= 2048 && ref_env.is_none() {
            Some("k-chain".to_string())
        } else {
            spark_runtime::cublaslt::bf16_fixed_algo_description(n as u32, k as u32, ref_m)
        };
        println!(
            "fixed {name:<12} N={n:<5} K={k:<5} [{}]: {} | {}",
            algo.unwrap_or_default(),
            if bad.is_empty() {
                "row-invariant, deterministic".to_string()
            } else {
                bad.join("; ")
            },
            times.join("; ")
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let mut seed = 0x5eed_u64;
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if which == "all" || which == "fixed" {
        check_fixed(g, &mut seed)?;
    }
    if which == "all" || which == "hc" {
        check_hc(g, &mut seed)?;
    }
    Ok(())
}
