// SPDX-License-Identifier: AGPL-3.0-only
//! Vendored Marlin NVFP4 W4A16 GEMM vs our tensor-core W4A16 GEMV tiers
//! (`w4a16_gemv_tc8/16/32`) at GLM-5.3 Flash per-rank (TP2) dense decode
//! shapes and M = 1, 4, 8, 16, 24, 32.
//!
//! Both sides consume the same synthetic modelopt NVFP4 weight ([N, K/2] E2M1
//! + [N, K/16] E4M3 + f32 scale_2); Marlin's copy goes through the REAL
//! load-time packing (`ops::marlin_pack_nvfp4`, the Nemotron sidecar path).
//! Marlin output is checked against the tc tier (worst error in BF16 ulps of
//! the larger magnitude) before and after its timed loop, so the self-reset
//! of the lock workspace is exercised too.
//!
//! Every Marlin instantiation that fits the shape is run in both reduce modes:
//! `fp32` (global reduce through C_tmp + locks; vLLM's default) and `atomic`
//! (BF16 atomicAdd into C; the Nemotron sidecar setting). The m8 kernels run
//! for M <= 8 only: their `parallel` split strides A/C by 16 rows, so M = 16,
//! 24, 32 leave rows unwritten (measured; vLLM never dispatches them there).
//! The cfg4 kernels (M-tile 32, tm=2, 128 threads) run for M > 8.
//!
//! Error is reported two ways: per element, in BF16 ulps (1/128) of the
//! larger magnitude (`ok` <= MAX_ULPS), and as the worst absolute difference in
//! BF16 ulps of the output RMS. A config within MAX_ULPS only on the RMS scale
//! is `lossy` (the atomic reduce rounds each K-split partial to BF16, so
//! outputs near cancellation lose relative precision); beyond both it is `BAD`.
//!
//! Timings are cold-weight: rotating weight copies (> 96 MiB total) so each
//! launch streams from DRAM; median of TRIALS batched wall-clock averages.
//! GB/s = NVFP4 weight + block-scale bytes / time.
//!
//! Exit: 0 ok, 1 some Marlin config BAD vs the tc tier, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example marlin_vs_tc_bench

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP: usize = 16;
const SCALE2: f32 = 0.0123_f32;
const SMEM: u32 = 96 * 1024;
const TRIALS: usize = 7;
const MAX_ULPS: f32 = 4.0;
const MS: [u32; 6] = [1, 4, 8, 16, 24, 32];

/// GLM-5.3 Flash per-rank (TP2) dense decode projections (N x K).
const SHAPES: [(&str, usize, usize); 6] = [
    ("kda q/k/v/o", 4096, 4096),
    ("shared gate/up", 2048, 4096),
    ("shared down", 4096, 2048),
    ("dense ffn gate/up", 12288, 4096),
    ("dense ffn down", 4096, 12288),
    ("mla q_b", 8192, 1536),
];

struct MarlinKernel {
    name: &'static str,
    kh: KernelHandle,
    /// 8 = m8 (M <= 8), 32 = cfg4 (tm=2, 8 < M <= 32).
    m_tile: u32,
    n_align: usize,
    k_align: usize,
}

impl MarlinKernel {
    fn serves(&self, m: u32, n: usize, k: usize) -> bool {
        let m_ok = if self.m_tile == 8 {
            m <= 8
        } else {
            m > 8 && m <= 32
        };
        m_ok && n % self.n_align == 0 && k % self.k_align == 0
    }
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect())
}

/// (per-element, RMS-scaled) worst |x - y| in BF16 ulps (1/128): per element
/// of the larger magnitude (small absolute floor for ~0 outputs), and of the
/// reference RMS. NaN / unwritten outputs count as infinite.
fn worst_ulps(x: &[f32], y: &[f32]) -> (f32, f32) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let nan_inf = |d: f32| if d.is_nan() { f32::INFINITY } else { d };
    x.iter().zip(y).fold((0f32, 0f32), |(e, r), (a, b)| {
        let d = (a - b).abs();
        (
            e.max(nan_inf(d / (a.abs().max(b.abs()) / 128.0 + 5e-4))),
            r.max(nan_inf(d / (rms / 128.0))),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn tc_launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n as u32, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(ws)
        .arg_f32(SCALE2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

/// Same argument list as `ops::marlin_nvfp4_m8`, with the reduce mode exposed.
#[allow(clippy::too_many_arguments)]
fn marlin_launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    sms: u32,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    c_tmp: DevicePtr,
    s: DevicePtr,
    gs: DevicePtr,
    locks: DevicePtr,
    m: u32,
    n: usize,
    k: usize,
    atomic: bool,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([sms, 1, 1])
        .block([128, 1, 1])
        .shared_mem(SMEM)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL) // bias
        .arg_ptr(DevicePtr::NULL) // a_scales
        .arg_ptr(s)
        .arg_ptr(gs)
        .arg_ptr(DevicePtr::NULL) // zp
        .arg_ptr(DevicePtr::NULL) // g_idx
        .arg_i32((k / GROUP) as i32)
        .arg_i32(m as i32)
        .arg_i32(n as i32)
        .arg_i32(k as i32)
        .arg_i32(k as i32) // lda
        .arg_ptr(locks)
        .arg_i32(0) // has_bias
        .arg_i32(atomic as i32)
        .arg_i32(1) // use_fp32_reduce
        .arg_i32(SMEM as i32)
        .launch(0)
}

/// Median over TRIALS of the per-launch wall time of back-to-back launches
/// rotating through `copies` weight copies.
fn median_time(g: &dyn GpuBackend, copies: usize, f: &dyn Fn(usize) -> Result<()>) -> Result<f64> {
    let reps = (2 * copies).max(16);
    f(0)?;
    let mut v = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        g.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..reps {
            f(i % copies)?;
        }
        g.synchronize(0)?;
        v.push(t0.elapsed().as_secs_f64() / reps as f64);
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(v[TRIALS / 2])
}

struct Row {
    shape: String,
    m: u32,
    tier: &'static str,
    t_tc: f64,
    best: String,
    t_best: f64,
    err_best: f32,
    /// Worst (per-element, RMS-scaled) error over every config run.
    err_all: (f32, f32),
    gbytes: f64,
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let sms = g.sm_count()?;
    let tier = |name: &str| g.kernel("w4a16_gemv", name);
    let marlin = |name: &str| g.kernel("marlin_nvfp4_gemm", name);
    let (Ok(tc8), Ok(tc16), Ok(tc32), Ok(repack)) = (
        tier("w4a16_gemv_tc8"),
        tier("w4a16_gemv_tc16"),
        tier("w4a16_gemv_tc32"),
        g.kernel("marlin_repack", "atlas_marlin_repack_w4"),
    ) else {
        eprintln!("w4a16_gemv_tc* / atlas_marlin_repack_w4 absent from this target");
        std::process::exit(2);
    };
    let mut kernels = Vec::new();
    for (name, m_tile, n_align, k_align) in [
        ("m8", 8, 64, 128),
        ("m8_k64n128", 8, 128, 64),
        ("cfg4", 32, 64, 128),
        ("cfg4_k64n128", 32, 128, 64),
    ] {
        match marlin(&format!("atlas_marlin_nvfp4_{name}")) {
            Ok(kh) => kernels.push(MarlinKernel {
                name,
                kh,
                m_tile,
                n_align,
                k_align,
            }),
            Err(e) => eprintln!("atlas_marlin_nvfp4_{name} absent: {e}"),
        }
    }
    if kernels.is_empty() {
        std::process::exit(2);
    }
    println!(
        "SMs {sms}, Marlin smem {} KiB, {TRIALS} trials, cold weights > 96 MiB",
        SMEM / 1024
    );

    // Workspace sized like vLLM (c_tmp: sms * 64 rows * 256 cols fp32) with 2x headroom.
    let c_tmp = g.alloc(2 * sms as usize * 64 * 256 * 4)?;
    let locks = g.alloc(1024 * 4)?;
    let mut rows = Vec::new();
    let mut fail = false;
    for (name, n, k) in SHAPES {
        let mut rng = Lcg(0x5EED ^ (n * 31 + k) as u64);
        let a: Vec<u8> = (0..32 * k)
            .flat_map(|_| bf16::from_f32(rng.f() * 3.0 - 1.5).to_bits().to_le_bytes())
            .collect();
        let w: Vec<u8> = (0..n * k / 2).map(|_| (rng.f() * 256.0) as u8).collect();
        let ws: Vec<u8> = (0..n * k / GROUP)
            .map(|_| 0x30u8 + (rng.f() * 24.0) as u8)
            .collect();
        let gbytes = (w.len() + ws.len()) as f64;
        let copies = (96usize << 20).div_ceil(w.len() + ws.len()).max(2);
        let ad = up(g, &a)?;
        let rot: Vec<(DevicePtr, DevicePtr)> = (0..copies)
            .map(|_| Ok((up(g, &w)?, up(g, &ws)?)))
            .collect::<Result<_>>()?;

        // Marlin packing (repack + scale permutation + global scale), then
        // rotated copies of the packed buffers.
        let (tmp_in, tmp_out, gs) = (g.alloc(w.len())?, g.alloc(w.len())?, g.alloc(4)?);
        let (mw0, ms0) = (g.alloc(w.len())?, g.alloc(ws.len())?);
        ops::marlin_pack_nvfp4(
            g, repack, &w, &ws, SCALE2, n, k, tmp_in, tmp_out, mw0, ms0, gs, sms, SMEM,
        )?;
        let mrot: Vec<(DevicePtr, DevicePtr)> = (0..copies)
            .map(|_| {
                let (mw, ms) = (g.alloc(w.len())?, g.alloc(ws.len())?);
                g.copy_d2d(mw0, mw, w.len())?;
                g.copy_d2d(ms0, ms, ws.len())?;
                Ok((mw, ms))
            })
            .collect::<Result<_>>()?;
        let (c_tc, c_mar) = (g.alloc(32 * n * 2)?, g.alloc(32 * n * 2)?);

        for m in MS {
            let (tier_name, tck) = match m {
                1..=8 => ("tc8", tc8),
                9..=16 => ("tc16", tc16),
                _ => ("tc32", tc32),
            };
            let cnt = m as usize * n;
            tc_launch(g, tck, ad, rot[0].0, rot[0].1, c_tc, m, n, k)?;
            g.synchronize(0)?;
            let reference = down(g, c_tc, cnt)?;
            let t_tc = median_time(g, copies, &|i| {
                tc_launch(g, tck, ad, rot[i].0, rot[i].1, c_tc, m, n, k)
            })?;
            println!(
                "{name} [{n}x{k}] M={m:2}: {tier_name:4} {:7.1}us {:5.0}GB/s",
                t_tc * 1e6,
                gbytes / t_tc / 1e9
            );
            let mut best: Option<(String, f64, f32)> = None;
            let mut err_all = (0f32, 0f32);
            for mk in kernels.iter().filter(|mk| mk.serves(m, n, k)) {
                for atomic in [false, true] {
                    let run = |i: usize| {
                        let (mw, ms) = mrot[i];
                        marlin_launch(
                            g, mk.kh, sms, ad, mw, c_mar, c_tmp, ms, gs, locks, m, n, k, atomic,
                        )
                    };
                    // Poison C (BF16 NaN) so unwritten rows fail the check.
                    g.memset_async(c_mar, 0xFF, cnt * 2, 0)?;
                    g.memset_async(locks, 0, 1024 * 4, 0)?;
                    run(0)?;
                    g.synchronize(0)?;
                    let pre = worst_ulps(&reference, &down(g, c_mar, cnt)?);
                    let t = median_time(g, copies, &run)?;
                    let post = worst_ulps(&reference, &down(g, c_mar, cnt)?);
                    let err = (pre.0.max(post.0), pre.1.max(post.1));
                    let ok = err.0 <= MAX_ULPS;
                    let status = match (ok, err.1 <= MAX_ULPS) {
                        (true, _) => "ok",
                        (false, true) => "lossy",
                        _ => "BAD",
                    };
                    fail |= status == "BAD";
                    err_all = (err_all.0.max(err.0), err_all.1.max(err.1));
                    let cfg = format!("{}/{}", mk.name, if atomic { "atomic" } else { "fp32" });
                    println!(
                        "    marlin {cfg:20} {:7.1}us {:5.0}GB/s  worst elem {:.2}/{:.2} rms {:.2}/{:.2} ulps (pre/post) {status}  ({:+.1}% vs tc)",
                        t * 1e6,
                        gbytes / t / 1e9,
                        pre.0,
                        post.0,
                        pre.1,
                        post.1,
                        (t_tc / t - 1.0) * 100.0,
                    );
                    if ok && best.as_ref().is_none_or(|b| t < b.1) {
                        best = Some((cfg, t, err.0));
                    }
                }
            }
            let (best, t_best, err_best) = best.unwrap_or(("none".into(), f64::NAN, f32::NAN));
            rows.push(Row {
                shape: format!("{name} {n}x{k}"),
                m,
                tier: tier_name,
                t_tc,
                best,
                t_best,
                err_best,
                err_all,
                gbytes,
            });
        }
        for p in [ad, tmp_in, tmp_out, gs, mw0, ms0, c_tc, c_mar] {
            g.free(p)?;
        }
        for (x, y) in rot.into_iter().chain(mrot) {
            g.free(x)?;
            g.free(y)?;
        }
    }

    println!(
        "\n| shape | M | tier | tier us | tier GB/s | best Marlin | Marlin us | Marlin GB/s | Marlin/tier speedup | best err (elem ulps) | worst any cfg (elem / rms ulps) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {:.1} | {:.0} | {} | {:.1} | {:.0} | {:.2}x | {:.2} | {:.1} / {:.2} |",
            r.shape,
            r.m,
            r.tier,
            r.t_tc * 1e6,
            r.gbytes / r.t_tc / 1e9,
            r.best,
            r.t_best * 1e6,
            r.gbytes / r.t_best / 1e9,
            r.t_tc / r.t_best,
            r.err_best,
            r.err_all.0,
            r.err_all.1,
        );
    }
    std::process::exit(if fail { 1 } else { 0 });
}
