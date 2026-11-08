// SPDX-License-Identifier: AGPL-3.0-only
//! Wide vs narrow row-exact GEMV timing on the Qwen3.8-27B shapes
//! (`ATLAS_WIDE_GEMV`, `kernels/gb10/qwen3.6-27b/nvfp4/wide_gemv.cu`).
//!
//! At M ∈ {1, 4, 8, 9, 12, 16} rows, per shape, it times what the exact
//! verify launches today ("narrow") against each wide variant (nb2, nb4),
//! and prints us and weight GB/s:
//!
//! | family | narrow (today) | wide |
//! |---|---|---|
//! | NVFP4 | `w4a16_gemv` at 1, `batch4` to 4, `batch5..8`, `batch16` | `w4a16_gemv_wide{8,9,12,16}_nb{2,4}` |
//! | Q+gate | `w4a16_gemv_qg_batch4` chunks + `qg_batch3/2` / `qg` | `w4a16_gemv_qg_wide*` |
//! | FP8 | `w8a16_gemv` at 1, `w8a16_gemv_batch4` to 4, `batch16` | `w8a16_gemv_wide*` |
//!
//! COLD WEIGHTS: launches cycle over enough weight copies (512 MB) that each
//! streams its weight from DRAM, as a verify step does. Each (M, variant)
//! also byte-compares one wide launch with the narrow launches (the full
//! gate is `wide_gemv_bitparity_microtest`).
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example wide_gemv_bench
//!
//! Env: ATLAS_PEAK_GBPS (default 273) for the %-of-peak column;
//! WIDE_BENCH_ITERS (default 30).

use anyhow::Result;
use spark_model::layers::ops::wide_gemv::{self, Family, NBS, TIERS, WideKernel};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 5;
const COLD_CYCLE_BYTES: usize = 512 << 20;
const MS: &[u32] = &[1, 4, 8, 9, 12, 16];
const MAX_ROWS: usize = 16;
const SCALE2: f32 = 0.0123;

/// `(label, family, N, K, heads, head_dim)`.
const SHAPES: &[(&str, Family, u32, u32, u32, u32)] = &[
    (
        "nvfp4 gate/up  17408x5120 ",
        Family::W4a16,
        17408,
        5120,
        0,
        0,
    ),
    (
        "nvfp4 down     5120x17408 ",
        Family::W4a16,
        5120,
        17408,
        0,
        0,
    ),
    (
        "nvfp4 o_proj   5120x6144  ",
        Family::W4a16,
        5120,
        6144,
        0,
        0,
    ),
    (
        "nvfp4 k/v      1024x5120  ",
        Family::W4a16,
        1024,
        5120,
        0,
        0,
    ),
    (
        "nvfp4 lm_head  248320x5120",
        Family::W4a16,
        248_320,
        5120,
        0,
        0,
    ),
    (
        "q+gate         12288x5120 ",
        Family::Qg,
        12288,
        5120,
        24,
        256,
    ),
    (
        "fp8 gdn qkvz   16384x5120 ",
        Family::W8a16,
        16384,
        5120,
        0,
        0,
    ),
    (
        "fp8 gdn out    5120x6144  ",
        Family::W8a16,
        5120,
        6144,
        0,
        0,
    ),
];

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The kernels the exact verify runs today.
struct Narrow {
    w4_1: KernelHandle,
    /// `w4a16_gemv_batch{4,5,6,7,8,16}`.
    w4: [(u32, KernelHandle); 6],
    /// `w4a16_gemv_qg`, `_qg_batch2/3/4` (fixed rows).
    qg: [KernelHandle; 4],
    w8_1: KernelHandle,
    w8_4: KernelHandle,
    w8_16: KernelHandle,
}

impl Narrow {
    fn resolve(g: &dyn GpuBackend) -> Result<Self> {
        let k = |m: &str, f: &str| g.kernel(m, f);
        let w4 = |w: u32| -> Result<(u32, KernelHandle)> {
            Ok((w, k("w4a16_gemv", &format!("w4a16_gemv_batch{w}"))?))
        };
        Ok(Self {
            w4_1: k("w4a16_gemv", "w4a16_gemv")?,
            w4: [w4(4)?, w4(5)?, w4(6)?, w4(7)?, w4(8)?, w4(16)?],
            qg: [
                k("w4a16_gemv", "w4a16_gemv_qg")?,
                k("w4a16_gemv", "w4a16_gemv_qg_batch2")?,
                k("w4a16_gemv", "w4a16_gemv_qg_batch3")?,
                k("w4a16_gemv", "w4a16_gemv_qg_batch4")?,
            ],
            w8_1: k("w8a16_gemv", "w8a16_gemv")?,
            w8_4: k("w8a16_gemv_batch4", "w8a16_gemv_batch4")?,
            w8_16: k("w8a16_gemv_batch4", "w8a16_gemv_batch16")?,
        })
    }

    /// The narrow launches for `m` rows: `(kernel, rows, takes an M arg)`,
    /// row-consecutive.
    fn plan(&self, family: Family, m: u32) -> Vec<(KernelHandle, u32, bool)> {
        match family {
            Family::W4a16 if m == 1 => vec![(self.w4_1, 1, false)],
            Family::W4a16 => {
                let (_, h) = *self.w4.iter().find(|(w, _)| *w >= m).expect("m <= 16");
                vec![(h, m, true)]
            }
            Family::Qg => {
                let mut out = Vec::new();
                let mut left = m;
                while left > 0 {
                    let r = left.min(4);
                    out.push((self.qg[r as usize - 1], r, false));
                    left -= r;
                }
                out
            }
            Family::W8a16 if m == 1 => vec![(self.w8_1, 1, false)],
            Family::W8a16 if m <= 4 => vec![(self.w8_4, m, true)],
            Family::W8a16 => vec![(self.w8_16, m, true)],
        }
    }
}

/// One shape's operands: activations, `copies` weights (packed / FP8 bytes
/// and scales), two outputs.
struct Bufs {
    a: DevicePtr,
    weights: Vec<(DevicePtr, DevicePtr)>,
    c: [DevicePtr; 2],
}

#[allow(clippy::too_many_arguments)]
fn launch_narrow(
    g: &dyn GpuBackend,
    plan: &[(KernelHandle, u32, bool)],
    (family, n, k, heads, hd): (Family, u32, u32, u32, u32),
    a: DevicePtr,
    (w, s): (DevicePtr, DevicePtr),
    c: DevicePtr,
) -> Result<()> {
    let mut r0 = 0usize;
    for &(h, rows, takes_m) in plan {
        let (at, ct) = (a.offset(r0 * k as usize * 2), c.offset(r0 * n as usize * 2));
        let mut l = KernelLaunch::new(g, h)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(at)
            .arg_ptr(w)
            .arg_ptr(s);
        if family != Family::W8a16 {
            l = l.arg_f32(SCALE2);
        }
        l = l.arg_ptr(ct);
        if takes_m {
            l = l.arg_u32(rows);
        }
        l = l.arg_u32(n).arg_u32(k);
        if family == Family::Qg {
            l = l.arg_u32(heads).arg_u32(hd);
        }
        l.launch(0)?;
        r0 += rows as usize;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_wide(
    g: &dyn GpuBackend,
    kernel: &WideKernel,
    sms: u32,
    (family, n, k, heads, hd): (Family, u32, u32, u32, u32),
    m: u32,
    a: DevicePtr,
    (w, s): (DevicePtr, DevicePtr),
    c: DevicePtr,
) -> Result<()> {
    let q = QuantizedWeight {
        weight: w,
        weight_scale: s,
        weight_scale_2: SCALE2,
        ..QuantizedWeight::null()
    };
    match family {
        Family::W4a16 => wide_gemv::launch_w4a16(g, kernel, sms, a, &q, c, (m, n, k), n, 0),
        Family::Qg => wide_gemv::launch_qg(g, kernel, sms, a, &q, c, (m, n, k), (heads, hd), 0),
        Family::W8a16 => wide_gemv::launch_w8a16(g, kernel, sms, a, (w, s), c, (m, n, k), n, 0),
    }
}

/// Mean us a call of `run(i)` over `iters` calls (weight copy `i`).
fn time(g: &dyn GpuBackend, iters: usize, run: &dyn Fn(usize) -> Result<()>) -> Result<f64> {
    for i in 0..WARMUP {
        run(i)?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for i in 0..iters {
        run(WARMUP + i)?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

fn alloc_shape(
    g: &dyn GpuBackend,
    rng: &mut XorShift,
    family: Family,
    n: u32,
    k: u32,
) -> Result<(Bufs, usize)> {
    let (n, k) = (n as usize, k as usize);
    let (w_bytes, s_bytes) = match family {
        Family::W8a16 => (n * k, n.div_ceil(128) * k.div_ceil(128) * 4),
        _ => (n * k / 2, n * k / 16),
    };
    let weight_bytes = w_bytes + s_bytes;
    let copies = COLD_CYCLE_BYTES.div_ceil(weight_bytes).clamp(1, 16);
    // BF16 activations in +-[0.5, 2): random sign, exponent 0x7E/0x7F, random mantissa.
    let a_host: Vec<u8> = (0..MAX_ROWS * k)
        .flat_map(|_| {
            let r = rng.next();
            ((((r >> 63) as u16) << 15) | (0x3F00 | ((r >> 8) as u16 & 0xFF))).to_le_bytes()
        })
        .collect();
    let w_host: Vec<u8> = (0..w_bytes).map(|_| (rng.next() >> 32) as u8).collect();
    let s_host: Vec<u8> = match family {
        // FP32 block scales ~0.05.
        Family::W8a16 => (0..s_bytes / 4)
            .flat_map(|_| 0.05f32.to_le_bytes())
            .collect(),
        // E4M3 0x30..0x3F: 0.5 .. 0.94.
        _ => (0..s_bytes)
            .map(|_| 0x30 + (rng.next() >> 60) as u8)
            .collect(),
    };
    let up = |bytes: &[u8]| -> Result<DevicePtr> {
        let p = g.alloc(bytes.len())?;
        g.copy_h2d(bytes, p)?;
        Ok(p)
    };
    let a = up(&a_host)?;
    let mut weights = Vec::with_capacity(copies);
    for _ in 0..copies {
        weights.push((up(&w_host)?, up(&s_host)?));
    }
    let c = [g.alloc(MAX_ROWS * n * 2)?, g.alloc(MAX_ROWS * n * 2)?];
    Ok((Bufs { a, weights, c }, weight_bytes))
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b ptx set: build with ATLAS_TARGET_MODEL=qwen3.8-27b");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let sms = g.sm_count()?;
    let peak: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);
    let iters: usize = std::env::var("WIDE_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let narrow = Narrow::resolve(g)?;
    let wides: Vec<WideKernel> = Family::ALL
        .iter()
        .flat_map(|&f| NBS.iter().flat_map(move |&nb| TIERS.map(|r| (f, r, nb))))
        .map(|(f, r, nb)| WideKernel::resolve(g, f, r, nb))
        .filter(|w| w.handle.0 != 0)
        .collect();
    eprintln!(
        "kernel set {}, {sms} SMs, peak {peak:.0} GB/s, {iters} iters, cold weights; \
         {} wide tiers resolved",
        set.target.model,
        wides.len()
    );

    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut all_identical = true;
    for &(label, family, n, k, heads, hd) in SHAPES {
        let shape = (family, n, k, heads, hd);
        let (bufs, weight_bytes) = alloc_shape(g, &mut rng, family, n, k)?;
        let copies = bufs.weights.len();
        let floor_us = weight_bytes as f64 / (peak * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.1} MB x {copies}, floor {floor_us:.0} us ──",
            weight_bytes as f64 / 1e6
        );
        let report = |name: &str, m: u32, us: f64, same: Option<bool>| {
            let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
            let same = same.map_or(String::new(), |s| format!("  identical={s}"));
            eprintln!(
                "  {name:<26} M={m:<2} {us:>9.1} us {gbps:>6.1} GB/s {:>5.1}% peak{same}",
                100.0 * gbps / peak
            );
        };
        for &m in MS {
            let plan = narrow.plan(family, m);
            let us = time(g, iters, &|i| {
                launch_narrow(g, &plan, shape, bufs.a, bufs.weights[i % copies], bufs.c[0])
            })?;
            report("narrow", m, us, None);
            launch_narrow(g, &plan, shape, bufs.a, bufs.weights[0], bufs.c[0])?;
            g.synchronize(0)?;
            let mut want = vec![0u8; m as usize * n as usize * 2];
            g.copy_d2h(bufs.c[0], &mut want)?;
            for &nb in &NBS {
                let Some(w) = wides
                    .iter()
                    .filter(|w| w.family == family && w.nb == nb && w.serves(m, k))
                    .min_by_key(|w| w.rows)
                else {
                    continue;
                };
                let us = time(g, iters, &|i| {
                    launch_wide(
                        g,
                        w,
                        sms,
                        shape,
                        m,
                        bufs.a,
                        bufs.weights[i % copies],
                        bufs.c[1],
                    )
                })?;
                g.memset(bufs.c[1], 0, MAX_ROWS * n as usize * 2)?;
                launch_wide(g, w, sms, shape, m, bufs.a, bufs.weights[0], bufs.c[1])?;
                g.synchronize(0)?;
                let mut got = vec![0u8; want.len()];
                g.copy_d2h(bufs.c[1], &mut got)?;
                all_identical &= got == want;
                report(
                    &wide_gemv::kernel_name(family, w.rows, nb),
                    m,
                    us,
                    Some(got == want),
                );
            }
        }
        g.free(bufs.a)?;
        for p in bufs.c {
            g.free(p)?;
        }
        for (w, s) in bufs.weights {
            g.free(w)?;
            g.free(s)?;
        }
    }
    if !all_identical {
        eprintln!("FAIL — a wide launch differed from the narrow launches (run the microtest)");
        std::process::exit(1);
    }
    Ok(())
}
