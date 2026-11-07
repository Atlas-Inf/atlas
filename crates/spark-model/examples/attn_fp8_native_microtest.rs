// SPDX-License-Identifier: AGPL-3.0-only
//! GPU gate for `ATLAS_DENSE_NATIVE_FP8_ATTN` at the Qwen3.8-27B attention
//! shapes (q+gate [12288, 5120], k/v [1024, 5120], o [5120, 6144]), FP8 E4M3
//! weights with one per-tensor scale broadcast to the 128x128 grid, as the
//! loader builds them from nvidia/Qwen3.8-27B-NVFP4:
//!
//! 1. ACCURACY: `w8a16_gemv` (serial decode) against an FP64 CPU reference
//!    of the exact dequantized weights, and against BF16-rounded dequantized
//!    weights (the precision the FP8 -> BF16 -> NVFP4 path starts from).
//! 2. TWIN: `ops::fp8_rows_plan` launches (verify and batched decode) at
//!    1..=4, 8, 16, 24 and 36 rows against that many serial `w8a16_gemv`
//!    calls: zero differing bytes (1..=4 rows is the exact K=2/3/4 verify).
//! 3. PREFILL: the BF16-MMA FP8 kernel (`ATLAS_DENSE_PREFILL_BF16MMA`) at 64
//!    rows against the serial rows (cosine), skipped when not in the build.
//!
//! GPU REQUIRED. Build with the qwen3.8-27b (or qwen3.6-27b) kernels, then:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example attn_fp8_native_microtest
//! Exit 0 = pass, 1 = FAIL.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const SHAPES: [(&str, usize, usize); 3] = [
    ("q+gate", 12288, 5120),
    ("k/v", 1024, 5120),
    ("o", 5120, 6144),
];
/// The per-tensor scale (nvidia's attention scales are O(1e-3..1e-2)).
const SCALE: f32 = 0.0068;
const TWIN_ROWS: [usize; 8] = [1, 2, 3, 4, 8, 16, 24, 36];
const MAX_ROWS: usize = 64;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

/// E4M3 (bias 7) as the kernels' LUT: 0x7f / 0xff (NaN) read as 0.
fn e4m3(b: u8) -> f64 {
    let (s, e, m) = (b >> 7, (b >> 3) & 15, b & 7);
    if e == 15 && m == 7 {
        return 0.0;
    }
    let v = if e == 0 {
        m as f64 / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn f(bits: u16) -> f64 {
    bf16::from_bits(bits).to_f64()
}

fn cos(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb + 1e-300)
}

struct Kernels {
    gemv: KernelHandle,
    rows: ops::Fp8RowsKernels,
}

/// One projection's device state.
struct Proj {
    name: &'static str,
    w: Fp8Weight,
    bytes: Vec<u8>,
    x: Vec<u16>,
    x_dev: DevicePtr,
}

impl Proj {
    fn new(
        g: &dyn GpuBackend,
        rng: &mut Lcg,
        (name, n, k): (&'static str, usize, usize),
    ) -> Result<Self> {
        let bytes: Vec<u8> = (0..n * k).map(|_| rng.next() as u8).collect();
        let grid = vec![SCALE; n.div_ceil(128) * k.div_ceil(128)];
        let grid_b: Vec<u8> = grid.iter().flat_map(|s| s.to_le_bytes()).collect();
        let x: Vec<u16> = (0..MAX_ROWS * k)
            .map(|_| bf16::from_f64(rng.f() * 2.0 - 1.0).to_bits())
            .collect();
        let x_b: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(Self {
            name,
            w: Fp8Weight {
                weight: up(g, &bytes)?,
                row_scale: up(g, &grid_b)?,
                n: n as u32,
                k: k as u32,
                scale_format: WeightQuantFormat::Fp8BlockScaled,
            },
            bytes,
            x,
            x_dev: up(g, &x_b)?,
        })
    }

    fn nk(&self) -> (usize, usize) {
        (self.w.n as usize, self.w.k as usize)
    }

    /// `rows` serial `w8a16_gemv` rows into `out`.
    fn serial(&self, g: &dyn GpuBackend, ks: &Kernels, out: DevicePtr, rows: usize) -> Result<()> {
        let (n, k) = self.nk();
        for r in 0..rows {
            ops::w8a16_gemv(
                g,
                ks.gemv,
                self.x_dev.offset(r * k * 2),
                self.w.weight,
                self.w.row_scale,
                out.offset(r * n * 2),
                n as u32,
                k as u32,
                0,
            )?;
        }
        Ok(())
    }
}

/// Leg 1 for row 0: the GEMV against FP64 references.
fn accuracy(g: &dyn GpuBackend, ks: &Kernels, p: &Proj, out: DevicePtr) -> Result<bool> {
    let (n, k) = p.nk();
    p.serial(g, ks, out, 1)?;
    g.synchronize(0)?;
    let gpu: Vec<f64> = down_bf16(g, out, n)?.into_iter().map(f).collect();
    let x: Vec<f64> = p.x[..k].iter().map(|&b| f(b)).collect();
    let (mut exact, mut bf16w) = (vec![0f64; n], vec![0f64; n]);
    for (row, (e, b)) in exact.iter_mut().zip(bf16w.iter_mut()).enumerate() {
        for (j, xv) in x.iter().enumerate() {
            let w = e4m3(p.bytes[row * k + j]) * SCALE as f64;
            *e += xv * w;
            *b += xv * bf16::from_f64(w).to_f64();
        }
    }
    let peak = exact.iter().fold(0f64, |m, v| m.max(v.abs()));
    let max_err = gpu
        .iter()
        .zip(&exact)
        .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
    let (c_exact, c_bf16) = (cos(&gpu, &exact), cos(&gpu, &bf16w));
    let pass = c_exact >= 0.99999 && max_err <= 0.01 * peak;
    println!(
        "ACCURACY {:7} [{n}, {k}]: cos(exact)={c_exact:.9} max_err/peak={:.2e} \
         cos(bf16 weights)={c_bf16:.9} {}",
        p.name,
        max_err / peak,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Leg 2: plan launches against serial rows, byte for byte.
fn twin(g: &dyn GpuBackend, ks: &Kernels, p: &Proj, [a, b]: [DevicePtr; 2]) -> Result<bool> {
    let (n, k) = p.nk();
    let mut pass = true;
    for rows in TWIN_ROWS {
        let Some(plan) = ops::fp8_rows_plan(rows, true, ops::gemv_vl2_enabled(), ks.rows) else {
            println!(
                "TWIN {:7} rows={rows:2}: no plan (kernel missing) FAIL",
                p.name
            );
            pass = false;
            continue;
        };
        for l in &plan {
            ops::w8a16_gemv_batch4_div(
                g,
                l.kernel,
                p.x_dev.offset(l.first * k * 2),
                p.w.weight,
                p.w.row_scale,
                a.offset(l.first * n * 2),
                l.rows as u32,
                n as u32,
                k as u32,
                l.div,
                0,
            )?;
        }
        p.serial(g, ks, b, rows)?;
        g.synchronize(0)?;
        let (va, vb) = (down_bf16(g, a, rows * n)?, down_bf16(g, b, rows * n)?);
        let diff = va.iter().zip(&vb).filter(|(x, y)| x != y).count();
        pass &= diff == 0;
        println!(
            "TWIN {:7} rows={rows:2} launches={}: differing={diff} {}",
            p.name,
            plan.len(),
            if diff == 0 { "PASS" } else { "FAIL" }
        );
    }
    Ok(pass)
}

/// Leg 3: the BF16-MMA FP8 prefill kernel at MAX_ROWS rows.
fn prefill(g: &dyn GpuBackend, ks: &Kernels, p: &Proj, [a, b]: [DevicePtr; 2]) -> Result<bool> {
    // SAFETY: single-threaded example, before any reader of the variable.
    unsafe { std::env::set_var("ATLAS_DENSE_PREFILL_BF16MMA", "1") };
    let mma = ops::DenseBf16mma::load(g);
    let levers = ops::ModelLevers::from_env();
    let uniform = ops::fp8_scale_k_uniform(g, &p.w)?;
    if !mma.serves_fp8(&p.w, uniform, MAX_ROWS as u32, &levers) {
        println!(
            "PREFILL {:7}: BF16-MMA FP8 kernel not served (not in this build?) SKIP",
            p.name
        );
        return Ok(true);
    }
    let lowp = ops::LowpArgs {
        group: 0,
        codes: DevicePtr::NULL,
        scales: DevicePtr::NULL,
    };
    let (n, _) = p.nk();
    mma.fp8(g, p.x_dev, &p.w, None, a, MAX_ROWS as u32, lowp, 0)?;
    p.serial(g, ks, b, MAX_ROWS)?;
    g.synchronize(0)?;
    let va: Vec<f64> = down_bf16(g, a, MAX_ROWS * n)?.into_iter().map(f).collect();
    let vb: Vec<f64> = down_bf16(g, b, MAX_ROWS * n)?.into_iter().map(f).collect();
    let worst = (0..MAX_ROWS)
        .map(|r| cos(&va[r * n..(r + 1) * n], &vb[r * n..(r + 1) * n]))
        .fold(1f64, f64::min);
    let pass = worst >= 0.99999;
    println!(
        "PREFILL {:7} rows={MAX_ROWS}: worst row cos vs serial GEMV={worst:.9} {}",
        p.name,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b ptx set");
    eprintln!("kernel set: {}", set.target.model);
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let ks = Kernels {
        gemv: g.kernel("w8a16_gemv", "w8a16_gemv")?,
        rows: ops::Fp8RowsKernels {
            batch4: g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4")?,
            batch8: KernelHandle(0),
            batch8_dyn_vl2: KernelHandle(0),
            batch8_dyn_vl2_n2: KernelHandle(0),
            batch16: g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?,
        },
    };
    let mut rng = Lcg(0x4154_544e_4650_3801);
    let mut pass = true;
    for shape in SHAPES {
        let p = Proj::new(g, &mut rng, shape)?;
        ensure!(p.w.n % 128 == 0 && p.w.k % 128 == 0);
        let (n, _) = p.nk();
        let outs = [g.alloc(MAX_ROWS * n * 2)?, g.alloc(MAX_ROWS * n * 2)?];
        pass &= accuracy(g, &ks, &p, outs[0])?;
        pass &= twin(g, &ks, &p, outs)?;
        pass &= prefill(g, &ks, &p, outs)?;
        for ptr in outs.into_iter().chain([p.w.weight, p.w.row_scale, p.x_dev]) {
            g.free(ptr)?;
        }
    }
    if pass {
        println!("ALL PASS");
        Ok(())
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
