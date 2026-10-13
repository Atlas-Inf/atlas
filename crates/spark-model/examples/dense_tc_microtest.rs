// SPDX-License-Identifier: AGPL-3.0-only
//! Accuracy + bandwidth gate for `dense_gemv_bf16_tc16/32` against the scalar
//! `dense_gemv_bf16_batchm` (run in 8-row slices as the reference) at GLM
//! target-head, DFlash2 drafter and MLA projection shapes.
//!
//! Exit: 0 pass, 1 any leg beyond 2 BF16 ulps, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example dense_tc_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const SHAPES: [(&str, usize, usize); 5] = [
    ("target head half [77440 x 4096]", 77440, 4096),
    ("drafter q/o      [ 4096 x 4096]", 4096, 4096),
    ("drafter gate/up  [12288 x 4096]", 12288, 4096),
    ("drafter down     [ 4096 x12288]", 4096, 12288),
    ("drafter k/v odd  [ 1027 x 4096]", 1027, 4096),
];

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

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    tc: bool,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let (grid, block) = if tc {
        (div_ceil(n, 16), 256)
    } else {
        (div_ceil(n, 4), 256)
    };
    KernelLaunch::new(g, kh)
        .grid([grid, 1, 1])
        .block([block, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(n)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m_ = "dense_gemv_bf16_batchm";
    let (Ok(bm), Ok(t16), Ok(t32)) = (
        g.kernel(m_, "dense_gemv_bf16_batchm"),
        g.kernel(m_, "dense_gemv_bf16_tc16"),
        g.kernel(m_, "dense_gemv_bf16_tc32"),
    ) else {
        eprintln!("dense_gemv_bf16 batchm/tc kernels absent from this target");
        std::process::exit(2);
    };
    let mut fail = false;
    for (name, n, k) in SHAPES {
        let mut rng = Lcg(0xD15C ^ (n * 131 + k) as u64);
        let a: Vec<u8> = (0..32 * k)
            .flat_map(|_| bf16::from_f32(rng.f() * 2.0 - 1.0).to_bits().to_le_bytes())
            .collect();
        let w: Vec<u8> = (0..n * k)
            .flat_map(|_| {
                bf16::from_f32((rng.f() * 2.0 - 1.0) * 0.05)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let (ad, wd) = (up(g, &a)?, up(g, &w)?);
        let (c_ref, c_tc) = (g.alloc(32 * n * 2)?, g.alloc(32 * n * 2)?);
        let copies = (96usize << 20).div_ceil(w.len()).max(2);
        let rot: Vec<DevicePtr> = (0..copies).map(|_| up(g, &w)).collect::<Result<_>>()?;
        for m in [8u32, 16, 24, 32] {
            for s in (0..m).step_by(8) {
                let rows = (m - s).min(8);
                launch(
                    g,
                    bm,
                    false,
                    ad.offset(s as usize * k * 2),
                    wd,
                    c_ref.offset(s as usize * n * 2),
                    rows,
                    n as u32,
                    k as u32,
                )?;
            }
            let tk = if m <= 16 { t16 } else { t32 };
            launch(g, tk, true, ad, wd, c_tc, m, n as u32, k as u32)?;
            g.synchronize(0)?;
            let (r, t) = (
                down(g, c_ref, m as usize * n)?,
                down(g, c_tc, m as usize * n)?,
            );
            let mut worst = 0f32;
            for (x, y) in r.iter().zip(&t) {
                let tol = x.abs().max(y.abs()) * (2.0 / 128.0) + 1e-3;
                worst = worst.max((x - y).abs() / tol);
            }
            let ok = worst <= 1.0;
            fail |= !ok;
            let time = |f: &dyn Fn(DevicePtr) -> Result<()>| -> Result<f64> {
                let reps = 4 * copies;
                g.synchronize(0)?;
                let t0 = std::time::Instant::now();
                for i in 0..reps {
                    f(rot[i % copies])?;
                }
                g.synchronize(0)?;
                Ok(t0.elapsed().as_secs_f64() / reps as f64)
            };
            let t_ref = time(&|wp| {
                for s in (0..m).step_by(8) {
                    let rows = (m - s).min(8);
                    launch(
                        g,
                        bm,
                        false,
                        ad.offset(s as usize * k * 2),
                        wp,
                        c_ref.offset(s as usize * n * 2),
                        rows,
                        n as u32,
                        k as u32,
                    )?;
                }
                Ok(())
            })?;
            let t_tc = time(&|wp| launch(g, tk, true, ad, wp, c_tc, m, n as u32, k as u32))?;
            let bytes = (n * k * 2) as f64;
            println!(
                "{name} M={m}: worst/tol {worst:.3} {}  batchm x{} {:7.1}us  tc {:7.1}us {:5.0}GB/s",
                if ok { "ok" } else { "FAIL" },
                m.div_ceil(8),
                t_ref * 1e6,
                t_tc * 1e6,
                bytes / t_tc / 1e9,
            );
        }
    }
    std::process::exit(if fail { 1 } else { 0 });
}
