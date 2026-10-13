// SPDX-License-Identifier: AGPL-3.0-only
//! Accuracy + bandwidth gate for the tensor-core `w4a16_gemv_tc8` tier
//! against the scalar `w4a16_gemv_batch8` at GLM-5.3 Flash TP2 projection
//! shapes and every M the tier serves (1..=8).
//!
//! tc8 sums in a different order, so the gate is numeric, not bytewise: the
//! worst |tc8 - batch8| must stay within 2 BF16 ulps of the output magnitude
//! (both round an FP32 accumulator once). Timings are cold-weight (a scratch
//! flush between reps) and report effective weight bandwidth.
//!
//! Exit: 0 pass, 1 any leg out of tolerance, 2 kernels absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_tc8_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const SCALE2: f32 = 0.0123_f32;

/// GLM-5.3 Flash per-rank (TP2) projection shapes, plus an odd-N tail.
const SHAPES: [(&str, usize, usize); 6] = [
    ("kda qkv/o     [4096 x 4096]", 4096, 4096),
    ("shared gate/up[2048 x 4096]", 2048, 4096),
    ("shared down   [4096 x 2048]", 4096, 2048),
    ("dense ffn     [12288 x 4096]", 12288, 4096),
    ("mla q_b       [8192 x 1536]", 8192, 1536),
    ("tail odd-N    [4099 x 4096]", 4099, 4096),
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
    ws: DevicePtr,
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
        .arg_ptr(ws)
        .arg_f32(SCALE2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (Ok(b8), Ok(tc8), Ok(tc16), Ok(tc32)) = (
        g.kernel("w4a16_gemv", "w4a16_gemv_batch8"),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc8"),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc16"),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc32"),
    ) else {
        eprintln!("w4a16_gemv_batch8 / w4a16_gemv_tc8 absent from this target");
        std::process::exit(2);
    };
    let tc_ld = [
        g.kernel("w4a16_gemv", "w4a16_gemv_tc8_ld").ok(),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc16_ld").ok(),
        g.kernel("w4a16_gemv", "w4a16_gemv_tc32_ld").ok(),
    ];
    let mut fail = false;
    for (name, n, k) in SHAPES {
        let mut rng = Lcg(0x5EED ^ (n * 31 + k) as u64);
        let a: Vec<u8> = (0..32 * k)
            .flat_map(|_| bf16::from_f32(rng.f() * 3.0 - 1.5).to_bits().to_le_bytes())
            .collect();
        let w: Vec<u8> = (0..n * k / 2).map(|_| (rng.f() * 256.0) as u8).collect();
        let ws: Vec<u8> = (0..n * k / GROUP_SIZE)
            .map(|_| 0x30u8 + (rng.f() * 24.0) as u8)
            .collect();
        let (ad, wd, wsd) = (up(g, &a)?, up(g, &w)?, up(g, &ws)?);
        // Rotating weight copies (> L2) so back-to-back timed launches stream
        // from DRAM while their launch overhead overlaps the previous kernel.
        let copies = (96usize << 20).div_ceil(w.len() + ws.len()).max(2);
        let rot: Vec<(DevicePtr, DevicePtr)> = (0..copies)
            .map(|_| Ok((up(g, &w)?, up(g, &ws)?)))
            .collect::<Result<_>>()?;
        let (c_ref, c_tc) = (g.alloc(32 * n * 2)?, g.alloc(32 * n * 2)?);
        let weight_bytes = (n * k / 2 + n * k / GROUP_SIZE) as f64;
        // The scalar reference runs in 8-row slices (its row cap).
        let reference = |w: DevicePtr, s: DevicePtr, m: u32| -> Result<()> {
            for r in (0..m).step_by(8) {
                let rows = (m - r).min(8);
                launch(
                    g,
                    b8,
                    false,
                    ad.offset(r as usize * k * 2),
                    w,
                    s,
                    c_ref.offset(r as usize * n * 2),
                    rows,
                    n as u32,
                    k as u32,
                )?;
            }
            Ok(())
        };
        for m in (1..=8u32).chain([12, 16, 24, 32]) {
            let tck = match m {
                1..=8 => tc8,
                9..=16 => tc16,
                _ => tc32,
            };
            reference(wd, wsd, m)?;
            launch(g, tck, true, ad, wd, wsd, c_tc, m, n as u32, k as u32)?;
            g.synchronize(0)?;
            let (r, t) = (
                down(g, c_ref, m as usize * n)?,
                down(g, c_tc, m as usize * n)?,
            );
            let mut worst = 0f32;
            for (x, y) in r.iter().zip(&t) {
                // 2 BF16 ulps of the larger magnitude (+ a floor for ~0 outputs).
                let tol = x.abs().max(y.abs()) * (2.0 / 128.0) + 1e-3;
                worst = worst.max((x - y).abs() / tol);
            }
            let ok = worst <= 1.0;
            fail |= !ok;
            let time = |f: &dyn Fn(DevicePtr, DevicePtr) -> Result<()>| -> Result<f64> {
                let reps = 4 * copies;
                g.synchronize(0)?;
                let t0 = std::time::Instant::now();
                for i in 0..reps {
                    let (w, s) = rot[i % copies];
                    f(w, s)?;
                }
                g.synchronize(0)?;
                Ok(t0.elapsed().as_secs_f64() / reps as f64)
            };
            let t8 = time(&|w, s| reference(w, s, m))?;
            let ttc = time(&|w, s| launch(g, tck, true, ad, w, s, c_tc, m, n as u32, k as u32))?;
            println!(
                "{name} M={m}: worst/tol {worst:.3} {}  batch8 {:6.1}us {:5.0}GB/s  tc {:6.1}us {:5.0}GB/s",
                if ok { "ok" } else { "FAIL" },
                t8 * 1e6,
                weight_bytes / t8 / 1e9,
                ttc * 1e6,
                weight_bytes / ttc / 1e9,
            );
        }
        // Strided tiers: at the natural stride they must equal the plain tier
        // bit for bit; over the two K-halves of the weight (rows k/2 bytes
        // apart, contiguous half-width activations) their sum must match the
        // full product within the same tolerance.
        if let [Some(l8), Some(l16), Some(l32)] = tc_ld {
            let half = k / 2;
            let a_half = |h: usize| -> Vec<u8> {
                a.chunks_exact(k * 2)
                    .flat_map(|row| row[h * half * 2..(h + 1) * half * 2].to_vec())
                    .collect()
            };
            let (a0, a1) = (up(g, &a_half(0))?, up(g, &a_half(1))?);
            let (c0, c1) = (g.alloc(32 * n * 2)?, g.alloc(32 * n * 2)?);
            let ld = |kh: KernelHandle,
                      a: DevicePtr,
                      w: DevicePtr,
                      s: DevicePtr,
                      c: DevicePtr,
                      m: u32,
                      kk: u32| {
                KernelLaunch::new(g, kh)
                    .grid([div_ceil(n as u32, 16), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(w)
                    .arg_ptr(s)
                    .arg_f32(SCALE2)
                    .arg_ptr(c)
                    .arg_u32(m)
                    .arg_u32(n as u32)
                    .arg_u32(kk)
                    .arg_u32((k / 2) as u32)
                    .arg_u32((k / GROUP_SIZE) as u32)
                    .launch(0)
            };
            for m in [1u32, 8, 16, 32] {
                let (tck, ldk) = match m {
                    1..=8 => (tc8, l8),
                    9..=16 => (tc16, l16),
                    _ => (tc32, l32),
                };
                launch(g, tck, true, ad, wd, wsd, c_tc, m, n as u32, k as u32)?;
                ld(ldk, ad, wd, wsd, c_ref, m, k as u32)?;
                ld(ldk, a0, wd, wsd, c0, m, half as u32)?;
                ld(
                    ldk,
                    a1,
                    wd.offset(half / 2),
                    wsd.offset(half / GROUP_SIZE),
                    c1,
                    m,
                    half as u32,
                )?;
                g.synchronize(0)?;
                let cnt = m as usize * n;
                let (full, same) = (down(g, c_tc, cnt)?, down(g, c_ref, cnt)?);
                let (h0, h1) = (down(g, c0, cnt)?, down(g, c1, cnt)?);
                let bitwise = full
                    .iter()
                    .zip(&same)
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                let mut worst = 0f32;
                for ((x, y0), y1) in full.iter().zip(&h0).zip(&h1) {
                    let y = y0 + y1;
                    // Each half rounds to BF16 on its own, so scale by the halves too.
                    let tol = x.abs().max(y0.abs() + y1.abs()) * (3.0 / 128.0) + 2e-3;
                    worst = worst.max((x - y).abs() / tol);
                }
                let ok = bitwise && worst <= 1.0;
                fail |= !ok;
                println!(
                    "{name} M={m}: strided natural-stride bitwise {bitwise}, K-halves worst/tol {worst:.3} {}",
                    if ok { "ok" } else { "FAIL" }
                );
            }
        }
    }
    std::process::exit(if fail { 1 } else { 0 });
}
