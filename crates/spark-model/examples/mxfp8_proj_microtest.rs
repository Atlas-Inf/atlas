// SPDX-License-Identifier: AGPL-3.0-only
//! Correctness and speed of the native MXFP8 kernels (`mxfp8_proj.cu`).
//!
//! Checks, on random E4M3 bytes and E8M0 scales of a real GDN shape:
//! 1. `mxfp8_proj_dequant` is byte-identical to `atlas_core::modelopt_codec::mxfp8_to_bf16`;
//! 2. `mxfp8_proj_gemv` matches an f64 reference over that exact BF16 weight;
//! 3. every row of `mxfp8_proj_batchm` (M = 1..=8) is bit-identical to `mxfp8_proj_gemv`;
//!
//! then times `mxfp8_proj_gemv` against `dense_gemv_bf16` over the dequantized
//! weight (the path FP4-Local ran before this kernel). Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example mxfp8_proj_microtest [-- N K]

use anyhow::{Result, ensure};
use atlas_core::modelopt_codec::mxfp8_to_bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MAXM: usize = 8;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f32 {
        ((self.next() as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn bf16_at(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
}

fn gemv(
    g: &dyn GpuBackend,
    k_gemv: KernelHandle,
    (a, w, s, c): (DevicePtr, DevicePtr, DevicePtr, DevicePtr),
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, k_gemv)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

fn time_ms(g: &dyn GpuBackend, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // GDN in_proj_qkvz of Qwen3.8-Flash-Next: [16384, 2560].
    let n: usize = args.get(1).map_or(Ok(16384), |v| v.parse())?;
    let k: usize = args.get(2).map_or(Ok(2560), |v| v.parse())?;
    ensure!(k.is_multiple_of(32), "K must be a multiple of 32");

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_gemv = g.kernel("mxfp8_proj", "mxfp8_proj_gemv")?;
    let k_batch = g.kernel("mxfp8_proj", "mxfp8_proj_batchm")?;
    let k_deq = g.kernel("mxfp8_proj", "mxfp8_proj_dequant")?;
    let k_bf16 = g.kernel("gemv", "dense_gemv_bf16")?;

    let mut rng = Lcg(0x4d58_4650_3800_0001);
    // Avoid the NaN bytes 0x7f/0xff: the reference maps them to 0 as well,
    // but real checkpoints never carry them.
    let weight: Vec<u8> = (0..n * k)
        .map(|_| match (rng.next() & 0xFF) as u8 {
            0x7F | 0xFF => 0,
            b => b,
        })
        .collect();
    // Scales around the checkpoint's range: 2^-12 .. 2^-4.
    let scale: Vec<u8> = (0..n * k / 32)
        .map(|_| 115 + (rng.next() % 9) as u8)
        .collect();
    let act: Vec<u8> = (0..MAXM * k)
        .flat_map(|_| {
            let v = half::bf16::from_f32(rng.unit() * 2.0 - 1.0);
            v.to_bits().to_le_bytes()
        })
        .collect();

    let (w_d, s_d, a_d) = (up(g, &weight)?, up(g, &scale)?, up(g, &act)?);
    let deq_d = g.alloc(n * k * 2)?;
    let c1 = g.alloc(MAXM * n * 2)?;
    let cb = g.alloc(MAXM * n * 2)?;
    let mut pass = true;

    // 1. Dequant is the codec, byte for byte.
    KernelLaunch::new(g, k_deq)
        .grid([div_ceil((n * k / 16) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(w_d)
        .arg_ptr(s_d)
        .arg_ptr(deq_d)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)?;
    g.synchronize(0)?;
    let deq = down(g, deq_d, n * k * 2)?;
    let want = mxfp8_to_bf16(&weight, &scale, n, k)?;
    let bad = deq
        .chunks_exact(2)
        .zip(want.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "dequant vs codec: {bad} of {} values differ {}",
        n * k,
        verdict(bad == 0)
    );
    pass &= bad == 0;

    // 2. GEMV against an f64 reference over the same exact weight.
    gemv(g, k_gemv, (a_d, w_d, s_d, c1), n, k)?;
    g.synchronize(0)?;
    let got = down(g, c1, n * 2)?;
    let (mut worst, mut num, mut den) = (0f64, 0f64, 0f64);
    for row in 0..n {
        let r: f64 = (0..k)
            .map(|i| f64::from(bf16_at(&act, i)) * f64::from(bf16_at(&want, row * k + i)))
            .sum();
        let e = (f64::from(bf16_at(&got, row)) - r).abs();
        worst = worst.max(e / r.abs().max(1.0));
        num += e * e;
        den += r * r;
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    // BF16 output rounding alone is 2^-9 relative.
    let ok = rel_l2 < 4e-3 && worst < 1.6e-2;
    println!(
        "gemv vs f64: rel_l2 {rel_l2:.3e}, worst {worst:.3e} {}",
        verdict(ok)
    );
    pass &= ok;

    // 3. Batched rows are the single-row kernel's bits.
    for m in 1..=MAXM {
        KernelLaunch::new(g, k_batch)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_d)
            .arg_ptr(w_d)
            .arg_ptr(s_d)
            .arg_ptr(cb)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(0)?;
        for t in 0..m {
            gemv(
                g,
                k_gemv,
                (a_d.offset(t * k * 2), w_d, s_d, c1.offset(t * n * 2)),
                n,
                k,
            )?;
        }
        g.synchronize(0)?;
        let (b, r) = (down(g, cb, m * n * 2)?, down(g, c1, m * n * 2)?);
        let bad = b
            .chunks_exact(2)
            .zip(r.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        println!(
            "batchm M={m}: {bad} mismatches vs mxfp8_proj_gemv {}",
            verdict(bad == 0)
        );
        pass &= bad == 0;
    }

    // Speed: native MXFP8 vs the BF16 expansion it replaces.
    let reps = 200;
    let t_mx = time_ms(g, reps, || gemv(g, k_gemv, (a_d, w_d, s_d, c1), n, k))?;
    let t_bf = time_ms(g, reps, || {
        KernelLaunch::new(g, k_bf16)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_d)
            .arg_ptr(deq_d)
            .arg_ptr(c1)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(0)
    })?;
    let gbs = |bytes: usize, ms: f64| bytes as f64 / ms / 1e6;
    println!(
        "mxfp8_proj_gemv {t_mx:.4} ms ({:.0} GB/s)  dense_gemv_bf16 {t_bf:.4} ms ({:.0} GB/s)  speedup {:.2}x",
        gbs(n * k + n * k / 32, t_mx),
        gbs(n * k * 2, t_bf),
        t_bf / t_mx
    );
    for m in [2usize, 4, 8] {
        let t = time_ms(g, reps, || {
            KernelLaunch::new(g, k_batch)
                .grid([div_ceil(n as u32, 4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_d)
                .arg_ptr(w_d)
                .arg_ptr(s_d)
                .arg_ptr(cb)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(0)
        })?;
        println!(
            "mxfp8_proj_batchm M={m} {t:.4} ms ({:.2}x of M=1)",
            t / t_mx
        );
    }

    if pass {
        println!("ALL PASS");
        Ok(())
    } else {
        std::process::exit(1)
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}
