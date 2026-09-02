// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle + timing harness for the two Strix BF16 dense GEMMs the Qwen3.8
//! BF16-preservation prefill routes through: `gemm::dense_gemm_bf16` (scalar,
//! historically bit-verified vs CPU at router shapes) and
//! `gemm::dense_gemm_bf16_pipelined` (the HIP WMMA port whose gb10 sibling is
//! unresolved on this target).
//!
//! Motivation: the preservation serve prefills long prompts at ~12 tok/s with
//! a degenerate first token (`<|audio_pad|>`) and a period-2 decode loop, while
//! 16-token prompts are clean. Every failing prompt crossed a partial 128-row
//! M tile of the pipelined kernel; this harness sweeps M across tile edges,
//! compares every row against the scalar kernel and (small M) a CPU f32
//! reference, and times both kernels.
//!
//! ```text
//! cargo build --release -p spark-model --no-default-features \
//!   --features cuda,gpu-examples --example dense_gemm_bf16_oracle
//! ```

use std::time::Instant;

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bf16_val(&mut self) -> u16 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        bf16::from_f32((unit - 0.5) * 0.5).to_bits()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        if aa == bb { 1.0 } else { 0.0 }
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn to_le_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    pipelined: bool,
    m: usize,
    n: usize,
    k: usize,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
) -> Result<Vec<f32>> {
    let output_bytes = m * n * 2;
    // 0xFFFF = BF16 NaN sentinel: unwritten tails fail finiteness.
    gpu.memset(c, 0xFF, output_bytes)?;
    let started = Instant::now();
    let mut launch = KernelLaunch::new(gpu, kernel);
    if pipelined {
        launch = launch
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1]);
    } else {
        launch = launch
            .grid([n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1])
            .block([16, 16, 1]);
    }
    launch
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    println!("    kernel wall {:.3?} (M={m} N={n} K={k})", started.elapsed());
    let mut raw = vec![0u8; output_bytes];
    gpu.copy_d2h(c, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect())
}

fn cpu_reference(a_bits: &[u16], b_bits: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let a: Vec<f32> = a_bits.iter().map(|&x| bf16::from_bits(x).to_f32()).collect();
    let b: Vec<f32> = b_bits.iter().map(|&x| bf16::from_bits(x).to_f32()).collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(16)
        .min(n);
    let cols = n.div_ceil(threads);
    let a_ref = &a;
    let b_ref = &b;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for n0 in (0..n).step_by(cols) {
            let n1 = (n0 + cols).min(n);
            handles.push(scope.spawn(move || {
                let mut local = vec![0.0f32; m * (n1 - n0)];
                for n in n0..n1 {
                    for mm in 0..m {
                        let a_row = &a_ref[mm * k..(mm + 1) * k];
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += a_row[kk] * b_ref[n * k + kk];
                        }
                        local[mm * (n1 - n0) + n - n0] = bf16::from_f32(sum).to_f32();
                    }
                }
                local
            }));
        }
        let mut out = vec![0.0f32; m * n];
        for (i, handle) in handles.into_iter().enumerate() {
            let local = handle.join().expect("cpu worker");
            let n0 = i * cols;
            let n1 = (n0 + cols).min(n);
            for mm in 0..m {
                out[mm * n + n0..mm * n + n1]
                    .copy_from_slice(&local[mm * (n1 - n0)..(mm + 1) * (n1 - n0)]);
            }
        }
        out
    })
}

fn diff_stats(gpu: &[f32], reference: &[f32], m: usize, n: usize) -> (f64, f64, f64, usize) {
    let mut max_abs = 0.0f64;
    let mut exact = 0usize;
    for (x, y) in gpu.iter().zip(reference) {
        let d = f64::from(*x - *y).abs();
        max_abs = max_abs.max(d);
        if d == 0.0 {
            exact += 1;
        }
    }
    let mut row_min = f64::INFINITY;
    let mut worst = 0usize;
    for row in 0..m {
        let range = row * n..(row + 1) * n;
        let v = cosine(&gpu[range.clone()], &reference[range]);
        if v < row_min {
            row_min = v;
            worst = row;
        }
    }
    (max_abs, exact as f64 / (m * n) as f64, row_min, worst)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k: usize = args.first().map_or(Ok(5120), |s| s.parse())?;
    let reps: usize = args.get(1).map_or(Ok(3), |s| s.parse())?;
    if k % 16 != 0 {
        bail!("K must be a multiple of the pipelined K-step 16");
    }

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let scalar = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let pipelined = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let tc = gpu
        .kernel("gemm_tc", "dense_gemm_tc")
        .map(|k| Some(k))
        .unwrap_or_else(|e| {
            println!("gemm_tc: SKIP ({e})");
            None
        });
    println!(
        "kernels resolved: scalar + pipelined + tc={} (K={k}, reps={reps})",
        tc.is_some()
    );

    // (M, N, run_cpu_reference)
    let cases = [
        (16usize, 5120usize, true),
        (128, 5120, true),
        (129, 5120, true),
        (512, 5120, false),
        (513, 5120, true),
        (1024, 5120, false),
        (2049, 5120, false),
        (513, 8192, false),
        (2049, 8192, false),
        // The preserved-FFN shape that profiled at 7.3 s/layer in the serve.
        (823, 17408, false),
    ];

    for (m, n, run_cpu) in cases {
        println!("== M={m} N={n} K={k} ==");
        let mut rng = Rng(0x51A7_C0DE ^ (m as u64) ^ ((n as u64) << 8));
        let a_bits: Vec<u16> = (0..m * k).map(|_| rng.bf16_val()).collect();
        let b_bits: Vec<u16> = (0..n * k).map(|_| rng.bf16_val()).collect();
        let a = upload(gpu, &to_le_bytes(&a_bits))?;
        let b = upload(gpu, &to_le_bytes(&b_bits))?;
        let c = gpu.alloc(m * n * 2)?;

        // Timing reps for the pipelined kernel (the production prefill path).
        let mut walls = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let _ = run_gemm(gpu, pipelined, true, m, n, k, a, b, c)?;
            walls.push(t.elapsed());
        }
        walls.sort();
        let best = walls[0].as_secs_f64();
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        println!(
            "    pipelined: best {:.1} ms -> {:.0} tok/s, {:.2} TFLOPS",
            best * 1e3,
            m as f64 / best,
            flops / best / 1e12
        );
        let piped = run_gemm(gpu, pipelined, true, m, n, k, a, b, c)?;
        if piped.iter().any(|x| !x.is_finite()) {
            println!("    pipelined: NONFINITE OUTPUT (unwritten tail or NaN)");
        }
        if let Some(tc_kernel) = tc {
            // tc wrapper geometry: grid (ceil(N/64), ceil(M/16)), block 128.
            gpu.memset(c, 0xFF, m * n * 2)?;
            let t = Instant::now();
            KernelLaunch::new(gpu, tc_kernel)
                .grid([n.div_ceil(64) as u32, m.div_ceil(16) as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(c)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let wall = t.elapsed().as_secs_f64();
            println!(
                "    tc:        {wall:.1} ms -> {:.0} tok/s, {:.2} TFLOPS",
                m as f64 / wall,
                flops / wall / 1e12
            );
            let mut raw = vec![0u8; m * n * 2];
            gpu.copy_d2h(c, &mut raw)?;
            let tc_out: Vec<f32> = raw
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect();
            let (max_abs, exact, row_min, worst) = diff_stats(&tc_out, &piped, m, n);
            println!(
                "    tc vs pipelined: max_abs={max_abs:.6e} exact={exact:.6} row_cosine_min={row_min:.8} worst_row={worst}"
            );
        }
        let scalar = run_gemm(gpu, scalar, false, m, n, k, a, b, c)?;
        let (max_abs, exact, row_min, worst) = diff_stats(&piped, &scalar, m, n);
        println!(
            "    pipelined vs scalar: max_abs={max_abs:.6e} exact={exact:.6} row_cosine_min={row_min:.8} worst_row={worst}"
        );

        if run_cpu {
            let t = Instant::now();
            let reference = cpu_reference(&a_bits, &b_bits, m, n, k);
            println!("    cpu reference built in {:.1?}", t.elapsed());
            for (name, out) in [("scalar", &scalar), ("pipelined", &piped)] {
                let (max_abs, exact, row_min, worst) = diff_stats(out, &reference, m, n);
                println!(
                    "    {name} vs cpu: max_abs={max_abs:.6e} exact={exact:.6} row_cosine_min={row_min:.8} worst_row={worst}"
                );
            }
        }
        for ptr in [a, b, c] {
            let _ = gpu.free(ptr);
        }
    }
    println!("DONE");
    Ok(())
}
