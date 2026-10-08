// SPDX-License-Identifier: AGPL-3.0-only

//! Wall-time A/B on the Qwen3.6/3.8-27B dense-FFN prefill shapes:
//!
//!   fused      production `w4a16_gemm_t_m128` — in-kernel NVFP4->BF16 dequant
//!              + BF16 WMMA. Walls at ~19 TF/s on gfx1151 (occupancy AND the
//!              smem-staged dequant pipeline).
//!   deq+lt     `dequant_nvfp4_to_bf16` into a transient [n,k] bf16 scratch,
//!              then `cublaslt::bf16_gemm_act_weight_t` (hipBLASLt on native
//!              HIP). The Oct-4 oracle bench measured hipBLASLt at ~29-36 TF/s
//!              on the fn-gdn/27b bf16 shapes, so this arm trades one weight
//!              scratch (~89-178 MB, per call site, not persistent) for the
//!              vendor GEMM path.
//!
//! Reports fused ms/TF vs lt ms/TF plus the combined deq+lt figure — the
//! dequant pass is measured when `dequant_nvfp4_bf16` is registered for the
//! target (strix-hip gains it via the common/ symlink), analytical otherwise.
//!
//! Usage: cargo run --release -p spark-model --example w4a16_lt_bench --features cuda,gpu-examples

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::cublaslt;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const GROUP_SIZE: usize = 16;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// Timed loop for the production fused kernel (`w4a16_gemm_t_m128`,
/// transposed packed [K/2,N] + [K/16,N] scale layout).
#[allow(clippy::too_many_arguments)]
fn time_fused(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed_t: DevicePtr,
    scale_t: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let go = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed_t)
            .arg_ptr(scale_t)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        go()?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        go()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}


/// Same launch contract as the fused m128 but with M_TILE=64 grids
/// (w4a16_gemm_t_m64_bf16 / w4a16_gemm_t_k64): grid [N/128, M/64].
#[allow(clippy::too_many_arguments)]
fn time_m64(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed_t: DevicePtr,
    scale_t: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let go = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed_t)
            .arg_ptr(scale_t)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32) // ldb — transposed B rows are exactly N apart
            .launch(stream)
    };
    for _ in 0..3 {
        go()?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        go()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

/// fp8_fp8_gemm_t_m128: A[M,K] e4m3 x B[N,K] e4m3 -> C[M,N] bf16. The "crush
/// both operands" MMA — m16n8k32.e4m3 class math, half the smem bytes.
fn time_f8f8(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a8: DevicePtr,
    b8: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let go = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(b8)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        go()?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        go()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}


/// Timed loop for `dequant_nvfp4_to_bf16` alone (natural [n,k/2] packed +
/// [n,k/16] scale layout -> bf16 [n,k] scratch). grid=[N], block=256.
#[allow(clippy::too_many_arguments)]
fn time_dequant(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    packed: DevicePtr,
    scales: DevicePtr,
    out_bf16: DevicePtr,
    scale2: f32,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let go = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(packed)
            .arg_ptr(scales)
            .arg_ptr(out_bf16)
            .arg_f32(scale2)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    go()?;
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        go()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let fused = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    let m64k = gpu.kernel("w4a16", "w4a16_gemm_t")?;      // N64/M64 base — lowest smem
    let k64k = gpu.kernel("w4a16", "w4a16_gemm_t_k64")?;   // deep-K step
    let f8f8 = gpu.kernel("w4a16", "fp8_fp8_gemm_t_m128")?; // fp8 x fp8 MMA
    let dq = gpu
        .kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")
        .unwrap_or(KernelHandle(0)); // absent on targets that skip the module

    // 27B dense FFN: H=5120, inter=17408. gate/up: N=17408 K=5120;
    // down: N=5120 K=17408. Prefill chunk M: 1024 / 2048 / 4096.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gate/up M=1024", 1024, 17408, 5120),
        ("down    M=1024", 1024, 5120, 17408),
        ("gate/up M=2048", 2048, 17408, 5120),
        ("down    M=2048", 2048, 5120, 17408),
        ("gate/up M=4096", 4096, 17408, 5120),
    ];

    println!("=== w4a16 fused vs dequant->scratch + hipBLASLt bf16 ===\n");
    println!(
        "{:<16} {:>6} {:>6} {:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>8} {:>6} | {:>6} {:>6} {:>6}",
        "shape", "M", "N", "K", "fused ms", "fused TF", "lt ms", "lt TF", "deq~ms", "d+lt x", "m64 TF", "k64 TF", "f8f8 TF"
    );
    println!("{}", "-".repeat(100));

    let mut rng = Rng(0x1234);
    for &(label, m, n, k) in shapes {
        let num_groups = k / GROUP_SIZE;
        // Baseline's transposed layouts: packed [K/2,N], scale [K/16,N].
        let packed_t: Vec<u8> = (0..(k / 2) * n).map(|_| rng.next_u64() as u8).collect();
        let scale_t: Vec<u8> = (0..num_groups * n)
            .map(|_| (((5 + (rng.next_u64() % 5)) as u8) << 3) & 0x7F)
            .collect();
        // Dequant leg's natural layouts: packed [N,K/2], scale [N,K/16].
        // Natural-layout packed [N,K/2] + scales [N,K/16] for the dequant arm —
        // produces the real bf16 scratch the lt GEMM then consumes.
        let packed_n: Vec<u8> = (0..n * (k / 2)).map(|_| rng.next_u64() as u8).collect();
        let scale_n: Vec<u8> = (0..n * (k / 16))
            .map(|_| (((5 + (rng.next_u64() % 5)) as u8) << 3) & 0x7F)
            .collect();
        let a: Vec<u8> = (0..m * k * 2).map(|_| rng.next_u64() as u8).collect();

        let a_ptr = upload(gpu, &a)?;
        let pt_ptr = upload(gpu, &packed_t)?;
        let st_ptr = upload(gpu, &scale_t)?;
        let pn_ptr = upload(gpu, &packed_n)?;
        let sn_ptr = upload(gpu, &scale_n)?;
        let w_bf16 = gpu.alloc(n * k * 2)?;
        let c = gpu.alloc(m * n * 2)?;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let iters = if m >= 4096 { 20 } else { 40 };

        let t_fused = time_fused(gpu, stream, fused, a_ptr, pt_ptr, st_ptr, 0.5, c, m, n, k, iters)?;
        // Real dequant timing (registered under the raw stem name).
        let t_dq = match dq.0 {
            0 => (n as f64 * k as f64 * 2.5) / 200e9, // module absent: est. only
            _ => time_dequant(gpu, stream, dq, pn_ptr, sn_ptr, w_bf16, 0.5, n, k, 6)?,
        };
        // hipBLASLt arm: act [m,k] bf16 x weight [n,k] bf16 (transposed-B
        // contract) -> out [m,n] bf16.
        let lt_once = || cublaslt::bf16_gemm_act_weight_t(a_ptr.0, w_bf16.0, c.0, m as u32, n as u32, k as u32, stream);
        lt_once()?;
        gpu.synchronize(stream)?;
        let t0 = Instant::now();
        for _ in 0..iters {
            lt_once()?;
        }
        gpu.synchronize(stream)?;
        let t_lt = t0.elapsed().as_secs_f64() / iters as f64;

        // fp8 arm: weight scratch halves to e4m3 [n,k] (+ block scales), act
        // also e4m3 — cheaper dequant write AND the fp8-lt GEMM. Uses the
        // blkscaled entry if present; skipped cleanly when unavailable.
        // Occupancy arms on the same transposed-packed contract:
        // m64 lossless bf16 twin (M_TILE=64, ~half smem_A -> 2-3 CTAs/CU) and
        // t_k64 (deep-K step). Same packed_t/scale_t buffers.
        let t_m64 = time_m64(gpu, stream, m64k, a_ptr, pt_ptr, st_ptr, 0.5, c, m, n, k, iters)?;
        let t_k64 = time_m64(gpu, stream, k64k, a_ptr, pt_ptr, st_ptr, 0.5, c, m, n, k, iters)?;
        // fp8fp8: both operands e4m3 (random bytes — speed only).
        let a8 = upload(gpu, &(0..m * k).map(|_| rng.next_u64() as u8).collect::<Vec<u8>>())?;
        let b8 = upload(gpu, &(0..n * k).map(|_| rng.next_u64() as u8).collect::<Vec<u8>>())?;
        let t_f8f8 = time_f8f8(gpu, stream, f8f8, a8, b8, c, m, n, k, iters)?;

        println!(
            "{label:<16} {m:>6} {n:>6} {k:>6} | {:>8.3}ms {:>6.1} | {:>8.3}ms {:>6.1} | {:>7.3}ms {:>5.2}x | {:>6.1} | {:>6.1} | {:>6.1}",
            t_fused * 1e3,
            flops / t_fused / 1e12,
            t_lt * 1e3,
            flops / t_lt / 1e12,
            t_dq * 1e3,
            t_fused / (t_dq + t_lt),
            flops / t_m64 / 1e12,
            flops / t_k64 / 1e12,
            flops / t_f8f8 / 1e12,
        );

        for ptr in [a_ptr, pt_ptr, st_ptr, pn_ptr, sn_ptr, w_bf16, a8, b8, c] {
            let _ = gpu.free(ptr);
        }
    }
    println!("{}", "-".repeat(100));
    Ok(())
}
