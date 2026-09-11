// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle + microbench for the batch4 W4A8 DP4A decode GEMVs.
//!
//! Compares, at M ∈ {2,3,4}, the DP4A batch4 kernels against the float
//! `w4a16_gemv_batch4` reference (cosine gate) and times each arm. M=4 uses the
//! guard-free specialization (`w4a16_gemv_dp4a_batch4_d4`); M=2..3 use the
//! runtime-row-guarded kernel (`..._d4_dyn`) that the scheduler still dispatches
//! for K=2/K=3 verify. A final cross-check requires the two M=4 instantiations
//! to agree bit-for-bit, so the guard-free rewrite cannot silently change
//! numerics.

use anyhow::{Result, bail, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const MAXM: usize = 4;
const GROUP_SIZE: usize = 16;
const COSINE_GATE: f64 = 0.999;
const M_VALUES: [usize; 3] = [2, 3, 4];

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        let unit = ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32);
        low + (high - low) * unit
    }
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn launch_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    output: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(weight)
        .arg_ptr(scale)
        .arg_f32(1.0)
        .arg_ptr(output)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n = args.get(1).map_or(Ok(17408usize), |value| value.parse())?;
    let k = args.get(2).map_or(Ok(5120usize), |value| value.parse())?;
    ensure!(n.is_multiple_of(4));
    ensure!(k.is_multiple_of(GROUP_SIZE));

    let mut rng = Rng(0x51a7);
    let a_bits: Vec<u16> = (0..MAXM * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_bytes: Vec<u8> = a_bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let weight: Vec<u8> = (0..n * k / 2)
        .map(|_| (rng.next_u64() & 0xff) as u8)
        .collect();
    let groups = k / GROUP_SIZE;
    let scale: Vec<u8> = (0..n * groups)
        .map(|_| {
            let exponent = 6 + (rng.next_u64() % 3) as u8;
            let mantissa = (rng.next_u64() % 8) as u8;
            (exponent << 3) | mantissa
        })
        .collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let float_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch4")?;
    let quant_kernel = gpu.kernel("w4a16_gemv_dp4a", "quantize_act_int8_g16_batch4_d4")?;
    let fixed_kernel = gpu.kernel("w4a16_gemv_dp4a", "w4a16_gemv_dp4a_batch4_d4")?;
    let dyn_kernel = gpu.kernel("w4a16_gemv_dp4a", "w4a16_gemv_dp4a_batch4_d4_dyn")?;
    let fixed_dual = gpu.kernel("w4a16_gemv_dp4a", "w4a16_gemv_dp4a_dual_batch4_d4")?;
    let dyn_dual = gpu.kernel("w4a16_gemv_dp4a", "w4a16_gemv_dp4a_dual_batch4_d4_dyn")?;

    let a = upload(gpu, &a_bytes)?;
    let weight = upload(gpu, &weight)?;
    let scale = upload(gpu, &scale)?;
    let quantized = gpu.alloc(MAXM * k)?;
    let activation_scale = gpu.alloc(MAXM * groups * 4)?;
    let reference = gpu.alloc(MAXM * n * 2)?;
    let candidate = gpu.alloc(MAXM * n * 2)?;
    let candidate2 = gpu.alloc(MAXM * n * 2)?;

    let launch_quant = |m: usize| {
        KernelLaunch::new(gpu, quant_kernel)
            .grid([groups as u32, m as u32, 1])
            .block([GROUP_SIZE as u32, 1, 1])
            .arg_ptr(a)
            .arg_ptr(quantized)
            .arg_ptr(activation_scale)
            .arg_u32(m as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    let launch_dp4a = |kernel: KernelHandle, m: usize| {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(quantized)
            .arg_ptr(activation_scale)
            .arg_ptr(weight)
            .arg_ptr(scale)
            .arg_f32(1.0)
            .arg_ptr(candidate)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    let launch_dual = |kernel: KernelHandle, m: usize| {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(quantized)
            .arg_ptr(activation_scale)
            .arg_ptr(weight)
            .arg_ptr(scale)
            .arg_f32(1.0)
            .arg_ptr(candidate)
            .arg_ptr(weight)
            .arg_ptr(scale)
            .arg_f32(1.0)
            .arg_ptr(candidate2)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };

    let read = |ptr, rows: usize| -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; rows * n * 2];
        gpu.copy_d2h(ptr, &mut bytes)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect())
    };

    let time = |operation: &dyn Fn() -> Result<()>| -> Result<f64> {
        for _ in 0..10 {
            operation()?;
        }
        gpu.synchronize(stream)?;
        let started = Instant::now();
        for _ in 0..50 {
            operation()?;
        }
        gpu.synchronize(stream)?;
        Ok(started.elapsed().as_secs_f64() * 1e6 / 50.0)
    };

    let mut worst_cosine = 1.0f64;
    for m in M_VALUES {
        // m == 4 is the guard-free specialization; m == 2..3 keeps the guard.
        let (dp4a_kernel, dual_kernel) = if m == 4 {
            (fixed_kernel, fixed_dual)
        } else {
            (dyn_kernel, dyn_dual)
        };
        launch_batch4(
            gpu,
            float_kernel,
            a,
            weight,
            scale,
            reference,
            m,
            n,
            k,
            stream,
        )?;
        launch_quant(m)?;
        launch_dual(dual_kernel, m)?;
        gpu.synchronize(stream)?;

        let reference_values = read(reference, m)?;
        let candidate_values = [read(candidate, m)?, read(candidate2, m)?];
        let mut arm_cosine = 1.0f64;
        let mut arm_max_abs = 0.0f32;
        for values in &candidate_values {
            for row in 0..m {
                let (mut dot, mut reference_norm, mut candidate_norm) = (0.0f64, 0.0f64, 0.0f64);
                for index in row * n..(row + 1) * n {
                    let want = reference_values[index] as f64;
                    let actual = values[index] as f64;
                    dot += want * actual;
                    reference_norm += want * want;
                    candidate_norm += actual * actual;
                    arm_max_abs = arm_max_abs.max((reference_values[index] - values[index]).abs());
                }
                arm_cosine = arm_cosine.min(dot / (reference_norm.sqrt() * candidate_norm.sqrt()));
            }
        }
        worst_cosine = worst_cosine.min(arm_cosine);

        let float_us = time(&|| {
            launch_batch4(
                gpu,
                float_kernel,
                a,
                weight,
                scale,
                reference,
                m,
                n,
                k,
                stream,
            )
        })?;
        let quant_us = time(&|| launch_quant(m))?;
        let dp4a_us = time(&|| launch_dp4a(dp4a_kernel, m))?;
        let dual_us = time(&|| launch_dual(dual_kernel, m))?;
        println!(
            "N={n} K={k} M={m} cosine={arm_cosine:.9} max_abs={arm_max_abs:.6} float={float_us:.1}us quant={quant_us:.1}us dp4a={dp4a_us:.1}us dual={dual_us:.1}us dual_vs_two={:.3}x single_speedup={:.3}x dual_speedup={:.3}x",
            2.0 * dp4a_us / dual_us,
            float_us / (quant_us + dp4a_us),
            2.0 * float_us / (quant_us + dual_us),
        );
    }

    // The guard-free M=4 specialization must agree bit-for-bit with the
    // runtime-guarded instantiation at the same M.
    launch_quant(4)?;
    launch_dp4a(fixed_kernel, 4)?;
    gpu.synchronize(stream)?;
    let fixed_values = read(candidate, 4)?;
    launch_dp4a(dyn_kernel, 4)?;
    gpu.synchronize(stream)?;
    let dyn_values = read(candidate, 4)?;
    if fixed_values != dyn_values {
        let differing = fixed_values
            .iter()
            .zip(&dyn_values)
            .filter(|(a, b)| a != b)
            .count();
        bail!("guard-free vs guarded M=4 mismatch: {differing} elements differ");
    }
    println!("M=4 fixed-vs-dyn bit-identical: PASS");

    if worst_cosine < COSINE_GATE {
        bail!("cosine {worst_cosine:.9} below {COSINE_GATE}");
    }
    Ok(())
}
