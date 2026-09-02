// SPDX-License-Identifier: AGPL-3.0-only

//! True CPU-reference oracle for the production Strix NVFP4 M128 prefill GEMMs.
//!
//! The CPU independently dequantizes logical E2M1 weights with E4M3 block
//! scales, accumulates `A @ W^T` in f32, and rounds only the output to BF16.
//! The GPU receives the exact production transposes `[K/2,N]` and `[K/16,N]`.
//!
//! ```text
//! cargo run --release -p spark-model --features cuda,gpu-examples \
//!   --example w4a16_cpu_reference_microtest -- [M] [N] [K] [seed]
//! ```
//! Defaults are `M=129 N=512 K=512 seed=0x51A7C0DE`.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const GROUP_SIZE: usize = 16;
const COSINE_GATE: f64 = 0.999;
const GLOBAL_SCALE: f32 = 0.125;
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
// Finite positive E4M3 normals spanning 0.03125 through 2.5.
const SCALE_PALETTE: [u8; 6] = [0x10, 0x20, 0x29, 0x32, 0x3C, 0x42];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn signed_quarter(&mut self) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (unit - 0.5) * 0.5
    }
}

struct Shape {
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
}

fn parse_cli() -> Result<Shape> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 4 {
        bail!("usage: w4a16_cpu_reference_microtest [M] [N] [K] [seed]");
    }
    let dim = |i: usize, default: usize, name: &str| -> Result<usize> {
        args.get(i).map_or(Ok(default), |s| {
            s.parse().with_context(|| format!("invalid {name}: {s}"))
        })
    };
    let seed = args.get(3).map_or(Ok(0x51A7_C0DE), |s| {
        let (digits, radix) = s
            .strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))
            .map_or((s.as_str(), 10), |hex| (hex, 16));
        u64::from_str_radix(digits, radix).with_context(|| format!("invalid seed: {s}"))
    })?;
    let shape = Shape {
        m: dim(0, 129, "M")?,
        n: dim(1, 512, "N")?,
        k: dim(2, 512, "K")?,
        seed,
    };
    if shape.m == 0 || shape.n == 0 || shape.k == 0 {
        bail!("M, N, and K must be nonzero");
    }
    if shape.n % 16 != 0 || shape.k % 32 != 0 {
        bail!("production M128 kernels require N % 16 == 0 and K % 32 == 0");
    }
    if shape.m > u32::MAX as usize || shape.n > u32::MAX as usize || shape.k > u32::MAX as usize {
        bail!("M, N, and K must fit in u32 kernel arguments");
    }
    checked_product(shape.m, shape.k, "M*K")?;
    let output_elements = checked_product(shape.m, shape.n, "M*N")?;
    checked_product(output_elements, 2, "2*M*N")?;
    checked_product(shape.n, shape.k / 2, "N*K/2")?;
    checked_product(shape.n, shape.k / GROUP_SIZE, "N*K/16")?;
    Ok(shape)
}

fn checked_product(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("{label} overflows usize"))
}

fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    let value = if exp == 0 {
        mant as f32 * 2f32.powi(-9)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    };
    sign * value
}

struct Inputs {
    a_bits: Vec<u16>,
    packed_nt: Vec<u8>,
    scale_nt: Vec<u8>,
    packed_t: Vec<u8>,
    scale_t: Vec<u8>,
}

fn generate(shape: &Shape) -> Inputs {
    let mut rng = Rng(shape.seed);
    let half_k = shape.k / 2;
    let groups = shape.k / GROUP_SIZE;
    let a_bits = (0..shape.m * shape.k)
        .map(|_| bf16::from_f32(rng.signed_quarter()).to_bits())
        .collect();
    let mut packed_nt: Vec<u8> = (0..shape.n * half_k)
        .map(|_| {
            let bits = rng.next();
            (((bits >> 12) as u8 & 0x0F) << 4) | ((bits >> 4) as u8 & 0x0F)
        })
        .collect();
    // One byte guarantees both E2M1 +0 (low nibble) and -0 (high nibble).
    packed_nt[0] = 0x80;
    let mut scale_nt: Vec<u8> = (0..shape.n * groups)
        .map(|_| SCALE_PALETTE[(rng.next() % SCALE_PALETTE.len() as u64) as usize])
        .collect();
    for (g, &scale) in SCALE_PALETTE.iter().enumerate().take(groups) {
        scale_nt[g] = scale;
    }

    // Verbatim layout transform from QuantizedWeight::transpose_for_gemm.
    let mut packed_t = vec![0; shape.n * half_k];
    for i in 0..shape.n {
        for j in 0..half_k {
            packed_t[j * shape.n + i] = packed_nt[i * half_k + j];
        }
    }
    let mut scale_t = vec![0; shape.n * groups];
    for i in 0..shape.n {
        for j in 0..groups {
            scale_t[j * shape.n + i] = scale_nt[i * groups + j];
        }
    }
    debug_assert_eq!(E2M1[0].to_bits(), 0);
    debug_assert_eq!(E2M1[8].to_bits(), 0x8000_0000);
    debug_assert_eq!(e4m3_to_f32(0x10), 0.03125);
    debug_assert_eq!(e4m3_to_f32(0x38), 1.0);
    debug_assert_eq!(e4m3_to_f32(0x42), 2.5);
    debug_assert_eq!(packed_t[0], 0x80);
    Inputs {
        a_bits,
        packed_nt,
        scale_nt,
        packed_t,
        scale_t,
    }
}

fn dequant_row(n: usize, shape: &Shape, input: &Inputs, out: &mut [f32]) {
    let half_k = shape.k / 2;
    let groups = shape.k / GROUP_SIZE;
    let packed = &input.packed_nt[n * half_k..(n + 1) * half_k];
    let scales = &input.scale_nt[n * groups..(n + 1) * groups];
    for (group, &scale_byte) in scales.iter().enumerate() {
        let scale = e4m3_to_f32(scale_byte) * GLOBAL_SCALE;
        for pair in 0..GROUP_SIZE / 2 {
            let k = group * GROUP_SIZE + pair * 2;
            let byte = packed[k / 2];
            out[k] = E2M1[(byte & 0x0F) as usize] * scale;
            out[k + 1] = E2M1[(byte >> 4) as usize] * scale;
        }
    }
}

fn cpu_reference(shape: &Shape, input: &Inputs) -> Result<Vec<f32>> {
    let a: Vec<f32> = input
        .a_bits
        .iter()
        .map(|&x| bf16::from_bits(x).to_f32())
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(16)
        .min(shape.n);
    let cols_per_thread = shape.n.div_ceil(threads);
    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for n0 in (0..shape.n).step_by(cols_per_thread) {
            let n1 = (n0 + cols_per_thread).min(shape.n);
            let a = &a;
            handles.push((
                n0,
                n1,
                scope.spawn(move || {
                    let cols = n1 - n0;
                    let mut local = vec![0.0f32; shape.m * cols];
                    let mut weight = vec![0.0f32; shape.k];
                    for n in n0..n1 {
                        dequant_row(n, shape, input, &mut weight);
                        for m in 0..shape.m {
                            let a_row = &a[m * shape.k..(m + 1) * shape.k];
                            let mut sum = 0.0f32;
                            for k in 0..shape.k {
                                sum += a_row[k] * weight[k];
                            }
                            local[m * cols + n - n0] = bf16::from_f32(sum).to_f32();
                        }
                    }
                    local
                }),
            ));
        }
        let mut output = vec![0.0f32; shape.m * shape.n];
        for (n0, n1, handle) in handles {
            let local = handle
                .join()
                .map_err(|_| anyhow!("CPU reference worker panicked"))?;
            let cols = n1 - n0;
            for m in 0..shape.m {
                output[m * shape.n + n0..m * shape.n + n1]
                    .copy_from_slice(&local[m * cols..(m + 1) * cols]);
            }
        }
        Ok::<_, anyhow::Error>(output)
    })?;
    if chunks.iter().any(|x| !x.is_finite()) {
        bail!("CPU reference produced a nonfinite BF16 output");
    }
    Ok(chunks)
}

fn to_le_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    shape: &Shape,
    a: DevicePtr,
    packed_t: DevicePtr,
    scale_t: DevicePtr,
    output: DevicePtr,
) -> Result<Vec<f32>> {
    let output_elements = checked_product(shape.m, shape.n, "M*N")?;
    let output_bytes = checked_product(output_elements, 2, "2*M*N")?;
    // 0xFFFF is a BF16 NaN sentinel: any unwritten M/N tail fails finiteness.
    gpu.memset(output, 0xFF, output_bytes)?;
    KernelLaunch::new(gpu, kernel)
        .grid([
            shape.n.div_ceil(128) as u32,
            shape.m.div_ceil(128) as u32,
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(packed_t)
        .arg_ptr(scale_t)
        .arg_f32(GLOBAL_SCALE)
        .arg_ptr(output)
        .arg_u32(shape.m as u32)
        .arg_u32(shape.n as u32)
        .arg_u32(shape.k as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let mut raw = vec![0u8; output_bytes];
    gpu.copy_d2h(output, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect())
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

struct Stats {
    cosine: f64,
    max_abs: f64,
    mean_abs: f64,
    row_min: f64,
    row_mean: f64,
    row_max: f64,
    worst_row: usize,
    finite: bool,
}

fn compare(gpu: &[f32], cpu: &[f32], shape: &Shape) -> Stats {
    let finite = gpu.iter().chain(cpu).all(|x| x.is_finite());
    if !finite {
        return Stats {
            cosine: f64::NAN,
            max_abs: f64::NAN,
            mean_abs: f64::NAN,
            row_min: f64::NAN,
            row_mean: f64::NAN,
            row_max: f64::NAN,
            worst_row: 0,
            finite: false,
        };
    }
    let mut max_abs = 0.0f64;
    let mut abs_sum = 0.0f64;
    for (&x, &y) in gpu.iter().zip(cpu) {
        let diff = (x as f64 - y as f64).abs();
        max_abs = max_abs.max(diff);
        abs_sum += diff;
    }
    let mut row_min = f64::INFINITY;
    let mut row_max = f64::NEG_INFINITY;
    let mut row_sum = 0.0;
    let mut worst_row = 0;
    for row in 0..shape.m {
        let range = row * shape.n..(row + 1) * shape.n;
        let value = cosine(&gpu[range.clone()], &cpu[range]);
        if value < row_min {
            row_min = value;
            worst_row = row;
        }
        row_max = row_max.max(value);
        row_sum += value;
    }
    Stats {
        cosine: cosine(gpu, cpu),
        max_abs,
        mean_abs: abs_sum / gpu.len() as f64,
        row_min,
        row_mean: row_sum / shape.m as f64,
        row_max,
        worst_row,
        finite: true,
    }
}

fn main() -> Result<()> {
    let shape = parse_cli()?;
    println!(
        "W4A16 CPU oracle: M={} N={} K={} seed=0x{:X} global_scale={GLOBAL_SCALE}",
        shape.m, shape.n, shape.k, shape.seed
    );

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let mut kernels = vec![(
        "w4a16_gemm_t_m128",
        gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
    )];
    match gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16") {
        Ok(kernel) => kernels.push(("w4a16_gemm_t_m128_bf16", kernel)),
        Err(error) => println!("w4a16_gemm_t_m128_bf16: SKIP ({error})"),
    }

    let input = generate(&shape);
    let started = Instant::now();
    let reference = cpu_reference(&shape, &input)?;
    println!(
        "CPU f32 dequant+matmul completed in {:.3?}",
        started.elapsed()
    );

    let a = upload(gpu, &to_le_bytes(&input.a_bits))?;
    let packed_t = upload(gpu, &input.packed_t)?;
    let scale_t = upload(gpu, &input.scale_t)?;
    let output_elements = checked_product(shape.m, shape.n, "M*N")?;
    let output = gpu.alloc(checked_product(output_elements, 2, "2*M*N")?)?;
    let mut all_pass = true;
    for (name, kernel) in kernels {
        let actual = run_kernel(gpu, kernel, &shape, a, packed_t, scale_t, output)?;
        let stats = compare(&actual, &reference, &shape);
        let pass = stats.finite && stats.cosine >= COSINE_GATE && stats.row_min >= COSINE_GATE;
        all_pass &= pass;
        println!(
            "{name}: cosine={:.8} max_abs={:.6e} mean_abs={:.6e} {}",
            stats.cosine,
            stats.max_abs,
            stats.mean_abs,
            if pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  row-wise cosine: min={:.8} mean={:.8} max={:.8} worst_row={}",
            stats.row_min, stats.row_mean, stats.row_max, stats.worst_row
        );
    }
    for ptr in [a, packed_t, scale_t, output] {
        let _ = gpu.free(ptr);
    }
    if !all_pass {
        bail!("nonfinite output or cosine below {COSINE_GATE}");
    }
    println!("RESULT: PASS (available kernels vs BF16-rounded CPU reference)");
    Ok(())
}
