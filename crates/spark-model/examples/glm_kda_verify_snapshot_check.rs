// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exactness check for GLM K=5 convolution/recurrent snapshot kernels.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const K: usize = 5;
const DCONV: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uni(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * ((self.next() >> 11) as f32 / (1u64 << 53) as f32)
    }
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes];
    gpu.copy_d2h(ptr, &mut out)?;
    Ok(out)
}

fn require_equal(name: &str, a: &[u8], b: &[u8]) -> Result<()> {
    let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
    println!("  {name}: {diff} differing bytes / {}", a.len());
    if diff != 0 {
        bail!("{name} is not bit-identical");
    }
    Ok(())
}

fn check_conv(gpu: &dyn GpuBackend, stream: u64, rng: &mut Rng) -> Result<()> {
    let dim = 3 * 8 * 128;
    let state_elems = dim * DCONV;
    let input: Vec<u16> = (0..K * dim).map(|_| bf16(rng.uni(-2.0, 2.0))).collect();
    let weight: Vec<u16> = (0..state_elems).map(|_| bf16(rng.uni(-0.5, 0.5))).collect();
    let state: Vec<f32> = (0..state_elems).map(|_| rng.uni(-1.0, 1.0)).collect();
    let input_p = upload(gpu, &bytes_u16(&input))?;
    let weight_p = upload(gpu, &bytes_u16(&weight))?;
    let serial_state = upload(gpu, &bytes_f32(&state))?;
    let fused_state = upload(gpu, &bytes_f32(&state))?;
    let serial_out = gpu.alloc(K * dim * 2)?;
    let fused_out = gpu.alloc(K * dim * 2)?;
    let serial_inter = gpu.alloc((K - 1) * state_elems * 4)?;
    let fused_inter = gpu.alloc((K - 1) * state_elems * 4)?;
    let base = gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill_tp")?;
    let fused = gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill_tp_snap")?;

    for t in 0..K {
        KernelLaunch::new(gpu, base)
            .grid([div_ceil(dim as u32, 32), 1, 1])
            .block([32, 8, 1])
            .arg_ptr(serial_state)
            .arg_ptr(input_p.offset(t * dim * 2))
            .arg_ptr(weight_p)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(serial_out.offset(t * dim * 2))
            .arg_u32(dim as u32)
            .arg_u32(DCONV as u32)
            .arg_u32(1)
            .arg_u32(dim as u32)
            .arg_u32(dim as u32)
            .launch(stream)?;
        if t + 1 < K {
            gpu.copy_d2d_async(
                serial_state,
                serial_inter.offset(t * state_elems * 4),
                state_elems * 4,
                stream,
            )?;
        }
    }
    KernelLaunch::new(gpu, fused)
        .grid([div_ceil(dim as u32, 32), 1, 1])
        .block([32, 8, 1])
        .arg_ptr(fused_state)
        .arg_ptr(input_p)
        .arg_ptr(weight_p)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(fused_out)
        .arg_ptr(fused_inter)
        .arg_u64(state_elems as u64)
        .arg_u32(dim as u32)
        .arg_u32(DCONV as u32)
        .arg_u32(K as u32)
        .arg_u32(dim as u32)
        .arg_u32(dim as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    require_equal(
        "conv output",
        &download(gpu, serial_out, K * dim * 2)?,
        &download(gpu, fused_out, K * dim * 2)?,
    )?;
    require_equal(
        "conv final state",
        &download(gpu, serial_state, state_elems * 4)?,
        &download(gpu, fused_state, state_elems * 4)?,
    )?;
    require_equal(
        "conv rollback states",
        &download(gpu, serial_inter, (K - 1) * state_elems * 4)?,
        &download(gpu, fused_inter, (K - 1) * state_elems * 4)?,
    )
}

fn check_recurrent(gpu: &dyn GpuBackend, stream: u64, rng: &mut Rng) -> Result<()> {
    let heads = 4usize;
    let dim = 128usize;
    let plane = heads * dim;
    let state_elems = heads * dim * dim;
    let qkv: Vec<u16> = (0..K * 3 * plane)
        .map(|_| bf16(rng.uni(-1.0, 1.0)))
        .collect();
    let gate: Vec<u16> = (0..K * plane).map(|_| bf16(rng.uni(-1.0, 1.0))).collect();
    let beta: Vec<u16> = (0..K * heads).map(|_| bf16(rng.uni(-1.0, 1.0))).collect();
    let a_log: Vec<f32> = (0..heads).map(|_| rng.uni(-0.5, 0.5)).collect();
    let dt_bias: Vec<f32> = (0..plane).map(|_| rng.uni(-0.5, 0.5)).collect();
    let state: Vec<f32> = (0..state_elems).map(|_| rng.uni(-0.2, 0.2)).collect();
    let qkv_p = upload(gpu, &bytes_u16(&qkv))?;
    let gate_p = upload(gpu, &bytes_u16(&gate))?;
    let beta_p = upload(gpu, &bytes_u16(&beta))?;
    let a_log_p = upload(gpu, &bytes_f32(&a_log))?;
    let dt_bias_p = upload(gpu, &bytes_f32(&dt_bias))?;
    let serial_state = upload(gpu, &bytes_f32(&state))?;
    let fused_state = upload(gpu, &bytes_f32(&state))?;
    let serial_out = gpu.alloc(K * plane * 2)?;
    let fused_out = gpu.alloc(K * plane * 2)?;
    let serial_inter = gpu.alloc((K - 1) * state_elems * 4)?;
    let fused_inter = gpu.alloc((K - 1) * state_elems * 4)?;
    let base = gpu.kernel("kda", "kda_recurrent_bf16")?;
    let fused = gpu.kernel("kda", "kda_recurrent_bf16_verify_snap")?;

    for t in 0..K {
        KernelLaunch::new(gpu, base)
            .grid([heads as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(qkv_p.offset(t * 3 * plane * 2))
            .arg_ptr(gate_p.offset(t * plane * 2))
            .arg_ptr(beta_p.offset(t * heads * 2))
            .arg_ptr(a_log_p)
            .arg_ptr(dt_bias_p)
            .arg_ptr(serial_state)
            .arg_ptr(serial_out.offset(t * plane * 2))
            .arg_u32(1)
            .arg_u32(heads as u32)
            .arg_u32(dim as u32)
            .arg_f32(-8.0)
            .launch(stream)?;
        if t + 1 < K {
            gpu.copy_d2d_async(
                serial_state,
                serial_inter.offset(t * state_elems * 4),
                state_elems * 4,
                stream,
            )?;
        }
    }
    KernelLaunch::new(gpu, fused)
        .grid([heads as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(qkv_p)
        .arg_ptr(gate_p)
        .arg_ptr(beta_p)
        .arg_ptr(a_log_p)
        .arg_ptr(dt_bias_p)
        .arg_ptr(fused_state)
        .arg_ptr(fused_out)
        .arg_ptr(fused_inter)
        .arg_u64(state_elems as u64)
        .arg_u32(K as u32)
        .arg_u32(heads as u32)
        .arg_u32(dim as u32)
        .arg_f32(-8.0)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    require_equal(
        "recurrent output",
        &download(gpu, serial_out, K * plane * 2)?,
        &download(gpu, fused_out, K * plane * 2)?,
    )?;
    require_equal(
        "recurrent final state",
        &download(gpu, serial_state, state_elems * 4)?,
        &download(gpu, fused_state, state_elems * 4)?,
    )?;
    require_equal(
        "recurrent rollback states",
        &download(gpu, serial_inter, (K - 1) * state_elems * 4)?,
        &download(gpu, fused_inter, (K - 1) * state_elems * 4)?,
    )
}

fn check_qkv(gpu: &dyn GpuBackend, stream: u64, rng: &mut Rng) -> Result<()> {
    let m = K;
    let n = 128usize;
    let k = 256usize;
    let input: Vec<u16> = (0..m * k).map(|_| bf16(rng.uni(-1.0, 1.0))).collect();
    let input_p = upload(gpu, &bytes_u16(&input))?;
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    for plane in 0..3usize {
        let packed: Vec<u8> = (0..n * k / 2)
            .map(|i| ((i * (17 + plane * 2) + plane * 13) & 0xff) as u8)
            .collect();
        let scale: Vec<u8> = (0..n * k / 16)
            .map(|i| [0x30, 0x38, 0x40][(i + plane) % 3])
            .collect();
        weights.push(upload(gpu, &packed)?);
        scales.push(upload(gpu, &scale)?);
    }
    let scale2 = [1.0f32, 0.5, 1.25];
    let separate = gpu.alloc(3 * m * n * 2)?;
    let fused_out = gpu.alloc(3 * m * n * 2)?;
    let base = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch5")?;
    let fused = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch5_qkv")?;
    for plane in 0..3usize {
        KernelLaunch::new(gpu, base)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input_p)
            .arg_ptr(weights[plane])
            .arg_ptr(scales[plane])
            .arg_f32(scale2[plane])
            .arg_ptr(separate.offset(plane * m * n * 2))
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
    }
    KernelLaunch::new(gpu, fused)
        .grid([div_ceil(n as u32, 4), 1, 3])
        .block([256, 1, 1])
        .arg_ptr(input_p)
        .arg_ptr(weights[0])
        .arg_ptr(scales[0])
        .arg_f32(scale2[0])
        .arg_ptr(weights[1])
        .arg_ptr(scales[1])
        .arg_f32(scale2[1])
        .arg_ptr(weights[2])
        .arg_ptr(scales[2])
        .arg_f32(scale2[2])
        .arg_ptr(fused_out)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    require_equal(
        "native-FP4 QKV output",
        &download(gpu, separate, 3 * m * n * 2)?,
        &download(gpu, fused_out, 3 * m * n * 2)?,
    )
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let mut rng = Rng(0x53_F1A5_A5);
    println!("=== GLM K=5 inline snapshot exactness ===");
    check_conv(gpu, stream, &mut rng)?;
    check_recurrent(gpu, stream, &mut rng)?;
    check_qkv(gpu, stream, &mut rng)?;
    println!("PASS: state kernels and native-FP4 QKV are bit-identical");
    Ok(())
}
