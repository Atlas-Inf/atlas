// SPDX-License-Identifier: AGPL-3.0-only

//! Ignored, CUDA-only check of the Qwen ViT head-72 flash-attention kernel
//! against an FP64 host reference: partial, exact and multi-tile sequences.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const HEADS: usize = 2;
const D: usize = 72;

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Deterministic values in roughly [-2, 2), so scores span a wide range.
fn values(n: usize, seed: u32) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bf16((state >> 8) as f32 / (1u32 << 22) as f32 - 2.0)
        })
        .collect()
}

fn upload(gpu: &dyn GpuBackend, v: &[u16]) -> Result<spark_runtime::gpu::DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

#[test]
#[ignore = "requires a CUDA host"]
fn vit_flash_attention_d72_matches_fp64_reference() -> Result<()> {
    let target = atlas_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .context("resolve the 27B CUDA kernel target")?;
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)
        .context("create CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let kernel = gpu.kernel("vision_encoder", "vit_flash_attention_d72")?;
    let stream = gpu.default_stream();
    let hidden = HEADS * D;
    for seq in [4usize, 40, 64, 300, 1028] {
        let q = values(HEADS * seq * D, 3 * seq as u32);
        let k = values(HEADS * seq * D, 3 * seq as u32 + 1);
        let qkv = values(seq * 3 * hidden, 3 * seq as u32 + 2);
        let (dq, dk, dqkv) = (upload(gpu, &q)?, upload(gpu, &k)?, upload(gpu, &qkv)?);
        let dout = gpu.alloc(seq * hidden * 2)?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(seq as u32, 64), HEADS as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(dq)
            .arg_ptr(dk)
            .arg_ptr(dqkv)
            .arg_ptr(dout)
            .arg_u32(seq as u32)
            .arg_u32(HEADS as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut out = vec![0u8; seq * hidden * 2];
        gpu.copy_d2h(dout, &mut out)?;
        for p in [dq, dk, dqkv, dout] {
            gpu.free(p)?;
        }
        let (mut worst, mut sum, mut n) = (0.0f64, 0.0f64, 0usize);
        for h in 0..HEADS {
            for i in 0..seq {
                let qi = |d: usize| f32_of(q[(h * seq + i) * D + d]) as f64;
                let scores: Vec<f64> = (0..seq)
                    .map(|j| {
                        (0..D)
                            .map(|d| qi(d) * f32_of(k[(h * seq + j) * D + d]) as f64)
                            .sum::<f64>()
                            / (D as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = p.iter().sum();
                for d in 0..D {
                    let want = (0..seq)
                        .map(|j| p[j] * f32_of(qkv[j * 3 * hidden + 2 * hidden + h * D + d]) as f64)
                        .sum::<f64>()
                        / total;
                    let at = (i * hidden + h * D + d) * 2;
                    let got = f32_of(u16::from_le_bytes([out[at], out[at + 1]])) as f64;
                    worst = worst.max((got - want).abs());
                    sum += (got - want).abs();
                    n += 1;
                }
            }
        }
        eprintln!(
            "seq {seq}: max |flash - fp64| = {worst:.5}, mean {:.6}",
            sum / n as f64
        );
        // BF16 output and BF16 probabilities: a few BF16 ulps at |v| <= 2.
        ensure!(worst < 0.01, "seq {seq}: max |flash - fp64| = {worst}");
    }
    Ok(())
}
