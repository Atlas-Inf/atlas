// SPDX-License-Identifier: AGPL-3.0-only

//! Ignored, CUDA-only check of the GLM visual flash-attention kernel against
//! an FP32 host reference on sequences spanning several key tiles.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const HEADS: usize = 3;
const HEAD_DIM: usize = 64;

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

fn reference(qkv: &[u16], seq: usize) -> Vec<f32> {
    let hidden = HEADS * HEAD_DIM;
    let at = |row: usize, part: usize, head: usize, d: usize| {
        f32_of(qkv[row * 3 * hidden + part * hidden + head * HEAD_DIM + d])
    };
    let mut out = vec![0.0f32; seq * hidden];
    for head in 0..HEADS {
        for q in 0..seq {
            let scores: Vec<f32> = (0..seq)
                .map(|k| {
                    (0..HEAD_DIM)
                        .map(|d| at(q, 0, head, d) * at(k, 1, head, d))
                        .sum::<f32>()
                        / (HEAD_DIM as f32).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = probs.iter().sum();
            for d in 0..HEAD_DIM {
                out[q * hidden + head * HEAD_DIM + d] =
                    (0..seq).map(|k| probs[k] * at(k, 2, head, d)).sum::<f32>() / total;
            }
        }
    }
    out
}

#[test]
#[ignore = "requires a CUDA host"]
fn glm_vision_flash_attention_matches_fp32_reference() -> Result<()> {
    let target = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("resolve exact GLM CUDA kernel target")?;
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)
        .context("create CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let kernel = gpu.kernel("glm_vision_encoder", "glm_vision_flash_attention")?;
    let stream = gpu.default_stream();
    let hidden = HEADS * HEAD_DIM;
    // One partial tile, exact tiles, and several tiles with a partial tail.
    for seq in [40usize, 128, 300] {
        let qkv = values(seq * 3 * hidden, seq as u32);
        let qkv_bytes: Vec<u8> = qkv.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_qkv = gpu.alloc(qkv_bytes.len())?;
        let d_out = gpu.alloc(seq * hidden * 2)?;
        gpu.copy_h2d(&qkv_bytes, d_qkv)?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(seq as u32, 64), HEADS as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(d_qkv)
            .arg_ptr(d_out)
            .arg_u32(seq as u32)
            .arg_u32(HEADS as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut out_bytes = vec![0u8; seq * hidden * 2];
        gpu.copy_d2h(d_out, &mut out_bytes)?;
        gpu.free(d_qkv)?;
        gpu.free(d_out)?;
        let expected = reference(&qkv, seq);
        let mut worst = 0.0f32;
        for (i, pair) in out_bytes.chunks_exact(2).enumerate() {
            let got = f32_of(u16::from_le_bytes([pair[0], pair[1]]));
            worst = worst.max((got - expected[i]).abs());
        }
        // BF16 output and BF16 probabilities: a few BF16 ulps at |v| <= 2.
        ensure!(worst < 0.01, "seq {seq}: max |flash - reference| = {worst}");
    }
    Ok(())
}
