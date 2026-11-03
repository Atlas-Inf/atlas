// SPDX-License-Identifier: AGPL-3.0-only

//! GB10 checks of the native GLM tower at the shipped geometry, on random
//! BF16 weights: chunking the token-local stages leaves every output bit
//! unchanged, and the full reference sequences encode.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{
    GlmCapacity, GlmVisionBlockWeights, GlmVisionEncoder, GlmVisionMergerWeights, GlmVisionWeights,
};
use crate::VisionItem;

const PATCH_DIM: usize = 3 * 2 * 14 * 14;

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Deterministic values in `[-scale, scale] + offset` (xorshift).
fn values(n: usize, seed: u32, scale: f32, offset: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            offset + scale * ((state as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

struct Uploader<'a> {
    gpu: &'a dyn GpuBackend,
    seed: u32,
}

impl Uploader<'_> {
    fn tensor(&mut self, n: usize, scale: f32, offset: f32) -> Result<DevicePtr> {
        self.seed += 1;
        let bytes: Vec<u8> = values(n, self.seed, scale, offset)
            .into_iter()
            .flat_map(|v| bf16_bits(v).to_le_bytes())
            .collect();
        let ptr = self.gpu.alloc(bytes.len())?;
        self.gpu.copy_h2d(&bytes, ptr)?;
        Ok(ptr)
    }
}

fn shipped_config(depth: usize) -> atlas_core::config::VisionConfig {
    let mut config = super::test_tower::config();
    (config.depth, config.hidden_size, config.num_heads) = (depth, 1024, 16);
    (config.intermediate_size, config.out_hidden_size) = (4096, 4096);
    config.projection_intermediate_size = 10240;
    config.max_pixels = None;
    config
}

fn tower(gpu: &dyn GpuBackend, depth: usize) -> Result<GlmVisionEncoder> {
    let config = shipped_config(depth);
    let (h, i, o, p) = (1024, 4096, 4096, 10240);
    let mut up = Uploader { gpu, seed: 7 };
    let w = 1.0 / (h as f32).sqrt();
    let mut blocks = Vec::with_capacity(depth);
    for _ in 0..depth {
        blocks.push(GlmVisionBlockWeights {
            norm1_w: up.tensor(h, 0.1, 1.0)?,
            qkv_w: up.tensor(3 * h * h, w, 0.0)?,
            qkv_b: up.tensor(3 * h, 0.02, 0.0)?,
            q_norm_w: up.tensor(h, 0.1, 1.0)?,
            k_norm_w: up.tensor(h, 0.1, 1.0)?,
            proj_w: up.tensor(h * h, w, 0.0)?,
            proj_b: up.tensor(h, 0.02, 0.0)?,
            norm2_w: up.tensor(h, 0.1, 1.0)?,
            gate_up_w: up.tensor(2 * i * h, w, 0.0)?,
            gate_up_b: up.tensor(2 * i, 0.02, 0.0)?,
            down_w: up.tensor(h * i, 1.0 / (i as f32).sqrt(), 0.0)?,
            down_b: up.tensor(h, 0.02, 0.0)?,
        });
    }
    let weights = GlmVisionWeights {
        patch_embed_w: up.tensor(h * PATCH_DIM, 1.0 / (PATCH_DIM as f32).sqrt(), 0.0)?,
        patch_embed_b: up.tensor(h, 0.02, 0.0)?,
        blocks,
        post_layernorm_w: up.tensor(h, 0.1, 1.0)?,
        downsample_w: up.tensor(o * 4 * h, 1.0 / ((4 * h) as f32).sqrt(), 0.0)?,
        downsample_b: up.tensor(o, 0.02, 0.0)?,
        merger: GlmVisionMergerWeights {
            proj_w: up.tensor(o * o, 1.0 / (o as f32).sqrt(), 0.0)?,
            post_norm_w: up.tensor(o, 0.1, 1.0)?,
            post_norm_b: up.tensor(o, 0.02, 0.0)?,
            gate_up_w: up.tensor(2 * p * o, 1.0 / (o as f32).sqrt(), 0.0)?,
            down_w: up.tensor(o * p, 1.0 / (p as f32).sqrt(), 0.0)?,
        },
    };
    GlmVisionEncoder::new(Some(weights), &config, GlmCapacity::budgets(&config), gpu)
}

fn backend() -> Result<spark_runtime::cuda_backend::AtlasCudaBackend> {
    let target = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("resolve exact GLM CUDA kernel target")?;
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)
        .context("create CUDA backend")
}

/// Encode `items` and return the packed merged rows' bytes and the seconds.
fn encode(
    enc: &GlmVisionEncoder,
    gpu: &dyn GpuBackend,
    items: &[&VisionItem],
) -> Result<(Vec<u8>, f64)> {
    let stream = gpu.default_stream();
    let started = std::time::Instant::now();
    let geometry = enc.forward_items(items, gpu, stream)?;
    gpu.synchronize(stream)?;
    let seconds = started.elapsed().as_secs_f64();
    let rows: usize = geometry.iter().map(|g| g.2).sum();
    let mut out = vec![0u8; rows * enc.out_hidden_size * 2];
    gpu.copy_d2h(enc.buf_out, &mut out)?;
    Ok((out, seconds))
}

fn all_finite(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16).is_finite())
}

fn image(grid_h: usize, grid_w: usize, seed: u32) -> VisionItem {
    VisionItem::image(
        values(grid_h * grid_w * PATCH_DIM, seed, 2.0, 0.0),
        grid_h,
        grid_w,
    )
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn chunked_token_local_stages_are_bit_exact() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = &gpu;
    let mut enc = tower(gpu, 2)?;
    // 72x100 patches: two full 4096-row chunks and a partial one; 1800
    // merged rows across two merger chunks.
    let item = image(72, 100, 11);
    let (reference, _) = encode(&enc, gpu, &[&item])?;
    ensure!(all_finite(&reference), "non-finite tower output");
    for chunk in [1000, 512, 64] {
        enc.row_chunk = chunk;
        let (got, _) = encode(&enc, gpu, &[&item])?;
        ensure!(got == reference, "row chunk {chunk} changed the output");
    }
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn the_largest_reference_sequences_encode() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = &gpu;
    let enc = tower(gpu, 24)?;
    // An 8000-token still (2240x2800 px) and one two-frame 30000-token video
    // group (4200x4480 px): the processor's largest sequences.
    for (name, grid_h, grid_w) in [
        ("8000-token still", 160, 200),
        ("30000-token group", 300, 400),
    ] {
        let item = image(grid_h, grid_w, grid_h as u32);
        let (out, seconds) = encode(&enc, gpu, &[&item])?;
        ensure!(all_finite(&out), "{name}: non-finite output");
        eprintln!(
            "{name}: {} patches, {} rows in {seconds:.2} s",
            grid_h * grid_w,
            grid_h * grid_w / 4
        );
    }
    Ok(())
}
