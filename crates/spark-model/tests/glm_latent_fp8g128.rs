// SPDX-License-Identifier: AGPL-3.0-only
//! `fp8_g128` GLM latent cache: writer layout and precision oracle.
//! No checkpoint required. Run only on an explicitly available CUDA GPU.
#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const BLOCK: usize = 16;
const BLOCK_BYTES: usize = BLOCK * 528;

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits(((value.to_bits() + 0x7FFF + ((value.to_bits() >> 16) & 1)) >> 16) << 16)
}

fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((code >> 3) & 0xF) as i32;
    let man = (code & 7) as f32;
    sign * if exp == 0 {
        man / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

#[test]
#[ignore = "requires an available GB10 GPU and compiled kernels; no model weights"]
fn writer_packs_values_then_scales_per_block_within_e4m3_rounding() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let (tokens, stride, blocks) = (37usize, 640usize, 4usize);
    // Deterministic spread over ~5 decades, one all-zero group, one outlier.
    let value = |t: usize, d: usize| -> f32 {
        if t == 5 && d < 128 {
            return 0.0;
        }
        let x = ((t * 131 + d * 17) % 997) as f32 / 997.0 - 0.5;
        let mag = 10f32.powi(((t + d / 128) % 5) as i32 - 3);
        if t == 9 && d == 300 { 900.0 } else { x * mag }
    };
    let mut src = vec![0u8; tokens * stride * 2];
    let mut rows = vec![[0f32; 512]; tokens];
    for t in 0..tokens {
        for d in 0..512 {
            let v = bf16_round(value(t, d));
            rows[t][d] = v;
            src[(t * stride + d) * 2..][..2]
                .copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    // Reverse physical order catches accidental logical addressing.
    let slot = |t: usize| ((blocks - 1 - t / BLOCK) * BLOCK + t % BLOCK) as i64;
    let slots: Vec<u8> = (0..tokens).flat_map(|t| slot(t).to_le_bytes()).collect();
    let key = upload(&gpu, &src)?;
    let slot_ptr = upload(&gpu, &slots)?;
    let cache = gpu.alloc(blocks * BLOCK_BYTES)?;
    gpu.memset(cache, 0, blocks * BLOCK_BYTES)?;
    ops::glm_latent_cache_write_fp8g128(
        &gpu,
        gpu.kernel("reshape_and_cache", "glm_latent_cache_write_fp8g128")?,
        key,
        cache,
        slot_ptr,
        tokens as u32,
        BLOCK as u32,
        stride as u32,
        stream,
    )?;
    let mut out = vec![0u8; blocks * BLOCK_BYTES];
    gpu.copy_d2h_on_stream(cache, &mut out, stream)?;
    for t in 0..tokens {
        let s = slot(t) as usize;
        let base = s / BLOCK * BLOCK_BYTES;
        let values = &out[base + s % BLOCK * 512..][..512];
        let scales = &out[base + BLOCK * 512 + s % BLOCK * 16..][..16];
        for g in 0..4 {
            let group = &rows[t][g * 128..(g + 1) * 128];
            let amax = group.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0e-4);
            let scale = f32::from_le_bytes(scales[g * 4..g * 4 + 4].try_into().unwrap());
            ensure!(
                scale == amax * (1.0 / 448.0) as f32,
                "token {t} group {g}: scale {scale} vs amax {amax}"
            );
            for (i, &v) in group.iter().enumerate() {
                let got = e4m3(values[g * 128 + i]) * scale;
                // E4M3 keeps 3 mantissa bits; subnormals keep scale * 2^-9.
                let tol = (v.abs() / 16.0).max(scale * 2f32.powi(-10)) * 1.0001;
                ensure!(
                    (got - v).abs() <= tol,
                    "token {t} dim {}: {got} vs {v}",
                    g * 128 + i
                );
            }
        }
    }
    for p in [key, slot_ptr, cache] {
        gpu.free(p)?;
    }
    Ok(())
}

fn kv_pad(
    gpu: &dyn GpuBackend,
    (symbol, shared_mem): (&str, u32),
    [q, cache, indices, out, table]: [DevicePtr; 5],
    rows: u32,
) -> Result<()> {
    spark_runtime::kernel_args::KernelLaunch::new(
        gpu,
        gpu.kernel("glm_sparse_prefill_kv_reuse", symbol)?,
    )
    .grid([1, rows, 1])
    .block([256, 1, 1])
    .shared_mem(shared_mem)
    .arg_ptr(q)
    .arg_ptr(cache)
    .arg_ptr(cache)
    .arg_ptr(indices)
    .arg_ptr(out)
    .arg_ptr(table)
    .arg_u32(rows)
    .arg_u32(32)
    .arg_u32(512)
    .arg_u32(2051)
    .arg_u32(BLOCK as u32)
    .arg_f32(0.0625)
    .launch(gpu.default_stream())
}

#[test]
#[ignore = "requires an available GB10 GPU and compiled kernels; no model weights"]
fn fp8_kv_pad_and_pipe_are_bit_identical_to_bf16_kv_pad_on_the_dequantized_view() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    // Rows 0-2 select a full 2048; rows 3-6 are early-context rows at
    // positions 0/5/37/1001, mostly -1 with whole invalid tiles.
    let early = [0usize, 5, 37, 1001];
    let (tokens, blocks, rows) = (2600usize, 163usize, 3 + early.len());
    let bf16 = |v: f32| ((bf16_round(v).to_bits() >> 16) as u16).to_le_bytes();
    let hash = |a: usize, b: usize| ((a * 2654435761 + b * 40503) % 10007) as f32 / 10007.0 - 0.5;
    let latents: Vec<u8> = (0..tokens)
        .flat_map(|t| {
            (0..512).flat_map(move |d| bf16(hash(t, d) * (1.0 + (d % 128) as f32 / 16.0)))
        })
        .collect();
    // Scrambled physical placement: logical block b lives at (b * 37) % blocks.
    let table: Vec<u8> = (0..blocks as u32)
        .flat_map(|b| ((b * 37) % blocks as u32).to_le_bytes())
        .collect();
    let slots: Vec<u8> = (0..tokens)
        .flat_map(|t| (((t / BLOCK * 37) % blocks * BLOCK + t % BLOCK) as i64).to_le_bytes())
        .collect();
    let queries: Vec<u8> = (0..rows * 32 * 512)
        .flat_map(|i| bf16(hash(i, 7) * 0.25))
        .collect();
    // Full rows select 2048 distinct tokens (stride-coprime walk) and three -1
    // pads; an early row at p selects its whole pools, -1 pads, then the tail.
    let select = |r: usize, j: usize| -> i32 {
        if r < 3 {
            return if j < 2048 {
                ((j * 1031 + r * 17) % tokens) as i32
            } else {
                -1
            };
        }
        let p = early[r - 3];
        let pooled = (p + 1) / 4 * 4;
        match j {
            j if j < pooled => j as i32,
            j if j >= 2048 && pooled + j - 2048 <= p => (pooled + j - 2048) as i32,
            _ => -1,
        }
    };
    let indices: Vec<u8> = (0..rows)
        .flat_map(|r| (0..2051).flat_map(move |j| select(r, j).to_le_bytes()))
        .collect();
    let key = upload(&gpu, &latents)?;
    let slot = upload(&gpu, &slots)?;
    let table = upload(&gpu, &table)?;
    let q = upload(&gpu, &queries)?;
    let ids = upload(&gpu, &indices)?;
    let cache = gpu.alloc(blocks * BLOCK_BYTES)?;
    let view = gpu.alloc(tokens * 1024)?;
    let identity = upload(
        &gpu,
        &(0..(tokens / BLOCK) as u32 + 1)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let out_bf16 = gpu.alloc(rows * 32 * 1024)?;
    let out_fp8 = gpu.alloc(rows * 32 * 1024)?;
    let out_pipe = gpu.alloc(rows * 32 * 1024)?;
    ops::glm_latent_cache_write_fp8g128(
        &gpu,
        gpu.kernel("reshape_and_cache", "glm_latent_cache_write_fp8g128")?,
        key,
        cache,
        slot,
        tokens as u32,
        BLOCK as u32,
        512,
        stream,
    )?;
    ops::glm_latent_dequant_fp8g128(
        &gpu,
        gpu.kernel("reshape_and_cache", "glm_latent_dequant_fp8g128")?,
        cache,
        table,
        view,
        tokens as u32,
        BLOCK as u32,
        stream,
    )?;
    kv_pad(
        &gpu,
        ("glm_sparse_mla_prefill_bf16_head32_tc_kv_pad", 69376),
        [q, view, ids, out_bf16, identity],
        rows as u32,
    )?;
    kv_pad(
        &gpu,
        ("glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad", 69376),
        [q, cache, ids, out_fp8, table],
        rows as u32,
    )?;
    kv_pad(
        &gpu,
        ("glm_sparse_mla_prefill_fp8g128_head32_tc_pipe", 80128),
        [q, cache, ids, out_pipe, table],
        rows as u32,
    )?;
    let (mut a, mut b) = (vec![0u8; rows * 32 * 1024], vec![0u8; rows * 32 * 1024]);
    gpu.copy_d2h_on_stream(out_bf16, &mut a, stream)?;
    gpu.copy_d2h_on_stream(out_fp8, &mut b, stream)?;
    let mut c = vec![0u8; rows * 32 * 1024];
    gpu.copy_d2h_on_stream(out_pipe, &mut c, stream)?;
    ensure!(b == c, "fp8 pipe differs from fp8 kv_pad");
    ensure!(a.iter().any(|&x| x != 0), "attention output is all zero");
    ensure!(
        a == b,
        "fp8 kv_pad differs from bf16 kv_pad on the dequantized view"
    );
    // The view is the probe's arithmetic: every latent within E4M3 rounding.
    let mut v = vec![0u8; tokens * 1024];
    gpu.copy_d2h_on_stream(view, &mut v, stream)?;
    for (i, (got, want)) in v.chunks_exact(2).zip(latents.chunks_exact(2)).enumerate() {
        let f = |b: &[u8]| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
        let (g, w) = (f(got), f(want));
        ensure!(
            (g - w).abs() <= w.abs() / 15.0 + 1e-3,
            "view element {i}: {g} vs {w}"
        );
    }
    for p in [
        key, slot, table, q, ids, cache, view, identity, out_bf16, out_fp8, out_pipe,
    ] {
        gpu.free(p)?;
    }
    Ok(())
}
