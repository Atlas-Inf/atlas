// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`) that replay single operations of DeepSeek's
//! reference through this engine's kernels, on the release's own weights:
//! each takes an operation's recorded input and must give its recorded output.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::ops::proj -- --ignored --nocapture --test-threads=1

use std::path::Path;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;
use crate::layers::deepseek_v41::cache_recipe as recipe;
use crate::layers::deepseek_v41::test_support::{
    Recording, backend, bf16s, download, le_bytes, upload,
};
use crate::layers::ops::rms_norm;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;
use crate::weight_map::{DenseWeight, fp8_block_scales_to_rows};

fn checkpoint() -> Result<String> {
    std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")
}

fn tiers(gpu: &dyn GpuBackend) -> Result<[KernelHandle; 3]> {
    Ok([
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
        gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
    ])
}

/// An FP8 projection of the release, with its 32 x 32 block scales repeated
/// into the per-row layout the MXFP8 tiers read. `(weight, n, k)`.
fn load_fp8(gpu: &dyn GpuBackend, prefix: &str) -> Result<(Mxfp8Weight, usize, usize)> {
    let ckpt = checkpoint()?;
    let (dtype, shape, values) =
        read_checkpoint_tensor(Path::new(&ckpt), &format!("{prefix}.weight"))?;
    let (_, _, grid) = read_checkpoint_tensor(Path::new(&ckpt), &format!("{prefix}.scale"))?;
    anyhow::ensure!(dtype == "F8_E4M3", "{prefix}: {dtype}");
    let (n, k) = (shape[0], shape[1]);
    let w = Mxfp8Weight {
        data: upload(gpu, &values)?,
        scales: upload(gpu, &fp8_block_scales_to_rows(&grid, n, k)?)?,
        tiled: false,
    };
    Ok((w, n, k))
}

/// Rows `picks` of a recorded `[.., rows, width]` tensor.
fn rows_of(all: &[u8], width: usize, picks: &[usize]) -> Vec<u8> {
    picks
        .iter()
        .flat_map(|&row| all[row * width * 2..(row + 1) * width * 2].iter().copied())
        .collect()
}

/// How close `got` is to the recorded `want`: the share of BF16 values equal,
/// and the relative L2 distance.
fn closeness(got: &[u16], want: &[u16]) -> (f64, f64) {
    let value = |bits: u16| f64::from(recipe::bf16_to_f32(bits));
    let same = got.iter().zip(want).filter(|(a, b)| a == b).count();
    let (mut diff, mut norm) = (0.0, 0.0);
    for (&a, &b) in got.iter().zip(want) {
        diff += (value(a) - value(b)).powi(2);
        norm += value(b).powi(2);
    }
    (same as f64 / got.len() as f64, (diff / norm).sqrt())
}

/// Prefill rows spread over the prompt.
fn picks(rec: &Recording, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| i * (rec.prefill - 1) / (count - 1))
        .collect()
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn fp8_projections_replay_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let (k, tiers, stream) = (Kernels::load(gpu)?, tiers(gpu)?, gpu.default_stream());
    for (layer, op) in [
        (0, "attn.wq_a"),
        (0, "attn.wq_b"),
        (0, "attn.wkv"),
        (0, "attn.wo_b"),
        (0, "ffn.shared_experts.w1"),
        (0, "ffn.shared_experts.w3"),
        (0, "ffn.shared_experts.w2"),
        (1, "engram.wkv"),
        (2, "attn.wq_b"),
        (2, "attn.wo_b"),
        (2, "attn.indexer.wq_b"),
    ] {
        let (w, n, width) = load_fp8(gpu, &format!("layers.{layer}.{op}"))?;
        let name = format!("L{layer}.op.{op}");
        let (_, input) = rec.prefill(&format!("{name}.in"))?;
        let (_, output) = rec.prefill(&format!("{name}.out"))?;
        // 40 rows: one launch of 32 and one of 8.
        let picks = picks(&rec, 40);
        let x = upload(gpu, &rows_of(&input, width, &picks))?;
        let quantized = gpu.alloc(picks.len() * width * 2)?;
        let y = gpu.alloc(picks.len() * n * 2)?;
        fp8_linear(
            gpu,
            &k,
            &tiers,
            &w,
            x,
            quantized,
            y,
            picks.len(),
            n,
            width,
            stream,
        )?;
        let got = download(gpu, y, picks.len() * n)?;
        let (same, distance) = closeness(&got, &bf16s(&rows_of(&output, n, &picks)));
        println!(
            "{name} [{n}, {width}]: {:.3}% of values equal, {distance:.2e} away",
            100.0 * same
        );
        assert!(
            same > 0.995 && distance < 2e-4,
            "{name}: {same} equal, {distance:e} away"
        );

        // One decode step, one row.
        let (_, input) = rec.step(&format!("{name}.in"), 7)?;
        let (_, output) = rec.step(&format!("{name}.out"), 7)?;
        gpu.copy_h2d(&input, x)?;
        fp8_linear(gpu, &k, &tiers, &w, x, quantized, y, 1, n, width, stream)?;
        let (same, distance) = closeness(&download(gpu, y, n)?, &bf16s(&output));
        assert!(
            same > 0.99 && distance < 4e-4,
            "{name} decode: {same} equal, {distance:e} away"
        );
        for p in [x, quantized, y, w.data, w.scales] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn norms_replay_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let ckpt = checkpoint()?;
    let gpu = backend()?;
    let kernel = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    for (layer, op) in [
        (0, "attn_norm"),
        (0, "attn.q_norm"),
        (0, "attn.kv_norm"),
        (0, "ffn_norm"),
        (2, "attn.compressor.norm"),
        (2, "attn.indexer.k_norm"),
    ] {
        let name = format!("L{layer}.op.{op}");
        let (_, shape, weight) =
            read_checkpoint_tensor(Path::new(&ckpt), &format!("layers.{layer}.{op}.weight"))?;
        let width = shape[0];
        let (in_shape, input) = rec.prefill(&format!("{name}.in"))?;
        let (_, output) = rec.prefill(&format!("{name}.out"))?;
        let rows = in_shape[1].min(64);
        let (x, y) = (
            upload(gpu, &input[..rows * width * 2])?,
            gpu.alloc(rows * width * 2)?,
        );
        let weight = DenseWeight {
            weight: upload(gpu, &weight)?,
        };
        rms_norm(
            gpu,
            kernel,
            x,
            &weight,
            y,
            rows as u32,
            width as u32,
            1e-20,
            gpu.default_stream(),
        )?;
        let got = download(gpu, y, rows * width)?;
        let (same, distance) = closeness(&got, &bf16s(&output[..rows * width * 2]));
        println!(
            "{name} [{width}]: {:.3}% of values equal, {distance:.2e} away",
            100.0 * same
        );
        assert!(
            same > 0.995 && distance < 2e-4,
            "{name}: {same} equal, {distance:e} away"
        );
        for p in [x, y, weight.weight] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

/// The two rotary ladders: window-only layers turn at the base theta with no
/// YaRN, compressing layers at the compressed theta with it.
fn freqs(layer: usize) -> Vec<f32> {
    if layer < 2 {
        recipe::rope_freqs(64, 0, 10_000.0, 16.0, 32.0, 1.0)
    } else {
        recipe::rope_freqs(64, 65536, 160_000.0, 16.0, 32.0, 1.0)
    }
}

fn positions(gpu: &dyn GpuBackend, picks: &[usize]) -> Result<DevicePtr> {
    let positions: Vec<u32> = picks.iter().map(|&p| p as u32).collect();
    upload(gpu, &le_bytes(&positions, u32::to_le_bytes))
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn queries_and_window_keys_replay_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let (k, stream) = (Kernels::load(gpu)?, gpu.default_stream());
    for layer in 0..3 {
        let l = format!("L{layer}.");
        let picks = picks(&rec, 48);
        let at = positions(gpu, &picks)?;
        let ladder = upload(gpu, &le_bytes(&freqs(layer), f32::to_le_bytes))?;

        // q: the projection's output with each head's last 64 values rotated.
        let (_, projected) = rec.prefill(&format!("{l}op.attn.wq_b.out"))?;
        let (_, want) = rec.prefill(&format!("{l}sa_q"))?;
        let q = upload(gpu, &rows_of(&projected, 64 * HEAD_DIM, &picks))?;
        rotate(
            gpu,
            k.rotate,
            q,
            at,
            ladder,
            picks.len(),
            64,
            HEAD_DIM,
            64,
            false,
            stream,
        )?;
        let got = download(gpu, q, picks.len() * 64 * HEAD_DIM)?;
        let (same, distance) = closeness(&got, &bf16s(&rows_of(&want, 64 * HEAD_DIM, &picks)));
        println!(
            "layer {layer} q: {:.4}% of values equal, {distance:.2e} away",
            100.0 * same
        );
        assert!(same > 0.9999, "layer {layer} q: {same} equal");

        // Window key: the normed projection, rotated, then FP8.
        let (_, normed) = rec.prefill(&format!("{l}op.attn.kv_norm.out"))?;
        let (_, want) = rec.prefill(&format!("{l}sa_kv"))?;
        let kv = upload(gpu, &rows_of(&normed, HEAD_DIM, &picks))?;
        rotate(
            gpu,
            k.rotate,
            kv,
            at,
            ladder,
            picks.len(),
            1,
            HEAD_DIM,
            64,
            false,
            stream,
        )?;
        fp8_quant_dequant(
            gpu,
            k.fp8_quant_dequant,
            kv,
            kv,
            picks.len() * HEAD_DIM,
            stream,
        )?;
        let got = download(gpu, kv, picks.len() * HEAD_DIM)?;
        let (same, distance) = closeness(&got, &bf16s(&rows_of(&want, HEAD_DIM, &picks)));
        println!(
            "layer {layer} window key: {:.4}% of values equal, {distance:.2e} away",
            100.0 * same
        );
        assert!(same > 0.9999, "layer {layer} window key: {same} equal");
        for p in [at, ladder, q, kv] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn the_output_projection_replays_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let (k, tiers, stream) = (Kernels::load(gpu)?, tiers(gpu)?, gpu.default_stream());
    const GROUPS: usize = 8;
    const RANK: usize = 1024;
    let group_in = 64 / GROUPS * HEAD_DIM;
    for layer in [0, 2] {
        let l = format!("L{layer}.");
        let (wo_a, n, width) = load_fp8(gpu, &format!("layers.{layer}.attn.wo_a"))?;
        anyhow::ensure!(
            (n, width) == (GROUPS * RANK, group_in),
            "wo_a [{n}, {width}]"
        );
        let picks = picks(&rec, 12);
        let at = positions(gpu, &picks)?;
        let ladder = upload(gpu, &le_bytes(&freqs(layer), f32::to_le_bytes))?;
        let (_, attended) = rec.prefill(&format!("{l}sa_o"))?;
        let (_, want) = rec.prefill(&format!("{l}op.attn.wo_b.in"))?;
        let o = upload(gpu, &rows_of(&attended, 64 * HEAD_DIM, &picks))?;
        let low = gpu.alloc(picks.len() * GROUPS * RANK * 2)?;

        // Undo the query's rotation, quantize to FP8, then each group of 8
        // heads through its own [1024, 4096] block of wo_a.
        rotate(
            gpu,
            k.rotate,
            o,
            at,
            ladder,
            picks.len(),
            64,
            HEAD_DIM,
            64,
            true,
            stream,
        )?;
        fp8_quant_dequant(
            gpu,
            k.fp8_quant_dequant,
            o,
            o,
            picks.len() * 64 * HEAD_DIM,
            stream,
        )?;
        for row in 0..picks.len() {
            for group in 0..GROUPS {
                let block = Mxfp8Weight {
                    data: wo_a.data.offset(group * RANK * group_in),
                    scales: wo_a.scales.offset(group * RANK * group_in / 32),
                    tiled: false,
                };
                crate::layers::ops::mxfp8_gemv_rows(
                    gpu,
                    &tiers,
                    &block,
                    o.offset((row * GROUPS + group) * group_in * 2),
                    low.offset((row * GROUPS + group) * RANK * 2),
                    1,
                    RANK,
                    group_in,
                    RANK,
                    stream,
                )?;
            }
        }
        let got = download(gpu, low, picks.len() * GROUPS * RANK)?;
        let (same, distance) = closeness(&got, &bf16s(&rows_of(&want, GROUPS * RANK, &picks)));
        println!(
            "layer {layer} wo_a: {:.3}% of values equal, {distance:.2e} away",
            100.0 * same
        );
        assert!(
            same > 0.995 && distance < 2e-4,
            "layer {layer} wo_a: {same} equal, {distance:e} away"
        );
        for p in [at, ladder, o, low, wo_a.data, wo_a.scales] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
