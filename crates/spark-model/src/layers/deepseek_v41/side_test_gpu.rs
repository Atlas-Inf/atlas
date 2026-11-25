// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`): the compressor of layer 2 and the Engram of layer
//! 1, replayed from DeepSeek's recorded inputs on the release's weights.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::side -- --ignored --nocapture --test-threads=1

use std::path::Path;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;
use crate::layers::deepseek_v41::parts::PARTS;
use crate::layers::deepseek_v41::test_support::{
    Recording, backend, bf16s, checkpoint, closeness, download, le_bytes, picks, rows_of, upload,
};
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

const HIDDEN: usize = 5120;
const KEY: usize = 512;

fn tensor(name: &str) -> Result<Vec<u8>> {
    Ok(read_checkpoint_tensor(Path::new(&checkpoint()?), name)?.2)
}

fn f32s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    gpu.synchronize(gpu.default_stream())?;
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn the_compressor_replays_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let (k, stream) = (SideKernels::load(gpu)?, gpu.default_stream());
    let wkv = upload(gpu, &tensor("layers.2.attn.compressor.wkv.weight")?)?;
    let wgate = upload(gpu, &tensor("layers.2.attn.compressor.wgate.weight")?)?;
    let (_, inputs) = rec.prefill("L2.attn_in")?;
    let (_, keys) = rec.prefill("L2.op.attn.compressor.wkv.out")?;
    let (_, gates) = rec.prefill("L2.op.attn.compressor.wgate.out")?;
    let (_, pooled) = rec.prefill("L2.op.attn.compressor.norm.in")?;
    let (kv, score, out) = (
        gpu.alloc(2 * KEY * 4)?,
        gpu.alloc(2 * KEY * 4)?,
        gpu.alloc(KEY * 2)?,
    );
    let (mut diff, mut norm, mut same, mut seen) = (0.0f64, 0.0f64, 0usize, 0usize);
    // Groups of two tokens, spread over the prompt.
    for group in picks(&rec, 80)
        .into_iter()
        .map(|p| p / 2)
        .filter(|&g| 2 * g + 1 < rec.prefill)
    {
        for token in 0..2 {
            let x = upload(gpu, &rows_of(&inputs, HIDDEN, &[2 * group + token]))?;
            gemv_f32(
                gpu,
                k.gemv_f32,
                x,
                wkv,
                kv.offset(token * KEY * 4),
                KEY,
                HIDDEN,
                stream,
            )?;
            gemv_f32(
                gpu,
                k.gemv_f32,
                x,
                wgate,
                score.offset(token * KEY * 4),
                KEY,
                HIDDEN,
                stream,
            )?;
            gpu.synchronize(stream)?;
            gpu.free(x)?;
        }
        for (dev, recorded) in [(kv, &keys), (score, &gates)] {
            let got = f32s(gpu, dev, 2 * KEY)?;
            let want = &recorded[2 * group * KEY * 4..(2 * group + 2) * KEY * 4];
            for (g, w) in got.iter().zip(want.chunks_exact(4)) {
                let w = f64::from(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                diff += (f64::from(*g) - w).powi(2);
                norm += w * w;
            }
        }
        compress_pool(gpu, k.compress_pool, kv, score, out, 2, KEY, stream)?;
        let got = download(gpu, out, KEY)?;
        let want = bf16s(&rows_of(&pooled, KEY, &[group]));
        same += got.iter().zip(&want).filter(|(a, b)| a == b).count();
        seen += KEY;
    }
    let apart = (diff / norm).sqrt();
    println!(
        "compressor: projections {apart:.1e} away; {:.3}% of pooled values equal",
        100.0 * same as f64 / seen as f64
    );
    assert!(apart < 1e-5, "compressor projections {apart:e} away");
    assert!(
        same * 1000 >= seen * 995,
        "{same} of {seen} pooled values equal"
    );
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn engram_replays_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let k = SideKernels::load(gpu)?;
    let picks = picks(&rec, 64);
    // q_weight * k_weight, as FP32.
    let (q, key) = (
        bf16s(&tensor("layers.1.engram.q_weight")?),
        bf16s(&tensor("layers.1.engram.k_weight")?),
    );
    let product: Vec<f32> = q
        .iter()
        .zip(&key)
        .map(|(&a, &b)| bf16_to_f32(a) * bf16_to_f32(b))
        .collect();
    anyhow::ensure!(product.len() == 4 * HIDDEN, "engram weights");
    let weight = upload(gpu, &le_bytes(&product, f32::to_le_bytes))?;
    let stream_buf = upload(
        gpu,
        &rows_of(&rec.prefill("L1.engram_in")?.1, 4 * HIDDEN, &picks),
    )?;
    // The projection as the ranks gather it: four parts, a part every
    // token's quarter of the values.
    let rows = rows_of(&rec.prefill("L1.op.engram.wkv.out")?.1, 5 * HIDDEN, &picks);
    let (part, width) = (5 * HIDDEN / PARTS * 2, 5 * HIDDEN * 2);
    let in_parts: Vec<u8> = (0..PARTS)
        .flat_map(|at| {
            let rows = &rows;
            (0..picks.len()).flat_map(move |row| {
                rows[row * width + at * part..row * width + (at + 1) * part]
                    .iter()
                    .copied()
            })
        })
        .collect();
    let kv = upload(gpu, &in_parts)?;
    engram_gate(
        gpu,
        k.engram_gate,
        stream_buf,
        kv,
        weight,
        picks.len(),
        HIDDEN,
        4,
        1e-20,
        gpu.default_stream(),
    )?;
    let got = download(gpu, stream_buf, picks.len() * 4 * HIDDEN)?;
    let want = bf16s(&rows_of(
        &rec.prefill("L1.engram_out")?.1,
        4 * HIDDEN,
        &picks,
    ));
    let (same, distance) = closeness(&got, &want);
    println!(
        "engram: {:.4}% of values equal, {distance:.2e} away",
        100.0 * same
    );
    // The lookup must have changed the stream for the check to mean anything.
    let before = bf16s(&rows_of(
        &rec.prefill("L1.engram_in")?.1,
        4 * HIDDEN,
        &picks,
    ));
    let moved = before.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(
        moved * 10 > want.len(),
        "Engram changed only {moved} of {} values",
        want.len()
    );
    assert!(same > 0.999, "engram: {same} equal");
    Ok(())
}
