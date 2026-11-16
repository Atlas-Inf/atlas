// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the Engram layers from recorded hash ids to the
//! stream they leave, on the release's table and weights.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::engram -- --ignored --nocapture --test-threads=1

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, bf16s, checkpoint, closeness, download, picks, rows_of, upload,
};
use crate::weight_map::test_checkpoint::{CheckpointRows, read_checkpoint_tensor};

/// An Engram layer's weights apart from its table, in a store.
fn store(gpu: &dyn GpuBackend, dir: &Path, layer: usize) -> Result<WeightStore> {
    let mut map = HashMap::new();
    for name in ["wkv.weight", "wkv.scale", "q_weight", "k_weight"] {
        let name = format!("layers.{layer}.engram.{name}");
        let (dtype, shape, bytes) = read_checkpoint_tensor(dir, &name)?;
        let tensor = WeightTensor {
            ptr: upload(gpu, &bytes)?,
            shape,
            dtype: WeightDtype::from_safetensors_str(&dtype)?,
        };
        map.insert(name, tensor);
    }
    Ok(WeightStore::from_map(map))
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn engram_layers_replay_the_reference_from_hash_ids() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let dir = checkpoint()?;
    let dir = Path::new(&dir);
    let (copies, hidden) = (SHAPE.hc.copies, SHAPE.hidden);
    let picks = picks(&rec, 64);
    for layer in rec.sample(&[1], &[14]) {
        let kernels = BlockKernels::load(gpu)?;
        let weights = EngramWeights::load(
            &store(gpu, dir, layer)?,
            gpu,
            layer,
            &SHAPE,
            Held::ALL,
            stream,
        )?;
        let engram = Engram::new(gpu, kernels, &SHAPE, Held::ALL, weights)?;
        let scratch = GemmScratch::alloc(gpu, 1, engram.weights.columns * ENGRAM_DIM)?;
        let columns = engram.weights.columns;
        let table = format!("layers.{layer}.engram.embed");
        let mut rows = CheckpointRows::open(dir, &format!("{table}.weight"))?;
        let mut scales = CheckpointRows::open(dir, &format!("{table}.scale"))?;
        anyhow::ensure!(
            rows.row_bytes == ENGRAM_DIM && scales.row_bytes == ENGRAM_DIM / ENGRAM_SCALE_GROUP,
            "{table}: rows of {} values, {} scales",
            rows.row_bytes,
            scales.row_bytes
        );

        let l = format!("L{layer}.");
        let (shape, ids) = rec.prefill(&format!("{l}engram_hash_ids"))?;
        anyhow::ensure!(shape[2] == columns, "{} hash ids a token", shape[2]);
        let recorded = |name: &str, width: usize| -> Result<Vec<u8>> {
            Ok(rows_of(
                &rec.prefill(&format!("{l}{name}"))?.1,
                width,
                &picks,
            ))
        };
        let read = columns * ENGRAM_DIM;
        let kv = (copies + 1) * hidden;
        let inputs = recorded("engram_in", copies * hidden)?;
        let want_values = bf16s(&recorded("op.engram.wkv.in", read)?);
        let want_kv = bf16s(&recorded("op.engram.wkv.out", kv)?);
        let want = bf16s(&recorded("engram_out", copies * hidden)?);

        let streams = gpu.alloc(copies * hidden * 2)?;
        let (row_dev, scale_dev) = (
            gpu.alloc(columns * rows.row_bytes)?,
            gpu.alloc(columns * scales.row_bytes)?,
        );
        // The rows are read in the ids' order, so column c is row c.
        let in_order: Vec<u8> = (0..columns as u32).flat_map(u32::to_le_bytes).collect();
        let slots = upload(gpu, &in_order)?;
        let (mut values, mut projected, mut got) = (Vec::new(), Vec::new(), Vec::new());
        for (at, &position) in picks.iter().enumerate() {
            let (mut row_bytes, mut scale_bytes) = (Vec::new(), Vec::new());
            for id in ids[position * columns * 8..(position + 1) * columns * 8].chunks_exact(8) {
                let id = i64::from_le_bytes(id.try_into().expect("8 bytes")) as usize;
                rows.read(id, &mut row_bytes)?;
                scales.read(id, &mut scale_bytes)?;
            }
            gpu.synchronize(stream)?;
            gpu.copy_h2d(&row_bytes, row_dev)?;
            gpu.copy_h2d(&scale_bytes, scale_dev)?;
            gpu.copy_h2d(
                &inputs[at * copies * hidden * 2..(at + 1) * copies * hidden * 2],
                streams,
            )?;
            engram.apply(
                gpu, &scratch, row_dev, scale_dev, slots, slots, streams, 1, None, stream,
            )?;
            values.extend(download(gpu, engram.values, read)?);
            projected.extend(download(gpu, engram.kv, kv)?);
            got.extend(download(gpu, streams, copies * hidden)?);
        }
        let (kv_same, kv_distance) = closeness(&projected, &want_kv);
        let (same, distance) = closeness(&got, &want);
        println!(
            "layer {layer} Engram, {} tokens x {columns} rows: rows as values {}; projection {:.4}% of values equal, {kv_distance:.2e} away; stream {:.4}% equal, {distance:.2e} away",
            picks.len(),
            if values == want_values {
                "bit-identical"
            } else {
                "differ"
            },
            100.0 * kv_same,
            100.0 * same
        );
        assert!(
            values == want_values,
            "layer {layer}: looked-up values differ"
        );
        assert!(
            kv_same > 0.999 && same > 0.999,
            "layer {layer}: {kv_same} and {same} equal"
        );
        // The lookup must have changed the stream for the check to mean anything.
        let moved = bf16s(&inputs)
            .iter()
            .zip(&want)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            moved * 10 > want.len(),
            "layer {layer}: Engram changed only {moved} values"
        );
    }
    Ok(())
}
