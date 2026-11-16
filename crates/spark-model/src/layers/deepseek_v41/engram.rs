// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1's Engram, in plain form: before two of the blocks, a token's
//! n-gram hash ids fetch rows of a table, a projection turns the rows into a
//! key per copy of the stream and one value, and the value is added into
//! each copy through a gate that compares the copy with its key (`Engram` in
//! the release's `inference/model.py`).
//!
//! The table is 384 million rows of 256 FP8 values, 101 GB a layer. This
//! takes the rows a token reads, as the table stores them; finding them is
//! the caller's.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cache_recipe::bf16_to_f32;
use super::ops;
use super::shape::{BlockKernels, Shape};
use super::side;
use super::weights::{Fp8Proj, fp8, tensor};

/// Values in one table row.
pub const ENGRAM_DIM: usize = 256;
/// Values of a row under one scale.
pub const ENGRAM_SCALE_GROUP: usize = 32;

/// An Engram layer's weights, apart from its table.
pub struct EngramWeights {
    /// `[(copies + 1) * hidden, columns * 256]`: a key per copy, then the
    /// value.
    pub wkv: Fp8Proj,
    /// `[copies, hidden]` FP32: `q_weight * k_weight`, which the reference
    /// only ever uses as that product.
    pub weight: DevicePtr,
    /// Rows a token reads: one per n-gram size and hash head.
    pub columns: usize,
}

impl EngramWeights {
    pub fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        layer: usize,
        shape: &Shape,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.engram");
        let wkv = fp8(store, gpu, &format!("{prefix}.wkv"))?;
        let gate = shape.hc.copies * shape.hidden;
        ensure!(
            wkv.n == gate + shape.hidden && wkv.k.is_multiple_of(ENGRAM_DIM),
            "{prefix}.wkv [{}, {}]",
            wkv.n,
            wkv.k
        );
        let part = |name: &str| -> Result<Vec<f32>> {
            let name = format!("{prefix}.{name}");
            let p = tensor(store, &name, WeightDtype::BF16)?;
            ensure!(
                store.get(&name)?.num_elements() == gate,
                "{name}: not [copies, hidden]"
            );
            let mut bytes = vec![0u8; gate * 2];
            gpu.copy_d2h(p, &mut bytes)?;
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        };
        let product: Vec<u8> = part("q_weight")?
            .iter()
            .zip(part("k_weight")?)
            .flat_map(|(q, k)| (q * k).to_le_bytes())
            .collect();
        let weight = gpu.alloc(product.len())?;
        gpu.copy_h2d(&product, weight)?;
        Ok(Self {
            columns: wkv.k / ENGRAM_DIM,
            wkv,
            weight,
        })
    }
}

/// One Engram layer on one token at a time.
pub struct Engram {
    pub weights: EngramWeights,
    kernels: BlockKernels,
    shape: Shape,
    /// `[columns * 256]` BF16: the rows as values.
    pub(super) values: DevicePtr,
    quantized: DevicePtr,
    /// `[(copies + 1) * hidden]` BF16: the keys, then the value.
    pub(super) kv: DevicePtr,
}

impl Engram {
    pub fn new(
        gpu: &dyn GpuBackend,
        kernels: BlockKernels,
        shape: &Shape,
        weights: EngramWeights,
    ) -> Result<Self> {
        let read = weights.columns * ENGRAM_DIM * 2;
        Ok(Self {
            values: gpu.alloc(read)?,
            quantized: gpu.alloc(read)?,
            kv: gpu.alloc(weights.wkv.n * 2)?,
            kernels,
            shape: *shape,
            weights,
        })
    }

    /// Add a token's lookup into its stream (`[copies, hidden]` BF16, in
    /// place). `rows` (`[columns, 256]` FP8) and `scales` (`[columns, 8]`
    /// E8M0) are the table rows its hash ids name, in the ids' order.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        rows: DevicePtr,
        scales: DevicePtr,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (k, w, s) = (&self.kernels, &self.weights, &self.shape);
        side::engram_rows(
            gpu,
            k.side.engram_rows,
            rows,
            scales,
            self.values,
            w.columns * ENGRAM_DIM,
            stream,
        )?;
        ops::fp8_linear(
            gpu,
            &k.cache,
            &k.tiers,
            &w.wkv.w,
            self.values,
            self.quantized,
            self.kv,
            1,
            w.wkv.n,
            w.wkv.k,
            stream,
        )?;
        side::engram_gate(
            gpu,
            k.side.engram_gate,
            streams,
            self.kv,
            w.weight,
            1,
            s.hidden,
            s.hc.copies,
            s.norm_eps,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "engram_test_gpu.rs"]
mod test_gpu;
