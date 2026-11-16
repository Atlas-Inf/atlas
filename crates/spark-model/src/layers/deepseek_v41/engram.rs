// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1's Engram, in plain form: before two of the blocks, a token's
//! n-gram hash ids fetch rows of a table, a projection turns the rows into a
//! key per copy of the stream and one value, and the value is added into
//! each copy through a gate that compares the copy with its key (`Engram` in
//! the release's `inference/model.py`).
//!
//! The table is 384 million rows of 256 FP8 values, 101 GB a layer. This
//! takes the rows a token reads, as the table stores them and wherever they
//! sit; `engram_table.rs` finds them on disk.
//!
//! The projection is in four column parts, like every wide one (`parts.rs`):
//! a part reads a quarter of the token's rows, so a rank fetches only the
//! rows of the parts it holds, and the ranks gather the partial sums.

use std::ops::Range;

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cache_recipe::bf16_to_f32;
use super::parts::{self, Held, PARTS, PartKernels};
use super::shape::{BlockKernels, Shape};
use super::side;
use super::weights::{Parted, column_parts, tensor};

/// Values in one table row.
pub const ENGRAM_DIM: usize = 256;
/// Values of a row under one scale.
pub const ENGRAM_SCALE_GROUP: usize = 32;

/// An Engram layer's weights, apart from its table.
pub struct EngramWeights {
    /// `[(copies + 1) * hidden, columns * 256]` in four column parts: a key
    /// per copy, then the value.
    pub wkv: Parted,
    /// Rows of `wkv`.
    pub rows: usize,
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
        kernels: &PartKernels,
        layer: usize,
        shape: &Shape,
        held: Held,
        stream: u64,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.engram");
        let (wkv, rows, width) =
            column_parts(store, gpu, kernels, &format!("{prefix}.wkv"), held, stream)?;
        let gate = shape.hc.copies * shape.hidden;
        ensure!(
            rows == gate + shape.hidden && width.is_multiple_of(ENGRAM_DIM),
            "{prefix}.wkv [{rows}, {PARTS} x {width}]"
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
            columns: PARTS * width / ENGRAM_DIM,
            wkv,
            rows,
            weight,
        })
    }
}

/// Of the `columns` rows a token's lookup reads, those a rank holding `held`
/// reads: the rows its parts of the projection take.
pub fn held_columns(columns: usize, held: Held) -> Range<usize> {
    let per = columns / PARTS;
    held.first * per..(held.first + held.count) * per
}

/// One Engram layer on one token at a time.
pub struct Engram {
    pub weights: EngramWeights,
    kernels: BlockKernels,
    shape: Shape,
    held: Held,
    /// `[columns * 256]` BF16: the rows as values.
    pub(super) values: DevicePtr,
    /// `[PARTS, rows]` FP32: the parts of the projection, and scratch as
    /// large for a gather that is not in place.
    kv_parts: DevicePtr,
    gather_recv: DevicePtr,
    /// `[(copies + 1) * hidden]` BF16: the keys, then the value.
    pub(super) kv: DevicePtr,
}

impl Engram {
    pub fn new(
        gpu: &dyn GpuBackend,
        kernels: BlockKernels,
        shape: &Shape,
        held: Held,
        weights: EngramWeights,
    ) -> Result<Self> {
        let read = weights.columns * ENGRAM_DIM * 2;
        Ok(Self {
            values: gpu.alloc(read)?,
            kv_parts: gpu.alloc(PARTS * weights.rows * 4)?,
            gather_recv: gpu.alloc(PARTS * weights.rows * 4)?,
            kv: gpu.alloc(weights.rows * 2)?,
            kernels,
            shape: *shape,
            held,
            weights,
        })
    }

    /// The columns of a token's lookup this rank reads.
    pub fn columns(&self) -> Range<usize> {
        held_columns(self.weights.columns, self.held)
    }

    /// Add a token's lookup into its stream (`[copies, hidden]` BF16, in
    /// place). The rows its hash ids name are, in the ids' order, rows
    /// `row_slots` (`[columns]` u32) of `rows` (256 FP8 values a row) and
    /// rows `scale_slots` of `scales` (8 E8M0 scales a row); only
    /// [`Self::columns`] of them are read. `comm` reaches the ranks that
    /// hold the other parts.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        rows: DevicePtr,
        scales: DevicePtr,
        row_slots: DevicePtr,
        scale_slots: DevicePtr,
        streams: DevicePtr,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (k, w, s) = (&self.kernels, &self.weights, &self.shape);
        let per = w.columns / PARTS;
        let width = per * ENGRAM_DIM;
        for part in self.held.range() {
            let values = self.values.offset(part * width * 2);
            side::engram_rows(
                gpu,
                k.side.engram_rows,
                rows,
                scales,
                row_slots.offset(part * per * 4),
                scale_slots.offset(part * per * 4),
                values,
                per,
                stream,
            )?;
            // The projection rounds the values to FP8 on the way in.
            let wkv = w.wkv[part].context("a held part of the Engram projection")?;
            parts::project(
                gpu,
                k.parts.fp8_f32_q,
                values,
                wkv.data,
                wkv.scales,
                self.kv_parts.offset(part * w.rows * 4),
                w.rows,
                width,
                stream,
            )?;
        }
        parts::gather(
            gpu,
            comm,
            self.held,
            self.kv_parts,
            self.gather_recv,
            w.rows * 4,
            stream,
        )?;
        parts::round(
            gpu,
            k.parts.round,
            self.kv_parts,
            self.kv,
            1,
            w.rows,
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
