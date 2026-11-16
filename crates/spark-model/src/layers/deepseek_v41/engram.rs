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
//! The projection is in four parts by its output rows, like every wide one
//! (`parts.rs`), and it reads a token's rows in four parts too: a rank
//! fetches the table rows of a quarter of the hash columns, the ranks gather
//! the values, a rank projects a quarter of the output from all of them, and
//! the ranks gather that.

use std::ops::Range;

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cache_recipe::bf16_to_f32;
use super::gemm::{GEMM_ROWS, GemmScratch, Input, gemms, operand};
use super::parts::{self, Held, PARTS};
use super::shape::{BlockKernels, Shape};
use super::side;
use super::weights::{Parted, Slice, fp8, tensor};
use crate::layers::ops::MxW8a8;

/// Values in one table row.
pub const ENGRAM_DIM: usize = 256;
/// Values of a row under one scale.
pub const ENGRAM_SCALE_GROUP: usize = 32;
/// Tokens one lookup takes: their rows are held in memory together.
pub const ENGRAM_ROWS: usize = 256;

/// An Engram layer's weights, apart from its table.
pub struct EngramWeights {
    /// `[(copies + 1) * hidden, columns * 256]` in four parts by rows: a key
    /// per copy, then the value. And the same as GEMM operands.
    pub wkv: Parted,
    gemm: [Option<MxW8a8>; PARTS],
    /// Rows of `wkv`, and its columns.
    pub rows: usize,
    k: usize,
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
        held: Held,
        stream: u64,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.engram");
        let whole = fp8(store, gpu, &format!("{prefix}.wkv"))?;
        let (rows, k) = (whole.n, whole.k);
        let gate = shape.hc.copies * shape.hidden;
        ensure!(
            rows == gate + shape.hidden
                && rows.is_multiple_of(PARTS)
                && k.is_multiple_of(PARTS * ENGRAM_DIM),
            "{prefix}.wkv [{rows}, {k}]"
        );
        // A rank keeps the rows of its parts.
        let part_rows = rows / PARTS;
        let (mut wkv, mut gemm): (Parted, _) = ([None; PARTS], [None; PARTS]);
        gpu.synchronize(stream)?;
        for at in held.range() {
            let from = whole.rows_from(at * part_rows);
            let slice = Slice {
                data: gpu.alloc(part_rows * k)?,
                scales: gpu.alloc(part_rows * k / 32)?,
            };
            gpu.copy_d2d(from.data, slice.data, part_rows * k)?;
            gpu.copy_d2d(from.scales, slice.scales, part_rows * k / 32)?;
            wkv[at] = Some(slice);
            if gemms() {
                gemm[at] = Some(operand(gpu, slice, part_rows, k, stream)?);
            }
        }
        store.reclaim(gpu, &format!("{prefix}.wkv.weight"))?;
        gpu.free(whole.w.scales)?;
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
            columns: k / ENGRAM_DIM,
            wkv,
            gemm,
            rows,
            k,
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

/// One Engram layer on up to [`ENGRAM_ROWS`] tokens at a time.
pub struct Engram {
    pub weights: EngramWeights,
    kernels: BlockKernels,
    shape: Shape,
    held: Held,
    /// `[PARTS, tokens, columns / PARTS * 256]` BF16: the rows as values, a
    /// part's tokens together.
    pub(super) values: DevicePtr,
    /// `[PARTS, tokens, (copies + 1) * hidden / PARTS]` BF16: the
    /// projection, the keys and then the value of a token, in parts.
    pub(super) kv: DevicePtr,
    /// Scratch for a gather that is not in place.
    gather_recv: DevicePtr,
}

impl Engram {
    pub fn new(
        gpu: &dyn GpuBackend,
        kernels: BlockKernels,
        shape: &Shape,
        held: Held,
        weights: EngramWeights,
    ) -> Result<Self> {
        let read = ENGRAM_ROWS * weights.columns * ENGRAM_DIM * 2;
        Ok(Self {
            values: gpu.alloc(read)?,
            kv: gpu.alloc(ENGRAM_ROWS * weights.rows * 2)?,
            gather_recv: gpu.alloc(read.max(ENGRAM_ROWS * weights.rows * 2))?,
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

    /// Columns of a lookup one part of the projection reads.
    pub fn per_part(&self) -> usize {
        self.weights.columns / PARTS
    }

    /// Add the lookups of `tokens` tokens into their streams (`[tokens,
    /// copies, hidden]` BF16, in place). The rows their hash ids name are
    /// rows `row_slots` of `rows` (256 FP8 values a row) and rows
    /// `scale_slots` of `scales` (8 E8M0 scales a row). The slots are u32,
    /// `[tokens, columns / PARTS]` for each part this rank holds, one part
    /// after another. `comm` reaches the ranks that hold the other parts,
    /// and `scratch` is where a batch's values are rounded for a GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        scratch: &GemmScratch,
        rows: DevicePtr,
        scales: DevicePtr,
        row_slots: DevicePtr,
        scale_slots: DevicePtr,
        streams: DevicePtr,
        tokens: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (k, w, s) = (&self.kernels, &self.weights, &self.shape);
        ensure!(tokens <= ENGRAM_ROWS, "Engram: {tokens} tokens at once");
        let per = self.per_part();
        let (width, part_n) = (per * ENGRAM_DIM, w.rows / PARTS);
        // A part of every token's values: the table rows of a quarter of the
        // hash columns.
        for part in self.held.range() {
            let slots = (part - self.held.first) * tokens * per * 4;
            side::engram_rows(
                gpu,
                k.side.engram_rows,
                rows,
                scales,
                row_slots.offset(slots),
                scale_slots.offset(slots),
                self.values.offset(part * tokens * width * 2),
                tokens * per,
                stream,
            )?;
        }
        let values = tokens * width * 2;
        parts::gather(
            gpu,
            comm,
            self.held,
            self.values,
            self.gather_recv,
            values,
            values,
            stream,
        )?;
        // A part of the projection from all of them, which it rounds to FP8
        // on the way in.
        for part in self.held.range() {
            let out = self.kv.offset(part * tokens * part_n * 2);
            let gemm = w.gemm[part].as_ref().filter(|_| tokens >= GEMM_ROWS);
            match gemm {
                Some(operand) => {
                    let x = Input {
                        x: self.values,
                        stride: width,
                        piece: width,
                        piece_stride: tokens * width,
                    };
                    scratch.project(gpu, x, operand, out, tokens, stream)?
                }
                None => parts::project_pieces(
                    gpu,
                    k.parts.fp8_pieces_q,
                    self.values,
                    w.wkv[part].context("a held part of the Engram projection")?,
                    out,
                    part_n,
                    tokens,
                    part_n,
                    w.k,
                    width,
                    tokens * width,
                    stream,
                )?,
            }
        }
        let projected = tokens * part_n * 2;
        parts::gather(
            gpu,
            comm,
            self.held,
            self.kv,
            self.gather_recv,
            projected,
            projected,
            stream,
        )?;
        side::engram_gate(
            gpu,
            k.side.engram_gate,
            streams,
            self.kv,
            w.weight,
            tokens,
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
