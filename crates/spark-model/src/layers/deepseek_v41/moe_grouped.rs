// SPDX-License-Identifier: AGPL-3.0-only

//! The routed experts of a batch of rows, grouped by expert
//! (`dsv41_experts_*` in `dsv41_expert.cu`).
//!
//! A row alone reads its six experts' weights, and a batch a row at a time
//! reads six a row: 384 experts' worth every 64 rows. Rows that chose the
//! same expert can share one read of it. The host sorts the batch's (row,
//! slot) pairs by expert and cuts each expert's run into tiles of up to
//! eight, and three launches take every tile: both projections in, the
//! activation, the projection out. A row's bytes are those of
//! [`super::moe::experts`] on it alone.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::moe::MoeKernels;
use super::parts::TILE;
use super::weights::ExpertSlab;

/// A batch's (row, slot) pairs in expert order, as the kernels read them.
pub struct Plan {
    /// `[tiles, 3]` u32: expert, first pair, pairs.
    pub tiles: Vec<u32>,
    /// A pair's row of the batch, its row of the output (`row * slots +
    /// slot`), and its routing weight.
    pub row_of: Vec<i32>,
    pub out_row: Vec<i32>,
    pub weights: Vec<f32>,
}

impl Plan {
    /// `chosen` and `weights` are `[rows, top_k]`, a row's experts ascending,
    /// as the router leaves them by expert. A row has `slots` outputs.
    pub fn new(
        chosen: &[u32],
        weights: &[f32],
        top_k: usize,
        slots: usize,
        experts: usize,
    ) -> Self {
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); experts];
        for (pair, &expert) in chosen.iter().enumerate() {
            by_expert[expert as usize].push((pair / top_k, pair % top_k));
        }
        let mut plan = Self {
            tiles: Vec::new(),
            row_of: Vec::with_capacity(chosen.len()),
            out_row: Vec::with_capacity(chosen.len()),
            weights: Vec::with_capacity(chosen.len()),
        };
        for (expert, pairs) in by_expert.iter().enumerate() {
            for tile in pairs.chunks(TILE) {
                plan.tiles
                    .extend([expert as u32, plan.row_of.len() as u32, tile.len() as u32]);
                for &(row, slot) in tile {
                    plan.row_of.push(row as i32);
                    plan.out_row.push((row * slots + slot) as i32);
                    plan.weights.push(weights[row * top_k + slot]);
                }
            }
        }
        plan
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len() / 3
    }

    pub fn pairs(&self) -> usize {
        self.row_of.len()
    }
}

/// Device buffers of a plan and of the launches' working values, for
/// batches of up to `rows` rows.
#[derive(Clone, Copy)]
pub struct GroupedScratch {
    pub tiles: DevicePtr,
    pub row_of: DevicePtr,
    pub out_row: DevicePtr,
    pub weights: DevicePtr,
    /// `[pairs, 2, part rows]` and `[pairs, part rows]` BF16.
    pub gate_up: DevicePtr,
    pub inner: DevicePtr,
}

impl GroupedScratch {
    pub fn alloc(
        gpu: &dyn GpuBackend,
        rows: usize,
        top_k: usize,
        experts: usize,
        part_rows: usize,
    ) -> Result<Self> {
        let pairs = rows * top_k;
        Ok(Self {
            // An expert's run ends in at most one tile that is not full.
            tiles: gpu.alloc((experts + pairs / TILE) * 3 * 4)?,
            row_of: gpu.alloc(pairs * 4)?,
            out_row: gpu.alloc(pairs * 4)?,
            weights: gpu.alloc(pairs * 4)?,
            gate_up: gpu.alloc(pairs * 2 * part_rows * 2)?,
            inner: gpu.alloc(pairs * part_rows * 2)?,
        })
    }

    /// Put `plan` on the device. Nothing queued may still read the last.
    pub fn upload(&self, gpu: &dyn GpuBackend, plan: &Plan) -> Result<()> {
        let words = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let ints = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let floats: Vec<u8> = plan.weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        gpu.copy_h2d(&words(&plan.tiles), self.tiles)?;
        gpu.copy_h2d(&ints(&plan.row_of), self.row_of)?;
        gpu.copy_h2d(&ints(&plan.out_row), self.out_row)?;
        gpu.copy_h2d(&floats, self.weights)
    }
}

/// One part of the routed experts of every row of a batch: `out` (`[rows *
/// slots, hidden]` FP32) gets, at each pair's output row, the part's partial
/// of that expert's output. `x` is the rows' input in FP8 (`[rows, hidden]`
/// BF16), `part_rows` the part's share of an expert's hidden rows.
#[allow(clippy::too_many_arguments)]
pub fn experts(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    slab: &ExpertSlab,
    plan: &Plan,
    scratch: &GroupedScratch,
    x: DevicePtr,
    out: DevicePtr,
    part_rows: usize,
    hidden: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        part_rows.is_multiple_of(32) && hidden.is_multiple_of(32),
        "dsv41_experts: {part_rows} rows of {hidden}"
    );
    let tiles = plan.tile_count() as u32;
    if tiles == 0 {
        return Ok(());
    }
    let (data, scales) = slab.layout();
    let layout = |launch: KernelLaunch<'_>| {
        launch
            .arg_u64(slab.stride() as u64)
            .arg_u32(data as u32)
            .arg_u32(scales as u32)
            .launch(stream)
    };
    layout(
        KernelLaunch::new(gpu, kernels.experts_in)
            .grid([div_ceil(part_rows as u32, 16), 2 * tiles, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(slab.base)
            .arg_ptr(scratch.tiles)
            .arg_ptr(scratch.row_of)
            .arg_ptr(scratch.gate_up)
            .arg_u32(part_rows as u32)
            .arg_u32(hidden as u32),
    )?;
    activate(
        gpu,
        kernels,
        scratch.gate_up,
        scratch.weights,
        scratch.inner,
        plan.pairs(),
        part_rows,
        limit,
        stream,
    )?;
    layout(
        KernelLaunch::new(gpu, kernels.experts_out)
            .grid([div_ceil(hidden as u32, 16), tiles, 1])
            .block([256, 1, 1])
            .arg_ptr(scratch.inner)
            .arg_ptr(slab.base)
            .arg_ptr(scratch.tiles)
            .arg_ptr(scratch.out_row)
            .arg_ptr(out)
            .arg_u32(hidden as u32)
            .arg_u32(part_rows as u32),
    )
}

/// `inner` (`[count, part_rows]` BF16) = the FP8 activation of `gate_up`
/// (`[count, 2, part_rows]`) under `weights` (`[count]` FP32, or none for the
/// shared expert's 1).
#[allow(clippy::too_many_arguments)]
pub fn activate(
    gpu: &dyn GpuBackend,
    kernels: &MoeKernels,
    gate_up: DevicePtr,
    weights: DevicePtr,
    inner: DevicePtr,
    count: usize,
    part_rows: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.experts_act)
        .grid([div_ceil((count * part_rows / 32) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_up)
        .arg_ptr(weights)
        .arg_ptr(inner)
        .arg_u32(count as u32)
        .arg_u32(part_rows as u32)
        .arg_f32(limit)
        .launch(stream)
}
