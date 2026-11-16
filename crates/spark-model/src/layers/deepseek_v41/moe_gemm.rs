// SPDX-License-Identifier: AGPL-3.0-only

//! The routed experts of a large batch as grouped GEMMs.
//!
//! The grouped kernels of `moe_grouped.rs` take an expert's rows 32 at a time,
//! each time reading the batch's activations again for every 16 rows of its
//! weights: some 12 TFLOPS however large the batch. With thousands of rows a
//! batch an expert has a hundred or more, which is a GEMM. Here every
//! expert's projection of its rows is one group of one launch
//! (`spark_runtime::cutlass::mx84_grouped`): the rows' activations rounded to
//! E4M3 codes by the model's own rule, against the expert's FP4 weights where
//! they are, on the block-scaled tensor cores that multiply the two
//! unrounded. Only the experts' scales are laid out again, a launch a layer.
//!
//! A launch reads every expert's weights whatever the batch, so this pays
//! from some thousands of rows up ([`expert_gemm_rows`]). The products are the row-at-a-time
//! kernels'; the FP32 sums are in another order.

use anyhow::{Result, ensure};
use spark_runtime::cutlass::{Mx84Group, mx84_grouped, mx84_workspace_bytes};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::moe_grouped::Plan;
use super::parts::PARTS;
use super::shape::Shape;
use super::weights::ExpertSlab;
use crate::layers::ops::{mx_lt_inner_pad, mx_lt_scale_bytes};

/// The fewest rows a batch's routed experts go through the grouped GEMM
/// with, unless `ATLAS_DSV41_EXPERT_GEMM_ROWS` names another (0 for never).
const EXPERT_GEMM_ROWS: usize = 4096;

/// That number as this process has it.
pub fn expert_gemm_rows() -> usize {
    std::env::var("ATLAS_DSV41_EXPERT_GEMM_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EXPERT_GEMM_ROWS)
}

/// A plan's pairs as the groups of a launch: an expert's run of pairs each.
pub struct Runs {
    /// Expert, its first pair, its pairs, and the row its scales start at.
    runs: Vec<[usize; 4]>,
    /// A pair's row of the activations' scales: a run starts on a multiple
    /// of 128 rows, where a tile of scales starts.
    sf_rows: Vec<u32>,
    /// `[rows, top_k]`: the pair of each row's each chosen expert.
    pair_of: Vec<i32>,
}

impl Runs {
    /// `plan` is of `rows` rows, each with `top_k` chosen experts among its
    /// `slots` outputs.
    pub fn new(plan: &Plan, rows: usize, top_k: usize, slots: usize) -> Self {
        let mut runs: Vec<[usize; 4]> = Vec::new();
        let mut sf_rows = Vec::with_capacity(plan.pairs());
        for tile in plan.tiles.chunks_exact(3) {
            let (expert, first, count) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
            match runs.last_mut() {
                Some(run) if run[0] == expert => run[2] += count,
                _ => {
                    let padded = runs.last().map_or(0, |r| r[3] + r[2].div_ceil(128) * 128);
                    runs.push([expert, first, count, padded]);
                }
            }
        }
        for run in &runs {
            sf_rows.extend((0..run[2]).map(|i| (run[3] + i) as u32));
        }
        let mut pair_of = vec![0i32; rows * top_k];
        for (pair, &out_row) in plan.out_row.iter().enumerate() {
            let (row, slot) = (out_row as usize / slots, out_row as usize % slots);
            pair_of[row * top_k + slot] = pair as i32;
        }
        Self {
            runs,
            sf_rows,
            pair_of,
        }
    }
}

/// The kernels and working buffers of the grouped GEMMs, for batches of up
/// to `rows` rows.
pub struct ExpertGemm {
    quant: KernelHandle,
    slab_scales: KernelHandle,
    pairs_sum: KernelHandle,
    /// `[pairs, hidden]` E4M3, and their scales in tiles, a run to itself.
    codes: DevicePtr,
    sf: DevicePtr,
    sf_rows: DevicePtr,
    pair_of: DevicePtr,
    /// One part of every expert's scales in tiles: w1's, w3's, w2's.
    sfb: DevicePtr,
    /// `[pairs, hidden / PARTS]` BF16: one part of each pair's output.
    pair_out: DevicePtr,
    workspace: DevicePtr,
    workspace_bytes: usize,
    /// The fewest rows of a batch this takes.
    pub from: usize,
    pairs: usize,
    experts: usize,
    /// Bytes of an expert's tiled scales: one projection in, and out.
    tiled_in: usize,
    tiled_out: usize,
}

impl ExpertGemm {
    pub fn alloc(gpu: &dyn GpuBackend, s: &Shape, rows: usize) -> Result<Self> {
        let (pairs, part_rows, part_n) = (rows * s.top_k, s.inter / PARTS, s.hidden / PARTS);
        ensure!(
            s.hidden.is_multiple_of(128) && s.inter.is_multiple_of(128) && part_n.is_multiple_of(8),
            "deepseek_v41: experts [{}, {}] are not a grouped GEMM's",
            s.inter,
            s.hidden
        );
        let kernel = |name: &str| gpu.kernel("dsv41_mx", name);
        // Every run's scales start a tile, so each may leave one unfilled.
        let sf_rows = pairs + s.experts * 128;
        let (tiled_in, tiled_out) = (
            mx_lt_scale_bytes(part_rows, s.hidden),
            mx_lt_scale_bytes(part_n, s.inter),
        );
        let workspace_bytes = mx84_workspace_bytes(2 * s.experts);
        Ok(Self {
            quant: kernel("dsv41_mx_quant")?,
            slab_scales: kernel("dsv41_slab_scales")?,
            pairs_sum: kernel("dsv41_pairs_sum")?,
            codes: gpu.alloc(pairs * s.hidden)?,
            sf: gpu.alloc(mx_lt_scale_bytes(sf_rows, s.hidden))?,
            sf_rows: gpu.alloc(pairs * 4)?,
            pair_of: gpu.alloc(pairs * 4)?,
            sfb: gpu.alloc(s.experts * (2 * tiled_in + tiled_out))?,
            pair_out: gpu.alloc(pairs * part_n * 2)?,
            workspace: gpu.alloc(workspace_bytes)?,
            workspace_bytes,
            from: expert_gemm_rows(),
            pairs,
            experts: s.experts,
            tiled_in,
            tiled_out,
        })
    }

    /// Put `runs` on the device. Nothing queued may still read the last.
    pub fn upload(&self, gpu: &dyn GpuBackend, runs: &Runs) -> Result<()> {
        ensure!(
            runs.sf_rows.len() <= self.pairs && runs.runs.len() <= self.experts,
            "deepseek_v41: {} pairs of {} experts, room for {} of {}",
            runs.sf_rows.len(),
            runs.runs.len(),
            self.pairs,
            self.experts
        );
        let rows: Vec<u8> = runs.sf_rows.iter().flat_map(|r| r.to_le_bytes()).collect();
        let pairs: Vec<u8> = runs.pair_of.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&rows, self.sf_rows)?;
        gpu.copy_h2d(&pairs, self.pair_of)
    }

    /// `rows` rows of `k` values of `x` as the runs' activations: rounded to
    /// FP8, a row `stride` values after the last and in pieces as
    /// `gemm::Input` has them; row m is row `a_rows[m]` of `x`, or row m.
    #[allow(clippy::too_many_arguments)]
    fn quantize(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        a_rows: DevicePtr,
        rows: usize,
        k: usize,
        stride: usize,
        piece: [usize; 2],
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.quant)
            .grid([div_ceil((rows * k / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.codes)
            .arg_ptr(self.sf)
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_u32(stride as u32)
            .arg_u32(piece[0] as u32)
            .arg_u64(piece[1] as u64)
            .arg_u32(mx_lt_inner_pad(k) as u32)
            .arg_ptr(a_rows)
            .arg_ptr(self.sf_rows)
            .launch(stream)
    }

    /// One launch of the runs through a projection `[n, k]` of their
    /// experts: `weights` is an expert's packed weights and the offset of its
    /// tiled scales in an expert's, and `out` where a run's first row goes.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        runs: &Runs,
        projections: &[(&dyn Fn(usize) -> DevicePtr, usize, DevicePtr)],
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let per = 2 * self.tiled_in + self.tiled_out;
        let pad = mx_lt_inner_pad(k);
        let mut groups = Vec::with_capacity(runs.runs.len() * projections.len());
        for (weights, scales_at, out) in projections {
            for &[expert, first, count, sf_first] in &runs.runs {
                groups.push(Mx84Group {
                    a: self.codes.offset(first * k).0,
                    sfa: self.sf.offset(sf_first * pad).0,
                    b: weights(expert).0,
                    sfb: self.sfb.offset(expert * per + scales_at).0,
                    d: out.offset(first * n * 2).0,
                    rows: count,
                });
            }
        }
        mx84_grouped(
            &groups,
            n,
            k,
            self.workspace.0,
            self.workspace_bytes,
            stream,
        )
    }

    /// The way in: `gate_up` gets every pair's `part_rows` gate values and,
    /// `pairs * part_rows` values on, its up values (BF16). `x` is the
    /// batch's input (`[rows, hidden]` BF16) and `row_of` each pair's row.
    #[allow(clippy::too_many_arguments)]
    pub fn project_in(
        &self,
        gpu: &dyn GpuBackend,
        slab: &ExpertSlab,
        runs: &Runs,
        x: DevicePtr,
        row_of: DevicePtr,
        gate_up: DevicePtr,
        part_rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        let pairs = runs.sf_rows.len();
        let (data, scales) = slab.layout();
        let (part_n, inter) = (hidden / PARTS, part_rows * PARTS);
        let bytes = self.experts * (2 * self.tiled_in + self.tiled_out);
        KernelLaunch::new(gpu, self.slab_scales)
            .grid([div_ceil(bytes as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(slab.base)
            .arg_ptr(self.sfb)
            .arg_u32(self.experts as u32)
            .arg_u64(slab.stride() as u64)
            .arg_u32(data as u32)
            .arg_u32(scales as u32)
            .arg_u32(part_rows as u32)
            .arg_u32(hidden as u32)
            .arg_u32(part_n as u32)
            .arg_u32(inter as u32)
            .launch(stream)?;
        self.quantize(gpu, x, row_of, pairs, hidden, hidden, [0, 0], stream)?;
        let up = gate_up.offset(pairs * part_rows * 2);
        self.launch(
            runs,
            &[
                (&|e| slab.w1(e).data, 0, gate_up),
                (&|e| slab.w3(e).data, self.tiled_in, up),
            ],
            part_rows,
            hidden,
            stream,
        )
    }

    /// The way out: one part of every pair's output, kept here for
    /// [`Self::sum`]. `inner` is every part of the pairs' hidden activations,
    /// a part `piece_stride` values after the last ([`project_in`] must have
    /// run for this slab: it lays out the scales).
    #[allow(clippy::too_many_arguments)]
    pub fn project_out(
        &self,
        gpu: &dyn GpuBackend,
        slab: &ExpertSlab,
        runs: &Runs,
        inner: DevicePtr,
        piece_stride: usize,
        part_rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        let (pairs, inter) = (runs.sf_rows.len(), part_rows * PARTS);
        let piece = [part_rows, piece_stride];
        self.quantize(
            gpu,
            inner,
            DevicePtr::NULL,
            pairs,
            inter,
            part_rows,
            piece,
            stream,
        )?;
        self.launch(
            runs,
            &[(&|e| slab.w2(e).data, 2 * self.tiled_in, self.pair_out)],
            hidden / PARTS,
            inter,
            stream,
        )
    }

    /// `y` (`[rows, n]` BF16) = each row's experts' outputs and then the
    /// shared expert's (`shared`, `[rows, n]` BF16), added as
    /// `parts::slots_sum` adds them.
    #[allow(clippy::too_many_arguments)]
    pub fn sum(
        &self,
        gpu: &dyn GpuBackend,
        shared: DevicePtr,
        y: DevicePtr,
        rows: usize,
        top_k: usize,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(!shared.is_null(), "dsv41_pairs_sum: no shared expert");
        KernelLaunch::new(gpu, self.pairs_sum)
            .grid([div_ceil((rows * n) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.pair_out)
            .arg_ptr(self.pair_of)
            .arg_ptr(shared)
            .arg_ptr(y)
            .arg_u32(top_k as u32)
            .arg_u32(n as u32)
            .arg_u32(rows as u32)
            .launch(stream)
    }
}
