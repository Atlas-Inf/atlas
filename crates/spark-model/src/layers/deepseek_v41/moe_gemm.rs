// SPDX-License-Identifier: AGPL-3.0-only

//! The routed experts of a large batch on the block-scaled tensor cores.
//!
//! The grouped kernels of `moe_grouped.rs` take an expert's rows 32 at a time,
//! each time reading the batch's activations again for every 16 rows of its
//! weights, and widen both operands to BF16 to multiply them: some 12 TFLOPS
//! however large the batch. This hardware multiplies the two formats as they
//! are (`dsv41_mxg.cu`): the rows' activations rounded to E4M3 codes by the
//! model's own rule, against the expert's FP4 weights where the loader put
//! them. A launch takes every (expert, 128 of its pairs, 128 of its weight
//! rows) at once.
//!
//! A launch streams every chosen expert's weights from memory, which no
//! batch makes quicker, so this pays from some thousands of rows up
//! ([`expert_gemm_rows`]). The products are the row-at-a-time kernels'; the
//! FP32 sums are in another order.

use std::ops::Range;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::moe_grouped::Plan;
use super::parts::PARTS;
use super::shape::Shape;
use super::weights::ExpertSlab;

/// The fewest rows a batch's routed experts go through these with, unless
/// `ATLAS_DSV41_EXPERT_GEMM_ROWS` names another (0 for never).
const EXPERT_GEMM_ROWS: usize = 2048;

/// Eighths of a batch's pairs in the first of the two spans the ranks
/// gather their hidden activations in (`block_moe_big.rs`), unless
/// `ATLAS_DSV41_MOE_SPAN` names another; 0 for one span, gathered after
/// every pair is in. The first span is the larger: its gather has the
/// second's launches to hide under, and the second's only the first's way
/// out.
const FIRST_SPAN: usize = 5;

fn first_span() -> usize {
    std::env::var("ATLAS_DSV41_MOE_SPAN")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&eighths| eighths < 8)
        .unwrap_or(FIRST_SPAN)
}

/// A tile's pairs, a block's threads and its shared memory
/// (`MXG_THREADS`, `MXG_SHARED` in `dsv41_mxg.cu`).
const TILE: usize = 128;
const SHARED: u32 = 49_664;

/// That number as this process has it.
pub fn expert_gemm_rows() -> usize {
    std::env::var("ATLAS_DSV41_EXPERT_GEMM_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(EXPERT_GEMM_ROWS)
}

/// A plan's pairs as the tiles of a launch: up to [`TILE`] of an expert's
/// run of pairs each.
pub struct Runs {
    /// `[tiles, 3]` u32: expert, first pair, pairs.
    tiles: Vec<u32>,
    pairs: usize,
    /// `[rows, top_k]`: the pair of each row's each chosen expert.
    pair_of: Vec<i32>,
}

impl Runs {
    /// `plan` is of `rows` rows, each with `top_k` chosen experts among its
    /// `slots` outputs.
    pub fn new(plan: &Plan, rows: usize, top_k: usize, slots: usize) -> Self {
        // The plan's tiles are an expert's run cut small: join them, and cut
        // the run again.
        let mut runs: Vec<[usize; 3]> = Vec::new();
        for tile in plan.tiles.chunks_exact(3) {
            let (expert, first, count) = (tile[0] as usize, tile[1] as usize, tile[2] as usize);
            match runs.last_mut() {
                Some(run) if run[0] == expert => run[2] += count,
                _ => runs.push([expert, first, count]),
            }
        }
        let mut tiles = Vec::new();
        for [expert, first, count] in runs {
            for at in (0..count).step_by(TILE) {
                tiles.extend([expert, first + at, (count - at).min(TILE)].map(|v| v as u32));
            }
        }
        let mut pair_of = vec![0i32; rows * top_k];
        for (pair, &out_row) in plan.out_row.iter().enumerate() {
            let (row, slot) = (out_row as usize / slots, out_row as usize % slots);
            pair_of[row * top_k + slot] = pair as i32;
        }
        Self {
            tiles,
            pairs: plan.pairs(),
            pair_of,
        }
    }

    fn tile_count(&self) -> usize {
        self.tiles.len() / 3
    }

    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Every tile, and every pair.
    pub fn whole(&self) -> Span {
        Span {
            tiles: 0..self.tile_count(),
            pairs: 0..self.pairs,
        }
    }

    /// The tiles in two spans, the first of about `eighths` eighths of the
    /// pairs: cut at the tile nearest that which starts on a pair that is
    /// a multiple of [`SPAN_PAIRS`]. None where no tile inside does.
    pub fn halves(&self, eighths: usize) -> Option<[Span; 2]> {
        let want = self.pairs * eighths / 8;
        let (tile, pair) = (1..self.tile_count())
            .map(|tile| (tile, self.tiles[tile * 3 + 1] as usize))
            .filter(|&(_, pair)| pair.is_multiple_of(SPAN_PAIRS) && pair < self.pairs)
            .min_by_key(|&(_, pair)| pair.abs_diff(want))?;
        Some([
            Span {
                tiles: 0..tile,
                pairs: 0..pair,
            },
            Span {
                tiles: tile..self.tile_count(),
                pairs: pair..self.pairs,
            },
        ])
    }
}

/// A span starts on a pair whose scales start on a 16-byte boundary of a
/// part ([`Sent`]), which the ranks' copies want: a pair's scales are a
/// multiple of two bytes.
const SPAN_PAIRS: usize = 8;

/// A run of a launch's tiles, and the pairs in them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub tiles: Range<usize>,
    pub pairs: Range<usize>,
}

/// One part of a batch's hidden activations as a rank sends it: the pairs'
/// codes, their scales, and then the shared expert's rows as BF16 values
/// (its projection out is a GEMM of its own, which rounds them itself).
#[derive(Clone, Copy)]
pub struct Sent {
    /// Bytes of a part, and from one part to the next.
    pub bytes: usize,
    /// Where the scales and the shared expert's rows start in a part.
    pub scales: usize,
    pub shared: usize,
}

impl Sent {
    pub fn new(pairs: usize, rows: usize, part_rows: usize) -> Self {
        // Each on a 16-byte boundary, which the copies want.
        let scales = pairs * part_rows;
        let shared = scales + (pairs * part_rows / 32).next_multiple_of(16);
        Self {
            bytes: (shared + rows * part_rows * 2).next_multiple_of(16),
            scales,
            shared,
        }
    }
}

/// One launch's two halves of 64 weight rows a block: where each half's
/// projection is in an expert, how far apart the blocks' rows and the
/// halves' are, and where each half's outputs go.
struct Halves {
    projections: [usize; 2],
    row_step: usize,
    half_row: usize,
    out: [DevicePtr; 2],
}

/// The kernels and working buffers of these launches, for batches of up to
/// `rows` rows.
pub struct ExpertGemm {
    quant: KernelHandle,
    scales_rows: KernelHandle,
    multiply: KernelHandle,
    pairs_sum: KernelHandle,
    /// The batch's input as E4M3 codes, `[rows, hidden]`; and scales, its
    /// `[rows, hidden / 32]` on the way in and the pairs' `[pairs, inter /
    /// 32]` on the way out.
    codes: DevicePtr,
    sf: DevicePtr,
    tiles: DevicePtr,
    pair_of: DevicePtr,
    /// `[pairs, hidden / PARTS]` BF16: one part of each pair's output.
    pair_out: DevicePtr,
    /// The fewest rows of a batch this takes.
    pub from: usize,
    /// Eighths of the pairs in the first span ([`Runs::halves`]).
    pub first_span: usize,
    pairs: usize,
    max_tiles: usize,
}

impl ExpertGemm {
    pub fn alloc(gpu: &dyn GpuBackend, s: &Shape, rows: usize) -> Result<Self> {
        let (pairs, part_rows, part_n) = (rows * s.top_k, s.inter / PARTS, s.hidden / PARTS);
        ensure!(
            s.hidden.is_multiple_of(128)
                && s.inter.is_multiple_of(128)
                && part_rows.is_multiple_of(64)
                && part_n.is_multiple_of(128),
            "deepseek_v41: experts [{}, {}] are not the tensor-core kernel's",
            s.inter,
            s.hidden
        );
        // An expert's run ends in at most one tile that is not full.
        let max_tiles = s.experts + pairs / TILE;
        Ok(Self {
            quant: gpu.kernel("dsv41_mx", "dsv41_mx_quant")?,
            scales_rows: gpu.kernel("dsv41_mx", "dsv41_mx_scales_rows")?,
            multiply: gpu.kernel("dsv41_mxg", "dsv41_mxg")?,
            pairs_sum: gpu.kernel("dsv41_mx", "dsv41_pairs_sum")?,
            // Rows gathered a part each fill whole parts.
            codes: gpu.alloc(rows.next_multiple_of(PARTS) * s.hidden)?,
            sf: gpu.alloc(pairs * s.inter / 32)?,
            tiles: gpu.alloc(max_tiles * 3 * 4)?,
            pair_of: gpu.alloc(pairs * 4)?,
            pair_out: gpu.alloc(pairs * part_n * 2)?,
            from: expert_gemm_rows(),
            first_span: first_span(),
            pairs,
            max_tiles,
        })
    }

    /// Put `runs` on the device. Nothing queued may still read the last.
    pub fn upload(&self, gpu: &dyn GpuBackend, runs: &Runs) -> Result<()> {
        ensure!(
            runs.pairs <= self.pairs && runs.tile_count() <= self.max_tiles,
            "deepseek_v41: {} pairs in {} tiles, room for {} in {}",
            runs.pairs,
            runs.tile_count(),
            self.pairs,
            self.max_tiles
        );
        let tiles: Vec<u8> = runs.tiles.iter().flat_map(|r| r.to_le_bytes()).collect();
        let pairs: Vec<u8> = runs.pair_of.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&tiles, self.tiles)?;
        gpu.copy_h2d(&pairs, self.pair_of)
    }

    /// Rows `first..first + rows` of the batch's input as codes and scales,
    /// rounded to FP8: `x` is those rows' `k` values each (BF16). Every
    /// row's are in place before [`Self::project_in`]: rounded here, or a
    /// rank's own rows here and the others' gathered ([`Self::input`]).
    pub fn round_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        first: usize,
        rows: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.quant)
            .grid([div_ceil((rows * k / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.codes.offset(first * k))
            .arg_ptr(self.sf.offset(first * k / 32))
            .arg_u32(rows as u32)
            .arg_u32(k as u32)
            .arg_u32(k as u32)
            .arg_u32(0)
            .arg_u64(0)
            .arg_u32(0)
            .launch(stream)
    }

    /// The batch's input as [`Self::round_rows`] leaves it: codes, `hidden`
    /// bytes a row, and scales, `hidden / 32`. Half the bytes of its values,
    /// and what the ranks gather of each other's rows.
    pub fn input(&self) -> (DevicePtr, DevicePtr) {
        (self.codes, self.sf)
    }

    /// One launch of `tiles` of the tiles through `n` weight rows of `k`
    /// values of their experts. A pair's codes are row `a_rows[pair]` of `codes`, or
    /// the pair's own with no `a_rows`, in pieces of `piece[0]` values
    /// `piece[1]` bytes apart, or in one run when that is 0; its outputs are
    /// `out_stride` values after the pair's before.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        gpu: &dyn GpuBackend,
        slab: &ExpertSlab,
        tiles: Range<usize>,
        codes: DevicePtr,
        piece: [usize; 2],
        a_rows: DevicePtr,
        halves: Halves,
        n: usize,
        k: usize,
        out_stride: usize,
        stream: u64,
    ) -> Result<()> {
        if tiles.is_empty() {
            return Ok(());
        }
        let (data, scales) = slab.layout();
        let at = |projection: usize| (projection * (data + scales)) as u32;
        let [p0, p1] = halves.projections;
        KernelLaunch::new(gpu, self.multiply)
            .grid([(n / halves.row_step) as u32, tiles.len() as u32, 1])
            .block([TILE as u32, 1, 1])
            .shared_mem(SHARED)
            .arg_ptr(codes)
            .arg_ptr(self.sf)
            .arg_ptr(a_rows)
            .arg_ptr(slab.base)
            .arg_ptr(self.tiles.offset(tiles.start * 3 * 4))
            .arg_ptr(halves.out[0])
            .arg_ptr(halves.out[1])
            .arg_u64(slab.stride() as u64)
            .arg_u32(at(p0))
            .arg_u32(at(p1))
            .arg_u32(at(p0) + data as u32)
            .arg_u32(at(p1) + data as u32)
            .arg_u32(halves.row_step as u32)
            .arg_u32(halves.half_row as u32)
            .arg_u32(k as u32)
            .arg_u32(out_stride as u32)
            .arg_u32(piece[0] as u32)
            .arg_u64(piece[1] as u64)
            .launch(stream)
    }

    /// The way in, of the pairs of `tiles`: `gate_up` gets a pair's
    /// `part_rows` gate values and, `pairs * part_rows` values on, its up
    /// values (BF16), from the batch's input in codes
    /// ([`Self::round_rows`]); `row_of` is each pair's row.
    #[allow(clippy::too_many_arguments)]
    pub fn project_in(
        &self,
        gpu: &dyn GpuBackend,
        slab: &ExpertSlab,
        runs: &Runs,
        tiles: Range<usize>,
        row_of: DevicePtr,
        gate_up: DevicePtr,
        part_rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        // A block's two halves are the same 64 rows of the gate and of the
        // up projection, which read the same codes.
        let halves = Halves {
            projections: [0, 1],
            row_step: 64,
            half_row: 0,
            out: [gate_up, gate_up.offset(runs.pairs * part_rows * 2)],
        };
        self.launch(
            gpu,
            slab,
            tiles,
            self.codes,
            [0, 0],
            row_of,
            halves,
            part_rows,
            hidden,
            part_rows,
            stream,
        )
    }

    /// The way out, of the pairs of `span`: one part of a pair's output,
    /// kept here for [`Self::sum`]. `inner` is every part of the pairs'
    /// hidden activations as the ranks sent them ([`Sent`]).
    #[allow(clippy::too_many_arguments)]
    pub fn project_out(
        &self,
        gpu: &dyn GpuBackend,
        slab: &ExpertSlab,
        span: &Span,
        inner: DevicePtr,
        sent: Sent,
        part_rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        let (inter, part_n) = (part_rows * PARTS, hidden / PARTS);
        if span.pairs.is_empty() {
            return Ok(());
        }
        // A pair's scales in one run; its codes are read where they are.
        let first = span.pairs.start;
        KernelLaunch::new(gpu, self.scales_rows)
            .grid([div_ceil((span.pairs.len() * inter / 32) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(inner.offset(sent.scales + first * part_rows / 32))
            .arg_ptr(self.sf.offset(first * inter / 32))
            .arg_u32(span.pairs.len() as u32)
            .arg_u32((part_rows / 32) as u32)
            .arg_u32(PARTS as u32)
            .arg_u64(sent.bytes as u64)
            .launch(stream)?;
        // 128 rows of the down projection a block.
        let halves = Halves {
            projections: [2, 2],
            row_step: 128,
            half_row: 64,
            out: [self.pair_out, self.pair_out],
        };
        self.launch(
            gpu,
            slab,
            span.tiles.clone(),
            inner,
            [part_rows, sent.bytes],
            DevicePtr::NULL,
            halves,
            part_n,
            inter,
            part_n,
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

#[cfg(test)]
#[path = "moe_gemm_test_gpu.rs"]
mod test_gpu;
